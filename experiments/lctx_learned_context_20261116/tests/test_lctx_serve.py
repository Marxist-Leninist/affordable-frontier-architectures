"""serving library tests (v2): a whole prompt, prompt chunks over a cache (t0 > 0, bottom-right causal), token-by-token
decode and fp16/bf16 caches reproduce the trainer's formula; eviction keeps the answer; pack files are checked against
the pack's own record; the export tool picks the pinned gate of the pack's step and refuses anything else.
python3 -I test_lctx_serve.py SRC_DIR TOOLS_DIR"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SRC, TOOLS = sys.argv.pop(1), sys.argv.pop(1)
sys.path.insert(0, SRC)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lctx_core as LC  # noqa: E402
from lctx_serve import LctxServe  # noqa: E402

torch.set_num_threads(2)


def setup(T=60, E=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.stack([LC.r_from_rate(torch.tensor([0.4, 0.03, 1e-6])), LC.r_from_rate(torch.tensor([0.1, 1.5, 1e-9]))]).float()
    w = torch.randn(2, 3, E, generator=g) * 0.4
    q = F.normalize(torch.randn(1, 3, T, E, generator=g), dim=-1) * math.sqrt(E) * 2
    k = F.normalize(torch.randn(1, 3, T, E, generator=g), dim=-1)
    v = torch.randn(1, 3, T, E, generator=g)
    return r, w, q, k, v


def ref(layer, r, w, q, k, v):
    return LC.ref_attention(q.double(), k.double(), v.double(), r[layer].double(), w[layer].double(), 1.0,
                            amax=6.0).float()


class Serve(unittest.TestCase):
    def test_whole_prompt_chunks_and_decode_match_training(self):
        r, w, q, k, v = setup()
        sv = LctxServe(r, w, 6.0)
        for layer in range(2):
            want = ref(layer, r, w, q, k, v)
            out, kb = sv.attend(layer, q, k, v, None, 0, 1.0, hgroup=2)
            self.assertLess(float((out - want).abs().max()), 2e-5)
            # chunks 20 + 40, then 7 + 53, over the growing cache
            for cut in (20, 7):
                o1, kb1 = sv.attend(layer, q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], None, 0, 1.0)
                o2, kb2 = sv.attend(layer, q[:, :, cut:], k, v, kb1, cut, 1.0)
                self.assertLess(float((torch.cat([o1, o2], 2) - want).abs().max()), 2e-5, cut)
                self.assertLess(float((torch.cat([kb1, kb2], -1) - kb).abs().max()), 1e-4)
            # token by token after a 20-token prompt
            o, KB = sv.attend(layer, q[:, :, :20], k[:, :, :20], v[:, :, :20], None, 0, 1.0)
            outs = [o]
            for t in range(20, 60):
                ot, kt = sv.attend(layer, q[:, :, t:t + 1], k[:, :, :t + 1], v[:, :, :t + 1], KB, t, 1.0)
                outs.append(ot)
                KB = torch.cat([KB, kt], -1)
            self.assertLess(float((torch.cat(outs, 2) - want).abs().max()), 2e-5)

    def test_half_precision_caches(self):
        r, w, q, k, v = setup(T=256)
        sv = LctxServe(r, w, 6.0)
        want = ref(1, r, w, q, k, v)
        for dt in (torch.float16, torch.bfloat16):
            o1, kb1 = sv.attend(1, q[:, :, :100].to(dt), k[:, :, :100].to(dt), v[:, :, :100].to(dt), None, 0, 1.0)
            o2, _ = sv.attend(1, q[:, :, 100:].to(dt), k.to(dt), v.to(dt), kb1, 100, 1.0)
            err = float((torch.cat([o1, o2], 2).float() - want).abs().max())
            self.assertLess(err, 0.05 if dt == torch.bfloat16 else 0.01, (dt, err))

    def test_eviction_keeps_the_answer(self):
        r, w, q, k, v = setup(T=120)
        sv = LctxServe(r, w, 6.0)
        layer = 1  # head 1 forgets fast (rate 1.5), head 2 almost never (rate 1e-9)
        want = ref(layer, r, w, q, k, v)
        kb = torch.cumsum(sv.rates(layer, v), -1)
        outs, kept = [], []
        for t in range(120):
            row = kb[..., :t + 1]
            keep = ~LctxServe.evictable(row, 40.0)
            o = []
            for h in range(3):
                idx = keep[0, h].nonzero().flatten()
                kbh = row[:, h:h + 1, idx]
                logits = (q[:, h:h + 1, t:t + 1] @ k[:, h:h + 1, idx].transpose(-1, -2)) + (kbh - kbh[..., -1:])[..., None, :]
                o.append(torch.softmax(logits, -1) @ v[:, h:h + 1, idx])
            outs.append(torch.cat(o, 1))
            kept.append(keep.sum(-1)[0].tolist())
        self.assertLess(float((torch.cat(outs, 2) - want).abs().max()), 1e-5)
        self.assertLess(kept[-1][1], 40)
        self.assertEqual(kept[-1][2], 120)

    def test_pack_files_and_export(self):
        d = tempfile.mkdtemp()
        try:
            import hashlib
            run, pack = os.path.join(d, 'run'), os.path.join(d, 'pack')
            os.makedirs(os.path.join(run, 'lctx.pinned'))
            os.makedirs(os.path.join(run, 'lctx.ring'))
            os.makedirs(pack)
            st = LC.LctxState(2, 3, 16)
            st.r[0, 0] = -2.0
            st.w[1, 2, 3] = 0.7
            st.updates = 7
            sha = hashlib.sha256(st.params_bytes()).hexdigest()
            torch.save(dict(r=st.r.clone(), w=st.w.clone(), trainer_step=40, updates=7, L=2, H=3, E=16, amax=6.0,
                            sha256=sha), os.path.join(run, 'lctx.pinned', 's000000040_u0000007.pt'))
            st2 = LC.LctxState(2, 3, 16)  # the trainer moved on: a later state
            st2.updates = 9
            torch.save(dict(st2.state_dict(), trainer_step=44), os.path.join(run, 'lctx.state.pt'))
            rec = dict(version='lctx-v1.1-20261116', updates=7, geometry=[2, 3, 16], amax=6.0, sha256=sha,
                       state_file=os.path.join(run, 'lctx.state.pt'))
            json.dump(dict(step=40, lctx=rec), open(os.path.join(pack, 'state.json'), 'w'))
            ex = lambda: subprocess.run([sys.executable, '-I', os.path.join(TOOLS, 'export_lctx.py'), pack],  # noqa: E731
                                        capture_output=True, text=True)
            out = ex()
            self.assertIn('EXPORTED', out.stdout, out.stdout + out.stderr)
            self.assertIn('lctx.pinned', out.stdout)
            sv = LctxServe.from_pack(pack, expect=rec)
            self.assertTrue(torch.equal(sv.r, st.r) and torch.equal(sv.w, st.w))
            self.assertEqual(sv.meta['trainer_step'], 40)
            with self.assertRaises(ValueError):
                LctxServe.from_pack(pack, expect=dict(rec, sha256='0' * 64))
            with self.assertRaises(ValueError):
                LctxServe.from_pack(pack, expect=dict(rec, updates=8))
            # a record whose gate is nowhere: refused, never the live state instead
            json.dump(dict(step=41, lctx=dict(rec, updates=8)), open(os.path.join(pack, 'state.json'), 'w'))
            self.assertIn('REFUSED', ex().stdout)
            # a matching update count with different parameters: refused (sha256)
            json.dump(dict(step=40, lctx=dict(rec, sha256='f' * 64)), open(os.path.join(pack, 'state.json'), 'w'))
            self.assertIn('REFUSED', ex().stdout)
            # a pack from before the mod
            json.dump(dict(step=30), open(os.path.join(pack, 'state.json'), 'w'))
            self.assertIn('NO_LCTX', ex().stdout)
            # a damaged params file, missing files with and without a record
            blob = bytearray(open(os.path.join(pack, 'lctx.params.f32'), 'rb').read())
            blob[5] ^= 1
            open(os.path.join(pack, 'lctx.params.f32'), 'wb').write(bytes(blob))
            with self.assertRaises(ValueError):
                LctxServe.from_pack(pack)
            os.remove(os.path.join(pack, 'lctx.json'))
            self.assertIsNone(LctxServe.from_pack(pack))
            with self.assertRaises(ValueError):
                LctxServe.from_pack(pack, expect=rec)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
