"""tests for the built lctx hotmod (v1.1) against a fake trainer namespace (CPU).
python3 -I test_lctx_hotmod.py HOTMOD_FILE [V1_HOTMOD_FILE]"""
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

MOD = sys.argv.pop(1)
MOD_V1 = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1].endswith('.py') else None
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
SRC = open(MOD, encoding='utf-8').read()
SRC_V1 = open(MOD_V1, encoding='utf-8').read() if MOD_V1 else None

TRAINER = r'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
_MM_CAUSAL = True
ATTN_T = 11.3137
ATTN_DTYPE = 'fp32'
def _attn_t(g=None):
    return ATTN_T
def _sdpa(q, k, v):
    if ATTN_T is not None:
        t = _attn_t()
        if t != 1.0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=_MM_CAUSAL, scale=t / math.sqrt(q.shape[-1]))
    return F.scaled_dot_product_attention(q, k, v, is_causal=_MM_CAUSAL)
STATUS_CALLS = []
def _rs_status(run, step, *a, **kw):
    STATUS_CALLS.append(step)
class Block(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        q, k = F.normalize(q, dim=-1) * math.sqrt(D // self.H), F.normalize(k, dim=-1)
        a = _sdpa(q, k, v).transpose(1, 2).reshape(B, T, D)
        return x + self.o(a)
class Model(nn.Module):
    def __init__(self, L, d, H):
        super().__init__()
        self.c = dict(L=L, d=d, H=H)
        self.blocks = nn.ModuleList(Block(d, H) for _ in range(L))
    def forward(self, x, ckpt=True):
        for b in self.blocks:
            x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False) if ckpt and torch.is_grad_enabled() else b(x)
        return x
'''


def make_trainer(L=3, d=32, H=4, seed=0):
    T = {'__name__': 'fake_trainer'}
    exec(compile(TRAINER, 'fake_trainer.py', 'exec'), T)
    T['_RS'] = types.SimpleNamespace(cur_step=1, pending=None)
    T['TO'] = types.SimpleNamespace(inv_scale=1.0)
    torch.manual_seed(seed)
    return T, T['Model'](L, d, H)


def install(T, model, run, step=0, extra=None, src=None):
    ns = {'__name__': 'hotmod_lctx', 'HOTMOD': dict(model=model, trainer=T, run=run, step=step,
                                                     extra_state=extra if extra is not None else {})}
    exec(compile(src or SRC, MOD, 'exec'), ns)
    return ns


def train_step(T, model, x, step, run, extra):
    """forward + backward + the trainer's save of state.json + the boundary (cur_step is the step in flight)"""
    T['_RS'].cur_step = step
    y = model(x)
    (y ** 2).mean().backward()
    with open(os.path.join(run, 'state.json'), 'w') as f:
        json.dump({'step': step, 'lctx': extra.get('lctx')}, f)
    T['_RS'].cur_step = step + 1
    T['_rs_status'](run, step)
    return y


def rate(x):
    return float(F.softplus(torch.tensor(float(x), dtype=torch.float64)))


class Hotmod(unittest.TestCase):
    def setUp(self):
        self.run = tempfile.mkdtemp(prefix='lctx_test_')
        os.makedirs(os.path.join(self.run, 'hotmods'))

    def tearDown(self):
        shutil.rmtree(self.run, ignore_errors=True)

    def x(self, seed=1, B=2, T_=24, d=32):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(B, T_, d, generator=g)

    def test_install_identity_updates_state_and_record(self):
        T, model = make_trainer()
        x = self.x()
        with torch.no_grad():
            base = model(x, ckpt=False)
        orig, orig_status = T['_sdpa'], T['_rs_status']
        extra = {}
        ns = install(T, model, self.run, extra=extra)
        self.assertTrue(ns['HOTMOD_RESULT']['installed'], ns['HOTMOD_RESULT'])
        self.assertIsNot(T['_sdpa'], orig)
        self.assertIs(T['_rs_status'].__kwdefaults__['_prev'], orig_status)  # chain walkers see through it
        self.assertIs(T['_sdpa'].__defaults__[0], orig)
        self.assertEqual(extra['lctx']['installed'], True)  # every launch records it, before any update
        with torch.no_grad():
            mod = model(x, ckpt=False)
        self.assertLess(float((mod - base).abs().max()), 1e-4)  # rate 1e-6: the model as it was
        S = T['_LCTX']
        r0 = S['st'].r.clone()
        for step in range(1, 7):
            train_step(T, model, x, step, self.run, extra)
        self.assertEqual(S['held'], 2)
        self.assertEqual(S['st'].updates, 4)
        self.assertGreater(float((S['st'].r - r0).abs().max()), 0.05)
        self.assertGreater(float(S['st'].w.abs().max()), 0.0)  # w learns from the first update (no rate gate)
        rec = extra['lctx']
        self.assertEqual(rec['updates'], 4)
        import hashlib
        self.assertEqual(rec['sha256'], hashlib.sha256(S['st'].params_bytes()).hexdigest())
        st = json.load(open(os.path.join(self.run, 'lctx.status.json')))
        self.assertEqual((st['updates'], st['gate_errors'], st['recompute_errors']), (4, 0, 0))
        self.assertEqual(len(os.listdir(os.path.join(self.run, 'lctx.ring'))), 4)
        self.assertFalse(os.path.exists(os.path.join(self.run, 'lctx.inflight')))
        snap = (S['st'].r.clone(), S['st'].updates, json.dumps(extra, sort_keys=True))
        mtime = os.path.getmtime(os.path.join(self.run, 'lctx.state.pt'))
        with torch.no_grad():
            model(x)  # a no_grad pass changes nothing (deveval's contract)
        self.assertTrue(torch.equal(S['st'].r, snap[0]))
        self.assertEqual((S['st'].updates, json.dumps(extra, sort_keys=True)), snap[1:])
        self.assertEqual(os.path.getmtime(os.path.join(self.run, 'lctx.state.pt')), mtime)

    def test_gradients_into_the_model_match_the_formula(self):
        T, model = make_trainer()
        ns = install(T, model, self.run)
        S = T['_LCTX']
        S['st'].r.fill_(float(ns['r_from_rate'](0.2)))
        S['st'].w.normal_(0, 0.3)
        x = self.x()
        ps = list(model.parameters())
        model(x).pow(2).mean().backward()
        g_mod = [p.grad.clone() for p in ps]
        for p in ps:
            p.grad = None

        def ref_sdpa(q, k, v):
            i = S['idx'][id(sys._getframe(1).f_locals['self'])]
            r, w = S['st'].layer_params(i, 'cpu')
            return ns['ref_attention'](q, k, v, r, w, 11.3137 / math.sqrt(q.shape[-1]), amax=6.0)

        T['_sdpa'] = ref_sdpa
        model(x).pow(2).mean().backward()
        for a, p in zip(g_mod, ps):
            self.assertLess(float((a - p.grad).abs().max()), 2e-5)

    def test_bidirectional_is_bit_identical(self):
        T, model = make_trainer()
        x = self.x()
        T['_MM_CAUSAL'] = False
        with torch.no_grad():
            base = model(x, ckpt=False)
        ns = install(T, model, self.run)
        T['_LCTX']['st'].r.fill_(float(ns['r_from_rate'](0.5)))
        with torch.no_grad():
            mod = model(x, ckpt=False)
        self.assertTrue(torch.equal(mod, base))
        self.assertEqual(T['_LCTX']['bidir_calls'], 3)

    def test_kill_switch_at_the_next_boundary_and_record(self):
        T, model = make_trainer()
        x = self.x()
        with torch.no_grad():
            base = model(x, ckpt=False)
        extra = {}
        ns = install(T, model, self.run, extra=extra)
        S = T['_LCTX']
        S['st'].r.fill_(float(ns['r_from_rate'](0.5)))
        open(os.path.join(self.run, 'hotmods', 'lctx.off'), 'w').close()
        with torch.no_grad():
            self.assertFalse(torch.equal(model(x, ckpt=False), base))  # still on inside this step
        T['_rs_status'](self.run, 1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x, ckpt=False), base))
        self.assertFalse(extra['lctx']['active'])
        os.remove(os.path.join(self.run, 'hotmods', 'lctx.off'))
        T['_rs_status'](self.run, 2)
        self.assertTrue(S['active'])
        self.assertTrue(extra['lctx']['active'])

    def test_two_strike_crash_marker(self):
        T, model = make_trainer()
        mk = os.path.join(self.run, 'lctx.inflight')
        with open(mk, 'w') as f:
            json.dump({'step': 5, 'post_crash': False}, f)
        ns = install(T, model, self.run)  # first strike: installs, flagged
        self.assertTrue(ns['HOTMOD_RESULT']['installed'])
        S = T['_LCTX']
        self.assertTrue(S['post_crash'])
        model(self.x())  # a grad forward writes this launch's marker with post_crash
        self.assertTrue(json.load(open(mk))['post_crash'])
        T2, model2 = make_trainer()  # this launch also dies mid-step: the next launch refuses
        extra2 = {}
        ns2 = install(T2, model2, self.run, extra=extra2)
        self.assertFalse(ns2['HOTMOD_RESULT']['installed'])
        self.assertIn('two launches in a row', ns2['HOTMOD_RESULT']['msg'])
        self.assertEqual(extra2['lctx']['installed'], False)  # the refusal is recorded too
        self.assertNotIn('_LCTX', T2)
        # a clean backward resets the strike count
        os.remove(mk)
        T3, model3 = make_trainer()
        with open(mk, 'w') as f:
            json.dump({'step': 9}, f)
        install(T3, model3, self.run)
        S3 = T3['_LCTX']
        model3(self.x()).pow(2).mean().backward()
        self.assertFalse(S3['post_crash'])
        self.assertFalse(os.path.exists(mk))
        open(os.path.join(self.run, 'hotmods', 'lctx.off'), 'w').close()
        T4, model4 = make_trainer()
        self.assertFalse(install(T4, model4, self.run)['HOTMOD_RESULT']['installed'])

    def test_pin_at_pause_and_rewind_rules(self):
        T, model = make_trainer()
        x = self.x()
        extra = {}
        ns = install(T, model, self.run, extra=extra)
        for step in range(1, 6):
            train_step(T, model, x, step, self.run, extra)
        S = T['_LCTX']
        rec5 = dict(extra['lctx'])
        T['_RS'].pending = 'pause'
        T['_rs_status'](self.run, 5)  # the boundary of a step that is about to be paused (an export)
        T['_RS'].pending = None
        pins = os.listdir(os.path.join(self.run, 'lctx.pinned'))
        self.assertEqual(pins, ['s000000005_u%07d.pt' % rec5['updates']])
        pin = torch.load(os.path.join(self.run, 'lctx.pinned', pins[0]))
        self.assertEqual(pin['sha256'], rec5['sha256'])
        rho_5 = S['st'].r.clone()
        for step in range(6, 9):
            train_step(T, model, x, step, self.run, extra)
        # relaunch from the step-5 checkpoint: its record says updates = rec5's -> the pinned / ring copy
        with open(os.path.join(self.run, 'state.json'), 'w') as f:
            json.dump({'step': 5, 'lctx': rec5}, f)
        T2, model2 = make_trainer()
        ns2 = install(T2, model2, self.run, step=5)
        self.assertIn('update %d' % rec5['updates'], ns2['HOTMOD_RESULT']['source'])
        self.assertTrue(torch.equal(T2['_LCTX']['st'].r, rho_5))
        newest = torch.load(os.path.join(self.run, 'lctx.state.pt'))['updates']
        T2['_RS'].cur_step = 6
        model2(x).pow(2).mean().backward()  # held
        model2(x).pow(2).mean().backward()  # held
        model2(x).pow(2).mean().backward()  # first update of this launch: the newer state is kept aside, not lost
        self.assertTrue(any(n.startswith('lctx.state.pt.prev-') for n in os.listdir(self.run)))
        kept = [n for n in os.listdir(self.run) if n.startswith('lctx.state.pt.prev-')][0]
        self.assertEqual(torch.load(os.path.join(self.run, kept))['updates'], newest)
        # a record with no saved copy of its update count: the newest state, never a silent reset
        with open(os.path.join(self.run, 'state.json'), 'w') as f:
            json.dump({'step': 3, 'lctx': dict(rec5, updates=999)}, f)
        T3, model3 = make_trainer()
        ns3 = install(T3, model3, self.run, step=3)
        self.assertTrue(ns3['HOTMOD_RESULT']['source'].startswith('NEWEST'))
        # no record: the checkpoint predates the mod -> fresh
        with open(os.path.join(self.run, 'state.json'), 'w') as f:
            json.dump({'step': 0}, f)
        T4, model4 = make_trainer()
        ns4 = install(T4, model4, self.run, step=0)
        self.assertIn('no lctx record', ns4['HOTMOD_RESULT']['source'])
        # 3 -> 5 layers keeps the saved layers
        with open(os.path.join(self.run, 'state.json'), 'w') as f:
            json.dump({'step': 5, 'lctx': rec5}, f)
        T5, model5 = make_trainer(L=5)
        install(T5, model5, self.run, step=5)
        st5 = T5['_LCTX']['st']
        self.assertTrue(torch.equal(st5.r[:3], rho_5))
        self.assertTrue(bool((st5.r[3:] == st5.r[3, 0]).all()))

    def test_forward_failure_inside_the_attention_falls_back_for_the_step(self):
        T, model = make_trainer()
        x = self.x()
        with torch.no_grad():
            base = model(x, ckpt=False)
        ns = install(T, model, self.run)
        S = T['_LCTX']
        real = ns['_lc_kernel']
        calls = {'n': 0}

        def flaky(*a, **kw):  # inside LctxAttn.forward, AFTER the gate ran (the 16 Nov review's scenario)
            if torch._C._current_graph_task_id() == -1:
                calls['n'] += 1
                if calls['n'] == 2:
                    raise RuntimeError('simulated out of memory in the forward')
            return real(*a, **kw)

        ns['_lc_kernel'] = flaky
        ns['HGROUP'] = 2
        model(x).pow(2).mean().backward()  # checkpointed: no CheckpointError, the recompute takes the same path
        self.assertEqual(S['fallback'], {0})
        T['_rs_status'](self.run, 1)
        self.assertEqual(S['fallback'], set())
        self.assertEqual(S['fail_streak'], 1)
        ns['_lc_kernel'] = real
        for step in (2, 3):
            calls['n'] = 0
            ns['_lc_kernel'] = flaky
            model(x).pow(2).mean().backward()
            T['_rs_status'](self.run, step)
        self.assertTrue(S['disabled'])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x, ckpt=False), base))

    def test_recompute_failure_inside_the_attention_retries(self):
        T, model = make_trainer()
        ns = install(T, model, self.run)
        S = T['_LCTX']
        S['st'].r.fill_(float(ns['r_from_rate'](0.2)))
        x = self.x()
        ps = list(model.parameters())
        model(x).pow(2).mean().backward()
        ref = [p.grad.clone() for p in ps]
        for p in ps:
            p.grad = None
        real = ns['_lc_kernel']
        state = {'n': 0}

        def flaky(*a, **kw):  # fails once inside LctxAttn.forward during a checkpoint recompute
            if torch._C._current_graph_task_id() != -1 and state['n'] == 0 and a[0].shape[1] > 1:
                state['n'] += 1
                raise RuntimeError('simulated out of memory in the recompute')
            return real(*a, **kw)

        ns['_lc_kernel'] = flaky
        model(x).pow(2).mean().backward()
        self.assertEqual(S['recompute_errors'], 1)
        for a, p in zip(ref, ps):
            self.assertLess(float((a - p.grad).abs().max()), 1e-6)

    def test_layer_found_through_an_extra_wrapper(self):
        T, model = make_trainer()
        install(T, model, self.run)
        inner = T['_sdpa']

        def outer(q, k, v):  # another mod wrapping _sdpa later
            return inner(q, k, v)

        T['_sdpa'] = outer
        with torch.no_grad():
            model(self.x(), ckpt=False)
        self.assertEqual(T['_LCTX']['skipped_calls'], 0)
        self.assertEqual(T['_LCTX']['calls'], 3)

    @unittest.skipIf(SRC_V1 is None, 'no v1 hotmod given')
    def test_hot_swap_from_v1_in_process(self):
        T, model = make_trainer()
        orig, orig_status = T['_sdpa'], T['_rs_status']
        extra = {}
        install(T, model, self.run, extra=extra, src=SRC_V1)
        S1 = T['_LCTX']
        x = self.x()
        for step in range(1, 5):  # v1: 2 held + 2 updates
            T['_RS'].cur_step = step
            model(x).pow(2).mean().backward()
            T['_rs_status'](self.run, step)
        self.assertEqual(S1['st'].updates, 2)
        lam_v1 = S1['st'].rho.double().exp()
        ns = install(T, model, self.run, step=4, extra=extra)  # the new file at a boundary, same process
        S2 = T['_LCTX']
        self.assertTrue(ns['HOTMOD_RESULT']['installed'], ns['HOTMOD_RESULT'])
        self.assertIn('carried over', ns['HOTMOD_RESULT']['source'])
        self.assertTrue(S1['retired'])
        self.assertIs(S2['prev'], orig)
        self.assertIs(T['_rs_status'].__kwdefaults__['_prev'], orig_status)  # the v1 wrapper left the chain
        self.assertLess(float((F.softplus(S2['st'].r.double()) - lam_v1).abs().div(lam_v1).max()), 1e-5)
        self.assertEqual(S2['st'].updates, 2)
        for step in range(5, 9):
            train_step(T, model, x, step, self.run, extra)
        self.assertEqual(S2['st'].updates, 4)
        self.assertEqual(extra['lctx']['version'], ns['VG'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
