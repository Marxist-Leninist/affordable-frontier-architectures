"""lctx_serve.py -- serve a pack trained with the lctx hotmod (learned context length) the way it was trained.

The trainer's causal attention is   softmax_j( s q_i.k_j + kb_j - kb_i )  over j <= i,  kb = running sum over tokens of
    a_t = min(softplus(r_h + w_h . unit(v_t)), amax)        (per layer, per head; v_t = the head's value)
Bidirectional attention (NAT, diffusion) is unchanged.

Serving needs, per layer: the rates of every token as it enters the KV cache (from its v), the running sum kb per head
next to the cache (float32 whatever the cache dtype), and the bias kb_j - kb_i on every causal logit.  Keys with
kb_i - kb_j > theta can be EVICTED for good (kb never decreases): a head that learned a short context keeps a short cache.

attend() is the one call for a prompt chunk or a decode token: queries t0 .. t0+T-1 over keys 0 .. t0+T-1 (the cache
including the chunk), bottom-right causal (key j visible to query i iff j <= t0 + i; never SDPA's top-left is_causal
when T < S), the bias recentred per row in float32 and handed to SDPA as attn_mask in the query dtype.

Pack files (written by tools/export_lctx.py next to the pack):
    lctx.json        {"version", "form", "glue", "trainer_step", "updates", "L", "H", "E", "amax", "sha256", "params_bytes"}
    lctx.params.f32  r (L*H) + w (L*H*E), little-endian float32, in that order (sha256 = the trainer record's)
"""
import hashlib
import json
import os

import torch
import torch.nn.functional as F

VERSION = 'lctx-serve-v2-20261116'
FORM = 'softplus(r+w.unit(v))'


class LctxServe:
    def __init__(self, r, w, amax, meta=None):
        self.r, self.w, self.amax = r.float(), w.float(), float(amax)
        self.L, self.H = self.r.shape
        self.E = self.w.shape[-1]
        self.meta = dict(meta or {})
        self._dev = {}

    # ---- files
    @staticmethod
    def blob(r, w):
        return torch.cat([r.float().flatten(), w.float().flatten()]).numpy().astype('<f4').tobytes()

    @staticmethod
    def write(pack_dir, r, w, amax, info):
        L, H = r.shape
        E = w.shape[-1]
        blob = LctxServe.blob(r, w)
        meta = dict(info, version=VERSION, form=FORM, L=int(L), H=int(H), E=int(E), amax=float(amax),
                    sha256=hashlib.sha256(blob).hexdigest(), params_bytes=len(blob))
        tmp = os.path.join(pack_dir, 'lctx.params.f32.tmp')
        with open(tmp, 'wb') as f:
            f.write(blob)
        os.replace(tmp, os.path.join(pack_dir, 'lctx.params.f32'))
        tmp = os.path.join(pack_dir, 'lctx.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(pack_dir, 'lctx.json'))
        return meta

    @classmethod
    def from_pack(cls, pack_dir, expect=None):
        """None when the pack has no lctx files and its checkpoint (expect = the pack's state.json['lctx']) was not
        trained with an active gate; raises when the files are inconsistent, missing although required, or not the
        checkpoint's own (update count, geometry and, from v1.1 records on, the parameter sha256)"""
        jp, bp = os.path.join(pack_dir, 'lctx.json'), os.path.join(pack_dir, 'lctx.params.f32')
        if not os.path.exists(jp):
            if expect and int(expect.get('updates') or 0) > 0:
                raise ValueError('the checkpoint was trained with lctx (%s updates) but the pack has no lctx.json; '
                                 'run tools/export_lctx.py on it' % expect.get('updates'))
            return None
        with open(jp) as f:
            meta = json.load(f)
        if meta.get('form') != FORM:
            raise ValueError('lctx.json form %r is not %r' % (meta.get('form'), FORM))
        with open(bp, 'rb') as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != meta['sha256'] or len(blob) != meta['params_bytes']:
            raise ValueError('lctx.params.f32 does not match lctx.json')
        L, H, E = int(meta['L']), int(meta['H']), int(meta['E'])
        if len(blob) != 4 * L * H * (E + 1):
            raise ValueError('lctx.params.f32 has the wrong size for its geometry')
        if expect:
            if int(expect.get('updates', -1)) != int(meta.get('updates', -2)) or \
                    list(expect.get('geometry') or []) != [L, H, E]:
                raise ValueError('lctx files are not the checkpoint\'s own (updates/geometry differ)')
            if expect.get('sha256') and expect['sha256'] != meta['sha256']:
                raise ValueError('lctx parameters differ from the checkpoint\'s record (sha256)')
        t = torch.frombuffer(bytearray(blob), dtype=torch.float32)
        n1 = L * H
        r, w = t[:n1].view(L, H).clone(), t[n1:].view(L, H, E).clone()
        if not (bool(torch.isfinite(r).all()) and bool(torch.isfinite(w).all())):
            raise ValueError('lctx parameters are not finite')
        return cls(r, w, meta['amax'], meta)

    # ---- math
    def params(self, layer, device):
        key = (layer, str(device))
        p = self._dev.get(key)
        if p is None:
            p = (self.r[layer].to(device), self.w[layer].to(device))
            self._dev[key] = p
        return p

    def rates(self, layer, v):
        """a (B, H, T) float32 for value vectors v (B, H, T, E) of this layer"""
        r, w = self.params(layer, v.device)
        vf = v.float()
        u = vf / torch.linalg.vector_norm(vf, dim=-1, keepdim=True).clamp_min(1e-12)
        x = torch.einsum('bhtd,hd->bht', u, w) + r[None, :, None]
        return torch.clamp(F.softplus(x), max=self.amax)

    def attend(self, layer, q, K, V, kb_prev, t0, scale, hgroup=8):
        """queries q (B, H, T, E) at positions t0 .. t0+T-1 over the cache K, V (B, H, S = t0+T, E) that already holds
        this chunk; kb_prev (B, H, t0) float32 (None when t0 = 0).  Returns (out (B, H, T, Ev), kb_new (B, H, T))."""
        B, H, T, E = q.shape
        S = K.shape[2]
        if S != t0 + T:
            raise ValueError('the cache must hold exactly keys 0 .. t0+T-1 (S=%d, t0=%d, T=%d)' % (S, t0, T))
        a = self.rates(layer, V[:, :, t0:])
        base = kb_prev[..., -1:].float() if t0 > 0 else torch.zeros(B, H, 1, device=q.device)
        kb_new = base + torch.cumsum(a, dim=-1)
        kb_all = kb_new if t0 == 0 else torch.cat([kb_prev.float(), kb_new], dim=-1)
        vis = torch.arange(S, device=q.device)[None, :] <= (t0 + torch.arange(T, device=q.device))[:, None]
        out = q.new_empty((B, H, T, V.shape[-1]))
        hg = max(1, min(int(hgroup), H))
        for h0 in range(0, H, hg):
            h1 = min(H, h0 + hg)
            bias = (kb_all[:, h0:h1, None, :] - kb_new[:, h0:h1, :, None])  # <= 0 on visible keys, 0 at the diagonal
            bias = bias.masked_fill(~vis, float('-inf')).to(q.dtype)
            out[:, h0:h1] = F.scaled_dot_product_attention(q[:, h0:h1], K[:, h0:h1], V[:, h0:h1], attn_mask=bias,
                                                           scale=scale)
            del bias
        return out, kb_new

    @staticmethod
    def evictable(kb_keys, theta=40.0):
        """bool (B, H, n): keys that no current or later query of that head can give more than e^(2T - theta) weight"""
        return (kb_keys[..., -1:] - kb_keys) > theta
