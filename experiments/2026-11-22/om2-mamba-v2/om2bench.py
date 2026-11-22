#!/usr/bin/env python3
"""OM2 study (Claude lane, 22 Nov 2026): our Mamba v2 and a state-gated retrieval hybrid.

Matched CPU benchmark. One process = one (arm, seed) run on one CPU thread.
Data: the P056 human-only three-domain corpus (article / fiction / dialogue),
copied read-only into ./data (train.bin sha256 pinned below).

Arms (byte level, 4 layers, d=128, tied embeddings, same data stream per seed):
  tf      Transformer: RoPE, 4 heads, GELU MLP (ff picked to match parameters), KV-cached decode
  m2hf    Hugging Face transformers Mamba2ForCausalLM (torch path, no CUDA kernels)
  om1     our Mamba v1 (P055/P057 own.py mixer and doubling scan, unchanged, scaled up)
  om2     our Mamba v2: chunked scalar-decay (SSD-style) core with 4x the state of om1,
          signed decay (transitions in (-1,1)), RMS-normalised B and C, conv on x/B/C,
          gated output norm, per-head skip
  om2pos  om2 with positive-only decay (ablation of the signed decay)
  om2p    om2 + placebo sidecar: same parameters as the store, each token reads only itself
  om2a    om2 + always-on bounded exact-token store (one-head attention over the last 256 tokens)
  om2g    om2 + state-gated store: a gate on the Mamba output decides per token whether to
          read the store; sparsity cost on the gate; evaluation skips reads with gate <= 0.5
"""
import argparse, base64, hashlib, json, math, os, random, resource, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('OM2_DATA', str(HERE / 'data')))
TRAIN_SHA = 'e58bc4084cfc06c5cf96ba65fb1fb38a244a691d2f09d389b09603eba92c8552'
D_MODEL, LAYERS, VOCAB = 128, 4, 256
ARMS = ('tf', 'm2hf', 'm2', 'om1', 'om2', 'om2pos', 'om2p', 'om2a', 'om2g')
DOMAINS = ('article', 'fiction', 'dialogue')
LN2 = math.log(2.0)


def sha(b):
    return hashlib.sha256(b).hexdigest()


class RMS(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# ----------------------------------------------------------------------------- Transformer
def rope_tables(T, dh, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
    f = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.H, self.dh = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

    def step(self, x, pos, cache, cos, sin):
        B = x.shape[0]
        q, k, v = self.qkv(x).view(B, 1, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        c, s = cos[pos:pos + 1], sin[pos:pos + 1]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)
        if cache is not None:
            k = torch.cat((cache[0], k), 2)
            v = torch.cat((cache[1], v), 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).reshape(B, 1, -1)), (k, v)


class TFBlock(nn.Module):
    def __init__(self, d, H, ff):
        super().__init__()
        self.n1, self.att, self.n2 = RMS(d), Attn(d, H), RMS(d)
        self.f1, self.f2 = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.att(self.n1(x), cos, sin)
        return x + self.f2(F.gelu(self.f1(self.n2(x))))

    def step(self, x, pos, cache, cos, sin):
        a, cache = self.att.step(self.n1(x), pos, cache, cos, sin)
        x = x + a
        return x + self.f2(F.gelu(self.f1(self.n2(x)))), cache


class TF(nn.Module):
    def __init__(self, ff, d=D_MODEL, L=LAYERS, H=4, maxT=1024):
        super().__init__()
        self.e = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.blocks = nn.ModuleList(TFBlock(d, H, ff) for _ in range(L))
        self.n = RMS(d)
        c, s = rope_tables(maxT, d // H)
        self.register_buffer('cos', c, persistent=False)
        self.register_buffer('sin', s, persistent=False)

    def forward(self, x, hard=False):
        h = self.e(x)
        for b in self.blocks:
            h = b(h, self.cos, self.sin)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        pos, caches = state if state is not None else (0, [None] * len(self.blocks))
        h = self.e(tok)[:, None]
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step(h, pos, c, self.cos, self.sin)
            new.append(c)
        return F.linear(self.n(h), self.e.weight)[:, 0], (pos + 1, new)


# ----------------------------------------------------------------------------- our Mamba v1
def scan_doubling(a, b):
    """P055/P057 own.py scan, unchanged: inclusive affine scan by doubling."""
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        if k >= a.shape[1]:
            break
        b = torch.cat([b[:, :k], b[:, k:] + a[:, k:] * b[:, :-k]], dim=1)
        a = torch.cat([a[:, :k], a[:, k:] * a[:, :-k]], dim=1)
    return b


class OM1Mixer(nn.Module):
    """P055/P057 own.py Mixer, unchanged apart from its size arguments."""

    def __init__(self, d=D_MODEL, m=2 * D_MODEL, n=8, k=4, r=8):
        super().__init__()
        self.m, self.n, self.k = m, n, k
        self.input = nn.Linear(d, 2 * m, bias=False)
        self.dw = nn.Conv1d(m, m, k, groups=m)
        self.param = nn.Linear(m, r + 2 * n, bias=False)
        self.dt = nn.Linear(r, m)
        self.log_rates = nn.Parameter(torch.arange(1, n + 1, dtype=torch.float32).log()[None].repeat(m, 1))
        self.skip = nn.Parameter(torch.ones(m))
        self.output = nn.Linear(m, d, bias=False)
        with torch.no_grad():
            self.dt.weight.uniform_(-r ** -.5, r ** -.5)
            u = torch.rand(m) * (math.log(.1) - math.log(.001)) + math.log(.001)
            dt = u.exp().clamp_min(1e-4)
            self.dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def transitions(self, u):
        q, b, c = self.param(u).split((self.dt.in_features, self.n, self.n), -1)
        dt = F.softplus(self.dt(q))
        lam = self.log_rates.exp().clamp_max(1e4)
        a = torch.exp(-dt[..., None] * lam)
        update = (dt * u)[..., None] * b.unsqueeze(-2)
        return a, update, c

    def forward(self, x):
        v, g = self.input(x).chunk(2, -1)
        u = F.silu(self.dw(F.pad(v.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2))
        a, b, c = self.transitions(u)
        states = scan_doubling(a, b)
        y = (states * c.unsqueeze(-2)).sum(-1) + u * self.skip
        return self.output(y * F.silu(g))

    def step(self, x, cached=None):
        v, g = self.input(x).chunk(2, -1)
        if cached is None:
            conv = v.new_zeros(v.shape[0], self.m, self.k - 1)
            ss = v.new_zeros(v.shape[0], self.m, self.n)
        else:
            conv, ss = cached
        wind = torch.cat((conv, v.unsqueeze(-1)), -1)
        u = F.silu((wind * self.dw.weight.squeeze(1)).sum(-1) + self.dw.bias)
        a, b, c = self.transitions(u)
        ss = a * ss + b
        y = (ss * c.unsqueeze(-2)).sum(-1) + u * self.skip
        return self.output(y * F.silu(g)), (wind[:, :, 1:], ss)


# ----------------------------------------------------------------------------- our Mamba v2
_MASKS = {}


def _masks(T):
    if T not in _MASKS:
        _MASKS[T] = (torch.tril(torch.ones(T, T, dtype=torch.bool), -1),
                     torch.tril(torch.ones(T, T, dtype=torch.bool), 0))
    return _MASKS[T]


def segsum(x):
    """out[..., i, j] = sum_{j<k<=i} x[..., k] for i >= j, -inf above the diagonal (stable form)."""
    T = x.size(-1)
    strict, incl = _masks(T)
    x = x[..., None].expand(*x.shape, T).masked_fill(~strict, 0.0)
    return torch.cumsum(x, -2).masked_fill(~incl, float('-inf'))


class OM2Mixer(nn.Module):
    """Scalar decay per head, state H x P x N, computed exactly in chunks with matmuls.

    signed=True lets every transition lie in (-1, 1): a = 2 exp(-dt A) - 1. Products of
    transitions are carried as exp(sum log|a|) times a parity sign, so the chunked form
    stays exact (checked against the recurrent step in the parity gate).
    """

    def __init__(self, d=D_MODEL, expand=2, P=32, N=32, k=4, Q=32, signed=True, bcnorm=True):
        super().__init__()
        di = expand * d
        H = di // P
        self.di, self.H, self.P, self.N, self.k, self.Q, self.signed = di, H, P, N, k, Q, signed
        self.inp = nn.Linear(d, 2 * di + 2 * N + H, bias=False)
        cw = di + 2 * N
        self.conv = nn.Conv1d(cw, cw, k, groups=cw)
        self.bn, self.cn = (RMS(N), RMS(N)) if bcnorm else (nn.Identity(), nn.Identity())
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.Dskip = nn.Parameter(torch.ones(H))
        self.gn = RMS(di)
        self.out = nn.Linear(di, d, bias=False)

    def _split(self, zxbcdt):
        return zxbcdt.split((self.di, self.di + 2 * self.N, self.H), -1)

    def _post_conv(self, xbc):
        x, Bm, Cm = xbc.split((self.di, self.N, self.N), -1)
        return x, self.bn(Bm), self.cn(Cm)

    def _decay(self, dt):
        A = self.A_log.exp()
        if self.signed:
            a = 2.0 * torch.exp(-dt * A) - 1.0
            return a, torch.log(a.abs().clamp_min(1e-6)), (a < 0).to(dt.dtype)
        la = -dt * A
        return la.exp(), la, None

    def forward(self, u):
        b, T, _ = u.shape
        z, xbc, dt = self._split(self.inp(u))
        xbc = F.silu(self.conv(F.pad(xbc.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2))
        x, Bm, Cm = self._post_conv(xbc)
        dt = F.softplus(dt + self.dt_bias)
        _, la, neg = self._decay(dt)
        xh = x.view(b, T, self.H, self.P)
        y = self._ssd(xh, dt, la, neg, Bm, Cm) + xh * self.Dskip[:, None]
        return self.out(self.gn(y.reshape(b, T, self.di) * F.silu(z)))

    def _ssd(self, x, dt, la, neg, Bm, Cm):
        b, T, H, P = x.shape
        Q, N = self.Q, self.N
        pad = (-T) % Q
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt, la = F.pad(dt, (0, 0, 0, pad)), F.pad(la, (0, 0, 0, pad))
            Bm, Cm = F.pad(Bm, (0, 0, 0, pad)), F.pad(Cm, (0, 0, 0, pad))
            if neg is not None:
                neg = F.pad(neg, (0, 0, 0, pad))
        Tp = T + pad
        c = Tp // Q
        X = (x * dt[..., None]).view(b, c, Q, H, P)
        A = la.view(b, c, Q, H).permute(0, 3, 1, 2)          # b H c Q
        Bc, Cc = Bm.view(b, c, Q, N), Cm.view(b, c, Q, N)
        Acs = A.cumsum(-1)
        Lm = torch.exp(segsum(A))                               # b H c Q Q
        s = None
        if neg is not None:
            par = torch.remainder(neg.view(b, c, Q, H).permute(0, 3, 1, 2).cumsum(-1), 2.0)
            s = 1.0 - 2.0 * par                                 # sign of prod a[0..l] in chunk
            Lm = Lm * (s[..., :, None] * s[..., None, :])
        CB = torch.einsum('bcln,bcsn->bcls', Cc, Bc)
        Yd = torch.einsum('bcls,bhcls,bcshp->bclhp', CB, Lm, X)
        ds = torch.exp(Acs[..., -1:] - Acs)
        if s is not None:
            ds = ds * (s[..., -1:] * s)
        st = torch.einsum('bcln,bhcl,bclhp->bchpn', Bc, ds, X)
        dc = torch.exp(segsum(F.pad(Acs[..., -1], (1, 0))))   # b H c+1 c+1
        if s is not None:
            cp = torch.cumprod(F.pad(s[..., -1], (1, 0), value=1.0), -1)
            dc = dc * (cp[..., :, None] * cp[..., None, :])
        st = torch.cat((torch.zeros_like(st[:, :1]), st), 1)
        st_in = torch.einsum('bhzc,bchpn->bzhpn', dc, st)[:, :-1]
        od = torch.exp(Acs)
        if s is not None:
            od = od * s
        Yo = torch.einsum('bcln,bchpn,bhcl->bclhp', Cc, st_in, od)
        return (Yd + Yo).reshape(b, Tp, H, P)[:, :T]

    def step(self, u, cache=None):
        b = u.shape[0]
        z, xbc, dt = self._split(self.inp(u))
        if cache is None:
            conv = u.new_zeros(b, xbc.shape[-1], self.k - 1)
            h = u.new_zeros(b, self.H, self.P, self.N)
        else:
            conv, h = cache
        win = torch.cat((conv, xbc[..., None]), -1)
        xbc = F.silu((win * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        x, Bm, Cm = self._post_conv(xbc)
        dt = F.softplus(dt + self.dt_bias)
        a, _, _ = self._decay(dt)
        xh = x.view(b, self.H, self.P)
        h = a[..., None, None] * h + (dt[..., None] * xh)[..., None] * Bm[:, None, None, :]
        y = (h * Cm[:, None, None, :]).sum(-1) + xh * self.Dskip[:, None]
        return self.out(self.gn(y.reshape(b, self.di) * F.silu(z))), (win[..., 1:], h)


class Store(nn.Module):
    """Bounded exact-token store read by one attention head (keys/values of the last W tokens).

    mode 'gated':   read weight g = sigmoid(w . h); sparsity cost on mean g; eval may skip g <= tau
    mode 'always':  read every token (no gate)
    mode 'placebo': same parameters and gate, but each token reads only its own value (no lookup)
    """

    def __init__(self, d=D_MODEL, dk=64, W=256, mode='gated', tau=0.5):
        super().__init__()
        self.mode, self.W, self.tau, self.dk = mode, W, tau, dk
        self.n = RMS(d)
        self.q = nn.Linear(d, dk, bias=False)
        self.k = nn.Linear(d, dk, bias=False)
        self.v = nn.Linear(d, dk, bias=False)
        self.o = nn.Linear(dk, d, bias=False)
        nn.init.zeros_(self.o.weight)
        self.g = nn.Linear(d, 1)
        nn.init.zeros_(self.g.bias)
        self.reset_stats()

    def reset_stats(self):
        self.stat_open, self.stat_total, self.stat_gsum = 0, 0, 0.0

    def _gate(self, hn):
        return None if self.mode == 'always' else torch.sigmoid(self.g(hn))

    def forward(self, h, hard=False):
        hn = self.n(h)
        v = self.v(hn)
        g = self._gate(hn)
        if self.mode == 'placebo':
            r = v
        else:
            q, k = self.q(hn), self.k(hn)
            T = h.shape[1]
            mask = None
            if T > self.W:
                i = torch.arange(T)
                mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.W)
            r = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], attn_mask=mask,
                                               is_causal=mask is None)[:, 0]
        pen = None
        if g is not None:
            pen = g.mean()
            if not self.training:
                self.stat_open += int((g > self.tau).sum())
                self.stat_total += g.numel()
                self.stat_gsum += float(g.sum())
            if hard:
                g = g * (g > self.tau).to(g.dtype)
            r = g * r
        return h + self.o(r), pen

    def step(self, h, sc, hard=True):
        hn = self.n(h)
        v = self.v(hn)
        g = self._gate(hn)
        if self.mode == 'placebo':
            gg = g * (g > self.tau).to(g.dtype) if hard else g
            return h + self.o(gg * v), None
        k = self.k(hn)
        if sc is None:
            K, V = k[:, None], v[:, None]
        else:
            K = torch.cat((sc[0], k[:, None]), 1)[:, -self.W:]
            V = torch.cat((sc[1], v[:, None]), 1)[:, -self.W:]
        if g is not None and hard and bool((g <= self.tau).all()):
            return h, (K, V)                      # gate closed: the lookup is skipped entirely
        q = self.q(hn)
        att = torch.softmax((K @ q[:, :, None])[..., 0] / math.sqrt(self.dk), -1)
        r = (att[..., None] * V).sum(1)
        if g is not None:
            r = (g * (g > self.tau).to(g.dtype) if hard else g) * r
        return h + self.o(r), (K, V)


class SSMLM(nn.Module):
    def __init__(self, mixers, store=None, store_after=2, d=D_MODEL):
        super().__init__()
        self.e = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.mix = nn.ModuleList(mixers)
        self.norms = nn.ModuleList(RMS(d) for _ in mixers)
        self.n = RMS(d)
        self.store, self.store_after = store, store_after
        self.last_penalty = None

    def forward(self, x, hard=False):
        h = self.e(x)
        pen = None
        for i, (m, nm) in enumerate(zip(self.mix, self.norms)):
            h = h + m(nm(h))
            if self.store is not None and i + 1 == self.store_after:
                h, pen = self.store(h, hard=hard)
        self.last_penalty = pen
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        h = self.e(tok)
        caches, sc = state if state is not None else ([None] * len(self.mix), None)
        new = []
        for i, (m, nm, c) in enumerate(zip(self.mix, self.norms, caches)):
            z, c = m.step(nm(h), c)
            h = h + z
            new.append(c)
            if self.store is not None and i + 1 == self.store_after:
                h, sc = self.store.step(h, sc, hard=hard)
        return F.linear(self.n(h), self.e.weight), (new, sc)


class HFM2(nn.Module):
    """transformers Mamba2ForCausalLM as shipped (torch path), sized like om2."""

    def __init__(self):
        super().__init__()
        from transformers import Mamba2Config, Mamba2ForCausalLM
        cfg = Mamba2Config(vocab_size=VOCAB, hidden_size=D_MODEL, state_size=32, num_heads=8, head_dim=32,
                           expand=2, n_groups=1, conv_kernel=4, num_hidden_layers=LAYERS, chunk_size=32,
                           use_cache=False, tie_word_embeddings=True, pad_token_id=0, bos_token_id=0,
                           eos_token_id=0)
        self.m = Mamba2ForCausalLM(cfg)

    def forward(self, x, hard=False):
        return self.m(input_ids=x, use_cache=False).logits

    def step(self, tok, state, hard=True):
        if state is None:
            out = self.m(input_ids=tok[:, None], use_cache=True)
            return out.logits[:, -1], (out.cache_params, 1)
        cache, pos = state
        out = self.m(input_ids=tok[:, None], cache_params=cache, use_cache=True,
                     cache_position=torch.tensor([pos]))
        return out.logits[:, -1], (out.cache_params, pos + 1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


TF_FF = int(os.environ.get('OM2_TF_FF', '176'))


def m2_from_hf():
    """Mamba-2 exactly as transformers defines and initialises it, computed with the om2 chunked kernel.

    The HF model is built first (its own init), its weights are copied in, and the logits of both are
    compared on real text; the max abs difference is returned and must stay below 1e-4.
    """
    hf = HFM2()
    m = SSMLM([OM2Mixer(signed=False, bcnorm=False) for _ in range(LAYERS)])
    sd = hf.m.state_dict()
    with torch.no_grad():
        m.e.weight.copy_(sd['backbone.embeddings.weight'])
        m.n.w.copy_(sd['backbone.norm_f.weight'])
        for i, (mx, nm) in enumerate(zip(m.mix, m.norms)):
            p = f'backbone.layers.{i}.'
            nm.w.copy_(sd[p + 'norm.weight'])
            mx.inp.weight.copy_(sd[p + 'mixer.in_proj.weight'])
            mx.conv.weight.copy_(sd[p + 'mixer.conv1d.weight'])
            mx.conv.bias.copy_(sd[p + 'mixer.conv1d.bias'])
            mx.dt_bias.copy_(sd[p + 'mixer.dt_bias'])
            mx.A_log.copy_(sd[p + 'mixer.A_log'])
            mx.Dskip.copy_(sd[p + 'mixer.D'])
            mx.gn.w.copy_(sd[p + 'mixer.norm.weight'])
            mx.out.weight.copy_(sd[p + 'mixer.out_proj.weight'])
    _, _, test, _ = load_all()
    x = torch.from_numpy(np.stack([test[i]['data'][:256] for i in range(4)]))
    with torch.no_grad():
        hf.eval(), m.eval()
        diff = float((hf(x) - m(x)).abs().max())
    m.train()
    assert diff < 1e-4, f'om2 kernel does not reproduce HF Mamba-2: {diff}'
    m.hf_equivalence_max_abs = diff
    return m


def build(arm):
    if arm == 'tf':
        return TF(TF_FF)
    if arm == 'm2hf':
        return HFM2()
    if arm == 'm2':
        return m2_from_hf()
    if arm == 'om1':
        return SSMLM([OM1Mixer() for _ in range(LAYERS)])
    signed = arm != 'om2pos'
    mixers = [OM2Mixer(signed=signed) for _ in range(LAYERS)]
    store = {'om2p': 'placebo', 'om2a': 'always', 'om2g': 'gated'}.get(arm)
    return SSMLM(mixers, store=Store(mode=store) if store else None)


# ----------------------------------------------------------------------------- data and evaluation
def load_windows(name):
    rows = json.loads((DATA / name).read_text())
    out, bad = [], 0
    for r in rows:
        raw = base64.b64decode(r['bytes_b64'])
        bad += int(r.get('sha256') not in (None, sha(raw)))
        out.append(dict(domain=r['domain'], doc=r['doc_id'], ctx=int(r['context']),
                        data=np.frombuffer(raw, np.uint8).astype(np.int64)))
    return out, bad


def load_all():
    raw = (DATA / 'train.bin').read_bytes()
    assert sha(raw) == TRAIN_SHA, 'train.bin hash mismatch'
    dev, bad_d = load_windows('dev_windows.json')
    test, bad_t = load_windows('test_windows.json')
    return np.frombuffer(raw, np.uint8), dev, test, dict(dev_sha_mismatch=bad_d, test_sha_mismatch=bad_t,
                                                          dev_file_sha=sha((DATA / 'dev_windows.json').read_bytes()),
                                                          test_file_sha=sha((DATA / 'test_windows.json').read_bytes()))


def eval_plan(wins, T, S=128):
    """Every target byte q >= ctx is scored exactly once with T-S+1 .. T bytes of context."""
    items = []
    for wi, w in enumerate(wins):
        Lw, q = len(w['data']), w['ctx']
        while q < Lw:
            s = max(0, min(q - (T - S + 1), Lw - 1 - T))
            qe = min(s + T, Lw - 1)
            items.append((wi, s, q, qe))
            q = qe + 1
    return items


@torch.no_grad()
def evaluate(model, wins, items, T, bs=32, hard=False):
    model.eval()
    nll = [np.zeros(len(w['data'])) for w in wins]
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        x = torch.from_numpy(np.stack([wins[wi]['data'][s:s + T] for wi, s, _, _ in ch]))
        y = torch.from_numpy(np.stack([wins[wi]['data'][s + 1:s + T + 1] for wi, s, _, _ in ch]))
        lg = model(x, hard=hard)
        l = F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1), reduction='none').view(len(ch), T).numpy()
        for j, (wi, s, q0, q1) in enumerate(ch):
            nll[wi][q0:q1 + 1] = l[j, q0 - s - 1:q1 - s]
    docs = []
    for w, l in zip(wins, nll):
        v = l[w['ctx']:]
        docs.append(dict(domain=w['domain'], doc=w['doc'], targets=int(len(v)), nll=float(v.sum())))
    return docs


def domain_bpb(docs):
    out = {}
    for dm in DOMAINS:
        t = sum(d['targets'] for d in docs if d['domain'] == dm)
        if t:
            out[dm] = sum(d['nll'] for d in docs if d['domain'] == dm) / t / LN2
    out['macro'] = statistics.mean(out[dm] for dm in DOMAINS if dm in out)
    return out


def copy_probes(wins, T, S=32, gaps=(16, 64, 128, 176), per_gap=96, seed=20261122):
    rng = random.Random(seed)
    texts = [w['data'] for w in wins]
    probes = []
    for gap in gaps:
        pre = T - 2 * S - gap
        for _ in range(per_gap):
            a = rng.randrange(len(texts))
            b = rng.randrange(len(texts))
            while b == a:
                b = rng.randrange(len(texts))
            A, Bt = texts[a], texts[b]
            st = rng.randrange(0, len(A) - (pre + S))
            sb = rng.randrange(0, len(Bt) - gap)
            seq = np.concatenate([A[st:st + pre + S], Bt[sb:sb + gap], A[st + pre:st + pre + S]])
            assert len(seq) == T
            probes.append((gap, pre, seq))
    return probes


@torch.no_grad()
def copy_eval(model, probes, T, S=32, skip=4, bs=32, hard=False):
    model.eval()
    res = {}
    for i in range(0, len(probes), bs):
        ch = probes[i:i + bs]
        x = torch.from_numpy(np.stack([p[2] for p in ch]))
        lg = model(x, hard=hard)
        lp = F.log_softmax(lg, -1)
        for j, (gap, pre, seq) in enumerate(ch):
            r = res.setdefault(gap, dict(n=0, acc2=0, nll2=0.0, acc1=0, nll1=0.0))
            for first, base in ((True, pre), (False, T - S)):
                pos = torch.arange(base + skip, base + S)          # target positions
                tgt = x[j, pos]
                l = -lp[j, pos - 1, tgt]
                acc = int((lg[j, pos - 1].argmax(-1) == tgt).sum())
                if first:
                    r['acc1'] += acc
                    r['nll1'] += float(l.sum())
                else:
                    r['acc2'] += acc
                    r['nll2'] += float(l.sum())
            r['n'] += S - skip
    out = {}
    for gap, r in sorted(res.items()):
        out[str(gap)] = dict(targets=r['n'], second_acc=r['acc2'] / r['n'], second_bpb=r['nll2'] / r['n'] / LN2,
                             first_acc=r['acc1'] / r['n'], first_bpb=r['nll1'] / r['n'] / LN2,
                             copy_gain_bpb=(r['nll1'] - r['nll2']) / r['n'] / LN2)
    return out


@torch.no_grad()
def parity(model, x):
    model.eval()
    full = model(x, hard=True)
    st, outs = None, []
    for t in range(x.shape[1]):
        lg, st = model.step(x[:, t], st, hard=True)
        outs.append(lg)
    return float((torch.stack(outs, 1) - full).abs().max())


@torch.no_grad()
def timing(model, sample, T, reps=12, prompt=64, gen=192):
    model.eval()
    x = torch.from_numpy(sample[:T])[None]
    for _ in range(3):
        model(x, hard=True)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x, hard=True)
        ts.append(time.perf_counter() - t0)
    st, lg = None, None
    for t in sample[:prompt]:
        lg, st = model.step(torch.tensor([int(t)]), st, hard=True)
    tok = lg.argmax(-1)
    per, out = [], []
    for _ in range(gen):
        t0 = time.perf_counter()
        lg, st = model.step(tok, st, hard=True)
        per.append(time.perf_counter() - t0)
        tok = lg.argmax(-1)
        out.append(int(tok))
    per_s = sorted(per)
    return dict(prefill_tok_s=T / statistics.median(ts), prefill_ms_median=1000 * statistics.median(ts),
                decode_ms_median=1000 * statistics.median(per), decode_ms_p95=1000 * per_s[int(0.95 * len(per_s)) - 1],
                decode_tok_s=len(per) / sum(per),
                prompt=bytes(int(t) for t in sample[:prompt]).decode('utf8', 'replace'),
                greedy_sample=bytes(out).decode('utf8', 'replace'))


def wjson(path, obj):
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- one run
def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test, data_info = load_all()
    if a.smoke:
        dev, test = dev[:6], test[:6]
    T, B = a.ctx, a.batch
    model = build(a.arm)
    n_par = nparams(model)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=other, weight_decay=0.0)],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    def lr_at(i):
        if i <= a.warmup:
            return a.lr * i / a.warmup
        p = min(1.0, (i - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p)))

    dev_items, test_items = eval_plan(dev, T), eval_plan(test, T)
    rng = np.random.default_rng(a.seed)          # identical byte stream for every arm at this seed
    hi = len(train) - T - 1
    digest = hashlib.sha256()
    curve, train_s, losses, status = [], 0.0, [], 'ok'
    store = getattr(model, 'store', None)

    def dev_eval(step):
        t0 = time.perf_counter()
        bpb = domain_bpb(evaluate(model, dev, dev_items, T))
        curve.append(dict(step=step, train_s=train_s, dev_bpb=bpb['macro'], dev=bpb,
                          eval_s=time.perf_counter() - t0))
        model.train()

    dev_eval(0)
    model.train()
    t_start = time.time()
    for i in range(1, a.steps + 1):
        offs = rng.integers(0, hi, size=B)
        digest.update(offs.astype(np.int64).tobytes())
        t0 = time.perf_counter()
        xb = torch.from_numpy(np.stack([train[o:o + T + 1] for o in offs]).astype(np.int64))
        for gp in opt.param_groups:
            gp['lr'] = lr_at(i)
        lg = model(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
        pen = getattr(model, 'last_penalty', None)
        # The gate is free while the store learns to be useful, then its price ramps in
        # (a gate priced from step 1 closes before the store has anything to offer).
        cost = a.gate_cost * min(1.0, max(0.0, (i - a.gate_hold) / max(1, a.gate_ramp)))
        total = loss + cost * pen if pen is not None else loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_s += time.perf_counter() - t0
        lv = float(loss)
        losses.append(lv)
        if not math.isfinite(lv):
            status = f'diverged at step {i}'
            break
        if i % a.eval_every == 0 or i == a.steps:
            dev_eval(i)
        if i % 50 == 0 or i == a.steps:
            wjson(out / 'progress.json', dict(arm=a.arm, seed=a.seed, step=i, steps=a.steps, train_s=train_s,
                                              loss50=statistics.mean(losses[-50:]),
                                              gate50=None if pen is None else float(pen),
                                              last_dev=curve[-1]['dev_bpb'], wall_s=time.time() - t_start))
    torch.save(model.state_dict(), out / 'weights.pt')
    res = dict(arm=a.arm, seed=a.seed, status=status, params=n_par, steps=a.steps, batch=B, ctx=T, lr=a.lr,
               hf_equivalence_max_abs=getattr(model, 'hf_equivalence_max_abs', None),
               warmup=a.warmup, weight_decay=a.wd, gate_cost=a.gate_cost if store is not None else None,
               gate_hold=a.gate_hold if store is not None else None, gate_ramp=a.gate_ramp if store is not None else None,
               byte_presentations=a.steps * B * T, train_stream_sha=digest.hexdigest(), train_s=train_s,
               train_s_per_step=train_s / max(1, len(losses)), curve=curve,
               final_train_loss50=statistics.mean(losses[-50:]) / LN2 if losses else None, data=data_info,
               torch=torch.__version__, threads=a.threads, host=os.uname().nodename,
               weights_sha=sha((out / 'weights.pt').read_bytes()))
    if status == 'ok':
        if store is not None:
            store.reset_stats()
        t0 = time.perf_counter()
        docs = evaluate(model, test, test_items, T)
        res['test_eval_s'] = time.perf_counter() - t0
        res['test'] = domain_bpb(docs)
        res['test_docs'] = docs
        if store is not None and store.mode != 'always':
            res['gate_soft'] = dict(open_rate=store.stat_open / max(1, store.stat_total),
                                    mean_gate=store.stat_gsum / max(1, store.stat_total))
        if store is not None and store.mode == 'gated':
            store.reset_stats()
            hd = evaluate(model, test, test_items, T, hard=True)
            res['test_hard'] = domain_bpb(hd)
            res['test_docs_hard'] = hd
            res['gate_hard'] = dict(read_rate=store.stat_open / max(1, store.stat_total))
        probes = copy_probes(test, T)
        res['copy'] = copy_eval(model, probes, T, hard=store is not None and store.mode == 'gated')
        sample = next(w['data'] for w in test if w['domain'] == 'article')
        torch.manual_seed(7)
        px = torch.from_numpy(np.stack([test[0]['data'][:96], test[1]['data'][:96]]))
        res['parity_max_abs'] = parity(model, px)
        res['timing'] = timing(model, sample, T, reps=3 if a.smoke else 12, gen=16 if a.smoke else 192)
    res['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res['wall_s'] = time.time() - t_start
    wjson(out / 'result.json', res)
    print(json.dumps({k: res.get(k) for k in ('arm', 'seed', 'status', 'params', 'train_s', 'test',
                                               'parity_max_abs', 'peak_rss_mib')}), flush=True)


def probe(a):
    """Time a few training steps per arm and report parameters and peak memory (one arm per call)."""
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    train, dev, test, _ = load_all()
    m = build(a.arm)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(1)
    ts = []
    for i in range(a.steps):
        offs = rng.integers(0, len(train) - a.ctx - 1, size=a.batch)
        xb = torch.from_numpy(np.stack([train[o:o + a.ctx + 1] for o in offs]).astype(np.int64))
        t0 = time.perf_counter()
        lg = m(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
        pen = getattr(m, 'last_penalty', None)
        (loss + 0.02 * pen if pen is not None else loss).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ts.append(time.perf_counter() - t0)
    px = torch.from_numpy(np.stack([test[0]['data'][:96], test[1]['data'][:96]]))
    print(json.dumps(dict(arm=a.arm, params=nparams(m), s_per_step=statistics.median(ts[2:]) if len(ts) > 3 else ts[-1],
                          parity=parity(m, px), loss=float(loss),
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)), flush=True)


def speed(a):
    """Same-process, same-thread speed table for every arm (untrained weights; speed does not depend on them)."""
    torch.set_num_threads(a.threads)
    train, dev, test, _ = load_all()
    sample = next(w['data'] for w in test if w['domain'] == 'article')
    rng = np.random.default_rng(3)
    for arm in a.arms.split(','):
        torch.manual_seed(1)
        m = build(arm)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        ts = []
        for i in range(a.steps):
            offs = rng.integers(0, len(train) - a.ctx - 1, size=a.batch)
            xb = torch.from_numpy(np.stack([train[o:o + a.ctx + 1] for o in offs]).astype(np.int64))
            t0 = time.perf_counter()
            lg = m(xb[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
            pen = getattr(m, 'last_penalty', None)
            (loss + 0.02 * pen if pen is not None else loss).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            ts.append(time.perf_counter() - t0)
        tm = timing(m, sample, a.ctx)
        tm.pop('prompt'), tm.pop('greedy_sample')
        print(json.dumps(dict(arm=arm, params=nparams(m), train_s_per_step=statistics.median(ts[3:]), **tm)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run', 'probe', 'speed'))
    p.add_argument('--arm', choices=ARMS, default='om2')
    p.add_argument('--arms', default='tf,m2hf,m2,om1,om2,om2pos,om2p,om2a,om2g')
    p.add_argument('--seed', type=int, default=5617)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--ctx', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--gate-cost', type=float, default=0.02)
    p.add_argument('--gate-hold', type=int, default=500)
    p.add_argument('--gate-ramp', type=int, default=250)
    p.add_argument('--eval-every', type=int, default=300)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default='runs/tmp')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.mode == 'probe':
        probe(a)
    elif a.mode == 'speed':
        speed(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
