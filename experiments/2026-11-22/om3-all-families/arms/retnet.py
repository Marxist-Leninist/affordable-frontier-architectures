"""retnet -- RetNet multi-scale retention (MSR) layers for the OM3 byte-level benchmark.

SOURCE
  Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, F. Wei, "Retentive Network: A Successor to
  Transformer for Large Language Models", arXiv:2307.08621 (2023): Sec. 2.1 retention (parallel and
  recurrent forms), Sec. 2.2 gated multi-scale retention and retention score normalisation, Sec. 2.3
  the RetNet layer.  Rotation angles, decay table and initialisation follow the authors' reference code
  github.com/microsoft/torchscale (RetNetRelPos, MultiScaleRetention).

MODEL (471,296 parameters; target 471,264, +0.007%)
  tokens -> embedding E (256 x 128, N(0, 0.02^2), tied with the output layer) -> 4 RetNet layers
         -> LayerNorm -> logits = LN(x) E^T
  RetNet layer:  Y = MSR(LN(X)) + X,   X' = FFN(LN(Y)) + Y,   FFN(X) = gelu(X W_1) W_2  (width 106)

RETENTION (one head h, key/value head width 32)
  Q = (X W_Q) o Theta,  K = (X W_K) o conj(Theta) / sqrt(d_k),  V = X W_V
    Theta_n = e^{i n theta}: the rotation is applied to interleaved pairs (rotary form, as in the reference
    code), theta_j = 10000^(-j/15), j = 0..15 (reference RetNetRelPos angles: 1/10000^linspace(0, 1, 16)).
  gamma_h = 1 - 2^(-5-h),  h = 0..3   (0.96875, 0.984375, 0.9921875, 0.99609375)
  (paper Eq. 8, the torchscale default: e-folding 32/64/128/256 tokens. The paper's Sec. 3.1 experiments
  used gamma = 1 - exp(linspace(log 1/32, log 1/512, h)), 32/81/203/512 tokens; Eq. 8 is used here
  because it is the reference code's default.)
  D_nm = gamma^(n-m) for n >= m, 0 otherwise
  Parallel form (forward):    Retention(X) = (Q K^T o D) V
  Recurrent form (step):      S_n = gamma S_{n-1} + K_n^T V_n,   Retention(X_n) = Q_n S_n
  Retention score normalisation (paper Sec. 2.2), applied identically in both forms:
    D~_nm = D_nm / sqrt(sum_{i<=n} D_ni),   R = Q K^T o D~,   R~_nm = R_nm / max(|sum_{i<=n} R_ni|, 1),
    output = R~ V  (computed as (R V) / max(|sum_i R_ni|, 1): the factor is per row, the same numbers).
    Recurrent equivalent: c_n = gamma c_{n-1} + 1 (= row sum of D), z_n = gamma z_{n-1} + K_n,
    output_n = (Q_n S_n / sqrt(c_n)) / max(|Q_n z_n| / sqrt(c_n), 1).
  These factors are per (position, head) scalars, so the per-head GroupNorm below removes them
  (the paper: they "do not affect the final results"); they only keep the numbers in range.

GATED MULTI-SCALE RETENTION (4 heads, one gamma each)
  head_h = Retention_h(X)
  Y   = GroupNorm_h(Concat(head_1, ..., head_4))      per-head normalisation (mean and variance over the
                                                      head's 32 channels), no affine, eps 1e-6
  MSR(X) = (swish(X W_G) o Y) W_O

INITIALISATION (reference code): W_Q, W_K, W_V, W_G ~ xavier_uniform(gain 2^-2.5), W_O ~ xavier_uniform(gain 2^-1);
  FFN: PyTorch nn.Linear default; LayerNorm weight 1, bias 0.

SIMPLIFICATIONS / CHOICES VERSUS THE PAPER
  1. Widths: d = 128 and 4 layers (benchmark), 4 heads, key and value widths both d (value factor 1
     instead of the paper's 2d) and FFN width 106 (instead of 2d), so that the model meets the 471k
     parameter target; with value width 2d the MSR alone (8 d^2 per layer) would exceed the per-layer budget.
  2. GroupNorm without affine parameters (as in the reference code).
  3. Theta is the rotary form of xPos used by the reference code (pure rotation; the decay gamma supplies
     the length extrapolation term), not the xPos scale factor.
  4. R~ uses the paper's max(|row sum|, 1) in both forms (the reference code's parallel form divides by a
     detached clamp(sum |R|, 1, 5e4) and its recurrent form omits the factor; both are output-invariant
     through the GroupNorm up to eps; using the same factor in both forms keeps them exactly equivalent).
  5. FFN and projections without biases (paper equations); LayerNorm with bias (paper's LN).
  6. No chunkwise recurrent form (not needed at context 256), no dropout, no DeepNet/sub-LN rescaling,
     embedding std 0.02 and tied output layer (benchmark convention).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D, L, V = ob.D_MODEL, ob.LAYERS, ob.VOCAB    # 128, 4, 256
H = 4                                         # retention heads, one decay rate each
DK = DV = D                                   # key and value widths (value factor 1)
KH, VH = DK // H, DV // H                     # 32, 32
FF = 106                                      # FFN width chosen for the parameter target
GN_EPS = 1e-6


class MSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, DK, bias=False)
        self.wk = nn.Linear(D, DK, bias=False)
        self.wv = nn.Linear(D, DV, bias=False)
        self.wg = nn.Linear(D, DV, bias=False)
        self.wo = nn.Linear(DV, D, bias=False)
        for lin in (self.wq, self.wk, self.wv, self.wg):
            nn.init.xavier_uniform_(lin.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.wo.weight, gain=2 ** -1)
        self.register_buffer('log_gamma', torch.log(1.0 - 2.0 ** (-5.0 - torch.arange(H, dtype=torch.float32))),
                             persistent=False)
        self.register_buffer('angle', 1.0 / 10000 ** torch.linspace(0, 1, KH // 2), persistent=False)

    def _out(self, x, y):
        """y: (..., H, VH) retention output -> per-head GroupNorm, swish gate, output projection."""
        y = F.layer_norm(y, (VH,), None, None, GN_EPS).flatten(-2)
        return self.wo(F.silu(self.wg(x)) * y)

    def _dtilde(self, T):
        """D~ (H, T, T): gamma^(n-m) for n >= m, 0 above the diagonal, rows divided by sqrt of their sums."""
        pos = torch.arange(T, dtype=torch.float32)
        nm = pos[:, None] - pos[None, :]
        dmat = torch.exp(nm.clamp_min(0.0) * self.log_gamma[:, None, None]).masked_fill(nm < 0, 0.0)
        return dmat / dmat.sum(-1, keepdim=True).sqrt()

    def forward(self, x):
        B, T, _ = x.shape
        f = torch.arange(T, dtype=torch.float32)[:, None] * self.angle
        cos, sin = f.cos(), f.sin()
        q = ob.apply_rope(self.wq(x).view(B, T, H, KH).transpose(1, 2), cos, sin)
        k = ob.apply_rope(self.wk(x).view(B, T, H, KH).transpose(1, 2), cos, sin) * KH ** -0.5
        v = self.wv(x).view(B, T, H, VH).transpose(1, 2)
        r = (q @ k.transpose(-1, -2)) * self._dtilde(T)                          # R = Q K^T o D~
        y = (r @ v) / r.sum(-1, keepdim=True).abs().clamp_min(1.0)              # R~ V
        return self._out(x, y.transpose(1, 2))

    def step(self, x, st):
        B = x.shape[0]
        if st is None:
            pos, S, z, c = 0, x.new_zeros(B, H, KH, VH), x.new_zeros(B, H, KH), x.new_zeros(H)
        else:
            pos, S, z, c = st
        f = torch.tensor(float(pos)) * self.angle
        cos, sin = f.cos()[None], f.sin()[None]
        q = ob.apply_rope(self.wq(x).view(B, H, KH), cos, sin)
        k = ob.apply_rope(self.wk(x).view(B, H, KH), cos, sin) * KH ** -0.5
        v = self.wv(x).view(B, H, VH)
        g = self.log_gamma.exp()
        S = g[:, None, None] * S + k[..., :, None] * v[..., None, :]              # S_n = gamma S_{n-1} + K^T V
        z = g[:, None] * z + k
        c = g * c + 1.0
        rc = c.sqrt()
        num = (q[..., :, None] * S).sum(-2) / rc[:, None]                         # Q_n S_n / sqrt(c_n)
        s = (q * z).sum(-1) / rc                                                  # row sum of R
        y = num / s.abs().clamp_min(1.0)[..., None]
        return self._out(x, y), (pos + 1, S, z, c)


class RetNetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.msr, self.n2 = nn.LayerNorm(D), MSR(), nn.LayerNorm(D)
        self.f1, self.f2 = nn.Linear(D, FF, bias=False), nn.Linear(FF, D, bias=False)

    def forward(self, x):
        y = x + self.msr(self.n1(x))
        return y + self.f2(F.gelu(self.f1(self.n2(y))))

    def step(self, x, st):
        a, st = self.msr.step(self.n1(x), st)
        y = x + a
        return y + self.f2(F.gelu(self.f1(self.n2(y)))), st


class RetNetLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(V, D)
        nn.init.normal_(self.e.weight, std=0.02)
        self.layers = nn.ModuleList(RetNetLayer() for _ in range(L))
        self.n = nn.LayerNorm(D)
        self.last_penalty = None

    def forward(self, x, hard=False):
        h = self.e(x)
        for lay in self.layers:
            h = lay(h)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        h = self.e(tok)
        sts = state if state is not None else [None] * len(self.layers)
        new = []
        for lay, s in zip(self.layers, sts):
            h, s = lay.step(h, s)
            new.append(s)
        return F.linear(self.n(h), self.e.weight), new


def build():
    return RetNetLM()
