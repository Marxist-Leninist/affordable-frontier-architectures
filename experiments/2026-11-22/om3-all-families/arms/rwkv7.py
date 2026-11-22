"""RWKV-7-style arm (RWKV-7 "Goose") for the OM3 byte-level benchmark (om2bench.py frame).

Source paper: B. Peng et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution", arXiv:2503.14456 (2025),
section 4 (time mixing, channel mixing); reference code RWKV-LM/RWKV-v7 (RWKV_Tmix_x070, RWKV_CMix_x070), whose
initialisation is followed here.

Equations implemented (row vectors, per layer; D = 128 channels split into h = 2 heads of size N = 64;
lerp(p, q, m) = p + (q - p) * m; sigma = logistic sigmoid; * = elementwise):

 Time mixing, input x_t = LN1(h_t), with x_{-1} = 0:
  (1) x^s_t = lerp(x_t, x_{t-1}, mu_s),  s in {r, d, k, v, a, g}        token shift: learned per-channel lerp
  (2) r_t = x^r_t W_r,   k_t = x^k_t W_k,   v_t = x^v_t W_v
  (3) d_t = w0 + tanh(x^d_t A_d) B_d ;   w_t = exp(-e^{-0.5} sigma(d_t))  vector decay, every channel in (0.545, 1)
                                                                       (the paper's sigmoid-bounded form)
  (4) a_t = sigma(a0 + (x^a_t A_a) B_a)                                 vector in-context learning rate
  (5) layers l > 0: v_t <- lerp(v_t, v^(0)_t, sigma(v0 + (x^v_t A_v) B_v))  value residual (v^(0) = layer-0 v_t),
                                                                       written as in the reference code
  (6) g_t = sigma(x^g_t A_g) B_g                                        output gate
  (7) kappa_t = k_t * xi ;  kh_t = kappa_t / ||kappa_t||  (per head)    normalised removal key
  (8) kt_t = k_t * (1 + (a_t - 1) * alpha)                              replacement key
  (9) S_t = S_{t-1} (diag(w_t) - kh_t^T (a_t * kh_t)) + v_t^T kt_t      generalised delta rule, per head,
                                                                       S in R^{N x N}, S_{-1} = 0
 (10) p_t = GroupNorm_heads(r_t S_t^T) + (sum_head r_t * rho * kt_t) v_t  per-head group norm + bonus term
 (11) o_t = (g_t * p_t) W_o
 Channel mixing, input x'_t = LN2(h_t), x'_{-1} = 0:
 (12) o'_t = relu(lerp(x'_t, x'_{t-1}, mu'_k) W_k')^2 W_v'               squared-ReLU FFN with token shift
 Block: h_t <- h_t + o_t ; h_t <- h_t + o'_t ; block 0 first applies LN0 to the embedding.
 Model: logits = RMSNorm(h_t) E^T with the tied byte embedding E.

How (9) is evaluated.  step() runs (9) token by token (exact incremental decoding; the per-layer state is the
previous token's LN1 and LN2 outputs and S).  forward() evaluates the same recurrence exactly with a chunked
diagonal-plus-low-rank (WY-style) form, chunk L = 16.  With the state transposed, Z = S^T, c_t = -kh_t and
b_t = a_t * kh_t, (9) reads Z_t = diag(w_t) Z_{t-1} + b_t^T (c_t Z_{t-1}) + kt_t^T v_t, and y_t = r_t Z_t.
Inside a chunk with start state Z_0 and Lambda_t = prod_{tau<=t} w_tau, the removal reads sa_t = c_t Z_{t-1}
satisfy the unit lower-triangular system
   sa_t = (c_t Lambda_{t-1}) Z_0 + sum_{s<t} [ (c_t Lambda_{t-1}).(b_s / Lambda_s) sa_s
                                              + (c_t Lambda_{t-1}).(kt_s / Lambda_s) v_s ],
solved for all chunks at once (torch.linalg.solve_triangular) in the form sa = Cp Z_0 + U; then
   y_t = (r_t Lambda_t) Z_0 + sum_{s<=t} [ (r_t Lambda_t).(b_s/Lambda_s) sa_s + (r_t Lambda_t).(kt_s/Lambda_s) v_s ]
   Z_L = diag(Lambda_L) Z_0 + sum_s [ (b_s Lambda_L/Lambda_s)^T sa_s + (kt_s Lambda_L/Lambda_s)^T v_s ].
The chunk-to-chunk recurrence (sa = Cp Z_0 + U, then Z_L) is the only sequential loop (T/16 steps).  This is
the same algebra as (9), not an approximation (agreement with the token-by-token loop is ~1e-13 in float64,
gradients included); w_t >= exp(-e^{-0.5}) = 0.545 bounds 1/Lambda by 0.545^-16 < 2e4, so float32 stays
finite.  The per-head GroupNorm is computed as layer_norm over each head's N channels followed by the
per-channel affine, which is the same function as GroupNorm(h groups, D channels).

Sizes: d_model 128, 4 blocks, 2 heads x 64, channel-mix hidden 100, LoRA ranks decay 16 / in-context rate 16 /
value residual 8 / gate 24; 470,784 parameters (target 471,264, -0.10%).

Simplifications and differences versus the paper:
 - Scale: channel-mix hidden width 100 (paper 4 D = 512) and LoRA ranks 16/16/8/24 (the reference code's size
   rule gives at least 32 each, and 32-64 for the gate, at this width) so that the model fits the benchmark's
   ~471k-parameter budget at d_model 128 with the paper's full-width r/k/v/o projections.
 - Benchmark frame shared with the other arms: byte vocabulary (256) and context 256; input embedding tied to
   the output head and initialised N(0, 0.02) (paper: World tokenizer, separate head, embedding uniform
   +-1e-4); final RMSNorm (paper: LayerNorm ln_out).  The block norms LN0/LN1/LN2 are LayerNorms as in the paper.
 - Plain float32 PyTorch on CPU, no custom CUDA kernel; the chunked form above computes the same function as
   the paper's sequential kernel.
 - Training follows the benchmark recipe (AdamW with weight decay on every matrix including the LoRA factors,
   one learning rate, float32), not the reference trainer's (per-parameter lr multipliers, decay on main
   weights only, bf16, L2Wrap logit penalty).
 - Initialisation follows the RWKV-LM v7 reference code (lerp schedules, w0/a0/v0, xi = k_k, alpha = k_a,
   rho = r_k, zero LoRA down-factors, orthogonal 0.1 up-factors, zero output projections, GroupNorm scale
   ((1+l)/L)^0.7, eps 64e-5) adapted to D = 128; no dropout, no state tuning. r/k/v and the channel-mix key
   are orthogonal (gains 1 / 0.1 / 1 / 1), as the v7 trainer's generate_init_weight sets them for a new model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D, LAYERS, VOCAB = ob.D_MODEL, ob.LAYERS, ob.VOCAB
HEAD = 64                                   # head size N (the paper's); D / HEAD = 2 heads
FFN = 100                                   # channel-mix hidden width
LORA_D, LORA_A, LORA_V, LORA_G = 16, 16, 8, 24
CHUNK = 16
GN_EPS = 64e-5
DECAY_SCALE = math.exp(-0.5)
_MASKS = {}


def _masks(L):
    """(mask of the dropped entries of [[M, Kc], [Ab, Ak]], identity) for an L x L chunk."""
    if L not in _MASKS:
        o = torch.ones(L, L, dtype=torch.bool)
        ns, ni = ~torch.tril(o, -1), ~torch.tril(o, 0)            # keep s < t  /  keep s <= t
        _MASKS[L] = (torch.cat((torch.cat((ns, ns), 1), torch.cat((ni, ni), 1)), 0), torch.eye(L))
    return _MASKS[L]


def _ortho(rows, cols, scale):
    w = torch.empty(rows, cols)
    nn.init.orthogonal_(w, gain=(math.sqrt(rows / cols) if rows > cols else 1.0) * scale)
    return w


def _shift(x):
    """x_{t-1} along time (dim 1) with x_{-1} = 0."""
    return F.pad(x, (0, 0, 1, -1))


def wkv7(r, lw, k, v, c, b, L=CHUNK):
    """Exact chunked evaluation of eq. (9) for all heads, from a zero state.

    Inputs (B, T, H, N): receptance r, log decay lw = log w_t (in (-e^-0.5, 0)), replacement key k (kt), value v,
    removal pair c = -kh and b = a * kh.  With Z = S^T (key x value):
        Z_t = diag(w_t) Z_{t-1} + b_t^T (c_t Z_{t-1}) + k_t^T v_t ,   y_t = r_t Z_t .
    Returns y (B, T, H, N).
    """
    B, T, H, N = r.shape
    pad = (-T) % L
    n = (T + pad) // L
    X = torch.stack((r, lw, k, v, c, b))
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
    r, lw, k, v, c, b = X.view(6, B, n, L, H, N).permute(0, 1, 4, 2, 3, 5).contiguous().unbind(0)  # B H n L N
    drop, eye = _masks(L)
    cs = lw.cumsum(-2)                                                         # log Lambda_t
    qs = (torch.stack((c, r), -3) * torch.stack((cs - lw, cs), -3).exp()).view(B, H, n, 2 * L, N)
    ks = (torch.stack((b, k), -3) * (-cs).exp().unsqueeze(-3)).view(B, H, n, 2 * L, N)
    # qs = [c_t Lambda_{t-1} ; r_t Lambda_t],  ks = [b_s ; k_s] / Lambda_s
    G = (qs @ ks.transpose(-1, -2)).masked_fill(drop, 0.0)                   # [[M, Kc], [Ab, Ak]]
    Gv = G[..., L:] @ v                                                        # [Kc v ; Ak v]
    sol = torch.linalg.solve_triangular(eye - G[..., :L, :L], torch.cat((qs[..., :L, :], Gv[..., :L, :]), -1),
                                        upper=False, unitriangular=True)
    Cp, U = sol[..., :N], sol[..., N:]                                         # sa = Cp Z_0 + U
    tail = (cs[..., -1:, :] - cs).exp()                                        # Lambda_L / Lambda_s
    blT = (b * tail).transpose(-1, -2)
    Pk = (k * tail).transpose(-1, -2) @ v                                      # sum_s (k_s Lambda_L/Lambda_s)^T v_s
    lamL = cs[..., -1, :].exp().unsqueeze(-1)                                  # Lambda_L
    BH = B * H
    Cps, Us, blTs, Pks, lamLs = (z.unbind(2) for z in (Cp, U, blT, Pk, lamL))
    Z = r.new_zeros(BH, N, N)
    Zs, SAs = [], []
    for i in range(n):                                                         # chunk-to-chunk recurrence
        Zs.append(Z)
        sa = torch.baddbmm(Us[i].reshape(BH, L, N), Cps[i].reshape(BH, L, N), Z)
        SAs.append(sa)
        if i + 1 < n:                                                          # Z_L = Lambda_L Z_0 + bl^T sa + Pk
            Z = torch.baddbmm(torch.addcmul(Pks[i].reshape(BH, N, N), lamLs[i].reshape(BH, N, 1), Z),
                              blTs[i].reshape(BH, N, L), sa)
    Zs = torch.stack(Zs, 1).view(B, H, n, N, N)
    SA = torch.stack(SAs, 1).view(B, H, n, L, N)
    y = qs[..., L:, :] @ Zs + G[..., L:, :L] @ SA + Gv[..., L:, :]            # (r Lambda) Z_0 + Ab sa + Ak v
    return y.permute(0, 2, 3, 1, 4).reshape(B, n * L, H, N)[:, :T]


class TimeMix(nn.Module):
    def __init__(self, layer, n_layer=LAYERS, C=D, N=HEAD):
        super().__init__()
        self.layer, self.C, self.N, self.H = layer, C, N, C // N
        r01 = layer / (n_layer - 1)
        r10 = 1.0 - layer / n_layer
        i = torch.arange(C, dtype=torch.float32)
        ddd = i / C
        lin = i / (C - 1) - 0.5
        zig = ((i % N) - (N - 1) / 2) / ((N - 1) / 2)
        zig = zig * zig.abs()
        P = nn.Parameter
        self.mu_r, self.mu_d = P(1 - ddd.pow(0.2 * r10)), P(1 - ddd.pow(0.9 * r10))
        self.mu_k, self.mu_v = P(1 - ddd.pow(0.7 * r10)), P(1 - ddd.pow(0.7 * r10))
        self.mu_a, self.mu_g = P(1 - ddd.pow(0.9 * r10)), P(1 - ddd.pow(0.2 * r10))
        self.w0 = P(-6 + 6 * (i / (C - 1)).pow(1 + r01 ** 0.3) + 0.5 + zig * 2.5)
        self.w1, self.w2 = P(torch.zeros(C, LORA_D)), P(_ortho(LORA_D, C, 0.1))
        self.a0 = P(-0.19 + zig * 0.3 + lin * 0.4)
        self.a1, self.a2 = P(torch.zeros(C, LORA_A)), P(_ortho(LORA_A, C, 0.1))
        if layer > 0:
            self.v0 = P(0.73 - lin * 0.4)
            self.v1, self.v2 = P(torch.zeros(C, LORA_V)), P(_ortho(LORA_V, C, 0.1))
        self.g1, self.g2 = P(torch.zeros(C, LORA_G)), P(_ortho(LORA_G, C, 0.1))
        self.xi = P(0.71 - lin * 0.1)                    # k_k: removal-key scale
        self.alpha = P(torch.full((C,), 1.02))           # k_a: replacement-key mix
        self.rho = P(torch.full((C,), -0.04))            # r_k: bonus scale (H x N, stored flat)
        self.Wr, self.Wk, self.Wv, self.Wo = (nn.Linear(C, C, bias=False) for _ in range(4))
        s = C ** -0.5
        with torch.no_grad():
            # orthogonal r/k/v as RWKV-LM v7 train.py generate_init_weight sets them for a new model (it overwrites
            # the uniform lines in RWKV_Tmix_x070.__init__): gain 1.0 for r and v, 0.1 for k (review fix, 22 Nov)
            nn.init.orthogonal_(self.Wr.weight, gain=1.0)
            nn.init.orthogonal_(self.Wk.weight, gain=0.1)
            nn.init.orthogonal_(self.Wv.weight, gain=1.0)
            self.Wo.weight.zero_()
        self.gn_w = P(torch.full((C,), ((1 + layer) / n_layer) ** 0.7))
        self.gn_b = P(torch.zeros(C))

    def _mix(self, x, xprev, v_first):
        """Eqs. (1)-(8) for any leading shape (..., C)."""
        mu = torch.stack((self.mu_r, self.mu_d, self.mu_k, self.mu_v, self.mu_a, self.mu_g))
        xr, xd, xk, xv, xa, xg = torch.lerp(x.unsqueeze(0), xprev.unsqueeze(0),
                                            mu.view((6,) + (1,) * (x.dim() - 1) + (self.C,))).unbind(0)  # eq. (1)
        r = self.Wr(xr)
        lw = -DECAY_SCALE * torch.sigmoid(self.w0 + torch.tanh(xd @ self.w1) @ self.w2)     # log w_t, eq. (3)
        k = self.Wk(xk)
        v = self.Wv(xv)
        if self.layer == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2))      # eq. (5)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)                                  # eq. (4)
        g = torch.sigmoid(xg @ self.g1) @ self.g2                                              # eq. (6)
        sh = x.shape[:-1] + (self.H, self.N)
        kh = F.normalize((k * self.xi).view(sh), dim=-1)                                        # eq. (7)
        kt = k * (1 + (a - 1) * self.alpha)                                                     # eq. (8)
        return r, lw, kt, v, a.view(sh), g, kh, v_first

    def _out(self, y, r, kt, v, g):
        """Eqs. (10)-(11); y is the WKV read-out r_t S_t^T with shape (..., H, N).

        GroupNorm with one group per head = normalisation over each head's N channels (layer_norm over the
        last axis), then the per-channel affine."""
        sh = y.shape
        hn = (self.H, self.N)
        p = torch.addcmul(self.gn_b.view(hn), F.layer_norm(y, (self.N,), eps=GN_EPS), self.gn_w.view(hn))
        p = p + (r.view(sh) * kt.view(sh) * self.rho.view(hn)).sum(-1, keepdim=True) * v.view(sh)
        return self.Wo(p.reshape(g.shape) * g)

    def forward(self, x, v_first):
        B, T, _ = x.shape
        r, lw, kt, v, a, g, kh, v_first = self._mix(x, _shift(x), v_first)
        sh = (B, T, self.H, self.N)
        y = wkv7(r.view(sh), lw.view(sh), kt.view(sh), v.view(sh), -kh, kh * a)
        return self._out(y, r, kt, v, g), v_first

    def step(self, x, state, v_first):
        B = x.shape[0]
        H, N = self.H, self.N
        if state is None:
            xprev, Z = torch.zeros_like(x), x.new_zeros(B, H, N, N)
        else:
            xprev, Z = state
        r, lw, kt, v, a, g, kh, v_first = self._mix(x, xprev, v_first)
        sa = (-kh).unsqueeze(-2) @ Z                                   # c_t Z_{t-1}       (B, H, 1, N)
        Z = (lw.view(B, H, N, 1).exp() * Z + (kh * a).unsqueeze(-1) @ sa
             + kt.view(B, H, N, 1) @ v.view(B, H, 1, N))                 # eq. (9), Z = S^T
        y = (r.view(B, H, 1, N) @ Z).view(B, H, N)                     # r_t S_t^T
        return self._out(y, r, kt, v, g), (x, Z), v_first


class ChannelMix(nn.Module):
    def __init__(self, layer, n_layer=LAYERS, C=D, hidden=FFN):
        super().__init__()
        ddd = torch.arange(C, dtype=torch.float32) / C
        self.mu_k = nn.Parameter(1 - ddd.pow((1.0 - layer / n_layer) ** 4))
        self.Wk = nn.Linear(C, hidden, bias=False)
        self.Wv = nn.Linear(hidden, C, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.Wk.weight, gain=1.0)          # generate_init_weight: ffn.key orthogonal
            self.Wv.weight.zero_()

    def forward(self, x, xprev):
        return self.Wv(torch.relu(self.Wk(torch.lerp(x, xprev, self.mu_k))) ** 2)          # eq. (12)


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.ln0 = nn.LayerNorm(D) if layer == 0 else None
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att, self.ffn = TimeMix(layer), ChannelMix(layer)

    def forward(self, h, v_first):
        if self.ln0 is not None:
            h = self.ln0(h)
        o, v_first = self.att(self.ln1(h), v_first)
        h = h + o
        x2 = self.ln2(h)
        return h + self.ffn(x2, _shift(x2)), v_first

    def step(self, h, st, v_first):
        if self.ln0 is not None:
            h = self.ln0(h)
        att_st, x2prev = st if st is not None else (None, None)
        o, att_st, v_first = self.att.step(self.ln1(h), att_st, v_first)
        h = h + o
        x2 = self.ln2(h)
        h = h + self.ffn(x2, torch.zeros_like(x2) if x2prev is None else x2prev)
        return h, (att_st, x2), v_first


class RWKV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(VOCAB, D)
        nn.init.normal_(self.e.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(i) for i in range(LAYERS))
        self.n = ob.RMS(D)
        self.last_penalty = None

    def forward(self, x, hard=False):
        h, v_first = self.e(x), None
        for blk in self.blocks:
            h, v_first = blk(h, v_first)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        h, v_first = self.e(tok), None
        states = state if state is not None else [None] * len(self.blocks)
        new = []
        for blk, st in zip(self.blocks, states):
            h, st, v_first = blk.step(h, st, v_first)
            new.append(st)
        return F.linear(self.n(h), self.e.weight), new


def build():
    return RWKV7()
