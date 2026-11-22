"""OM3 arm `titans`: Titans-style MAG (memory as a gate) model (delta group, Claude lane, 22 Nov 2026).

Source
  Ali Behrouz, Peilin Zhong, Vahab Mirrokni, "Titans: Learning to Memorize at Test Time", arXiv:2501.00663
  (2025). Google released no reference code; this is a Titans-style implementation written from the paper.

Equations implemented
  Persistent memory (paper Sec. 3.3): N_p = 4 learnable, input-independent tokens are prepended once,
    x~ = [p_1 .. p_Np] || x (positions 0 .. N_p-1); their outputs are dropped before the logits.
  Block (x 4, pre-norm, u = RMSNorm(h)), MAG variant (paper Sec. 4.2, "y = SW-Attn*(x~), o = y (x) M(x~)"):
    y = SW-Attn*(u)   causal attention, 4 heads x 32, RoPE, window 64 (query i sees keys i-63 .. i) plus the
                      always-visible persistent prefix (the paper's sliding-window attention with prefix)
    m = M(u)          neural long-term memory read-out (below)
    o = RMSNorm_y(y) * sigmoid(RMSNorm_m(m))      the learned gate: both branch outputs normalised with
                      learnable vector-valued weights, then a sigmoid non-linearity (the paper's description)
    h <- h + o ;  h <- h + W_2 GELU(W_1 RMSNorm(h))      (small MLP, hidden 36, to meet the block budget)
  Neural long-term memory (paper Sec. 3.1), 2 heads, M_t and S_t are d_v x d_k = 32 x 32 per head:
    q_t, k_t, v_t = SiLU(conv4(W_Q u)_t), SiLU(conv4(W_K u)_t), SiLU(conv4(W_V u)_t); q_t, k_t L2-normalised
                    (paper Sec. 4.4: depthwise causal conv after each projection, SiLU, l2-normalised q and k;
                    the projection is the pointwise half of the depthwise-separable convolution)
    alpha_t = sigmoid(w_a . u_t + b_a)     forgetting gate      (initial alpha 1e-3 and 1e-2 for the 2 heads)
    eta_t   = sigmoid(w_e . u_t + b_e)     surprise momentum    (initial ~0.5)
    theta_t = (1 - eta_t) sigmoid(w_th . u_t + b_th)   inner learning rate (initial ~0.05). The (1 - eta_t)
                      factor keeps the momentum-accumulated step 2 theta/(1 - eta) below 2 for unit keys, so the
                      linear-memory recurrence cannot grow without bound (review fix, 22 Nov 2026)
    loss  l(M_{t-1}; x_t) = ||M_{t-1} k_t - v_t||^2,   grad_M l = 2 (M_{t-1} k_t - v_t) k_t^T
    S_t = eta_t S_{t-1} - theta_t grad_M l(M_{t-1}; x_t)          (past surprise + momentary surprise)
    M_t = (1 - alpha_t) M_{t-1} + S_t                            (forgetting = weight decay)
    read  y_t = M_t q_t (after token t is written),  m_t = W_O concat_heads(y_t)
  M_0 = S_0 = 0 before the persistent tokens. The outer-loop weights (W_Q, W_K, W_V, gates, W_O, attention,
  MLP) are trained by back-propagating through the whole inner-loop recurrence, as in the paper.

Training form: the per-token recurrence evaluated exactly in chunks of C = 32 (algebra derived for this file).
  With r_t = 1 - alpha_t, a[t,s] = prod_{s<i<=t} r_i, h[s,j] = prod_{j<i<=s} eta_i, P = a @ h (lower
  triangular), A_t = prod_{i<=t} r_i, H_t = prod_{i<=t} eta_i, B_t = sum_{s<=t} a[t,s] H_s and the errors
  e_t = v_t - M_{t-1} k_t, the recurrence unrolls inside a chunk with incoming state (M0, S0) to
    M_t = A_t M0 + B_t S0 + sum_{j<=t} 2 theta_j P[t,j] e_j k_j^T,   S_t = H_t S0 + sum_{j<=t} 2 theta_j h[t,j] e_j k_j^T
  so the errors solve the unit lower-triangular system
    (I + L) E = V - diag(A_{t-1}) K M0^T - diag(B_{t-1}) K S0^T,   L[t,j] = 2 theta_j P[t-1,j] (k_t . k_j), j < t
  and Y = diag(A) Q M0^T + diag(B) Q S0^T + ((Q K^T) * P * 2 theta_j) E. step() runs the plain recurrence; the
  benchmark parity gate (forward vs step, max abs < 1e-4) checks one against the other.

Size: 4 blocks; attention 4 x 32; memory 2 heads x 32 x 32; MLP hidden 36; 4 persistent tokens
-> 471,704 parameters (target 471,264, +0.09 %).

Simplifications versus the paper (complete list)
  1. Linear memory: M is one d_v x d_k matrix per head, not a deep MLP memory (L_M >= 2 layers, which the
     paper argues for and ablates). This is the paper's L_M = 1 (linear) case, for which the memory update
     is a gated delta rule with momentum.
  2. Inner-loop granularity: every token takes its own gradient step at M_{t-1} (online, mini-batch of 1),
     computed exactly by the chunked algebra above. The paper's parallel training takes all gradients of a
     chunk at the chunk's starting weights (mini-batch gradient descent) for speed; that approximation is
     not made here.
  3. MAG gate: the paper only says the combination is a non-linear gating, with both outputs normalised by
     learnable vector-valued weights followed by a non-linearity; o = RMSNorm(y) * sigmoid(RMSNorm(m)) is this
     file's reading of it. The memory module's own normalisation and linear gate before its output
     projection (Sec. 4.4) are folded into this MAG gate (no separate gate projection).
  4. Gates: alpha, eta, theta come per token and per head from one linear layer plus sigmoid, with theta
     also scaled by (1 - eta) for stability (see above); the paper makes
     them data-dependent (optionally chunk-wise constant) without fixing the parameterisation. The gate
     biases above are this file's choice.
  5. Scale and data: 4 blocks, d_model 128, window 64, 4 persistent tokens, byte tokens, tied embeddings,
     context 256; the paper's models have hundreds of millions of parameters, subword tokens and long contexts.
  6. Capacity: the memory has 2 heads x 32 x 32 (2,048 state + 2,048 momentum floats per layer), less
     recurrent state than gdn (6,144) or Mamba-2 (8,192); attention takes most of the block budget. With
     the 64-token window only the gap-16 copy probe (distance 48) is inside attention; longer gaps rely on
     this memory. The paper gives no sizes.
  6. Initialisation: PyTorch defaults for linear and conv weights; embeddings and persistent tokens
     normal(0, 0.02).
  7. Feed-forward: a GELU MLP of hidden width 36 (a small MLP chosen to meet the parameter budget).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D = ob.D_MODEL
ATT_H, ATT_DH, WINDOW = 4, 32, 64
MEM_H, MEM_DK, MEM_DV = 2, 32, 32
N_PERSIST, FF, KCONV, CHUNK, MAXPOS = 4, 36, 4, 32, 8192


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def memory_chunked(q, k, v, lr1, leta, th2, C=CHUNK):
    """Exact chunkwise evaluation of the per-token memory recurrence from M_0 = S_0 = 0.

    q, k: (B, T, H, dk), L2-normalised; v: (B, T, H, dv); lr1 = log(1 - alpha), leta = log(eta),
    th2 = 2 theta: (B, T, H). Returns y: (B, T, H, dv) with y_t = M_t q_t.
    """
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % C
    if pad:                      # padded tail after every real token: q = k = v = theta = 0, alpha = 0, eta = 1
        q, k, v = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (q, k, v))
        lr1, leta, th2 = (F.pad(z, (0, 0, 0, pad)) for z in (lr1, leta, th2))
    n = (T + pad) // C
    q, k, v = (z.reshape(B, n, C, H, -1).permute(0, 3, 1, 2, 4) for z in (q, k, v))          # B H n C d
    lr1, leta, th2 = (z.reshape(B, n, C, H).permute(0, 3, 1, 2) for z in (lr1, leta, th2))  # B H n C
    a = torch.exp(ob.segsum(lr1))                         # a[t, s] = prod_{s<i<=t} (1 - alpha_i)
    h = torch.exp(ob.segsum(leta))                        # h[s, j] = prod_{j<i<=s} eta_i
    P = a @ h                                             # P[t, j] = sum_{j<=s<=t} a[t, s] h[s, j]
    At = lr1.cumsum(-1).exp()                             # A_t
    Ht = leta.cumsum(-1).exp()                            # H_t
    Bt = (a @ Ht[..., None])[..., 0]                      # B_t = sum_{s<=t} a[t, s] H_s
    A_prev = F.pad(At[..., :-1], (1, 0), value=1.0)       # A_{t-1}, with A_{-1} = 1
    B_prev = F.pad(Bt[..., :-1], (1, 0))                  # B_{t-1}, with B_{-1} = 0
    P_prev = F.pad(P[..., :-1, :], (0, 0, 1, 0))          # row t holds P[t-1, :] (strictly lower)
    L = P_prev * th2[..., None, :] * (k @ k.transpose(-1, -2))
    eye = torch.eye(C, dtype=k.dtype)
    rhs = torch.cat((v, k * A_prev[..., None], k * B_prev[..., None]), -1)
    E0, WM, WS = torch.linalg.solve_triangular(eye + L, rhs, upper=False, unitriangular=True).split((dv, dk, dk), -1)
    qkc = (q @ k.transpose(-1, -2)) * P * th2[..., None, :]
    qa, qb = q * At[..., None], q * Bt[..., None]
    km = k * (th2 * P[..., -1, :])[..., None]             # end-of-chunk write weights into M
    ks = k * (th2 * h[..., -1, :])[..., None]             # end-of-chunk write weights into S
    aC, bC, hC = (z[..., -1, None, None] for z in (At, Bt, Ht))
    Mt = k.new_zeros(B, H, dk, dv)                        # M^T and S^T (d_k x d_v)
    St = k.new_zeros(B, H, dk, dv)
    out = []
    for c in range(n):
        E = E0[:, :, c] - WM[:, :, c] @ Mt - WS[:, :, c] @ St
        out.append(qa[:, :, c] @ Mt + qb[:, :, c] @ St + qkc[:, :, c] @ E)
        Mt, St = (aC[:, :, c] * Mt + bC[:, :, c] * St + km[:, :, c].transpose(-1, -2) @ E,
                  hC[:, :, c] * St + ks[:, :, c].transpose(-1, -2) @ E)
    return torch.stack(out, 2).permute(0, 2, 3, 1, 4).reshape(B, n * C, H, dv)[:, :T]


class SlidingWindowAttention(nn.Module):
    """Causal multi-head attention over the last `window` positions plus an always-visible prefix, RoPE."""

    def __init__(self, d=D, H=ATT_H, dh=ATT_DH, window=WINDOW, prefix=N_PERSIST):
        super().__init__()
        self.H, self.dh, self.window, self.prefix = H, dh, window, prefix
        self.qkv = nn.Linear(d, 3 * H * dh, bias=False)
        self.o = nn.Linear(H * dh, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = ob.apply_rope(q, cos[:T], sin[:T]), ob.apply_rope(k, cos[:T], sin[:T])
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        mask = (j <= i) & ((j > i - self.window) | (j < self.prefix))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

    def step(self, x, pos, cache, cos, sin):
        B = x.shape[0]
        q, k, v = self.qkv(x).view(B, 1, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        c, s = cos[pos:pos + 1], sin[pos:pos + 1]
        q, k = ob.apply_rope(q, c, s), ob.apply_rope(k, c, s)
        kp, vp, kw, vw = cache if cache is not None else (k[:, :, :0],) * 4
        if pos < self.prefix:
            kp, vp = torch.cat((kp, k), 2), torch.cat((vp, v), 2)
        else:
            kw, vw = torch.cat((kw, k), 2)[:, :, -self.window:], torch.cat((vw, v), 2)[:, :, -self.window:]
        y = F.scaled_dot_product_attention(q, torch.cat((kp, kw), 2), torch.cat((vp, vw), 2))
        return self.o(y.transpose(1, 2).reshape(B, -1)), (kp, vp, kw, vw)


class NeuralMemory(nn.Module):
    """Linear long-term memory per head, written online by gradient steps with momentum and decay."""

    def __init__(self, d=D, H=MEM_H, dk=MEM_DK, dv=MEM_DV, kconv=KCONV):
        super().__init__()
        self.H, self.dk, self.dv, self.kc = H, dk, dv, kconv
        self.cw = H * (2 * dk + dv)
        self.inp = nn.Linear(d, self.cw, bias=False)                       # W_Q | W_K | W_V
        self.conv = nn.Conv1d(self.cw, self.cw, kconv, groups=self.cw, bias=False)
        self.gates = nn.Linear(d, 3 * H)                                    # alpha | eta | theta logits
        with torch.no_grad():
            a0 = torch.logspace(-3, -2, H)
            self.gates.bias.copy_(torch.cat((torch.log(a0 / (1 - a0)), torch.zeros(H),
                                             torch.full((H,), math.log(0.1 / 0.9)))))    # theta0 = 0.5 * 0.1
        self.out = nn.Linear(H * dv, d, bias=False)                         # W_O

    def _qkv(self, c):
        sh = c.shape[:-1]
        q, k, v = c.split((self.H * self.dk, self.H * self.dk, self.H * self.dv), -1)
        return (l2n(q.reshape(*sh, self.H, self.dk)), l2n(k.reshape(*sh, self.H, self.dk)),
                v.reshape(*sh, self.H, self.dv))

    def _gates(self, u):
        a, e, t = self.gates(u).split(self.H, -1)
        return -F.softplus(a), -F.softplus(-e), torch.sigmoid(-e) * torch.sigmoid(t)   # log(1-alpha), log(eta), theta

    def forward(self, u):
        c = F.silu(self.conv(F.pad(self.inp(u).transpose(1, 2), (self.kc - 1, 0))).transpose(1, 2))
        q, k, v = self._qkv(c)
        lr1, leta, th = self._gates(u)
        return self.out(memory_chunked(q, k, v, lr1, leta, 2.0 * th).flatten(-2))

    def step(self, u, cache=None):
        B = u.shape[0]
        if cache is None:
            conv = u.new_zeros(B, self.cw, self.kc - 1)
            M = u.new_zeros(B, self.H, self.dv, self.dk)
            S = torch.zeros_like(M)
        else:
            conv, M, S = cache
        win = torch.cat((conv, self.inp(u)[..., None]), -1)
        q, k, v = self._qkv(F.silu((win * self.conv.weight[:, 0]).sum(-1)))
        lr1, leta, th = self._gates(u)
        grad = 2.0 * ((M @ k[..., None])[..., 0] - v)[..., None] * k[:, :, None, :]   # grad of ||M k - v||^2
        S = leta.exp()[..., None, None] * S - th[..., None, None] * grad
        M = lr1.exp()[..., None, None] * M + S
        return self.out((M @ q[..., None])[..., 0].flatten(-2)), (win[..., 1:], M, S)


class MAGBlock(nn.Module):
    def __init__(self, d=D, ff=FF):
        super().__init__()
        self.n1, self.att, self.mem = ob.RMS(d), SlidingWindowAttention(d), NeuralMemory(d)
        self.ny, self.nm = ob.RMS(d), ob.RMS(d)
        self.n2, self.f1, self.f2 = ob.RMS(d), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def _mlp(self, h):
        return h + self.f2(F.gelu(self.f1(self.n2(h))))

    def forward(self, h, cos, sin):
        u = self.n1(h)
        return self._mlp(h + self.ny(self.att(u, cos, sin)) * torch.sigmoid(self.nm(self.mem(u))))

    def step(self, h, pos, cache, cos, sin):
        ac, mc = cache if cache is not None else (None, None)
        u = self.n1(h)
        y, ac = self.att.step(u, pos, ac, cos, sin)
        m, mc = self.mem.step(u, mc)
        return self._mlp(h + self.ny(y) * torch.sigmoid(self.nm(m))), (ac, mc)


class TitansMAG(nn.Module):
    def __init__(self, d=D, layers=ob.LAYERS):
        super().__init__()
        self.e = nn.Embedding(ob.VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.persist = nn.Parameter(torch.randn(N_PERSIST, d) * 0.02)
        self.blocks = nn.ModuleList(MAGBlock(d) for _ in range(layers))
        self.n = ob.RMS(d)
        c, s = ob.rope_tables(MAXPOS, ATT_DH)
        self.register_buffer('cos', c, persistent=False)
        self.register_buffer('sin', s, persistent=False)
        self.last_penalty = None

    def forward(self, x, hard=False):
        h = torch.cat((self.persist[None].expand(x.shape[0], -1, -1), self.e(x)), 1)
        for b in self.blocks:
            h = b(h, self.cos, self.sin)
        return F.linear(self.n(h[:, N_PERSIST:]), self.e.weight)

    def _advance(self, h, pos, caches):
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step(h, pos, c, self.cos, self.sin)
            new.append(c)
        return h, new

    def step(self, tok, state, hard=True):
        if state is None:                          # run the persistent tokens first
            pos, caches = 0, [None] * len(self.blocks)
            for p in range(N_PERSIST):
                _, caches = self._advance(self.persist[p][None].expand(tok.shape[0], -1), pos, caches)
                pos += 1
        else:
            pos, caches = state
        h, caches = self._advance(self.e(tok), pos, caches)
        return F.linear(self.n(h), self.e.weight), (pos + 1, caches)


def build():
    return TitansMAG()
