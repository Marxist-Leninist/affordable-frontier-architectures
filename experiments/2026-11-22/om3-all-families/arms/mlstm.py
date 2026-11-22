"""mlstm -- xLSTM mLSTM blocks (matrix memory with exponential gating) for the OM3 byte-level benchmark.

SOURCE
  M. Beck, K. Poeppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter,
  S. Hochreiter, "xLSTM: Extended Long Short-Term Memory", NeurIPS 2024, arXiv:2405.04517
  (Sec. 2.3 mLSTM cell and its log-space stabiliser, Sec. 2.4 mLSTM block with pre up-projection,
  appendix: stabilised parallel formulation of the mLSTM).
  Details the paper leaves open (gate inputs, block-diagonal q/k/v maps, initialisation, epsilon) follow
  the authors' reference code github.com/NX-AI/xlstm (mLSTMLayer, mLSTMCell, parallel_stabilized_simple,
  recurrent_step_stabilized_simple).

MODEL (470,688 parameters; target 471,264, -0.12%)
  tokens -> embedding E (256 x 128, N(0, 0.02^2), tied with the output layer)
         -> 4 x [ x <- x + mLSTMBlock(LN(x)) ] -> LN -> logits = LN(x) E^T
  LN = LayerNorm with a weight and no bias (as in the reference code).

mLSTM BLOCK (d = 128, projection factor 2 -> inner width 256, 4 heads of width d_h = 64)
  [x_m, z] = W_up LN(x)                          W_up: 128 -> 2 x 256 (pre up-projection)
  x_c      = swish(CausalConv1d_4(x_m))           depthwise, kernel 4, left zero padding, with bias
  q = W_q x_c,  k = W_k x_c,  v = W_v x_m         causal conv only on the q/k path; block-diagonal maps
                                                  (64 blocks of 4 x 4, no bias)
  it = W_i [q; k; v] + b_i,  ft = W_f [q; k; v] + b_f       one scalar per head and time step
  h~ = mLSTM cell (below), per head
  h  = GroupNorm_heads(h~) + s * x_c              per-head LayerNorm (weight, no bias); s = learnable skip
  y  = W_down (h * swish(z))                      output gate = swish of the second branch; W_down: 256 -> 128

mLSTM CELL (per head)
  C_t = f_t C_{t-1} + i_t v_t k_t^T               matrix memory, d_h x d_h
  n_t = f_t n_{t-1} + i_t k_t                     normaliser state
  h~_t = C_t q_t / max(|n_t^T q_t|, 1)
  with k_t scaled by 1/sqrt(d_h), exponential input gate i_t = exp(it_t), sigmoid forget gate f_t = sigmoid(ft_t).
  Log-space stabiliser (recurrent form, used by step()):
     m_t  = max(log f_t + m_{t-1}, it_t),  m_0 = -inf
     i'_t = exp(it_t - m_t),  f'_t = exp(log f_t + m_{t-1} - m_t)
     C'_t = f'_t C'_{t-1} + i'_t v_t k_t^T,  n'_t = f'_t n'_{t-1} + i'_t k_t     (C' = C e^-m, n' = n e^-m)
     h~_t = C'_t q_t / (max(|n'_t^T q_t|, exp(-m_t)) + eps)
  Stabilised parallel form (used by forward()):
     log D_ij = sum_{j<s<=i} log f_s + it_j  for j <= i,  -inf for j > i
     m_i = max_j log D_ij,   D'_ij = exp(log D_ij - m_i)
     Cbar = (Q K^T / sqrt(d_h)) o D'
     h~_i = sum_j Cbar_ij v_j / (max(|sum_j Cbar_ij|, exp(-m_i)) + eps)
  The parallel m_i equals the recurrent m_t, so both forms compute the same function and step()
  reproduces forward() up to float32 rounding.
  Implementation notes (no change to the function at float32 resolution):
   - the segment sums are (F_i - F_j) with the running sums F_t = sum_{s<=t} log f_s accumulated in float64
     and split into a float32 head and tail (F = hi + lo), so they stay accurate when gates saturate;
   - the row normaliser divides (Cbar V) instead of Cbar;
   - entries of D' below 1e-20 (more than 46 nats under their row maximum) are set to exactly 0: with
     saturated gates such entries are float32 denormals, which made the cell about 3x slower on CPU,
     and they are far below float32 resolution (outputs and gradients unchanged in the checks);
   - forward() evaluates the parallel form through ParallelMLSTM, an autograd.Function with a hand-written
     backward pass (the exact gradient of the expression above, including the paths through m_i and the
     max(|.|, exp(-m)) normaliser); it keeps two T x T matrices per layer instead of five.  parallel_mlstm()
     is the plain autograd version of the same function, kept as the reference it is checked against
     (float64 gradcheck; float32 gradients agree to ~1e-7 relative).

INITIALISATION (reference code): W_up, W_q, W_k, W_v ~ N(0, 2/(5d)); W_down ~ N(0, (2/(L sqrt(d)))^2), L = 4;
  W_i = W_f = 0, b_i ~ N(0, 0.1^2), b_f = linspace(3, 6) over the 4 heads; skip s = 1; norm weights = 1;
  causal conv: PyTorch default.

SIMPLIFICATIONS / CHOICES VERSUS THE PAPER
  1. Only mLSTM blocks (xLSTM[1:0]): no sLSTM blocks and no separate FFN; the pre up-projection block
     carries the MLP role, as in the paper's mLSTM-only models.
  2. The cell's output gate o_t = sigmoid(W_o x_t + b_o) is realised, as in the mLSTM block of the paper's
     architecture and in the reference code (z branch, ogate_act_fn = SiLU), by swish(z) applied after the
     group norm and the learnable skip; there is no separate sigmoid output gate.
  3. The gate pre-activations it, ft are linear in [q; k; v] (reference code) rather than in the block
     input x_t as the cell equations write them (w^T x_t + b).
  4. No biases in W_up, W_q, W_k, W_v, W_down (reference-code default); the cell equations include b_q, b_k, b_v.
  5. eps = 1e-6 is added to the normaliser (reference code; not in the paper's equation).
  6. Forget gate: sigmoid (one of the paper's two options; the exp forget gate is not used).
  7. Embedding initialised with std 0.02 (benchmark convention) instead of the reference small-init;
     no dropout; optimiser, schedule and weight decay are the benchmark's.
  8. No chunkwise or fused kernel: training and evaluation use the exact O(T^2) stabilised parallel form.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D, L, V = ob.D_MODEL, ob.LAYERS, ob.VOCAB    # 128, 4, 256
PF, NH, KC, QB = 2, 4, 4, 4                   # projection factor, heads, conv kernel, q/k/v block size
DI = PF * D                                   # inner width 256
DH = DI // NH                                 # head width 64
EPS = 1e-6                                    # normaliser guard (reference code)
TINY = 1e-20                                  # D' entries at or below this are set to 0 (denormal guard)
_UPPER = {}


def _upper(T):
    """Boolean (T, T) mask of the strictly upper triangle (j > i)."""
    if T not in _UPPER:
        _UPPER[T] = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    return _UPPER[T]


def _log_d(it, lf):
    """log D_ij = sum_{j<s<=i} lf_s + it_j for j <= i, -inf for j > i.  it, lf: (..., T) -> (..., T, T)."""
    cs = torch.cumsum(lf.double(), -1)                    # F_t, float64
    hi = cs.to(it.dtype)
    lo = (cs - hi.double()).to(it.dtype)                  # F = hi + lo
    logd = hi[..., :, None] - hi[..., None, :]            # correctly rounded hi_i - hi_j
    logd += lo[..., :, None]
    logd += (it - lo)[..., None, :]
    return logd.masked_fill_(_upper(it.shape[-1]), float('-inf'))


def parallel_mlstm(q, k, v, it, ft):
    """Reference (plain autograd) stabilised parallel mLSTM. q, k, v: (B, NH, T, DH); it, ft: (B, NH, T)."""
    logd = _log_d(it, F.logsigmoid(ft))
    m = logd.amax(-1, keepdim=True)                                          # m_i
    c = (q @ (k / math.sqrt(q.shape[-1])).transpose(-1, -2)) * torch.exp(logd - m)   # Cbar
    nrm = torch.maximum(c.sum(-1, keepdim=True).abs(), torch.exp(-m)) + EPS
    return (c @ v) / nrm


class ParallelMLSTM(torch.autograd.Function):
    """parallel_mlstm(q, k, v, it, ft) with lf = logsigmoid(ft) as input, and a hand-written backward."""

    @staticmethod
    def forward(ctx, q, k, v, it, lf):
        ks = k / math.sqrt(q.shape[-1])
        logd = _log_d(it, lf)
        m, idx = logd.max(-1, keepdim=True)
        E = F.threshold_(logd.sub_(m).exp_(), TINY, 0.0)                     # D'
        C = torch.matmul(q, ks.transpose(-1, -2)).mul_(E)                    # Cbar
        s = C.sum(-1, keepdim=True)
        a = torch.exp(-m)
        den = torch.maximum(s.abs(), a) + EPS
        h = torch.matmul(C, v).div_(den)
        ctx.save_for_backward(q, ks, v, E, C, s, a, den, h, idx)
        return h

    @staticmethod
    def backward(ctx, g):
        q, ks, v, E, C, s, a, den, h, idx = ctx.saved_tensors
        dN = g / den                                                         # h = N / den, N = Cbar V
        dden = -(g * h).sum(-1, keepdim=True) / den
        big = s.abs() >= a                                                   # den = max(|s|, a) + eps
        ds = torch.where(big, dden * torch.sign(s), torch.zeros_like(s))
        da = torch.where(big, torch.zeros_like(s), dden)
        dv = torch.matmul(C.transpose(-1, -2), dN)
        dC = torch.matmul(dN, v.transpose(-1, -2)).add_(ds)                  # from N and from s = rowsum(Cbar)
        dlogd = dC * C                                                       # through D' = exp(log D - m)
        dS = dC.mul_(E)                                                      # through Q K^T
        dm = -dlogd.sum(-1, keepdim=True) - a * da                          # m enters D' and a = exp(-m)
        dlogd.scatter_add_(-1, idx, dm)                                      # m = max_j log D_ij
        dq = torch.matmul(dS, ks)
        dk = torch.matmul(dS.transpose(-1, -2), q) / math.sqrt(q.shape[-1])
        dit = dlogd.sum(-2)                                                  # log D_ij = F_i - F_j + it_j
        dcs = dlogd.sum(-1) - dit
        dlf = dcs.flip(-1).cumsum(-1).flip(-1)                               # F_t = sum_{s<=t} lf_s
        return dq, dk, dv, dit, dlf


def recurrent_mlstm(q, k, v, it, ft, state):
    """One stabilised recurrent mLSTM step. q, k, v: (B, NH, DH); it, ft: (B, NH); state (C', n', m) or None."""
    if state is None:
        B = q.shape[0]
        C = q.new_zeros(B, NH, DH, DH)
        n = q.new_zeros(B, NH, DH)
        m = q.new_full((B, NH), float('-inf'))
    else:
        C, n, m = state
    lf = F.logsigmoid(ft)
    m_new = torch.maximum(lf + m, it)
    fa = torch.exp(lf + m - m_new)
    ia = torch.exp(it - m_new)
    ks = k / math.sqrt(DH)
    C = fa[..., None, None] * C + ia[..., None, None] * (ks[..., :, None] * v[..., None, :])   # C[a, b] = sum k_a v_b
    n = fa[..., None] * n + ia[..., None] * ks
    num = (q[..., :, None] * C).sum(-2)                                      # q^T C'
    den = torch.maximum((q * n).sum(-1).abs(), torch.exp(-m_new)) + EPS
    return num / den[..., None], (C, n, m_new)


class LayerNormNoBias(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.w, None, self.eps)


class Headwise(nn.Module):
    """Block-diagonal linear map: DI/QB blocks of QB x QB, no bias (reference LinearHeadwiseExpand)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.empty(DI // QB, QB, QB))
        nn.init.normal_(self.w, 0.0, math.sqrt(2.0 / (5 * D)))

    def forward(self, x):
        s = x.shape
        y = torch.einsum('...hi,hoi->...ho', x.reshape(*s[:-1], DI // QB, QB), self.w)
        return y.reshape(s)


class MLSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNormNoBias(D)
        self.up = nn.Linear(D, 2 * DI, bias=False)
        self.conv = nn.Conv1d(DI, DI, KC, groups=DI)
        self.wq, self.wk, self.wv = Headwise(), Headwise(), Headwise()
        self.igate = nn.Linear(3 * DI, NH)
        self.fgate = nn.Linear(3 * DI, NH)
        self.gn_w = nn.Parameter(torch.ones(DI))
        self.skip = nn.Parameter(torch.ones(DI))
        self.down = nn.Linear(DI, D, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.up.weight, 0.0, math.sqrt(2.0 / (5 * D)))
            nn.init.normal_(self.down.weight, 0.0, 2.0 / L / math.sqrt(D))
            nn.init.zeros_(self.igate.weight)
            nn.init.normal_(self.igate.bias, 0.0, 0.1)
            nn.init.zeros_(self.fgate.weight)
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, NH))

    def _gates(self, q, k, v):
        g = torch.cat((q, k, v), -1)
        return self.igate(g), self.fgate(g)

    def _out(self, h, xc, z):
        """h: (..., NH, DH) cell output -> per-head group norm, learnable skip, output gate, down-projection."""
        hn = F.layer_norm(h, (DH,), None, None, 1e-5).flatten(-2) * self.gn_w
        return self.down((hn + self.skip * xc) * F.silu(z))

    def forward(self, x):
        B, T, _ = x.shape
        xm, z = self.up(self.norm(x)).chunk(2, -1)
        xc = F.silu(self.conv(F.pad(xm.transpose(1, 2), (KC - 1, 0))).transpose(1, 2))
        q, k, v = self.wq(xc), self.wk(xc), self.wv(xm)
        it, ft = self._gates(q, k, v)

        def heads(t):
            return t.reshape(B, T, NH, DH).transpose(1, 2)

        h = ParallelMLSTM.apply(heads(q), heads(k), heads(v), it.transpose(1, 2), F.logsigmoid(ft).transpose(1, 2))
        return x + self._out(h.transpose(1, 2), xc, z)

    def step(self, x, cache):
        B = x.shape[0]
        xm, z = self.up(self.norm(x)).chunk(2, -1)
        conv, cell = (x.new_zeros(B, DI, KC - 1), None) if cache is None else cache
        win = torch.cat((conv, xm[..., None]), -1)
        xc = F.silu((win * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        q, k, v = self.wq(xc), self.wk(xc), self.wv(xm)
        it, ft = self._gates(q, k, v)
        h, cell = recurrent_mlstm(q.reshape(B, NH, DH), k.reshape(B, NH, DH), v.reshape(B, NH, DH), it, ft, cell)
        return x + self._out(h, xc, z), (win[..., 1:], cell)


class MLSTMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(V, D)
        nn.init.normal_(self.e.weight, std=0.02)
        self.blocks = nn.ModuleList(MLSTMBlock() for _ in range(L))
        self.n = LayerNormNoBias(D)
        self.last_penalty = None

    def forward(self, x, hard=False):
        h = self.e(x)
        for b in self.blocks:
            h = b(h)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        h = self.e(tok)
        caches = state if state is not None else [None] * len(self.blocks)
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step(h, c)
            new.append(c)
        return F.linear(self.n(h), self.e.weight), new


def build():
    return MLSTMLM()
