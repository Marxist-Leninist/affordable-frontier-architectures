"""griffin: Griffin, gated linear recurrences (RG-LRU) mixed with local multi-query attention (OM3, hybrids group).

Source paper
  [G]  S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen,
       S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. De Freitas, C. Gulcehre 2024,
       "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models",
       arXiv:2402.19427.
  [RG] google-deepmind/recurrentgemma, the paper's reference code, consulted only where the paper is silent.
  RoPE: J. Su et al. 2021, arXiv:2104.09864. Multi-query attention: N. Shazeer 2019, arXiv:1911.02150.

Defining mechanism ([G]): residual blocks whose temporal mixing is either a recurrent block built around the
Real-Gated Linear Recurrent Unit (RG-LRU: a diagonal gated linear recurrence whose gates do not depend on the
state) or local sliding-window multi-query attention, in the pattern two recurrent blocks then one local-attention
block. Decoding needs only a fixed-size state: a D_RNN vector plus a conv window per recurrent block, and the last
W keys/values of the single KV head for the attention block.

Model (bytes, vocab 256, D = 128, tied embedding E with std 0.02, final RMSNorm), residual blocks
  rec, rec, local-attn, rec                      ([G]: two recurrent blocks, then one local-MQA block, repeated)
Residual block ([G] Fig. 2a):   h = h + Temporal(RMS(h));   h = h + MLP(RMS(h))
Gated MLP ([G] Fig. 2b):        MLP(x) = W_d (GeLU(W_1 x + b_1) * (W_2 x + b_2)) + b_d,   hidden width F = 114
Recurrent block ([G] Fig. 2c), D_RNN = 176:
  GeLU branch:       g_t = GeLU(W_y x_t + b_y)
  recurrent branch:  u_t = W_x x_t + b_x;   v_t = sum_{j=0..3} w_j * u_{t-j}   (causal depthwise Conv1D, filter 4,
                     no bias: the paper's 4 D_RNN parameters);   h_t = RG-LRU(v)_t
  output:            W_o (h_t * g_t) + b_o
RG-LRU ([G] eqs. 1-4), elementwise over the D_RNN channels; the two gate matrices are block-diagonal with 16 blocks
([G] Sec. 4.1 uses 16 blocks for both the recurrence gate and the input gate):
  r_t = sigmoid(W_a v_t + b_a)                                    recurrence gate
  i_t = sigmoid(W_i v_t + b_i)                                    input gate
  a = sigmoid(Lambda);   log a_t = c r_t log a = -c r_t softplus(-Lambda),   c = 8    (computed in log space, [G])
  h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * v_t),   h_{-1} = 0
  init ([G]): Lambda such that a^c ~ U[0.9, 0.999]; W_a, W_i LeCun normal (std 1/sqrt(fan-in)); b_a = b_i = 0.
  forward(): the diagonal recurrence for all t at once with om2bench.scan_doubling (an exact inclusive affine scan
  in log2(T) doubling steps, exact for up to 1024 steps; longer inputs are scanned in spans of 1024 with the state
  carried from span to span); step(): the explicit recurrence h = a_t h + b_t.
Local attention ([G], MQA): 4 query heads of 32 share one key head and one value head of 32; RoPE on q and k;
  token t attends to positions t-63 .. t (a 64-token sliding window that includes t itself);
  y = W_o softmax(q k^T / sqrt(32) + window mask) v + b_o. step() keeps a rolling cache of the last 64 rotated keys
  and values, so decode memory is constant.
Decoding state: (position, [(conv window, h) for each recurrent block, (K, V) window for the attention block]).
Parameters: 471,040 (target 471,264, -0.05%).

Simplifications versus the paper [G]:
  1. Scale: D = 128, 4 residual blocks, byte vocabulary, context 256 (the smallest model in [G]: D = 768,
     D_RNN = 1024, 12 blocks, M = 3, 2048-token sequences).
  2. MLP hidden width F = 114 (expansion M ~ 0.9) instead of M = 3, so that the parameter target holds while the
     recurrent block keeps the paper's width ratio D_RNN ~ 4D/3 (176 = 4D/3 rounded to a multiple of the 16 gate
     blocks). The MLP then has about as many weights per block as the benchmark's TF arm (ff = 176).
  3. Local attention window 64 instead of 1024 tokens (the context here is 256); 4 query heads of 32 instead of the
     paper's fixed head dimension of 128, which at D = 128 would leave one head (MQA and MHA would coincide).
  4. Initialisation: only Lambda and the gate weights follow the paper. Every other layer uses the PyTorch default
     (the paper does not specify them), and the embedding follows the benchmark convention (normal, std 0.02; no
     sqrt(D) input scaling and no logit soft-cap as in [RG]).
  5. Biases: the paper specifies only the gate biases. Following [RG], W_y, W_x, W_o, both MLP layers and the
     attention output projection have biases; q, k and v do not.
  6. [RG] also drops the sqrt(1 - a_t^2) factor on the first token of each segment and rotates only half of each
     head's dimensions with RoPE. The paper mentions neither; this arm applies the paper's equation at every
     position and rotates all head dimensions (as the benchmark's TF arm does).
  7. Backward pass only: the derivative of sqrt(1 - a_t^2) is clipped at 1000, as in [RG]; forward values are
     unchanged. 1 - a_t^2 is computed as -expm1(2 log a_t) for accuracy when a_t is close to 1.
  8. float32 on CPU; exact (erf) GeLU; no model-parallel sharding; no document resets (training windows are random
     byte spans).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D_RNN, GATE_BLOCKS, FF, HEADS, WINDOW, CONV_K, C_RGLRU = 176, 16, 114, 4, 64, 4, 8.0
PATTERN = ('rec', 'rec', 'attn', 'rec')
MAX_SQRT_GRADIENT = 1000.0
SCAN_SPAN = 1024        # ob.scan_doubling doubles up to k = 512, so it is exact for sequences of up to 1024 steps


def linear_scan(a, b):
    """h_t = a_t h_{t-1} + b_t along dim 1 with h_{-1} = 0, exact for any length: ob.scan_doubling on spans of
    SCAN_SPAN steps, each span adding cumprod(a) times the last state of the span before it."""
    T = a.shape[1]
    if T <= SCAN_SPAN:
        return ob.scan_doubling(a, b)
    outs, h = [], None
    for s in range(0, T, SCAN_SPAN):
        hs = ob.scan_doubling(a[:, s:s + SCAN_SPAN], b[:, s:s + SCAN_SPAN])
        if h is not None:
            hs = hs + torch.cumprod(a[:, s:s + SCAN_SPAN], 1) * h[:, None]
        outs.append(hs)
        h = hs[:, -1]
    return torch.cat(outs, 1)


class SqrtBoundDerivative(torch.autograd.Function):
    """sqrt(x) whose derivative is clipped at MAX_SQRT_GRADIENT in the backward pass ([RG])."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / torch.sqrt(torch.clamp_min(4.0 * x, 1.0 / MAX_SQRT_GRADIENT ** 2))


class BlockDiagonalLinear(nn.Module):
    """y = x W + b with W block-diagonal (nb square blocks); LeCun-normal weights, zero bias."""

    def __init__(self, width, nb):
        super().__init__()
        assert width % nb == 0
        self.nb, self.bw = nb, width // nb
        self.w = nn.Parameter(torch.randn(nb, self.bw, self.bw) / math.sqrt(self.bw))
        self.b = nn.Parameter(torch.zeros(nb, self.bw))

    def forward(self, x):
        xb = x.unflatten(-1, (self.nb, self.bw))
        return (torch.einsum('...hi,hij->...hj', xb, self.w) + self.b).flatten(-2)


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit ([G] eqs. 1-4)."""

    def __init__(self, width, nb, c=C_RGLRU):
        super().__init__()
        self.c = c
        self.gate_a = BlockDiagonalLinear(width, nb)       # recurrence gate r_t
        self.gate_i = BlockDiagonalLinear(width, nb)       # input gate i_t
        ac = torch.empty(width, dtype=torch.float64).uniform_(0.9, 0.999)    # a^c ~ U[0.9, 0.999]
        a = ac ** (1.0 / c)
        self.Lambda = nn.Parameter(torch.log(a / (1.0 - a)).float())        # a = sigmoid(Lambda)

    def coefficients(self, v):
        r = torch.sigmoid(self.gate_a(v))
        i = torch.sigmoid(self.gate_i(v))
        log_a = -self.c * r * F.softplus(-self.Lambda)     # log a_t = c r_t log sigmoid(Lambda)
        a = torch.exp(log_a)
        mult = SqrtBoundDerivative.apply(-torch.expm1(2.0 * log_a))         # sqrt(1 - a_t^2)
        return a, mult * (i * v)

    def forward(self, v):                                  # v: (B, T, width) -> h: (B, T, width)
        a, b = self.coefficients(v)
        return linear_scan(a, b)

    def step(self, v, h):                                  # v: (B, width); h: (B, width), None means h_{-1} = 0
        a, b = self.coefficients(v)
        return b if h is None else a * h + b


class RecurrentBlock(nn.Module):
    """[G] Fig. 2c: GeLU branch times (linear -> causal Conv1D(4) -> RG-LRU), then a linear layer back to D."""

    def __init__(self, d, width=D_RNN, nb=GATE_BLOCKS, k=CONV_K):
        super().__init__()
        self.k = k
        self.lin_y = nn.Linear(d, width)
        self.lin_x = nn.Linear(d, width)
        self.conv = nn.Conv1d(width, width, k, groups=width, bias=False)
        self.lru = RGLRU(width, nb)
        self.lin_out = nn.Linear(width, d)

    def forward(self, x):                                  # x: (B, T, d)
        g = F.gelu(self.lin_y(x))
        u = self.lin_x(x)
        v = self.conv(F.pad(u.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        return self.lin_out(self.lru(v) * g)

    def step(self, x, cache):                              # x: (B, d); cache: (last k-1 conv inputs, h)
        g = F.gelu(self.lin_y(x))
        u = self.lin_x(x)
        if cache is None:
            win, h = u.new_zeros(*u.shape, self.k - 1), None
        else:
            win, h = cache
        win = torch.cat((win, u[..., None]), -1)
        v = (win * self.conv.weight[:, 0]).sum(-1)
        h = self.lru.step(v, h)
        return self.lin_out(h * g), (win[..., 1:], h)


class LocalMQA(nn.Module):
    """Sliding-window multi-query attention: H query heads share a single key/value head; RoPE on q and k."""

    def __init__(self, d, H=HEADS, W=WINDOW):
        super().__init__()
        self.H, self.dh, self.W = H, d // H, W
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * self.dh, bias=False)   # the one shared key head and value head
        self.o = nn.Linear(d, d)

    def forward(self, x, cos, sin):                        # x: (B, T, d)
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)              # B H T dh
        k, v = self.kv(x).view(B, T, 2, 1, self.dh).permute(2, 0, 3, 1, 4)     # each B 1 T dh
        q, k = ob.apply_rope(q, cos[:T], sin[:T]), ob.apply_rope(k, cos[:T], sin[:T])
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.W)  # s in (t - W, t]
        y = F.scaled_dot_product_attention(q, k.expand(B, self.H, T, self.dh), v.expand(B, self.H, T, self.dh),
                                           attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

    def step(self, x, pos, cache, cos, sin):               # x: (B, 1, d); cache: last <= W rotated keys, values
        B = x.shape[0]
        q = self.q(x).view(B, 1, self.H, self.dh).transpose(1, 2)              # B H 1 dh
        k, v = self.kv(x).view(B, 1, 2, 1, self.dh).permute(2, 0, 3, 1, 4)     # each B 1 1 dh
        c, s = cos[pos:pos + 1], sin[pos:pos + 1]
        q, k = ob.apply_rope(q, c, s), ob.apply_rope(k, c, s)
        if cache is not None:
            k = torch.cat((cache[0], k), 2)[:, :, -self.W:]
            v = torch.cat((cache[1], v), 2)[:, :, -self.W:]
        n = k.shape[2]
        y = F.scaled_dot_product_attention(q, k.expand(B, self.H, n, self.dh), v.expand(B, self.H, n, self.dh))
        return self.o(y.transpose(1, 2).reshape(B, 1, -1)), (k, v)


class GatedMLP(nn.Module):
    """[G] Fig. 2b: two branches of width F, GeLU on one, merged by elementwise product, then a linear layer."""

    def __init__(self, d, f=FF):
        super().__init__()
        self.up = nn.Linear(d, 2 * f)
        self.down = nn.Linear(f, d)

    def forward(self, x):
        gate, lin = self.up(x).chunk(2, -1)
        return self.down(F.gelu(gate) * lin)


class ResidualBlock(nn.Module):
    """[G] Fig. 2a: RMSNorm -> temporal mixing -> skip; RMSNorm -> gated MLP -> skip."""

    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = ob.RMS(d), ob.RMS(d)
        self.mix = RecurrentBlock(d) if kind == 'rec' else LocalMQA(d)
        self.mlp = GatedMLP(d)

    def forward(self, x, cos, sin):
        n = self.n1(x)
        x = x + (self.mix(n) if self.kind == 'rec' else self.mix(n, cos, sin))
        return x + self.mlp(self.n2(x))

    def step(self, x, pos, cache, cos, sin):               # x: (B, 1, d)
        n = self.n1(x)
        if self.kind == 'rec':
            y, cache = self.mix.step(n[:, 0], cache)
            y = y[:, None]
        else:
            y, cache = self.mix.step(n, pos, cache, cos, sin)
        x = x + y
        return x + self.mlp(self.n2(x)), cache


class Griffin(nn.Module):
    def __init__(self, d=ob.D_MODEL, pattern=PATTERN, maxT=1024):
        super().__init__()
        self.e = nn.Embedding(ob.VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.pattern = tuple(pattern)
        self.blocks = nn.ModuleList(ResidualBlock(k, d) for k in self.pattern)
        self.n = ob.RMS(d)
        self.dh = d // HEADS
        c, s = ob.rope_tables(maxT, self.dh)
        self.register_buffer('cos', c, persistent=False)
        self.register_buffer('sin', s, persistent=False)
        self.last_penalty = None

    def _rope(self, T):
        if T > self.cos.shape[0]:                          # grow the RoPE tables for long decodes
            c, s = ob.rope_tables(max(T, 2 * self.cos.shape[0]), self.dh)
            self.cos, self.sin = c.to(self.cos.device), s.to(self.sin.device)
        return self.cos, self.sin

    def forward(self, x, hard=False):
        cos, sin = self._rope(x.shape[1])
        h = self.e(x)
        for b in self.blocks:
            h = b(h, cos, sin)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        pos, caches = state if state is not None else (0, [None] * len(self.blocks))
        cos, sin = self._rope(pos + 1)
        h = self.e(tok)[:, None]
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step(h, pos, c, cos, sin)
            new.append(c)
        return F.linear(self.n(h), self.e.weight)[:, 0], (pos + 1, new)


def build():
    return Griffin()
