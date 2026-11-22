"""hyb: Transformer-Mamba hybrid with a Jamba-style interleaved layer schedule (OM3 study, hybrids group).

Source papers
  [J]  O. Lieber et al. 2024, "Jamba: A Hybrid Transformer-Mamba Language Model", arXiv:2403.19887.
  [M2] T. Dao and A. Gu 2024, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through
       Structured State Space Duality" (Mamba-2), arXiv:2405.21060.
  [A]  A. Vaswani et al. 2017, "Attention Is All You Need", arXiv:1706.03762; rotary position embedding (RoPE)
       from J. Su et al. 2021, "RoFormer", arXiv:2104.09864.

Defining mechanism ([J]): one residual stack that interleaves attention layers with Mamba layers. Most layers are
linear-time state-space mixers with a constant-size recurrent state; the attention layers keep an exact KV cache.
Incremental decoding therefore carries both kinds of cache at once: a conv window plus an SSM state for every Mamba
layer, and a growing (K, V) cache for every attention layer.

Model (bytes, vocab 256, d = 128, 4 pre-norm residual blocks, tied embedding E with std 0.02, final RMSNorm):
  h = E[x]
  block 1  Mamba-2:    h = h + Mamba2(RMS(h))
  block 2  attention:  h = h + Attn(RMS(h));  h = h + W2 GELU(W1 RMS(h))        (W1: 128 -> 171, W2: 171 -> 128)
  block 3  Mamba-2:    h = h + Mamba2(RMS(h))
  block 4  attention:  as block 2
  logits = RMS(h) E^T
Mamba-2 mixer ([M2]) = om2bench.OM2Mixer(signed=False, bcnorm=False), the om2bench 'm2' mixer that reproduces the
Hugging Face Mamba2 layer to < 1e-4: expand 2 (d_inner 256), 8 heads of P = 32 channels, state N = 32, conv k = 4:
  [z, xBC, dt] = W_in u;   xBC = SiLU(causal depthwise conv_4(xBC));   (x, B, C) = split(xBC)
  dt_t = softplus(dt_t + dt_bias);   a_t = exp(-dt_t * exp(A_log))                  (one scalar decay per head)
  H_t = a_t H_{t-1} + dt_t x_t B_t^T   (H_t is P x N per head);   y_t = H_t C_t + D * x_t
  out = W_out RMSNorm(y * SiLU(z))
  forward(): the exact chunked SSD form of [M2] (chunks of 32: masked intra-chunk matmuls plus inter-chunk state
  passing); step(): the recurrence above.
Attention block = om2bench.TFBlock ([A]): 4 heads of 32, (q, k, v) = W_qkv x, RoPE on q and k,
  softmax(q k^T / sqrt(32) + causal mask) v, then W_o, then the GELU MLP. step() appends the rotated k and v to the
  layer's KV cache.
Decoding state: (position, [(conv window, SSM state) or (K, V) cache for each of the 4 blocks]).
Parameters: 471,088 (target 471,264, -0.04%); ff = 171 is the MLP width that lands closest to the target.

Simplifications versus Jamba [J] (the 4-block schedule is fixed by the OM3 study design):
  1. 4 layers at attention:Mamba = 1:1. Jamba uses blocks of l = 8 layers at a:m = 1:7 (one attention layer per
     block).
  2. Mamba-2 (SSD, one scalar decay per head) replaces the Mamba-1 selective scan that Jamba builds on.
  3. In Jamba every layer, Mamba or attention, is followed by an MLP, and every e = 2 layers that MLP is a mixture
     of experts (n = 16 experts, top K = 2). Here only the attention blocks have an MLP (dense, GELU) and the Mamba
     blocks are mixer-only, as in Mamba / Mamba-2 stacks; there is no MoE.
  4. RoPE in the attention layers, where Jamba uses no explicit positional information; multi-head attention with
     4 KV heads, where Jamba uses grouped-query attention.
  5. No RMSNorm on the inner Mamba activations (Jamba adds it to stop loss spikes at 7B scale); the m2 mixer keeps
     only its gated output norm and leaves B and C unnormalised (bcnorm=False).
  6. Tiny scale: d = 128, 4 layers, byte vocabulary, context 256, float32 on CPU.
"""
import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

FF = 171
SCHEDULE = ('mamba', 'attn', 'mamba', 'attn')


class MambaBlock(nn.Module):
    """Pre-norm residual block around the om2bench Mamba-2 mixer: h + Mamba2(RMS(h))."""

    def __init__(self, d):
        super().__init__()
        self.n = ob.RMS(d)
        self.mix = ob.OM2Mixer(d=d, signed=False, bcnorm=False)

    def forward(self, h, cos, sin):
        return h + self.mix(self.n(h))

    def step(self, h, pos, cache, cos, sin):            # h: (B, 1, d); cache: (conv window, SSM state)
        y, cache = self.mix.step(self.n(h[:, 0]), cache)
        return h + y[:, None], cache


class HybridLM(nn.Module):
    def __init__(self, d=ob.D_MODEL, heads=4, ff=FF, schedule=SCHEDULE, maxT=1024):
        super().__init__()
        self.e = nn.Embedding(ob.VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.schedule = tuple(schedule)
        self.blocks = nn.ModuleList(MambaBlock(d) if k == 'mamba' else ob.TFBlock(d, heads, ff)
                                    for k in self.schedule)
        self.n = ob.RMS(d)
        self.dh = d // heads
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
        for b, c in zip(self.blocks, caches):              # Mamba caches and attention KV caches side by side
            h, c = b.step(h, pos, c, cos, sin)
            new.append(c)
        return F.linear(self.n(h), self.e.weight)[:, 0], (pos + 1, new)


def build():
    return HybridLM()
