"""Exact incremental decoding for ASI-Arch #879 FusionGatedFIRNet (`delta_net_psfr`).

Why this exists
---------------
The exported layer (original/delta_net_psfr.py, kept byte-for-byte) recomputes everything from zero state on
every call: `_retention_delta_chunkwise` always starts S at zeros (the cache it writes is never read back) and both
`_DepthwiseFIRConv1d` branches left-pad zeros. Feeding a sequence in pieces therefore gives different outputs from
one full-prefix call (trained first-layer max error 1.54 / 1.64 / 2.04 for seeds 1-3 in the v44 audit).

This module reuses the layer's own parameters and reproduces the full-prefix computation exactly when the input
arrives as a prefill followed by any number of pieces (normally one token at a time).

Semantics reproduced (all checked against the original full-prefix forward in verify_psfr_cache.py)
------------------------------------------------------------------------------------------------
* q, k, v: projection -> causal depthwise ShortConvolution (kernel 4, SiLU) -> 2 heads x 32 channels.
* beta = sigmoid(b_proj(x)); inside the kernel q and k are L2-normalised with eps 1e-6.
* Delta rule inside a 32-token chunk: S <- S + beta * k (v - k^T S)^T, and the output is q^T S (post-update).
* Chunks sit on ABSOLUTE positions 0, 32, 64, ... After each complete chunk the next chunk starts from
  S_end - (1 - lambda) * S_chunk_start, lambda = 0.6 + 0.4 * sigmoid(retention_param) per head. This is the
  source's chunk-boundary retention (not a per-token decay). We apply it lazily, when the first token of the next
  chunk arrives, so a partial chunk is never finalised early.
* FIR branches (kernels 3 and 63) read v_direct, so the cache keeps its last 62 steps.
* Fusion-gate features, the epsilon-floored sigmoid gates, the per-head RMSNorm and o_proj are token-local.

Per-layer cache: three conv tails (3 steps each), the 62-step FIR tail, S_chunk_start, S, and the position.
Its size is fixed (independent of context length).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

CHUNK = 32


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


class PSFRState:
    """Cache for one PSFR layer. Tensors are never modified in place; every step returns a new state."""

    __slots__ = ("conv_q", "conv_k", "conv_v", "fir", "s_in", "s", "pos")

    def __init__(self, conv_q, conv_k, conv_v, fir, s_in, s, pos):
        self.conv_q, self.conv_k, self.conv_v = conv_q, conv_k, conv_v
        self.fir, self.s_in, self.s, self.pos = fir, s_in, s, int(pos)

    def tensors(self):
        return (self.conv_q, self.conv_k, self.conv_v, self.fir, self.s_in, self.s)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


def empty_state(layer, batch: int, dtype, device) -> PSFRState:
    kc = layer.q_conv1d.kernel_size[0] - 1
    kf = layer.fir_long.kernel_size - 1
    assert layer.fir_short.kernel_size - 1 <= kf
    h, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim
    z = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)
    return PSFRState(z(batch, layer.key_dim, kc), z(batch, layer.key_dim, kc), z(batch, layer.value_dim, kc),
                     z(batch, kf, h, dv), z(batch, h, dk, dv), z(batch, h, dk, dv), 0)


def _short_conv(conv, x: torch.Tensor, tail: torch.Tensor):
    """Causal depthwise conv continuing from `tail` (the previous k-1 inputs, channels-first)."""
    xt = x.transpose(1, 2)
    combined = torch.cat((tail, xt), -1)
    y = F.conv1d(combined, conv.weight, conv.bias, groups=conv.groups).transpose(1, 2)
    if conv.activation == "silu":
        y = F.silu(y)
    return y, combined[..., -tail.shape[-1]:]


def _fir(fir_module, hist: torch.Tensor, x: torch.Tensor):
    """Depthwise causal FIR over [history, new]; returns the outputs for the new steps only."""
    k = fir_module.kernel_size
    b, t, h, d = x.shape
    seq = torch.cat((hist[:, hist.shape[1] - (k - 1):], x), 1)            # [B, k-1+T, H, D]
    seq = seq.permute(0, 2, 3, 1).reshape(b, h * d, -1)
    w = fir_module.filters.reshape(h * d, 1, k)
    y = F.conv1d(seq, w, groups=h * d)                                      # [B, H*D, T]
    return y.reshape(b, h, d, t).permute(0, 3, 1, 2)


def _delta_tokens(q, k, v, beta, lam, s_in, s, pos):
    """Token-by-token delta rule with lazy chunk-boundary retention. q,k are already L2-normalised. [B,H,T,D]."""
    outs = []
    lam_m1 = (lam - 1.0)[None, :, None, None]
    for t in range(q.shape[2]):
        if pos % CHUNK == 0 and pos > 0:          # previous chunk is complete: start the next one
            s = s + lam_m1 * s_in
            s_in = s
        k_t = k[:, :, t]
        resid = v[:, :, t] - torch.einsum("bhk,bhkv->bhv", k_t, s)
        s = s + beta[:, :, t, None, None] * k_t[..., :, None] * resid[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], s))
        pos += 1
    return torch.stack(outs, 2), s_in, s, pos


def _delta_chunks(q, k, v, beta, lam, s_in, s, pos):
    """Whole chunks starting on a chunk boundary, with the source's chunk algebra but a carried-in state."""
    b, h, t, dk = q.shape
    assert t % CHUNK == 0 and pos % CHUNK == 0
    n = t // CHUNK
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = (z.reshape(b, h, n, CHUNK, z.shape[-1]) for z in (q, k, v, k_beta))
    eye = torch.eye(CHUNK, dtype=q.dtype, device=q.device)
    tri = torch.triu(torch.ones_like(eye, dtype=torch.bool), 0)
    strict = torch.triu(torch.ones_like(eye, dtype=torch.bool), 1)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, CHUNK):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + eye
    u = inv @ v
    w = inv @ k_beta
    lam_m1 = (lam - 1.0)[None, :, None, None]
    out = torch.empty_like(v)
    for idx in range(n):
        if pos > 0:
            s = s + lam_m1 * s_in
        s_in = s
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ s
        out[:, :, idx] = q_i @ s + attn_local @ u_i
        s = s + k_i.transpose(-1, -2) @ u_i
        pos += CHUNK
    return out.reshape(b, h, t, -1), s_in, s, pos


def _delta(q, k, v, beta, lam, st: PSFRState, use_chunks: bool = True):
    """Dispatch: finish the current partial chunk token by token, whole chunks in chunk form, then the tail."""
    q, k = _l2(q), _l2(k)
    t, pos, s_in, s = q.shape[2], st.pos, st.s_in, st.s
    head = min(t, (-pos) % CHUNK)
    body = ((t - head) // CHUNK) * CHUNK if use_chunks else 0
    pieces, a = [], 0
    for span, fn in ((head, _delta_tokens), (body, _delta_chunks), (t - head - body, _delta_tokens)):
        if span:
            o, s_in, s, pos = fn(q[:, :, a:a + span], k[:, :, a:a + span], v[:, :, a:a + span],
                                 beta[:, :, a:a + span], lam, s_in, s, pos)
            pieces.append(o)
            a += span
    return torch.cat(pieces, 2), s_in, s, pos


def layer_step(layer, x: torch.Tensor, st: PSFRState, use_chunks: bool = True):
    """Run one PSFR layer on new hidden states x [B, T, D] that continue the sequence held in `st`."""
    assert not layer.use_gate and layer.use_beta and not layer.allow_neg_eigval and layer.use_retention
    assert layer.qk_activation == "silu" and layer.qk_norm == "l2"
    b, t, _ = x.shape
    h, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim
    q_lin, cq = _short_conv(layer.q_conv1d, layer.q_proj(x), st.conv_q)
    k_lin, ck = _short_conv(layer.k_conv1d, layer.k_proj(x), st.conv_k)
    v_lin, cv = _short_conv(layer.v_conv1d, layer.v_proj(x), st.conv_v)
    q = q_lin.reshape(b, t, h, dk)
    k = k_lin.reshape(b, t, h, dk)
    v_direct = v_lin.reshape(b, t, h, dv)
    beta = layer.b_proj(x).sigmoid()
    lam = layer.retention_min + (1.0 - layer.retention_min) * torch.sigmoid(layer.retention_param)
    delta_out, s_in, s, pos = _delta(q.transpose(1, 2), k.transpose(1, 2), v_direct.transpose(1, 2),
                                     beta.transpose(1, 2), lam, st, use_chunks)
    delta_out = delta_out.transpose(1, 2)
    local_short = _fir(layer.fir_short, st.fir, v_direct)
    local_long = _fir(layer.fir_long, st.fir, v_direct)
    stat = lambda z: z.abs().mean(-1)
    feat = torch.cat([x, stat(local_short), stat(local_long), stat(delta_out), stat(v_direct)], -1)
    wts = layer.fusion_gate(feat)
    w_s, w_l, w_d, w_v = torch.split(wts, 1, dim=-1)
    fused = w_s * local_short + w_l * local_long + w_d * delta_out + w_v * v_direct
    out = layer.o_proj(layer.o_norm(fused).reshape(b, t, h * dv))
    fir = torch.cat((st.fir, v_direct), 1)[:, -st.fir.shape[1]:]
    return out, PSFRState(cq, ck, cv, fir, s_in, s, pos)


def lm_step(model, tokens: torch.Tensor, states=None, use_chunks: bool = True):
    """Run the AlternatingLM (PSFR arm) on new tokens [B, T], continuing from `states` (None = empty)."""
    x = model.emb(tokens)
    if states is None:
        states = [empty_state(blk.mix.layer, tokens.shape[0], x.dtype, x.device) for blk in model.blocks]
    new = []
    for blk, st in zip(model.blocks, states):
        z, st = layer_step(blk.mix.layer, blk.n1(x), st, use_chunks)
        x = x + z
        y = blk.n2(x)
        x = x + blk.down(F.silu(blk.gate(y)) * blk.up(y))
        new.append(st)
    return model.head(model.norm(x)), new


def state_nbytes(states) -> int:
    return sum(st.nbytes() for st in states)
