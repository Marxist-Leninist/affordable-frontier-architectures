"""Exact, fast CPU path for Hugging Face transformers Mamba-2 (Mamba2ForCausalLM).

On CPU, transformers runs Mamba-2 through `Mamba2Mixer.torch_forward`, a "naive" chunked SSD
that materialises (batch, chunk, l, s, heads, state) products and repeats B/C for every head.
This module computes the same function with einsum contractions that compute C.B^T once per
group of heads, so it does less work and allocates far less memory, while matching the original
to floating-point rounding (see test_equivalence.py).

    from transformers import Mamba2ForCausalLM
    from mamba2_cpu_ssd import patch_mamba2
    model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
    patch_mamba2(model)          # training, scoring and prefill now use the fast kernel

Only the full-sequence path (training, scoring, prompt prefill) is replaced. Single-token
decoding with a cache keeps transformers' own recurrent step, which is already O(1) per token;
the patched prefill writes the same conv and SSM cache states, so generation continues exactly.
The CUDA path (mamba_ssm kernels) is untouched. Patching swaps each mixer's class to
FastMamba2Mixer (a Mamba2Mixer subclass defined here), so patched models pickle and reload
(torch.save / torch.load) as long as this module is importable.

Attribution: ssd_chunked is adapted from the ssd_minimal reference in state-spaces/mamba
(mamba_ssm/modules/ssd_minimal.py, Apache-2.0, (c) 2024 Albert Gu and Tri Dao; the algorithm is the
state-space-duality form of Dao and Gu 2024, arXiv:2405.21060), extended to compute C.B^T once per
group of heads. fast_torch_forward follows transformers' Mamba2Mixer.torch_forward (Apache-2.0,
(c) 2024 state-spaces/mamba2 org and HuggingFace Inc. team) outside the SSD core.
See THIRD_PARTY_NOTICES.md.
"""
import hashlib
import inspect
import warnings

import torch
from torch.nn import functional as F

__version__ = '1.1.0'
# sha256 of inspect.getsource(Mamba2Mixer.torch_forward) in transformers 4.57.6, the version tested
TESTED_TORCH_FORWARD_SHA = '16cec33185821c28fe9bb846275da051d49f31dce5ea81b4919e3bdc9e08520e'

_MASKS = {}
_FAST = None


def _masks(T, device):
    key = (T, str(device))
    if key not in _MASKS:
        _MASKS[key] = (torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), -1),
                       torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), 0))
    return _MASKS[key]


def segsum(x):
    """out[..., i, j] = sum_{j < k <= i} x[..., k] for i >= j, and -inf above the diagonal."""
    T = x.size(-1)
    strict, incl = _masks(T, x.device)
    x = x[..., None].expand(*x.shape, T).masked_fill(~strict, 0.0)
    return torch.cumsum(x, -2).masked_fill(~incl, float('-inf'))


def ssd_chunked(x, dt, A, B, C, chunk_size, initial_state=None):
    """Exact chunked selective state space (Mamba-2 SSD) with grouped B and C.

    x:  (b, T, H, P)   values (before multiplying by dt)
    dt: (b, T, H)      step sizes after softplus and clamping
    A:  (H,)           negative real decay rates (A = -exp(A_log))
    B, C: (b, T, G, N) input and output projections, H divisible by G
    Returns y (b, T, H, P) without the D skip term, and the final state (b, H, P, N).
    All operands are promoted to one dtype (float32 for float32/bf16/fp16 models, float64 for float64
    models) and autocast is disabled inside, like the elementwise SSD core of transformers' torch path.
    """
    dtype = torch.promote_types(torch.promote_types(x.dtype, dt.dtype), torch.promote_types(B.dtype, C.dtype))
    dtype = torch.promote_types(dtype, torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, A, B, C = (t.to(dtype) for t in (x, dt, A, B, C))
        b, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        R = H // G
        Q = chunk_size
        la = A * dt                                  # log of each step's decay (<= 0)
        X = x * dt[..., None]
        pad = (-T) % Q
        if pad:
            X = F.pad(X, (0, 0, 0, 0, 0, pad))
            la = F.pad(la, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))
        c = (T + pad) // Q
        X = X.reshape(b, c, Q, G, R, P)
        Bc = B.reshape(b, c, Q, G, N)
        Cc = C.reshape(b, c, Q, G, N)
        Ah = la.reshape(b, c, Q, G, R).permute(0, 3, 4, 1, 2)          # b G R c Q
        Acs = Ah.cumsum(-1)
        Lm = torch.exp(segsum(Ah))                                       # b G R c Q Q
        CB = torch.einsum('bclgn,bcsgn->bgcls', Cc, Bc)                 # one C.B^T per group
        Yd = torch.einsum('bgcls,bgrcls,bcsgrp->bclgrp', CB, Lm, X)      # within-chunk outputs
        ds = torch.exp(Acs[..., -1:] - Acs)
        st = torch.einsum('bclgn,bgrcl,bclgrp->bcgrpn', Bc, ds, X)       # state each chunk adds
        if initial_state is None:
            st0 = torch.zeros_like(st[:, :1])
        else:
            st0 = initial_state.reshape(b, 1, G, R, P, N).to(dtype)
        st = torch.cat((st0, st), 1)
        dc = torch.exp(segsum(F.pad(Acs[..., -1], (1, 0))))              # b G R c+1 c+1
        new = torch.einsum('bgrzc,bcgrpn->bzgrpn', dc, st)               # states at chunk starts
        Yo = torch.einsum('bclgn,bcgrpn,bgrcl->bclgrp', Cc, new[:, :-1], torch.exp(Acs))
        y = (Yd + Yo).reshape(b, c * Q, H, P)[:, :T]
        return y, new[:, -1].reshape(b, H, P, N)


def _hf():
    from transformers.models.mamba2 import modeling_mamba2
    return modeling_mamba2


def torch_forward_source_sha():
    """sha256 of the installed transformers Mamba2Mixer.torch_forward source, or None if it cannot be read."""
    try:
        return hashlib.sha256(inspect.getsource(_hf().Mamba2Mixer.torch_forward).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def fast_torch_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """Drop-in replacement for Mamba2Mixer.torch_forward (same arguments, same outputs and cache)."""
    hf = _hf()
    if cache_params is not None and cache_position is not None and cache_position[0] > 0:
        # one-token decoding step: transformers' own recurrent update
        return hf.Mamba2Mixer.torch_forward(self, hidden_states, cache_params, cache_position, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
    projected_states = self.in_proj(hidden_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, xbc, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    if cache_params is not None:
        t = xbc.transpose(1, 2)
        conv_states = F.pad(t, (cache_params.conv_kernel_size - t.shape[-1], 0))
        cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)
    xbc = self.act(self.conv1d(xbc.transpose(1, 2))[..., :seq_len].transpose(1, 2))
    xbc = hf.apply_mask_to_padding_states(xbc, attention_mask)
    x, B, C = torch.split(xbc, [self.intermediate_size, self.n_groups * self.ssm_state_size,
                                self.n_groups * self.ssm_state_size], dim=-1)
    A = -torch.exp(self.A_log.float())
    dt = F.softplus(dt + self.dt_bias)
    dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
    x = x.reshape(batch_size, seq_len, -1, self.head_dim).float()
    B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
    y, ssm_state = ssd_chunked(x, dt, A, B, C, self.chunk_size)
    y = (y + x * self.D[:, None]).reshape(batch_size, seq_len, -1)
    if cache_params is not None:
        cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)
    scan_output = self.norm(y, gate)
    return self.out_proj(scan_output.to(dtype))


def _fast_class():
    global _FAST
    if _FAST is None:
        class FastMamba2Mixer(_hf().Mamba2Mixer):
            """transformers Mamba2Mixer whose full-sequence CPU path uses ssd_chunked (fast_torch_forward)."""
            torch_forward = fast_torch_forward

        FastMamba2Mixer.__module__ = __name__
        FastMamba2Mixer.__qualname__ = 'FastMamba2Mixer'
        _FAST = FastMamba2Mixer
    return _FAST


def __getattr__(name):           # lets pickle find mamba2_cpu_ssd.FastMamba2Mixer without importing transformers early
    if name == 'FastMamba2Mixer':
        return _fast_class()
    raise AttributeError(name)


def patch_mamba2(model, check_source=True):
    """Route every Mamba2Mixer in `model` through the fast kernel. Returns the number of patched mixers."""
    hf = _hf()
    if check_source:
        sha = torch_forward_source_sha()
        if sha is None:
            warnings.warn('cannot read the transformers Mamba2Mixer.torch_forward source, so this install could not be '
                          'checked against the tested version; run test_equivalence.py before relying on the patch')
        elif sha != TESTED_TORCH_FORWARD_SHA:
            warnings.warn('transformers Mamba2Mixer.torch_forward differs from the version this patch was tested '
                          'against (4.57.6); run test_equivalence.py before relying on it')
    fast = _fast_class()
    n = 0
    for m in model.modules():
        if type(m) is hf.Mamba2Mixer:
            m.__class__ = fast
        if type(m) is fast:
            n += 1
    return n


def unpatch_mamba2(model):
    """Undo patch_mamba2. Returns the number of mixers restored."""
    hf = _hf()
    fast = _fast_class()
    n = 0
    for m in model.modules():
        if type(m) is fast:
            m.__class__ = hf.Mamba2Mixer
            n += 1
    return n
