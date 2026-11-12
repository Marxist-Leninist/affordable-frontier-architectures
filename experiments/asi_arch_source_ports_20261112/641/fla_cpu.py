"""Plain-PyTorch stand-ins for the few flash-linear-attention pieces the ASI-Arch gallery programs import.

These are dependency references for CPU runs. They are not the official fused FLA kernels and no numerical
parity with those kernels is claimed. Semantics follow the adapter published with the v44 study in this
repository (RMSNorm eps 1e-5, gated RMSNorm = rmsnorm(x) * silu(g), causal depthwise ShortConvolution,
l2norm eps 1e-6), with two additions: ShortConvolution keeps FLA's own default activation ("silu") when a
program does not pass one, and the typing-only imports (fla.models.utils.Cache,
transformers.processing_utils.Unpack) are stubbed.

Gallery source runs unchanged except that @torch.compile decorator lines are dropped. Masked or packed
sequences are refused. Only programs that pass static_audit.py are loaded.
"""
import sys
import types
import typing

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, bias=False, eps=1e-5, **kw):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if (elementwise_affine and bias) else None

    def forward(self, x, *a, **kw):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class FusedRMSNormGated(RMSNorm):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kw):
        super().__init__(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
        if activation not in ("swish", "silu", "sigmoid"):
            raise NotImplementedError(f"gated norm activation {activation!r}")
        self.activation = activation

    def forward(self, x, o, *a, **kw):
        y = super().forward(x)
        return y * (torch.sigmoid(o) if self.activation == "sigmoid" else F.silu(o))


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kw):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.activation = activation
        self.k = kernel_size

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kw):
        if cu_seqlens is not None or mask is not None:
            raise NotImplementedError("masked/packed sequences are intentionally unsupported")
        x = x.transpose(1, 2)
        old = x.new_zeros(x.shape[0], x.shape[1], self.k - 1) if cache is None else cache[..., -(self.k - 1):]
        combined = torch.cat((old, x), -1)
        y = super().forward(combined).transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        if residual is not None:
            y = y + residual
        return y, (combined[..., -self.k:].clone() if output_final_state else None)


def _unsupported(*a, **kw):
    raise NotImplementedError("masked/packed sequences are intentionally unsupported")


def l2norm(x, eps=1e-6, **kw):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)


class Cache:
    def __init__(self, *a, **kw):
        self.layers = {}

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def update(self, *a, layer_idx=0, **kw):
        self.layers[layer_idx] = kw


def install():
    for name in ("fla", "fla.layers", "fla.layers.utils", "fla.modules", "fla.modules.l2norm", "fla.models",
                 "fla.models.utils", "transformers", "transformers.processing_utils"):
        sys.modules[name] = types.ModuleType(name)
    for n in ("get_unpad_data", "index_first_axis", "pad_input"):
        setattr(sys.modules["fla.layers.utils"], n, _unsupported)
    for n, o in (("RMSNorm", RMSNorm), ("FusedRMSNormGated", FusedRMSNormGated), ("ShortConvolution", ShortConvolution)):
        setattr(sys.modules["fla.modules"], n, o)
    sys.modules["fla.modules.l2norm"].l2norm = l2norm
    sys.modules["fla.models.utils"].Cache = Cache
    sys.modules["transformers.processing_utils"].Unpack = typing.Unpack


def load_program(index, source):
    """Run one audited gallery program (a str) and return its module namespace."""
    install()
    kept = "\n".join(line for line in source.splitlines() if not line.strip().startswith("@torch.compile"))
    mod = types.ModuleType(f"asi_gallery_{index}")
    exec(compile(kept, f"<ASI-Arch data/106.json record {index}>", "exec"), mod.__dict__)
    return mod
