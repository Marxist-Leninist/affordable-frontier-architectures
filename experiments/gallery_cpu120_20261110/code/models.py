"""Byte-level language-model wrapper shared by every arm.

256-way untied embedding and output head, three pre-norm residual blocks (mixer, then SwiGLU feed-forward).

* Gallery arms: width 64, mixer = an unchanged ASI-Arch gallery `DeltaNet(hidden_size=64, num_heads=2)`.
* Transformer control: width 96, mixer = causal attention with rotary positions, 4 heads.

Feed-forward widths are chosen so the whole model is the nearest feasible size to 350,000 parameters; no
dummy parameters are added. This shape reproduces, to the parameter, all 18 gallery parameter counts and the
Transformer count (349,920) recorded for the earlier 350k-parameter cohort (see SMOKE.json).
"""
import hashlib
import inspect
import json

import torch
from torch import nn
from torch.nn import functional as F

import fla_cpu

LAYERS, TARGET = 3, 350_000
GALLERY_WIDTH, TRANSFORMER_WIDTH = 64, 96


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Attention(nn.Module):
    """Causal self-attention with rotary positions.

    The rotation tables are built in float32 and the rotation is applied in float32 (float64 when the model is
    in float64), so BF16 autocast cannot merge neighbouring positions.
    """

    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.d = heads, d
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        k = d // heads
        ang = torch.arange(4096, dtype=torch.float32)[:, None] * 10000.0 ** (-torch.arange(0, k, 2, dtype=torch.float32) / k)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def rope(self, x):
        t = x.shape[-2]
        work = x if x.dtype == torch.float64 else x.float()
        c, s = self.cos[:t].to(work.dtype), self.sin[:t].to(work.dtype)
        u, v = work[..., ::2], work[..., 1::2]
        return torch.stack((u * c - v * s, u * s + v * c), -1).flatten(-2).to(x.dtype)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [a.reshape(b, t, self.h, d // self.h).transpose(1, 2) for a in (q, k, v)]
        q, k = self.rope(q), self.rope(k)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(z.transpose(1, 2).reshape(b, t, d))


class GalleryMixer(nn.Module):
    def __init__(self, module, width, layer_idx):
        super().__init__()
        kwargs = {"hidden_size": width, "num_heads": 2}
        if "layer_idx" in inspect.signature(module.DeltaNet.__init__).parameters:
            kwargs["layer_idx"] = layer_idx
        self.layer = module.DeltaNet(**kwargs)

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class Block(nn.Module):
    def __init__(self, width, mixer, inner):
        super().__init__()
        self.width = width
        self.n1, self.n2, self.mix = Norm(width), Norm(width), mixer
        self.set_inner(inner)

    def set_inner(self, inner):
        self.gate = nn.Linear(self.width, inner, bias=False)
        self.up = nn.Linear(self.width, inner, bias=False)
        self.down = nn.Linear(inner, self.width, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        z = self.n2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class LM(nn.Module):
    def __init__(self, width, mixers, inners):
        super().__init__()
        self.emb = nn.Embedding(256, width)
        self.blocks = nn.ModuleList([Block(width, m, i) for m, i in zip(mixers, inners)])
        self.norm = Norm(width)
        self.head = nn.Linear(width, 256, bias=False)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def count(m):
    return sum(p.numel() for p in m.parameters())


def build(arm, seed, gallery=None):
    """arm is 'transformer' or a gallery index. Returns (model, info)."""
    torch.manual_seed(1_000_003 * seed + 17)
    if arm == "transformer":
        width = TRANSFORMER_WIDTH
        mixers = [Attention(width) for _ in range(LAYERS)]
        name = "transformer_w96_rope"
    else:
        width = GALLERY_WIDTH
        rec = gallery[int(arm)]
        module = fla_cpu.load_program(rec["index"], rec["program"])
        mixers = [GalleryMixer(module, width, i) for i in range(LAYERS)]
        name = rec["name"]
    model = LM(width, mixers, [1] * LAYERS)
    nonff = count(model) - LAYERS * 3 * width
    total = max(LAYERS * 8, round((TARGET - nonff) / (3 * width)))
    inners = [total // LAYERS + (i < total % LAYERS) for i in range(LAYERS)]
    torch.manual_seed(2_000_003 * seed + 29)
    for b, inner in zip(model.blocks, inners):
        b.set_inner(inner)
    return model, {"arm": str(arm), "name": name, "width": width, "parameters": count(model), "ffn_inner": inners,
                   "mixer_parameters": sum(count(b.mix) for b in model.blocks), "target": TARGET}


def load_gallery(path, audit_path=None):
    """Gallery records by index. With audit_path, only programs that passed static_audit.py are returned."""
    records = {r["index"]: r for r in json.loads(open(path, encoding="utf-8").read())}
    if audit_path:
        audit = json.loads(open(audit_path, encoding="utf-8").read())
        clean = {row["index"]: row["sha256"] for row in audit["rows"] if not row["problems"] and row["defines_DeltaNet"]}
        records = {i: r for i, r in records.items()
                   if clean.get(i) == hashlib.sha256(r["program"].encode("utf-8")).hexdigest()}
    return records
