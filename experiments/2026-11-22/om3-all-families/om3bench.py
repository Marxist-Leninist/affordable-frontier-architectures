#!/usr/bin/env python3
"""OM3: the OM2 benchmark (om2bench.py, unchanged) with more architecture arms loaded from arms/<arm>.py.

Training, data, evaluation, recall probes and timing are om2bench's own; this file only extends build().
Extra arm 'm2k' is OM2's 'm2' compute (Mamba-2 on the chunked kernel) without the transformers import, for speed tables.
Extra arm 'tfplain' is the Transformer with its attention written out in ordinary PyTorch ops (matmul, causal mask,
softmax, matmul) instead of PyTorch's fused scaled_dot_product_attention: a speed control only, the same function to
float32 rounding, so the speed table can show how much of the Transformer's per-step speed comes from that kernel.
Usage: om3bench.py run --arm ARM --seed S --out DIR [om2bench run options]
"""
import argparse, importlib.util, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import om2bench as ob  # noqa: E402

_orig_build = ob.build


class PlainAttn(ob.Attn):
    """ob.Attn with the attention written out (no fused kernel); used only by the 'tfplain' speed control."""

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = ob.apply_rope(q, cos[:T], sin[:T]), ob.apply_rope(k, cos[:T], sin[:T])
        s = (q @ k.transpose(-1, -2)) * self.dh ** -0.5
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float('-inf'))
        return self.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1))

    def step(self, x, pos, cache, cos, sin):
        B = x.shape[0]
        q, k, v = self.qkv(x).view(B, 1, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        c, s = cos[pos:pos + 1], sin[pos:pos + 1]
        q, k = ob.apply_rope(q, c, s), ob.apply_rope(k, c, s)
        if cache is not None:
            k = torch.cat((cache[0], k), 2)
            v = torch.cat((cache[1], v), 2)
        y = ((q @ k.transpose(-1, -2)) * self.dh ** -0.5).softmax(-1) @ v
        return self.o(y.transpose(1, 2).reshape(B, 1, -1)), (k, v)


def load_arm(arm):
    spec = importlib.util.spec_from_file_location(f'arm_{arm}', HERE / 'arms' / f'{arm}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(arm):
    if arm in ob.ARMS:
        return _orig_build(arm)
    if arm == 'm2k':
        return ob.SSMLM([ob.OM2Mixer(signed=False, bcnorm=False) for _ in range(ob.LAYERS)])
    if arm == 'tfplain':
        m = _orig_build('tf')
        for mod in m.modules():
            if type(mod) is ob.Attn:
                mod.__class__ = PlainAttn
        return m
    return load_arm(arm).build()


ob.build = build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run',))
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, default=5617)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--ctx', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--gate-cost', type=float, default=0.02)
    p.add_argument('--gate-hold', type=int, default=500)
    p.add_argument('--gate-ramp', type=int, default=250)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default='runs/tmp')
    p.add_argument('--smoke', action='store_true')
    ob.run(p.parse_args())


if __name__ == '__main__':
    main()
