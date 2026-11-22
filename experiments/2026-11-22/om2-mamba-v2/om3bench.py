#!/usr/bin/env python3
"""OM3: the OM2 benchmark (om2bench.py, unchanged) with more architecture arms loaded from arms/<arm>.py.

Training, data, evaluation, recall probes and timing are om2bench's own; this file only extends build().
Extra arm 'm2k' is OM2's 'm2' compute (Mamba-2 on the chunked kernel) without the transformers import, for speed tables.
Usage: om3bench.py run --arm ARM --seed S --out DIR [om2bench run options]
"""
import argparse, importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import om2bench as ob  # noqa: E402

_orig_build = ob.build


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
