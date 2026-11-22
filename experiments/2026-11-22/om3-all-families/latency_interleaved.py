#!/usr/bin/env python3
"""Prefill and decode latency for every arm, measured INTERLEAVED in one process so that interference from other jobs
on a shared machine hits every model alike. Same quantities as om2bench.timing(): prefill = one forward pass over 256
bytes of a test article; decode = per-token time of 192 greedy steps after a 64-byte prompt (positions 65-256).
All models are built first (one thread, untrained weights, except arms named in --trained ARM=PATH, e.g. the
state-gated store, whose decode skips its lookup when its trained gate is shut). Then for ITERS iterations, in an order
rotated every iteration, each model does one timed prefill and one timed 192-token decode from a copy of its prompt
state. Reported per arm: medians over all iterations and tokens, and medians per block of iterations (BLOCKS blocks)
for the spread analysis.
Usage: latency_interleaved.py ARM[,ARM...] OUT_JSON [--iters 20] [--warmup 2] [--blocks 3] [--trained ARM=PATH ...]"""
import argparse, copy, json, os, statistics, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import om3bench  # noqa: E402,F401  (patches om2bench.build)
import om2bench as ob  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('arms')
p.add_argument('out')
p.add_argument('--iters', type=int, default=20)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--blocks', type=int, default=3)
p.add_argument('--trained', nargs='*', default=[])
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(1)
arms = a.arms.split(',')
trained = dict(x.split('=', 1) for x in a.trained)
_, _, test, _ = ob.load_all()
sample = next(w['data'] for w in test if w['domain'] == 'article')
x = torch.from_numpy(sample[:256])[None]
models, start = {}, {}
with torch.no_grad():
    for arm in arms:
        m = ob.build(arm)
        if arm in trained:
            m.load_state_dict(torch.load(trained[arm], map_location='cpu'))
        m.eval()
        for _ in range(3):
            m(x, hard=True)
        st, lg = None, None
        for t in sample[:64]:
            lg, st = m.step(torch.tensor([int(t)]), st, hard=True)
        models[arm], start[arm] = m, (st, lg.argmax(-1))
    pre = {arm: [] for arm in arms}
    dec = {arm: [] for arm in arms}           # list per iteration of 192 per-token seconds
    for it in range(a.warmup + a.iters):
        k = it % len(arms)
        for arm in arms[k:] + arms[:k]:
            m = models[arm]
            t0 = time.perf_counter()
            m(x, hard=True)
            tp = time.perf_counter() - t0
            st, tok = copy.deepcopy(start[arm][0]), start[arm][1].clone()
            per = []
            for _ in range(192):
                t0 = time.perf_counter()
                lg, st = m.step(tok, st, hard=True)
                per.append(time.perf_counter() - t0)
                tok = lg.argmax(-1)
            if it >= a.warmup:
                pre[arm].append(tp)
                dec[arm].append(per)
out = {}
for arm in arms:
    allp = sorted(t for per in dec[arm] for t in per)
    nb = a.iters // a.blocks
    blocks = [range(b * nb, (b + 1) * nb if b < a.blocks - 1 else a.iters) for b in range(a.blocks)]
    out[arm] = dict(
        prefill_tok_s=256 / statistics.median(pre[arm]), prefill_ms_median=1000 * statistics.median(pre[arm]),
        decode_ms_median=1000 * statistics.median(allp), decode_ms_p95=1000 * allp[int(0.95 * len(allp)) - 1],
        decode_tok_s=len(allp) / sum(allp),
        per_iteration_decode_ms_median=[1000 * statistics.median(per) for per in dec[arm]],
        per_iteration_prefill_ms=[1000 * t for t in pre[arm]],
        blocks=[dict(prefill_tok_s=256 / statistics.median([pre[arm][i] for i in blk]),
                     decode_ms_median=1000 * statistics.median([t for i in blk for t in dec[arm][i]]),
                     decode_ms_p95=1000 * sorted(t for i in blk for t in dec[arm][i])[int(0.95 * 192 * len(blk)) - 1])
                for blk in blocks],
        trained_weights=arm in trained)
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
Path(a.out).write_text(json.dumps(dict(cpu=cpu, threads=1, iters=a.iters, warmup=a.warmup, blocks=a.blocks, torch=torch.__version__,
                                       time=time.strftime('%Y-%m-%d %H:%M:%S'), arms=out), indent=1) + '\n')
for arm in arms:
    o = out[arm]
    print(f"{arm:8s} prefill {o['prefill_tok_s']:8.0f} tok/s  decode {o['decode_ms_median']:.2f} ms (p95 {o['decode_ms_p95']:.2f}) "
          f"blocks {[round(b['decode_ms_median'], 2) for b in o['blocks']]}", flush=True)
