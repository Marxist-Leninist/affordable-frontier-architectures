#!/usr/bin/env python3
"""bench_speed.py -- cost of one training step for every variant, measured so that a busy machine cannot bias it.

Each round runs every variant for a few steps, in a rotating order, single-threaded. The figure is process CPU time
(not wall time), and each variant is compared with the baseline OF THE SAME ROUND, so drift in the machine's load
hits both sides alike. Also reports the forward matmul FLOPs (torch's counter) and the CPU time spent in the chain.

  python3 bench_speed.py --data data/tinystories --out bench.json [--rounds 10 --k 5 --bs 16 ...]
"""
import argparse
import json
import os
import statistics as st
import time

import numpy as np
import torch

from markov import MarkovPrior
from model import GPT, lm_loss, variant_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--variants', default='base,mc1_poe,mc3_poe,mc3_mix,mkattn,mc3_poe_mk')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    names = a.variants.split(',')
    assert names[0] == 'base'
    meta = json.load(open(os.path.join(a.data, 'meta.json')))
    V = meta['V']
    tr = torch.from_numpy(np.load(os.path.join(a.data, 'train.npy')).astype(np.int64))
    tables = np.load(os.path.join(a.data, 'markov.npz'))
    priors, M = {}, {}
    btables = None
    for n in names:
        cfg = variant_cfg(n)
        pk = (cfg['order'], cfg['cv'] == 'lobo')
        if cfg['prior'] and pk not in priors:
            if pk[1] and btables is None:
                btables = os.path.join(a.data, 'blocks_cache')
            priors[pk] = MarkovPrior(tables, cfg['order'], btables=btables if pk[1] else None)
        torch.manual_seed(1)
        model = GPT(V, a.T, a.d, a.L, a.H, cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        M[n] = (model, opt, priors.get(pk))
    g = torch.Generator().manual_seed(7)
    ar = torch.arange(a.T)

    def batch():
        ix = torch.randint(0, len(tr) - a.T - 1, (a.bs,), generator=g)
        idx = ix[:, None] + ar[None, :]
        return tr[idx], tr[idx + 1], idx + 1

    def step(n, x, y, tpos, timer=None):
        model, opt, prior = M[n]
        loss, _ = lm_loss(model, prior, x, y, train=True, timer=timer, tpos=tpos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    flops = {}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x, y, tp = batch()
        for n in names:
            with FlopCounterMode(display=False) as fc:
                lm_loss(M[n][0], M[n][2], x, y, train=True, tpos=tp)
            flops[n] = fc.get_total_flops()
    except Exception as e:  # noqa: BLE001
        flops = {'error': repr(e)}
    for n in names:                       # warm-up, not timed
        for _ in range(3):
            step(n, *batch())
    cpu = {n: [] for n in names}
    wall = {n: [] for n in names}
    pri = {n: [] for n in names}
    ratio = {n: [] for n in names}
    for r in range(a.rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        batches = [batch() for _ in range(a.k)]           # the same batches for every variant in this round
        this = {}
        for n in order:
            timer = {}
            c0, w0 = time.process_time(), time.perf_counter()
            for x, y, tp in batches:
                step(n, x, y, tp, timer)
            this[n] = 1000 * (time.process_time() - c0) / a.k
            cpu[n].append(this[n])
            wall[n].append(1000 * (time.perf_counter() - w0) / a.k)
            pri[n].append(1000 * timer.get('prior_cpu', 0.0) / a.k)
        for n in names:
            ratio[n].append(this[n] / this['base'])
        print(f'round {r}: ' + ' '.join(f'{n}={this[n]:.0f}ms' for n in names), flush=True)
    out = {'args': vars(a), 'load': os.getloadavg(), 'torch': torch.__version__, 'flops_forward': flops, 'variants': {}}
    for n in names:
        rs = sorted(ratio[n])
        out['variants'][n] = {'cpu_ms_per_step_median': st.median(cpu[n]), 'wall_ms_per_step_median': st.median(wall[n]),
                              'chain_cpu_ms_per_step_median': st.median(pri[n]),
                              'ratio_vs_base_median': st.median(rs), 'ratio_min': rs[0], 'ratio_max': rs[-1],
                              'ratio_q1': rs[len(rs) // 4], 'ratio_q3': rs[(3 * len(rs)) // 4],
                              'params': sum(p.numel() for p in M[n][0].parameters()),
                              'flops_ratio': (flops[n] / flops['base']) if 'error' not in flops else None}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({n: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} for n, d in out['variants'].items()}, indent=1))
    print('DONE')


if __name__ == '__main__':
    main()
