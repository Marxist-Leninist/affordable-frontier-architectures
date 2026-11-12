#!/usr/bin/env python3
"""quality_time.py --data tinystories [--seed 1] -- quality per unit of training time, from the finished runs.

Quality = nats of held-out loss removed from the know-nothing level ln(V) (a uniform guess over the vocabulary).
Time    = CPU minutes of training, plus the one-off counting time for variants that use a Markov chain.

A rate taken at the end of each run is misleading (every model learns fast at first and slowly later), so the rate is
taken where the models can be compared: (a) the time each needs to reach the SAME quality, the classical
transformer's final loss, and (b) the quality each has after the SAME time.
Each variant is shown at its own best learning rate (chosen on the dev loss).
"""
import argparse
import glob
import json
import math
import os

# one-off chain counting time, seconds (prep logs on GETH: build_tables, and build_block_tables for the *_bo variants)
CHAIN_S = {'tinystories': (4.2, 22.8), 'wikitext': (5.1, 36.2)}


def at_time(curve, t):
    """dev loss at total time t (minutes), linear in log(time) between evaluations; None before the first one"""
    if t < curve[0][0]:
        return None
    for (t0, l0), (t1, l1) in zip(curve, curve[1:]):
        if t0 <= t <= t1:
            f = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return l0 + f * (l1 - l0)
    return curve[-1][1]


def time_to(curve, target):
    """minutes until the dev loss is <= target; None if never"""
    if curve[0][1] <= target:
        return curve[0][0]
    for (t0, l0), (t1, l1) in zip(curve, curve[1:]):
        if l1 <= target < l0:
            f = (l0 - target) / (l0 - l1)
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='tinystories')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--steps', type=int, default=2000)
    a = ap.parse_args()
    meta = json.load(open(os.path.join('data', a.data, 'meta.json')))
    lnV = math.log(meta['V'])
    best = {}
    for f in glob.glob(os.path.join('runs', f'{a.tag}_{a.data}_*_s{a.seed}.json')):
        r = json.load(open(f))
        g = r['args']
        if r.get('status') != 'done' or g['steps'] != a.steps:
            continue
        v = g['variant']
        if v not in best or r['evals'][-1]['dev'] < best[v]['evals'][-1]['dev']:
            best[v] = r
    if 'base' not in best:
        raise SystemExit('no finished classical run')
    rows = []
    for v, r in best.items():
        cv = r['cfg'].get('cv', 'loo')
        start = 0.0 if not r['cfg']['prior'] else (CHAIN_S[a.data][0] + (CHAIN_S[a.data][1] if cv == 'lobo' else 0.0)) / 60
        curve = [(start + e['cpu_s'] / 60, e['dev']) for e in r['evals']]
        rows.append(dict(v=v, lr=r['args']['lr'], test=r['test_loss'], dev=r['evals'][-1]['dev'], total=curve[-1][0],
                         ms=r['cpu_ms_per_step'], curve=curve, start=start))
    base = next(x for x in rows if x['v'] == 'base')
    Tc, Lc = base['total'], base['dev']
    chain = meta['chain_alone_loss'][str(meta['order'])]
    rows.append(dict(v='chain alone (order 3)', lr=0, test=chain['test'], dev=chain['dev'], total=CHAIN_S[a.data][0] / 60,
                     ms=0.0, curve=[(CHAIN_S[a.data][0] / 60, chain['dev'])], start=CHAIN_S[a.data][0] / 60))
    rows.sort(key=lambda x: (x['v'] != 'base', x['test']))
    out = []
    print(f'{a.data} seed {a.seed}: ln(V) = {lnV:.3f} nats; classical transformer: lr {base["lr"]:g}, final dev loss {Lc:.3f}, '
          f'test {base["test"]:.3f}, {Tc:.1f} CPU-min for {a.steps} steps\n')
    print('| model | best lr | test loss | CPU-min, full run | CPU-min to reach the classical final quality | quality per CPU-min '
          'at that point (nats/min) | relative to classical | loss after 10% / 25% / 50% / 100% of the classical run time |')
    print('|---|---|---|---|---|---|---|---|')
    for x in rows:
        tt = time_to(x['curve'], Lc)
        rate = (lnV - Lc) / tt if tt else None
        rel = (Tc / tt) if tt else None
        pts = [at_time(x['curve'], f * Tc) for f in (0.1, 0.25, 0.5, 1.0)]
        x.update(time_to_base=tt, rate=rate, rel=rel, at=pts)
        out.append({k: x[k] for k in ('v', 'lr', 'test', 'dev', 'total', 'ms', 'time_to_base', 'rate', 'rel', 'at')})
        print(f"| {x['v']} | {x['lr']:g} | {x['test']:.3f} | {x['total']:.1f} | {'never' if tt is None else format(tt, '.1f')} | "
              f"{'-' if rate is None else format(rate, '.2f')} | {'-' if rel is None else 'x' + format(rel, '.2f')} | "
              + ' / '.join('n/a' if p is None else format(p, '.3f') for p in pts) + ' |')
    os.makedirs('results', exist_ok=True)
    json.dump({'data': a.data, 'seed': a.seed, 'lnV': lnV, 'classical_minutes': Tc, 'classical_final_dev': Lc, 'rows': out},
              open(os.path.join('results', f'quality_time_{a.data}_s{a.seed}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
