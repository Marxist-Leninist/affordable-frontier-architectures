#!/usr/bin/env python3
"""tune_discounts.py --name X -- how much better would the chain be with discounts chosen by likelihood?

Chain only, no transformer. The discounts in use are the closed-form n1/(n1+2 n2). Here a coordinate search picks D_k
to maximise the chain's leave-one-block-out likelihood on TRAIN text (dev and test are not consulted), and the dev and
test losses are then reported for both sets of discounts. Nothing in the stored tables is changed.
"""
import argparse
import json
import os

import numpy as np
import torch

from markov import MarkovPrior

GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]


def held(pr, ids, T):
    nw = (len(ids) - 1) // T
    tot = 0.0
    for s in range(0, nw, 64):
        idx = torch.arange(s, min(nw, s + 64))[:, None] * T + torch.arange(T)[None, :]
        tot += float(-pr.target_logprob(ids[idx], ids[idx + 1]).sum())
    return tot / (nw * T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--windows', type=int, default=1500)
    a = ap.parse_args()
    torch.set_num_threads(1)
    d = os.path.join('data', a.name)
    tables = np.load(os.path.join(d, 'markov.npz'))
    order = int(tables['order'])
    pr = MarkovPrior(tables, order, btables=os.path.join(d, 'blocks_cache'))
    tr, dev, test = (torch.from_numpy(np.load(os.path.join(d, f + '.npy')).astype(np.int64)) for f in ('train', 'dev', 'test'))
    g = torch.Generator().manual_seed(123)
    st = torch.randint(0, len(tr) - a.T - 1, (a.windows,), generator=g)

    def train_bo():
        tot = 0.0
        for s in range(0, len(st), 100):
            idx = st[s:s + 100, None] + torch.arange(a.T)[None, :]
            tot += float(-pr.target_logprob(tr[idx], tr[idx + 1], mode='lobo', tpos=idx + 1).sum())
        return tot / (len(st) * a.T)

    D0 = dict(pr.D)
    out = {'closed_form': {'D': D0, 'train_block_out': train_bo(), 'dev': held(pr, dev, a.T), 'test': held(pr, test, a.T)}}
    print('closed form', {k: round(v, 3) for k, v in D0.items()}, {k: round(v, 4) for k, v in out['closed_form'].items() if k != 'D'}, flush=True)
    best = train_bo()
    for rnd in range(2):
        for k in range(2, order + 1):
            keep = pr.D[k]
            for dv in GRID:
                pr.D[k] = dv
                v = train_bo()
                if v < best - 1e-6:
                    best, keep = v, dv
            pr.D[k] = keep
        print('round', rnd, {k: v for k, v in pr.D.items()}, round(best, 4), flush=True)
    out['tuned'] = {'D': dict(pr.D), 'train_block_out': best, 'dev': held(pr, dev, a.T), 'test': held(pr, test, a.T)}
    out['gain_dev'] = out['closed_form']['dev'] - out['tuned']['dev']
    out['gain_test'] = out['closed_form']['test'] - out['tuned']['test']
    print('tuned', pr.D, {k: round(v, 4) for k, v in out['tuned'].items() if k != 'D'})
    print(f"chain-alone gain from tuned discounts: dev {out['gain_dev']:+.4f}  test {out['gain_test']:+.4f} nats")
    json.dump(out, open(os.path.join(d, 'discount_tuning.json'), 'w'), indent=1, default=float)
    print('DONE')


if __name__ == '__main__':
    main()
