#!/usr/bin/env python3
"""fuse_eval.py [--tokens 58000000] -- does mixing a counted Markov chain into the served 698B model's predictions
lower its held-out loss?  Offline, no request to the worker.

Inputs (copied by fetch.sh):
  dev_depths.json             per-token NLL of the served model (full depth) on the first 128 targets of the
                              trainer's held-out windows, collected by the depth-audit lane through /score
  devset_v3_first129.json     the token ids of those windows (first 129 of each)
  devset_v3.bin               the full windows, int32 [8, 2049] (used to check the chain against the QA lane's numbers)
  fwedu_tokens_116m.bin       fineweb-edu tokens, o200k_base, dev documents already removed (QA lane's cache)
Fusion tested: a mixture  p = (1 - g) * p_model + g * p_chain  with ONE weight g for all positions. It needs only the
two probabilities of the actual next token, so it can be measured exactly from what is already on disk. g is chosen on
the other windows and read on the held-out one (leave-one-window-out), and also shown at a fixed 0.5.
"""
import argparse
import json
import math
import time

import numpy as np

from chain_big import Chain

V = 200064
GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


def mix(m, c, g):
    """per-token NLL of the mixture from per-token NLLs of the model (m) and the chain (c)"""
    return -np.logaddexp(math.log(1 - g) - m, math.log(g) - c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=58_000_000)
    ap.add_argument('--out', default='fusion_result.json')
    a = ap.parse_args()
    t0 = time.time()
    tok = np.fromfile('fwedu_tokens_116m.bin', dtype=np.int32, offset=128, count=a.tokens)
    assert tok.min() >= 0 and tok.max() < V, (tok.min(), tok.max())
    ch = Chain(tok, V)
    print(f'chain built from {len(tok):,} tokens in {time.time() - t0:.0f} s: {len(ch.u2):,} bigram types, '
          f'{len(ch.u3):,} trigram types, discounts {ch.D2:.3f} {ch.D3:.3f}', flush=True)
    res = {'chain_tokens': int(len(tok)), 'bigram_types': int(len(ch.u2)), 'trigram_types': int(len(ch.u3))}

    # 1) the chain alone on the trainer's full held-out windows (compare: QA lane, Kneser-Ney, same token count)
    dev = np.fromfile('devset_v3.bin', dtype=np.int32).reshape(8, 2049)
    full = {o: float(-np.log(ch.target_prob(dev, order=o)).mean()) for o in (2, 3)}
    uni = float(-np.log((ch.c1[dev[:, 1:]] + ch.delta) / (ch.N + ch.delta * V)).mean())
    res['chain_alone_full_dev'] = {'unigram': uni, 'order1_bigram': full[2], 'order2_trigram': full[3]}
    print(f'chain alone on the full held-out windows (16,384 targets): unigram {uni:.3f}  first-order {full[2]:.3f}  '
          f'second-order {full[3]:.3f}   [model: 6.30-6.38]', flush=True)

    # 2) fusion on the tokens the served model has been scored on
    wins = [w['ids'] for w in json.load(open('devset_v3_first129.json'))['windows']]
    dd = json.load(open('dev_depths.json'))
    have = sorted(int(k.split('_')[0]) for k in dd if k.endswith('_full'))
    M = np.array([dd[f'{w}_full']['per_token'] for w in have])                  # [W, 128] model NLL
    ids = np.array([wins[w] for w in have])
    assert ids.shape[1] == M.shape[1] + 1
    assert all(np.array_equal(ids[i], dev[w, :ids.shape[1]]) for i, w in enumerate(have)), 'windows differ from devset_v3'
    C3 = -np.log(ch.target_prob(ids, order=3))
    C2 = -np.log(ch.target_prob(ids, order=2))
    U = -np.log((ch.c1[ids[:, 1:]] + ch.delta) / (ch.N + ch.delta * V))
    W, T = M.shape
    rows = {'model alone (698B, step 7036, full depth)': M, 'chain alone, second-order': C3, 'chain alone, first-order': C2,
            'unigram alone': U}
    for name, c in (('second-order chain', C3), ('first-order chain', C2), ('unigram', U)):
        rows[f'mixture with {name}, g = 0.5'] = mix(M, c, 0.5)
        if W >= 2:                                                             # g picked on the other windows
            cv = np.zeros_like(M)
            picks = []
            for w in range(W):
                others = [i for i in range(W) if i != w]
                g = min(GRID, key=lambda gg: mix(M[others], c[others], gg).mean())
                picks.append(g)
                cv[w] = mix(M[w], c[w], g)
            rows[f'mixture with {name}, g fitted on the other windows'] = cv
            res[f'g_picks_{name}'] = picks
    print(f'\nserved model scored on windows {have}: {W * T} tokens (the first {T} targets of each window)\n')
    print('| predictor | mean NLL | vs model | first 16 tokens | tokens 17-128 | per window |')
    print('|---|---|---|---|---|---|')
    base = M.mean()
    out = {}
    for name, x in rows.items():
        d = x - M
        se = float(d.mean(1).std(ddof=1) / math.sqrt(W)) if W > 1 else float('nan')
        out[name] = {'mean': float(x.mean()), 'minus_model': float(x.mean() - base), 'se_windows': se,
                     'first16': float(x[:, :16].mean()), 'rest': float(x[:, 16:].mean()), 'per_window': [float(v) for v in x.mean(1)]}
        print(f"| {name} | {x.mean():.3f} | {x.mean() - base:+.3f} | {x[:, :16].mean():.3f} | {x[:, 16:].mean():.3f} | "
              + ' '.join(f'{v:.2f}' for v in x.mean(1)) + ' |')
    res.update({'windows': have, 'tokens': int(W * T), 'rows': out, 'uniform_guess': math.log(V), 'seconds': time.time() - t0})
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'\nuniform guess over the vocabulary = {math.log(V):.2f} nats; done in {time.time() - t0:.0f} s')
    print('FUSE-DONE')


if __name__ == '__main__':
    main()
