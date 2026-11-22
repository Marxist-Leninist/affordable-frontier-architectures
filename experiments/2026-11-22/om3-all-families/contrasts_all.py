#!/usr/bin/env python3
"""Paired held-out contrasts for OM2 + OM3 runs: test macro bits/byte of arm A minus arm B, averaged over the seeds both
have, with a descriptive 95% interval from a domain-stratified document bootstrap (2,000 resamples, the same documents
drawn for both arms and every seed). Same method as OM2's summarize.py, vectorised with numpy.
Arm om2g is the state-gated store scored as designed (reads only where its gate exceeds 0.5); om2gsoft is the same
trained model read softly at every token (the score OM2 reported).
Usage: contrasts_all.py RUNS_DIR[,RUNS_DIR...] OUT_JSON [A:B ...]   (default pairs: every arm against tf, m2 and om2)"""
import json, math, sys
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
rows = {}
for d in sys.argv[1].split(','):
    for f in sorted(list(Path(d).glob('*/result.json')) + list(Path(d).glob('*.json'))):   # folders or flat run files
        r = json.loads(f.read_text())
        if isinstance(r, dict) and r.get('status') == 'ok' and 'test_docs' in r and 'arm' in r:
            if r['arm'] == 'om2g' and 'test_docs_hard' in r:      # the state-gated store, scored as designed (hard gate)
                rows.setdefault('om2g', {})[r['seed']] = r['test_docs_hard']
                rows.setdefault('om2gsoft', {})[r['seed']] = r['test_docs']   # the same model read softly at every token
            else:
                rows.setdefault(r['arm'], {})[r['seed']] = r['test_docs']
if len(sys.argv) > 3:
    pairs = [tuple(p.split(':')) for p in sys.argv[3:]]
else:
    pairs = [(a, b) for b in ('tf', 'm2', 'om2') for a in sorted(rows) if a != b and b in rows]
ref = next(iter(next(iter(rows.values())).values()))
doms = sorted({d['domain'] for d in ref})
idx = {dm: np.array([i for i, d in enumerate(ref) if d['domain'] == dm]) for dm in doms}
for arm, by_seed in rows.items():
    for s, docs in by_seed.items():
        assert [d['doc'] for d in docs] == [d['doc'] for d in ref], (arm, s, 'test documents differ')
rng = np.random.default_rng(20261122)
B = 2000
picks = {dm: rng.integers(0, len(ix), size=(B, len(ix))) for dm, ix in idx.items()}


def arrays(docs):
    return np.array([d['nll'] for d in docs]), np.array([d['targets'] for d in docs], dtype=np.float64)


def macro(nll, tg):
    return float(np.mean([nll[idx[dm]].sum() / tg[idx[dm]].sum() / LN2 for dm in doms]))


def boot_macro(nll, tg):
    return np.mean([nll[idx[dm]][picks[dm]].sum(1) / tg[idx[dm]][picks[dm]].sum(1) / LN2 for dm in doms], axis=0)


out = {}
for a, b in pairs:
    if a not in rows or b not in rows:
        continue
    seeds = sorted(set(rows[a]) & set(rows[b]))
    if not seeds:
        continue
    per_seed, boots = [], []
    for s in seeds:
        na, ta = arrays(rows[a][s])
        nb, tb = arrays(rows[b][s])
        per_seed.append(macro(na, ta) - macro(nb, tb))
        boots.append(boot_macro(na, ta) - boot_macro(nb, tb))
    bt = np.sort(np.mean(boots, axis=0))
    out[f'{a} minus {b}'] = dict(mean_bpb=float(np.mean(per_seed)), per_seed=per_seed,
                                 interval95=[float(bt[49]), float(bt[1949])], seeds=seeds)
Path(sys.argv[2]).write_text(json.dumps(dict(method='domain-stratified document bootstrap, 2000 resamples, numpy seed 20261122',
                                             contrasts=out), indent=1) + '\n')
for k, v in out.items():
    print(f"{k:22s} {v['mean_bpb']:+.4f} [{v['interval95'][0]:+.4f}, {v['interval95'][1]:+.4f}] seeds {v['seeds']}")
