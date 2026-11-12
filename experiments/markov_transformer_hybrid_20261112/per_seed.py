#!/usr/bin/env python3
"""per_seed.py DATA [VARIANT:LR ...] -- per-seed test losses, the dev curve of each run, and seed-paired differences
against the classical model at ITS best learning rate (default pairs for the two corpora)."""
import glob
import json
import math
import sys

data = sys.argv[1]
pairs = sys.argv[2:] or {'tinystories': ['base:4e-3', 'mc3_poe:2e-3', 'mc3_poe_bo:2e-3', 'mkattn:4e-3'],
                         'wikitext': ['base:4e-3', 'mc3_poe_bo:2e-3', 'mc3_mix_bo:2e-3', 'mkattn:4e-3']}[data]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
runs = {}
for spec in pairs:
    v, lr = spec.split(':')
    for f in sorted(glob.glob('runs/main_%s_%s_lr%s_s*.json' % (data, v, lr))):
        d = json.load(open(f))
        if d.get('status') != 'done':
            continue
        seed = int(f.rsplit('_s', 1)[1].split('.')[0])
        runs.setdefault(spec, {})[seed] = d
for spec in pairs:
    for seed, d in sorted(runs.get(spec, {}).items()):
        ev = d['evals']
        pick = [e for e in ev if e['step'] in (100, 200, 400, 800, 1200, 1600, 2000)]
        extra = {k: round(v, 3) for k, v in (ev[-1].get('extra') or {}).items()} if isinstance(ev[-1].get('extra'), dict) else ''
        print('%-18s s%d test %.4f | dev %s | cpu ms/step %.0f %s' % (spec, seed, d['test_loss'], ' '.join('%d:%.3f' % (e['step'], e['dev']) for e in pick), d.get('cpu_ms_per_step', 0), extra))
base = runs.get(pairs[0], {})
print()
for spec in pairs[1:]:
    r = runs.get(spec, {})
    seeds = sorted(set(r) & set(base))
    if not seeds:
        continue
    diff = [r[s]['test_loss'] - base[s]['test_loss'] for s in seeds]
    m = sum(diff) / len(diff)
    line = '%-18s minus %s, seeds %s: %s | mean %+.4f' % (spec, pairs[0], seeds, ' '.join('%+.4f' % x for x in diff), m)
    if len(diff) >= 2:
        sd = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1))
        h = T975[len(diff) - 1] * sd / math.sqrt(len(diff))
        line += ' | paired 95%% interval [%+.4f, %+.4f] (n=%d)' % (m - h, m + h, len(diff))
    cost = [r[s].get('cpu_ms_per_step', 0) / base[s].get('cpu_ms_per_step', 1) for s in seeds]
    line += ' | CPU per step x%.2f' % (sum(cost) / len(cost))
    print(line)
