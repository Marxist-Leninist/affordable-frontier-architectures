#!/usr/bin/env python3
"""stall_look.py -- TinyStories seed 3: why did both chain hybrids stop improving at about 2.66 from step 400, while
seeds 1-2 went on to 2.48? Prints each run's dev curve, the training loss, the learned chain weight and anything else
the evals recorded, side by side with the classical run of the same seed."""
import glob
import json

for v in ('base_lr4e-3', 'mc3_poe_lr2e-3', 'mc3_poe_bo_lr2e-3', 'mc3_mix_bo_lr2e-3'):
    for s in (1, 2, 3):
        f = 'runs/main_tinystories_%s_s%d.json' % (v, s)
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        ev = d['evals']
        keys = sorted(set(k for e in ev for k in e) - {'step', 'dev', 'train_ema', 'cpu_s', 'wall_s', 'prior_cpu_s'})
        print('%-22s s%d test %.4f status %s chain_alone %s' % (v, s, d.get('test_loss', float('nan')), d.get('status'), d.get('chain_alone_test_loss')))
        for e in ev:
            if e['step'] in (25, 50, 100, 200, 400, 800, 1000, 1200, 1600, 2000):
                print('    step %5d dev %.4f train_ema %.4f %s' % (e['step'], e['dev'], e['train_ema'], ' '.join('%s=%s' % (k, round(e[k], 4) if isinstance(e[k], float) else e[k]) for k in keys)))
print('final-step train vs dev gap (overfitting check) per run:')
for f in sorted(glob.glob('runs/main_tinystories_*_s3.json')) + sorted(glob.glob('runs/main_tinystories_*_s1.json')):
    d = json.load(open(f))
    if d.get('status') == 'done':
        e = d['evals'][-1]
        print('  %-50s train_ema %.4f dev %.4f gap %+.4f' % (f.split('/')[-1], e['train_ema'], e['dev'], e['dev'] - e['train_ema']))
