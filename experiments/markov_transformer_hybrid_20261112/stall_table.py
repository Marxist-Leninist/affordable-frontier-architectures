#!/usr/bin/env python3
"""stall_table.py -- TinyStories seeds 1-8: the plain hybrid (mc3_poe), the aux hybrid (mc3_poe_aux), the classical
model and the classical model with the chain mixed in afterwards. A run counts as stalled when its dev loss falls by
less than 0.05 nats between steps 400 and 2,000 (the classical model and the healthy hybrid runs fall 0.15-0.9).
Seed-paired 95 % t-intervals over the seeds that have every run. Read-only."""
import json
import math
import os

T975 = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624}


def load(name):
    f = 'runs/%s.json' % name
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d if d.get('status') == 'done' else None


def dev_at(d, step):
    return [e for e in d['evals'] if e['step'] == step][0]['dev']


rows = {}
for s in range(1, 9):
    b = load('main_tinystories_base_lr4e-3_s%d' % s)
    p = load('post_tinystories_base_lr4e-3_s%d' % s) if s <= 3 else b
    h = load('main_tinystories_mc3_poe_lr2e-3_s%d' % s)
    a = load('main_tinystories_mc3_poe_aux_lr2e-3_s%d' % s)
    if not (b and p and h and a and 'posthoc_chain' in p):
        print('seed %d incomplete' % s)
        continue
    r = dict(base=b['test_loss'], post=p['posthoc_chain']['test']['mix'], g=p['posthoc_chain']['g_dev'], hyb=h['test_loss'], aux=a['test_loss'],
             hyb_fall=dev_at(h, 400) - dev_at(h, 2000), aux_fall=dev_at(a, 400) - dev_at(a, 2000), base_fall=dev_at(b, 400) - dev_at(b, 2000))
    rows[s] = r
    print('seed %d: classical %.4f | + chain afterwards %.4f (g %.2f) | plain hybrid %.4f (dev fall 400->2000 %.3f%s) | aux hybrid %.4f (fall %.3f%s) | classical fall %.3f' % (
        s, r['base'], r['post'], r['g'], r['hyb'], r['hyb_fall'], ' STALLED' if r['hyb_fall'] < 0.05 else '', r['aux'], r['aux_fall'], ' STALLED' if r['aux_fall'] < 0.05 else '', r['base_fall']))
seeds = sorted(rows)
n = len(seeds)
print('seeds with every run: %s (n=%d)' % (seeds, n))
print('plain hybrid stalled on %d of %d seeds; aux hybrid on %d of %d' % (sum(rows[s]['hyb_fall'] < 0.05 for s in seeds), n, sum(rows[s]['aux_fall'] < 0.05 for s in seeds), n))


def ci(d):
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    h = T975[len(d) - 1] * sd / math.sqrt(len(d))
    return m, m - h, m + h, sum(1 for x in d if x < 0)


for a_, b_, label in (('hyb', 'base', 'plain hybrid minus classical'), ('aux', 'base', 'aux hybrid minus classical'),
                      ('post', 'base', 'chain afterwards minus classical'), ('aux', 'post', 'aux hybrid minus chain afterwards'),
                      ('hyb', 'post', 'plain hybrid minus chain afterwards')):
    d = [rows[s][a_] - rows[s][b_] for s in seeds]
    m, lo, hi, neg = ci(d)
    print('%-38s mean %+.4f [%+.4f, %+.4f], negative on %d of %d seeds | %s' % (label, m, lo, hi, neg, n, ' '.join('%+.3f' % x for x in d)))
print('aux test losses: %s' % ' '.join('%.3f' % rows[s]['aux'] for s in seeds))
print('chain-afterwards test losses: %s; g chosen %s' % (' '.join('%.3f' % rows[s]['post'] for s in seeds), sorted(set(rows[s]['g'] for s in seeds))))
print('STALL-TABLE-DONE')
