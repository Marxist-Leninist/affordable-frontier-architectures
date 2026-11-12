#!/usr/bin/env python3
"""post_table.py -- the classical model with the chain added AFTERWARDS (weights chosen on dev, scored on test) against
the hybrids trained with the chain, seed by seed, with seed-paired 95 % t-intervals. Read-only."""
import json
import math

T975 = 4.302652729911275


def load(f):
    return json.load(open('runs/' + f))


def ci(diff):
    m = sum(diff) / len(diff)
    sd = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1))
    h = T975 * sd / math.sqrt(len(diff))
    return m, m - h, m + h


for data, trained in (('tinystories', ['mc3_poe_lr2e-3', 'mc3_poe_aux_lr2e-3', 'mc3_poe_bo_lr2e-3']),
                      ('wikitext', ['mc3_poe_bo_lr2e-3', 'mc3_mix_bo_lr2e-3', 'mc3_poe_bo_aux_lr2e-3'])):
    print('=== %s' % data)
    post = {}
    for s in (1, 2, 3):
        p = load('post_%s_base_lr4e-3_s%d.json' % (data, s))
        o = load('main_%s_base_lr4e-3_s%d.json' % (data, s))
        q = p['posthoc_chain']
        post[s] = q
        print('seed %d: classical test %.4f (original run %.4f, same model: %s) | chain alone %.4f | added afterwards: mixture g=%.2f %.4f, logits lam=%.1f %.4f' % (
            s, q['test']['model'], o['test_loss'], abs(q['test']['model'] - o['test_loss']) < 1e-9, q['test']['chain'], q['g_dev'], q['test']['mix'], q['lam_dev'], q['test']['logits']))
    for form in ('mix', 'logits'):
        d = [post[s]['test'][form] - post[s]['test']['model'] for s in (1, 2, 3)]
        print('added afterwards (%s) minus classical: %s | mean %+.4f [%+.4f, %+.4f]' % ((form,) + (' '.join('%+.4f' % x for x in d),) + ci(d)))
        d = [post[s]['test'][form] - post[s]['test']['chain'] for s in (1, 2, 3)]
        print('added afterwards (%s) minus chain alone: %s | mean %+.4f [%+.4f, %+.4f]' % ((form,) + (' '.join('%+.4f' % x for x in d),) + ci(d)))
    best = {s: min(post[s]['test']['mix'], post[s]['test']['logits']) for s in (1, 2, 3)}
    for v in trained:
        t = {s: load('main_%s_%s_s%d.json' % (data, v, s))['test_loss'] for s in (1, 2, 3)}
        for form in ('mix', 'logits'):
            d = [t[s] - post[s]['test'][form] for s in (1, 2, 3)]
            print('trained %-22s minus added-afterwards (%-6s): %s | mean %+.4f [%+.4f, %+.4f]' % ((v, form) + (' '.join('%+.4f' % x for x in d),) + ci(d)))
print('POST-TABLE-DONE')
