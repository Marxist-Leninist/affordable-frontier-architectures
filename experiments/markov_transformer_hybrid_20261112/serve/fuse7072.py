#!/usr/bin/env python3
"""fuse7072.py LOOP_EVAL_JSON TABLE_DIR [TABLE_DIR ...] -- the 888B pack (step 7072) mixed offline with each chain
table set, on every held-out window whose plain per-token scores the agillm-ui lane's loop_eval saved (keys
"<w>_plain" -> per_token, the first 128 targets of the devset_v3 windows). No worker time. Run from
/root/markov_800b_20261112."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, 'serve')
import chainmix512 as cmx  # noqa: E402

res = json.load(open(sys.argv[1]))
wins = [w['ids'] for w in json.load(open('devset_v3_first129.json'))['windows']]
have = sorted(int(k.split('_')[0]) for k in res if k.endswith('_plain') and isinstance(res[k], dict) and res[k].get('per_token'))
M = np.array([res['%d_plain' % w]['per_token'] for w in have], dtype=np.float64)
print('888B step 7072, windows %s, %d targets: model alone %.4f nats (per window %s)' % (have, M.size, M.mean(), ' '.join('%.2f' % v for v in M.mean(1))))
for k in sorted(res):
    if 'chain' in k and isinstance(res[k], dict) and res[k].get('per_token'):
        print('  on file: %s mean %.4f (%d tokens)' % (k, float(np.mean(res[k]['per_token'])), len(res[k]['per_token'])))
for folder in sys.argv[2:]:
    t = cmx.Tables(folder)
    C = np.array([-np.log(t.target_prob(wins[w][:M.shape[1] + 1])) for w in have])
    line = '%s (%s): chain alone %.4f' % (folder.rstrip('/').split('/')[-1], t.meta['id'], C.mean())
    best = (None, 9e9)
    for g in [i / 20 for i in range(1, 20)]:
        v = float((-np.logaddexp(math.log(1 - g) - M, math.log(g) - C)).mean())
        if v < best[1]:
            best = (g, v)
        if g in (0.3, 0.5, 0.7):
            line += ' | mix %.1f %.4f' % (g, v)
    line += ' | best g %.2f -> %.4f' % best
    print(line, flush=True)
print('FUSE7072-DONE')
