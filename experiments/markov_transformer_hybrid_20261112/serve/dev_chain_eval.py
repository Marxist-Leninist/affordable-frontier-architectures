#!/usr/bin/env python3
"""dev_chain_eval.py TABLE_DIR [TABLE_DIR ...] -- each table set alone on the trainer's 8 held-out windows (all 2,048
targets of each, and the first 128), and mixed with the served model where its per-token scores are on file
(dev_depths.json: the 698B pack, step 7036). Run from /root/markov_800b_20261112."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, 'serve')
import chainmix512 as cmx  # noqa: E402

dev = np.fromfile('devset_v3.bin', dtype=np.int32).reshape(8, 2049)
try:
    dd = json.load(open('dev_depths.json'))
    have = sorted(int(k.split('_')[0]) for k in dd if k.endswith('_full'))
    M = np.array([dd['%d_full' % w]['per_token'] for w in have])
except Exception:  # noqa: BLE001
    have, M = [], None
for folder in sys.argv[1:]:
    t = cmx.Tables(folder)
    t.selftest()
    C = np.array([-np.log(t.target_prob([int(v) for v in dev[w]])) for w in range(8)])      # [8, 2048]
    line = '%s (%s): chain alone, 8 x 2048 targets %.4f | per window %s | first 128 of each %.4f' % (
        folder.rstrip('/').split('/')[-1], t.meta['id'], C.mean(), ' '.join('%.2f' % v for v in C.mean(1)), C[:, :128].mean())
    if M is not None:
        Cm = C[have][:, :M.shape[1]]
        line += ' | with the 698B model on windows %s (%d targets): model %.3f, chain %.3f' % (have, M.size, M.mean(), Cm.mean())
        for g in (0.3, 0.5, 0.7):
            line += ', mix %.1f %.3f' % (g, float((-np.logaddexp(math.log(1 - g) - M, math.log(g) - Cm)).mean()))
    print(line, flush=True)
print('DEV-EVAL-DONE')
