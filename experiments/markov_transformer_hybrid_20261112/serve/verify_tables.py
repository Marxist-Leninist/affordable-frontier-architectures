#!/usr/bin/env python3
"""verify_tables.py TABLE_DIR -- the tables that will be served must reproduce the offline fusion measurement
(fuse_eval.py, which used chain_big.py): chain alone and the 0.5 mixture on the tokens the served model was scored on."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, 'serve')
import chainmix512 as cmx

t = cmx.Tables(sys.argv[1])
t.selftest()
wins = [w['ids'] for w in json.load(open('devset_v3_first129.json'))['windows']]
dd = json.load(open('dev_depths.json'))
have = sorted(int(k.split('_')[0]) for k in dd if k.endswith('_full'))
M = np.array([dd['%d_full' % w]['per_token'] for w in have])
C = np.array([-np.log(t.target_prob(wins[w])) for w in have])
mix = -np.logaddexp(math.log(0.5) - M, math.log(0.5) - C)
print('tables', t.meta['id'], 'tokens', int(t.meta['tokens']), 'windows', have, 'scored tokens', M.size)
print('model alone %.3f | chain alone %.3f | mixture g=0.5 %.3f | per window %s' % (
    M.mean(), C.mean(), mix.mean(), ' '.join('%.2f' % v for v in mix.mean(1))))
ref = json.load(open('fusion_result_116000000.json'))
if len(sys.argv) > 2 and sys.argv[2] == 'noassert':
    a = ref['rows']['chain alone, second-order']['mean']
    b = ref['rows']['mixture with second-order chain, g = 0.5']['mean']
    print('for comparison, the plain 116M-token tables on windows %s: chain alone %.3f | mixture %.3f' % (ref['windows'], a, b))
    print('VERIFY-DONE (no assertion: these are different tables)')
elif ref['windows'] == have:
    a = ref['rows']['chain alone, second-order']['mean']
    b = ref['rows']['mixture with second-order chain, g = 0.5']['mean']
    print('offline measurement: chain alone %.3f | mixture %.3f | difference %.2e %.2e' % (a, b, abs(a - C.mean()), abs(b - mix.mean())))
    assert abs(a - C.mean()) < 1e-6 and abs(b - mix.mean()) < 1e-6
    print('VERIFY-OK: the served tables reproduce the offline measurement')
else:
    print('more windows have been scored since the offline run (%s vs %s): numbers above are the current ones' % (have, ref['windows']))
    print('VERIFY-OK')
