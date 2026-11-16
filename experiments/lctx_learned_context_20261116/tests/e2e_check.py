"""checks for e2e_lctx.sh (v1.1): python3 e2e_check.py E2E_DIR  (prints PASS/FAIL per check, exits 1 on any FAIL)"""
import glob
import hashlib
import json
import os
import re
import sys

import torch

D = sys.argv[1]
fails = []


def log(name):
    p = os.path.join(D, name)
    return open(p, errors='replace').read() if os.path.exists(p) else ''


def losses(text):
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^step (\d+) loss ([0-9.]+)', text, re.M)}


def check(cond, what):
    print(('PASS ' if cond else 'FAIL ') + what)
    if not cond:
        fails.append(what)


def status(run):
    p = os.path.join(D, run, 'lctx.status.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def record(run):
    p = os.path.join(D, run, 'state.json')
    return (json.load(open(p)) if os.path.exists(p) else {}).get('lctx') or {}


L = {n: log(n + '.e2e.log') for n in ('A', 'B', 'B2', 'S', 'P', 'D', 'E1', 'E2', 'E3', 'F')}
la, lb = losses(L['A']), losses(L['B'])
for name in ('A', 'B', 'B2', 'S', 'P', 'D', 'E3', 'F'):
    check('Traceback' not in L[name], f'{name}: no traceback')
check(len(la) >= 16 and len(lb) >= 16, f'A and B ran 16 steps ({len(la)}, {len(lb)})')
check('[lctx] installed lctx-v1.1' in L['B'] and 'self-tests passed' in L['B'], 'B: v1.1 installed, self-tests passed')
check('[lctx] update 1 ' in L['B'] and '[lctx] update 3 ' in L['B'], 'B: updates started after the warm passes')
d12 = max(abs(la[s] - lb[s]) for s in (1, 2)) if all(s in la and s in lb for s in (1, 2)) else 9
check(d12 < 2e-3, f'B: steps 1-2 (rate 1e-6, held) match A within 2e-3 (max diff {d12:.2e})')
print(f'INFO B vs A max |loss diff| {max(abs(la[s] - lb[s]) for s in la if s in lb):.3e}; A last {la.get(16)} B last {lb.get(16)}')
st = status('runB')
errs = {k: st.get(k) for k in ('gate_errors', 'recompute_errors', 'fails', 'update_errors', 'persist_errors')}
check(st.get('updates', 0) >= 14 and not any(errs.values()), f"B2 status: updates {st.get('updates')}, errors {errs}")
print('INFO B2 summary', json.dumps(st.get('summary')), 'identity', st.get('identity_max_abs_diff'))
rb = record('runB')
check(rb.get('version') == 'lctx-v1.1-20261116' and rb.get('updates', 0) >= 14 and rb.get('installed') and
      len(rb.get('sha256') or '') == 64, 'B2: state.json carries the v1.1 record with a sha256')
check(re.search(r'\[lctx\] installed.*state \S+lctx\.state\.pt \(update \d+, step', L['B2']) is not None,
      'B2: resume loaded the saved state with the checkpoint\'s update count')
check(len(losses(L['B2'])) >= 6, 'B2: resumed 6 steps')
check('[lctx] installed lctx-v1-20261116' in L['S'] and '[lctx] installed lctx-v1.1' in L['S'] and
      'carried over' in L['S'] and len(losses(L['S'])) >= 14, 'S: v1 -> v1.1 swapped in process, run completed')
ss = status('runS')
check(ss.get('version') == 'lctx-v1.1-20261116' and not any(ss.get(k) for k in errs), 'S: v1.1 status clean after the swap')
pins = sorted(glob.glob(os.path.join(D, 'runP', 'lctx.pinned', '*.pt')))
check(len(pins) >= 1 and 'pinned the gate' in L['P'] and len(losses(L['P'])) >= 10, f'P: PAUSE pinned the gate ({len(pins)})')
if pins:
    pin = torch.load(pins[0], map_location='cpu')
    blob = torch.cat([pin['r'].flatten(), pin['w'].flatten()]).numpy().astype('<f4').tobytes()
    check(hashlib.sha256(blob).hexdigest() == pin['sha256'], 'P: the pin\'s sha256 is the sha256 of its parameters')
check('[lctx] OFF' in L['D'] and len(losses(L['D'])) >= 12 and record('runD').get('active') is False,
      'D: kill switch took effect, record says inactive')
check('WARNING: the last process died inside a step' in L['E2'] and '[lctx] installed' in L['E2'],
      'E: first strike: the relaunch installs again with a warning')
check('two launches in a row' in L['E3'] and len(losses(L['E3'])) >= 1, 'E: second strike: refused, the trainer still trains')
check(record('runE').get('installed') is False or '[lctx] installed' in L['F'], 'E: the refusal is in the record')
check('[lctx] installed' in L['F'] and len(losses(L['F'])) >= 4, 'F: after deleting the marker it installs and trains')
print('SUMMARY', 'PASS' if not fails else 'FAIL %d: %s' % (len(fails), fails))
sys.exit(1 if fails else 0)
