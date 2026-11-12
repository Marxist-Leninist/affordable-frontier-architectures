"""test_alignment.py SIM.py PLUGIN.py DATA_DIR -- the plugin's slot kinds against the simulator's OWN prepare() (not my
re-implementation of its masks), and its counting stream against the simulator's own training windows.
A few seconds of numpy / torch on CPU; not a simulator run."""
import importlib.util
import os
import sys

import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sim_path, plug_path, data = sys.argv[1:4]
os.environ['AGILLM_SIM_DATA'] = data
sim = load(sim_path, 'sim')
plug = load(plug_path, 'plug')
V, T, B, MASK = sim.SIM['V'], sim.SIM['ctx'], sim.SIM['batch'], sim.SIM['mask']
train = np.load(data + '/train.npy', mmap_mode='r')
STEPS = 5
ch = plug.Chain(train, V, T, B, MASK, steps=STEPS)
ok = True


def check(cond, what):
    global ok
    print(('PASS ' if cond else 'FAIL ') + what)
    ok = ok and bool(cond)


# 1. the counting stream never contains a training window of those steps (the simulator's own window draw)
keep = plug.Chain._keep(len(train), T, B, STEPS, ch.info['margin'])
rng = np.random.default_rng(1000)
bad = 0
for step in range(STEPS):
    starts = rng.integers(0, len(train) - T - 2, B)                 # the line in agillm_mini.main()
    for s in starts:
        bad += int(keep[int(s):int(s) + T + 1].sum())
        m = ch.info['margin']
        bad += int(keep[max(0, int(s) - m):int(s) + T + 1 + m].sum())
check(bad == 0, 'no counted token lies in a training window of the run or within the margin around it (%d steps checked)' % STEPS)
check(abs(keep.mean() - ch.info['kept_fraction']) < 1e-12, 'the chain reports the fraction it really counted (%.3f)' % keep.mean())

# 2. every row the plugin adds points at the right context token and the right distance, for the simulator's real masks
g = np.random.default_rng(5)
x = torch.from_numpy(np.stack([np.asarray(train[int(s):int(s) + T + 1]) for s in g.integers(0, len(train) - T - 2, B)]).astype(np.int64))
xn = x.numpy()
for mode in ('ar', 'sat_fixed', 'sat_variable', 'nat'):
    inputs, targets, sel, causal = sim.prepare(x, mode, np.random.default_rng(11), MASK)
    inp = inputs.numpy()
    (slot, ctx, gap, kind), (ss, sv, sl) = ch.lifts(inp, causal, T)
    b, p = slot // T, slot % T
    if causal:
        tpos = p + 1                                                 # the target of slot p is x[p+1]
        want_slots = B * T - int(((inp == MASK) & (np.arange(T)[None, :] - np.maximum.accumulate(np.where(inp != MASK, np.arange(T)[None, :], -1), axis=1) > 3)).sum())
    else:
        tpos = p                                                     # NAT: the masked token itself
        want_slots = None
    q = tpos - gap                                                   # where the context token sits
    good = (q >= 0).all() and (inp[b, q] != MASK).all() and (ctx == xn[b, q]).all()
    between_masked = all((inp[bb, qq + 1:pp + 1] == MASK).all() for bb, qq, pp in zip(b[inp[b, p] == MASK], q[inp[b, p] == MASK], p[inp[b, p] == MASK]))
    one_each = len(np.unique(slot)) == len(slot)
    if causal:
        kinds_ok = ((kind == 0) == (inp[b, p] != MASK)).all() and ((kind >= 1) & (kind <= 3) == (inp[b, p] == MASK)).all() and (gap[kind == 0] == 1).all() and (gap[kind > 0] == kind[kind > 0] + 1).all()
        cover = len(slot) == want_slots
    else:
        kinds_ok = (kind == 3 + gap).all() and (inp[b, p] == MASK).all()
        prefix_end = np.maximum.accumulate(np.where(inp != MASK, np.arange(T)[None, :], -1), axis=1)[:, -1]
        cover = len(slot) == int(sum(min(4, T - 1 - pe) for pe in prefix_end))
    tri_ok = True
    if len(ss):
        tb, tp = ss // T, ss % T
        tri_ok = causal and (inp[tb, tp] != MASK).all() and (tp >= 1).all() and (inp[tb, tp - 1] != MASK).all() and (sl > 0).all() and len(set(zip(ss.tolist(), sv.tolist()))) == len(ss)
    check(good and between_masked and one_each and kinds_ok and cover and tri_ok,
          '%s: %d rows, kinds %s; context token, distance to the target, kind and coverage all agree with the simulator\'s masks; %d trigram entries'
          % (mode, len(slot), np.bincount(kind, minlength=plug.NKIND).tolist(), len(ss)))

# 3. the in-place add inside autograd: gradient reaches lam and the logits' own gradient is untouched by the constant rows
lam = torch.ones(plug.NKIND, requires_grad=True)
base = torch.randn(B * T, V, requires_grad=True)
z = base * 1.0                                                       # a non-leaf, like the table readout
inputs, targets, sel, causal = sim.prepare(x, 'sat_variable', np.random.default_rng(3), MASK)
out = plug.apply_chain(ch, z, lam, inputs.numpy(), causal, T)
loss = torch.nn.functional.cross_entropy(out, targets.reshape(-1))
loss.backward()
ref = torch.softmax(out.detach(), -1)
ref[torch.arange(B * T), targets.reshape(-1)] -= 1
check(torch.allclose(base.grad, ref / (B * T), atol=1e-6) and bool(torch.isfinite(lam.grad).all()) and float(lam.grad.abs().sum()) > 0,
      'autograd through the in-place add: d loss / d logits = softmax - onehot, and lam receives a finite non-zero gradient %s' % [round(float(v), 4) for v in lam.grad])
print('ALIGN-OK' if ok else 'ALIGN-FAILED')
