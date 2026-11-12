"""test_equiv.py SIM.py PLUGIN.py DATA_DIR -- the single-node chain add (_ChainAdd) against the first version, which
added in place through ordinary autograd ops. Same forward values, same gradient to the logits, same gradient to the
10 weights, on the simulator's own batch layouts (prepare() of agillm_mini.py), at ramp 1 and 0.5, with and without
the n-gram rows. Also what each keeps for backward (saved-tensor hooks) and whether the graph has CopySlices nodes,
the copies of the whole logits gradient that went over the GPU cap. CPU only, a few seconds."""
import importlib.util
import os
import sys

import numpy as np
import torch

QS_REF = None


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ref_apply(ch, flat, lam, ids, causal, Th, lifts=True, unigram=True, ramp=1.0):
    """the first version, verbatim apart from names (it ran on 12 Nov and went over the cap in backward)"""
    dev = flat.device
    if unigram:
        flat += (ramp * lam[9]).to(flat.dtype) * ch.logp1_on(dev, flat.dtype)
    if not lifts or not ch.Lq:
        return flat
    (slot, ctx, gap, kind), (ss, sv, sl) = ch.lifts(ids, causal, Th)
    if len(slot):
        w = lam[torch.from_numpy(kind).to(dev)] * (ramp / QS_REF)
        st = torch.from_numpy(slot).to(dev)
        for a in range(0, len(slot), 512):
            R = torch.from_numpy(ch.byte_rows(ctx[a:a + 512], gap[a:a + 512])).to(dev)
            flat.index_add_(0, st[a:a + 512], (w[a:a + 512, None] * R.to(torch.float32)).to(flat.dtype))
    if len(ss):
        flat.index_put_((torch.from_numpy(ss).to(dev), torch.from_numpy(sv).to(dev)),
                        ((ramp * lam[8]) * torch.from_numpy(sl).to(dev)).to(flat.dtype), accumulate=True)
    return flat


def graph_nodes(t):
    seen, out, stack = set(), [], [t.grad_fn]
    while stack:
        f = stack.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        out.append(type(f).__name__)
        stack.extend(n for n, _ in f.next_functions)
    return out


sim_path, plug_path, data = sys.argv[1:4]
os.environ['AGILLM_SIM_DATA'] = data
sim = load(sim_path, 'sim')
plug = load(plug_path, 'plug')
QS_REF = plug.QS
V, T, B, MASK = sim.SIM['V'], sim.SIM['ctx'], sim.SIM['batch'], sim.SIM['mask']
train = np.load(data + '/train.npy', mmap_mode='r')
ch = plug.Chain(train, V, T, B, MASK, steps=5)
g = np.random.default_rng(5)
x = torch.from_numpy(np.stack([np.asarray(train[int(s):int(s) + T + 1]) for s in g.integers(0, len(train) - T - 2, B)]).astype(np.int64))
ok = True
torch.manual_seed(0)
lam0 = torch.rand(plug.NKIND) * 1.5 + 0.25                  # not all ones, so a mixed-up weight index would show
for mode in ('ar', 'sat_fixed', 'sat_variable', 'nat', 'local'):
    for ramp, lifts in ((1.0, True), (0.5, True), (1.0, False)):
        if mode == 'local':
            inputs, causal = x[:, :-1].clone(), False
            inputs[torch.rand(inputs.shape) < 0.3] = MASK
            lifts_here = False
        else:
            inputs, _, _, causal = sim.prepare(x, mode, np.random.default_rng(11), MASK)
            lifts_here = lifts
        ids = inputs.numpy()
        G = torch.randn(B, T, V)                              # loss = sum(G * out): checks every gradient entry
        base0 = torch.randn(B, T, V) * 2
        res = []
        for impl in ('ref', 'new'):
            base = base0.clone().requires_grad_(True)
            lam = lam0.clone().requires_grad_(True)
            saved = []

            def pack(t):
                saved.append(t.numel() * t.element_size())
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                z = base * 1.0                                  # a non-leaf, like the readout
                if impl == 'ref':
                    ref_apply(ch, z.view(-1, V), lam, ids, causal, T, lifts=lifts_here, unigram=True, ramp=ramp)
                    out = z
                else:
                    out = plug.apply_chain(ch, z, lam, ids, causal, T, lifts=lifts_here, unigram=True, ramp=ramp)
            (out * G).sum().backward()
            res.append((out.detach().clone(), base.grad.clone(), lam.grad.clone(), sum(saved), graph_nodes(out)))
        (o1, gb1, gl1, s1, n1), (o2, gb2, gl2, s2, n2) = res
        d_out = float((o1 - o2).abs().max())
        d_gb = float((gb1 - gb2).abs().max())
        d_gl = float(((gl1 - gl2).abs() / (gl1.abs() + 1e-3)).max())
        good = d_out < 1e-5 and d_gb < 1e-6 and d_gl < 1e-4 and n2.count('CopySlices') == 0
        ok = ok and good
        print('%s %-12s ramp %.1f rows %-5s | max diff: output %.1e, d/d logits %.1e, d/d lam (relative) %.1e | saved for backward: first version %.1f MiB, now %.2f MiB | CopySlices nodes: first %d, now %d'
              % ('PASS' if good else 'FAIL', mode, ramp, str(lifts_here), d_out, d_gb, d_gl, s1 / 2 ** 20, s2 / 2 ** 20, n1.count('CopySlices'), n2.count('CopySlices')))
print('EQUIV-OK' if ok else 'EQUIV-FAILED')
