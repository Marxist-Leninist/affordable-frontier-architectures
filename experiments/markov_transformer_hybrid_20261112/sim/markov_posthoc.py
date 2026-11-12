"""markov_posthoc -- the plain simulator model with the counted Markov chain mixed in AFTERWARDS (no training with it).

Training is exactly `base`: this plugin adds no parameters and does not touch the readout, so the run's own metrics are
a base run's. After the last training step (in post_step, before the harness's final evaluation, so on the same model)
it scores, on the harness's own 64 evaluation windows and with the harness's own deterministic layouts (AR, SAT fixed,
NAT), the mixture  -log((1 - g) p_model + g p_chain)  for a grid of g. p_chain is the chain markov_corpus_prior trains
with (same counts: train-stream text no training window touches), evaluated as softmax(log P1 + lifts) with every
weight at 1. The weight is chosen on one half of the windows and scored on the other half, both ways (2-fold), so no
number is chosen on the windows it is reported on. The results go to the log (lines "markov_posthoc ...") and to
/workspace/fable52_markov_20261112/sim/posthoc_s<seed>.json.

On the small corpora (section 1 of the write-up) this "added afterwards" baseline beat every hybrid trained with the
chain; this plugin asks the same question under AGILLM's objectives.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markov_corpus_prior.py')
_s = importlib.util.spec_from_file_location('markov_corpus_prior_core', _p)
core = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = core
_s.loader.exec_module(core)

META = dict(id='markov_posthoc', title='Plain model with the counted Markov chain mixed in afterwards (scored at the end, not trained with)',
            slot='hybrid', one_line='Trains exactly like base; after the last step scores (1-g) p_model + g p_chain on the evaluation windows, '
                                    'g chosen on the other half of the windows. No parameters, no training change.',
            storage='the same count tables as markov_corpus_prior, on the host')
GRID = [round(0.05 * i, 2) for i in range(0, 21)]
OUT_DIR = os.environ.get('MARKOV_POSTHOC_DIR', '/workspace/fable52_markov_20261112/sim')


def _arg(name, default):
    a = sys.argv
    for i, t in enumerate(a[:-1]):
        if t == name:
            return a[i + 1]
    return default


@torch.no_grad()
def _score(model, c):
    import __main__ as sim                      # the harness: its prepare() and SIM
    T, V, mask = c['ctx'], c['V'], c['mask']
    test = np.load(core.DATA + '/test.npy', mmap_mode='r')
    smoke = os.environ.get('AGILLM_SIM_SMOKE') == '1'
    pos = np.linspace(0, len(test) - T - 2, 8 if smoke else 64).astype(np.int64)
    W = torch.from_numpy(np.stack([np.asarray(test[int(s):int(s) + T + 1]) for s in pos]).astype(np.int64))
    ch = core.Chain(np.load(core.DATA + '/train.npy', mmap_mode='r'), V, T, c['batch'], mask, steps=core._run_steps())
    dev = next(model.parameters()).device
    lam = torch.ones(core.NKIND)
    out = {}
    was_training = model.training
    model.eval()                                # as the harness's own evaluate() does
    for mode in ('ar', 'sat_fixed', 'nat'):
        tm_all, tc_all = [], []
        for a in range(0, len(W), 4):           # 4 windows at a time: 1,024 rows x 8,192
            x = W[a:a + 4].to(dev)
            inputs, targets, sel, causal = sim.prepare(x, mode, np.random.default_rng(7), mask, deterministic=True)
            logits, _ = model.logits(inputs, causal=causal)
            rows = logits.reshape(-1, V).float()
            cl = core.apply_chain(ch, torch.zeros(rows.shape, dtype=torch.float32), lam, inputs.cpu().numpy(), causal, T)
            if sel is not None:
                keep = sel.reshape(-1)
                rows, cl = rows[keep], cl[keep.cpu()]
            tg = targets.reshape(-1)
            tm = torch.log_softmax(rows, -1).gather(1, tg[:, None])[:, 0].cpu().double()
            tc = torch.log_softmax(cl, -1).gather(1, tg.cpu()[:, None])[:, 0].double()
            per_window = int(len(tg) // len(x))
            tm_all.append(tm.view(len(x), per_window))
            tc_all.append(tc.view(len(x), per_window))
        tm, tc = torch.cat(tm_all), torch.cat(tc_all)                 # [windows, tokens]

        def mix(g, rows_sel):
            if g <= 0:
                return float(-tm[rows_sel].mean())
            if g >= 1:
                return float(-tc[rows_sel].mean())
            return float(-torch.logaddexp(np.log(1 - g) + tm[rows_sel], np.log(g) + tc[rows_sel]).mean())
        n = len(tm)
        halves = (torch.arange(n) < n // 2, torch.arange(n) >= n // 2)
        held, picks = [], []
        for pick_on, report_on in ((halves[0], halves[1]), (halves[1], halves[0])):
            g = min(GRID, key=lambda gg: mix(gg, pick_on))
            picks.append(g)
            held.append(mix(g, report_on))
        out[mode] = dict(model=mix(0.0, slice(None)), chain=mix(1.0, slice(None)), mixture_heldout=float(np.mean(held)), g_picked=picks,
                         by_g={str(g): mix(g, slice(None)) for g in GRID})
        print('markov_posthoc %s: model %.4f | chain %.4f | mixture added afterwards (g picked on the other half: %s) %.4f' % (
            mode, out[mode]['model'], out[mode]['chain'], picks, out[mode]['mixture_heldout']), flush=True)
    if was_training:
        model.train()
    return out, ch.info


_GUARDED = []


def patch(Block, Model, cfg):
    """no change to the model (returns None); only the attempt guard of markov_corpus_prior, extended to this plugin's
    own queue lines, so a failing run can never be retried for ever by the queue's runner"""
    if not _GUARDED:
        _GUARDED.append(True)
        core.MY_PLUGINS = tuple(core.MY_PLUGINS) + ('markov_posthoc',)
        core._attempt_guard()
    return None


def post_step(model, step, c):
    steps = int(_arg('--steps', '1200'))
    if step + 1 != steps:
        return
    t0 = time.time()
    res, info = _score(model, c)
    seed = _arg('--seed', '0')
    rec = dict(seed=int(seed), steps=steps, scored=res, chain=info, seconds=round(time.time() - t0, 1))
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, 'posthoc_s%s.json.tmp' % seed), 'w') as f:
            json.dump(rec, f)
        os.replace(os.path.join(OUT_DIR, 'posthoc_s%s.json.tmp' % seed), os.path.join(OUT_DIR, 'posthoc_s%s.json' % seed))
    except OSError as e:
        print('markov_posthoc: could not write the result file: %r' % (e,), flush=True)
    print('markov_posthoc done in %.0f s' % rec['seconds'], flush=True)
