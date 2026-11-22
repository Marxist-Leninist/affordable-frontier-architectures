#!/usr/bin/env python3
"""Standard OM3 checks for one arm module arms/<arm>.py: parameters, parity, causality, a 12-step training smoke.
Usage: test_arm.py ARM   (prints one JSON line)"""
import importlib.util, json, resource, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import om2bench as ob  # noqa: E402

TARGET = 471264


def load(arm):
    spec = importlib.util.spec_from_file_location(f'arm_{arm}', HERE / 'arms' / f'{arm}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    arm = sys.argv[1]
    torch.set_num_threads(1)
    torch.manual_seed(5617)
    mod = load(arm)
    model = mod.build()
    n = ob.nparams(model)
    train, dev, test, _ = ob.load_all()
    px = torch.from_numpy(np.stack([test[0]['data'][:96], test[1]['data'][:96]]))
    par = ob.parity(model, px)
    with torch.no_grad():
        model.eval()
        a = model(px, hard=True)
        x2 = px.clone()
        x2[:, 60:] = torch.randint(0, 256, x2[:, 60:].shape)
        b = model(x2, hard=True)
    causal = float((a[:, :60] - b[:, :60]).abs().max())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(1)
    ts, losses = [], []
    for i in range(12):
        offs = rng.integers(0, len(train) - 257, size=8)
        xb = torch.from_numpy(np.stack([train[o:o + 257] for o in offs]).astype(np.int64))
        t0 = time.perf_counter()
        lg = model(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), xb[:, 1:].reshape(-1))
        pen = getattr(model, 'last_penalty', None)
        (loss + 0.02 * pen if pen is not None else loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ts.append(time.perf_counter() - t0)
        losses.append(float(loss))
    # parity and causality again with the smoke-trained weights: at init, arms with zero-init output projections
    # (rwkv7) output a constant per block, so the checks above never touch their token mixer
    par_t = ob.parity(model, px)
    with torch.no_grad():
        model.eval()
        causal_t = float((model(px, hard=True)[:, :60] - model(x2, hard=True)[:, :60]).abs().max())
    model.train()
    tm = ob.timing(model, test[0]['data'], 256, reps=2, gen=32)
    out = dict(arm=arm, params=n, params_vs_target=n / TARGET - 1, parity_max_abs=par, causal_max_abs=causal,
               parity_trained=par_t, causal_trained=causal_t, s_per_step=statistics.median(ts[2:]), loss_first=losses[0], loss_last=losses[-1],
               peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               decode_ms_median=tm['decode_ms_median'], prefill_tok_s=tm['prefill_tok_s'])
    out['all_requirements_met'] = bool(abs(out['params_vs_target']) <= 0.03 and max(par, par_t) < 1e-4 and max(causal, causal_t) < 1e-6 and
                                       losses[-1] < losses[0] and out['peak_rss_mib'] < 2048)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
