#!/usr/bin/env python3
"""train.py -- one training run of one variant. Same seed => same backbone start and same batches in every variant.

  python3 train.py --data data/tinystories --variant mc3_poe --seed 1 --out runs/x.json

Writes a JSON with the held-out curve (dev loss at fixed steps), the final test loss, and the measured cost per step
(process CPU seconds, so other jobs on the machine do not distort it; the run is single-threaded by default).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from markov import MarkovPrior
from model import GPT, lm_loss, variant_cfg


def lr_at(step, steps, lr, warmup, floor=0.1):
    if step <= warmup:
        return lr * step / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, prior, ids, T, n_windows, bs=32):
    """mean loss over the first n_windows non-overlapping windows of a held-out stream (full chain counts)"""
    model.eval()
    tot, n, acc = 0.0, 0, {}
    ar = torch.arange(T)
    for s in range(0, n_windows, bs):
        w = torch.arange(s, min(n_windows, s + bs))
        idx = w[:, None] * T + ar[None, :]
        loss, info = lm_loss(model, prior, ids[idx], ids[idx + 1], train=False)
        tot += float(loss) * idx.numel()
        n += idx.numel()
        for k, v in info.items():
            acc[k] = acc.get(k, 0.0) + v * idx.numel()
    model.train()
    return tot / n, {k: v / n for k, v in acc.items()}


@torch.no_grad()
def posthoc_chain(model, a, dev, test, ndev, ntest, bs=32):
    """The trained classical model with the chain added AFTERWARDS (no retraining), in the two forms the hybrids use:
    mixture  -log((1 - g) p_model + g p_chain)        (what chain_mix serves on the 888B)
    logits   -log softmax(logits + lam * log p_chain)  (the product of experts mc3_poe trains with)
    g and lam are chosen on the dev windows, then scored once on the test windows. Chain: order 4 (3 tokens of
    context), full counts of the train split, the same chain the mc3 variants use at test time."""
    import torch.nn.functional as F
    prior = MarkovPrior(np.load(os.path.join(a.data, 'markov.npz')), 4)
    gs = [round(0.05 * i, 2) for i in range(1, 20)]
    lams = [round(0.1 * i, 1) for i in range(0, 16)]
    ar = torch.arange(a.T)
    model.eval()

    def batches(ids, nwin):
        for s in range(0, nwin, bs):
            w = torch.arange(s, min(nwin, s + bs))
            idx = w[:, None] * a.T + ar[None, :]
            x, y = ids[idx], ids[idx + 1]
            logits, _ = model(x)
            lpc = prior.dense_logprob(x).float()
            yield logits.float(), lpc, y

    def score(ids, nwin, gs, lams):
        mix, poe, alone, chain, n = [0.0] * len(gs), [0.0] * len(lams), 0.0, 0.0, 0
        for logits, lpc, y in batches(ids, nwin):
            tm = F.log_softmax(logits, -1).gather(-1, y[..., None])[..., 0]
            tc = lpc.gather(-1, y[..., None])[..., 0]
            alone += float(-tm.sum())
            chain += float(-tc.sum())
            for i, g in enumerate(gs):
                mix[i] += float(-torch.logaddexp(math.log(1 - g) + tm, math.log(g) + tc).sum())
            for i, lam in enumerate(lams):
                poe[i] += float(-F.log_softmax(logits + lam * lpc, -1).gather(-1, y[..., None]).sum())
            n += y.numel()
        return [v / n for v in mix], [v / n for v in poe], alone / n, chain / n

    dmix, dpoe, dalone, dchain = score(dev, ndev, gs, lams)
    g = gs[int(np.argmin(dmix))]
    lam = lams[int(np.argmin(dpoe))]
    tmix, tpoe, talone, tchain = score(test, ntest, [g], [lam])
    model.train()
    return {'g_dev': g, 'lam_dev': lam, 'dev': {'model': dalone, 'chain': dchain, 'mix': min(dmix), 'logits': min(dpoe)},
            'test': {'model': talone, 'chain': tchain, 'mix': tmix[0], 'logits': tpoe[0]},
            'dev_mix_by_g': dict(zip(map(str, gs), dmix)), 'dev_logits_by_lam': dict(zip(map(str, lams), dpoe))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--dev_windows', type=int, default=300)
    ap.add_argument('--test_windows', type=int, default=4000)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--posthoc_chain', type=int, default=0,
                    help='classical model only: after training, also score it combined with the full-count order-3 chain '
                         '(mixture and logit sum), weights chosen on dev, scored on test: the chain added afterwards')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    # denormal floats make CPU arithmetic many times slower and carry no information; without this a run whose logits
    # spread widely is charged a cost that has nothing to do with its architecture
    ftz = torch.set_flush_denormal(True)

    meta = json.load(open(os.path.join(a.data, 'meta.json')))
    V = meta['V']
    tr = torch.from_numpy(np.load(os.path.join(a.data, 'train.npy')).astype(np.int64))
    dev = torch.from_numpy(np.load(os.path.join(a.data, 'dev.npy')).astype(np.int64))
    test = torch.from_numpy(np.load(os.path.join(a.data, 'test.npy')).astype(np.int64))
    ndev = min(a.dev_windows, (len(dev) - 1) // a.T)
    ntest = min(a.test_windows, (len(test) - 1) // a.T)

    cfg = variant_cfg(a.variant)
    prior = None
    if cfg['prior']:
        bt = os.path.join(a.data, 'blocks_cache') if cfg['cv'] == 'lobo' else None
        prior = MarkovPrior(np.load(os.path.join(a.data, 'markov.npz')), cfg['order'], btables=bt)   # counts from train only

    torch.manual_seed(a.seed)
    model = GPT(V, a.T, a.d, a.L, a.H, cfg)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith('gate')]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith('gate'))]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': rest, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(100003 * a.seed + 17)       # batch order depends on the seed only
    eval_steps = sorted(set([s for s in (25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800) if s < a.steps]
                            + [int(round(a.steps * f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]))
    res = {'name': os.path.splitext(os.path.basename(a.out))[0], 'args': vars(a), 'cfg': cfg, 'V': V,
           'params': sum(p.numel() for p in model.parameters()), 'params_extra': model.extra_params(),
           'dev_windows': ndev, 'test_windows': ntest, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
           'flush_denormal': bool(ftz),
           'load_start': os.getloadavg()[0], 'started': time.time(), 'evals': [], 'status': 'running'}

    def save():
        tmp = a.out + '.tmp'
        json.dump(res, open(tmp, 'w'))
        os.replace(tmp, a.out)

    timer, cpu, wall, ema = {}, 0.0, 0.0, None
    ar = torch.arange(a.T)
    for step in range(1, a.steps + 1):
        c0, w0 = time.process_time(), time.perf_counter()
        ix = torch.randint(0, len(tr) - a.T - 1, (a.bs,), generator=g)
        idx = ix[:, None] + ar[None, :]
        x, y = tr[idx], tr[idx + 1]
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step, a.steps, a.lr, a.warmup)
        loss, info = lm_loss(model, prior, x, y, train=True, timer=timer, tpos=idx + 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        cpu += time.process_time() - c0
        wall += time.perf_counter() - w0
        lv = loss.item()
        if not math.isfinite(lv):
            res['status'] = f'diverged at step {step}'
            save()
            raise SystemExit(3)
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step in eval_steps:
            dv, einfo = evaluate(model, prior, dev, a.T, ndev)
            res['evals'].append({'step': step, 'dev': dv, 'train_ema': ema, 'cpu_s': cpu, 'wall_s': wall,
                                 'prior_cpu_s': timer.get('prior_cpu', 0.0), **einfo})
            save()
            print(f'step {step:6d} dev {dv:.4f} train_ema {ema:.4f} cpu {cpu:.0f}s wall {wall:.0f}s {einfo}', flush=True)
        if step % 20 == 0 and os.path.exists('STOP'):
            res['status'] = f'stopped at step {step}'
            save()
            raise SystemExit(4)

    res['test_loss'], res['test_info'] = evaluate(model, prior, test, a.T, ntest)
    res['dev_loss_final'] = res['evals'][-1]['dev']
    if prior is not None:
        tot = 0.0
        for s in range(0, ntest, 64):
            w = torch.arange(s, min(ntest, s + 64))
            idx = w[:, None] * a.T + ar[None, :]
            tot += float(-prior.target_logprob(test[idx], test[idx + 1]).sum())
        res['chain_alone_test_loss'] = tot / (ntest * a.T)
    if cfg['mk']:
        res['mk_b_per_layer'] = [torch.sigmoid(b.mk_beta.detach()).tolist() for b in model.blocks]
    if a.posthoc_chain and cfg['prior'] is None:
        res['posthoc_chain'] = posthoc_chain(model, a, dev, test, ndev, ntest)
    res.update({'cpu_ms_per_step': 1000 * cpu / a.steps, 'wall_ms_per_step': 1000 * wall / a.steps,
                'prior_cpu_ms_per_step': 1000 * timer.get('prior_cpu', 0.0) / a.steps,
                'load_end': os.getloadavg()[0], 'finished': time.time(), 'status': 'done'})
    save()
    print(f"DONE test {res['test_loss']:.4f} cpu_ms/step {res['cpu_ms_per_step']:.1f}", flush=True)


if __name__ == '__main__':
    main()
