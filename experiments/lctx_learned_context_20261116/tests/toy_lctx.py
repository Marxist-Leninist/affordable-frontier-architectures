"""toy_lctx.py -- does the learned context length train, what does it decide, what does it cost or buy?

A small model with the 888B's attention (q = sqrt(hd) unit(q), k = unit(k), RoPE, logit T * cos with T = sqrt(hd),
causal SDPA, ReLU^2 MLP, RMS norms), byte-level text.  Like the 888B, it is first trained WITHOUT the mod, then
continued from that checkpoint three ways:
    base   no mod
    lctx   the mod (rho0 = log 1e-5, the hotmod's optimiser and knobs), span_cost 0: the loss alone decides
    lctx_c the mod with span_cost > 0 (a price on context, so heads that do not need it give it back)
Reported per arm: validation loss at the training context and at 4x it, the learned spans per head, and how much of
a pruned KV cache (theta = 40) each head keeps at 4x context, with the loss of the pruned run.

python3 -I toy_lctx.py SRC_DIR DATA_FILE OUT_DIR phase [arm]     phase: pre | cont
"""
import json
import math
import os
import sys
import time

SRC, DATA, OUT, PHASE = sys.argv[1:5]
ARM = sys.argv[5] if len(sys.argv) > 5 else 'base'
sys.path.insert(0, SRC)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lctx_core as LC  # noqa: E402

torch.set_num_threads(int(os.environ.get('TOY_THREADS', '2')))
CFG = dict(d=192, L=4, H=4, ff=768, ctx=256, batch=8, lr=2e-3, pre_steps=int(os.environ.get('TOY_PRE', '1500')),
           cont_steps=int(os.environ.get('TOY_CONT', '1500')), eval_every=250, eval_batches=12, seed=0)
HD = CFG['d'] // CFG['H']
TATT = math.sqrt(HD)
SCALE = TATT / math.sqrt(HD)


def rope(x, base=10000.0):
    T, hd = x.shape[-2], x.shape[-1]
    f = base ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
    ang = torch.arange(T, device=x.device).float()[:, None] * f[None]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, ff = c['d'], c['ff']
        self.H = c['H']
        self.n1, self.n2, self.nh, self.no = RMS(d), RMS(d), RMS(ff), RMS(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x, lc=None, keep_theta=None, stats=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        q, k = rope(F.normalize(q, dim=-1) * math.sqrt(D // self.H)), rope(F.normalize(k, dim=-1))
        if lc is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
        elif keep_theta is not None:  # evaluation with a pruned KV cache
            rho, w, c, amax = lc[:4]
            kb = LC.KeyBias.apply(v, rho, w, c, amax, None, None)
            a, keep = LC.pruned_attention(q, k, v, kb, SCALE, keep_theta)
            if stats is not None:  # per head: the KV cache a decoder still needs at the LAST position, as a share of T
                stats.append((keep[:, :, -1, :].sum(dim=(0, 2)).double() / (T * B)).tolist())
        else:
            rho, w, c, amax, sink = lc
            a = LC.lctx_attention(q, k, v, rho, w, c, SCALE, amax=amax, sink=sink)
        x = x + self.o(self.no(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.down(self.nh(F.relu(self.up(self.n2(x))).square()))


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.emb = nn.Embedding(256, c['d'])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c['L']))
        self.nf = RMS(c['d'])

    def forward(self, ids, lcs=None, keep_theta=None, stats=None):
        x = self.emb(ids)
        for i, b in enumerate(self.blocks):
            x = b(x, None if lcs is None else lcs[i], keep_theta, stats)
        return self.nf(x) @ self.emb.weight.t()


def batches(data, ctx, batch, gen):
    n = len(data) - ctx - 2
    while True:
        ix = torch.randint(0, n, (batch,), generator=gen)
        yield torch.stack([data[i:i + ctx + 1] for i in ix.tolist()])


def lcs_for(st, knobs, sinks=True):
    out = []
    for layer in range(st.L):
        rho, w, c = st.layer_params(layer, 'cpu')
        sink = (lambda dr, dw, dc, _l=layer: st.add(_l, dr, dw, dc)) if sinks else None
        out.append((rho, w, c, knobs['amax'], sink))
    return out


@torch.no_grad()
def evaluate(model, val, ctx, st, knobs, n, theta=None):
    gen = torch.Generator().manual_seed(4321 + ctx)
    it = batches(val, ctx, 4, gen)
    tot, cnt, stats = 0.0, 0, []
    for _ in range(n):
        x = next(it)
        lcs = None if st is None else lcs_for(st, knobs, sinks=False)
        logits = model(x[:, :-1], lcs, theta, stats if theta is not None else None)
        tot += float(F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1), reduction='sum'))
        cnt += x[:, 1:].numel()
    kept = None
    if stats:
        per = torch.tensor(stats, dtype=torch.float64).view(n, CFG['L'], CFG['H']).mean(0)
        kept = [[round(float(v), 4) for v in row] for row in per]
    return tot / cnt, kept


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = open(DATA, 'rb').read()
    data = torch.tensor(list(raw), dtype=torch.long)
    cut = int(0.9 * len(data))
    train, val = data[:cut], data[cut:]
    torch.manual_seed(CFG['seed'])
    model = Model(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], betas=(0.9, 0.95), weight_decay=0.0)
    pre = os.path.join(OUT, 'pre.pt')
    log = []
    st, knobs = None, None
    if PHASE == 'pre':
        steps, start = CFG['pre_steps'], 0
    else:
        sd = torch.load(pre, map_location='cpu')
        model.load_state_dict(sd['model'])
        opt.load_state_dict(sd['opt'])
        steps, start = CFG['cont_steps'], CFG['pre_steps']
        if ARM.startswith('lctx'):
            st = LC.LctxState(CFG['L'], CFG['H'], HD)
            knobs = LC.clean_knobs({'span_cost': float(os.environ.get('TOY_SPAN_COST', '0.02'))} if ARM == 'lctx_c'
                                   else {})
    total = CFG['pre_steps'] + CFG['cont_steps']
    gen = torch.Generator().manual_seed(CFG['seed'] + (0 if PHASE == 'pre' else 1))
    it = batches(train, CFG['ctx'], CFG['batch'], gen)
    t0 = time.time()
    for s in range(start, start + steps):
        lr = CFG['lr'] * min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total)))
        for g in opt.param_groups:
            g['lr'] = lr
        x = next(it)
        lcs = None if st is None else lcs_for(st, knobs)
        logits = model(x[:, :-1], lcs)
        loss = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st is not None:
            st.step(knobs)
        if (s + 1) % CFG['eval_every'] == 0 or s + 1 == start + steps:
            v1, _ = evaluate(model, val, CFG['ctx'], st, knobs, CFG['eval_batches'])
            rec = dict(step=s + 1, train=float(loss), val=v1, sec=round(time.time() - t0, 1))
            if st is not None:
                rec['lctx'] = st.summary()
            log.append(rec)
            print(json.dumps(rec), flush=True)
    res = dict(arm=ARM if PHASE == 'cont' else 'pre', cfg=CFG, log=log)
    if PHASE == 'pre':
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), pre)
    v1, _ = evaluate(model, val, CFG['ctx'], st, knobs, 40)
    v4, _ = evaluate(model, val, 4 * CFG['ctx'], st, knobs, 12)
    v8, _ = evaluate(model, val, 8 * CFG['ctx'], st, knobs, 6)
    res.update(val_ctx=v1, val_4x=v4, val_8x=v8)
    if PHASE == 'cont':
        torch.save(model.state_dict(), os.path.join(OUT, ARM + '.model.pt'))
    if st is not None:
        res['spans'] = [[round(s_, 1) for s_ in LC.spans(r)] for r in st.rho]
        res['w_norm'] = [[round(float(x), 3) for x in row] for row in st.w.norm(dim=-1)]
        v4p, kept = evaluate(model, val, 4 * CFG['ctx'], st, knobs, 12, theta=40.0)
        res.update(val_4x_pruned=v4p, kv_kept_4x=kept, pruned_minus_full=v4p - v4)
        sweep = []  # the memory / quality trade-off of evicting keys whose bias fell more than theta behind
        for th in (4.0, 8.0, 12.0, 16.0, 24.0, 40.0):
            for mult, n, full in ((4, 12, v4), (8, 6, v8)):
                vp, kp = evaluate(model, val, mult * CFG['ctx'], st, knobs, n, theta=th)
                mean_kept = sum(sum(r) for r in kp) / (len(kp) * len(kp[0]))
                sweep.append(dict(theta=th, ctx=mult * CFG['ctx'], loss=vp, loss_minus_full=vp - full,
                                  kv_kept_mean=round(mean_kept, 4)))
                print('SWEEP ' + json.dumps(sweep[-1]), flush=True)
        res['kv_sweep'] = sweep
        torch.save(st.state_dict(), os.path.join(OUT, ARM + '.lctx.pt'))
    with open(os.path.join(OUT, (ARM if PHASE == 'cont' else 'pre') + '.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('RESULT ' + json.dumps({k: res[k] for k in res if k not in ('cfg', 'log')}), flush=True)


if __name__ == '__main__':
    main()
