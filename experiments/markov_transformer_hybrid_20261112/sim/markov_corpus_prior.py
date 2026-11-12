"""markov_corpus_prior -- a counted Markov chain beside the transformer, in the AGILLM objective simulator.

The network is one model, a table of counts is another, and the counts cost no training steps. The readout becomes

    logits[v] = network[v] + lam_u * log P1(v) + lam[kind] * lift(v | context)
    lift(v | context) = log( P_chain(v | context) / (back-off weight * lower-order P(v)) )

P_chain is interpolated absolute discounting over counted n-grams. With every lam = 1 and a silent network the logits
are exactly the chain's distribution (the common factor cancels in the softmax); the lift is zero for tokens the
context says nothing about. The network then only has to learn what the counts do not know. Every lam is one trained
scalar (10 fp parameters, all 1.0 at the start).

  9    every readout, all objectives: lam_u * log P1(v), the unigram. The tied binary table has a per-token scale but no
       additive per-token term, so the network cannot simply store token frequency; the counts can.
  0    causal slot with a visible input (AR, and the first slot of a SAT group): next token given the last visible token
       (full bigram row), plus 8: a sparse trigram correction when the token before it is visible too
  1-3  causal slot with a masked input (SAT, 1-3 slots after the group's first): the target is 2-4 tokens after the
       last visible token -> skip-gram rows for gaps 2, 3, 4
  4-7  NAT masked slot 1-4 tokens after the end of the prefix -> the rows for gaps 1-4, with their own lam
  local denoise, and NAT slots further than 4 from the prefix: the unigram only.

WHERE THE COUNTS COME FROM, and why this is not leakage: from the simulator's own train stream, but ONLY from tokens
that no training step of the run ever sees. The training windows are known in advance (the simulator draws them from
default_rng(1000) for every candidate), so every one of the 1,200 steps' windows is cut out of the counting stream
together with 512 tokens on each side, and no counted n-gram may straddle a cut. 56.9 % of the 22M-token stream
(12.56M tokens) is left. The eval windows are from the held-out test documents, which are never counted. So on
training text and on eval text alike the chain has not seen the window it is scoring. (The cut-out follows --steps;
a run so long that under 30 % of the stream is left to count stops with an error instead of counting training text.)
What the comparison with `base` then says: same steps, same training windows, same network; this candidate also
reads counts of text that neither model trains on. Counting 12.5M tokens takes seconds; training on them would take
five times the run. `unigram_bias` (same file, lifts off) is the control for the unigram part alone, and
`markov_corpus_late` switches the lifts on at step 600 of an ordinary base run, which is what adding the chain to a
model that is already trained would be.

Cost: the bigram and skip tables are full rows (V x V) stored as one byte per entry (the lift in twelfths of a nat),
67 MB each on the host; a step gathers one row per slot (16 MB), moves it to the GPU in pieces of 4 MB and adds it in
place. The trigram part is sparse: at most 64 tokens per context, the rest pruned with their mass given back to the
back-off weight, as in any pruned n-gram model. No latents, no KV, nothing streamed. At the real size the same thing
is a table of counts over the o200k vocabulary (1.1 GB for 116M counted tokens, memory-mapped, sparse rows; it
already serves beside the 888B as chainmix512) and one lookup per token, against a 350 s training step.
Causal by construction: a slot's lift reads only input tokens at or left of it.
The ctx-gain evaluation goes through readout() too, so its numbers include the chain.
"""
import os

import numpy as np
import torch
import torch.nn as nn

META = dict(id='markov_corpus_prior', title='Counted Markov chain beside the transformer: n-gram log-probabilities added to the logits', slot='hybrid',
            one_line='Every readout adds lam * log P1 (unigram); every causal slot and the NAT slots just after the prefix add '
                     'lam * log(P_chain / back-off) from n-gram and skip-gram counts taken on text the run never trains on; '
                     '10 fp parameters, no extra latents.',
            storage='count tables on the host, not in the model: 4 x 67 MB of byte rows + ~0.1 GB trigram here; ~1.1 GB memory-mapped '
                    'sparse rows at o200k scale for 116M counted tokens')

QUEUE = os.environ.get('MARKOV_QUEUE_FILE', '/workspace/asi_agillm/queue.txt')
ATTEMPTS = os.environ.get('MARKOV_ATTEMPTS_DIR', '/workspace/fable52_markov_20261112/sim/attempts')
MY_PLUGINS = ('markov_corpus_prior', 'markov_corpus_late', 'unigram_bias')


def _attempt_guard():
    """The queue's runner retries a failing run for ever (its failure count never passes 1, reported in #training 6540).
    So these plugins count their own attempts per result file; on the third attempt of the same run they take their own
    queue lines for that seed out of queue.txt (new file + rename) and stop, instead of failing again and again.
    Only inside the queue on the box (the result path is under /workspace/asi_agillm/runs/)."""
    import sys
    a = sys.argv
    out = next((a[i + 1] for i, t in enumerate(a[:-1]) if t == '--out'), None)
    seed = next((a[i + 1] for i, t in enumerate(a[:-1]) if t == '--seed'), None)
    if not out or seed is None or not (out.startswith('/workspace/asi_agillm/runs/') or os.environ.get('MARKOV_GUARD_TEST') == '1'):
        return
    os.makedirs(ATTEMPTS, exist_ok=True)
    f = os.path.join(ATTEMPTS, os.path.basename(out) + '.count')
    try:
        with open(f) as fh:
            n = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        n = 0
    n += 1
    with open(f + '.tmp', 'w') as fh:
        fh.write(str(n))
    os.replace(f + '.tmp', f)
    if n <= 2:
        return
    lines = open(QUEUE).read().splitlines()
    keep = [ln for ln in lines if not (len(ln.split()) >= 2 and ln.split()[0] in MY_PLUGINS and ln.split()[1] == str(seed))]
    with open(QUEUE + '.tmp-fable52markov', 'w') as fh:
        fh.write('\n'.join(keep) + '\n')
    os.replace(QUEUE + '.tmp-fable52markov', QUEUE)
    raise RuntimeError('markov_corpus_prior attempt guard: attempt %d of %s; removed %d of my queue lines for seed %s so the runner '
                       'does not retry it for ever' % (n, os.path.basename(out), len(lines) - len(keep), seed))


K3 = 64                    # trigram candidates kept per context; the rest are pruned into the back-off weight
MARGIN, EXCL_STEPS = 512, 1200
DELTA = 0.5
QS = 12.0                  # byte rows hold round(12 * lift): lifts up to 21.2 nats in steps of 1/12
NKIND = 10                 # 0 causal next (bigram row) | 1-3 causal gap 2-4 | 4-7 NAT gap 1-4 | 8 trigram correction | 9 unigram
DATA = os.environ.get('AGILLM_SIM_DATA', '/workspace/asi_agillm/data')


def _run_steps():
    """how many steps' training windows to cut out of the counting stream: those of this run (--steps), and never fewer
    than the standard 1,200, so a short probe counts exactly what a full run counts. The 5-step smoke test on the
    1.5M-token sample cuts out its own 5 steps only (1,200 steps' windows would cover that whole sample)."""
    import sys
    a, steps = sys.argv, EXCL_STEPS
    for i, t in enumerate(a):
        if t == '--steps' and i + 1 < len(a):
            steps = int(a[i + 1])
        elif t.startswith('--steps='):
            steps = int(t.split('=', 1)[1])
    if os.environ.get('AGILLM_SIM_SMOKE') == '1':
        return steps
    return max(steps, EXCL_STEPS)


def _disc(cnt):
    n1 = float(np.count_nonzero(cnt == 1))
    n2 = float(np.count_nonzero(cnt == 2))
    return min(0.95, max(0.05, n1 / (n1 + 2.0 * n2))) if n1 > 0 and n2 > 0 else 0.75


class Chain:
    """numpy count tables + the per-batch lift rows"""

    def __init__(self, train, V, T, B, mask_id, steps=EXCL_STEPS, margin=MARGIN, tables=True):
        try:                                        # if the box runs short of host memory this process goes first, not the site's worker
            with open('/proc/self/oom_score_adj', 'w') as f:
                f.write('800')
        except OSError:
            pass
        N = len(train)
        self.V, self.mask = V, mask_id
        keep = self._keep(N, T, B, steps, margin)
        if keep.mean() < 0.3 and margin:
            margin = 0
            keep = self._keep(N, T, B, steps, margin)
        if keep.mean() < 0.3:
            raise RuntimeError('markov_corpus_prior: under 30 % of the train stream is left to count after the training windows are cut out')
        self.info = dict(stream_tokens=int(N), counted_tokens=int(keep.sum()), kept_fraction=float(keep.mean()), margin=int(margin), excluded_steps=int(steps))
        x = np.asarray(train).astype(np.int32)      # int32 here, int64 only for the keys: the box is short of host memory
        c1 = np.bincount(x[keep], minlength=V).astype(np.float64)
        self.p1 = (c1 + DELTA) / (c1.sum() + DELTA * V)
        self.logp1 = np.log(self.p1).astype(np.float32)
        self._logp1_t = {}
        self.Lq, self.gam = {}, {}
        if not tables:                              # the unigram-only control needs nothing else
            return
        cs = np.concatenate([[0], np.cumsum(keep, dtype=np.int32)])

        def whole(i0, i1, length):                  # mask over starts i0 .. i1-1: all positions start .. start+length-1 are counted text
            return cs[i0 + length:i1 + length] - cs[i0:i1] == length
        disc, types = [], []
        for g in (1, 2, 3, 4):
            ok = whole(0, N - g, g + 1)
            uk, cnt = np.unique(x[:N - g][ok].astype(np.int64) * V + x[g:][ok], return_counts=True)
            ctx, nxt = uk // V, uk % V
            tot = np.bincount(ctx, weights=cnt, minlength=V)
            n1 = np.bincount(ctx, minlength=V).astype(np.float64)
            D = _disc(cnt)
            gam = np.where(tot > 0, D * n1 / np.maximum(tot, 1), 1.0)
            s = (cnt - D) / tot[ctx]
            Lq = np.zeros((V, V), dtype=np.uint8)
            Lq[ctx, nxt] = np.clip(np.rint(QS * np.log1p(s / (gam[ctx] * self.p1[nxt]))), 0, 255).astype(np.uint8)
            self.Lq[g], self.gam[g] = Lq, gam
            disc.append(D)
            types.append(int(len(uk)))
            if g == 1:
                self.keys1, self.s1 = uk, s.astype(np.float32)   # every bigram, for the trigram candidates' bigram probability
        ok = whole(0, N - 2, 3)
        full = (x[:N - 2][ok].astype(np.int64) * V + x[1:N - 1][ok]) * V + x[2:][ok]
        uf, cnt = np.unique(full, return_counts=True)
        del full, ok, x, cs
        self.uc3, start, n1 = np.unique(uf // V, return_index=True, return_counts=True)
        tot = np.add.reduceat(cnt, start).astype(np.float64)
        D3 = _disc(cnt)
        row = np.repeat(np.arange(len(self.uc3)), n1)
        s3 = (cnt - D3) / tot[row]
        order = np.lexsort((-s3, row))
        rank = np.arange(len(row)) - start[row[order]]
        sel = rank < K3
        self.t_nxt = (uf % V)[order][sel].astype(np.int32)
        kept = s3[order][sel]
        self.t_s = kept.astype(np.float32)
        self.tptr = np.concatenate([[0], np.cumsum(np.minimum(n1, K3))])
        # pruning done the n-gram way: what the dropped candidates held goes to the back-off weight, so a context with
        # more than K3 continuations still sums to one. For an unpruned context this is exactly D3 * types / total.
        self.gam3 = 1.0 - np.add.reduceat(kept, self.tptr[:-1])
        n_tri, pruned = int(len(uf)), int(len(uf) - len(kept))
        del uf, cnt, row, s3, order, rank, sel, kept
        self.info.update(bigram_types=types[0], skip_types=types[1:], trigram_types=n_tri, trigram_pruned=pruned, trigram_contexts=int(len(self.uc3)),
                         discounts=[round(v, 4) for v in disc + [D3]],
                         table_mb=round(4 * V * V / 1e6 + (self.t_nxt.nbytes + self.t_s.nbytes + self.uc3.nbytes) / 1e6))

    @staticmethod
    def _keep(N, T, B, steps, margin):
        keep = np.ones(N, dtype=bool)
        rng = np.random.default_rng(1000)           # the simulator's data_rng: identical windows for every candidate
        for _ in range(steps):
            for s in rng.integers(0, N - T - 2, B):
                keep[max(0, int(s) - margin):int(s) + T + 1 + margin] = False
        return keep

    def logp1_on(self, dev, dtype):
        key = (str(dev), dtype)
        if key not in self._logp1_t:
            self._logp1_t[key] = torch.from_numpy(self.logp1).to(device=dev, dtype=dtype)
        return self._logp1_t[key]

    def lifts(self, ids, causal, Th):
        """ids [B,T] int64 (inputs, MASK where masked). Slots are the last Th positions.
        -> rows (slot [n], context token [n], gap [n], kind [n]) for the full byte rows, and
           sparse (slot [m], token [m], lift [m]) for the trigram correction. slot = b * Th + local position."""
        B, T = ids.shape
        V = self.V
        vis = ids != self.mask
        pos = np.arange(T)[None, :]
        last = np.maximum.accumulate(np.where(vis, pos, -1), axis=1)      # last visible position <= p, -1 if none
        slot = np.zeros((B, T), dtype=bool)
        slot[:, T - Th:] = True
        rs, rc, rg, rk = [], [], [], []
        sp = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))

        def add(bb, pp, ctx, g, kind):
            rs.append(bb * Th + pp - (T - Th))
            rc.append(ctx)
            rg.append(np.full(len(bb), g, dtype=np.int64))
            rk.append(np.full(len(bb), kind, dtype=np.int64))
        if causal:
            bb, pp = np.nonzero(vis & slot)                               # visible input: predict the next token
            if len(bb):
                c1 = ids[bb, pp]
                add(bb, pp, c1, 1, 0)
                prev_ok = (pp >= 1) & vis[bb, np.maximum(pp - 1, 0)]
                ck = ids[bb, np.maximum(pp - 1, 0)] * V + c1
                r = np.minimum(np.searchsorted(self.uc3, ck), len(self.uc3) - 1)
                j = np.nonzero(prev_ok & (self.uc3[r] == ck))[0]
                if len(j):
                    rj = r[j]
                    e = self.tptr[rj][:, None] + np.arange(K3)[None, :]
                    val3 = e < self.tptr[rj + 1][:, None]
                    e = np.where(val3, e, 0)
                    cid3, s3 = self.t_nxt[e].astype(np.int64), self.t_s[e].astype(np.float64)
                    k2 = c1[j][:, None] * V + cid3                        # the bigram probability of each trigram candidate
                    q = np.minimum(np.searchsorted(self.keys1, k2), len(self.keys1) - 1)
                    p2 = self.gam[1][c1[j]][:, None] * self.p1[cid3] + np.where(self.keys1[q] == k2, self.s1[q], 0.0)
                    d3 = np.log1p(s3 / (self.gam3[rj][:, None] * p2))     # on top of the bigram row: log(P3 / (a3 * P2))
                    sl = (bb[j] * Th + pp[j] - (T - Th))[:, None] + np.zeros((1, K3), dtype=np.int64)
                    sp = (sl[val3], cid3[val3], d3[val3])
            bb, pp = np.nonzero(~vis & slot)                              # masked input (SAT): target is d+1 after the last visible token
            if len(bb):
                d = pp - last[bb, pp]
                for g in (2, 3, 4):
                    m = (last[bb, pp] >= 0) & (d == g - 1)
                    if m.any():
                        add(bb[m], pp[m], ids[bb[m], last[bb[m], pp[m]]], g, g - 1)
        else:
            bb, pp = np.nonzero(~vis & slot)                              # NAT: the masked token itself, d after the end of the prefix
            if len(bb):
                d = pp - last[bb, pp]
                for g in (1, 2, 3, 4):
                    m = (last[bb, pp] >= 0) & (d == g)
                    if m.any():
                        add(bb[m], pp[m], ids[bb[m], last[bb[m], pp[m]]], g, 3 + g)
        if rs:
            rows = tuple(np.concatenate(a) for a in (rs, rc, rg, rk))
        else:
            z = np.zeros(0, dtype=np.int64)
            rows = (z, z, z, z)
        return rows, sp

    def byte_rows(self, ctx, gap):
        out = np.empty((len(ctx), self.V), dtype=np.uint8)
        for g in (1, 2, 3, 4):
            m = gap == g
            if m.any():
                out[m] = self.Lq[g][ctx[m]]
        return out


CHUNK = 256                # slots per piece: 2 MB of byte rows, 8 MB once widened to fp32


class _Plan:
    """what one readout call adds: computed once on the host, used by the forward add and again by the backward"""

    def __init__(self, ch, ids, causal, Th, lifts, unigram, ramp):
        self.ch, self.unigram, self.ramp = ch, bool(unigram), float(ramp)
        z = np.zeros(0, dtype=np.int64)
        if lifts and ch.Lq:
            (self.slot, self.ctx, self.gap, self.kind), (self.ss, self.sv, self.sl) = ch.lifts(ids, causal, Th)
        else:
            self.slot = self.ctx = self.gap = self.kind = self.ss = self.sv = z
            self.sl = np.zeros(0)

    def pieces(self, dev):
        for a in range(0, len(self.slot), CHUNK):
            b = a + CHUNK
            yield (torch.from_numpy(self.slot[a:b]).to(dev), torch.from_numpy(self.kind[a:b]).to(dev),
                   torch.from_numpy(self.ch.byte_rows(self.ctx[a:b], self.gap[a:b])).to(dev))


class _ChainAdd(torch.autograd.Function):
    """logits += chain, IN PLACE, as one autograd node. The gradient to the logits is the incoming gradient itself
    (d(x + c)/dx = 1); the gradient to the 10 weights is gathered piece by piece in backward from the host tables.
    Nothing the size of the logits is kept for backward, and no copy of the logits gradient is made (adding in place
    to a view of the logits, as the first version did, makes autograd copy the whole gradient in backward once per
    add: that went over the queue's 720 MiB cap on 12 Nov)."""

    @staticmethod
    def forward(ctx, logits, lam, plan):
        flat = logits.view(-1, logits.shape[-1])
        dev, dt = flat.device, flat.dtype
        lam_d = lam.detach()
        if plan.unigram:
            flat.add_(plan.ch.logp1_on(dev, dt), alpha=float(plan.ramp * lam_d[9]))
        if len(plan.slot):
            w = lam_d.to(dt) * (plan.ramp / QS)
            for st, kd, R in plan.pieces(dev):
                Rf = R.to(dt)
                Rf.mul_(w[kd][:, None])
                flat.index_add_(0, st, Rf)
        if len(plan.ss):
            flat.index_put_((torch.from_numpy(plan.ss).to(dev), torch.from_numpy(plan.sv).to(dev)),
                            (plan.ramp * lam_d[8].to(dt)) * torch.from_numpy(plan.sl).to(device=dev, dtype=dt), accumulate=True)
        ctx.mark_dirty(logits)
        ctx.plan = plan
        return logits

    @staticmethod
    def backward(ctx, grad):
        plan = ctx.plan
        g = grad.reshape(-1, grad.shape[-1])
        dev = g.device
        gl = torch.zeros(NKIND, dtype=torch.float32, device=dev)
        if plan.unigram:
            gl[9] = plan.ramp * torch.dot(g.sum(0, dtype=torch.float32), plan.ch.logp1_on(dev, torch.float32))
        if len(plan.slot):
            for st, kd, R in plan.pieces(dev):
                dots = (g.index_select(0, st) * R.to(g.dtype)).sum(1, dtype=torch.float32)
                gl.index_add_(0, kd, dots * (plan.ramp / QS))
        if len(plan.ss):
            ss, sv = torch.from_numpy(plan.ss).to(dev), torch.from_numpy(plan.sv).to(dev)
            gl[8] = plan.ramp * (g[ss, sv].float() * torch.from_numpy(plan.sl).to(device=dev, dtype=torch.float32)).sum()
        return grad, gl, None


def apply_chain(ch, logits, lam, ids, causal, Th, lifts=True, unigram=True, ramp=1.0):
    """adds the chain to the logits ([..., V], last two dims = slots x vocabulary) in place and returns them;
    lam = the 10 trained weights; ramp in (0, 1] scales all of it (the late variant fades the chain in)"""
    return _ChainAdd.apply(logits, lam, _Plan(ch, ids, causal, Th, lifts, unigram, ramp))


_GUARDED = []


def patch(Block, Model, cfg, unigram=True, lifts=True, start_step=0, ramp_steps=0):
    """unigram / lifts: which parts of the chain are added. start_step > 0: nothing is added until that many training
    steps are done, then the chain fades in linearly over ramp_steps steps (needs this module's post_step)."""
    if not _GUARDED:                                   # patch() runs twice per run (simulator and 600B census configs)
        _GUARDED.append(True)
        _attempt_guard()

    class MarkovModel(Model):
        def __init__(self, c, BlockCls):
            super().__init__(c, BlockCls)
            self.chain_lam = nn.Parameter(torch.ones(NKIND))
            self._ids, self._causal, self._local, self._chain = None, True, False, None
            self.chain_step = 0                                # training steps done; post_step() keeps it current

        def chain_ramp(self):
            if not start_step:
                return 1.0
            if ramp_steps <= 0:
                return 1.0 if self.chain_step >= start_step else 0.0
            return min(1.0, max(0.0, (self.chain_step - start_step) / float(ramp_steps)))

        def chain(self):
            if self._chain is None:
                path = DATA + '/train.npy'
                if self.c['V'] != 8192 or not os.path.exists(path):
                    self._chain = False                        # the 600B census config, or no data: nothing to count from
                else:
                    self._chain = Chain(np.load(path, mmap_mode='r'), self.c['V'], self.c['ctx'], self.c['batch'], self.c['mask'],
                                        steps=_run_steps(), tables=lifts)
                    print('markov_corpus_prior chain: ' + str(self._chain.info), flush=True)
            return self._chain

        def hidden(self, ids, causal=True):
            self._ids, self._causal, self._local = ids, causal, False
            return super().hidden(ids, causal)

        def forward_local(self, ids, group, sigma, noise):
            self._ids, self._causal, self._local = ids, False, True
            return super().forward_local(ids, group, sigma, noise)

        def readout(self, h):
            logits = self.table.readout(h)
            ids = self._ids
            if ids is None:
                return logits
            ramp = self.chain_ramp()
            if ramp <= 0.0:
                return logits
            ch = self.chain()
            if not ch:
                return logits
            return apply_chain(ch, logits, self.chain_lam, ids.detach().cpu().numpy(), self._causal, h.shape[1],
                               lifts=lifts and not self._local, unigram=unigram, ramp=ramp)

    return Block, MarkovModel


def post_step(model, step, c):
    model.chain_step = step + 1
    if torch.cuda.is_available() and (step + 1 in (1, 10, 40, 150, 600, 1200) or (step + 1) % 300 == 0):
        print('markov_corpus_prior gpu: step %d max_allocated %.1f MiB max_reserved %.1f MiB lam %s' % (
            step + 1, torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20,
            [round(float(v), 3) for v in model.chain_lam.detach().cpu()]), flush=True)


if __name__ == '__main__':
    # self-check (not a simulator run): the chain alone, with no network, on held-out windows. A silent network plus
    # the chain must score what the exact chain scores (test_exact.py recounts that independently); a few seconds of numpy.
    import sys
    import torch.nn.functional as F
    V, T, B, MASK = 8192, 256, 8, 8191
    train = np.load(DATA + '/train.npy', mmap_mode='r')
    test = np.load(DATA + '/test.npy', mmap_mode='r')
    ch = Chain(train, V, T, B, MASK, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    print(ch.info)
    posn = np.linspace(0, len(test) - T - 2, 64).astype(np.int64)
    W = np.stack([np.asarray(test[int(s):int(s) + T + 1]) for s in posn]).astype(np.int64)
    x, y = W[:, :-1], torch.from_numpy(W[:, 1:])
    lam = torch.ones(NKIND)

    def run(inp, causal, lam=lam, n=64 * T, **kw):
        return apply_chain(ch, torch.zeros(n, V), lam, inp, causal, T, **kw)
    bi = torch.ones(NKIND)
    bi[8] = 0.0
    print('AR, no network: unigram alone %.4f | + bigram rows %.4f | + trigram correction %.4f' % (
        float(F.cross_entropy(run(x, True, lifts=False), y.reshape(-1))), float(F.cross_entropy(run(x, True, bi), y.reshape(-1))),
        float(F.cross_entropy(run(x, True), y.reshape(-1)))))
    x2 = x.copy()
    x2[:, 1::2] = MASK                                           # SAT-fixed layout: every second input masked
    (slot, ctx, gap, kind), _ = ch.lifts(x2, True, T)
    print('SAT-fixed, no network: unigram + chain %.4f | slot kinds %s' % (float(F.cross_entropy(run(x2, True), y.reshape(-1))), np.bincount(kind, minlength=NKIND).tolist()))
    x3 = x.copy()
    x3[:, T // 2:] = MASK                                        # NAT layout: masked suffix, targets are the masked tokens
    sel = np.zeros((64, T), dtype=bool)
    sel[:, T // 2:] = True
    tg = torch.from_numpy(x[sel])
    (slot, ctx, gap, kind), _ = ch.lifts(x3, False, T)
    st = torch.from_numpy(sel.reshape(-1))
    print('NAT, no network: unigram alone %.4f | unigram + chain %.4f | slot kinds %s' % (
        float(F.cross_entropy(run(x3, False, lifts=False)[st], tg)), float(F.cross_entropy(run(x3, False)[st], tg)), np.bincount(kind, minlength=NKIND).tolist()))
    half = float(F.cross_entropy(run(x, True, ramp=0.5), y.reshape(-1)))
    print('ramp 0.5 (half of every term): %.4f' % half)
    xa = x[:2].copy()
    xb = xa.copy()
    xb[:, 40] = (xb[:, 40] + 1) % 8190
    fa = run(xa, True, n=2 * T).view(2, T, V)
    fb = run(xb, True, n=2 * T).view(2, T, V)
    print('causal: slots before the edited token unchanged:', bool(torch.equal(fa[:, :40], fb[:, :40])), '| later slots changed:', not bool(torch.equal(fa[:, 40:], fb[:, 40:])))
    part = apply_chain(ch, torch.zeros(2 * 100, V), lam, xa, True, 100).view(2, 100, V)   # a right-aligned slice, as in the ctx-gain eval
    print('right-aligned slice equals the full pass:', bool(torch.equal(part, fa[:, T - 100:])))
    print('SELFCHECK-DONE')
