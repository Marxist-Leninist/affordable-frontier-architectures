#!/usr/bin/env python3
"""markov.py -- a token-level Markov chain (n-gram model) built by counting, to be used beside a transformer.

Model: interpolated absolute discounting. For order k (context c = the last k-1 tokens):
    P_k(w | c) = max(C(c, w) - D_k, 0) / C(c)  +  D_k * N1+(c) / C(c) * P_{k-1}(w | shorter c)
    P_1(w)     = (C(w) + delta) / (N + delta * V)
A context that was never seen (C(c) = 0) passes straight through to the lower order. N1+(c) is the number of distinct
tokens seen after c. D_k = n1 / (n1 + 2 n2) from the count-of-counts of order k (Ney, Essen, Kneser 1994).
order=2 is the textbook first-order Markov chain: the next token depends on the current token only.

All counts come from the TRAINING stream. On training text the chain must not be allowed to contain the answer it is
helping to predict, otherwise it looks far more reliable there than it is on new text and a model trained beside it
learns to trust it too much. Three modes:
    full   all counts (dev / test text, generation)
    loo    leave-one-out: the position's own n-gram event is removed from every count
    lobo   leave-one-block-out: every event of the position's block (a group of whole documents) is removed. This is
           ordinary cross-fitting; it also removes the repetition inside a document, which leave-one-out cannot.

Inside a window only the window's own tokens are used: at window index j the chain sees min(j + 1, order - 1) context
tokens, which is exactly what the transformer beside it can see.
"""
import os
import warnings

import numpy as np
import torch


def _pack(a, k1, V):
    """int64 keys of the k1-token contexts a[t .. t+k1-1] for t = 0 .. len(a)-k1 (oldest token most significant)."""
    n = len(a) - k1 + 1
    key = np.zeros(n, dtype=np.int64)
    for m in range(k1):
        key = key * V + a[m:m + n]
    return key


def _discount(n1c, n2c):
    if n1c <= 0 or n2c <= 0:
        return 0.75
    return float(min(0.95, max(0.05, n1c / (n1c + 2.0 * n2c))))


def build_tables(ids, V, order, delta=0.5):
    """Count tables for orders 1..order from one token stream (numpy). Returns a dict of arrays (np.savez-able)."""
    ids = np.asarray(ids).astype(np.int64)
    N = len(ids)
    assert N > order and ids.min() >= 0 and ids.max() < V
    assert float(V) ** order < 2.0 ** 62, 'context keys would overflow int64'
    t = {'V': np.int64(V), 'order': np.int64(order), 'delta': np.float64(delta), 'N': np.int64(N)}
    t['C1'] = np.bincount(ids, minlength=V).astype(np.int64)
    D = []
    C2 = np.bincount(ids[:-1] * V + ids[1:], minlength=V * V).reshape(V, V).astype(np.int32)
    t['C2'] = C2
    D.append(_discount(int((C2 == 1).sum()), int((C2 == 2).sum())))
    for k in range(3, order + 1):
        k1 = k - 1
        ctx = _pack(ids[:N - 1], k1, V)            # contexts ending at i = k1-1 .. N-2
        full = ctx * V + ids[k1:]                  # followed by the token at i+1
        ufull, cnt = np.unique(full, return_counts=True)
        uctx, start, n1 = np.unique(ufull // V, return_index=True, return_counts=True)
        t[f'ufull{k}'] = ufull.astype(np.int64)
        t[f'cnt{k}'] = cnt.astype(np.int32)
        t[f'uctx{k}'] = uctx.astype(np.int64)
        t[f'indptr{k}'] = np.append(start, len(ufull)).astype(np.int64)
        t[f'tot{k}'] = np.add.reduceat(cnt, start).astype(np.int64)
        t[f'n1{k}'] = n1.astype(np.int64)
        D.append(_discount(int((cnt == 1).sum()), int((cnt == 2).sum())))
    t['D'] = np.array(D, dtype=np.float64)
    return t


def build_block_tables(ids, V, order, blk_of_pos, tables):
    """Per-block counts for leave-one-block-out. blk_of_pos[i] = block id of token position i (blocks are groups of
    whole documents). An n-gram event belongs to the block of its TARGET token. Stored per (block, n-gram): the block's
    count and the change it makes to the stored global value when the block is removed."""
    ids = np.asarray(ids).astype(np.int64)
    N = len(ids)
    blk = np.asarray(blk_of_pos).astype(np.int64)
    assert len(blk) == N and blk.min() >= 0
    nb = int(blk.max()) + 1
    D = np.asarray(tables['D'], dtype=np.float64)
    out = {'nb': np.int64(nb), 'blk': blk.astype(np.int32)}
    out['C1b'] = np.bincount(blk * V + ids, minlength=nb * V).reshape(nb, V).astype(np.int32)
    C2 = np.asarray(tables['C2'])
    tot2 = C2.sum(1).astype(np.float64)
    for k in range(2, order + 1):
        k1 = k - 1
        VK = V ** k
        assert nb * float(VK) < 2.0 ** 62
        full = _pack(ids[:N - 1], k1, V) * V + ids[k1:]
        ubfull, bcnt = np.unique(blk[k1:] * VK + full, return_counts=True)      # sorted by (block, context, next)
        fe = ubfull % VK
        ce = fe // V
        if k == 2:
            Cg = C2[ce, fe % V].astype(np.float64)
            totg = tot2[ce]
        else:
            gi = np.searchsorted(tables[f'ufull{k}'], fe)
            assert np.array_equal(np.asarray(tables[f'ufull{k}'])[gi], fe)
            Cg = np.asarray(tables[f'cnt{k}'])[gi].astype(np.float64)
            totg = np.asarray(tables[f'tot{k}'])[np.searchsorted(tables[f'uctx{k}'], ce)].astype(np.float64)
        Dk = D[k - 2]
        ubctx, start = np.unique(ubfull // V, return_index=True)               # rows keyed by (block, context)
        out[f'bfull{k}'] = ubfull.astype(np.int64)
        out[f'bcnt{k}'] = bcnt.astype(np.int32)
        out[f'bdn{k}'] = ((np.maximum(Cg - bcnt - Dk, 0) - (Cg - Dk)) / totg).astype(np.float32)
        out[f'bctx{k}'] = ubctx.astype(np.int64)
        out[f'bindptr{k}'] = np.append(start, len(ubfull)).astype(np.int64)
        out[f'btot{k}'] = np.add.reduceat(bcnt, start).astype(np.int64)
        out[f'blost{k}'] = np.add.reduceat((bcnt == Cg).astype(np.int64), start)   # continuations that vanish
    return out


BLOCK_DTYPES = {'blk': np.int32, 'C1b': np.int32, 'bfull': np.int64, 'bcnt': np.float32, 'bdn': np.float32,
                'bctx': np.int64, 'bindptr': np.int64, 'btot': np.float32, 'blost': np.float32}


def save_block_cache(bt, folder, order):
    """write the block tables as one .npy per array in their run-time dtypes (counts are exact in float32)"""
    os.makedirs(folder, exist_ok=True)
    for name in ['blk', 'C1b'] + [f'{n}{k}' for k in range(2, order + 1) for n in BLOCK_DTYPES if n not in ('blk', 'C1b')]:
        np.save(os.path.join(folder, name + '.npy'), np.asarray(bt[name]).astype(BLOCK_DTYPES[name.rstrip('0123456789')]))


class MarkovPrior:
    """Torch (CPU) evaluator. order <= the tables' order picks a shorter chain from the same counts.
    btables (the dict from build_block_tables, or a folder from save_block_cache) enables mode='lobo'."""

    def __init__(self, tables, order=None, btables=None):
        V = int(tables['V'])
        self.V = V
        max_order = int(tables['order'])
        self.order = int(order or max_order)
        assert 2 <= self.order <= max_order
        self.delta = float(tables['delta'])
        self.N = float(tables['N'])
        Dall = np.asarray(tables['D'], dtype=np.float64)
        self.D = {k: float(Dall[k - 2]) for k in range(2, self.order + 1)}
        self.C1 = torch.from_numpy(np.asarray(tables['C1']).astype(np.float64))
        self.P1 = ((self.C1 + self.delta) / (self.N + self.delta * V)).float()
        self.C2 = torch.from_numpy(np.ascontiguousarray(np.asarray(tables['C2']).astype(np.int32)))
        self.tot2 = self.C2.sum(1, dtype=torch.float64)
        self.n12 = (self.C2 > 0).sum(1).double()
        self.S2 = ((self.C2.double() - self.D[2]).clamp_(min=0) / self.tot2.clamp(min=1).unsqueeze(1)).float()
        self.hi = {}
        for k in range(3, self.order + 1):
            ufull = torch.from_numpy(np.asarray(tables[f'ufull{k}']).astype(np.int64))
            cnt = torch.from_numpy(np.asarray(tables[f'cnt{k}']).astype(np.float64))
            uctx = torch.from_numpy(np.asarray(tables[f'uctx{k}']).astype(np.int64))
            indptr = torch.from_numpy(np.asarray(tables[f'indptr{k}']).astype(np.int64))
            tot = torch.from_numpy(np.asarray(tables[f'tot{k}']).astype(np.float64))
            n1 = torch.from_numpy(np.asarray(tables[f'n1{k}']).astype(np.float64))
            rows = torch.repeat_interleave(torch.arange(len(uctx)), indptr[1:] - indptr[:-1])
            spv = ((cnt - self.D[k]) / tot[rows]).float()
            self.hi[k] = dict(ufull=ufull, cnt=cnt, uctx=uctx, indptr=indptr, tot=tot, n1=n1, nxt=ufull % V, spv=spv)
        self.bt = None
        if btables is not None:
            # btables: the dict from build_block_tables, or a folder written by save_block_cache. The folder is
            # memory-mapped read-only, so parallel runs share one copy (about one entry per training token per order).
            def arr(name, dtype):
                if isinstance(btables, str):
                    a = np.load(os.path.join(btables, name + '.npy'), mmap_mode='r')
                    assert a.dtype == dtype, (name, a.dtype)
                else:
                    a = np.asarray(btables[name]).astype(dtype)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')          # "array is not writable": it is only ever read
                    return torch.from_numpy(a)
            self.blk = arr('blk', np.int32)
            self.C1b = arr('C1b', np.int32).double()
            self.Nb = self.C1b.sum(1)
            self.P1blk = ((self.C1.unsqueeze(0) - self.C1b + self.delta)
                          / (self.N - self.Nb + self.delta * V).unsqueeze(1)).float()
            self.bt = {}
            for k in range(2, self.order + 1):
                self.bt[k] = dict(bfull=arr(f'bfull{k}', np.int64), bcnt=arr(f'bcnt{k}', np.float32),
                                  bdn=arr(f'bdn{k}', np.float32), bctx=arr(f'bctx{k}', np.int64),
                                  bindptr=arr(f'bindptr{k}', np.int64), btot=arr(f'btot{k}', np.float32),
                                  blost=arr(f'blost{k}', np.float32))

    # ------------------------------------------------------------------ per-order scalars for a batch of windows
    def _stats(self, x, y, mode, tpos=None):
        """x [B,T] int64 window tokens; y [B,T] next tokens or None; tpos [B,T] stream positions of the targets (lobo).
        Returns {k: (valid, gam, sc, own_e, den, row, brow)} for k = 2..order, every entry [B,T]:
          valid  the order-k context is inside the window and has (remaining) count > 0
          gam    back-off weight D_k * N1+ / C (1 where not valid)
          sc     factor that turns the stored (C(c,w) - D)/C(c) values into (C(c,w) - D)/C_eff(c) (0 where not valid)
          own_e  (remaining) count of the event (context, y);  den  C_eff(c) clamped to >= 1;  row  table row (k >= 3)
          brow   (row in the block table, found there) for lobo, else None
        """
        assert mode in ('full', 'loo', 'lobo')
        assert mode == 'full' or y is not None or mode == 'lobo'
        B, T = x.shape
        V = self.V
        j = torch.arange(T)
        zero = torch.zeros((), dtype=torch.float64)
        blk = None
        if mode == 'lobo':
            assert self.bt is not None and tpos is not None
            blk = self.blk[tpos].long()
        out = {}
        key = x.clone()                      # running context key, most recent token least significant
        mult = V
        for k in range(2, self.order + 1):
            if k == 2:
                found = torch.ones_like(x, dtype=torch.bool)
                tot = self.tot2[x]
                n1 = self.n12[x]
                own = self.C2[x, y].double() if y is not None else None
                row = None
            else:
                m = k - 2                    # bring in the token m places back
                shifted = torch.zeros_like(x)
                shifted[:, m:] = x[:, :T - m]
                key = key + shifted * mult
                mult *= V
                h = self.hi[k]
                idx = torch.searchsorted(h['uctx'], key).clamp_(max=len(h['uctx']) - 1)
                found = (h['uctx'][idx] == key) & (j >= m).unsqueeze(0)
                tot = torch.where(found, h['tot'][idx], zero)
                n1 = torch.where(found, h['n1'][idx], zero)
                row = idx
                if y is not None:
                    fk = key * V + y
                    fi = torch.searchsorted(h['ufull'], fk).clamp_(max=len(h['ufull']) - 1)
                    own = torch.where((h['ufull'][fi] == fk) & found, h['cnt'][fi], zero)
                else:
                    own = None
            brow = None
            if mode == 'loo':
                tot_e = tot - (own >= 1).double()
                n1_e = n1 - (own == 1).double()
                own_e = (own - 1).clamp(min=0)
            elif mode == 'lobo':
                bt = self.bt[k]
                rk = blk * (V ** (k - 1)) + key
                bi = torch.searchsorted(bt['bctx'], rk).clamp_(max=len(bt['bctx']) - 1)
                bfound = (bt['bctx'][bi] == rk) & found
                tot_e = tot - torch.where(bfound, bt['btot'][bi].double(), zero)
                n1_e = n1 - torch.where(bfound, bt['blost'][bi].double(), zero)
                own_e = None
                if y is not None:
                    fk = rk * V + y
                    fi = torch.searchsorted(bt['bfull'], fk).clamp_(max=len(bt['bfull']) - 1)
                    own_e = own - torch.where((bt['bfull'][fi] == fk) & bfound, bt['bcnt'][fi].double(), zero)
                brow = (bi, bfound)
            else:
                tot_e, n1_e, own_e = tot, n1, own
            valid = found & (tot_e > 0)
            den = tot_e.clamp(min=1)
            gam = torch.where(valid, self.D[k] * n1_e / den, torch.ones((), dtype=torch.float64))
            sc = torch.where(valid, tot / den, zero)
            out[k] = (valid, gam, sc, own_e, den, row, brow)
        out['blk'] = blk
        return out

    def _target(self, st, y, mode):
        """probability of the actual next token, float64 [B,T]"""
        V = self.V
        c1 = self.C1[y]
        if mode == 'loo':
            p = ((c1 - 1).clamp(min=0) + self.delta) / (self.N - 1 + self.delta * V)
        elif mode == 'lobo':
            blk = st['blk']
            p = (c1 - self.C1b[blk, y] + self.delta) / (self.N - self.Nb[blk] + self.delta * V)
        else:
            p = (c1 + self.delta) / (self.N + self.delta * V)
        for k in range(2, self.order + 1):
            valid, gam, sc, own_e, den, row, brow = st[k]
            p = torch.where(valid, (own_e - self.D[k]).clamp(min=0) / den + gam * p, p)
        return p

    @staticmethod
    def _mode(loo, mode):
        return mode or ('loo' if loo else 'full')

    @torch.no_grad()
    def target_logprob(self, x, y, loo=False, mode=None, tpos=None):
        """log P_chain(y | context) for every position, float64 [B,T]. No dense vectors are built."""
        mode = self._mode(loo, mode)
        return self._target(self._stats(x, y, mode, tpos), y, mode).log()

    @torch.no_grad()
    def dense_logprob(self, x, y=None, loo=False, mode=None, tpos=None):
        """log P_chain(. | context) for every position and every next token, float32 [B,T,V].
        loo / lobo need y (the targets of the training window); lobo also needs tpos."""
        mode = self._mode(loo, mode)
        B, T = x.shape
        V = self.V
        st = self._stats(x, y if mode != 'full' else None, mode, tpos)
        w = torch.ones(B * T, dtype=torch.float64)
        wk = {}
        for k in range(self.order, 1, -1):
            valid, gam, sc, own_e, den, row, brow = st[k]
            wk[k] = (w * sc.reshape(-1)).float()
            w = w * gam.reshape(-1)
        P = self.S2.index_select(0, x.reshape(-1))                  # [BT,V]
        P.mul_(wk[2].unsqueeze(1))
        if mode == 'lobo':
            P.addcmul_(self.P1blk.index_select(0, st['blk'].reshape(-1)), w.float().unsqueeze(1))
        else:
            s1 = (self.N + self.delta * V) / (self.N - 1 + self.delta * V) if mode == 'loo' else 1.0
            P.addr_((w * s1).float(), self.P1)
        Pf = P.view(-1)
        for k in range(3, self.order + 1):
            valid, gam, sc, own_e, den, row, brow = st[k]
            pos = torch.nonzero(valid.reshape(-1)).squeeze(1)
            if pos.numel() == 0:
                continue
            h = self.hi[k]
            e, rep = _expand(h['indptr'], row.reshape(-1)[pos])
            Pf.index_add_(0, pos[rep] * V + h['nxt'][e], h['spv'][e] * wk[k][pos][rep])
        if mode == 'lobo':
            for k in range(2, self.order + 1):
                valid, gam, sc, own_e, den, row, brow = st[k]
                bi, bfound = brow
                pos = torch.nonzero((bfound & valid).reshape(-1)).squeeze(1)
                if pos.numel() == 0:
                    continue
                bt = self.bt[k]
                e, rep = _expand(bt['bindptr'], bi.reshape(-1)[pos])
                Pf.index_add_(0, pos[rep] * V + bt['bfull'][e] % V, bt['bdn'][e] * wk[k][pos][rep])
            P.clamp_(min=1e-12)          # removed entries cancel only to float32 rounding
        if mode != 'full' and y is not None:
            # the target entry is the one that matters for the loss: write its exact (float64) value
            Pf[torch.arange(B * T) * V + y.reshape(-1)] = self._target(st, y, mode).reshape(-1).float()
        return P.log_().view(B, T, V)


def _expand(indptr, r):
    """all table entries of the rows r: (entry index, index into r) flattened"""
    starts = indptr[r]
    lens = indptr[r + 1] - starts
    rep = torch.repeat_interleave(torch.arange(r.numel()), lens)
    offs = torch.arange(int(lens.sum())) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    return starts[rep] + offs, rep


# ---------------------------------------------------------------------- slow reference, used only by the tests
def brute_dist(ids, V, order, D, delta, ctx, loo_y=None, drop=None):
    """Next-token distribution after ctx (list, most recent token last; up to order-1 of them are used), computed by
    direct counting over ids. loo_y: remove one occurrence of the event (context, loo_y) at every order first.
    drop: bool array over token positions; an event is ignored when its target position is flagged, and so is the
    flagged token's own unigram count (leave-one-block-out)."""
    ids = np.asarray(ids).astype(np.int64)
    N = len(ids)
    keep = np.ones(N, dtype=bool) if drop is None else ~np.asarray(drop)
    C1 = np.bincount(ids[keep], minlength=V).astype(np.float64)
    Neff = float(keep.sum())
    if loo_y is not None:
        C1[loo_y] = max(C1[loo_y] - 1, 0)
        Neff -= 1
    p = (C1 + delta) / (Neff + delta * V)
    for k in range(2, order + 1):
        k1 = k - 1
        if len(ctx) < k1:
            break
        c = np.asarray(ctx[-k1:], dtype=np.int64)
        cnt = np.zeros(V, dtype=np.float64)
        ok = keep[k1:].copy()                            # start t of a context, next token at t+k1
        for m in range(k1):
            ok &= ids[m:m + N - k1] == c[m]
        np.add.at(cnt, ids[k1:][ok], 1.0)
        if loo_y is not None and cnt[loo_y] >= 1:
            cnt[loo_y] -= 1
        tot = cnt.sum()
        if tot <= 0:
            continue
        n1 = float((cnt > 0).sum())
        p = np.maximum(cnt - D[k], 0) / tot + D[k] * n1 / tot * p
    return p
