#!/usr/bin/env python3
"""chain_big.py -- the same Markov chain as markov.py (interpolated absolute discounting, orders 1..3), with every
table sparse so it works at a 200,064-token vocabulary. numpy only.

    P3(w | u v) = max(C(u v w) - D3, 0) / C(u v)  +  D3 N1+(u v) / C(u v) * P2(w | v)
    P2(w | v)   = max(C(v w) - D2, 0) / C(v)      +  D2 N1+(v) / C(v)     * P1(w)
    P1(w)       = (C(w) + delta) / (N + delta V)
A context never seen passes through to the lower order. Inside a window only the window's own tokens are context.
"""
import numpy as np


def _disc(cnt):
    n1 = float(np.count_nonzero(cnt == 1))
    n2 = float(np.count_nonzero(cnt == 2))
    return min(0.95, max(0.05, n1 / (n1 + 2.0 * n2))) if n1 > 0 and n2 > 0 else 0.75


class Chain:
    def __init__(self, tok, V, delta=0.5, order=3):
        tok = np.asarray(tok)
        self.V, self.delta, self.N, self.order = int(V), float(delta), int(len(tok)), int(order)
        assert float(V) ** 3 < 2.0 ** 63
        self.c1 = np.bincount(tok, minlength=V).astype(np.float64)
        t = tok.astype(np.int64)
        k2 = t[:-1] * V + t[1:]
        self.u2, n2 = np.unique(k2, return_counts=True)
        self.n2 = n2.astype(np.float64)
        self.D2 = _disc(n2)
        self.c2k, start, types = np.unique(self.u2 // V, return_index=True, return_counts=True)
        self.c2tot = np.add.reduceat(n2, start).astype(np.float64)
        self.c2n1 = types.astype(np.float64)
        self.c2ptr = np.append(start, len(self.u2))
        if order >= 3:
            k3 = k2[:-1] * V + t[2:]
            del k2
            self.u3, n3 = np.unique(k3, return_counts=True)
            del k3
            self.n3 = n3.astype(np.float64)
            self.D3 = _disc(n3)
            self.c3k, start, types = np.unique(self.u3 // V, return_index=True, return_counts=True)
            self.c3tot = np.add.reduceat(n3, start).astype(np.float64)
            self.c3n1 = types.astype(np.float64)
            self.c3ptr = np.append(start, len(self.u3))

    @staticmethod
    def _find(keys, q):
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return i, keys[i] == q

    def target_prob(self, ids, order=None):
        """ids: int array [W, T+1] (windows). Returns P(ids[:, j+1] | window context up to j), float64 [W, T]."""
        order = order or self.order
        ids = np.asarray(ids).astype(np.int64)
        x, y = ids[:, :-1], ids[:, 1:]
        V = self.V
        p = (self.c1[y] + self.delta) / (self.N + self.delta * V)
        i, ok = self._find(self.u2, x * V + y)
        own = np.where(ok, self.n2[i], 0.0)
        r, okc = self._find(self.c2k, x)
        tot = np.where(okc, self.c2tot[r], 0.0)
        n1 = np.where(okc, self.c2n1[r], 0.0)
        den = np.maximum(tot, 1.0)
        p = np.where(tot > 0, np.maximum(own - self.D2, 0) / den + self.D2 * n1 / den * p, p)
        if order >= 3:
            xp = np.zeros_like(x)
            xp[:, 1:] = x[:, :-1]                      # the token before the current one
            ctx = xp * V + x
            inwin = np.zeros(x.shape, dtype=bool)
            inwin[:, 1:] = True
            i, ok = self._find(self.u3, ctx * V + y)
            own = np.where(ok & inwin, self.n3[i], 0.0)
            r, okc = self._find(self.c3k, ctx)
            okc &= inwin
            tot = np.where(okc, self.c3tot[r], 0.0)
            n1 = np.where(okc, self.c3n1[r], 0.0)
            den = np.maximum(tot, 1.0)
            p = np.where(tot > 0, np.maximum(own - self.D3, 0) / den + self.D3 * n1 / den * p, p)
        return p

    def next_dist(self, prev2, prev1):
        """full next-token distribution after (prev2, prev1) as a dense float64 [V] vector (prev2 = None: one token
        of context). This is what a decoder would mix into the model's distribution."""
        V = self.V
        p = (self.c1 + self.delta) / (self.N + self.delta * V)
        r, ok = self._find(self.c2k, np.int64(prev1))
        if ok:
            a, b = self.c2ptr[r], self.c2ptr[r + 1]
            tot = self.c2tot[r]
            p = p * (self.D2 * self.c2n1[r] / tot)
            p[self.u2[a:b] % V] += np.maximum(self.n2[a:b] - self.D2, 0) / tot
        if prev2 is not None and self.order >= 3:
            r, ok = self._find(self.c3k, np.int64(prev2) * V + np.int64(prev1))
            if ok:
                a, b = self.c3ptr[r], self.c3ptr[r + 1]
                tot = self.c3tot[r]
                p = p * (self.D3 * self.c3n1[r] / tot)
                p[self.u3[a:b] % V] += np.maximum(self.n3[a:b] - self.D3, 0) / tot
        return p
