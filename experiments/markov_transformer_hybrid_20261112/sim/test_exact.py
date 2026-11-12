"""test_exact.py PLUGIN.py DATA_DIR [STEPS] -- an independent recount. The exact interpolated chain (no pruning, no
byte rounding), from counts made HERE, scored on the same held-out windows as the plugin's self-check. The only thing
taken from the plugin is which tokens are counted (Chain._keep). Also prints two estimators the plugin does not use
(count-dependent discounts, Kneser-Ney lower order), to show what is left on the table. numpy only."""
import importlib.util
import os
import sys

import numpy as np

plug_path, data = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
os.environ['AGILLM_SIM_DATA'] = data
spec = importlib.util.spec_from_file_location('plug', plug_path)
plug = importlib.util.module_from_spec(spec)
sys.modules['plug'] = plug
spec.loader.exec_module(plug)
V, T, B, DELTA = 8192, 256, 8, 0.5
train = np.load(data + '/train.npy', mmap_mode='r')
test = np.load(data + '/test.npy', mmap_mode='r')
N = len(train)
keep = plug.Chain._keep(N, T, B, steps, plug.MARGIN)
x = np.asarray(train).astype(np.int64)


def table(n):
    """sorted keys and counts of the n-grams that lie wholly in counted text"""
    m = N - n + 1
    ok = keep[:m].copy()
    key = x[:m].copy()
    for j in range(1, n):
        ok &= keep[j:m + j]
        key = key * V + x[j:m + j]
    return np.unique(key[ok], return_counts=True)


def look(keys, vals, q):
    i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return np.where(keys[i] == q, vals[i], 0)


def disc1(cnt):
    n1, n2 = float((cnt == 1).sum()), float((cnt == 2).sum())
    return min(0.95, max(0.05, n1 / (n1 + 2 * n2)))


def disc3(cnt):
    n = [float((cnt == k).sum()) for k in (1, 2, 3, 4)]
    Y = n[0] / (n[0] + 2 * n[1])
    return np.array([0.0, 1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])


class Order:
    """one order of the chain: P(y | ctx) = (c - D(c)) / tot + gam(ctx) * lower(y)"""

    def __init__(self, keys, cnt, modified=False):
        self.keys, self.cnt = keys, cnt.astype(np.float64)
        self.ctx, start, self.types = np.unique(keys // V, return_index=True, return_counts=True)
        self.tot = np.add.reduceat(self.cnt, start)
        if modified:
            D = disc3(cnt)
            d = D[np.minimum(cnt, 3)]
            self.D = D
        else:
            d = np.full(len(cnt), disc1(cnt))
            self.D = d[0]
        self.d = d
        self.gam = np.add.reduceat(d, start) / self.tot

    def prob(self, ctx, y, lower):
        """ctx, y arrays; lower = P_lower(y) array. Where the context was never counted the lower order is returned."""
        i = np.minimum(np.searchsorted(self.ctx, ctx), len(self.ctx) - 1)
        seen = self.ctx[i] == ctx
        j = np.minimum(np.searchsorted(self.keys, ctx * V + y), len(self.keys) - 1)
        hit = self.keys[j] == ctx * V + y
        s = np.where(hit, (self.cnt[j] - self.d[j]) / self.tot[i], 0.0)
        return np.where(seen, s + self.gam[i] * lower, lower)


c1 = np.bincount(x[keep], minlength=V).astype(np.float64)
p1 = (c1 + DELTA) / (c1.sum() + DELTA * V)
k2, n2 = table(2)
k3, n3 = table(3)
print('counted tokens %d (%.1f %%) | bigram types %d | trigram types %d' % (keep.sum(), 100 * keep.mean(), len(k2), len(k3)))

posn = np.linspace(0, len(test) - T - 2, 64).astype(np.int64)
W = np.stack([np.asarray(test[int(s):int(s) + T + 1]) for s in posn]).astype(np.int64)
a = np.concatenate([np.full((64, 1), -1), W[:, :-2]], axis=1).reshape(-1)      # token before the context token (-1 at slot 0)
b = W[:, :-1].reshape(-1)
y = W[:, 1:].reshape(-1)
has_a = a >= 0


def nll(p):
    return float(-np.log(p).mean())


out = {}
for name, mod in (('one discount per order (what the plugin uses)', False), ('count-dependent discounts D1, D2, D3+', True)):
    o2, o3 = Order(k2, n2, mod), Order(k3, n3, mod)
    P2 = o2.prob(b, y, p1[y])
    P3 = np.where(has_a, o3.prob(np.where(has_a, a, 0) * V + b, y, P2), P2)
    out[name] = (nll(p1[y]), nll(P2), nll(P3))
    print('%s: unigram %.4f | bigram %.4f | trigram %.4f | discounts %s / %s' % (name, *out[name], np.round(o2.D, 3), np.round(o3.D, 3)))

# Kneser-Ney lower orders for the trigram: the bigram it backs off to counts distinct left contexts, not occurrences
kb, nb = np.unique(k3 % (V * V), return_counts=True)            # (b, y) -> number of distinct a seen before it
ku, nu = np.unique(k2 % V, return_counts=True)                  # y -> number of distinct b seen before it
pc1 = np.full(V, DELTA)
pc1[ku] += nu
pc1 /= pc1.sum()
for name, mod in (('Kneser-Ney lower orders, one discount', False), ('Kneser-Ney lower orders, count-dependent discounts', True)):
    ob, o3 = Order(kb, nb, mod), Order(k3, n3, mod)
    P2c = ob.prob(b, y, pc1[y])
    o2 = Order(k2, n2, mod)
    P2 = o2.prob(b, y, p1[y])                                    # slot 0 has no left neighbour: the ordinary bigram
    P3 = np.where(has_a, o3.prob(np.where(has_a, a, 0) * V + b, y, P2c), P2)
    print('%s: trigram %.4f' % (name, nll(P3)))
print('EXACT-DONE')
