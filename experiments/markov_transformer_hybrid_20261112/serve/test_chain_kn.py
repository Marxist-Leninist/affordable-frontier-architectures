"""test_chain_kn.py -- the Kneser-Ney tables of chain_kn_build.py against direct dictionary counting, read through the
UNCHANGED loader of chainmix512.py (the one the worker runs).  python3 test_chain_kn.py"""
import os
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_kn_build as kn  # noqa: E402
import chainmix512 as cmx  # noqa: E402
from test_chainmix512 import V, stream  # noqa: E402


def brute_kn(tok, meta, prev2, prev1, extra=()):
    c3, tw = Counter(), {}
    n = 0.0
    for t, w in [(tok, 1)] + [(e, w) for e, w, _ in extra]:
        t = [int(x) for x in t]
        n += w * len(t)
        for g in zip(t, t[1:], t[2:]):
            c3[g] += w
            tw[g] = max(tw.get(g, 1), w)
    cont2 = Counter()
    for (a, b, c), w in tw.items():
        cont2[(b, c)] += w
    cont1 = Counter()
    for (b, c) in cont2:
        cont1[c] += 1
    alpha = n / float(sum(cont1.values()))
    p = np.array([(alpha * cont1[v] + meta['delta']) / (n + meta['delta'] * V) for v in range(V)])
    row = {c: k for (b, c), k in cont2.items() if b == prev1}
    if row:
        tot = float(sum(row.values()))
        p = p * meta['D2'] * len(row) / tot
        for c, k in row.items():
            p[c] += (k - meta['D2']) / tot
    if prev2 is not None:
        row = {c: k for (a, b, c), k in c3.items() if a == prev2 and b == prev1}
        if row:
            tot = float(sum(row.values()))
            p = p * meta['D3'] * len(row) / tot
            for c, k in row.items():
                p[c] += (k - meta['D3']) / tot
    return p


class KnTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = stream(6000, 0)
        cls.dir = tempfile.mkdtemp(prefix='chainkn-unit-')
        cls.meta = kn.build_kn(cls.tok, V, cls.dir, source='unit', eos=0)
        cls.t = cmx.Tables(cls.dir)

    def test_same_format_as_the_worker_reads(self):
        self.assertEqual(self.meta['format'], cmx.FORMAT)
        self.t.selftest()
        self.assertTrue(cmx.install(V, self.dir))
        self.assertTrue(cmx.info()['supported'])

    def test_matches_direct_counting(self):
        rng = np.random.default_rng(5)
        worst = 0.0
        for _ in range(150):
            a, b = int(rng.integers(V)), int(rng.integers(V))
            for ctx in ((None, b), (a, b)):
                p = self.t.next_dist(*ctx)
                worst = max(worst, float(np.abs(p - brute_kn(self.tok, self.meta, *ctx)).max()))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
                self.assertGreater(float(p.min()), 0.0)
        self.assertLess(worst, 1e-12)

    def test_extra_text_and_its_weight(self):
        ex, ex2 = stream(900, 9), stream(500, 13)
        d = tempfile.mkdtemp(prefix='chainkn-unit-x-')
        extra = [(ex, 4, 'lean'), (ex2, 2, 'second')]
        meta = kn.build_kn(self.tok, V, d, source='unit', extra=extra)
        t = cmx.Tables(d)
        t.selftest()
        self.assertEqual(meta['tokens'], len(self.tok) + 4 * len(ex) + 2 * len(ex2))
        self.assertEqual((meta['D2'], meta['D3']), (self.meta['D2'], self.meta['D3']))      # discounts from the main stream
        rng = np.random.default_rng(6)
        worst = 0.0
        for _ in range(80):
            a, b = int(rng.integers(V)), int(rng.integers(V))
            for ctx in ((None, b), (a, b)):
                p = t.next_dist(*ctx)
                worst = max(worst, float(np.abs(p - brute_kn(self.tok, meta, *ctx, extra=extra)).max()))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertLess(worst, 1e-12)

    def test_phrase_bound_token_is_not_offered_everywhere(self):
        # token 5 occurs often, but only ever after token 4; after an unrelated context the Kneser-Ney chain must give it
        # less than the plain chain does, and after token 4 about the same
        rng = np.random.default_rng(1)
        body = rng.integers(6, V, size=20000)
        at = rng.choice(len(body) - 2, size=1500, replace=False)
        body[at] = 4
        body[at + 1] = 5
        tok = body.astype(np.int32)
        d1, d2 = tempfile.mkdtemp(prefix='chainkn-unit-p-'), tempfile.mkdtemp(prefix='chainkn-unit-k-')
        cmx.build(tok, V, d1, source='plain')
        kn.build_kn(tok, V, d2, source='kn')
        plain, knt = cmx.Tables(d1), cmx.Tables(d2)
        ctx = (int(tok[100]) if tok[100] != 4 else 7, 9)
        pairs = set(zip(tok[:-1].tolist(), tok[1:].tolist()))
        self.assertNotIn((9, 5), pairs)
        self.assertLess(float(knt.next_dist(*ctx)[5]), 0.2 * float(plain.next_dist(*ctx)[5]))
        self.assertGreater(float(knt.next_dist(None, 4)[5]), 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=1)
