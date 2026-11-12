"""test_chainmix512.py -- the chain tables and the mixing arithmetic of chainmix512.py against direct counting. No worker,
no pack, no GPU:  python3 test_chainmix512.py"""
import math
import os
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainmix512 as cmx  # noqa: E402

V = 37


def stream(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, V, size=(V, V, 3))
    ids = np.zeros(n, dtype=np.int32)
    ids[0], ids[1] = rng.integers(V), rng.integers(V)
    for i in range(2, n):
        ids[i] = rng.integers(V) if rng.random() < 0.2 else A[ids[i - 2], ids[i - 1], rng.integers(3)]
    return ids


def brute(tok, meta, prev2, prev1, extra=()):
    """the chain's distribution by dictionary counting"""
    c1, c2, c3 = Counter(), Counter(), Counter()
    n = 0.0
    for t, w in [(tok, 1)] + [(e, w) for e, w, _ in extra]:
        t = [int(x) for x in t]
        n += w * len(t)
        for x in t:
            c1[x] += w
        for a, b in zip(t, t[1:]):
            c2[(a, b)] += w
        for a, b, c in zip(t, t[1:], t[2:]):
            c3[(a, b, c)] += w
    p = np.array([(c1[v] + meta['delta']) / (n + meta['delta'] * V) for v in range(V)])
    row = {b: c for (a, b), c in c2.items() if a == prev1}
    if row:
        tot = float(sum(row.values()))
        p = p * meta['D2'] * len(row) / tot
        for b, c in row.items():
            p[b] += (c - meta['D2']) / tot
    if prev2 is not None:
        row = {c: k for (a, b, c), k in c3.items() if a == prev2 and b == prev1}
        if row:
            tot = float(sum(row.values()))
            p = p * meta['D3'] * len(row) / tot
            for c, k in row.items():
                p[c] += (k - meta['D3']) / tot
    return p


class ChainTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = stream(6000, 0)
        cls.dir = tempfile.mkdtemp(prefix='chainmix-unit-')
        cls.meta = cmx.build(cls.tok, V, cls.dir, source='unit', eos=0)
        cls.t = cmx.Tables(cls.dir)

    def test_matches_direct_counting(self):
        rng = np.random.default_rng(5)
        worst = 0.0
        for _ in range(120):
            a, b = int(rng.integers(V)), int(rng.integers(V))
            for ctx in ((None, b), (a, b)):
                p = self.t.next_dist(*ctx)
                worst = max(worst, float(np.abs(p - brute(self.tok, self.meta, *ctx)).max()))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
                self.assertGreater(float(p.min()), 0.0)
        self.assertLess(worst, 1e-12)

    def test_target_prob_is_the_entry_of_next_dist(self):
        ids = [int(x) for x in self.tok[100:140]]
        tp = self.t.target_prob(ids)
        self.assertEqual(len(tp), len(ids) - 1)
        self.assertAlmostEqual(float(tp[0]), float(self.t.next_dist(None, ids[0])[ids[1]]), places=15)
        self.assertAlmostEqual(float(tp[7]), float(self.t.next_dist(ids[6], ids[7])[ids[8]]), places=15)

    def test_extra_text_is_counted_with_its_weight(self):
        ex = stream(900, 9)
        d = tempfile.mkdtemp(prefix='chainmix-unit-x-')
        meta = cmx.build(self.tok, V, d, source='unit', extra=[(ex, 4, 'lean')])
        t = cmx.Tables(d)
        self.assertEqual(meta['tokens'], len(self.tok) + 4 * len(ex))
        self.assertEqual((meta['D2'], meta['D3']), (self.meta['D2'], self.meta['D3']))      # discounts from the main stream
        rng = np.random.default_rng(6)
        for _ in range(60):
            a, b = int(rng.integers(V)), int(rng.integers(V))
            p = t.next_dist(a, b)
            self.assertLess(float(np.abs(p - brute(self.tok, meta, a, b, extra=[(ex, 4, 'lean')])).max()), 1e-12)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=9)

    def test_merge(self):
        k, c = cmx._merge(np.array([1, 5, 9]), np.array([2, 1, 4]), np.array([0, 5, 9, 11]), np.array([7, 3, 1, 1]))
        self.assertEqual((k.tolist(), c.tolist()), ([0, 1, 5, 9, 11], [7, 2, 4, 5, 1]))

    def test_wrong_size_or_vocabulary_is_refused(self):
        self.assertFalse(cmx.install(V + 1, self.dir))
        self.assertIn('vocabulary', cmx.STATE['error'])
        self.assertFalse(cmx.install(V, ''))
        self.assertTrue(cmx.install(V, self.dir))
        self.assertTrue(cmx.info()['supported'])


class Mixing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = stream(6000, 0)
        cls.dir = tempfile.mkdtemp(prefix='chainmix-unit-m-')
        cmx.build(cls.tok, V, cls.dir, source='unit')
        assert cmx.install(V, cls.dir)
        cls.t = cmx.STATE['tables']

    def test_parse(self):
        self.assertIsNone(cmx.parse({}))
        self.assertIsNone(cmx.parse({'chain_mix': 0}))
        self.assertIsNone(cmx.parse({'chain_mix': 0.5, 'mode': 'nat'}))
        self.assertEqual(cmx.parse({'chain_mix': 0.5, 'mode': 'nat'}, True)['g'], 0.5)       # /score has no mode
        self.assertEqual(cmx.parse({'chain_mix': 0.5})['g'], 0.5)
        for bad in (1.0, -0.1, 'x', True, float('nan'), [0.5]):
            with self.assertRaises(ValueError):
                cmx.parse({'chain_mix': bad})

    def test_mix_logits_is_the_mixture(self):
        g = torch.Generator().manual_seed(3)
        logits = torch.randn(V, generator=g) * 3
        hist = [int(x) for x in self.tok[50:60]]
        cm = cmx.parse({'chain_mix': 0.3})
        out = cmx.mix_logits(cm, logits.clone(), hist)
        ref = 0.7 * torch.softmax(logits.double(), -1) + 0.3 * torch.from_numpy(self.t.next_dist(hist[-2], hist[-1]))
        self.assertLess(float((out.double().exp() - ref).abs().max()), 1e-6)
        self.assertAlmostEqual(float(torch.logsumexp(out.double(), -1)), 0.0, places=5)
        self.assertEqual(out.dtype, logits.dtype)
        self.assertEqual(cm['tokens'], 1)
        one = cmx.mix_logits(cmx.parse({'chain_mix': 0.3}), logits.clone(), hist[-1:])      # one token of context
        ref1 = 0.7 * torch.softmax(logits.double(), -1) + 0.3 * torch.from_numpy(self.t.next_dist(None, hist[-1]))
        self.assertLess(float((one.double().exp() - ref1).abs().max()), 1e-6)

    def test_mix_rows_equals_mix_logits_row_by_row(self):
        g = torch.Generator().manual_seed(4)
        x = [int(v) for v in self.tok[200:212]]
        lg = torch.randn(5, V, generator=g)
        cm = cmx.parse({'chain_mix': 0.6}, True)
        out = cmx.mix_rows(cm, lg, x, 3)                              # rows are input positions 3 .. 7
        for j in range(5):
            ref = cmx.mix_logits(cmx.parse({'chain_mix': 0.6}), lg[j].clone(), x[:3 + j + 1])
            self.assertLess(float((out[j] - ref).abs().max()), 1e-5)
        self.assertEqual(cm['tokens'], 5)
        first = cmx.mix_rows(cmx.parse({'chain_mix': 0.6}, True), lg[:1], x, 0)               # position 0: one context token
        ref0 = cmx.mix_logits(cmx.parse({'chain_mix': 0.6}), lg[0].clone(), x[:1])
        self.assertLess(float((first[0] - ref0).abs().max()), 1e-5)

    def test_tables_can_be_replaced_without_a_restart(self):
        import json
        keep = dict(cmx.STATE)
        root = tempfile.mkdtemp(prefix='chainmix-unit-root-')
        try:
            a = cmx.build(self.tok, V, os.path.join(root, 'a'), source='a')
            b = cmx.build(stream(3000, 21), V, os.path.join(root, 'b'), source='b')
            cur = os.path.join(root, 'current.json')
            json.dump({'tables': 'a'}, open(cur, 'w'))
            self.assertTrue(cmx.install(V, root))
            self.assertTrue(cmx.info()['reloadable'])
            self.assertEqual(cmx.STATE['tables'].meta['id'], a['id'])
            cmx.parse({'chain_mix': 0.5})
            self.assertEqual(cmx.STATE['tables'].meta['id'], a['id'])           # nothing changed, nothing reloaded
            json.dump({'tables': 'b', 'note': 'second set'}, open(cur, 'w'))
            cmx.parse({'chain_mix': 0.5})                                        # the next chain request picks it up
            self.assertEqual(cmx.STATE['tables'].meta['id'], b['id'])
            json.dump({'tables': 'missing-folder-xyz'}, open(cur, 'w'))
            cm = cmx.parse({'chain_mix': 0.5})                                   # a bad update keeps the tables in use
            self.assertEqual(cmx.STATE['tables'].meta['id'], b['id'])
            self.assertIn('reload_error', cmx.info())
            self.assertEqual(cm['g'], 0.5)
            json.dump({'tables': '../a'}, open(cur, 'w'))
            cmx.parse({'chain_mix': 0.5})
            self.assertEqual(cmx.STATE['tables'].meta['id'], b['id'])           # no path tricks
            json.dump({'tables': 'a', 'again': True}, open(cur, 'w'))
            cmx.parse({'chain_mix': 0.5})
            self.assertEqual(cmx.STATE['tables'].meta['id'], a['id'])
            self.assertNotIn('reload_error', cmx.info())
        finally:
            cmx.STATE.clear()
            cmx.STATE.update(keep)

    def test_measured_sentence_comes_from_current_json_or_the_tables(self):
        import json
        keep = dict(cmx.STATE)
        root = tempfile.mkdtemp(prefix='chainmix-unit-meas-')
        try:
            cmx.build(self.tok, V, os.path.join(root, 'a'), source='a')
            cur = os.path.join(root, 'current.json')
            json.dump({'tables': 'a'}, open(cur, 'w'))
            self.assertTrue(cmx.install(V, root))
            self.assertEqual(cmx.info()['measured'], cmx.NOT_MEASURED)
            json.dump({'tables': 'a', 'measured': 'step 1: model 7.0, mixed 5.0'}, open(cur, 'w'))
            self.assertEqual(cmx.info()['measured'], 'step 1: model 7.0, mixed 5.0')        # /health picks it up, no request needed
            meta_path = os.path.join(root, 'a', 'meta.json')
            meta = json.load(open(meta_path))
            meta['measured'] = 'stored with the tables'
            json.dump(meta, open(meta_path, 'w'))
            json.dump({'tables': 'a', 'note': 'no sentence here'}, open(cur, 'w'))
            self.assertEqual(cmx.info()['measured'], 'stored with the tables')
            json.dump({'tables': 'a', 'measured': 7}, open(cur, 'w'))                        # not a sentence: ignored
            self.assertEqual(cmx.info()['measured'], 'stored with the tables')
        finally:
            cmx.STATE.clear()
            cmx.STATE.update(keep)

    def test_a_request_keeps_its_tables_while_others_are_loaded(self):
        import json
        keep = dict(cmx.STATE)
        root = tempfile.mkdtemp(prefix='chainmix-unit-keep-')
        try:
            a = cmx.build(self.tok, V, os.path.join(root, 'a'), source='a')
            b = cmx.build(stream(3000, 21), V, os.path.join(root, 'b'), source='b')
            cur = os.path.join(root, 'current.json')
            json.dump({'tables': 'a'}, open(cur, 'w'))
            self.assertTrue(cmx.install(V, root))
            ta = cmx.STATE['tables']
            cm = cmx.parse({'chain_mix': 0.5})                                    # a request starts on tables a
            json.dump({'tables': 'b', 'measured': 'second set'}, open(cur, 'w'))
            self.assertEqual(cmx.info()['tables']['id'], b['id'])                 # /health reloads meanwhile
            hist = [int(x) for x in self.tok[50:60]]
            logits = torch.randn(V, generator=torch.Generator().manual_seed(8))
            out = cmx.mix_logits(cm, logits.clone(), hist)
            ref = 0.5 * torch.softmax(logits.double(), -1) + 0.5 * torch.from_numpy(ta.next_dist(hist[-2], hist[-1]))
            self.assertLess(float((out.double().exp() - ref).abs().max()), 1e-6)   # still tables a
            self.assertEqual(cmx.summary(cm)['table'], a['id'])
            self.assertEqual(cmx.summary(cmx.parse({'chain_mix': 0.5}))['table'], b['id'])   # the next request is on b
        finally:
            cmx.STATE.clear()
            cmx.STATE.update(keep)

    def test_not_installed_is_refused_with_the_reason(self):
        keep = dict(cmx.STATE)
        cmx.STATE.update(tables=None, error='test reason')
        try:
            with self.assertRaises(ValueError) as e:
                cmx.parse({'chain_mix': 0.5})
            self.assertIn('test reason', str(e.exception))
            self.assertIsNone(cmx.parse({}))
            self.assertFalse(cmx.info()['supported'])
        finally:
            cmx.STATE.update(keep)


if __name__ == '__main__':
    assert math.isfinite(1.0)
    unittest.main(verbosity=1)
