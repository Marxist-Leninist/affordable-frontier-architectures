"""test_chainmix_hooks.py RELDIR -- the CHAINMIX512 hooks inside serve128 (osel_check, run_ar, _score, health) on the CPU
FakeModel of fake_serve.py (real o200k tokenizer, vocabulary 200,064). No pack, no GPU. The chain used here is counted
from one sentence repeated 40 times, so it knows that sentence by heart and nothing else."""
import json
import math
import os
import sys
import tempfile
import unittest

REL = sys.argv.pop(1)
import fake_serve
S, M = fake_serve.load(REL)
import numpy as np
import torch
import chainmix512 as cmx
import loop512
loop512.install(M)            # serve128.load() does this for the real model (needed by the loops + chain test)

TXT ='The history of the Roman Empire is long, and the city of Rome was founded, according to tradition, in 753 BC by Romulus.'
IDS = S.ENC.encode(TXT, disallowed_special=())
STREAM = []
for _ in range(40):
    STREAM += [S.EOS] + IDS
TAB = tempfile.mkdtemp(prefix='chainmix-hooks-')
cmx.build(np.asarray(STREAM, dtype=np.int32), M.V, TAB, source='hook test', eos=S.EOS)
assert cmx.install(M.V, TAB), cmx.STATE['error']
PROMPT = 'The history of the Roman Empire is long'
PIDS = S.ENC.encode(PROMPT, disallowed_special=())
assert PIDS == IDS[:len(PIDS)], 'the prompt must tokenise to a prefix of the sentence'
GREEDY = dict(prompt=PROMPT, max_new_tokens=10, temperature=0, top_p=1.0, top_k=0, seed=1)
_N = [0]


def gen(**kw):
    _N[0] += 1
    req = dict(GREEDY, request_id='cm-%d' % _N[0])
    req.update(kw)
    ev = []
    res = S.run(dict(req), ev.append)
    return res, [e['data'] for e in ev if e.get('event') == 'stream_tok']


class ChainMixHooks(unittest.TestCase):
    def test_health_and_revision(self):
        h = S.health()
        self.assertIn('+chainmix512-v1-20261112', h['runtime_revision'])
        self.assertIn('+loop512-v1-20261112', h['runtime_revision'])
        c = h['chain_mix']
        self.assertTrue(c['supported'])
        self.assertEqual(c['tables']['vocab'], M.V)
        json.dumps(c)

    def test_plain_request_is_untouched(self):
        a, _ = gen()
        b, _ = gen(chain_mix=0)
        self.assertNotIn('chain_mix', a['stats'])
        self.assertNotIn('chain_mix', b['stats'])
        self.assertNotIn('chain_mix', a['stats']['sampling_applied'])
        self.assertEqual(a['completion'], b['completion'])

    def test_the_chain_steers_generation(self):
        plain, _ = gen()
        r, toks = gen(chain_mix=0.98)
        want = S.ENC.decode(IDS[len(PIDS):len(PIDS) + 10])
        self.assertEqual(r['completion'], want)                       # the memorised sentence continues
        self.assertNotEqual(plain['completion'], want)
        self.assertEqual(r['stats']['chain_mix']['g'], 0.98)
        self.assertEqual(r['stats']['chain_mix']['tokens'], 10)
        self.assertEqual(r['stats']['sampling_applied']['chain_mix'], 0.98)
        self.assertEqual(gen()[0]['completion'], plain['completion'])   # the next plain request is untouched
        self.assertEqual(gen(chain_mix=0.98)[0]['completion'], want)    # deterministic

    def test_raw_rows_stay_the_model_alone(self):
        p, ptoks = gen(top_logprobs=3)
        r, toks = gen(chain_mix=0.98, top_logprobs=3)
        self.assertEqual([t['raw_top'] for t in toks][:1], [t['raw_top'] for t in ptoks][:1])   # same state, same model row
        self.assertEqual(toks[0]['top'][0]['id'], IDS[len(PIDS)])                                # drawn-from row: the chain's token

    def test_sampling_with_the_mix_completes(self):
        r, toks = gen(chain_mix=0.5, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.3, max_new_tokens=12)
        self.assertEqual(r['stats']['tokens'], 12)
        self.assertEqual(r['stats']['chain_mix']['tokens'], 12)

    def test_chain_mix_together_with_loops(self):
        want = S.ENC.decode(IDS[len(PIDS):len(PIDS) + 10])
        r, toks = gen(chain_mix=0.98, loop_passes=1)                  # one extra loop over the top blocks, then the mix
        self.assertEqual(r['completion'], want)
        self.assertEqual([t['depth'] for t in toks], [240] * 10)
        self.assertEqual(r['stats']['loop']['max_depth'], 240)
        self.assertEqual(r['stats']['chain_mix']['tokens'], 10)
        a, atoks = gen(chain_mix=0.98, loop_adaptive=True, loop_tau=0.0, loop_min=1)
        self.assertEqual(a['completion'], want)
        self.assertEqual([t['depth'] for t in atoks], [160] + [80] * 9)  # the stop rule reads the model's own confidence, not the mix
        self.assertTrue(all('loop_conf' in t for t in atoks[1:]))
        plain_loop, ptoks = gen(loop_adaptive=True, loop_tau=0.0, loop_min=1)
        self.assertEqual([t['depth'] for t in ptoks], [160] + [80] * 9)  # (confidences differ: the two runs chose different tokens)
        self.assertNotIn('chain_mix', plain_loop['stats'])
        self.assertIsNone(M.loop_plan)
        s = fake_serve.score(S, text=TXT, prepend_eos=True, chain_mix=0.5, loop_passes=1)
        self.assertEqual(s['loop']['max_depth'], 240)
        self.assertEqual(s['chain_mix']['tokens'], len(IDS))
        self.assertLess(s['mean_nll'], 1.0)

    def test_validation(self):
        for bad in (1.5, -0.2, 'x', True):
            with self.assertRaises(ValueError):
                gen(chain_mix=bad)
        self.assertEqual(gen()[0]['stats']['tokens'], 10)

    def test_score_is_the_mixture_of_the_two_probabilities(self):
        plain = fake_serve.score(S, text=TXT, prepend_eos=True, per_token=True)
        mixed = fake_serve.score(S, text=TXT, prepend_eos=True, per_token=True, chain_mix=0.5)
        again = fake_serve.score(S, text=TXT, prepend_eos=True, per_token=True)
        self.assertNotIn('chain_mix', plain)
        self.assertEqual(plain['per_token'], again['per_token'])       # no state left behind
        self.assertEqual(mixed['chain_mix']['g'], 0.5)
        self.assertEqual(mixed['chain_mix']['tokens'], len(IDS))
        self.assertGreater(plain['mean_nll'], 5.0)
        self.assertLess(mixed['mean_nll'], 1.0)                        # log 2 = 0.69 is the most a 0.5 mix can lose on a memorised token
        ids = [S.EOS] + IDS
        t = cmx.STATE['tables']
        for j in (0, 1, 5, len(IDS) - 1):
            pc = float(t.next_dist(ids[j - 1] if j >= 1 else None, ids[j])[ids[j + 1]])
            ref = -math.log(0.5 * math.exp(-plain['per_token'][j]) + 0.5 * pc)
            self.assertAlmostEqual(mixed['per_token'][j], ref, places=3)

    def test_missing_tables_refuse_only_chain_requests(self):
        keep = dict(cmx.STATE)
        cmx.STATE.update(tables=None, error='test: tables removed')
        try:
            with self.assertRaises(ValueError):
                gen(chain_mix=0.5)
            self.assertEqual(gen()[0]['stats']['tokens'], 10)
            self.assertFalse(S.health()['chain_mix']['supported'])
        finally:
            cmx.STATE.update(keep)


if __name__ == '__main__':
    unittest.main(verbosity=1)
