"""markov_corpus_late -- the counted Markov chain of markov_corpus_prior, added to a model that is ALREADY TRAINING.

Nothing is added during the first 600 steps or the step-600 evaluation: until then this is an ordinary `base` run of
the same seed (its step-150/300/600 numbers should equal base's, up to GPU nondeterminism). Then the n-gram lifts fade
in linearly over the next 100 steps and stay. No unigram term: a network that has trained for
600 steps already holds the token frequencies, and the lift is defined relative to them, so nothing is counted twice
at that level. This is the simulator's version of the question "can the chain be bolted onto the live model, or does a
model have to be born with it": compare its step-1200 numbers with base (same steps) and with markov_corpus_prior.
The counts, the cut-out of every training window and the slot rules are those of markov_corpus_prior.py (same folder).
"""
import importlib.util
import os
import sys

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markov_corpus_prior.py')
_s = importlib.util.spec_from_file_location('markov_corpus_prior_core', _p)
core = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = core
_s.loader.exec_module(core)

START, RAMP = (2, 2) if os.environ.get('AGILLM_SIM_SMOKE') == '1' else (600, 100)   # the 5-step smoke test must reach the chain too
META = dict(core.META, id='markov_corpus_late',
            title='Counted Markov chain switched on at step 600 of an ordinary run (n-gram lifts, faded in over 100 steps)',
            one_line='An ordinary base run for 600 steps; then lam * log(P_chain / back-off) from n-gram and skip-gram counts is faded '
                     'into the logits over 100 steps. Same counts as markov_corpus_prior; no unigram term.')
post_step = core.post_step


def patch(Block, Model, cfg):
    return core.patch(Block, Model, cfg, unigram=False, lifts=True, start_step=START, ramp_steps=RAMP)
