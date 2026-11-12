"""unigram_bias -- control for markov_corpus_prior: ONLY the unigram term, lam_u * log P1(v), on every readout.

The tied binary table has a per-token scale but no additive per-token term, so the base model has to spend network
capacity and steps on token frequency. This candidate hands it the frequencies as a fixed vector (counted on the same
text markov_corpus_prior counts: train-stream tokens that no training step of the run sees) with one trained weight.
What markov_corpus_prior gains beyond this candidate is what the n-gram context adds; what this candidate gains over
base is what a frequency bias alone is worth.
"""
import importlib.util
import os
import sys

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markov_corpus_prior.py')
_s = importlib.util.spec_from_file_location('markov_corpus_prior_core', _p)
core = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = core
_s.loader.exec_module(core)

META = dict(id='unigram_bias', title='Token-frequency bias on the readout (control for the Markov chain candidate)', slot='hybrid',
            one_line='Every readout adds lam_u * log P1(v), the counted unigram; 10 fp parameters of which one is used, no extra latents.',
            storage='one vector of V floats on the host (800 KB at the o200k vocabulary)')


post_step = core.post_step      # keeps the step count and prints GPU memory at a few steps


def patch(Block, Model, cfg):
    return core.patch(Block, Model, cfg, unigram=True, lifts=False)
