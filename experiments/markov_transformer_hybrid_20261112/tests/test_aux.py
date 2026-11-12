"""test_aux.py -- the *_aux variants: the reported loss is exactly the hybrid's, and the gradient is the hybrid's plus
aux times the gradient of the network's own chain-free loss. Uses the real TinyStories chain counts (run from the
experiment folder on GETH):  python3 tests/test_aux.py"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from markov import MarkovPrior  # noqa: E402
from model import GPT, lm_loss, variant_cfg  # noqa: E402

torch.manual_seed(0)
meta_V = int(np.load('data/tinystories/train.npy', mmap_mode='r').max()) + 1
tr = torch.from_numpy(np.load('data/tinystories/train.npy')[:200000].astype(np.int64))
prior = MarkovPrior(np.load('data/tinystories/markov.npz'), 4)
V = prior.V if hasattr(prior, 'V') else meta_V
T, B = 32, 4
ix = torch.randint(0, len(tr) - T - 1, (B,))
x = torch.stack([tr[i:i + T] for i in ix])
y = torch.stack([tr[i + 1:i + T + 1] for i in ix])
tpos = ix[:, None] + torch.arange(T)[None, :] + 1
ok = True
for name, aux in (('mc3_poe_aux', 0.5), ('mc3_poe_aux1', 1.0)):
    torch.manual_seed(1)
    m_aux = GPT(V, T, 32, 1, 2, variant_cfg(name))
    m_ref = copy.deepcopy(m_aux)
    m_ref.cfg = dict(m_ref.cfg, aux=0.0)
    l_aux, info = lm_loss(m_aux, prior, x, y, train=True, tpos=tpos)
    l_ref, _ = lm_loss(m_ref, prior, x, y, train=True, tpos=tpos)
    l_aux.backward()
    logits, _ = m_ref(x)
    net = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
    (l_ref + aux * net).backward()                                   # the gradient the aux variant should have
    same_value = abs(float(l_aux) - float(l_ref)) < 1e-6
    worst = max(float((p1.grad - p2.grad).abs().max()) for p1, p2 in zip(m_aux.parameters(), m_ref.parameters()))
    good = same_value and worst < 1e-6 and 'net_alone' in info
    ok = ok and good
    print('%s %s: reported loss %.6f vs hybrid %.6f, worst gradient difference %.2e, network alone %.4f' % (
        'PASS' if good else 'FAIL', name, float(l_aux), float(l_ref), worst, info.get('net_alone', float('nan'))))
    l_eval, info_e = lm_loss(m_aux, prior, x, y, train=False)
    l_eval_ref, _ = lm_loss(m_ref, prior, x, y, train=False)
    good = abs(float(l_eval) - float(l_eval_ref)) < 1e-6
    ok = ok and good
    print('%s %s at evaluation: %.6f vs %.6f (the aux term never touches what is reported)' % ('PASS' if good else 'FAIL', name, float(l_eval), float(l_eval_ref)))
print('AUX-OK' if ok else 'AUX-FAILED')
