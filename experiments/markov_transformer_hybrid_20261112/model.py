#!/usr/bin/env python3
"""model.py -- one small GPT backbone and the ways a Markov chain is combined with it.

Variants (all share the same backbone, initialisation, data order and optimiser):
  base           classical transformer (pre-LN GPT, learned positions, tied embeddings)
  mc1_poe        + first-order Markov chain (bigram counts) added to the logits:  logits + lam * log P_chain
  mc3_poe        + order-3 Markov chain (4-gram, interpolated absolute discounting) added to the logits
  mc3_mix        gated mixture: (1 - g) * P_transformer + g * P_chain, g = sigmoid(w . h) per position
  mkattn         attention applied as a 2-step Markov chain over positions: (1 - b) * A + b * A @ A, b learned per head
  mc3_poe_mk     mc3_poe and mkattn together
  mc3_poe_noloo  mc3_poe trained WITHOUT leave-one-out counts (ablation: shows what the leave-one-out step is worth)
  *_bo           the same hybrids trained with leave-one-BLOCK-out counts (cross-fitting over whole documents)
  *_aux          the same hybrid, and in training the network ALSO gets the gradient of its own chain-free loss
                 (weight `aux`). The reported loss is the hybrid's alone; only the gradient changes. Added on 12 Nov
                 after seed 3 of mc3_poe / mc3_poe_bo on TinyStories (and seed 1 at lr 4e-3) stopped improving at
                 about 2.66 from step 200-400, train and dev alike, while seeds 1-2 went on to 2.48: the counted chain
                 already supplies the word-pair statistics, so the network loses the easy gradient it normally builds
                 its first features from, and some starts never leave that plateau. On WikiText every hybrid seed
                 was flat after step 400 (3.65 -> 3.64) while the classical model kept improving.
The chain's counts are fixed (no gradient); `lam`, the gate and `b` are the only extra trained parameters.
On training windows the chain is evaluated with the variant's `cv` mode (loo / lobo / none); on held-out text always
with the full counts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {
    'base': dict(),
    'mc1_poe': dict(prior='poe', order=2),
    'mc3_poe': dict(prior='poe', order=4),
    'mc3_mix': dict(prior='mix', order=4),
    'mkattn': dict(mk=True),
    'mc3_poe_mk': dict(prior='poe', order=4, mk=True),
    'mc3_poe_noloo': dict(prior='poe', order=4, cv='none'),
    'mc3_poe_bo': dict(prior='poe', order=4, cv='lobo'),
    'mc3_mix_bo': dict(prior='mix', order=4, cv='lobo'),
    'mc1_poe_bo': dict(prior='poe', order=2, cv='lobo'),
    'mc3_poe_aux': dict(prior='poe', order=4, aux=0.5),
    'mc3_poe_bo_aux': dict(prior='poe', order=4, cv='lobo', aux=0.5),
    'mc3_poe_aux1': dict(prior='poe', order=4, aux=1.0),
}


def variant_cfg(name):
    c = dict(prior=None, order=0, mk=False, cv='loo', aux=0.0)
    c.update(VARIANTS[name])
    return c


class Block(nn.Module):
    def __init__(self, d, H, mk):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.mk = mk
        if mk:
            # sigmoid(-2) = 0.12: the walk starts mostly one-step. A raw parameter draws no random numbers.
            self.mk_beta = nn.Parameter(torch.full((H,), -2.0))

    def attend(self, x, mask):
        B, T, d = x.shape
        H = self.H
        q, k, v = self.qkv(x).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)      # each [B,H,T,dh]
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(d // H))
        att = att.masked_fill(mask[:T, :T], float('-inf')).softmax(-1)            # row-stochastic, lower triangular
        h = att @ v
        if self.mk:
            b = torch.sigmoid(self.mk_beta).view(1, H, 1, 1)
            h = (1 - b) * h + b * (att @ h)                                        # ((1-b) A + b A A) v
        return self.proj(h.transpose(1, 2).reshape(B, T, d))

    def forward(self, x, mask):
        x = x + self.attend(self.ln1(x), mask)
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V, T, d, L, H, cfg):
        super().__init__()
        self.cfg = cfg
        self.T = T
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, H, cfg['mk']) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * L))
        # extras are created after the backbone is initialised, so the backbone starts identical in every variant
        if cfg['prior'] == 'poe':
            self.lam = nn.Parameter(torch.ones(()))
        if cfg['prior'] == 'mix':
            self.gate = nn.Linear(d, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.register_buffer('mask', torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.wte(x) + self.wpe(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h, self.mask)
        h = self.lnf(h)
        return F.linear(h, self.wte.weight), h

    def extra_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n == 'lam' or n.startswith('gate') or n.endswith('mk_beta'))


def lm_loss(model, prior, x, y, train, timer=None, tpos=None):
    """Mean next-token loss (nats) of the model's actual predictive distribution, plus a few diagnostics.
    train=True evaluates the chain with the variant's cross-fitting mode (leave-one-out, leave-one-block-out, or none);
    train=False always uses the full counts. tpos = stream positions of the targets (needed for leave-one-block-out)."""
    cfg = model.cfg
    logits, h = model(x)
    V = logits.size(-1)
    info = {}
    mode = cfg['cv'] if (train and cfg['cv'] != 'none') else 'full'
    if cfg['prior'] == 'poe':
        t0 = _now(timer)
        lp = prior.dense_logprob(x, y if mode != 'full' else None, mode=mode, tpos=tpos)
        _add(timer, t0)
        loss = F.cross_entropy((logits + model.lam * lp).view(-1, V), y.reshape(-1))
        info['lam'] = float(model.lam.detach())
        if cfg.get('aux', 0.0) > 0:
            net = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
            info['net_alone'] = float(net.detach())
            if train:
                # value unchanged (the two aux terms cancel), gradient = hybrid loss + aux * network-alone loss
                loss = loss + cfg['aux'] * (net - net.detach())
    elif cfg['prior'] == 'mix':
        t0 = _now(timer)
        lpm = prior.target_logprob(x, y, mode=mode, tpos=tpos).reshape(-1).float()
        _add(timer, t0)
        lpt = -F.cross_entropy(logits.view(-1, V), y.reshape(-1), reduction='none')
        gl = model.gate(h).reshape(-1)
        loss = -torch.logaddexp(F.logsigmoid(-gl) + lpt, F.logsigmoid(gl) + lpm).mean()
        info['gate'] = float(torch.sigmoid(gl.detach()).mean())
    else:
        loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
    if cfg['mk']:
        info['mk_b'] = float(torch.sigmoid(torch.stack([b.mk_beta.detach() for b in model.blocks])).mean())
    return loss, info


@torch.no_grad()
def predictive_logprobs(model, prior, x):
    """[B,T,V] log of the distribution the model actually predicts from (full chain counts, as at test time)."""
    cfg = model.cfg
    logits, h = model(x)
    if cfg['prior'] == 'poe':
        return F.log_softmax(logits + model.lam * prior.dense_logprob(x), -1)
    if cfg['prior'] == 'mix':
        gl = model.gate(h)
        return torch.logaddexp(F.logsigmoid(-gl) + F.log_softmax(logits, -1), F.logsigmoid(gl) + prior.dense_logprob(x))
    return F.log_softmax(logits, -1)


def _now(timer):
    if timer is None:
        return None
    import time
    return time.process_time()


def _add(timer, t0):
    if timer is not None:
        import time
        timer['prior_cpu'] = timer.get('prior_cpu', 0.0) + time.process_time() - t0
