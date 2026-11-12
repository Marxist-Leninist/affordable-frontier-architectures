#!/usr/bin/env python3
"""tests/test_all.py -- correctness checks for the Markov chain, the leave-one-out counts and the model variants.
Plain asserts, no pytest:  python3 tests/test_all.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markov import MarkovPrior, brute_dist, build_block_tables, build_tables  # noqa: E402
from model import GPT, VARIANTS, lm_loss, predictive_logprobs, variant_cfg  # noqa: E402

V, ORDER, T = 12, 4, 20
torch.set_num_threads(1)


def make_stream(n, seed):
    """a noisy second-order source: contexts repeat, some are rare, some never appear"""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, V, size=(V, V, 3))
    ids = np.zeros(n, dtype=np.int64)
    ids[0], ids[1] = rng.integers(V), rng.integers(V)
    for i in range(2, n):
        ids[i] = rng.integers(V) if rng.random() < 0.15 else A[ids[i - 2], ids[i - 1], rng.integers(3)]
    return ids


STREAM = make_stream(4700, 0)
TRAIN, HELD = STREAM[:4000], STREAM[4000:]
TABLES = build_tables(TRAIN, V, ORDER)
# irregular blocks ("documents" of different lengths), one of them tiny, boundaries inside test windows
BOUNDS = [0, 7, 180, 505, 512, 900, 1310, 1800, 2222, 2600, 3105, 3500, 3990]
BLK = np.zeros(len(TRAIN), dtype=np.int64)
for _i, _s in enumerate(BOUNDS):
    BLK[_s:] = _i
BTABLES = build_block_tables(TRAIN, V, ORDER, BLK, TABLES)


def windows(ids, starts):
    idx = torch.tensor(starts)[:, None] + torch.arange(T)[None, :]
    t = torch.from_numpy(ids.astype(np.int64))
    return t[idx], t[idx + 1]


def windows_pos(ids, starts):
    idx = torch.tensor(starts)[:, None] + torch.arange(T)[None, :]
    t = torch.from_numpy(ids.astype(np.int64))
    return t[idx], t[idx + 1], idx + 1


def test_block_out_matches_brute_force():
    starts = [0, 170, 495, 500, 893, 2215, 3975, len(TRAIN) - T - 1]          # windows that straddle block boundaries
    x, y, tpos = windows_pos(TRAIN, starts)
    for order in (2, 3, 4):
        pr = MarkovPrior(TABLES, order, btables=BTABLES)
        P = pr.dense_logprob(x, y, mode='lobo', tpos=tpos).exp().double().numpy()
        tl = pr.target_logprob(x, y, mode='lobo', tpos=tpos).exp().numpy()
        worst, worst_t = 0.0, 0.0
        for b in range(x.shape[0]):
            for j in range(T):
                ctx = x[b, :j + 1].tolist()[-(order - 1):]
                ref = brute_dist(TRAIN, V, order, pr.D, pr.delta, ctx, drop=(BLK == BLK[int(tpos[b, j])]))
                worst = max(worst, float(np.abs(P[b, j] - ref).max()))
                worst_t = max(worst_t, abs(float(tl[b, j]) - ref[int(y[b, j])]))
        assert worst < 3e-6 and worst_t < 1e-9, (order, worst, worst_t)
        assert np.abs(P.sum(-1) - 1).max() < 1e-5
        full = float(pr.target_logprob(x, y).mean())
        loo = float(pr.target_logprob(x, y, loo=True).mean())
        lobo = float(pr.target_logprob(x, y, mode='lobo', tpos=tpos).mean())
        print(f'PASS leave-one-block-out == brute force with the whole block removed, order {order}, max abs diff '
              f'{worst:.1e} (target entry {worst_t:.1e}); train loss full {-full:.3f}, loo {-loo:.3f}, block-out {-lobo:.3f}')
    # without y the dense distribution is still right (no target overwrite needed)
    pr = MarkovPrior(TABLES, 4, btables=BTABLES)
    a = pr.dense_logprob(x, None, mode='lobo', tpos=tpos).exp()
    b = pr.dense_logprob(x, y, mode='lobo', tpos=tpos).exp()
    assert (a - b).abs().max() < 3e-6
    print('PASS block-out dense distribution does not depend on being given the targets')


def test_chain_matches_brute_force():
    x, y = windows(HELD, [0, 37, 200, 411])
    for order in (2, 3, 4):
        pr = MarkovPrior(TABLES, order)
        P = pr.dense_logprob(x).exp().double().numpy()
        worst = 0.0
        for b in range(x.shape[0]):
            for j in range(T):
                ctx = x[b, :j + 1].tolist()[-(order - 1):]
                ref = brute_dist(TRAIN, V, order, pr.D, pr.delta, ctx)
                worst = max(worst, float(np.abs(P[b, j] - ref).max()))
                assert abs(ref.sum() - 1) < 1e-9
        assert worst < 3e-6, (order, worst)
        assert np.abs(P.sum(-1) - 1).max() < 1e-5
        print(f'PASS chain == brute force (held-out windows), order {order}, max abs diff {worst:.1e}')


def test_leave_one_out_matches_brute_force():
    rng = np.random.default_rng(1)
    starts = rng.integers(0, len(TRAIN) - T - 1, size=5).tolist() + [0]      # include the very start of the stream
    x, y = windows(TRAIN, starts)
    for order in (2, 3, 4):
        pr = MarkovPrior(TABLES, order)
        P = pr.dense_logprob(x, y, loo=True).exp().double().numpy()
        worst = 0.0
        for b in range(x.shape[0]):
            for j in range(T):
                ctx = x[b, :j + 1].tolist()[-(order - 1):]
                ref = brute_dist(TRAIN, V, order, pr.D, pr.delta, ctx, loo_y=int(y[b, j]))
                worst = max(worst, float(np.abs(P[b, j] - ref).max()))
        assert worst < 3e-6, (order, worst)
        assert np.abs(P.sum(-1) - 1).max() < 1e-5
        full = pr.target_logprob(x, y).mean()
        loo = pr.target_logprob(x, y, loo=True).mean()
        assert loo < full, 'removing a position\'s own event must lower its probability on average'
        print(f'PASS leave-one-out == brute force with the event removed, order {order}, max abs diff {worst:.1e}; '
              f'train loss with own event {-full:.3f}, without {-loo:.3f}')


def test_target_equals_dense():
    for ids, loo in ((HELD, False), (TRAIN, True), (TRAIN, False)):
        x, y = windows(ids, [3, 90, 300])
        for order in (2, 3, 4):
            pr = MarkovPrior(TABLES, order)
            d = pr.dense_logprob(x, y, loo=loo).gather(-1, y.unsqueeze(-1)).squeeze(-1).double()
            t = pr.target_logprob(x, y, loo=loo)
            assert (d - t).abs().max() < 2e-5, (order, loo, float((d - t).abs().max()))
    print('PASS target-only probabilities == the target entry of the dense distribution')


def test_chain_is_causal():
    x, y = windows(TRAIN, [10, 500, 900])
    t0 = 11
    x2 = x.clone()
    x2[:, t0:] = (x[:, t0:] + 5) % V
    y2 = y.clone()
    y2[:, t0 - 1:] = (y[:, t0 - 1:] + 5) % V                        # y[j] is the token at j+1
    pr = MarkovPrior(TABLES, ORDER)
    a, b = pr.dense_logprob(x), pr.dense_logprob(x2)
    assert torch.equal(a[:, :t0], b[:, :t0]) and not torch.equal(a[:, t0:], b[:, t0:])
    a, b = pr.dense_logprob(x, y, loo=True), pr.dense_logprob(x2, y2, loo=True)
    assert torch.equal(a[:, :t0 - 1], b[:, :t0 - 1])
    print('PASS chain output at a position ignores every later token (full and leave-one-out counts)')


def tiny(name, seed=3):
    cfg = variant_cfg(name)
    torch.manual_seed(seed)
    m = GPT(V, T, 16, 2, 2, cfg)
    pr = MarkovPrior(TABLES, cfg['order'], btables=BTABLES if cfg['cv'] == 'lobo' else None) if cfg['prior'] else None
    return m, pr


def test_models_causal_normalised_and_loss_consistent():
    x, y = windows(HELD, [0, 100, 250])
    t0 = 9
    x2 = x.clone()
    x2[:, t0:] = (x[:, t0:] + 3) % V
    for name in VARIANTS:
        m, pr = tiny(name)
        with torch.no_grad():                                        # make the extras non-trivial
            for p in m.parameters():
                p.add_(0.3 * torch.randn_like(p))
        m.eval()
        lp, lp2 = predictive_logprobs(m, pr, x), predictive_logprobs(m, pr, x2)
        assert (lp.exp().sum(-1) - 1).abs().max() < 1e-4, name
        assert torch.allclose(lp[:, :t0], lp2[:, :t0], atol=1e-5), name
        assert not torch.allclose(lp[:, t0:], lp2[:, t0:], atol=1e-5), name
        with torch.no_grad():
            loss, _ = lm_loss(m, pr, x, y, train=False)
        ref = -lp.gather(-1, y.unsqueeze(-1)).mean()
        assert abs(float(loss) - float(ref)) < 2e-5, (name, float(loss), float(ref))
    print('PASS every variant: predictions sum to 1, ignore later tokens, and the reported loss is the loss of '
          'that same predictive distribution')


def test_same_backbone_start():
    ref = None
    for name in VARIANTS:
        m, _ = tiny(name, seed=5)
        sd = {k: v for k, v in m.state_dict().items() if k != 'lam' and not k.startswith('gate') and not k.endswith('mk_beta')}
        if ref is None:
            ref = sd
        else:
            assert sd.keys() == ref.keys() and all(torch.equal(sd[k], ref[k]) for k in ref), name
    print('PASS the backbone starts from identical weights in every variant (same seed)')


def test_markov_attention_reduces_to_baseline():
    x, _ = windows(HELD, [0, 50])
    base, _ = tiny('base')
    mk, _ = tiny('mkattn')
    with torch.no_grad():
        for b in mk.blocks:
            b.mk_beta.fill_(-1e9)                                     # b = 0: one-step attention only
        assert torch.allclose(base(x)[0], mk(x)[0], atol=1e-6)
        for b in mk.blocks:
            b.mk_beta.fill_(0.0)
        assert not torch.allclose(base(x)[0], mk(x)[0], atol=1e-6)
    print('PASS 2-step Markov attention equals the baseline when the second step is switched off')


def test_gradients():
    x, y, tpos = windows_pos(TRAIN, [5, 700, 1500])
    for name in VARIANTS:
        m, pr = tiny(name)
        loss, _ = lm_loss(m, pr, x, y, train=True, tpos=tpos)
        loss.backward()
        for n, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (name, n)
        if 'lam' in dict(m.named_parameters()):
            assert m.lam.grad.abs() > 0
        if name == 'mc3_mix':
            assert m.gate.bias.grad.abs() > 0
        if variant_cfg(name)['mk']:
            assert all(b.mk_beta.grad.abs().sum() > 0 for b in m.blocks)
    print('PASS every variant trains: finite gradients, and the extra parameters receive gradient')


def test_real_data_tables_are_train_only():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if not os.path.isdir(root):
        print('SKIP real-data checks (no data folder here)')
        return
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, 'meta.json')):
            continue
        tr, dv, te = (np.load(os.path.join(d, f + '.npy')) for f in ('train', 'dev', 'test'))
        tb = np.load(os.path.join(d, 'markov.npz'))
        Vd = int(tb['V'])
        assert int(tb['N']) == len(tr)
        assert np.array_equal(tb['C1'], np.bincount(tr.astype(np.int64), minlength=Vd))
        assert int(tb['C2'].sum()) == len(tr) - 1
        n = 4000
        assert not np.array_equal(tr[:n], dv[:n]) and not np.array_equal(tr[:n], te[:n]) and not np.array_equal(dv[:n], te[:n])
        # a rebuilt table from train alone must reproduce the stored one
        tb2 = build_tables(tr[:200000], Vd, int(tb['order']))
        assert tb2['C2'].sum() == 199999
        pr = MarkovPrior(tb, 4)
        t = torch.from_numpy(tr[:20 * 128 + 1].astype(np.int64))
        idx = torch.arange(20)[:, None] * 128 + torch.arange(128)[None, :]
        x, y = t[idx], t[idx + 1]
        P = pr.dense_logprob(x, y, loo=True).exp()
        assert (P.sum(-1) - 1).abs().max() < 2e-4, float((P.sum(-1) - 1).abs().max())
        P = pr.dense_logprob(x).exp()
        assert (P.sum(-1) - 1).abs().max() < 2e-4
        d1 = pr.dense_logprob(x, y, loo=True).gather(-1, y.unsqueeze(-1)).squeeze(-1).double()
        assert (d1 - pr.target_logprob(x, y, loo=True)).abs().max() < 1e-3
        bpath = os.path.join(d, 'blocks_cache')
        if os.path.isdir(bpath):
            prb = MarkovPrior(tb, 4, btables=bpath)
            assert int(prb.C1b.sum()) == len(tr) and len(prb.blk) == len(tr)
            P = prb.dense_logprob(x, y, mode='lobo', tpos=idx + 1)
            assert (P.exp().sum(-1) - 1).abs().max() < 2e-4, float((P.exp().sum(-1) - 1).abs().max())
            d2 = P.gather(-1, y.unsqueeze(-1)).squeeze(-1).double()
            assert (d2 - prb.target_logprob(x, y, mode='lobo', tpos=idx + 1)).abs().max() < 1e-3
            print(f'PASS real data {name}: block-out rows sum to 1 at real size ({int(prb.blk.max()) + 1} blocks, memory-mapped)')
        print(f'PASS real data {name}: counts come from the train stream only (N, unigram and bigram totals match), '
              'splits differ, real-size chain rows sum to 1')


if __name__ == '__main__':
    test_chain_matches_brute_force()
    test_leave_one_out_matches_brute_force()
    test_block_out_matches_brute_force()
    test_target_equals_dense()
    test_chain_is_causal()
    test_models_causal_normalised_and_loss_consistent()
    test_same_backbone_start()
    test_markov_attention_reduces_to_baseline()
    test_gradients()
    test_real_data_tables_are_train_only()
    print('ALL TESTS PASSED')
