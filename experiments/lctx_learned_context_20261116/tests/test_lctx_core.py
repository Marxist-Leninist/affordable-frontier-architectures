"""tests for lctx_core.py v2 (CPU, a few seconds).  python3 -I test_lctx_core.py <dir of lctx_core.py>"""
import math
import os
import sys
import unittest

sys.path.insert(0, sys.argv.pop(1) if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint, CheckpointError  # noqa: E402

import lctx_core as L  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def rnd(*s, dtype=torch.float64, seed=None, scale=1.0):
    g = torch.Generator().manual_seed(seed if seed is not None else 1234)
    return (torch.randn(*s, generator=g, dtype=torch.float64) * scale).to(dtype)


def qkv(B=2, H=4, T=37, E=16, dtype=torch.float64, seed=0):
    q = F.normalize(rnd(B, H, T, E, seed=seed), dim=-1) * math.sqrt(E)
    k = F.normalize(rnd(B, H, T, E, seed=seed + 1), dim=-1)
    v = rnd(B, H, T, E, seed=seed + 2)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def params(H=4, E=16, lo=-3.0, hi=0.0, wscale=0.5, seed=5, dtype=torch.float64):
    r = (lo + (hi - lo) * torch.rand(H, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)).to(dtype)
    w = rnd(H, E, seed=seed + 1, scale=wscale).to(dtype)
    return r, w


class Core(unittest.TestCase):
    def test_exact_vs_reference(self):
        for scale in (1.0, 0.25):
            for amax in (6.0, 0.3):
                q, k, v = qkv()
                r, w = params()
                a = L.lctx_attention(q, k, v, r, w, scale, amax=amax)
                ref = L.ref_attention(q, k, v, r, w, scale, amax=amax)
                self.assertLess(float((a - ref).abs().max()), 1e-11, (scale, amax))

    def test_identity_at_zero_rate(self):
        for dt, tol in ((torch.float64, 1e-12), (torch.float32, 2e-6)):
            q, k, v = qkv(dtype=dt)
            r = torch.full((4,), float('-inf'), dtype=dt)
            w = torch.zeros(4, 16, dtype=dt)
            a = L.lctx_attention(q, k, v, r, w, 1.0)
            b = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            self.assertLess(float((a - b).abs().max()), tol, dt)

    def test_gradients_vs_reference(self):
        q, k, v = qkv()
        r, w = params()
        wt = rnd(2, 4, 37, 16, seed=99)
        outs = []
        for fn in (L.lctx_attention, L.ref_attention):
            ts = [t.clone().requires_grad_() for t in (q, k, v, r, w)]
            (fn(*ts, 0.7, amax=6.0) * wt).sum().backward()
            outs.append([t.grad for t in ts])
        for name, ga, gb in zip('q k v r w'.split(), *outs):
            self.assertIsNotNone(ga, name)
            self.assertLess(float((ga - gb).abs().max()), 1e-9, name)
        self.assertGreater(float(outs[0][3].abs().max()), 1e-6)  # the rate really gets a gradient

    def test_sink_receives_the_same_gradients(self):
        q, k, v = qkv()
        r, w = params()
        got = {}

        def sink(dr, dw):
            got['r'], got['w'] = dr, dw

        wt = rnd(2, 4, 37, 16, seed=99)
        ts = [t.clone().requires_grad_() for t in (q, k, v)]
        (L.lctx_attention(*ts, r, w, 0.7, sink=sink) * wt).sum().backward()
        ref = [t.clone().requires_grad_() for t in (r, w)]
        (L.ref_attention(q, k, v, *ref, 0.7) * wt).sum().backward()
        for key, t in zip('rw', ref):
            self.assertLess(float((got[key] - t.grad).abs().max()), 1e-9, key)

    def test_chunking_does_not_change_anything(self):
        q, k, v = qkv(H=6)
        r, w = params(H=6)
        wt = rnd(2, 6, 37, 16, seed=98)
        base, bg = None, None
        for hc, hg in ((None, None), (1, 1), (2, 4), (4, 2), (5, 3)):
            ts = [t.clone().requires_grad_() for t in (q, k, v, r, w)]
            o = L.lctx_attention(*ts, 1.0, hchunk=hc, hgroup=hg)
            (o * wt).sum().backward()
            gs = [t.grad for t in ts]
            if base is None:
                base, bg = o.detach(), gs
                continue
            self.assertLess(float((o.detach() - base).abs().max()), 1e-12, (hc, hg))
            for a, b in zip(gs, bg):
                self.assertLess(float((a - b).abs().max()), 1e-11, (hc, hg))

    def test_clamped_rates_have_no_gradient(self):
        q, k, v = qkv()
        r = torch.full((4,), 50.0, dtype=torch.float64, requires_grad=True)
        w = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
        o = L.lctx_attention(q, k, v, r, w, 1.0, amax=6.0)
        ref = L.ref_attention(q, k, v, r.detach(), w.detach(), 1.0, amax=6.0)
        self.assertLess(float((o - ref).abs().max()), 1e-11)
        o.sum().backward()
        self.assertEqual(float(r.grad.abs().max()), 0.0)

    def test_a_token_can_forget_everything_at_a_long_span(self):
        # a head with a 1e6-token span; w aligned with one 'break' token's value: that token alone reaches amax
        H, E, T = 1, 16, 12
        v = rnd(1, H, T, E, seed=3)
        brk = 7
        u = F.normalize(v[0, 0, brk], dim=0)
        r = torch.full((H,), L.R0, dtype=torch.float64)
        w = (30.0 * u)[None]
        a = L.forget_rates(v, r, w, 6.0)[0, 0]
        self.assertAlmostEqual(float(a[brk]), 6.0, places=6)
        others = torch.cat([a[:brk], a[brk + 1:]])
        self.assertLess(float(others.max()), 0.05)  # ordinary tokens stay ~1000x below the break token's 6 nats

    def test_fp32_large_bias(self):
        q, k, v = qkv(B=1, H=2, T=512, E=32, dtype=torch.float32, seed=3)
        r = torch.tensor([float(L.r_from_rate(3.0)), float(L.r_from_rate(0.02))])
        w = torch.zeros(2, 32)
        a = L.lctx_attention(q, k, v, r, w, 11.3137 / math.sqrt(32), amax=6.0)
        ref = L.ref_attention(q.double(), k.double(), v.double(), r.double(), w.double(), 11.3137 / math.sqrt(32),
                              amax=6.0)
        self.assertLess(float((a.double() - ref).abs().max()), 2e-3)

    def test_pruning_bound_and_decode_cache(self):
        B, H, T, E = 1, 3, 96, 8
        q, k, v = qkv(B, H, T, E, seed=11)
        q = q * 11.3137 / math.sqrt(E)  # |logit| up to 11.31 like the 888B
        r = torch.tensor([float(L.r_from_rate(0.6)), float(L.r_from_rate(0.05)), float('-inf')], dtype=torch.float64)
        w = rnd(H, E, seed=12, scale=0.3)
        a = L.forget_rates(v, r, w, 6.0)
        kb = torch.cumsum(a, -1)
        full = L.ref_attention(q, k, v, r, w, 1.0)
        theta = 40.0
        pr, keep = L.pruned_attention(q, k, v, kb, 1.0, theta)
        self.assertLess(float((pr - full).abs().max()), 1e-6)
        dropped = (~keep & torch.ones(T, T, dtype=torch.bool).tril()).sum(dim=(-1, -2))[0]
        self.assertGreater(int(dropped[0]), 0)       # the fast head forgets
        self.assertEqual(int(dropped[2]), 0)         # rate 0: keeps everything
        for th, tol in ((float('inf'), 1e-12), (theta, 1e-6)):
            cache = L.DecodeCache(H, th)
            outs = [cache.step(q[0, :, t], k[0, :, t], v[0, :, t], a[0, :, t], 1.0) for t in range(T)]
            dec = torch.stack(outs, dim=1)
            self.assertLess(float((dec - full[0]).abs().max()), tol, th)
            if th == theta:
                lens = cache.cache_len()
                amin = float(a[0, 0].min())
                self.assertLessEqual(lens[0], int(theta / amin) + 1)
                self.assertLess(lens[0], T - 10)
                self.assertEqual(lens[2], T)

    def test_safe_gate_drops_only_the_gate_gradient(self):
        q, k, v = qkv()
        r, w = params()

        def bad_sink(*a):
            raise RuntimeError('sink failed')

        ts = [t.clone().requires_grad_() for t in (q, k, v)]
        L.LctxAttn.SAFE_GATE = True
        try:
            before = L.LctxAttn.gate_errors
            L.lctx_attention(*ts, r, w, 1.0, sink=bad_sink).sum().backward()
            self.assertEqual(L.LctxAttn.gate_errors, before + 1)
        finally:
            L.LctxAttn.SAFE_GATE = False
        ref = [t.clone().requires_grad_() for t in (q, k, v)]
        L.lctx_attention(*ref, r, w, 1.0, sink=lambda *a: None).sum().backward()
        for a, b in zip(ts, ref):
            self.assertLess(float((a.grad - b.grad).abs().max()), 1e-12)

    def test_backward_group_retry(self):
        q, k, v = qkv(H=4)
        r, w = params(H=4)
        calls = {'n': 0}

        def flaky(*a, **kw):
            calls['n'] += 1
            if calls['n'] == 3:  # forward = calls 1, 2; the first group's backward redo fails once (like an OOM)
                raise RuntimeError('simulated out of memory')
            return F.scaled_dot_product_attention(*a, **kw)

        ts = [t.clone().requires_grad_() for t in (q, k, v)]
        before = L.LctxAttn.retries
        L.lctx_attention(*ts, r, w, 1.0, hgroup=2, sdpa=flaky).sum().backward()
        ref = [t.clone().requires_grad_() for t in (q, k, v)]
        L.lctx_attention(*ref, r, w, 1.0).sum().backward()
        self.assertEqual(L.LctxAttn.retries, before + 1)
        for a, b in zip(ts, ref):
            self.assertLess(float((a.grad - b.grad).abs().max()), 1e-11)

    def test_state_optimiser_and_v1_migration(self):
        st = L.LctxState(2, 3, 4)
        k = L.clean_knobs({'lr_rho': 0.1})
        r0 = st.r.clone()
        for _ in range(5):
            st.add(0, torch.tensor([-1.0, 1.0, 0.0]), torch.ones(3, 4))
            st.step(k)
        self.assertGreater(float(st.r[0, 0]), float(r0[0, 0]) + 0.3)  # negative gradient: the rate grows
        self.assertLess(float(st.r[0, 1]), float(r0[0, 1]) - 0.3)
        self.assertEqual(float(st.r[1].sub(r0[1]).abs().max()), 0.0)  # no gradient, no step
        self.assertLess(float(st.w[0].max()), 0.0)  # w moves from the first update on (no v1 rate gate)
        st.add(1, torch.tensor([float('nan'), 0, 0]), torch.zeros(3, 4))
        self.assertEqual(st.step(k)['bad'], 1)
        r1 = st.r[1].clone()
        k2 = L.clean_knobs({'span_cost': 1.0, 'lr_rho': 0.1})
        for _ in range(3):
            st.add(1, torch.zeros(3), torch.zeros(3, 4))
            st.step(k2)
        self.assertTrue(bool((st.r[1] > r1 + 0.2).all()))  # a price on context shortens every span
        st2 = L.LctxState(2, 3, 4)
        st2.load_state_dict(st.state_dict())
        self.assertTrue(torch.equal(st2.r, st.r) and torch.equal(st2.w, st.w))
        self.assertEqual(L.clean_knobs({'lr_rho': 99, 'amax': -1, 'freeze': 'yes', 'ddep_from': 1}), L.DEFAULT_KNOBS)
        # v1 state: a = exp(rho) * 2 sigmoid(w.u + c) with w = c = 0 converts exactly
        rho = torch.tensor([[math.log(1e-6), math.log(3e-3), math.log(0.4)], [math.log(2e-5), -9.0, -1.0]])
        v1 = dict(L=2, H=3, E=4, rho=rho, w=torch.zeros(2, 3, 4), c=torch.zeros(2, 3),
                  m=dict(rho=torch.full((2, 3), 0.5, dtype=torch.float64), w=torch.zeros(2, 3, 4, dtype=torch.float64),
                         c=torch.zeros(2, 3, dtype=torch.float64)),
                  v=dict(rho=torch.full((2, 3), 0.25, dtype=torch.float64), w=torch.zeros(2, 3, 4, dtype=torch.float64),
                         c=torch.zeros(2, 3, dtype=torch.float64)), t=torch.tensor([3, 4]), updates=4)
        st3 = L.LctxState(2, 3, 4)
        self.assertIsNone(st3.load_state_dict(v1))
        self.assertLess(float((F.softplus(st3.r.double()) - rho.double().exp()).abs().div(rho.double().exp()).max()), 1e-5)
        self.assertEqual(float(st3.m['r'][0, 0]), 0.5)
        self.assertEqual(st3.updates, 4)
        v1['c'][0, 1] = 0.3
        self.assertIn('reset', L.LctxState(2, 3, 4).load_state_dict(v1))


class Checkpointed(unittest.TestCase):
    """the trainer runs every block under torch.utils.checkpoint(use_reentrant=False): each saved tensor may be unpacked
    once per backward (the 2026-11-10 incident), and a recompute must save exactly what the forward saved (the
    2026-11-16 review: a failure inside the attention after the gate had saved its tensors).  The bad variant shows the
    test can see the double-unpack failure."""

    def block(self, attn):
        torch.manual_seed(0)
        E, H, D = 8, 4, 32
        Wq, Wk, Wv = (torch.randn(D, D, dtype=torch.float64) / math.sqrt(D) for _ in range(3))
        r, w = params(H=H, E=E)

        def f(x):
            B, T, _ = x.shape
            q = (x @ Wq).view(B, T, H, E).transpose(1, 2)
            k = (x @ Wk).view(B, T, H, E).transpose(1, 2)
            v = (x @ Wv).view(B, T, H, E).transpose(1, 2)
            q, k = F.normalize(q, dim=-1) * math.sqrt(E), F.normalize(k, dim=-1)
            return attn(q, k, v, r, w).transpose(1, 2).reshape(B, T, D)
        return f

    def run_pair(self, attn, attn_ck=None):
        x0 = rnd(2, 21, 32, seed=7)
        x1 = x0.clone().requires_grad_()
        (self.block(attn)(x1) ** 2).sum().backward()
        x2 = x0.clone().requires_grad_()
        (checkpoint(self.block(attn_ck or attn), x2, use_reentrant=False) ** 2).sum().backward()
        return x1.grad, x2.grad

    def test_good_function_under_checkpoint(self):
        got = []
        g1, g2 = self.run_pair(lambda q, k, v, r, w: L.lctx_attention(q, k, v, r, w, 1.0, sink=lambda dr, dw: got.append(dr),
                                                                       hchunk=2, hgroup=2))
        self.assertLess(float((g1 - g2).abs().max()), 1e-12)
        self.assertEqual(len(got), 2)
        self.assertLess(float((got[0] - got[1]).abs().max()), 1e-12)

    def test_bad_function_is_caught(self):
        class Bad(L.LctxAttn):
            @staticmethod
            def backward(ctx, dout):
                ctx.saved_tensors  # the probe's mistake: a second unpack
                return L.LctxAttn.backward(ctx, dout)
        with self.assertRaises(CheckpointError):
            self.run_pair(lambda q, k, v, r, w: Bad.apply(q, k, v, r, w, 6.0, 1.0, lambda *a: None, 2, 2, None))

    def glue_like(self, fail_at):
        """the trainer glue's choices: forward failure -> the original attention for the whole step (forward AND
        recompute); recompute failure -> the same Function again in smaller pieces"""
        state = {'n': 0, 'fallback': False}

        def sdpa(*a, **kw):
            state['n'] += 1
            if state['n'] in fail_at:
                raise RuntimeError('simulated out of memory at sdpa call %d' % state['n'])
            return F.scaled_dot_product_attention(*a, **kw)

        def attn(q, k, v, r, w):
            in_bwd = torch._C._current_graph_task_id() != -1
            if state['fallback']:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            if not in_bwd:
                try:
                    return L.lctx_attention(q, k, v, r, w, 1.0, sink=lambda *a: None, hgroup=2, sdpa=sdpa)
                except Exception:
                    state['fallback'] = True
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            failed = False
            try:
                return L.lctx_attention(q, k, v, r, w, 1.0, sink=lambda *a: None, hgroup=2, sdpa=sdpa)
            except Exception:
                failed = True
            if failed:
                return L.lctx_attention(q, k, v, r, w, 1.0, sink=lambda *a: None, hgroup=1, hchunk=1, sdpa=sdpa)
        return attn, state

    def test_forward_failure_falls_back_consistently(self):
        attn, state = self.glue_like({2})  # the second SDPA group of the ORIGINAL forward fails (after the gate ran)
        x = rnd(2, 21, 32, seed=7).requires_grad_()
        (checkpoint(self.block(attn), x, use_reentrant=False) ** 2).sum().backward()  # no CheckpointError
        self.assertTrue(state['fallback'])
        ref = rnd(2, 21, 32, seed=7).requires_grad_()
        plain = lambda q, k, v, r, w: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)  # noqa: E731
        (self.block(plain)(ref) ** 2).sum().backward()
        self.assertLess(float((x.grad - ref.grad).abs().max()), 1e-12)

    def test_recompute_failure_retries_consistently(self):
        attn, state = self.glue_like({4})  # forward = 1, 2; the recompute's second group (call 4) fails once
        x = rnd(2, 21, 32, seed=7).requires_grad_()
        (checkpoint(self.block(attn), x, use_reentrant=False) ** 2).sum().backward()  # no CheckpointError
        self.assertFalse(state['fallback'])
        ok = lambda q, k, v, r, w: L.lctx_attention(q, k, v, r, w, 1.0, sink=lambda *a: None)  # noqa: E731
        ref = rnd(2, 21, 32, seed=7).requires_grad_()
        (self.block(ok)(ref) ** 2).sum().backward()
        self.assertLess(float((x.grad - ref.grad).abs().max()), 1e-11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
