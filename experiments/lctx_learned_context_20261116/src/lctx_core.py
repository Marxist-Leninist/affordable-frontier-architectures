"""lctx_core.py -- LEARNED CONTEXT LENGTH for the AGILLM cosine/RoPE attention (claude-code-lctx888-20261116).

Owner, 16 Nov 2026: "idea looking it apply to 88B a learned context length that model decides in training and
on inference".

WHAT (v2, the Forgetting Transformer's own gate, Lin et al., ICLR 2025, with a per-head learned bias):
  for every layer and head h, every token t gets a forget rate
      a_t = min(softplus(r_h + w_h . u_t), amax),     u_t = v_t / |v_t|   (the head's own value, detached)
  and causal attention becomes
      logit_ij = s * q_i . k_j  -  (a_{j+1} + ... + a_i)          (j <= i)
  i.e. a key fades by exp(-a) for every newer token.  1 / softplus(r_h) is the head's context length in tokens (learned,
  log scale while it is long); w_h lets any token raise its rate up to amax, so a head can keep a long context inside a
  document and still drop everything at a break (v1's lam * 2 sigmoid(.) capped a token at twice the head's rate).
  r = -inf is the original attention; the install rate softplus(r0) = 1e-6 is the model as it was.

HOW (exact, no T x T mask, any SDPA kernel): with kb_j = a_0 + ... + a_j (a running sum, >= 0),
  -(a_{j+1} + ... + a_i) = kb_j - kb_i, and kb_i is the same for every key of row i, so softmax only needs the KEY bias
  +kb_j.  One extra head dimension carries it:  q' = [q, 1/s, 0..], k' = [k, kb, 0..] (padded to a multiple of 8),
  SDPA(q', k', v, is_causal=True, scale=s) is the forgetting attention.
  LctxAttn is ONE autograd Function for gate + attention: its forward computes everything without a graph and saves
  q, k, v, r, w only once it has succeeded, so a failure inside it saves nothing (a caller can fall back to the original
  attention, or retry in smaller pieces, and a non-reentrant checkpoint recompute still sees the same saved tensors).
  Its backward reads ctx.saved_tensors once, redoes each head group's forward under its own small graph, and returns the
  exact gradients of q, k, v; the gate's gradients (r, w) are analytic and go to a sink (the trainer mod's host
  accumulators) or are returned (tests).  No gradient reaches v through the gate.

INFERENCE: kb never decreases, so once kb_i - kb_j > theta a key j is suppressed by e^-theta for query i AND for every
  later query: it can leave that head's KV cache for good.  With |logit| <= T_att (11.31 on the 888B) the dropped mass
  per key is at most e^(2 T_att - theta) (theta = 40: 1.1e-7).  See prune_keep / DecodeCache.
"""
import math

import torch
import torch.nn.functional as F

V = 'lctx-core-v2-20261116'
FORM = 'softplus(r+w.unit(v))'
TINY = 1e-12


def _unit(v):
    vf = v.detach()
    return vf / torch.linalg.vector_norm(vf, dim=-1, keepdim=True).clamp_min(TINY)


def _x(u, r, w):
    """x (B, H, T) = r_h + w_h . u_t"""
    return torch.einsum('bhtd,hd->bht', u, w.to(u.dtype)) + r.to(u.dtype)[None, :, None]


def r_from_rate(lam):
    """r with softplus(r) = lam (exact inverse; the v1 -> v2 migration of a head whose w, c are 0)"""
    lam = torch.as_tensor(lam, dtype=torch.float64)
    return torch.where(lam > 30.0, lam, torch.log(torch.expm1(lam.clamp_min(1e-300))))


def rate_from_r(r):
    return F.softplus(torch.as_tensor(r, dtype=torch.float64))


def forget_rates(v, r, w, amax):
    """a (B, H, T): the per-token forget rate of every head (no autograd)"""
    with torch.no_grad():
        return torch.clamp(F.softplus(_x(_unit(v), r.detach(), w.detach())), max=float(amax))


def _gate_kb(v, r, w, amax, hchunk):
    """kb (B, H, T) = running sum of the rates, head chunk by head chunk (bounds the u transient)"""
    H = v.shape[1]
    hc = max(1, min(int(hchunk or H), H))
    out = []
    for h0 in range(0, H, hc):
        h1 = min(H, h0 + hc)
        a = torch.clamp(F.softplus(_x(_unit(v[:, h0:h1]), r[h0:h1].detach(), w[h0:h1].detach())), max=float(amax))
        out.append(torch.cumsum(a, dim=-1))
        del a
    return out[0] if len(out) == 1 else torch.cat(out, dim=1)


def _gate_backward(v, r, w, amax, dkb, hchunk):
    """(dr (H,), dw (H, E)) from dL/dkb (B, H, T): dL/da_l = sum_{j >= l} dL/dkb_j; da/dx = sigmoid(x) where a < amax"""
    H, dt = v.shape[1], v.dtype
    hc = max(1, min(int(hchunk or H), H))
    dkb = dkb.to(dt)
    da_all = torch.flip(torch.cumsum(torch.flip(dkb, [-1]), dim=-1), [-1])
    dr = torch.zeros(H, device=v.device, dtype=dt)
    dw = torch.zeros(H, v.shape[-1], device=v.device, dtype=dt)
    for h0 in range(0, H, hc):
        h1 = min(H, h0 + hc)
        u = _unit(v[:, h0:h1])
        x = _x(u, r[h0:h1].detach(), w[h0:h1].detach())
        live = (F.softplus(x) < float(amax)).to(dt)
        dx = da_all[:, h0:h1] * torch.sigmoid(x) * live
        dr[h0:h1] = dx.sum(dim=(0, 2))
        dw[h0:h1] = torch.einsum('bht,bhtd->hd', dx, u)
        del u, x, live, dx
    return dr, dw


def ext_dim(e):
    """head dim with the bias column, padded to a multiple of 8 (what the fused kernels want)"""
    return e + 1 + ((-(e + 1)) % 8)


def _ext(qg, kg, kbg, scale):
    B, n, T, E = qg.shape
    Ex = ext_dim(E)
    qx = torch.cat([qg, qg.new_full((B, n, T, 1), 1.0 / scale), qg.new_zeros((B, n, T, Ex - E - 1))], dim=-1)
    kx = torch.cat([kg, kbg.unsqueeze(-1).to(kg.dtype), kg.new_zeros((B, n, kg.shape[2], Ex - E - 1))], dim=-1)
    return qx, kx


class LctxAttn(torch.autograd.Function):
    """gate + causal biased attention as ONE Function (see the module docstring).
    sink None: the gate gradients are returned for r, w (tests).  sink callable: sink(dr, dw) receives them and r, w
    get none (the trainer mod).  hchunk: heads per gate chunk; hgroup: heads per SDPA group (device transients)."""
    SAFE_GATE = False  # True in the trainer mod: an error in the gate's backward costs only that gate gradient
    gate_errors = 0
    retries = 0
    last_error = None

    @staticmethod
    def forward(ctx, q, k, v, r, w, amax, scale, sink, hchunk, hgroup, sdpa):
        ctx.set_materialize_grads(False)
        sdpa = sdpa or F.scaled_dot_product_attention
        B, H, T, E = q.shape
        hg = max(1, min(int(hgroup or H), H))
        with torch.no_grad():
            kb = _gate_kb(v, r, w, amax, hchunk)
            out = q.new_empty((B, H, T, v.shape[-1]))
            for h0 in range(0, H, hg):
                h1 = min(H, h0 + hg)
                qx, kx = _ext(q[:, h0:h1], k[:, h0:h1], kb[:, h0:h1], scale)
                out[:, h0:h1] = sdpa(qx, kx, v[:, h0:h1], is_causal=True, scale=scale)
                del qx, kx
            del kb
        ctx.save_for_backward(q, k, v, r, w)  # only after everything above succeeded
        ctx.amax, ctx.scale, ctx.sink, ctx.hc, ctx.hg, ctx.sdpa = float(amax), float(scale), sink, hchunk, hg, sdpa
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, r, w = ctx.saved_tensors  # exactly once (non-reentrant checkpoint allows one unpack per saved tensor)
        nones = (None,) * 11
        if dout is None:
            return nones
        nq, nk, nv, nr, nw = ctx.needs_input_grad[:5]
        gate = ctx.sink is not None or nr or nw
        if not (nq or nk or nv or gate):
            return nones
        with torch.no_grad():
            kb = _gate_kb(v, r, w, ctx.amax, ctx.hc)
        need = (nq, nk, nv, gate)
        res = [torch.empty_like(t) if n else None for t, n in zip((q, k, v, kb), need)]
        H = q.shape[1]

        def group(h0, h1):
            with torch.enable_grad():
                parts = [t[:, h0:h1].detach().requires_grad_(bool(n)) for t, n in zip((q, k, v, kb), need)]
                qx, kx = _ext(parts[0], parts[1], parts[3], ctx.scale)
                og = ctx.sdpa(qx, kx, parts[2], is_causal=True, scale=ctx.scale)
                ins = [p for p, n in zip(parts, need) if n]
                grads = torch.autograd.grad(og, ins, dout[:, h0:h1])
            gi = iter(grads)
            for i, n in enumerate(need):
                if n:
                    res[i][:, h0:h1] = next(gi)

        for h0 in range(0, H, ctx.hg):
            h1 = min(H, h0 + ctx.hg)
            failed = False
            try:
                group(h0, h1)
            except Exception:
                if h1 - h0 == 1:
                    raise
                failed = True
            if failed:  # outside the except: the failed attempt's frames (and their memory) are gone
                LctxAttn.retries += 1
                if q.is_cuda:
                    torch.cuda.empty_cache()
                for h in range(h0, h1):
                    group(h, h + 1)
        dr = dw = None
        if gate:
            try:
                g_r, g_w = _gate_backward(v, r, w, ctx.amax, res[3], ctx.hc)
                if ctx.sink is not None:
                    ctx.sink(g_r, g_w)
                else:
                    dr = g_r.to(r.dtype) if nr else None
                    dw = g_w.to(w.dtype) if nw else None
            except Exception as ex:
                if not LctxAttn.SAFE_GATE:
                    raise
                LctxAttn.gate_errors += 1  # the gate has no gradient path into the model: dropping it is safe
                LctxAttn.last_error = repr(ex)[:300]
        return res[0], res[1], res[2], dr, dw, None, None, None, None, None, None


def lctx_attention(q, k, v, r, w, scale, amax=6.0, sink=None, hchunk=None, hgroup=None, sdpa=None):
    """the whole thing for one layer: one LctxAttn call"""
    return LctxAttn.apply(q, k, v, r, w, float(amax), float(scale), sink, hchunk, hgroup, sdpa)


# ------------------------------------------------------------------------------------------------ references (tests)

def biased_sdpa(q, k, v, kb, scale, sdpa=None):
    """plain-autograd causal SDPA with key bias kb (B, H, T): softmax_j(s q_i.k_j + kb_j), j <= i"""
    qx, kx = _ext(q, k, kb, scale)
    return (sdpa or F.scaled_dot_product_attention)(qx, kx, v, is_causal=True, scale=scale)


def ref_attention(q, k, v, r, w, scale, amax=6.0):
    """explicit T x T reference: softmax(s q k^T + (D_i - D_j)) v with D = -cumsum(a), causal; differentiable in
    q, k, v (main path) and r, w (gate; v detached there)"""
    u = _unit(v)
    x = torch.einsum('bhtd,hd->bht', u, w.to(q.dtype)) + r.to(q.dtype)[None, :, None]
    a = torch.clamp(F.softplus(x), max=float(amax))
    D = -torch.cumsum(a, dim=-1)
    T = q.shape[-2]
    logits = scale * (q @ k.transpose(-1, -2)) + (D[..., :, None] - D[..., None, :])
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, float('-inf'))
    return torch.softmax(logits, dim=-1) @ v


def prune_keep(kb, theta):
    """bool (B, H, Tq, Tk): key j still in head h's cache for query i (j <= i and kb_i - kb_j <= theta)"""
    T = kb.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=kb.device).tril()
    return causal & ((kb[..., :, None] - kb[..., None, :]) <= theta)


def pruned_attention(q, k, v, kb, scale, theta):
    """reference attention that never looks at the keys prune_keep drops (what a pruned KV cache computes)"""
    keep = prune_keep(kb, theta)
    logits = scale * (q @ k.transpose(-1, -2)) + kb[..., None, :]
    logits = logits.masked_fill(~keep, float('-inf'))
    return torch.softmax(logits, dim=-1) @ v, keep


class DecodeCache:
    """per-head KV cache for token-by-token decoding with the learned forgetting: keys whose bias fell more than theta
    below the newest token's are evicted for good (kb never decreases).  Pure reference: one sequence, lists per head."""

    def __init__(self, H, theta):
        self.H, self.theta = H, float(theta)
        self.k = [[] for _ in range(H)]
        self.v = [[] for _ in range(H)]
        self.b = [[] for _ in range(H)]
        self.kb = [0.0] * H
        self.evicted = 0
        self.seen = 0

    def step(self, q, k, v, a, scale):
        """q, k, v (H, E) for the new token, a (H,) its forget rates; returns the attention output (H, Ev)"""
        out = []
        self.seen += 1
        for h in range(self.H):
            self.kb[h] += float(a[h])
            self.k[h].append(k[h])
            self.v[h].append(v[h])
            self.b[h].append(self.kb[h])
            keep = [i for i, b in enumerate(self.b[h]) if self.kb[h] - b <= self.theta]
            if len(keep) < len(self.b[h]):
                self.evicted += len(self.b[h]) - len(keep)
                self.k[h] = [self.k[h][i] for i in keep]
                self.v[h] = [self.v[h][i] for i in keep]
                self.b[h] = [self.b[h][i] for i in keep]
            K = torch.stack(self.k[h])
            Vh = torch.stack(self.v[h])
            bb = torch.tensor(self.b[h], dtype=K.dtype)
            p = torch.softmax(scale * (K @ q[h]) + bb, dim=-1)
            out.append(p @ Vh)
        return torch.stack(out)

    def cache_len(self):
        return [len(b) for b in self.b]


def spans(r):
    """context length per head in tokens (1 / softplus(r): the distance over which a key fades by e^-1 at w.u = 0)"""
    return [float('inf') if not math.isfinite(float(x)) else 1.0 / float(rate_from_r(float(x))) for x in r]


# ------------------------------------------------------------------------------------------------ the optimiser

R0 = float(r_from_rate(1e-6))  # install rate: the oldest of 2048 keys loses <= 0.002 nats of logit (the model as it was)
RHO0 = R0  # v1 name
DEFAULT_KNOBS = dict(lr_rho=0.03, lr_w=3e-3, b1=0.9, b2=0.99, wdecay=1e-3, rho_min=math.log(1e-7), rho_max=4.0,
                     wmax=32.0, span_cost=0.0, amax=6.0, freeze=False)
KNOB_LIMITS = dict(lr_rho=(0.0, 0.5), lr_w=(0.0, 0.1), b1=(0.0, 0.999), b2=(0.0, 0.99999), wdecay=(0.0, 0.1),
                   rho_min=(-30.0, 8.0), rho_max=(-30.0, 8.0), wmax=(0.0, 256.0), span_cost=(0.0, 10.0),
                   amax=(0.01, 30.0))


def clean_knobs(d, base=None):
    """knobs from a dict: unknown keys (v1's ddep_from) and out-of-range values are ignored (defaults or base kept).
    rho_min / rho_max bound r (softplus(r) is the head's rate)."""
    k = dict(base or DEFAULT_KNOBS)
    if isinstance(d, dict):
        for name, (lo, hi) in KNOB_LIMITS.items():
            x = d.get(name)
            if type(x) in (int, float) and math.isfinite(float(x)) and lo <= float(x) <= hi:
                k[name] = float(x)
        if type(d.get('freeze')) is bool:
            k['freeze'] = d['freeze']
    if k['rho_min'] > k['rho_max']:
        k['rho_min'], k['rho_max'] = DEFAULT_KNOBS['rho_min'], DEFAULT_KNOBS['rho_max']
    return k


class LctxState:
    """r (L, H), w (L, H, E) as float32 masters on the host, Adam moments in float64, one Adam step count per layer;
    gradients arrive per layer (add) and become one step per layer (step).  Pure torch on the CPU."""

    def __init__(self, L, H, E, r0=R0):
        self.L, self.H, self.E = int(L), int(H), int(E)
        self.r = torch.full((L, H), float(r0), dtype=torch.float32)
        self.w = torch.zeros((L, H, E), dtype=torch.float32)
        self.m = dict(r=torch.zeros(L, H, dtype=torch.float64), w=torch.zeros(L, H, E, dtype=torch.float64))
        self.v = {k: torch.zeros_like(t) for k, t in self.m.items()}
        self.t = torch.zeros(L, dtype=torch.int64)
        self.acc = {}
        self.updates = 0

    def add(self, layer, dr, dw):
        g = [t.detach().to('cpu', torch.float64) for t in (dr, dw)]
        cur = self.acc.get(layer)
        self.acc[layer] = g if cur is None else [a + b for a, b in zip(cur, g)]

    def pending(self):
        return len(self.acc)

    def drop(self):
        self.acc = {}

    def step(self, knobs, gscale=1.0):
        """one Adam step for every layer with a gradient; gscale multiplies the gradients (1 / loss scale).  Returns
        dict(layers=n, bad=n) - a layer whose gradient is not finite is skipped."""
        k = knobs
        acc, self.acc = self.acc, {}
        out = dict(layers=0, bad=0)
        if k['freeze']:
            return out
        b1, b2 = k['b1'], k['b2']
        for layer, (gr, gw) in sorted(acc.items()):
            gr, gw = gr * gscale, gw * gscale
            if not (bool(torch.isfinite(gr).all()) and bool(torch.isfinite(gw).all())):
                out['bad'] += 1
                continue
            if k['span_cost']:  # the price of context: d(span_cost * log span)/dr = -span_cost * sigmoid(r) / softplus(r)
                rr = self.r[layer].double()
                gr = gr - k['span_cost'] * torch.sigmoid(rr) / F.softplus(rr)
            self.t[layer] += 1
            t = int(self.t[layer])
            for name, g, lr in (('r', gr, k['lr_rho']), ('w', gw, k['lr_w'])):
                m, v = self.m[name][layer], self.v[name][layer]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                stp = lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-30)
                p = getattr(self, name)[layer]
                new = p.double()
                if name == 'w':
                    new = (new * (1.0 - k['wdecay']) - stp).clamp(-k['wmax'], k['wmax'])
                else:
                    new = (new - stp).clamp(k['rho_min'], k['rho_max'])
                p.copy_(new.to(torch.float32))
            out['layers'] += 1
        if out['layers']:
            self.updates += 1
        return out

    def layer_params(self, layer, device, dtype=torch.float32):
        return self.r[layer].to(device=device, dtype=dtype), self.w[layer].to(device=device, dtype=dtype)

    def params_bytes(self):
        """the exact little-endian float32 layout of the pack file: r (L*H) then w (L*H*E)"""
        return torch.cat([self.r.flatten(), self.w.flatten()]).numpy().astype('<f4').tobytes()

    def state_dict(self):
        return dict(version=V, form=FORM, L=self.L, H=self.H, E=self.E, r=self.r.clone(), w=self.w.clone(),
                    m={k: t.clone() for k, t in self.m.items()}, v={k: t.clone() for k, t in self.v.items()},
                    t=self.t.clone(), updates=self.updates)

    def load_state_dict(self, sd):
        """v2 (r, w) or v1 (rho, w, c: a = exp(rho) * 2 sigmoid(w.u + c)); v1 converts exactly while its w and c are 0
        (r = softplus^-1(exp(rho)), rho's Adam moments kept); a v1 w or c that is not 0 cannot be carried and is reset."""
        if (int(sd['L']), int(sd['H']), int(sd['E'])) != (self.L, self.H, self.E):
            raise ValueError('lctx state geometry differs')
        note = None
        if 'r' in sd:
            r, w = sd['r'].to(torch.float32), sd['w'].to(torch.float32)
            m, v = sd['m'], sd['v']
            mr, vr, mw, vw = m['r'], v['r'], m['w'], v['w']
        else:
            rho = sd['rho'].double()
            r = r_from_rate(rho.exp()).to(torch.float32)
            w = torch.zeros(self.L, self.H, self.E)
            mr, vr = sd['m']['rho'], sd['v']['rho']
            mw, vw = torch.zeros(self.L, self.H, self.E), torch.zeros(self.L, self.H, self.E)
            if bool(sd['w'].abs().gt(0).any()) or bool(sd['c'].abs().gt(0).any()):
                note = 'v1 w/c were not zero and were reset'
        for name, t in (('r', r), ('w', w)):
            if tuple(t.shape) != tuple(getattr(self, name).shape) or not bool(torch.isfinite(t).all()):
                raise ValueError('lctx state tensor %s is bad' % name)
            getattr(self, name).copy_(t)
        self.m['r'].copy_(mr)
        self.v['r'].copy_(vr)
        self.m['w'].copy_(mw)
        self.v['w'].copy_(vw)
        self.t.copy_(sd['t'])
        self.updates = int(sd.get('updates', 0))
        return note

    def summary(self):
        span = 1.0 / F.softplus(self.r.double())
        q = torch.quantile(span.flatten(), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64))
        return dict(updates=self.updates, span_min=float(q[0]), span_p10=float(q[1]), span_median=float(q[2]),
                    span_p90=float(q[3]), span_max=float(q[4]),
                    heads_under_256=int((span < 256).sum()), heads_under_2048=int((span < 2048).sum()),
                    heads=int(span.numel()), w_rms=float(self.w.double().pow(2).mean().sqrt()),
                    w_absmax=float(self.w.abs().max()))
