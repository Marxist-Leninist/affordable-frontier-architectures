#!/usr/bin/env python3
"""P039-L: GPU-scale matched rematch of the P039 pilot.

Four arms trained on the same windows, budget and optimiser:
  attn_ar      causal Transformer, next-byte loss on every position
  attn_diff    bidirectional Transformer, absorbing-state masked-diffusion ELBO loss
  hybrid_ar    causal Mamba-1-style selective scan + attention, layout M A M A
  hybrid_diff  the same hybrid made bidirectional (weight-shared reverse scan), masked-diffusion loss
Scale follows P045: width 256, 4 layers, 8 heads, FFN 1024, learned absolute positions over a
256-byte context, 20k steps x batch 16, AdamW 1e-3 with 500 warmup and cosine to 1e-4, wd 0.01,
clip 1, FP32. Data: the hash-pinned P045 train.bin / dev_windows.json / test_windows.json.
Reference PyTorch code written for this test; not stock Mamba, Jamba, LLaDA or MDLM code.
"""
import argparse, base64, hashlib, json, math, os, signal, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PINS = {'train.bin': 'e58bc4084cfc06c5cf96ba65fb1fb38a244a691d2f09d389b09603eba92c8552',
        'dev_windows.json': 'f98ff062cf409201aa61ff21de051b098c813aa22322c84609de3b76a85538e2',
        'test_windows.json': 'f729286378e9ec3695e91e6a61f4ed3d7a3407ee0a6e73dc1aec27e62d51f90f'}
D, LAYERS, HEADS, FFN, CTX = 256, 4, 8, 1024, 256
VOCAB_IN, MASK = 257, 256
S_STATE, DT_RANK, CONV_K = 16, 16, 4
E_INNER = 308          # Mamba mixer = 263,340 params vs attention mixer 263,168 (+0.07%)
SCAN_CHUNKS = 2        # scan split over channels; each chunk checkpointed to bound training memory
SCAN_TCHUNK = 64       # scan split over time with carried state, keeps log-space magnitudes small in fp32
PROMPT, SUFFIX, CASE_STRIDE = 240, 16, 64
ARMS = {'attn_ar': ('AAAA', True), 'attn_diff': ('AAAA', False),
        'hybrid_ar': ('MAMA', True), 'hybrid_diff': ('MAMA', False)}
DIFF_STEPS = (1, 2, 4, 8, 16)
SPACE = 32
LN2 = math.log(2)
ROOT = Path(__file__).resolve().parent
DEV = torch.device('cuda')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jdump(p, x):
    p = Path(p); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, indent=1) + '\n'); os.replace(t, p)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ----------------------------------------------------------------------------- selective scan
def selective_scan(u, delta, A, Bm, Cm, return_last=False):
    """h_t = exp(delta_t*A) h_{t-1} + delta_t u_t B_t ;  y_t = <h_t, C_t>.
    u, delta: (B,T,E); A: (E,S) negative; Bm, Cm: (B,T,S).  Parallel over time inside chunks of
    SCAN_TCHUNK via h_t = exp(Q_t) * sum_{s<=t} exp(-Q_s) b_s with Q = cumsum(delta*A); the positive
    and negative parts of b go through separate logcumsumexp scans (stable in fp32); the state is
    carried between time chunks."""
    T = u.shape[1]; ys = []; h0 = None
    for a in range(0, T, SCAN_TCHUNK):
        b = min(T, a + SCAN_TCHUNK)
        Q = torch.cumsum(delta[:, a:b].unsqueeze(-1) * A, dim=1)
        beta = (delta[:, a:b] * u[:, a:b]).unsqueeze(-1) * Bm[:, a:b].unsqueeze(2)
        lp = torch.log(F.relu(beta) + 1e-30) - Q
        ln = torch.log(F.relu(-beta) + 1e-30) - Q
        h = torch.exp(Q + torch.logcumsumexp(lp, 1)) - torch.exp(Q + torch.logcumsumexp(ln, 1))
        if h0 is not None:
            h = h + torch.exp(Q) * h0.unsqueeze(1)
        h0 = h[:, -1]
        ys.append((h * Cm[:, a:b].unsqueeze(2)).sum(-1))
    y = torch.cat(ys, 1)
    return (y, h0) if return_last else y


def selective_scan_ref(u, delta, A, Bm, Cm):
    b, t, e = u.shape
    h = u.new_zeros(b, e, A.shape[1]); ys = []
    for i in range(t):
        h = torch.exp(delta[:, i, :, None] * A) * h + (delta[:, i] * u[:, i])[:, :, None] * Bm[:, i, None, :]
        ys.append((h * Cm[:, i, None, :]).sum(-1))
    return torch.stack(ys, 1), h


# ----------------------------------------------------------------------------- modules
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D); self.out = nn.Linear(D, D)

    def forward(self, x, causal, cache=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        if cache is not None:
            assert causal
            if 'k' in cache:
                assert t == 1
                k = torch.cat([cache['k'], k], 2); v = torch.cat([cache['v'], v], 2)
            cache['k'], cache['v'] = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal and t > 1))
        return self.out(y.transpose(1, 2).reshape(b, t, D))


class Mamba(nn.Module):
    """Mamba-1-style selective SSM mixer: in_proj -> causal depthwise conv -> selective scan -> SiLU gate -> out_proj."""
    def __init__(self):
        super().__init__()
        E, S, R = E_INNER, S_STATE, DT_RANK
        self.in_proj = nn.Linear(D, 2 * E, bias=False)
        self.conv = nn.Conv1d(E, E, CONV_K, groups=E, padding=CONV_K - 1)
        self.x_proj = nn.Linear(E, R + 2 * S, bias=False)
        self.dt_proj = nn.Linear(R, E)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, S + 1, dtype=torch.float32)).repeat(E, 1))
        self.Dskip = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, D, bias=False)
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -R ** -0.5, R ** -0.5)
            dt = torch.exp(torch.rand(E) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        bounds = np.linspace(0, E, SCAN_CHUNKS + 1).round().astype(int)
        self.slices = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]

    def _pre(self, x):
        t = x.shape[1]
        u0, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u0.transpose(1, 2))[..., :t].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(u), [DT_RANK, S_STATE, S_STATE], -1)
        return u0, u, z, F.softplus(self.dt_proj(dt)), Bm, Cm

    def _scan(self, u, delta, Bm, Cm):
        A = -torch.exp(self.A_log)
        outs = []
        for a, b in self.slices:
            if torch.is_grad_enabled():
                outs.append(checkpoint(selective_scan, u[..., a:b], delta[..., a:b], A[a:b], Bm, Cm, use_reentrant=False))
            else:
                outs.append(selective_scan(u[..., a:b], delta[..., a:b], A[a:b], Bm, Cm))
        return torch.cat(outs, -1)

    def direction(self, x):
        _, u, z, delta, Bm, Cm = self._pre(x)
        y = self._scan(u, delta, Bm, Cm) + self.Dskip * u
        return self.out_proj(y * F.silu(z))

    def forward(self, x, causal, cache=None):
        if cache is None:
            if causal:
                return self.direction(x)
            return 0.5 * (self.direction(x) + self.direction(x.flip(1)).flip(1))
        assert causal
        if 'h' in cache:
            return self.step(x, cache)
        u0, u, z, delta, Bm, Cm = self._pre(x)          # prefill: parallel scan, keep the final states
        A = -torch.exp(self.A_log); ys, hs = [], []
        for a, b in self.slices:
            y, h = selective_scan(u[..., a:b], delta[..., a:b], A[a:b], Bm, Cm, return_last=True)
            ys.append(y); hs.append(h)
        conv_in = u0.transpose(1, 2); pad = CONV_K - 1
        if conv_in.shape[2] < pad:
            conv_in = F.pad(conv_in, (pad - conv_in.shape[2], 0))
        cache['conv'] = conv_in[:, :, -pad:].contiguous(); cache['h'] = torch.cat(hs, 1)
        y = torch.cat(ys, -1) + self.Dskip * u
        return self.out_proj(y * F.silu(z))

    def step(self, x, cache):
        u0, z = self.in_proj(x).chunk(2, -1)                          # (B,1,E)
        win = torch.cat([cache['conv'], u0.transpose(1, 2)], 2)       # (B,E,K)
        cache['conv'] = win[:, :, 1:]
        u = F.silu((win * self.conv.weight[:, 0, :]).sum(-1) + self.conv.bias)
        dt, Bm, Cm = torch.split(self.x_proj(u), [DT_RANK, S_STATE, S_STATE], -1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.exp(delta[..., None] * A) * cache['h'] + (delta * u)[..., None] * Bm[:, None, :]
        cache['h'] = h
        y = (h * Cm[:, None, :]).sum(-1) + self.Dskip * u
        return self.out_proj((y * F.silu(z[:, 0])).unsqueeze(1))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mix = Attention() if kind == 'A' else Mamba()
        self.ff = nn.Sequential(nn.Linear(D, FFN), nn.GELU(), nn.Linear(FFN, D))

    def forward(self, x, causal, cache=None):
        x = x + self.mix(self.n1(x), causal, cache)
        return x + self.ff(self.n2(x))


class LM(nn.Module):
    def __init__(self, arm):
        super().__init__()
        layout, self.causal = ARMS[arm]; self.arm = arm
        self.emb = nn.Embedding(VOCAB_IN, D); self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block(k) for k in layout)
        self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, 256, bias=False)

    def forward(self, x, cache=None, start=0):
        z = self.emb(x) + self.pos(torch.arange(start, start + x.shape[1], device=x.device))
        for i, blk in enumerate(self.blocks):
            z = blk(z, self.causal, None if cache is None else cache[i])
        return self.head(self.norm(z))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def eval_bs(m):
    return 256 if m.arm.startswith('attn') else 64


def make_opt(m, lr=1e-3):
    """AdamW wd 0.01 on everything except the scan's A_log and D skip (standard Mamba practice)."""
    nd = [p for n, p in m.named_parameters() if n.endswith('A_log') or n.endswith('Dskip')]
    ids = {id(p) for p in nd}
    rest = [p for p in m.parameters() if id(p) not in ids]
    groups = [{'params': rest, 'weight_decay': 0.01}]
    if nd:
        groups.append({'params': nd, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups, lr=lr)


# ----------------------------------------------------------------------------- data
def load_data(data_dir):
    out = {}
    for f, h in PINS.items():
        b = (Path(data_dir) / f).read_bytes()
        assert sha(b) == h, f'{f} hash mismatch'
        out[f] = b
    return out


def windows(raw_json):
    rows = json.loads(raw_json); rows = rows['windows'] if isinstance(rows, dict) else rows
    out = []
    for r in rows:
        b = base64.b64decode(r['bytes_b64'], validate=True)
        assert sha(b) == r['sha256']
        out.append((r['domain'], r['doc_id'], b))
    return out


def make_cases(wins, start=256, stride=CASE_STRIDE):
    """Cases = (240-byte prompt, next 16 bytes) at s = 256, 320, ... inside each window (all targets after the window's 256-byte context)."""
    X, meta = [], []
    for dom, doc, b in wins:
        s = start
        while s + SUFFIX <= len(b):
            X.append(np.frombuffer(b[s - PROMPT:s + SUFFIX], np.uint8)); meta.append((dom, doc, s)); s += stride
    return np.stack(X).astype(np.int64), meta


def macro(values, meta):
    """Equal-domain, then equal-document mean (the P045 convention). NaN entries are skipped."""
    per_doc = {}
    for v, (dom, doc, _) in zip(values, meta):
        per_doc.setdefault((dom, doc), []).append(v)
    dom = {}
    for (d, _), vs in per_doc.items():
        vs = [x for x in vs if x == x]
        if vs:
            dom.setdefault(d, []).append(sum(vs) / len(vs))
    per = {k: sum(v) / len(v) for k, v in dom.items()}
    return sum(per.values()) / len(per), per


class Stream:
    """Window starts depend only on the seed (identical for every arm); diffusion masks use a separate generator."""
    def __init__(self, seed, n):
        self.data = np.random.default_rng(seed); self.mask = np.random.default_rng(seed + 7919); self.n = n

    def starts(self, b):
        return self.data.integers(0, self.n - CTX - 1, size=b)

    def masks(self, b, t):
        u0 = self.mask.random()
        tt = ((u0 + np.arange(b) / b) % 1.0) * (1 - 1e-3) + 1e-3          # stratified t in [1e-3, 1)
        cond = self.mask.random(b) < 0.5                                      # half the rows keep a clean prefix
        pre = np.where(cond, self.mask.integers(16, CTX - SUFFIX + 1, size=b), 0)
        elig = np.arange(t)[None] >= pre[:, None]
        m = (self.mask.random((b, t)) < tt[:, None]) & elig
        w = 1.0 / (tt * elig.sum(1))                                          # ELBO weight 1/t, per eligible token
        return m, w


# ----------------------------------------------------------------------------- scoring
@torch.inference_mode()
def ar_suffix_nll(m, X):
    out = []
    for i in range(0, len(X), eval_bs(m)):
        x = torch.from_numpy(X[i:i + eval_bs(m)]).to(DEV)
        lg = m(x[:, :-1])[:, PROMPT - 1:]
        out.append(F.cross_entropy(lg.reshape(-1, 256), x[:, PROMPT:].reshape(-1), reduction='none').view(-1, SUFFIX).mean(1))
    return torch.cat(out).cpu().numpy() / LN2


@torch.inference_mode()
def diff_elbo(m, X, K=32, seed=12345):
    """Masked-diffusion NELBO of the 16-byte suffix given the clean prompt (an upper bound on its NLL), bits/byte.
    Midpoint quadrature over t with K points; one fixed random mask per (case, t)."""
    g = np.random.default_rng(seed)
    ts = (np.arange(K) + 0.5) / K
    rnd = g.random((K, len(X), SUFFIX))
    out = np.zeros(len(X))
    for i in range(0, len(X), eval_bs(m)):
        x = torch.from_numpy(X[i:i + eval_bs(m)]).to(DEV); n = len(x)
        acc = torch.zeros(n, device=DEV, dtype=torch.float64)
        for k, t in enumerate(ts):
            mk = torch.from_numpy(rnd[k, i:i + n] < t).to(DEV)
            xt = x.clone(); xt[:, PROMPT:] = xt[:, PROMPT:].masked_fill(mk, MASK)
            ce = F.cross_entropy(m(xt)[:, PROMPT:].reshape(-1, 256), x[:, PROMPT:].reshape(-1), reduction='none').view(n, SUFFIX)
            acc += ((ce * mk).sum(1) / t).double()
        out[i:i + n] = (acc / K / SUFFIX).cpu().numpy() / LN2
    return out


@torch.inference_mode()
def diff_l2r(m, X):
    """Exact NLL of the suffix under the diffusion network read left to right (position j sees prompt + bytes < j, rest masked)."""
    out = []
    for i in range(0, len(X), eval_bs(m)):
        x = torch.from_numpy(X[i:i + eval_bs(m)]).to(DEV); tot = torch.zeros(len(x), device=DEV)
        for j in range(SUFFIX):
            xt = x.clone(); xt[:, PROMPT + j:] = MASK
            tot += F.cross_entropy(m(xt)[:, PROMPT + j], x[:, PROMPT + j], reduction='none')
        out.append(tot / SUFFIX)
    return torch.cat(out).cpu().numpy() / LN2


@torch.inference_mode()
def gen_ar(m, X):
    """Greedy AR completion by full recompute (same tokens as the cached path, checked in smoke)."""
    out = []
    for i in range(0, len(X), eval_bs(m)):
        seq = torch.from_numpy(X[i:i + eval_bs(m), :PROMPT]).to(DEV)
        for _ in range(SUFFIX):
            seq = torch.cat([seq, m(seq)[:, -1].argmax(-1, keepdim=True)], 1)
        out.append(seq[:, PROMPT:])
    return torch.cat(out).cpu().numpy()


@torch.inference_mode()
def ar_prefill(m, prompt):
    cache = [dict() for _ in m.blocks]
    return m(prompt, cache=cache, start=0)[:, -1], cache


@torch.inference_mode()
def ar_generate_cached(m, prompt):
    lg, cache = ar_prefill(m, prompt); toks = []
    for j in range(SUFFIX):
        nxt = lg.argmax(-1, keepdim=True); toks.append(nxt)
        if j < SUFFIX - 1:
            lg = m(nxt, cache=cache, start=PROMPT + j)[:, -1]
    return torch.cat(toks, 1)


@torch.inference_mode()
def diff_decode(m, prompt, steps, order='conf'):
    """Commit SUFFIX/steps masked positions per pass: the most confident ones (conf) or the leftmost ones (l2r)."""
    n = len(prompt); per = SUFFIX // steps
    x = torch.cat([prompt, torch.full((n, SUFFIX), MASK, dtype=prompt.dtype, device=prompt.device)], 1)
    left = torch.arange(SUFFIX, 0, -1, device=prompt.device, dtype=torch.float32).expand(n, -1)
    for _ in range(steps):
        conf, tok = m(x)[:, PROMPT:].softmax(-1).max(-1)
        masked = x[:, PROMPT:] == MASK
        score = (conf if order == 'conf' else left).masked_fill(~masked, -1.0)
        ix = score.topk(per, -1).indices
        tail = x[:, PROMPT:]; tail.scatter_(1, ix, tok.gather(1, ix))
    assert not (x[:, PROMPT:] == MASK).any()
    return x[:, PROMPT:]


@torch.inference_mode()
def gen_diff(m, X, steps, order='conf'):
    out = []
    for i in range(0, len(X), eval_bs(m)):
        out.append(diff_decode(m, torch.from_numpy(X[i:i + eval_bs(m), :PROMPT]).to(DEV), steps, order))
    return torch.cat(out).cpu().numpy()


def gen_metrics(P, Y, meta):
    P = np.asarray(P).astype(np.int64); Y = np.asarray(Y).astype(np.int64)
    eq = P == Y
    ns = Y != SPACE
    per_case = {
        'byte_acc': eq.mean(1),
        'nonspace_acc': np.where(ns.sum(1) > 0, (eq & ns).sum(1) / np.maximum(ns.sum(1), 1), np.nan),
        'exact': eq.all(1).astype(float),
        'prefix_bytes': np.argmin(np.concatenate([eq, np.zeros((len(eq), 1), bool)], 1), 1).astype(float),
        'space_pred_rate': (P == SPACE).mean(1),
        'distinct_bytes': np.array([len(set(r.tolist())) for r in P], float),
        'degenerate': np.array([np.bincount(r, minlength=256).max() >= 12 for r in P], float),
    }
    res = {}
    for k, v in per_case.items():
        mac, per = macro(v.tolist(), meta)
        res[k] = {'macro': mac, 'domain': per}
    return res, per_case


@torch.inference_mode()
def dev_metric(m, X, meta):
    m.eval()
    v = ar_suffix_nll(m, X) if m.causal else diff_elbo(m, X, K=8, seed=777)
    m.train()
    return macro(v.tolist(), meta)[0]


def foreign_gpu_pids():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=20)
    except Exception:
        return ['nvidia-smi-failed']
    mine = set()
    try:
        mine = set(subprocess.check_output(['pgrep', '-f', 'p039l.py'], text=True).split())
    except Exception:
        pass
    return [p.strip() for p in out.split() if p.strip() and p.strip() not in mine]


# ----------------------------------------------------------------------------- training
STOP = [False]


def _sig(*_):
    STOP[0] = True


def lr_at(step, total, peak=1e-3, final=1e-4, warm=500):
    if step <= warm:
        return peak * step / warm
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def save_weights(m, path):
    from safetensors.torch import save_file
    tmp = Path(str(path) + '.tmp')
    save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(tmp)); os.replace(tmp, path)
    return sha(Path(path).read_bytes())


def load_weights(arm, path):
    from safetensors.torch import load_file
    m = LM(arm).to(DEV); m.load_state_dict(load_file(str(path), device='cpu')); m.eval()
    return m


def train_step(m, opt, tr, stream, step, total, batch):
    for g in opt.param_groups:
        g['lr'] = lr_at(step, total)
    idx = torch.from_numpy(stream.starts(batch)).to(DEV)[:, None] + torch.arange(CTX + 1, device=DEV)
    win = tr[idx].long()
    if m.causal:
        lg = m(win[:, :CTX])
        loss = F.cross_entropy(lg.reshape(-1, 256), win[:, 1:].reshape(-1))
    else:
        x0 = win[:, :CTX]
        mk, w = stream.masks(batch, CTX)
        mk = torch.from_numpy(mk).to(DEV); w = torch.from_numpy(w).float().to(DEV)
        lg = m(x0.masked_fill(mk, MASK))
        ce = F.cross_entropy(lg.reshape(-1, 256), x0.reshape(-1), reduction='none').view(batch, CTX)
        loss = ((ce * mk).sum(1) * w).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    return loss.detach(), gn.detach()


def cmd_train(a):
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    torch.set_num_threads(1)
    raw = load_data(a.data)
    tr = torch.from_numpy(np.frombuffer(raw['train.bin'], np.uint8).copy()).to(DEV)
    dX, dmeta = make_cases(windows(raw['dev_windows.json']))
    out = Path(a.out) / f'{a.arm}_{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('already done'); return
    torch.manual_seed(a.seed)
    m = LM(a.arm).to(DEV); m.train()
    opt = make_opt(m)
    stream = Stream(a.seed, len(tr))
    st = {'arm': a.arm, 'seed': a.seed, 'steps': a.steps, 'batch': a.batch, 'params': nparams(m), 'step': 0,
          'history': [], 'dev': [], 'best': None, 'train_s': 0.0, 'eval_s': 0.0, 'foreign_gpu': [], 'started': utc(),
          'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'code_sha256': sha(Path(__file__).read_bytes())}
    latest = out / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        stream.data.bit_generator.state = ck['rng_data']; stream.mask.bit_generator.state = ck['rng_mask']
        st = ck['state']; st.setdefault('resumed', []).append({'at_step': st['step'], 'utc': utc()})
        print('resumed at', st['step'], flush=True)

    def save_latest(step):
        st['step'] = step; st['peak_mem_bytes'] = torch.cuda.max_memory_allocated()
        ck = {'model': m.state_dict(), 'opt': opt.state_dict(), 'rng_data': stream.data.bit_generator.state,
              'rng_mask': stream.mask.bit_generator.state, 'state': st}
        torch.save(ck, str(latest) + '.tmp'); os.replace(str(latest) + '.tmp', latest)
        jdump(out / 'state.json', st)

    torch.cuda.reset_peak_memory_stats()
    acc_loss = torch.zeros((), device=DEV); acc_gn = torch.zeros((), device=DEV); acc_n = 0
    torch.cuda.synchronize(); t_mark = time.monotonic()
    step = st['step']
    while step < a.steps and not STOP[0]:
        step += 1
        loss, gn = train_step(m, opt, tr, stream, step, a.steps, a.batch)
        acc_loss += loss; acc_gn += gn; acc_n += 1
        if step % 100 == 0 or step == a.steps:
            torch.cuda.synchronize(); now = time.monotonic(); st['train_s'] += now - t_mark
            lv = float(acc_loss) / acc_n
            if not math.isfinite(lv):
                jdump(out / 'failed.json', {'step': step, 'loss': lv, 'utc': utc()}); raise SystemExit('non-finite loss')
            st['history'].append({'step': step, 'loss_bits': lv / LN2, 'grad_norm': float(acc_gn) / acc_n,
                                  'lr': lr_at(step, a.steps), 'train_s': st['train_s']})
            acc_loss.zero_(); acc_gn.zero_(); acc_n = 0
            if step % 1000 == 0:
                print(f'{a.arm} {a.seed} step {step} loss {lv / LN2:.4f} bits {st["train_s"]:.0f}s', flush=True)
            t_mark = time.monotonic()
        if step % a.eval_every == 0 or step == a.steps:
            t0 = time.monotonic()
            dv = dev_metric(m, dX, dmeta)
            st['dev'].append({'step': step, 'dev_bits': dv})
            st['foreign_gpu'].append({'step': step, 'pids': foreign_gpu_pids()})
            if st['best'] is None or dv < st['best']['dev_bits']:
                st['best'] = {'step': step, 'dev_bits': dv, 'sha256': save_weights(m, out / 'best.safetensors')}
            st['eval_s'] += time.monotonic() - t0
            save_latest(step)
            print(f'{a.arm} {a.seed} dev@{step} {dv:.4f} bits (best {st["best"]["dev_bits"]:.4f} @ {st["best"]["step"]})', flush=True)
            torch.cuda.synchronize(); t_mark = time.monotonic()
    if step >= a.steps:
        st['step'] = step; st['final_sha256'] = save_weights(m, out / 'final.safetensors'); st['finished'] = utc()
        st['peak_mem_bytes'] = torch.cuda.max_memory_allocated()
        jdump(out / 'state.json', st); jdump(out / 'done.json', {'utc': utc(), 'best': st['best']})
        latest.unlink(missing_ok=True)
        print('done', a.arm, a.seed, flush=True)
    else:
        save_latest(step); print('stopped at', step, flush=True); sys.exit(3)


# ----------------------------------------------------------------------------- evaluation
def cmd_evaluate(a):
    torch.set_num_threads(1)
    raw = load_data(a.data)
    X, meta = make_cases(windows(raw['test_windows.json']))
    Y = X[:, PROMPT:]
    res_dir = ROOT / 'results'; res_dir.mkdir(exist_ok=True)
    priv = ROOT / 'private'; priv.mkdir(exist_ok=True)
    cases = {'n_cases': len(X), 'n_docs': len({(d, doc) for d, doc, _ in meta}),
             'by_domain': {d: sum(1 for x in meta if x[0] == d) for d in ('article', 'fiction', 'dialogue')}}
    jdump(res_dir / 'cases.json', {**cases, 'prompt': PROMPT, 'suffix': SUFFIX, 'stride': CASE_STRIDE,
                                   'doc_ids': sorted({doc for _, doc, _ in meta})})
    np.savez_compressed(priv / 'case_meta.npz', dom=np.array([x[0] for x in meta]), doc=np.array([x[1] for x in meta]),
                        pos=np.array([x[2] for x in meta]), target=Y.astype(np.uint8))
    runs = sorted(p for p in Path(a.out).iterdir() if (p / 'done.json').exists())
    for rd in runs:
        arm, seed = rd.name.rsplit('_', 1)
        if a.only and rd.name not in a.only.split(','):
            continue
        dest = res_dir / f'eval_{rd.name}.json'
        if dest.exists() and not a.force:
            continue
        st = json.loads((rd / 'state.json').read_text())
        m = load_weights(arm, rd / 'best.safetensors')
        t0 = time.monotonic()
        r = {'arm': arm, 'seed': int(seed), 'best_step': st['best']['step'], 'best_dev_bits': st['best']['dev_bits'],
             'params': nparams(m), 'cases': cases, 'checkpoint_sha256': st['best']['sha256']}
        arrays = {}
        if m.causal:
            nll = ar_suffix_nll(m, X); arrays['nll_ar'] = nll
            r['suffix_bits'] = dict(zip(('macro', 'domain'), macro(nll.tolist(), meta)))
            P = gen_ar(m, X); arrays['pred_greedy'] = P.astype(np.uint8)
            r['gen'] = {'greedy': gen_metrics(P, Y, meta)[0]}
        else:
            el = diff_elbo(m, X, K=a.k); arrays['nelbo'] = el
            r['suffix_bits_elbo'] = dict(zip(('macro', 'domain'), macro(el.tolist(), meta)))
            l2 = diff_l2r(m, X); arrays['nll_l2r'] = l2
            r['suffix_bits_l2r'] = dict(zip(('macro', 'domain'), macro(l2.tolist(), meta)))
            r['gen'] = {}
            for s in DIFF_STEPS:
                P = gen_diff(m, X, s, 'conf'); arrays[f'pred_conf{s}'] = P.astype(np.uint8)
                r['gen'][f'conf{s}'] = gen_metrics(P, Y, meta)[0]
            P = gen_diff(m, X, SUFFIX, 'l2r'); arrays['pred_l2r16'] = P.astype(np.uint8)
            r['gen']['l2r16'] = gen_metrics(P, Y, meta)[0]
        r['eval_s'] = time.monotonic() - t0
        np.savez_compressed(priv / f'cases_{rd.name}.npz', **arrays)
        jdump(dest, r); print('evaluated', rd.name, round(r['eval_s'], 1), flush=True)
        del m; torch.cuda.empty_cache()


@torch.inference_mode()
def p045_protocol(m, wins):
    """P045's protocol: every target scored with exactly its preceding 256 bytes; equal domain then document mean."""
    bs = 512 if m.arm.startswith('attn') else 48
    dom = {}
    for d, doc, b in wins:
        t = torch.tensor(list(b), device=DEV); w = t.unfold(0, 257, 1); ls = []
        for batch in w.split(bs):
            ls.append(F.cross_entropy(m(batch[:, :256])[:, -1], batch[:, 256], reduction='none'))
        dom.setdefault(d, []).append(float(torch.cat(ls).mean()))
    per = {k: sum(v) / len(v) for k, v in dom.items()}
    mac = sum(per.values()) / len(per)
    return {'macro_nll_nats': mac, 'macro_bits': mac / LN2, 'domain_nll_nats': per}


def cmd_p045(a):
    raw = load_data(a.data); wins = windows(raw['test_windows.json'])
    for name in a.only.split(','):
        arm = name.rsplit('_', 1)[0]; dest = ROOT / 'results' / f'p045protocol_{name}.json'
        if dest.exists() or not (Path(a.out) / name / 'done.json').exists():
            continue
        m = load_weights(arm, Path(a.out) / name / 'best.safetensors'); t0 = time.monotonic()
        r = p045_protocol(m, wins); r['seconds'] = time.monotonic() - t0; r['run'] = name
        jdump(dest, r); print('p045', name, r['macro_nll_nats'], flush=True)
        del m; torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- timing
def _time(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); vals = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); vals.append(1e3 * (time.perf_counter() - t))
    return {'p50_ms': statistics.median(vals), 'p95_ms': sorted(vals)[max(0, math.ceil(0.95 * len(vals)) - 1)], 'n': reps}


def cmd_timing(a):
    torch.set_num_threads(1)
    raw = load_data(a.data)
    X, meta = make_cases(windows(raw['test_windows.json']))
    tr = torch.from_numpy(np.frombuffer(raw['train.bin'], np.uint8).copy()).to(DEV)
    res = {'utc': utc(), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'arms': {}}
    for arm in ARMS:
        cands = sorted(Path(a.out).glob(f'{arm}_*/best.safetensors'))
        m = load_weights(arm, cands[0]) if cands else LM(arm).to(DEV).eval()
        r = {'weights': cands[0].parent.name if cands else 'random-init', 'foreign_gpu_before': foreign_gpu_pids()}
        for bs in (1, 64):
            pr = torch.from_numpy(X[:bs, :PROMPT]).to(DEV)
            if m.causal:
                r[f'b{bs}_ar_cached_16'] = _time(lambda: ar_generate_cached(m, pr), a.reps)
                r[f'b{bs}_prefill_240'] = _time(lambda: ar_prefill(m, pr), a.reps)
            else:
                for s in DIFF_STEPS:
                    r[f'b{bs}_diff_{s}pass_16'] = _time(lambda: diff_decode(m, pr, s), a.reps)
        if m.causal:
            _, cache = ar_prefill(m, torch.from_numpy(X[:1, :PROMPT]).to(DEV))
            r['state_bytes_after_prompt'] = int(sum(v.numel() * v.element_size() for c in cache for v in c.values()))
        del m; torch.cuda.empty_cache()
        torch.manual_seed(0); mt = LM(arm).to(DEV); mt.train()
        opt = make_opt(mt, lr=1e-4); stream = Stream(0, len(tr))
        torch.cuda.reset_peak_memory_stats()
        r['train_step'] = _time(lambda: train_step(mt, opt, tr, stream, 1000, 20000, 16), a.train_reps, warm=10)
        r['train_peak_mem_bytes'] = torch.cuda.max_memory_allocated()
        r['foreign_gpu_after'] = foreign_gpu_pids()
        res['arms'][arm] = r; print(arm, json.dumps(r)[:500], flush=True)
        del mt, opt; torch.cuda.empty_cache()
    jdump(ROOT / 'results' / 'timing.json', res)


# ----------------------------------------------------------------------------- smoke
def cmd_smoke(a):
    torch.set_num_threads(1)
    rep = {'utc': utc(), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    raw = load_data(a.data); rep['hashes'] = 'ok'
    rel = lambda p, q: float((p - q).abs().max() / q.abs().max().clamp(min=1e-12))
    for name, dmax in (('extreme', 4.0), ('typical', 0.2)):
        g = torch.Generator().manual_seed(5)
        b, t, e, s = 2, 150, 6, S_STATE
        u = torch.randn(b, t, e, generator=g, dtype=torch.float64).requires_grad_()
        dl = (torch.rand(b, t, e, generator=g, dtype=torch.float64) * dmax).requires_grad_()
        A = (-(torch.rand(e, s, generator=g, dtype=torch.float64) * 15 + 0.5)).requires_grad_()
        Bm = torch.randn(b, t, s, generator=g, dtype=torch.float64).requires_grad_()
        Cm = torch.randn(b, t, s, generator=g, dtype=torch.float64).requires_grad_()
        y1, h1 = selective_scan(u, dl, A, Bm, Cm, return_last=True); y2, h2 = selective_scan_ref(u, dl, A, Bm, Cm)
        w = torch.randn_like(y1)
        g1 = torch.autograd.grad((y1 * w).sum(), [u, dl, A, Bm, Cm]); g2 = torch.autograd.grad((y2 * w).sum(), [u, dl, A, Bm, Cm])
        y3 = selective_scan(*(z.detach().float().cuda() for z in (u, dl, A, Bm, Cm))).double().cpu()
        rep[f'scan_{name}'] = {'fp64_y': rel(y1, y2), 'fp64_h_last': rel(h1, h2), 'fp64_grads': [rel(p, q) for p, q in zip(g1, g2)],
                               'fp32_y': rel(y3, y2.detach())}
        assert rep[f'scan_{name}']['fp64_y'] < 1e-9 and max(rep[f'scan_{name}']['fp64_grads']) < 1e-7, rep
        assert rep[f'scan_{name}']['fp32_y'] < (5e-3 if name == 'extreme' else 2e-4), rep
    dX, dmeta = make_cases(windows(raw['dev_windows.json']))
    tr = torch.from_numpy(np.frombuffer(raw['train.bin'], np.uint8).copy()).to(DEV)
    rep['arms'] = {}
    for arm in ARMS:
        torch.manual_seed(17); m = LM(arm).to(DEV); r = {'params': nparams(m)}
        opt = make_opt(m); stream = Stream(17, len(tr))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.monotonic()
        for i in range(1, a.smoke_steps + 1):
            loss, _ = train_step(m, opt, tr, stream, i, 20000, 16)
            if i == 10:
                torch.cuda.synchronize(); t0 = time.monotonic()
        torch.cuda.synchronize()
        r['ms_per_step'] = 1e3 * (time.monotonic() - t0) / max(1, a.smoke_steps - 10)
        r['train_peak_mem_bytes'] = torch.cuda.max_memory_allocated(); r['loss_bits_last'] = float(loss) / LN2
        m.eval()
        x = torch.from_numpy(dX[:8]).to(DEV)
        with torch.inference_mode():
            if m.causal:
                y = x.clone(); y[:, 200:] = (y[:, 200:] + 77) % 256
                r['causal_err'] = float((m(x)[:, :200] - m(y)[:, :200]).abs().max())
                full = m(x)
                lg0, cache = ar_prefill(m, x[:, :PROMPT]); lg = [lg0]
                for j in range(SUFFIX - 1):
                    lg.append(m(x[:, PROMPT + j:PROMPT + j + 1], cache=cache, start=PROMPT + j)[:, -1])
                r['cache_parity_err'] = float((torch.stack(lg, 1) - full[:, PROMPT - 1:PROMPT - 1 + SUFFIX]).abs().max())
                r['cached_greedy_equal'] = bool(torch.equal(ar_generate_cached(m, x[:, :PROMPT]).cpu(),
                                                            torch.from_numpy(gen_ar(m, dX[:8]))))
                assert r['causal_err'] < 1e-4 and r['cache_parity_err'] < 1e-3 and r['cached_greedy_equal'], r
            else:
                p1 = gen_diff(m, dX[:8], 4); z = dX[:8].copy(); z[:, PROMPT:] = (z[:, PROMPT:] + 91) % 256
                r['decode_ignores_targets'] = bool((gen_diff(m, z, 4) == p1).all())
                ea = diff_elbo(m, dX[:8], K=4); r['elbo_finite'] = bool(np.isfinite(ea).all())
                assert r['decode_ignores_targets'] and r['elbo_finite'], r
        t0 = time.monotonic(); r['dev_bits_after_smoke'] = dev_metric(m, dX, dmeta); r['dev_eval_s'] = time.monotonic() - t0
        rep['arms'][arm] = r; print(arm, r, flush=True)
        del m, opt; torch.cuda.empty_cache()
    pa = rep['arms']['attn_ar']['params']; ph = rep['arms']['hybrid_ar']['params']
    rep['param_gap_rel'] = (ph - pa) / pa
    assert abs(rep['param_gap_rel']) < 1e-3, rep
    rep['ok'] = True
    (ROOT / 'results').mkdir(exist_ok=True)
    jdump(ROOT / 'results' / 'smoke.json', rep); print(json.dumps(rep)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['smoke', 'train', 'evaluate', 'p045', 'timing'])
    ap.add_argument('--data', default=str(ROOT / 'data')); ap.add_argument('--out', default=str(ROOT / 'runs'))
    ap.add_argument('--arm', choices=list(ARMS)); ap.add_argument('--seed', type=int)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--eval-every', type=int, default=2000); ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--only', default=''); ap.add_argument('--force', action='store_true')
    ap.add_argument('--reps', type=int, default=30); ap.add_argument('--train-reps', type=int, default=40)
    ap.add_argument('--smoke-steps', type=int, default=60)
    a = ap.parse_args()
    {'smoke': cmd_smoke, 'train': cmd_train, 'evaluate': cmd_evaluate, 'p045': cmd_p045, 'timing': cmd_timing}[a.cmd](a)


if __name__ == '__main__':
    main()
