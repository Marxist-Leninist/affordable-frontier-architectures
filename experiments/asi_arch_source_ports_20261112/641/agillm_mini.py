#!/usr/bin/env python3
"""agillm_mini.py -- the ASI-Arch objective rebuilt for the AGILLM cluster: a tiny faithful mirror of the live 600B
training system (agillm128b.py sha 1241dead, architecture agillm128b-moe-local-v1), used to score candidate architectures.

MIRRORED EXACTLY (read side by side with agillm128b.py):
  * binary weights: int8 latents, binary read sign(P), init +-32, one fp32 scale per row, factored Adafactor R/C moments,
    bin8 step  P <- clamp(P + floor(U[0,1) - a*u/max(1,rms))), a = sign_lr 0.707 * warmup(30); no per-weight optimiser state
  * block: RMS n1 -> fused qkv -> unit q,k + RoPE -> SDPA with logit T*cos(q,k), T = sqrt(hd) -> RMS no -> o -> residual;
    RMS n2 -> top-1 router (nn.Linear on detached z) -> relu^2 expert with hidden RMS nh -> residual; routing balance aux 0.01
  * expert counts per layer from the same formula as moe_config (2-3 experts/layer)
  * objective schedule ar,sat_fixed,sat_variable,nat,ar,sat_variable,ar,ar + 1-in-9 local_denoise (4-block group, sigma band
    by group, masked noisy inputs, group conditioner + local norm, bidirectional), SAT shifted-causal-masked-groups, NAT
    unshifted-bidirectional-masked-suffix, same mask rules; AdamW(lr 1e-3, betas .9/.95, wd 0) on all fp params, warmup 30,
    grad clip 1.0; tokens/step = 2048 as live; tokenizer o200k_base; data from the same sources and weights as the live mix
REDUCED (and said so everywhere): vocab 8192 (o200k top-8190+UNK+MASK), d 256, L 12, H 4 (hd 64), ff 1024, ctx 256, batch 8;
  SAT-variable draws k uniformly in 1..4 instead of the learned contextual bandit.

PLUGIN API (one .py file per candidate, see plugins/):
    META = dict(id='...', title='...', slot='attention|pe|router|expert|storage|objective|hybrid|control', one_line='...')
    def patch(Block, Model, cfg): return Block2, Model2   # optional; subclass and override hooks, may edit cfg before build
  Block hooks: mix(h, causal) pe(x) attn_T() moe(z) kv_bytes_per_token()   Model hooks: hidden(ids, causal) forward_local(...)
  Big matrices MUST be BinLinear (or Block.lin) so that storage and streaming cost are measured from the real catalog.
  Optional module-level hooks: prepare(x,mode,rng,mask,deterministic=False) [training batches only; eval always uses the standard one],
    local_prepare(clean,c,group,sigma,rng)->(inputs,targets,sel,noise), update_filter(step,binlinear)->bool, post_step(model,step,c);
    cfg keys: schedule (list of modes), local_every. Model.take_extra_loss() returns+clears an extra loss added to the total.
    Use self.c (not a captured cfg) inside modules: patch() is also called with the real 600B config for the catalog census.
"""
import argparse, importlib.util, json, math, os, signal, sys, time, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

OBJECTIVE_VERSION = 'agillm-fit-v1'
DATA = os.environ.get('AGILLM_SIM_DATA', '/workspace/asi_agillm/data')
EXPERTS_600B = '22323232322323232323223232323223232323232232323232232323232322323232322323232323223232323223232323232232323232232323232322323232322323232323223232323223232323232232323232232323232322323232322323232323223232323223232323232232323232232323232322323232322323232323223232323223232323232232323232232323232322323232322323232323223232323223232323232232323232232323232322323232322323232323'
MODES = ('ar', 'sat_fixed', 'sat_variable', 'nat', 'ar', 'sat_variable', 'ar', 'ar')  # MULTIMODE_SCHEDULE
SIM = dict(V=8192, d=256, H=4, L=12, ff=1024, ctx=256, batch=8, local_group_size=4, router_balance=0.01, mask=8191, unk=8190,
           sign_lr=0.707, lr=1e-3, warmup=30, init_mag=32, local_every=9)
CFG_600 = dict(V=200064, d=8192, H=64, L=380, ff=32768, ctx=2048, batch=1, local_group_size=4, router_balance=0.01, mask=200063, unk=200063,
               sign_lr=0.707, lr=1e-3, warmup=30, init_mag=32, local_every=9, expert_counts=[int(c) for c in EXPERTS_600B])
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


def rope(x, base=10000.0):
    T, hd = x.shape[-2], x.shape[-1]
    f = base ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
    ang = torch.arange(T, device=x.device).float()[:, None] * f[None]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class BinLinear(nn.Module):
    """Mirror of TernLinear in the live bin8 mode: (d_out, d_in) int8 latents, binary read, one fp32 scale per row."""
    FP = False  # control only (fp_dense plugin): an ordinary fp32 matrix trained by AdamW instead of binary latents
    INIT = 32
    def __init__(self, d_in, d_out):
        super().__init__()
        self.shape = (d_out, d_in)
        if BinLinear.FP:
            self.register_buffer('P', torch.zeros(0, dtype=torch.int8)); self.w = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
        else:
            self.w = None
            self.register_buffer('P', (torch.randint(0, 2, (d_out, d_in), dtype=torch.int8) * 2 - 1) * BinLinear.INIT)
        self.scale = nn.Parameter(torch.full((d_out,), 1.0 / math.sqrt(d_in) if not BinLinear.FP else 1.0))
        self.register_buffer('R', torch.zeros(d_out, 1)); self.register_buffer('C', torch.zeros(1, d_in))
        self.register_buffer('t', torch.zeros((), dtype=torch.long))
        self._w = None
    def binary(self):
        if self.w is not None:
            return self.w
        if self._w is not None:
            return self._w
        w = torch.where(self.P >= 0, torch.ones((), device=self.P.device), -torch.ones((), device=self.P.device))
        if torch.is_grad_enabled():
            w.requires_grad_(True); self._w = w
        return w
    def forward(self, x):
        return F.linear(x, self.binary()) * self.scale
    @torch.no_grad()
    def latent_step(self, a, b2=0.99):
        """TernLinear.update, bin8 branch (TO.bin == 1). Returns (flips, seen, finite)."""
        w, self._w = self._w, None
        if self.w is not None or w is None or w.grad is None:
            return 0, 0, True
        g = w.grad
        if not bool(torch.isfinite(g.sum())):
            return 0, 0, False
        self.R.mul_(b2).add_(g.square().mean(1, keepdim=True) + 1e-30, alpha=1 - b2)
        self.C.mul_(b2).add_(g.square().mean(0, keepdim=True) + 1e-30, alpha=1 - b2)
        self.t += 1
        corr = 1 - b2 ** int(self.t)
        u = g * ((self.R / self.R.mean()) / corr).clamp_(min=1e-30).rsqrt_() * (self.C / corr).clamp_(min=1e-30).rsqrt_()
        rms = float((u.norm() / math.sqrt(u.numel())).clamp(min=1.0))
        Pf = self.P.float()
        new = (Pf + (torch.rand_like(u) + u * (-a / rms)).floor_()).clamp_(-127, 127)
        flips = int(((new >= 0) != (Pf >= 0)).sum())
        self.P.copy_(new.to(torch.int8))
        return flips, u.numel(), True


class BinTable(BinLinear):
    """Tied binary token table (V rows, d columns): embed and readout share the same latents, as --table bin."""
    def embed(self, ids):
        return self.binary()[ids] * self.scale[ids][..., None]
    def readout(self, h):
        return F.linear(h, self.binary()) * self.scale


class Expert(nn.Module):
    def __init__(self, c, lin):
        super().__init__(); self.up = lin(c['d'], c['ff']); self.nh = RMS(c['ff']); self.down = lin(c['ff'], c['d'])
    def forward(self, x):
        return self.down(self.nh(F.relu(self.up(x)).square()))


class Block(nn.Module):
    lin = staticmethod(lambda d_in, d_out: BinLinear(d_in, d_out))
    def __init__(self, c, index):
        super().__init__()
        d, ff = c['d'], c['ff']
        self.c, self.index, self.H = c, index, c['H']
        self.n1, self.n2, self.nh, self.no = RMS(d), RMS(d), RMS(ff), RMS(d)
        self.qkv, self.o = self.lin(d, 3 * d), self.lin(d, d)
        self.up, self.down = self.lin(d, ff), self.lin(ff, d)
        E = c['expert_counts'][index]
        self.extra_experts = nn.ModuleList(Expert(c, self.lin) for _ in range(E - 1))
        self.router = nn.Linear(d, E)
        self.init_router(index)
    def init_router(self, index):
        with torch.no_grad():
            if self.router.weight.is_meta:
                return
            g = torch.Generator(device='cpu').manual_seed(128000 + index)
            self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=g) * (0.01 / math.sqrt(self.router.in_features)))
            self.router.bias.zero_()
    def attn_T(self):
        return self.c.get('attn_T') or math.sqrt(self.c['d'] // self.H)
    def pe(self, x):
        return rope(x)
    def mix(self, h, causal):
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        hd = D // self.H
        q, k = self.pe(F.normalize(q, dim=-1) * math.sqrt(hd)), self.pe(F.normalize(k, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=self.attn_T() / math.sqrt(hd))
        return a.transpose(1, 2).reshape(B, T, D)
    def moe(self, z):
        probs = torch.softmax(self.router(z.detach().float()), dim=-1)
        sel = probs.argmax(-1); E = probs.shape[-1]
        load = F.one_hot(sel, E).float().mean(0)
        balance = E * (load.detach() * probs.mean(0)).sum()
        out = torch.zeros_like(z)
        for e in range(E):
            rows = torch.nonzero(sel == e, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            ze = z.index_select(0, rows)
            ye = self.down(self.nh(F.relu(self.up(ze)).square())) if e == 0 else self.extra_experts[e - 1](ze)
            out = out.index_copy(0, rows, ye)
        return out, balance
    def forward(self, x, causal):
        B, T, D = x.shape
        a = self.mix(self.n1(x), causal)
        x = x + self.o(self.no(a))
        out, aux = self.moe(self.n2(x).reshape(-1, D))
        return x + out.reshape(B, T, D), aux
    def kv_bytes_per_token(self):
        return 2 * self.c['d'] * 4  # one fp32 k and v per token for attention (live: 24,903,680 B/token at 600B)


Block.BinLinear = BinLinear; Block.RMS = RMS; Block.Expert = Expert; Block.rope = staticmethod(rope)


class Model(nn.Module):
    def __init__(self, c, BlockCls):
        super().__init__()
        self.c = c
        self.table = BinTable(c['d'], c['V'])
        self.blocks = nn.ModuleList(BlockCls(c, i) for i in range(c['L']))
        self.nf = RMS(c['d'])
        G = (c['L'] + c['local_group_size'] - 1) // c['local_group_size']
        self.local_conditioners = nn.ModuleList(nn.Linear(1, 2 * c['d']) for _ in range(G))
        self.local_norms = nn.ModuleList(RMS(c['d']) for _ in range(G))
        for cond in self.local_conditioners:
            nn.init.zeros_(cond.weight); nn.init.zeros_(cond.bias)
    def take_extra_loss(self):
        return 0.0   # hook: a plugin may accumulate an auxiliary loss during forward and return+clear it here
    def run_blocks(self, x, idx, causal):
        aux = []
        for i in idx:   # each block checkpointed on the GPU, as the live trainer does (one block's activations at a time)
            if torch.is_grad_enabled():
                x, a = _ckpt.checkpoint(self.blocks[i], x, causal, use_reentrant=False)
            else:
                x, a = self.blocks[i](x, causal)
            aux.append(a)
        return x, torch.stack(aux).mean()
    def hidden(self, ids, causal=True):
        x, aux = self.run_blocks(self.table.embed(ids), range(self.c['L']), causal)
        return self.nf(x), aux
    def readout(self, h):
        return self.table.readout(h)
    def logits(self, ids, causal=True):
        h, aux = self.hidden(ids, causal)
        return self.readout(h), aux
    def forward_local(self, ids, group, sigma, noise):
        x = math.sqrt(max(0.0, 1 - sigma ** 2)) * self.table.embed(ids) + sigma * noise.to(self.nf.g.device)
        t = x.new_tensor([[float(sigma)]])
        scale, shift = self.local_conditioners[group](t).chunk(2, dim=-1)
        x = x * (1 + 0.1 * torch.tanh(scale)) + shift
        start = group * self.c['local_group_size']
        x, aux = self.run_blocks(x, range(start, min(start + self.c['local_group_size'], self.c['L'])), False)
        return self.readout(self.local_norms[group](x)), aux


def census(ModelCls, BlockCls, cfg):
    """The candidate's storage/streaming bill at REAL 600B scale, from its actual module catalog (no training, no allocation)."""
    with torch.device('meta'):
        m = ModelCls(cfg, BlockCls)
    table = m.table
    body = [x for x in m.modules() if isinstance(x, BinLinear) and x is not table]
    latent_bytes = sum(x.shape[0] * x.shape[1] for x in body)
    fp_params = sum(p.numel() for p in m.parameters())
    return dict(latent_bytes=int(latent_bytes), table_bytes=int(table.shape[0] * table.shape[1]), n_matrices=len(body), fp_params=int(fp_params),
                kv_bytes_per_token=int(sum(b.kv_bytes_per_token() for b in m.blocks)))


def load_plugin(path):
    spec = importlib.util.spec_from_file_location('plugin_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare(x, mode, rng, mask, deterministic=False):
    """Mirror of MultimodeObjectives.prepare (GPU tensors)."""
    B, T = x.shape[0], x.shape[1] - 1
    inputs = x[:, :-1].clone(); targets = x[:, 1:].clone()
    selected = torch.ones_like(inputs, dtype=torch.bool); causal = True
    if mode in ('sat_fixed', 'sat_variable'):
        for b in range(B):
            i = 0
            while i < T:
                if mode == 'sat_fixed':
                    k = min(2, T - i)
                elif T - i >= 2:
                    k = min(int(rng.integers(1, 5)), T - i)
                else:
                    k = 1
                if k > 1:
                    inputs[b, i + 1:i + k] = mask
                i += k
    elif mode == 'nat':
        selected[:] = False
        targets = x[:, :-1].clone()
        for b in range(B):
            prefix = T // 2 if deterministic else min(T - 1, int(rng.integers(max(1, T // 4), max(2, 3 * T // 4 + 1))))
            inputs[b, prefix:] = mask; selected[b, prefix:] = True
        causal = False
    return inputs, targets[selected] if mode == 'nat' else targets, selected if mode == 'nat' else None, causal


def ce_eval(model, data_windows, c, mode, batch=8):
    tot, n = 0.0, 0
    rng = np.random.default_rng(7)
    for i in range(0, len(data_windows), batch):
        x = data_windows[i:i + batch].to(DEV)
        inputs, targets, sel, causal = prepare(x, mode, rng, c['mask'], deterministic=True)
        logits, _ = model.logits(inputs, causal=causal)
        rows = logits.reshape(-1, c['V']).float()
        if sel is not None:
            rows = rows[sel.reshape(-1)]
        tot += float(F.cross_entropy(rows, targets.reshape(-1), reduction='sum')); n += targets.numel()
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model, c, eval_windows, long_windows):
    model.eval()
    ar = ce_eval(model, eval_windows, c, 'ar')
    sat = ce_eval(model, eval_windows, c, 'sat_fixed')
    nat = ce_eval(model, eval_windows, c, 'nat')
    # context extension: NLL on the last 256 positions with the full 1024-token context vs only those 256 tokens as context
    full, trunc, n = 0.0, 0.0, 0
    for i in range(0, len(long_windows), 4):
        x = long_windows[i:i + 4].to(DEV)
        ids, tgt = x[:, :-1], x[:, 1:]
        h, _ = model.hidden(ids, causal=True)
        full += float(F.cross_entropy(model.readout(h[:, -256:]).reshape(-1, c['V']).float(), tgt[:, -256:].reshape(-1), reduction='sum'))
        h2, _ = model.hidden(ids[:, -256:], causal=True)
        trunc += float(F.cross_entropy(model.readout(h2).reshape(-1, c['V']).float(), tgt[:, -256:].reshape(-1), reduction='sum'))
        n += tgt[:, -256:].numel()
    model.train()
    return dict(ar=ar, sat=sat, nat=nat, ctx_full=full / n, ctx_256=trunc / n, ctx_gain=(trunc - full) / n)


@torch.no_grad()
def causality_check(model, c):
    model.eval()
    ids = torch.randint(0, c['V'] - 2, (2, 64), device=DEV)
    a, _ = model.logits(ids, causal=True)
    ids2 = ids.clone(); ids2[:, 40] = (ids2[:, 40] + 1) % (c['V'] - 2)
    b, _ = model.logits(ids2, causal=True)
    model.train()
    d = float((a[:, :40] - b[:, :40]).abs().max())
    return dict(max_abs_diff_before_edit=d, ok=bool(d < 1e-3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plugin', required=True); p.add_argument('--seed', type=int, default=1); p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--eval-at', default='150,300,600,1200'); p.add_argument('--out', required=True)
    p.add_argument('--mem-mib', type=int, default=int(os.environ.get('AGILLM_SIM_MEM_MIB', '720')))
    p.add_argument('--init-mag', type=int, default=None); p.add_argument('--sign-lr', type=float, default=None); p.add_argument('--lr', type=float, default=None)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    import hashlib
    res = dict(sim_sha256=hashlib.sha256(open(os.path.abspath(__file__),'rb').read()).hexdigest()[:16], objective=OBJECTIVE_VERSION, plugin=os.path.basename(a.plugin)[:-3], seed=a.seed, tag=a.tag, ok=False, started=time.time(), host=os.uname().nodename)
    def write():
        tmp = a.out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(res, f)
        os.replace(tmp, a.out)
    try:
        if DEV == 'cuda':
            torch.cuda.set_per_process_memory_fraction(min(0.5, a.mem_mib / (torch.cuda.get_device_properties(0).total_memory / 2**20)))
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        plugin = load_plugin(a.plugin)
        meta = dict(getattr(plugin, 'META', {})); meta.setdefault('id', res['plugin']); res['meta'] = meta
        c = dict(SIM)
        if a.init_mag is not None: c['init_mag'] = a.init_mag
        if a.sign_lr is not None: c['sign_lr'] = a.sign_lr
        if a.lr is not None: c['lr'] = a.lr
        c['expert_counts'] = [2 + int((i + 1) * 36 // 80 > i * 36 // 80) for i in range(c['L'])]
        BlockCls, ModelCls = Block, Model
        if hasattr(plugin, 'patch'):
            out = plugin.patch(Block, Model, c)
            if out is not None:
                BlockCls, ModelCls = out
        for key in ('V', 'd', 'H', 'L', 'ff', 'ctx', 'batch'):
            assert c[key] == SIM[key], f'plugin changed the frozen geometry field {key}'
        BinLinear.INIT = c['init_mag']
        V, T, B, d = c['V'], c['ctx'], c['batch'], c['d']
        res['cfg'] = {k: v for k, v in c.items()}
        # the catalog bill at real 600B scale
        cfg600 = dict(CFG_600)
        for k in ('attn_T', 'kv_rank'):
            if k in c: cfg600[k] = c[k]
        B600, M600 = BlockCls, ModelCls
        if hasattr(plugin, 'patch'):
            out600 = plugin.patch(Block, Model, cfg600)   # plugin may adjust expert_counts etc. for its 600B catalog too
            if out600 is not None:
                B600, M600 = out600
        res['census_600b'] = census(M600, B600, cfg600)
        model = ModelCls(c, BlockCls).to(DEV)
        bin_modules = [m for m in model.modules() if isinstance(m, BinLinear)]
        res['params_sim'] = dict(latents=int(sum(m.shape[0] * m.shape[1] for m in bin_modules if m.w is None)), fp=int(sum(p.numel() for p in model.parameters())))
        train = np.load(DATA + '/train.npy', mmap_mode='r'); test = np.load(DATA + '/test.npy', mmap_mode='r')
        res['data'] = json.load(open(DATA + '/meta.json'))
        smoke = os.environ.get('AGILLM_SIM_SMOKE') == '1'
        pos = np.linspace(0, len(test) - T - 2, 8 if smoke else 64).astype(np.int64)
        eval_windows = torch.from_numpy(np.stack([np.asarray(test[int(s):int(s) + T + 1]) for s in pos]).astype(np.int64))
        pos = np.linspace(0, len(test) - 1026, 2 if smoke else 16).astype(np.int64)
        long_windows = torch.from_numpy(np.stack([np.asarray(test[int(s):int(s) + 1025]) for s in pos]).astype(np.int64))
        res['causality'] = causality_check(model, c)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], betas=(0.9, 0.95), weight_decay=0.0)
        schedule = tuple(c.get('schedule', MODES))
        prep_train = getattr(plugin, 'prepare', None) or prepare
        update_filter = getattr(plugin, 'update_filter', None)
        post_step = getattr(plugin, 'post_step', None)
        body_mods = [m for m in bin_modules if m is not model.table and m.w is None]
        body_total = max(1, sum(m.shape[0] * m.shape[1] for m in body_mods))
        touched_sum = 0.0
        eval_at = set(int(s) for s in a.eval_at.split(','))
        data_rng = np.random.default_rng(1000)          # identical windows for every candidate (paired comparison)
        obj_rng = np.random.default_rng(2000 + a.seed)
        curve, local_log, step_ms, losses = [], [], [], {}
        attempts = cursor = local_cursor = skipped = 0
        flips = seen = 0
        t_start = time.time()
        G = len(model.local_norms)
        for step in range(a.steps):
            mult = min(1.0, (step + 1) / max(1, c['warmup']))
            sign_a = c['sign_lr'] * mult
            for g in opt.param_groups:
                g['lr'] = c['lr'] * mult
            starts = data_rng.integers(0, len(train) - T - 2, B)
            x = torch.from_numpy(np.stack([np.asarray(train[int(s):int(s) + T + 1]) for s in starts]).astype(np.int64)).to(DEV)
            timing = (step % 8 == 0 and DEV == 'cuda')
            if timing:
                torch.cuda.synchronize(); t0 = time.time()
            local = c['local_every'] and attempts % c['local_every'] == c['local_every'] - 1
            if local:
                group = local_cursor % G; local_cursor += 1
                low, high = 0.05 + 0.9 * (1 - (group + 1) / G), 0.05 + 0.9 * (1 - group / G)
                sigma = float(obj_rng.uniform(low, high))
                clean = x[:, :-1]
                chosen = obj_rng.random(tuple(clean.shape)) < sigma
                if not chosen.any():
                    chosen.flat[int(obj_rng.integers(chosen.size))] = True
                sel = torch.as_tensor(chosen, device=DEV, dtype=torch.bool)
                inputs = clean.clone(); inputs[sel] = c['mask']
                noise = torch.as_tensor(obj_rng.standard_normal((*clean.shape, d), dtype=np.float32), device=DEV)
                targets = clean[sel]
                if hasattr(plugin, 'local_prepare'):   # hook: replace the live local-denoise batch layout
                    inputs, targets, sel, noise = plugin.local_prepare(clean, c, group, sigma, obj_rng)
                logits, aux = model.forward_local(inputs, group, sigma, noise)
                rows = logits.reshape(-1, V)[sel.reshape(-1)]
                mode = 'local_denoise'
            else:
                mode = schedule[cursor % len(schedule)]; cursor += 1
                inputs, targets, sel, causal = prep_train(x, mode, obj_rng, c['mask'])
                logits, aux = model.logits(inputs, causal=causal)
                rows = logits.reshape(-1, V)
                if sel is not None:
                    rows = rows[sel.reshape(-1)]
                targets = targets.reshape(-1)
            attempts += 1
            loss = F.cross_entropy(rows.float(), targets)
            total = loss + c['router_balance'] * aux + model.take_extra_loss()
            opt.zero_grad(set_to_none=True)
            fine = bool(torch.isfinite(total))
            if fine:
                total.backward()
                touched_sum += sum(m.shape[0] * m.shape[1] for m in body_mods if m._w is not None and m._w.grad is not None) / body_total
                for m in bin_modules:
                    if update_filter is not None and m is not model.table and not update_filter(step, m):
                        m._w = None; continue
                    f, n, ok = m.latent_step(sign_a)
                    flips += f; seen += n; fine = fine and ok
                gs = [q.grad for q in model.parameters() if q.grad is not None]
                fine = fine and bool(torch.isfinite(sum(g.sum() for g in gs)))
                if fine:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            else:
                for m in bin_modules: m._w = None
            if not fine:
                skipped += 1
                if skipped > 30:
                    raise RuntimeError(f'diverged: {skipped} non-finite steps')
            if post_step is not None:
                post_step(model, step, c)
            if timing:
                torch.cuda.synchronize(); step_ms.append(1000 * (time.time() - t0))
            lv = float(loss.detach()) if math.isfinite(float(loss.detach())) else float('nan')
            losses.setdefault(mode, []).append(lv)
            if local:
                local_log.append((step, group, round(sigma, 4), round(lv, 4)))
            if step + 1 in eval_at:
                ev = evaluate(model, c, eval_windows, long_windows)
                ev.update(step=step + 1, tokens=(step + 1) * B * T, flip_pct=100 * flips / max(1, seen), skipped=skipped, wall_s=round(time.time() - t_start, 1))
                ev['train_recent'] = {k: float(np.nanmean(v[-20:])) for k, v in losses.items()}
                curve.append(ev); flips = seen = 0
                res['curve'] = curve; res['ms_per_step_median'] = float(np.median(step_ms)) if step_ms else None; res['touched_frac_mean'] = touched_sum / (step + 1)
                write()
                print(json.dumps({'step': step + 1, 'ar': round(ev['ar'], 4), 'sat': round(ev['sat'], 4), 'nat': round(ev['nat'], 4), 'ctx_gain': round(ev['ctx_gain'], 4),
                                  'flip_pct': round(ev['flip_pct'], 4), 'ms': res['ms_per_step_median'], 'mem_mib': int(torch.cuda.max_memory_allocated() / 2**20) if DEV == 'cuda' else 0}), flush=True)
        # local-denoise stability, the live run's HALT cause: spike = above 2x the running median of earlier local losses
        ll = [r[3] for r in local_log if math.isfinite(r[3])]
        spikes = sum(1 for i in range(10, len(ll)) if ll[i] > 2.0 * float(np.median(ll[:i])))
        res['local_denoise'] = dict(n=len(ll), mean_last20=float(np.mean(ll[-20:])) if ll else None, max=max(ll) if ll else None,
                                    spike_rate=spikes / max(1, len(ll) - 10), series_sample=local_log[::max(1, len(local_log) // 40)])
        res['steps'] = a.steps; res['skipped'] = skipped
        res['max_gpu_mib'] = int(torch.cuda.max_memory_allocated() / 2**20) if DEV == 'cuda' else 0
        res['wall_s'] = round(time.time() - t_start, 1)
        res['ok'] = True
    except Exception as e:
        res['error'] = repr(e)[:600]; res['traceback'] = traceback.format_exc()[-1800:]
        print('FAILED', res['error'], flush=True)
    res['finished'] = time.time()
    write()
    sys.exit(0 if res['ok'] else 1)


if __name__ == '__main__':
    main()
