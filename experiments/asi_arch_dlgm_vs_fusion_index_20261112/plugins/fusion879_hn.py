"""ROBUSTNESS VARIANT of fusion879.py: identical except that the fused output is RMS-normalised per head before it leaves the mixer, as the upstream
o_norm does (the main port keeps the block-wide norm only, the convention of dlgm458.py). Run next to dlgm458_hn.py to see whether the comparison depends on that convention.
ASI-Arch #879 FusionGatedFIRNet (delta_net_psfr: Parallel Sigmoid Fusion with Retention) on the agillm_mini binary block (claude-fable52, 2026-11-12).
Ported with the same layout and conventions as dlgm458.py so the two ASI-Arch finalists are measured like for like: attention at i%4==3 and the last
layer (unchanged SDPA, follows the causal flag, carries the bidirectional NAT/local modes); every other layer is Fusion and stays causal in every batch
type. q,k,v from the block's existing binary qkv matrix (no new big matrices); the block's own output norm and binary o projection are kept.
Fusion #879, as in the upstream source: 4-tap causal short conv (silu) on q,k,v; l2-normalised q,k; per-head beta; chunked delta-rule memory whose
carried state is multiplied by a learned per-head retention lambda in [0.6,1] at every 32-token chunk boundary (lambda ~1 at init); two Dirac-initialised
causal depthwise FIR branches on v (short 3 taps, long 63 taps); the value itself. Each of the four paths gets its OWN sigmoid gate in [0.02,1] per token
and head (no softmax budget): gate MLP on the hidden state plus the mean |x| of every path per head, bias short -1 / long -1 / delta +1 / value +3,
learned per-head temperature. The mixer equals the upstream DeltaNet class when its weights are copied in (plugins/_fusion879_test/test_fusion879_equiv.py).
Two forms: fusion879.py = faithful (per-channel kernels, full-width gate MLP); fusion879_lr.py = deployable at 600B scale (per-head kernels, gate and
beta through a fixed random sign projection to 32 dims), the same reduction as dlgm458_lr.py."""
import json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
FULL = not os.path.basename(__file__).endswith('_lr.py')
META = dict(id='fusion879_hn', title='FusionGatedFIRNet #879 (Fusion) hybrid, per-head output norm', slot='hybrid',
            one_line='Fusion layers (short+long causal FIR, delta memory with per-head retention, value; an independent sigmoid gate per path, token and head) on 3/4 of layers; attention on every 4th')
EPS, LAM_MIN, CHUNK = 0.02, 0.6, 32
def l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)   # fla.modules.l2norm
def delta_retention_chunked(q, k, v, beta, lam, C=CHUNK):
    """Upstream _retention_delta_chunkwise. Inside each C-token chunk the plain delta rule S_t = S_{t-1} + b_t k_t (v_t - S_{t-1}^T k_t)^T, o_t = S_t^T q_t
    (exact, UT transform); at every chunk boundary the carried state becomes lam * S_chunk_start + (the chunk's update).
    q,k (B,H,T,d) unit norm, v (B,H,T,d), beta (B,H,T), lam (H,)."""
    B, H, T, d = q.shape
    pad = (-T) % C
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta = F.pad(beta, (0, pad))
    N = (T + pad) // C
    q, k, v = (x.reshape(B, H, N, C, d) for x in (q, k, v))
    b = beta.reshape(B, H, N, C).unsqueeze(-1)
    I = torch.eye(C, dtype=q.dtype, device=q.device)
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
    incl = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), 0)
    Lm = (b * (k @ k.transpose(-1, -2))).masked_fill(~strict, 0.0) + I
    inv = torch.linalg.solve_triangular(Lm, I.expand_as(Lm), upper=False, unitriangular=True)
    W, U0 = inv @ (b * k), inv @ (b * v)
    Aqk = (q @ k.transpose(-1, -2)).masked_fill(~incl, 0.0)
    lam = lam.view(1, H, 1, 1).to(q.dtype)
    S = q.new_zeros(B, H, d, d); outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ S
        outs.append(q[:, :, n] @ S + Aqk[:, :, n] @ U)
        S = S * lam + k[:, :, n].transpose(-1, -2) @ U
    return torch.cat(outs, 2)[:, :, :T]
def conv(x, w, act=False):
    """causal depthwise conv over T of x (B,H,T,hd); w (H*hd,1,k) or (H,1,k) (per-head, shared across the head's channels). Same helper as dlgm458.py."""
    B, H, T, hd = x.shape
    ch = H * hd
    if w.shape[0] != ch:
        w = w.repeat_interleave(hd, dim=0)
    xc = x.permute(0, 1, 3, 2).reshape(B, ch, T)
    y = F.conv1d(F.pad(xc, (w.shape[-1] - 1, 0)), w, groups=ch)
    y = y.reshape(B, H, hd, T).permute(0, 1, 3, 2)
    return F.silu(y) if act else y
def post_step(model, step, c):
    """Log only, no effect on training: what retention, temperature and the four gates look like at the evaluation steps."""
    if step + 1 not in (150, 300, 600, 1200):
        return
    with torch.no_grad():
        rows = [b for b in model.blocks if not getattr(b, 'attn_layer', True) and hasattr(b, 'last_p')]
        if not rows:
            return
        lam = torch.stack([LAM_MIN + (1.0 - LAM_MIN) * torch.sigmoid(b.ret) for b in rows])
        temp = torch.stack([b.log_temp.exp() for b in rows])
        gates = torch.stack([b.last_p for b in rows]).mean((0, 1))
        print(json.dumps({'fusion_state_step': step + 1, 'lambda_min': round(float(lam.min()), 6), 'lambda_mean': round(float(lam.mean()), 6),
                          'temp_min': round(float(temp.min()), 4), 'temp_max': round(float(temp.max()), 4),
                          'mean_gate_short_long_delta_value': [round(float(x), 4) for x in gates]}), flush=True)
def patch(Block, Model, cfg):
    class FB(Block):
        def __init__(self, c, index):
            super().__init__(c, index)
            self.attn_layer = bool(index % 4 == 3 or index == c['L'] - 1)
            if self.attn_layer:
                return
            D, H = c['d'], self.H
            hd = D // H
            ch = H * hd if FULL else H
            def dirac(taps, noise=0.0):
                w = torch.zeros(ch, 1, taps); w[:, :, -1] = 1.0
                if noise:
                    w = w + torch.randn(ch, 1, taps) * noise
                return nn.Parameter(w)
            self.short_q, self.short_k, self.short_v = dirac(4), dirac(4), dirac(4)
            self.fs, self.fl = dirac(3, 1e-2), dirac(63, 1e-2)
            Rin = D if FULL else 32
            if not FULL:
                self.register_buffer('rp', (torch.randint(0, 2, (32, D), dtype=torch.int8) * 2 - 1).float() / math.sqrt(D))
            self.b = nn.Linear(Rin, H, bias=True)
            self.ret = nn.Parameter(torch.full((H,), math.log((1 - 1e-4) / 1e-4)))   # upstream retention_init 1.0 -> lambda 0.99996
            self.gate = nn.Sequential(nn.Linear(Rin + 4 * H, 2 * Rin), nn.GELU(), nn.Linear(2 * Rin, 4 * H))
            self.log_temp = nn.Parameter(torch.zeros(H))
            with torch.no_grad():
                if not self.gate[-1].bias.is_meta:
                    self.gate[-1].bias.copy_(torch.tensor([-1.0, -1.0, 1.0, 3.0]).repeat(H))   # per head: short, long, delta, value
        def mix(self, h, causal):
            if self.attn_layer:
                return super().mix(h, causal)
            B, T, D = h.shape
            H, hd = self.H, D // self.H
            q, k, v = self.qkv(h).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = conv(q, self.short_q, True), conv(k, self.short_k, True), conv(v, self.short_v, True)
            q, k = l2norm(q), l2norm(k)
            z = h if FULL else h @ self.rp.t()
            beta = torch.sigmoid(self.b(z)).permute(0, 2, 1)
            lam = LAM_MIN + (1.0 - LAM_MIN) * torch.sigmoid(self.ret)
            delta = delta_retention_chunked(q, k, v, beta, lam)
            short, long_ = conv(v, self.fs), conv(v, self.fl)
            stats = torch.cat([x.abs().mean(-1) for x in (short, long_, delta, v)], 1).permute(0, 2, 1)   # (B,T,4H): short heads, long heads, delta heads, value heads
            g = self.gate(torch.cat([z, stats], -1)).view(B, T, H, 4).permute(0, 2, 1, 3) / self.log_temp.exp().view(1, H, 1, 1)
            p = EPS + (1.0 - EPS) * torch.sigmoid(g)
            self.last_p = p.detach().mean((0, 2))   # (H,4) mean gate per head and path, for the log line only
            out = short * p[..., 0, None] + long_ * p[..., 1, None] + delta * p[..., 2, None] + v * p[..., 3, None]
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)   # per-head RMS as in the upstream o_norm (its learnable gain is the block's own norm gain)
            return out.transpose(1, 2).reshape(B, T, D)
    return FB, Model
