"""DEPLOYABLE MIDDLE FORM of dlgm458.py (claude-fable52, 2026-11-12; the original files are untouched): router MLP and beta through the fixed 32-dim sign
projection as in dlgm458_lr.py, while the short-conv and local/mid conv kernels stay per channel (about 50 taps per channel). Run next to fusion879_lrk.py.
ASI-Arch #458 DualScaleMemoryRouter (delta_net_dlgm) on the agillm_mini binary block (claude-opus55-600b-quality, 2026-11-09).
Layout as hybrid600: attention at i%4==3 and the last layer (unchanged SDPA, follows the causal flag, carries the bidirectional NAT/local modes);
every other layer is DLGM and stays causal in every batch type. q,k,v from the block's existing binary qkv matrix (no new big matrices).
DLGM: 4-tap causal short conv (silu) on q,k,v; l2-normalised q,k; per-head beta; chunked delta-rule memory (chunk 16) + causal depthwise convs on v
(local 7 taps, mid 31 taps) + identity, mixed per token and head by a softmax router (identity logit bias +1.5 = ~70% identity at init), as in #458.
Two forms: dlgm458.py = faithful (per-channel kernels, full-width router MLP); dlgm458_lr.py = deployable at 600B scale (per-head kernels, router and
beta through a fixed random sign projection to 32 dims, 0.4% of the fp cost)."""
import math, os
import torch, torch.nn as nn, torch.nn.functional as F
FULL = False   # router and beta through the fixed 32-dim sign projection, as in dlgm458_lr.py; kernels stay per channel (see ch below)
META = dict(id='dlgm458_lrk', title='DualScaleMemoryRouter #458 (DLGM) hybrid, low-rank router with per-channel kernels', slot='hybrid',
            one_line='DLGM layers (local+mid causal conv, delta memory, identity, token/head softmax router) on 3/4 of layers; attention on every 4th')
def delta_chunked(q, k, v, beta, C=16):
    """Plain delta rule S_t = S_{t-1} + b_t k_t (v_t - S_{t-1}^T k_t)^T, o_t = S_t^T q_t, chunked exactly (UT transform). q,k,v (B,H,T,d), beta (B,H,T)."""
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
    S = q.new_zeros(B, H, d, d); outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ S
        outs.append(q[:, :, n] @ S + Aqk[:, :, n] @ U)
        S = S + k[:, :, n].transpose(-1, -2) @ U
    return torch.cat(outs, 2)[:, :, :T]
def conv(x, w, act=False):
    """causal depthwise conv over T of x (B,H,T,hd); w (H*hd,1,k) or (H,1,k) (per-head, shared across the head's channels)."""
    B, H, T, hd = x.shape
    ch = H * hd
    if w.shape[0] != ch:
        w = w.repeat_interleave(hd, dim=0)
    xc = x.permute(0, 1, 3, 2).reshape(B, ch, T)
    y = F.conv1d(F.pad(xc, (w.shape[-1] - 1, 0)), w, groups=ch)
    y = y.reshape(B, H, hd, T).permute(0, 1, 3, 2)
    return F.silu(y) if act else y
def patch(Block, Model, cfg):
    class DB(Block):
        def __init__(self, c, index):
            super().__init__(c, index)
            self.attn_layer = bool(index % 4 == 3 or index == c['L'] - 1)
            if self.attn_layer:
                return
            D, H = c['d'], self.H
            hd = D // H
            ch = H * hd   # per-channel kernels kept in this variant
            def k0(taps):
                w = torch.zeros(ch, 1, taps); w[:, :, -1] = 1.0
                return nn.Parameter(w)
            self.short_q, self.short_k, self.short_v = k0(4), k0(4), k0(4)
            self.lc = nn.Parameter(torch.randn(ch, 1, 7) / math.sqrt(7))
            self.mc = nn.Parameter(torch.randn(ch, 1, 31) / math.sqrt(31))
            Rin = D if FULL else 32
            if not FULL:
                self.register_buffer('rp', (torch.randint(0, 2, (32, D), dtype=torch.int8) * 2 - 1).float() / math.sqrt(D))
            self.b = nn.Linear(Rin, H, bias=True)
            self.rt = nn.Sequential(nn.Linear(Rin, 2 * Rin), nn.SiLU(), nn.Linear(2 * Rin, 4 * H))
            with torch.no_grad():
                if not self.rt[-1].bias.is_meta:
                    self.rt[-1].bias.zero_(); self.rt[-1].bias[3::4] = 1.5
        def mix(self, h, causal):
            if self.attn_layer:
                return super().mix(h, causal)
            B, T, D = h.shape
            H, hd = self.H, D // self.H
            q, k, v = self.qkv(h).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = conv(q, self.short_q, True), conv(k, self.short_k, True), conv(v, self.short_v, True)
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            z = h if FULL else h @ self.rp.t()
            beta = torch.sigmoid(self.b(z)).permute(0, 2, 1)
            delta = delta_chunked(q, k, v, beta)
            local, mid = conv(v, self.lc), conv(v, self.mc)
            p = self.rt(z).view(B, T, H, 4).permute(0, 2, 1, 3).softmax(-1)
            out = local * p[..., 0, None] + mid * p[..., 1, None] + delta * p[..., 2, None] + v * p[..., 3, None]
            return out.transpose(1, 2).reshape(B, T, D)
    return DB, Model
