"""OM3 arm `gdn`: Gated DeltaNet (delta group, Claude lane, 22 Nov 2026).

Source
  Songlin Yang, Jan Kautz, Ali Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule",
  arXiv:2412.06464 (ICLR 2025). The layer layout follows the authors' reference layer (NVlabs/GatedDeltaNet,
  also shipped as flash-linear-attention fla/layers/gated_deltanet.py): q/k/v projections, short causal
  convolutions with SiLU, L2-normalised q and k, sigmoid beta, Mamba-2 style decay, gated RMS output norm,
  output projection.

Equations implemented (per layer and head; x_t = RMSNorm(h_t) is the block input; S_t is d_v x d_k)
  q_t = l2(SiLU(conv4(W_q x)_t)) / sqrt(d_k)    k_t = l2(SiLU(conv4(W_k x)_t))    v_t = SiLU(conv4(W_v x)_t)
  beta_t  = sigmoid(W_b x_t)
  alpha_t = exp(-softplus(W_a x_t + dt_bias) * exp(A_log))                  (Mamba-2 style gate)
  S_t = alpha_t (S_{t-1} - beta_t (S_{t-1} k_t) k_t^T) + beta_t v_t k_t^T     (gated delta rule)
  o_t = S_t q_t
  y_t = W_out concat_heads[ RMSNorm(o_t) * SiLU(W_g x_t) ]  (RMSNorm over d_v, weight shared by the heads)
  block: h <- h + y(RMSNorm(h)); byte embedding tied to the output layer, final RMSNorm (benchmark convention)
  l2(z) = z / sqrt(|z|^2 + 1e-6); conv4 = causal depthwise convolution of width 4 without bias; the
  1/sqrt(d_k) query scale is the reference kernel's default scale.

Training form: the paper's chunkwise algorithm (WY representation / UT transform extended with the gate),
chunk C = 64. In a chunk with incoming state Z = S_0^T (d_k x d_v), g_t = sum_{i<=t} log alpha_i inside the
chunk, Gam[t, j] = exp(g_t - g_j) for j <= t and 0 otherwise:
  A        = strictly_lower( diag(beta) (K K^T) * Gam )
  [U0 | W] = (I + A)^{-1} [ diag(beta) V | diag(beta * exp(g)) K ]     (exact unit-triangular solve)
  U        = U0 - W Z          (row t = beta_t (v_t - alpha_t S_{t-1} k_t)^T, the delta-rule "new value")
  O        = diag(exp(g)) Q Z + ((Q K^T) * Gam) U
  Z_next   = exp(g_C) Z + (diag(exp(g_C - g)) K)^T U
This equals the recurrence exactly in real arithmetic. step() runs the recurrence itself (in the paper's
d_v x d_k form above), and the benchmark parity gate (forward vs step, max abs < 1e-4) checks one against
the other.

Size: 4 blocks, d_model 128, 4 heads, d_k 32, d_v 48 (state 4 x 48 x 32 per layer) -> 470,880 parameters
(target 471,264, -0.08 %).

Simplifications versus the paper (complete list)
  1. Mixer-only blocks: the paper stacks every GDN token mixer with a SwiGLU MLP (Llama macro design). At this
     budget the suggested GDN width already fills the ~110k-parameter block, so there is no MLP (as in the
     om2 Mamba-2 arm of this benchmark).
  2. Scale and data: 4 layers, d_model 128, d_k 32, d_v 48, byte tokens, tied embeddings, context 256; the
     paper trains models of up to 1.3B parameters on subword text. Its hybrids (H1/H2, with sliding-window
     attention and Mamba-2 layers) are not used.
  3. Initialisation: PyTorch defaults for the linear and conv weights (the reference code re-initialises them
     from a small-std normal distribution, with scaled-down output projections); A ~ U(0, 16) and the
     inverse-softplus dt_bias with dt in [1e-3, 0.1] follow the reference code.
  4. Kernels: float32 PyTorch on CPU with torch.linalg.solve_triangular for (I + A)^{-1}, instead of the
     paper's Triton kernels (forward substitution on GPU); the mathematics is the same.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

HEADS, DK, DV, KCONV, CHUNK = 4, 32, 48, 4, 64


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_chunked(q, k, v, beta, la, C=CHUNK):
    """Exact chunkwise gated delta rule from a zero state.

    q, k: (B, T, H, dk), L2-normalised (q already scaled); v: (B, T, H, dv); beta and la = log alpha: (B, T, H).
    Returns o: (B, T, H, dv) with o_t = S_t q_t.
    """
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % C
    if pad:                      # padded tail after every real token: q = k = v = beta = 0, alpha = 1
        q, k, v = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (q, k, v))
        beta, la = F.pad(beta, (0, 0, 0, pad)), F.pad(la, (0, 0, 0, pad))
    n = (T + pad) // C
    q, k, v = (z.reshape(B, n, C, H, -1).permute(0, 3, 1, 2, 4) for z in (q, k, v))   # B H n C d
    beta, la = (z.reshape(B, n, C, H).permute(0, 3, 1, 2) for z in (beta, la))       # B H n C
    g = la.cumsum(-1)
    gam = torch.exp(ob.segsum(la))                       # exp(g_t - g_j) for j <= t, 0 above the diagonal
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
    kb = k * beta[..., None]
    A = ((kb @ k.transpose(-1, -2)) * gam).masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=k.dtype)
    rhs = torch.cat((v * beta[..., None], kb * g.exp()[..., None]), -1)
    U0, W = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True).split((dv, dk), -1)
    qk = (q @ k.transpose(-1, -2)) * gam
    qg = q * g.exp()[..., None]
    kd = k * (g[..., -1:] - g).exp()[..., None]
    gC = g[..., -1].exp()[..., None, None]
    Z = k.new_zeros(B, H, dk, dv)
    out = []
    for c in range(n):
        U = U0[:, :, c] - W[:, :, c] @ Z
        out.append(qg[:, :, c] @ Z + qk[:, :, c] @ U)
        Z = gC[:, :, c] * Z + kd[:, :, c].transpose(-1, -2) @ U
    return torch.stack(out, 2).permute(0, 2, 3, 1, 4).reshape(B, n * C, H, dv)[:, :T]


class GDNMixer(nn.Module):
    def __init__(self, d=ob.D_MODEL, H=HEADS, dk=DK, dv=DV, kconv=KCONV):
        super().__init__()
        self.H, self.dk, self.dv, self.kc = H, dk, dv, kconv
        self.cw = H * (2 * dk + dv)                                       # conv channels: q | k | v
        self.inp = nn.Linear(d, self.cw + H * dv + 2 * H, bias=False)    # W_q W_k W_v | W_g | W_a | W_b
        self.conv = nn.Conv1d(self.cw, self.cw, kconv, groups=self.cw, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(0, 16).clamp_min(1e-4)))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.onorm = ob.RMS(dv)
        self.out = nn.Linear(H * dv, d, bias=False)

    def _split(self, u):
        return self.inp(u).split((self.cw, self.H * self.dv, self.H, self.H), -1)

    def _qkv(self, c):
        sh = c.shape[:-1]
        q, k, v = c.split((self.H * self.dk, self.H * self.dk, self.H * self.dv), -1)
        q = l2n(q.reshape(*sh, self.H, self.dk)) * self.dk ** -0.5
        return q, l2n(k.reshape(*sh, self.H, self.dk)), v.reshape(*sh, self.H, self.dv)

    def _gates(self, a, b):
        return torch.sigmoid(b), -F.softplus(a + self.dt_bias) * self.A_log.exp()    # beta, log alpha

    def _out(self, o, gate):
        return self.out((self.onorm(o) * F.silu(gate.reshape(o.shape))).flatten(-2))

    def forward(self, u):
        c, gate, a, b = self._split(u)
        c = F.silu(self.conv(F.pad(c.transpose(1, 2), (self.kc - 1, 0))).transpose(1, 2))
        q, k, v = self._qkv(c)
        beta, la = self._gates(a, b)
        return self._out(gated_delta_chunked(q, k, v, beta, la), gate)

    def step(self, u, cache=None):
        c, gate, a, b = self._split(u)
        if cache is None:
            conv = u.new_zeros(u.shape[0], self.cw, self.kc - 1)
            S = u.new_zeros(u.shape[0], self.H, self.dv, self.dk)
        else:
            conv, S = cache
        win = torch.cat((conv, c[..., None]), -1)
        q, k, v = self._qkv(F.silu((win * self.conv.weight[:, 0]).sum(-1)))
        beta, la = self._gates(a, b)
        al, be, kr = la.exp()[..., None, None], beta[..., None, None], k[:, :, None, :]
        S = al * (S - be * (S @ k[..., None]) * kr) + be * v[..., None] * kr
        return self._out((S @ q[..., None])[..., 0], gate), (win[..., 1:], S)


def build():
    return ob.SSMLM([GDNMixer() for _ in range(ob.LAYERS)])
