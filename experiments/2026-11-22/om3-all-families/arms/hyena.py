"""Hyena operator of order 2 (Poli et al. 2023) as a 4-layer byte-level language model for the OM3 benchmark.

Source paper
  M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, C. Re,
  "Hyena Hierarchy: Towards Larger Convolutional Language Models", ICML 2023, arXiv:2302.10866:
  Definition 3.1 (eq. 4, the Hyena recurrence), Algorithms 1-3 (projection, filter, forward pass),
  eq. 7 (implicit filter h_t = Window(t) * (FFN o PositionalEncoding)(t)), Fig. 3.1 (exponential window plus a
  bias), Appendix D.3 (truncated complex exponential positional encoding), Table A.4 (filter FFN width 64,
  depth 4, sine frequency 14). Details the paper leaves open come from the authors' public reference code
  (HazyResearch "safari": standalone_hyena.py, src/models/sequence/hyena.py, long_conv_lm.py), marked below.

Equations implemented (one layer; width D = 128, order N = 2, sequence length T; c = channel)
  Projection (Alg. 1)  p_t = W_in u_t + b_in in R^{(N+1)D};   q = DepthwiseConv1d_{k=3}(p), causal, with bias:
                       q_t = w_0 p_{t-2} + w_1 p_{t-1} + w_2 p_t + b  (per channel);   (x^1, x^2, v) = split(q)
  Filters (Alg. 2, eq. 7, App. D.3), tap i = 0, 1, ...:   t_i = i / (l_max - 1),  l_max = 256
                       PE(i) = [t_i, cos(2 pi f_k i / l_max)_{k<K}, sin(2 pi f_k i / l_max)_{k<K}] in R^{2K+1},
                       f = linspace(1e-4, K-1, K), K = 8 (17 features)
                       FFN(e) = W4 sin(w*(W3 sin(w*(W2 sin(w*(W1 e + b1)) + b2)) + b3)) in R^{N x D}
                       (width 64, depth 4; w = one trainable frequency vector, initialised to 14)
                       h^n_{c,i} = FFN(PE(i))_{n,c} * (exp(-alpha_c t_i) + b)
                       alpha_c = |linspace(ln(1e-2)/1.5, ln(1e-2)/0.3, D)|_c (fixed),   b = 0.05 (fixed)
  Recurrence (Def. 3.1, Alg. 3)
                       z^1 = v;   z^{n+1}_t = x^n_t * ( sum_{s=0..t} h^n_{t-s} z^n_s + Dskip^n_c z^n_t ),  n = 1, 2
                       Hyena(u)_t = W_out z^3_t + b_out,  i.e.  W_out( x^2 * (h^2 conv (x^1 * (h^1 conv v))) )
    forward(): each long convolution is a causal FFT convolution with input and filter zero-padded to 2T,
               (h conv z) = irfft(rfft(z, 2T) * rfft(h, 2T))[:T]; its backward is the transposed convolution
               (a correlation), computed with FFTs of the same size.
    step():    the same convolutions as an exact direct sum over the stored history of z^1 and z^2, plus the last
               two projected inputs for the short convolution, so stepping token by token reproduces forward().
  Block and LM (benchmark shell shared with the other arms)
                       x <- x + Hyena(RMS(x));   x <- x + W_f2 gelu(W_f1 RMS(x))   (bias-free FFN, width 64)
                       logits = RMS(h_4) E^T with the byte embedding E (init std 0.02) tied to the output.

Simplifications and choices versus the paper
  1. Shell: byte vocabulary, 4 layers, d_model 128, RMSNorm pre-norm (the paper's GPT-style backbone uses
     LayerNorm), tied embeddings, final RMSNorm, no dropout, and a GELU FFN of width 64 = 0.5 d (paper: 2-4 d)
     so that the parameter count meets the benchmark target: 474,496 parameters (+0.69% vs 471,264). Block
     projections (W_in, W_out, FFN, short convolution) keep PyTorch's default init, like the other arms.
  2. Order: Definition 3.1 / Algorithms 2-3 literally, N = 2 with TWO implicit long filters h^1, h^2, as the task
     specifies. The public reference code's order-N operator has N-1 long filters (its first gate follows only
     the short convolution), so, run with that code, the paper's order-2 Pile models have one long filter.
  3. Positional encoding: K = 8 bands, 17 features (as in the paper's Fig. D.6 and D.9; the LM value is not
     stated). From the reference code: normalised time t = i/(l_max-1) as the first feature, first band
     frequency 1e-4 instead of 0 (no constant-zero feature), and a fixed encoding (the reference trains it with
     lr 1e-5 or 0). l_max = 256, the benchmark context; taps past l_max, which the benchmark never reaches, come
     from the same analytic formulas.
  4. Filter FFN: width 64, depth 4, sine activation, frequency initialised at 14 (Table A.4); one trainable
     frequency vector shared by the three sine layers (the reference reuses one Sin module). Its Linear layers
     start at N(0, 0.02) with zero biases, as the authors' LM code initialises every Linear (GPT-2 style
     _init_weights in long_conv_lm.py). With PyTorch's default init instead, the training loss over steps
     251-300 of a 300-step run of the benchmark recipe was 3.42 instead of 2.97 bits per byte (same seed, data).
  5. Window: exp(-alpha_c t) + b. The decay range (1e-2 reached at 30% .. 150% of l_max across channels) is the
     reference code's; the paper only says alpha varies across channels. The paper adds a bias to the window
     without giving its value: we use a fixed b = 0.05. Both filters use the same set of decay rates.
  6. Each long convolution carries the reference code's per-channel skip Dskip (fftconv's "+ u * D", a 1-D
     parameter initialised N(0, 1)); the paper's equations write h * z only. It equals adding Dskip to tap h_0.
  7. FFT size 2T (the paper says 2L-1; same result). The forward FFT runs in float64 (its inputs, outputs, all
     parameters and the logits are float32; the backward FFTs are float32). With a float32 forward FFT the
     round-off lets later bytes move earlier logits (measured on this model: 8.2e-7 at init for T = 96, 9.1e-7
     for T = 256, 5.2e-6 and 5.7e-6 after 300 training steps), so the benchmark's 1e-6 causality check fails
     once the model trains; with the float64 forward the measured change is exactly 0 at init and after
     training. Cost: 286 vs 247 ms per training step (+16%, same process). FFT_DTYPE = torch.float32 restores
     the reference precision.
  8. Optimiser: the benchmark's single AdamW setting covers every parameter (its weight decay hits 2-D weights,
     including the filter FFN's); the paper/reference give filter parameters their own lr and zero weight decay.
  9. Decoding: the paper gives no recurrent form for implicit filters; step() keeps the whole history (O(t) work
     per token per long filter): exact, but the state grows with the context, unlike an SSM.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import om2bench as ob

D_MODEL, LAYERS, VOCAB = ob.D_MODEL, ob.LAYERS, ob.VOCAB
ORDER = 2              # N: number of long filters / gates
L_MAX = 256            # filter time axis is normalised to the benchmark context
SHORT_K = 3            # short explicit depthwise convolution
PE_BANDS = 8           # K complex exponentials -> 2K+1 = 17 positional features
FILTER_WIDTH = 64      # Table A.4
FILTER_DEPTH = 4       # Table A.4 (number of Linear layers in the filter FFN)
SINE_FREQ = 14.0       # Table A.4
FILTER_INIT_STD = 0.02  # reference LM trainer init of the filter FFN (docstring item 4)
DECAY_FAST, DECAY_SLOW, DECAY_TARGET = 0.3, 1.5, 1e-2
WINDOW_BIAS = 0.05
FF = 64                # block FFN width, sized for the parameter target
FFT_DTYPE = torch.float64  # precision of the forward FFT only (docstring item 7); float32 = reference precision


class CausalFFTConv(torch.autograd.Function):
    """y_t = sum_{s<=t} h_{t-s} z_s for z (B, C, T), h (C, T): FFT convolution with zero padding to 2T.

    Forward in FFT_DTYPE. Backward in the gradient's dtype: dz = correlation of dy with h, and dh = correlation
    of dy with z summed over the batch, both as FFTs with the same zero padding.
    """

    @staticmethod
    def forward(ctx, z, h):
        T = z.shape[-1]
        n = 2 * T
        zf = torch.fft.rfft(z.to(FFT_DTYPE), n=n)
        hf = torch.fft.rfft(h.to(FFT_DTYPE), n=n)
        ctx.save_for_backward(zf, hf)
        return torch.fft.irfft(zf * hf, n=n)[..., :T].to(z.dtype)

    @staticmethod
    def backward(ctx, gy):
        zf, hf = ctx.saved_tensors
        T = gy.shape[-1]
        n = 2 * T
        gf = torch.fft.rfft(gy, n=n)
        gz = gh = None
        if ctx.needs_input_grad[0]:
            gz = torch.fft.irfft(gf * hf.to(gf.dtype).conj(), n=n)[..., :T]
        if ctx.needs_input_grad[1]:
            gh = torch.fft.irfft((gf * zf.to(gf.dtype).conj()).sum(0), n=n)[..., :T]
        return gz, gh


def fft_causal_conv(z, h):
    return CausalFFTConv.apply(z, h)


class HyenaFilter(nn.Module):
    """Implicit long filters h^1..h^N (Algorithm 2): FFN on a positional encoding, times an exponential window."""

    def __init__(self, d=D_MODEL, order=ORDER, l_max=L_MAX, bands=PE_BANDS, width=FILTER_WIDTH,
                 depth=FILTER_DEPTH, freq=SINE_FREQ, init_std=FILTER_INIT_STD, fast=DECAY_FAST, slow=DECAY_SLOW,
                 target=DECAY_TARGET, window_bias=WINDOW_BIAS):
        super().__init__()
        self.d, self.order, self.l_max, self.window_bias = d, order, l_max, window_bias
        dims = [2 * bands + 1] + [width] * (depth - 1)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.last = nn.Linear(width, order * d, bias=False)
        for lin in list(self.hidden) + [self.last]:
            nn.init.normal_(lin.weight, std=init_std)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        self.freq = nn.Parameter(torch.full((width,), float(freq)))
        self.skip = nn.Parameter(torch.randn(order * d))      # 1-D, like the reference filter bias
        self.register_buffer('bands', torch.linspace(1e-4, bands - 1, bands), persistent=False)
        alpha = torch.linspace(math.log(target) / slow, math.log(target) / fast, d).abs()
        self.register_buffer('alpha', alpha, persistent=False)

    def features(self, L):
        i = torch.arange(L, dtype=torch.float32, device=self.alpha.device)
        t = i / (self.l_max - 1)
        ang = (2.0 * math.pi / self.l_max) * i[:, None] * self.bands[None, :]
        return torch.cat((t[:, None], torch.cos(ang), torch.sin(ang)), -1), t

    def forward(self, L):
        """Taps 0..L-1 of every filter: (order, d, L)."""
        g, t = self.features(L)
        for lin in self.hidden:
            g = torch.sin(self.freq * lin(g))
        g = self.last(g).view(L, self.order, self.d)
        window = torch.exp(-t[:, None] * self.alpha[None, :]) + self.window_bias
        return (g * window[:, None, :]).permute(1, 2, 0)


class HyenaOperator(nn.Module):
    """y = W_out( x^N * (h^N conv ( ... x^1 * (h^1 conv v) ... )) ), Definition 3.1 / Algorithm 3."""

    def __init__(self, d=D_MODEL, order=ORDER, short_k=SHORT_K, **filter_kw):
        super().__init__()
        self.d, self.order, self.k = d, order, short_k
        w = (order + 1) * d
        self.in_proj = nn.Linear(d, w)
        self.short = nn.Conv1d(w, w, short_k, groups=w)
        self.filter = HyenaFilter(d, order, **filter_kw)
        self.out_proj = nn.Linear(d, d)

    def forward(self, u):
        T = u.shape[1]
        q = self.short(F.pad(self.in_proj(u).transpose(1, 2), (self.k - 1, 0)))     # (B, (N+1)d, T), causal
        *xs, z = q.split(self.d, 1)                                                   # x^1..x^N, v
        h = self.filter(T)                                                            # (N, d, T)
        skip = self.filter.skip.view(self.order, self.d)
        for n in range(self.order):
            z = xs[n] * (fft_causal_conv(z, h[n]) + skip[n][:, None] * z)
        return self.out_proj(z.transpose(1, 2))

    def step(self, u, cache):
        p = self.in_proj(u)                                                           # (B, (N+1)d)
        if cache is None:
            conv = p.new_zeros(p.shape[0], p.shape[1], self.k - 1)
            hist = [p.new_zeros(p.shape[0], self.d, 0) for _ in range(self.order)]
            taps = None
        else:
            conv, hist, taps = cache
        t = hist[0].shape[-1]
        if taps is None or taps.shape[-1] <= t:
            taps = self.filter(max(self.filter.l_max, 2 * (t + 1)))                 # (N, d, Lc), Lc > t
        win = torch.cat((conv, p[..., None]), -1)                                     # p_{t-2}, p_{t-1}, p_t
        q = (win * self.short.weight[:, 0]).sum(-1) + self.short.bias
        *xs, z = q.split(self.d, -1)
        skip = self.filter.skip.view(self.order, self.d)
        new = []
        for n in range(self.order):
            zs = torch.cat((hist[n], z[..., None]), -1)                               # z^n_0 .. z^n_t
            new.append(zs)
            z = xs[n] * ((zs * taps[n, :, :t + 1].flip(-1)).sum(-1) + skip[n] * z)
        return self.out_proj(z), (win[..., 1:], new, taps)


class HyenaBlock(nn.Module):
    def __init__(self, d=D_MODEL, ff=FF, **kw):
        super().__init__()
        self.n1, self.mix, self.n2 = ob.RMS(d), HyenaOperator(d, **kw), ob.RMS(d)
        self.f1, self.f2 = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def _ffn(self, x):
        return x + self.f2(F.gelu(self.f1(self.n2(x))))

    def forward(self, x):
        return self._ffn(x + self.mix(self.n1(x)))

    def step(self, x, cache):
        y, cache = self.mix.step(self.n1(x), cache)
        return self._ffn(x + y), cache


class HyenaLM(nn.Module):
    def __init__(self, d=D_MODEL, layers=LAYERS, ff=FF, **kw):
        super().__init__()
        self.e = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.e.weight, std=0.02)
        self.blocks = nn.ModuleList(HyenaBlock(d, ff, **kw) for _ in range(layers))
        self.n = ob.RMS(d)
        self.last_penalty = None

    def forward(self, x, hard=False):
        h = self.e(x)
        for b in self.blocks:
            h = b(h)
        return F.linear(self.n(h), self.e.weight)

    def step(self, tok, state, hard=True):
        caches = state if state is not None else [None] * len(self.blocks)
        h = self.e(tok)
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step(h, c)
            new.append(c)
        return F.linear(self.n(h), self.e.weight), new


def build():
    return HyenaLM()
