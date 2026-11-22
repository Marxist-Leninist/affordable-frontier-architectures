"""Fused sequential selective scan for P039-L (fp32 CUDA forward + backward, built with torch load_inline).

h_t = exp(delta_t * A) * h_{t-1} + delta_t * u_t * B_t ;   y_t = sum_s h_t[s] * C_t[s]
One thread per (batch, channel) keeps the S=16 states in registers and walks the sequence; the backward walks it
in reverse from the saved states. dB/dC are reduced over channels and dA over the batch outside the kernel.
This is the usual way Mamba-1 runs (a fused scan), so the hybrid's training and inference cost is not inflated by
a pure-PyTorch scan. Tested against a sequential fp64 reference in p039l.py smoke.
"""
import os
from pathlib import Path
import torch

CPP_SRC = r'''
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> fwd(torch::Tensor u, torch::Tensor dt, torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, bool save);
std::vector<torch::Tensor> bwd(torch::Tensor dy, torch::Tensor u, torch::Tensor dt, torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor hs);
'''

CUDA_SRC = r'''
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template <int S>
__global__ void scan_fwd_kernel(const float* __restrict__ u, const float* __restrict__ dt, const float* __restrict__ A,
                                const float* __restrict__ Bm, const float* __restrict__ Cm,
                                float* __restrict__ y, float* __restrict__ hs, float* __restrict__ hlast,
                                int Bn, int T, int E, bool save) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Bn * E) return;
  int b = idx / E, e = idx % E;
  float h[S], a[S];
#pragma unroll
  for (int s = 0; s < S; ++s) { h[s] = 0.f; a[s] = A[e * S + s]; }
  for (int t = 0; t < T; ++t) {
    long o = ((long)b * T + t) * E + e;
    float d = dt[o], du = d * u[o];
    const float* Bt = Bm + ((long)b * T + t) * S;
    const float* Ct = Cm + ((long)b * T + t) * S;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
      h[s] = expf(d * a[s]) * h[s] + du * Bt[s];
      acc += h[s] * Ct[s];
    }
    y[o] = acc;
    if (save) {
#pragma unroll
      for (int s = 0; s < S; ++s) hs[o * S + s] = h[s];
    }
  }
#pragma unroll
  for (int s = 0; s < S; ++s) hlast[((long)b * E + e) * S + s] = h[s];
}

template <int S>
__global__ void scan_bwd_kernel(const float* __restrict__ dy, const float* __restrict__ u, const float* __restrict__ dt,
                                const float* __restrict__ A, const float* __restrict__ Bm, const float* __restrict__ Cm,
                                const float* __restrict__ hs,
                                float* __restrict__ du, float* __restrict__ ddt, float* __restrict__ dA_part,
                                float* __restrict__ dB_full, float* __restrict__ dC_full, int Bn, int T, int E) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Bn * E) return;
  int b = idx / E, e = idx % E;
  float g[S], a[S], dAacc[S];
#pragma unroll
  for (int s = 0; s < S; ++s) { g[s] = 0.f; a[s] = A[e * S + s]; dAacc[s] = 0.f; }
  for (int t = T - 1; t >= 0; --t) {
    long o = ((long)b * T + t) * E + e;
    float d = dt[o], uu = u[o], gy = dy[o];
    const float* Bt = Bm + ((long)b * T + t) * S;
    const float* Ct = Cm + ((long)b * T + t) * S;
    float duacc = 0.f, ddtacc = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
      float hcur = hs[o * S + s];
      float hprev = t > 0 ? hs[(o - E) * S + s] : 0.f;
      float aa = expf(d * a[s]);
      float dh = g[s] + gy * Ct[s];          // total gradient reaching h_t
      dC_full[o * S + s] = gy * hcur;
      dB_full[o * S + s] = dh * d * uu;
      duacc += dh * d * Bt[s];
      ddtacc += dh * (a[s] * aa * hprev + uu * Bt[s]);
      dAacc[s] += dh * d * aa * hprev;
      g[s] = dh * aa;                         // carried to h_{t-1}
    }
    du[o] = duacc; ddt[o] = ddtacc;
  }
#pragma unroll
  for (int s = 0; s < S; ++s) dA_part[((long)b * E + e) * S + s] = dAacc[s];
}

static void check(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> fwd(torch::Tensor u, torch::Tensor dt, torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, bool save) {
  check(u, "u"); check(dt, "dt"); check(A, "A"); check(Bm, "Bm"); check(Cm, "Cm");
  int Bn = u.size(0), T = u.size(1), E = u.size(2), S = A.size(1);
  TORCH_CHECK(S == 16 && A.size(0) == E && Bm.size(2) == S && Cm.size(2) == S, "shape mismatch (S must be 16)");
  auto y = torch::empty_like(u);
  auto hs = save ? torch::empty({Bn, T, E, S}, u.options()) : torch::empty({1}, u.options());
  auto hl = torch::empty({Bn, E, S}, u.options());
  int threads = 128, blocks = (Bn * E + threads - 1) / threads;
  scan_fwd_kernel<16><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      u.data_ptr<float>(), dt.data_ptr<float>(), A.data_ptr<float>(), Bm.data_ptr<float>(), Cm.data_ptr<float>(),
      y.data_ptr<float>(), hs.data_ptr<float>(), hl.data_ptr<float>(), Bn, T, E, save);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, hs, hl};
}

std::vector<torch::Tensor> bwd(torch::Tensor dy, torch::Tensor u, torch::Tensor dt, torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor hs) {
  check(dy, "dy"); check(u, "u"); check(dt, "dt"); check(A, "A"); check(Bm, "Bm"); check(Cm, "Cm"); check(hs, "hs");
  int Bn = u.size(0), T = u.size(1), E = u.size(2), S = A.size(1);
  TORCH_CHECK(S == 16 && hs.dim() == 4, "shape mismatch");
  auto du = torch::empty_like(u), ddt = torch::empty_like(u);
  auto dAp = torch::empty({Bn, E, S}, u.options());
  auto dBf = torch::empty({Bn, T, E, S}, u.options()), dCf = torch::empty({Bn, T, E, S}, u.options());
  int threads = 128, blocks = (Bn * E + threads - 1) / threads;
  scan_bwd_kernel<16><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      dy.data_ptr<float>(), u.data_ptr<float>(), dt.data_ptr<float>(), A.data_ptr<float>(), Bm.data_ptr<float>(),
      Cm.data_ptr<float>(), hs.data_ptr<float>(), du.data_ptr<float>(), ddt.data_ptr<float>(), dAp.data_ptr<float>(),
      dBf.data_ptr<float>(), dCf.data_ptr<float>(), Bn, T, E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {du, ddt, dAp.sum(0), dBf.sum(2), dCf.sum(2)};
}
'''

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
        bd = Path(__file__).resolve().parent / 'build_scan'; bd.mkdir(exist_ok=True)
        _EXT = load_inline('p039l_scan_v1', CPP_SRC, CUDA_SRC, functions=['fwd', 'bwd'],
                           extra_cuda_cflags=['-O3'], build_directory=str(bd), verbose=False)
    return _EXT


class FusedScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, Bm, Cm):
        u, delta, A, Bm, Cm = (x.contiguous() for x in (u, delta, A, Bm, Cm))
        need = any(ctx.needs_input_grad)
        y, hs, _ = ext().fwd(u, delta, A, Bm, Cm, need)
        if need:
            ctx.save_for_backward(u, delta, A, Bm, Cm, hs)
        return y

    @staticmethod
    def backward(ctx, dy):
        u, delta, A, Bm, Cm, hs = ctx.saved_tensors
        du, ddt, dA, dB, dC = ext().bwd(dy.contiguous(), u, delta, A, Bm, Cm, hs)
        return du, ddt, dA, dB, dC


def fused_scan(u, delta, A, Bm, Cm):
    return FusedScan.apply(u, delta, A, Bm, Cm)


def fused_scan_last(u, delta, A, Bm, Cm):
    """Inference only: y and the final state, without storing per-step states."""
    y, _, hl = ext().fwd(u.contiguous(), delta.contiguous(), A.contiguous(), Bm.contiguous(), Cm.contiguous(), False)
    return y, hl
