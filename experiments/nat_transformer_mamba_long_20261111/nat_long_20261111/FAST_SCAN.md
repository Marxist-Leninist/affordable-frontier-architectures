# CPU affine scan acceleration

`fast_scan.scan_fast(a, b)` is a drop-in scan for `MambaDirection.scan`.
It evaluates `h[t] = a[t] * h[t-1] + b[t]`, with zero initial state.
Inputs must have equal nonempty `[batch, time, inner, state]` shapes, be CPU
strided tensors, and share FP32 or FP64 dtype. Noncontiguous input and upstream
gradient tensors are explicitly copied to contiguous storage. Finite inputs are
expected. GPU, FP16, BF16, mixed dtype, and higher-order differentiation are not
supported. This is a CPU training optimization, not a fused CUDA Mamba kernel.

The compiled C++ loops are O(B*T*I*N). Forward saves a and h. Backward computes
`q[t] = grad_h[t] + a[t+1]*q[t+1]`, `grad_b[t] = q[t]`, and
`grad_a[t] = q[t]*h[t-1]` (zero at t=0). No Torch C++ headers or ABI dependency,
external packages, OpenMP, fast-math, or fused multiply-add contraction are used.
The library has no global mutable numerical state. PyTorch owns all buffers and
keeps them alive through each synchronous ctypes call.

First use compiles `scan_cpu.cpp` with local g++ and caches the shared library
beside the source under a content/flags hash. Different processes use unique
temporary output filenames and an atomic replacement. The measured cold compile
and first-use times are recorded in `fast_scan_results.json`. Cached first use
still loads the library. On a different machine, delete the generated `.so` to
recompile locally. Source and tests are the reproducible deliverables.

Usage:

    from fast_scan import scan_fast
    direction.scan = scan_fast

Run validation and benchmark from the workspace root:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 frontier_cpu_runtime_20261110/venv/bin/python nat_long_20261111/test_fast_scan.py

Tests cover FP32/FP64 random shapes, noncontiguous tensors, single-time-step
sequences, first-order gradients, numerical float64 gradcheck, rejected shapes
and dtypes, real MambaDirection input/parameter gradients, and an AdamW update.
Benchmarks use newly initialized equivalent MambaDirection weights per scan,
13 complete training steps (3 warm-up, 10 measured), one PyTorch CPU thread, and
include forward, backward and AdamW. Timings exclude compilation. Sequential
reference is intentionally also measured; PyTorch's generic slicing backward
makes it much slower at larger shapes. Speed is specific to this runtime and
can vary with shared host load. No previous benchmark files are modified.
