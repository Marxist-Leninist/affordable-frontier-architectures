"""Single-thread CPU FP32/FP64 selective affine scan with first-order autograd.

Inputs [batch,time,inner,state]; noncontiguous inputs are normalized explicitly.
Finite input contract. No mixed precision, GPU, or higher-order derivatives.
A local g++ compiler is required on first use; compiled binary is source-hash keyed.
O(B*T*I*N) compute, saves a and h for backward. No Torch C++ ABI dependency.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import time
import torch
from torch.autograd.function import once_differentiable

_LIB = None
COMPILE_SECONDS = 0.0
FLAGS = ['-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off']

def load_library():
    global _LIB, COMPILE_SECONDS
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name('scan_cpu.cpp')
    digest = hashlib.sha256(source.read_bytes() + repr(FLAGS).encode()).hexdigest()[:16]
    target = source.with_name('scan_cpu_' + digest + '.so')
    if not target.exists():
        temp = target.with_suffix(f'.{os.getpid()}.tmp.so')
        start = time.perf_counter()
        try:
            subprocess.run(['g++', *FLAGS, str(source), '-o', str(temp)], check=True, capture_output=True)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        COMPILE_SECONDS = time.perf_counter() - start
    lib = ctypes.CDLL(str(target))
    for suffix in ('f32', 'f64'):
        for name, pointers in [('forward', 3), ('backward', 5)]:
            fn = getattr(lib, f'affine_{name}_{suffix}')
            fn.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * 3
            fn.restype = None
    _LIB = lib
    return lib

def _args(a):
    assert a.is_contiguous() and a.device.type == 'cpu'
    return (a.shape[0], a.shape[1], a.shape[2] * a.shape[3])

class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.contiguous(), b.contiguous()
        lib = load_library()
        h = torch.empty_like(b)
        suffix = 'f32' if a.dtype == torch.float32 else 'f64'
        getattr(lib, f'affine_forward_{suffix}')(a.data_ptr(), b.data_ptr(), h.data_ptr(), *_args(a))
        ctx.save_for_backward(a, h)
        ctx.suffix = suffix
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        a, h = ctx.saved_tensors
        grad = grad.contiguous()
        ga, gb = torch.empty_like(a), torch.empty_like(a)
        getattr(load_library(), f'affine_backward_{ctx.suffix}')(
            a.data_ptr(), h.data_ptr(), grad.data_ptr(), ga.data_ptr(), gb.data_ptr(), *_args(a))
        return ga, gb

def scan_fast(a, b):
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError('a and b must have equal [B,T,I,N] shapes')
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('scan_fast supports CPU only')
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64):
        raise TypeError('scan_fast requires matching float32 or float64 inputs')
    if any(n == 0 for n in a.shape):
        raise ValueError('scan_fast requires nonempty dimensions')
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError('scan_fast requires strided tensors')
    return _Scan.apply(a, b)
