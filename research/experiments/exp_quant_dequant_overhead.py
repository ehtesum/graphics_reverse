#!/usr/bin/env python3
"""
research/experiments/exp_quant_dequant_overhead.py
Forensic Characterization: Corner A1 - Quantized INT4 Nibble Unpacking & Dequantization Overhead
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5 Turing, Non-Tensor Core FP32/INT32 ALUs)

Methodology:
1. Benchmark identical matrix-vector dot products (GEMV) of size N=16,777,216 (16M parameters) across:
   a) Unquantized FP32: Direct 32-bit float loads (64 MB payload)
   b) Unquantized FP16: Direct 16-bit half2 loads (32 MB payload)
   c) Quantized Q4_0: 4-bit nibble packed in 32-bit words (8 MB payload) + FP32 scale per 32 weights
2. Measure kernel execution time (ms), effective memory throughput, and ALU dequantization penalty.
3. Quantify the exact trade-off between DRAM memory bandwidth savings (4x-8x) vs ALU bit-manipulation cost
   on Turing SM 7.5 silicon without hardware tensor cores.
"""

import ctypes
import json
import subprocess
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "quant_dequant_results.json"


def compile_gemv_kernels():
    ptx = """
.version 7.5
.target sm_75
.address_size 64

// 1. Unquantized FP32 GEMV Dot Product
.visible .entry GemvFP32(
    .param .u64 d_w,
    .param .u64 d_x,
    .param .u64 d_out,
    .param .u32 N
) {
    .reg .u64 %w, %x, %out, %w_addr, %x_addr;
    .reg .u32 %n, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride, %r_offset;
    .reg .f32 %w_val, %x_val, %acc;
    .reg .pred %p0;

    ld.param.u64 %w, [d_w];
    ld.param.u64 %x, [d_x];
    ld.param.u64 %out, [d_out];
    ld.param.u32 %n, [N];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;
    mov.f32 %acc, 0.0;

    $loop_fp32:
    setp.ge.u32 %p0, %gidx, %n;
    @%p0 bra $exit_fp32;

    shl.b32 %r_offset, %gidx, 2;
    cvt.u64.u32 %w_addr, %r_offset;
    add.u64 %w_addr, %w, %w_addr;
    cvt.u64.u32 %x_addr, %r_offset;
    add.u64 %x_addr, %x, %x_addr;

    ld.global.f32 %w_val, [%w_addr];
    ld.global.f32 %x_val, [%x_addr];
    fma.rn.f32 %acc, %w_val, %x_val, %acc;

    add.u32 %gidx, %gidx, %stride;
    bra $loop_fp32;

    $exit_fp32:
    shl.b32 %r_offset, %gidx, 2;
    // Warp shuffle reduction
    .reg .f32 %shfl_val;
    .reg .pred %p_dummy;
    shfl.sync.down.b32 %shfl_val|%p_dummy, %acc, 16, 31, 0xffffffff;
    add.f32 %acc, %acc, %shfl_val;
    shfl.sync.down.b32 %shfl_val|%p_dummy, %acc, 8, 31, 0xffffffff;
    add.f32 %acc, %acc, %shfl_val;
    shfl.sync.down.b32 %shfl_val|%p_dummy, %acc, 4, 31, 0xffffffff;
    add.f32 %acc, %acc, %shfl_val;
    shfl.sync.down.b32 %shfl_val|%p_dummy, %acc, 2, 31, 0xffffffff;
    add.f32 %acc, %acc, %shfl_val;
    shfl.sync.down.b32 %shfl_val|%p_dummy, %acc, 1, 31, 0xffffffff;
    add.f32 %acc, %acc, %shfl_val;

    .reg .pred %p_first;
    setp.eq.u32 %p_first, %tidx, 0;
    @%p_first st.global.f32 [%out], %acc;
    ret;
}

// 2. Quantized Q4_0 GEMV Dot Product (8 INT4 weights per 32-bit word, with scale)
.visible .entry GemvQ4_0(
    .param .u64 d_w_packed,
    .param .u64 d_scale,
    .param .u64 d_x,
    .param .u64 d_out,
    .param .u32 N_blocks // each block = 32 weights = 4 packed uint32 + 1 float scale
) {
    .reg .u64 %w_ptr, %scale_ptr, %x_ptr, %out_ptr;
    .reg .u32 %n_blocks, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride;
    .reg .f32 %acc, %scale, %x_val, %w_f32;
    .reg .u32 %packed, %nibble, %k, %block_idx;
    .reg .pred %p0;

    ld.param.u64 %w_ptr, [d_w_packed];
    ld.param.u64 %scale_ptr, [d_scale];
    ld.param.u64 %x_ptr, [d_x];
    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_blocks, [N_blocks];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;
    mov.f32 %acc, 0.0;

    $loop_q4:
    setp.ge.u32 %p0, %gidx, %n_blocks;
    @%p0 bra $exit_q4;

    // Each block has 1 uint32 packed word (8 weights) and 1 float scale
    .reg .u64 %sc_addr, %w_base, %x_base, %offset64;
    .reg .u32 %offset32;

    shl.b32 %offset32, %gidx, 2; // gidx * 4 bytes
    cvt.u64.u32 %offset64, %offset32;
    add.u64 %sc_addr, %scale_ptr, %offset64;
    ld.global.f32 %scale, [%sc_addr];

    add.u64 %w_base, %w_ptr, %offset64;

    shl.b32 %offset32, %gidx, 5; // gidx * 8 elements * 4 bytes = 32 bytes of X
    cvt.u64.u32 %offset64, %offset32;
    add.u64 %x_base, %x_ptr, %offset64;

    // Load packed word (8 weights)
    ld.global.u32 %packed, [%w_base];
    .reg .s32 %signed_nibble;

    // Unpack 8 nibbles from %packed
    // Nibble 0
    and.b32 %nibble, %packed, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 1
    shr.u32 %nibble, %packed, 4;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 4];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 2
    shr.u32 %nibble, %packed, 8;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 8];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 3
    shr.u32 %nibble, %packed, 12;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 12];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 4
    shr.u32 %nibble, %packed, 16;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 16];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 5
    shr.u32 %nibble, %packed, 20;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 20];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 6
    shr.u32 %nibble, %packed, 24;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 24];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    // Nibble 7
    shr.u32 %nibble, %packed, 28;
    and.b32 %nibble, %nibble, 0x0F;
    sub.s32 %signed_nibble, %nibble, 8;
    cvt.rn.f32.s32 %w_f32, %signed_nibble;
    mul.f32 %w_f32, %w_f32, %scale;
    ld.global.f32 %x_val, [%x_base + 28];
    fma.rn.f32 %acc, %w_f32, %x_val, %acc;

    add.u32 %gidx, %gidx, %stride;
    bra $loop_q4;

    $exit_q4:
    .reg .f32 %q_shfl;
    .reg .pred %q_dummy;
    shfl.sync.down.b32 %q_shfl|%q_dummy, %acc, 16, 31, 0xffffffff;
    add.f32 %acc, %acc, %q_shfl;
    shfl.sync.down.b32 %q_shfl|%q_dummy, %acc, 8, 31, 0xffffffff;
    add.f32 %acc, %acc, %q_shfl;
    shfl.sync.down.b32 %q_shfl|%q_dummy, %acc, 4, 31, 0xffffffff;
    add.f32 %acc, %acc, %q_shfl;
    shfl.sync.down.b32 %q_shfl|%q_dummy, %acc, 2, 31, 0xffffffff;
    add.f32 %acc, %acc, %q_shfl;
    shfl.sync.down.b32 %q_shfl|%q_dummy, %acc, 1, 31, 0xffffffff;
    add.f32 %acc, %acc, %q_shfl;

    .reg .pred %q_first;
    setp.eq.u32 %q_first, %tidx, 0;
    @%q_first st.global.f32 [%out_ptr], %acc;
    ret;
}
"""
    p_file = RAW_DIR / "quant_gemv.ptx"
    c_file = RAW_DIR / "quant_gemv.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER A1: Quantized INT4 Nibble Unpacking & Dequantization Overhead")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117 Turing) | Non-Tensor Core ALUs")
    print("=" * 80)

    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"[ERROR] Failed to load nvcuda.dll: {e}")
        sys.exit(1)

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    c_file = compile_gemv_kernels()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))

    func_fp32 = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_fp32), mod, b"GemvFP32")
    func_q4 = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_q4), mod, b"GemvQ4_0")

    # Allocate buffers for N = 8,388,608 elements (8M params)
    N = 8388608
    # FP32 weights: N * 4 bytes = 32 MB
    d_w_fp32 = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_w_fp32), N * 4)

    # Vector X: N * 4 bytes = 32 MB
    d_x = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_x), N * 4)

    # Q4_0 packed weights: (N / 8) * 4 bytes = 4 MB
    d_w_q4 = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_w_q4), (N // 8) * 4)

    # Q4_0 block scales: (N / 8) * 4 bytes = 4 MB
    N_blocks = N // 8
    d_scale = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_scale), N_blocks * 4)

    # Output scalar
    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)

    # Events
    ev_start = ctypes.c_void_p()
    ev_stop = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(ev_start), 0)
    cuda.cuEventCreate(ctypes.byref(ev_stop), 0)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    BLOCK_SIZE = 256
    NUM_BLOCKS = 112 # 14 SMs * 8

    # --- 1. Benchmark FP32 GEMV ---
    p_w_fp32 = ctypes.c_uint64(d_w_fp32.value)
    p_x = ctypes.c_uint64(d_x.value)
    p_out = ctypes.c_uint64(d_out.value)
    p_n = ctypes.c_uint32(N)

    args_fp32 = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(p_w_fp32), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_n), ctypes.c_void_p),
    )

    # Warmup
    cuda.cuLaunchKernel(func_fp32, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_fp32, None)
    cuda.cuStreamSynchronize(stream)

    cuda.cuEventRecord(ev_start, stream)
    ITERS = 20
    for _ in range(ITERS):
        cuda.cuLaunchKernel(func_fp32, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_fp32, None)
    cuda.cuEventRecord(ev_stop, stream)
    cuda.cuEventSynchronize(ev_stop)

    ms_fp32 = ctypes.c_float()
    cuda.cuEventElapsedTime(ctypes.byref(ms_fp32), ev_start, ev_stop)
    t_fp32_ms = ms_fp32.value / ITERS

    fp32_bytes = (N * 4) + (N * 4) # W + X
    fp32_bw = (fp32_bytes / 1e9) / (t_fp32_ms / 1000.0)
    fp32_flops = (2.0 * N) / (t_fp32_ms / 1000.0) / 1e9

    # --- 2. Benchmark Q4_0 GEMV ---
    p_w_q4 = ctypes.c_uint64(d_w_q4.value)
    p_sc = ctypes.c_uint64(d_scale.value)
    p_n_blocks = ctypes.c_uint32(N_blocks)

    args_q4 = (ctypes.c_void_p * 5)(
        ctypes.cast(ctypes.byref(p_w_q4), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_sc), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_n_blocks), ctypes.c_void_p),
    )

    cuda.cuLaunchKernel(func_q4, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_q4, None)
    cuda.cuStreamSynchronize(stream)

    cuda.cuEventRecord(ev_start, stream)
    for _ in range(ITERS):
        cuda.cuLaunchKernel(func_q4, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_q4, None)
    cuda.cuEventRecord(ev_stop, stream)
    cuda.cuEventSynchronize(ev_stop)

    ms_q4 = ctypes.c_float()
    cuda.cuEventElapsedTime(ctypes.byref(ms_q4), ev_start, ev_stop)
    t_q4_ms = ms_q4.value / ITERS

    q4_bytes = (N // 2) + (N_blocks * 4) + (N * 4) # W_packed + scale + X
    q4_bw = (q4_bytes / 1e9) / (t_q4_ms / 1000.0)
    q4_flops = (2.0 * N) / (t_q4_ms / 1000.0) / 1e9

    print(f"{'Precision':<16} | {'Execution Time':<18} | {'DRAM Footprint':<16} | {'Effective BW':<16} | {'Effective GFLOPS'}")
    print("-" * 85)
    print(f"{'FP32 GEMV':<16} | {t_fp32_ms:<15.3f} ms | {fp32_bytes / 1e6:<13.1f} MB | {fp32_bw:<13.2f} GB/s | {fp32_flops:.1f} GFLOPS")
    print(f"{'Q4_0 GEMV':<16} | {t_q4_ms:<15.3f} ms | {q4_bytes / 1e6:<13.1f} MB | {q4_bw:<13.2f} GB/s | {q4_flops:.1f} GFLOPS")

    speedup = t_fp32_ms / t_q4_ms
    print("-" * 85)
    print(f"[*] Q4_0 vs FP32 Speedup Ratio: {speedup:.2f}x ({(speedup - 1.0) * 100.0:+.1f}%)")

    results = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "elements_N": N,
        "fp32_results": {
            "time_ms": round(t_fp32_ms, 3),
            "footprint_mb": round(fp32_bytes / 1e6, 2),
            "effective_bw_gbps": round(fp32_bw, 2),
            "effective_gflops": round(fp32_flops, 1)
        },
        "q4_0_results": {
            "time_ms": round(t_q4_ms, 3),
            "footprint_mb": round(q4_bytes / 1e6, 2),
            "effective_bw_gbps": round(q4_bw, 2),
            "effective_gflops": round(q4_flops, 1)
        },
        "q4_vs_fp32_speedup": round(speedup, 3)
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFree_v2(d_w_fp32)
    cuda.cuMemFree_v2(d_x)
    cuda.cuMemFree_v2(d_w_q4)
    cuda.cuMemFree_v2(d_scale)
    cuda.cuMemFree_v2(d_out)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
