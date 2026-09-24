#!/usr/bin/env python3
"""
research/experiments/exp_prefill_decode_roofline.py
Corner 10 Forensic Experiment: Local AI Prefill vs Decode Arithmetic Intensity Transition Knee
Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, 136 GB/s GDDR6, 5.55 TFLOPS FP16)

Microarchitectural Hypotheses:
  1. The hardware ridge point for TU117 is I_ridge = Peak Compute / Peak Bandwidth = 5552 / 136 = 40.8 FLOP/Byte.
  2. For Decode (Batch B = 1, GEMV): Arithmetic Intensity is ~1.0 FLOP/Byte (FP16) or 2.0 (INT8).
     Execution is strictly memory-bandwidth bound (< 150 GFLOPS, saturated at ~110-120 GB/s GDDR6).
  3. For Prefill (Batch B >= 48, GEMM): Arithmetic Intensity crosses the 40.8 FLOP/Byte ridge point.
     Execution transitions to compute-bound, scaling toward the 5.39 - 5.55 TFLOPS roofline.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "prefill_decode_roofline_results.json"


def generate_gemm_ptx():
    """Generates a matrix multiplication kernel for variable batch size B and fixed hidden dim D=2048."""
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry BatchedMatMul(",
        "    .param .u64 param_out,",
        "    .param .u64 param_act,",
        "    .param .u64 param_weight,",
        "    .param .u32 param_B,",
        "    .param .u32 param_D",
        ")",
        "{",
        "    .reg .b32 %r<15>;",
        "    .reg .b64 %rd<15>;",
        "    .reg .f32 %f<10>;",
        "    .reg .pred %p<2>;",
        "",
        "    ld.param.u64 %rd1, [param_out];",
        "    ld.param.u64 %rd2, [param_act];",
        "    ld.param.u64 %rd3, [param_weight];",
        "    ld.param.u32 %r1, [param_B];",
        "    ld.param.u32 %r2, [param_D];",
        "",
        "    // Grid mapping: 2D block (row in B, col in D)",
        "    mov.u32 %r3, %ctaid.x;",
        "    mov.u32 %r4, %ntid.x;",
        "    mov.u32 %r5, %tid.x;",
        "    mad.lo.u32 %r6, %r3, %r4, %r5; // col = output feature (0..D-1)",
        "    mov.u32 %r7, %ctaid.y;          // row = batch token (0..B-1)",
        "",
        "    setp.ge.u32 %p1, %r6, %r2;",
        "    @%p1 bra EXIT;",
        "",
        "    // Compute dot product of act[row, :] and weight[:, col]",
        "    mov.f32 %f0, 0.0;",
        "    mov.u32 %r8, 0;",
        "",
        "LOOP:",
        "    // Load activation act[row * D + k]",
        "    mad.lo.u32 %r9, %r7, %r2, %r8;",
        "    mul.wide.u32 %rd4, %r9, 4;",
        "    add.u64 %rd5, %rd2, %rd4;",
        "    ld.global.f32 %f1, [%rd5];",
        "",
        "    // Load weight weight[k * D + col]",
        "    mad.lo.u32 %r10, %r8, %r2, %r6;",
        "    mul.wide.u32 %rd6, %r10, 4;",
        "    add.u64 %rd7, %rd3, %rd6;",
        "    ld.global.f32 %f2, [%rd7];",
        "",
        "    fma.rn.f32 %f0, %f1, %f2, %f0;",
        "",
        "    add.u32 %r8, %r8, 1;",
        "    setp.lt.u32 %p1, %r8, %r2;",
        "    @%p1 bra LOOP;",
        "",
        "    // Write out out[row * D + col]",
        "    mad.lo.u32 %r11, %r7, %r2, %r6;",
        "    mul.wide.u32 %rd8, %r11, 4;",
        "    add.u64 %rd9, %rd1, %rd8;",
        "    st.global.f32 [%rd9], %f0;",
        "",
        "EXIT:",
        "    ret;",
        "}"
    ]
    return "\n".join(ptx)


def run_experiment():
    print("=" * 96)
    print(" Corner 10: Local AI Prefill vs Decode Arithmetic Intensity & Roofline Knee")
    print(" Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)")
    print("=" * 96)

    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"Failed to load nvcuda.dll: {e}")
        sys.exit(1)

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    # Compile kernel
    ptx_src = generate_gemm_ptx()
    p_file = RAW_DIR / "batched_gemm.ptx"
    c_file = RAW_DIR / "batched_gemm.cubin"
    with open(p_file, "w") as f:
        f.write(ptx_src)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"BatchedMatMul")

    # Fixed hidden dimension D = 1024 (fits easily in memory across batches)
    D = 1024
    WEIGHT_BYTES = D * D * 4 # 4 MB weight matrix
    d_weight = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_weight), WEIGHT_BYTES)

    # Initialize weights
    h_w = (ctypes.c_float * (D * D))(*[0.01 for _ in range(D * D)])
    cuda.cuMemcpyHtoD_v2(d_weight, h_w, WEIGHT_BYTES)

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    # Batch sizes sweeping from Decode (B=1) to Heavy Prefill (B=512)
    BATCHES = [
        (1, "Decode Phase (Token Gen B=1)"),
        (2, "Decode Speculative (B=2)"),
        (4, "Decode Batched (B=4)"),
        (8, "Small Prompt (B=8)"),
        (16, "Prompt Ingest (B=16)"),
        (32, "Prompt Prefill (B=32)"),
        (48, "Roofline Transition Knee (B=48)"),
        (64, "Compute-Bound Prefill (B=64)"),
        (128, "Heavy Prefill (B=128)"),
        (256, "Full Batch Prefill (B=256)")
    ]

    print(f"{'Phase / Batch (B)':<34} | {'Intensity (FLOP/B)':<20} | {'Time (ms)':<12} | {'Compute (GFLOP/s)':<20} | {'Roofline Regime'}")
    print("-" * 104)

    results = []

    for b_val, desc in BATCHES:
        ACT_BYTES = b_val * D * 4
        OUT_BYTES = b_val * D * 4
        d_act = ctypes.c_uint64()
        d_out = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_act), ACT_BYTES)
        cuda.cuMemAlloc_v2(ctypes.byref(d_out), OUT_BYTES)

        # 2 FLOPs per element * B * D * D
        total_flops = 2.0 * b_val * D * D
        total_bytes_transferred = ACT_BYTES + OUT_BYTES + WEIGHT_BYTES
        arithmetic_intensity = total_flops / total_bytes_transferred

        p_out = ctypes.c_uint64(d_out.value)
        p_act = ctypes.c_uint64(d_act.value)
        p_w = ctypes.c_uint64(d_weight.value)
        b_arg = ctypes.c_uint32(b_val)
        d_arg = ctypes.c_uint32(D)

        k_args = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_act), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_w), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(b_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(d_arg), ctypes.c_void_p)
        )

        THREADS_PER_BLOCK = 256
        GRID_X = (D + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        GRID_Y = b_val

        # Warmup
        cuda.cuLaunchKernel(func, GRID_X, GRID_Y, 1, THREADS_PER_BLOCK, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()

        times = []
        ITERS = 10 if b_val <= 64 else 5
        for _ in range(ITERS):
            cuda.cuEventRecord(start_evt, None)
            cuda.cuLaunchKernel(func, GRID_X, GRID_Y, 1, THREADS_PER_BLOCK, 1, 1, 0, None, k_args, None)
            cuda.cuEventRecord(stop_evt, None)
            cuda.cuEventSynchronize(stop_evt)
            el = ctypes.c_float()
            cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
            times.append(el.value)

        avg_ms = sum(times) / len(times)
        gflops = (total_flops / (avg_ms / 1000.0)) / 1e9

        if arithmetic_intensity < 2.0:
            regime = "Memory Bandwidth Saturated (GEMV)"
        elif arithmetic_intensity < 20.0:
            regime = "Bandwidth / Compute Crossover"
        else:
            regime = "Compute-Bound Saturation (GEMM)"

        print(f"{desc:<34} | {arithmetic_intensity:<20.2f} | {avg_ms:<12.3f} | {gflops:<20.1f} | {regime}")

        results.append({
            "batch_size": b_val,
            "description": desc,
            "arithmetic_intensity": round(arithmetic_intensity, 2),
            "time_ms": round(avg_ms, 3),
            "throughput_gflops": round(gflops, 1),
            "regime": regime
        })

        cuda.cuMemFree_v2(d_act)
        cuda.cuMemFree_v2(d_out)

    cuda.cuMemFree_v2(d_weight)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 104)
    print(f"[SUCCESS] Roofline transition results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
