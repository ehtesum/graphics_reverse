#!/usr/bin/env python3
"""
research/experiments/exp_warp_divergence_depth.py
Corner 4 Forensic Experiment: GPU Warp Divergence Stack Depth & Penalty
Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)

Microarchitectural Hypotheses:
  1. Turing SM 7.5 features Independent Thread Scheduling (ITS) with per-thread Program Counters (PC).
  2. When branches diverge across a warp, divergent paths are executed serially.
  3. Diverging into N distinct paths across a 32-thread warp scales execution latency linearly
     by exactly N (from 1x for 1 path up to 32x for 32 distinct paths).
  4. SASS disassembly will reveal divergent branch targets and reconvergence points.
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
CUOBJDUMP = CUDA_BIN / "cuobjdump.exe"
OUT_FILE = RAW_DIR / "warp_divergence_depth_results.json"


def generate_divergence_ptx(num_paths):
    """Generates a PTX kernel that splits a warp into exactly num_paths divergent paths."""
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry DivergenceKernel_P{num_paths}(",
        "    .param .u64 param_out,",
        "    .param .u32 param_iters",
        ")",
        "{",
        "    .reg .b32 %r<35>;",
        "    .reg .b64 %rd<5>;",
        "    .reg .pred %p<35>;",
        "",
        "    ld.param.u64 %rd1, [param_out];",
        "    ld.param.u32 %r1, [param_iters];",
        "    mov.u32 %r2, %tid.x;",
        "",
        f"    // Determine path ID (0 to {num_paths - 1})",
        f"    rem.u32 %r3, %r2, {num_paths};",
        "    mov.u32 %r10, 1;",
        "    mov.u32 %r20, 0;",
        "",
        "LOOP:",
    ]

    for p in range(num_paths):
        ptx.append(f"    setp.eq.u32 %p{p}, %r3, {p};")
        ptx.append(f"    @%p{p} bra PATH_{p};")

    ptx.append("    bra END_PATHS;")

    for p in range(num_paths):
        ptx.extend([
            f"PATH_{p}:",
            f"    // Workload for Path {p} (16 unrolled ALU math ops)",
            f"    add.u32 %r10, %r10, {p + 1};",
            f"    mul.lo.u32 %r10, %r10, 3;",
            f"    add.u32 %r10, %r10, {p + 2};",
            f"    mul.lo.u32 %r10, %r10, 5;",
            f"    add.u32 %r10, %r10, {p + 3};",
            f"    mul.lo.u32 %r10, %r10, 7;",
            f"    add.u32 %r10, %r10, {p + 4};",
            f"    mul.lo.u32 %r10, %r10, 11;",
            f"    bra END_PATHS;"
        ])

    ptx.extend([
        "END_PATHS:",
        "    add.u32 %r20, %r20, 1;",
        "    setp.lt.u32 %p34, %r20, %r1;",
        "    @%p34 bra LOOP;",
        "",
        "    mul.wide.u32 %rd2, %r2, 4;",
        "    add.u64 %rd3, %rd1, %rd2;",
        "    st.global.u32 [%rd3], %r10;",
        "    ret;",
        "}"
    ])
    return "\n".join(ptx)


def run_experiment():
    print("=" * 96)
    print(" Corner 4: GPU Warp Divergence Stack Depth & Penalty Forensic Audit")
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

    PATHS = [1, 2, 4, 8, 16, 32]
    BLOCKS = 64
    THREADS = 1024 # 65,536 threads total
    LOOP_ITERS = 10000

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), BLOCKS * THREADS * 4)

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    p_out = ctypes.c_uint64(d_out.value)
    iters_arg = ctypes.c_uint32(LOOP_ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters_arg), ctypes.c_void_p)
    )

    print(f"{'Divergent Paths':<20} | {'Active Threads/Path':<24} | {'Kernel Time (ms)':<18} | {'Relative Slowdown'} | {'SIMT Efficiency'}")
    print("-" * 104)

    baseline_ms = 0.0
    results = []

    for num_p in PATHS:
        ptx_src = generate_divergence_ptx(num_p)
        p_file = RAW_DIR / f"diverge_p{num_p}.ptx"
        c_file = RAW_DIR / f"diverge_p{num_p}.cubin"
        with open(p_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, f"DivergenceKernel_P{num_p}".encode("utf-8"))

        # Warmup
        cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()

        times = []
        for _ in range(10):
            cuda.cuEventRecord(start_evt, None)
            cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
            cuda.cuEventRecord(stop_evt, None)
            cuda.cuEventSynchronize(stop_evt)
            el = ctypes.c_float()
            cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
            times.append(el.value)

        avg_ms = sum(times) / len(times)

        if num_p == 1:
            baseline_ms = avg_ms
            rel_str = "1.00x (Baseline)"
            eff_str = "100.0%"
        else:
            slowdown = avg_ms / baseline_ms
            rel_str = f"{slowdown:.2f}x"
            eff_str = f"{(1.0 / slowdown) * 100.0:.1f}%"

        threads_per_path = 32 // num_p
        label_str = f"{num_p} Distinct Paths"
        print(f"{label_str:<20} | {threads_per_path:<24} | {avg_ms:<18.3f} | {rel_str:<18} | {eff_str}")

        results.append({
            "num_paths": num_p,
            "threads_per_path": threads_per_path,
            "kernel_time_ms": round(avg_ms, 3),
            "slowdown": round(avg_ms / baseline_ms, 3),
            "simt_efficiency_pct": round((1.0 / (avg_ms / baseline_ms)) * 100.0, 1)
        })

        cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_out)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 104)
    print(f"[SUCCESS] Warp divergence telemetry saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
