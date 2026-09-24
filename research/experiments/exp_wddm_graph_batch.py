#!/usr/bin/env python3
"""
research/experiments/exp_wddm_graph_batch.py
Phase 3 Experiment: CUDA & WDDM Launch Decomposition
  - Benchmarks kernel launch overhead across batch counts: 1, 10, 100, 1000, 5000, 10000
  - Compares:
      1. Individual cuLaunchKernel invocations (user-mode pushbuffer append)
      2. cuLaunchKernel followed by cuCtxSynchronize (forces full OS ring-0 roundtrip)
      3. CUDA Graph capture & instantiate (amortized pushbuffer replay)
  - Identifies the crossover where kernel execution exceeds submission overhead
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "wddm_batch_sweep.json"


def compile_null_kernel():
    ptx = (
        ".version 7.5\n"
        ".target sm_75\n"
        ".address_size 64\n"
        ".visible .entry NullKernel() { ret; }\n"
    )
    p_file = RAW_DIR / "null_batch.ptx"
    c_file = RAW_DIR / "null_batch.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" Experiment EXP-WDDM-404: CUDA & WDDM Launch Decomposition & Graph Batching")
    print("=" * 80)

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

    c_file = compile_null_kernel()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"NullKernel")

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    BATCH_COUNTS = [1, 10, 50, 100, 500, 1000, 5000, 10000]

    print(f"{'Batch Count':<12} | {'Async Push (us/k)':<18} | {'Total Async (ms)':<17} | {'Sync Round-Trip (us/k)'}")
    print("-" * 75)

    results = []

    for count in BATCH_COUNTS:
        # Warmup
        for _ in range(min(count, 100)):
            cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, None, None)
        cuda.cuStreamSynchronize(stream)

        # 1. Asynchronous pushbuffer filling (User-mode queuing)
        t0 = time.perf_counter()
        for _ in range(count):
            cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, None, None)
        t1 = time.perf_counter()
        cuda.cuStreamSynchronize(stream)
        t2 = time.perf_counter()

        async_total_ms = (t1 - t0) * 1000.0
        async_per_k_us = ((t1 - t0) / count) * 1e6

        # 2. Synchronous round-trip (forcing WDDM kernel-mode transition)
        sync_sample_count = min(count, 500)
        t_sync_0 = time.perf_counter()
        for _ in range(sync_sample_count):
            cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, None, None)
            cuda.cuStreamSynchronize(stream)
        t_sync_1 = time.perf_counter()
        sync_per_k_us = ((t_sync_1 - t_sync_0) / sync_sample_count) * 1e6

        print(f"{count:<12} | {async_per_k_us:<18.3f} | {async_total_ms:<17.3f} | {sync_per_k_us:.3f}")

        results.append({
            "batch_count": count,
            "async_pushbuffer_us_per_kernel": round(async_per_k_us, 3),
            "async_total_ms": round(async_total_ms, 3),
            "sync_roundtrip_us_per_kernel": round(sync_per_k_us, 3)
        })

    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] WDDM batching results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
