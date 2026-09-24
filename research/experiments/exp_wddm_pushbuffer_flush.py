#!/usr/bin/env python3
"""
research/experiments/exp_wddm_pushbuffer_flush.py
Forensic Characterization: Corner H1 - WDDM Pushbuffer Flush Granularity & Auto-Submit Spike Profiling
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5), Windows 11 Build 22631, Driver 555.97, WDDM 2.7

Methodology:
1. Issue a continuous burst of N=4096 asynchronous cuLaunchKernel calls into a single stream with zero synchronization.
2. Measure per-launch latency in nanoseconds using high-resolution hardware timers (time.perf_counter_ns).
3. Detect latency spikes that indicate when user-mode driver (nvcuda.dll) flushes its pushbuffer
   to the OS kernel (dxgkrnl.sys / D3DKMTSubmitCommand).
4. Compute the exact command packet frequency, batch size, and tail latency distribution.
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "wddm_pushbuffer_flush_results.json"


def compile_null_kernel():
    ptx = (
        ".version 7.5\n"
        ".target sm_75\n"
        ".address_size 64\n"
        ".visible .entry NullKernel() { ret; }\n"
    )
    p_file = RAW_DIR / "null_kernel_h1.ptx"
    c_file = RAW_DIR / "null_kernel_h1.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER H1: WDDM Pushbuffer Flush Granularity & Auto-Submit Profiling")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117) | Windows 11 WDDM 2.7 | Driver 555.97")
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

    c_file = compile_null_kernel()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"NullKernel")

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    # Warmup
    print("[*] Warming up pushbuffer and driver state...")
    for _ in range(500):
        cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, None, None)
    cuda.cuStreamSynchronize(stream)

    BURST_SIZE = 4096
    print(f"[*] Recording microsecond timestamp for burst of {BURST_SIZE} asynchronous launches...")

    latencies_us = np.zeros(BURST_SIZE, dtype=np.float64)

    # Pre-allocate references to avoid Python object allocation in loop
    cuLaunch = cuda.cuLaunchKernel
    null_args = None

    t_burst_start = time.perf_counter_ns()
    t_prev = t_burst_start

    for i in range(BURST_SIZE):
        cuLaunch(func, 1, 1, 1, 1, 1, 1, 0, stream, null_args, null_args)
        t_now = time.perf_counter_ns()
        latencies_us[i] = (t_now - t_prev) / 1000.0
        t_prev = t_now

    t_burst_end = time.perf_counter_ns()
    
    # Measure drain / synchronization time
    t_sync_start = time.perf_counter_ns()
    cuda.cuStreamSynchronize(stream)
    t_sync_end = time.perf_counter_ns()

    drain_ms = (t_sync_end - t_sync_start) / 1e6
    burst_total_ms = (t_burst_end - t_burst_start) / 1e6
    mean_us = float(np.mean(latencies_us))
    median_us = float(np.median(latencies_us))
    p95_us = float(np.percentile(latencies_us, 95))
    p99_us = float(np.percentile(latencies_us, 99))
    max_us = float(np.max(latencies_us))
    min_us = float(np.min(latencies_us))

    # Detect flush spikes: points where latency exceeds 3x median
    threshold = max(median_us * 3.0, 10.0)
    spikes = np.where(latencies_us > threshold)[0]
    spike_intervals = np.diff(spikes) if len(spikes) > 1 else []

    print(f"\n[RESULTS] Telemetry Summary for N = {BURST_SIZE} Launches:")
    print(f"  - Total Burst Enqueue Time : {burst_total_ms:.3f} ms")
    print(f"  - cuStreamSynchronize Drain: {drain_ms:.3f} ms")
    print(f"  - Min Launch Latency       : {min_us:.3f} us")
    print(f"  - Median Launch Latency    : {median_us:.3f} us")
    print(f"  - Mean Launch Latency      : {mean_us:.3f} us")
    print(f"  - 95th Percentile Latency  : {p95_us:.3f} us")
    print(f"  - 99th Percentile Latency  : {p99_us:.3f} us")
    print(f"  - Max Launch Latency (Spike): {max_us:.3f} us")
    print(f"  - Detected Driver Submits  : {len(spikes)} flush spikes (threshold = {threshold:.2f} us)")
    
    if len(spike_intervals) > 0:
        mean_interval = float(np.mean(spike_intervals))
        median_interval = float(np.median(spike_intervals))
        print(f"  - Driver Pushbuffer Flush Interval: Mean = {mean_interval:.1f} launches, Median = {median_interval:.0f} launches")
    else:
        print("  - Driver buffered entire burst without autonomous mid-burst flush.")

    # Multi-burst sweep to test scaling behavior across burst sizes
    print("\n[*] Running Burst Size Scaling Sweep [1 to 8192 launches]...")
    burst_sweep_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    sweep_results = []

    print(f"{'Burst Size':<12} | {'Enqueue Time (ms)':<18} | {'Mean us/launch':<16} | {'Sync Drain (ms)':<16} | {'Total Wall (ms)'}")
    print("-" * 85)

    for b_size in burst_sweep_sizes:
        # Pre-sync
        cuda.cuStreamSynchronize(stream)
        t0 = time.perf_counter_ns()
        for _ in range(b_size):
            cuLaunch(func, 1, 1, 1, 1, 1, 1, 0, stream, null_args, null_args)
        t1 = time.perf_counter_ns()
        cuda.cuStreamSynchronize(stream)
        t2 = time.perf_counter_ns()

        enq_ms = (t1 - t0) / 1e6
        drain_ms_i = (t2 - t1) / 1e6
        total_ms_i = (t2 - t0) / 1e6
        mean_launch_us = (enq_ms * 1000.0) / b_size

        print(f"{b_size:<12} | {enq_ms:<18.3f} | {mean_launch_us:<16.3f} | {drain_ms_i:<16.3f} | {total_ms_i:.3f}")

        sweep_results.append({
            "burst_size": b_size,
            "enqueue_ms": round(enq_ms, 4),
            "mean_launch_us": round(mean_launch_us, 4),
            "sync_drain_ms": round(drain_ms_i, 4),
            "total_wall_ms": round(total_ms_i, 4)
        })

    # Save artifact
    output_data = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "driver": "555.97",
        "wddm_version": "2.7",
        "burst_4096_profile": {
            "burst_size": BURST_SIZE,
            "total_burst_ms": round(burst_total_ms, 3),
            "stream_sync_drain_ms": round(drain_ms, 3),
            "min_us": round(min_us, 3),
            "median_us": round(median_us, 3),
            "mean_us": round(mean_us, 3),
            "p95_us": round(p95_us, 3),
            "p99_us": round(p99_us, 3),
            "max_us": round(max_us, 3),
            "num_flush_spikes": int(len(spikes)),
            "flush_intervals": [int(x) for x in spike_intervals[:20]] if len(spike_intervals) > 0 else []
        },
        "burst_scaling_sweep": sweep_results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    print("-" * 85)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
