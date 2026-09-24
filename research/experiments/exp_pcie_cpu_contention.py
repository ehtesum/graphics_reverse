#!/usr/bin/env python3
"""
research/experiments/exp_pcie_cpu_contention.py
Step 5 Discovery Experiment: CPU Memory Saturation vs GPU PCIe Contention
Target: AMD Ryzen 7 4800HS (Renoir APU) + NVIDIA GeForce GTX 1650 (PCIe 3.0 x8)

Microarchitectural Hypotheses:
  1. On the Renoir APU, the PCIe Root Complex and the 8 CPU cores share the Unified Memory
     Controller (UMC) connected to dual-channel DDR4-3200.
  2. Saturated PCIe 3.0 x8 DMA transfers (~6.68 GB/s) compete with CPU AVX2 memory streaming (~32.3 GB/s)
     for memory controller read/write command queues and DRAM bank access.
  3. Concurrent execution reveals the arbitration policy and throughput penalty of the SoC fabric.
"""

import ctypes
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
OUT_FILE = RAW_DIR / "pcie_cpu_contention_results.json"
CPU_BENCH_EXE = ROOT / "cpu" / "benchmarks" / "exp_dram_bandwidth.exe"


def run_gpu_stream(duration_sec, stop_event, results_dict):
    """Continuously streams 128 MB chunks over PCIe 3.0 x8 until stop_event is set."""
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"Error loading nvcuda.dll: {e}")
        return

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    CHUNK_BYTES = 128 * 1024 * 1024 # 128 MB
    h_ptr = ctypes.c_void_p()
    cuda.cuMemAllocHost_v2(ctypes.byref(h_ptr), CHUNK_BYTES)
    d_ptr = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_ptr), CHUNK_BYTES)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    # Initialize buffer
    ctypes.memset(h_ptr, 0xAB, CHUNK_BYTES)

    total_bytes = 0
    t0 = time.perf_counter()

    while not stop_event.is_set():
        cuda.cuMemcpyHtoDAsync_v2(d_ptr, h_ptr, CHUNK_BYTES, stream)
        cuda.cuStreamSynchronize(stream)
        total_bytes += CHUNK_BYTES

    t1 = time.perf_counter()
    elapsed = t1 - t0
    bw_gb_s = (total_bytes / elapsed) / 1e9

    results_dict["gpu_bytes"] = total_bytes
    results_dict["gpu_elapsed"] = elapsed
    results_dict["gpu_bw_gb_s"] = bw_gb_s

    cuda.cuMemFreeHost(h_ptr)
    cuda.cuMemFree_v2(d_ptr)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuCtxDestroy_v2(ctx)


def run_cpu_bench(duration_sec):
    """Executes the multi-core AVX2 DRAM bandwidth benchmark."""
    out = subprocess.check_output([str(CPU_BENCH_EXE), str(duration_sec)], text=True)
    lines = out.strip().splitlines()
    res = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            res[k.strip()] = float(v.strip())
    return res


def main():
    print("=" * 88)
    print(" Step 5: CPU Memory Saturation vs GPU PCIe Contention Benchmark")
    print(" Architecture: AMD Renoir Unified Memory Controller + Turing PCIe 3.0 x8 Root Complex")
    print("=" * 88)

    DURATION = 2.5 # seconds per test phase

    # Phase 1: CPU Alone
    print("\n[Phase 1] Measuring Baseline CPU Multi-Core AVX2 DRAM Bandwidth (Alone)...")
    cpu_alone = run_cpu_bench(DURATION)
    cpu_alone_bw = cpu_alone.get("CPU_DRAM_BANDWIDTH_GB_S", 0.0)
    print(f"  -> CPU Alone Bandwidth: {cpu_alone_bw:.3f} GB/s")

    # Phase 2: GPU PCIe Alone
    print("\n[Phase 2] Measuring Baseline GPU PCIe 3.0 x8 Bandwidth (Alone)...")
    stop_evt = threading.Event()
    gpu_res_alone = {}
    th_gpu = threading.Thread(target=run_gpu_stream, args=(DURATION, stop_evt, gpu_res_alone))
    th_gpu.start()
    time.sleep(DURATION)
    stop_evt.set()
    th_gpu.join()
    gpu_alone_bw = gpu_res_alone.get("gpu_bw_gb_s", 0.0)
    print(f"  -> GPU PCIe Alone Bandwidth: {gpu_alone_bw:.3f} GB/s")

    # Phase 3: Concurrent CPU + GPU
    print("\n[Phase 3] Measuring Concurrent CPU AVX2 + GPU PCIe DMA Traffic...")
    stop_evt_conc = threading.Event()
    gpu_res_conc = {}
    th_gpu_conc = threading.Thread(target=run_gpu_stream, args=(DURATION, stop_evt_conc, gpu_res_conc))
    th_gpu_conc.start()
    time.sleep(0.2) # Allow GPU PCIe DMA pipeline to fully saturate

    cpu_conc = run_cpu_bench(DURATION)
    stop_evt_conc.set()
    th_gpu_conc.join()

    cpu_conc_bw = cpu_conc.get("CPU_DRAM_BANDWIDTH_GB_S", 0.0)
    gpu_conc_bw = gpu_res_conc.get("gpu_bw_gb_s", 0.0)

    cpu_drop_pct = ((cpu_alone_bw - cpu_conc_bw) / cpu_alone_bw) * 100.0
    gpu_drop_pct = ((gpu_alone_bw - gpu_conc_bw) / gpu_alone_bw) * 100.0
    total_system_dram_bw = cpu_conc_bw + gpu_conc_bw

    print("\n" + "=" * 88)
    print(" EMPIRICAL CONTENTION TELEMETRY SUMMARY")
    print("=" * 88)
    print(f" CPU DRAM Bandwidth (Alone)      : {cpu_alone_bw:.3f} GB/s")
    print(f" CPU DRAM Bandwidth (Concurrent) : {cpu_conc_bw:.3f} GB/s  (Drop: -{cpu_drop_pct:.2f}%)")
    print(f" GPU PCIe Bandwidth (Alone)      : {gpu_alone_bw:.3f} GB/s")
    print(f" GPU PCIe Bandwidth (Concurrent) : {gpu_conc_bw:.3f} GB/s  (Drop: -{gpu_drop_pct:.2f}%)")
    print(f" Combined System Memory Traffic  : {total_system_dram_bw:.3f} GB/s")
    print("-" * 88)

    if cpu_drop_pct < 10.0 and gpu_drop_pct < 10.0:
        diagnosis = "Near-Ideal Interconnect Isolation: AMD Renoir fabric prioritizes PCIe DMA without starving CPU queues"
    else:
        diagnosis = "Measurable Memory Controller Arbitration Bottleneck"
    print(f" Fabric Behavior Diagnosis: {diagnosis}\n")

    summary = {
        "cpu_alone_gb_s": round(cpu_alone_bw, 3),
        "cpu_concurrent_gb_s": round(cpu_conc_bw, 3),
        "cpu_degradation_pct": round(cpu_drop_pct, 2),
        "gpu_pcie_alone_gb_s": round(gpu_alone_bw, 3),
        "gpu_pcie_concurrent_gb_s": round(gpu_conc_bw, 3),
        "gpu_pcie_degradation_pct": round(gpu_drop_pct, 2),
        "combined_system_bw_gb_s": round(total_system_dram_bw, 3),
        "diagnosis": diagnosis
    }

    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Contention telemetry saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    main()
