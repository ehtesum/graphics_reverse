#!/usr/bin/env python3
"""
research/experiments/exp_pcie_duplex_saturation.py
Forensic Characterization: Corner H4 - PCIe Full-Duplex Bidirectional Saturation
Hardware Target: AMD Ryzen 7 4800HS <--> NVIDIA GeForce GTX 1650 (TU117) via PCIe 3.0 x8

Methodology:
1. Allocate page-locked pinned host buffers and device VRAM buffers of sizes [1, 4, 16, 64, 128 MB].
2. Measure:
   a) Simplex H2D (Host to Device) bandwidth across sizes.
   b) Simplex D2H (Device to Host) bandwidth across sizes.
   c) Full-Duplex Bidirectional (Simultaneous H2D + D2H on two concurrent streams).
3. Determine whether PCIe 3.0 x8 achieves full physical duplex saturation (~12.8 GB/s aggregate)
   or suffers from root complex DMA engine contention under Windows 11 WDDM 2.7.
"""

import ctypes
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
OUT_FILE = RAW_DIR / "pcie_duplex_results.json"


def run_experiment():
    print("=" * 80)
    print(" CORNER H4: PCIe Full-Duplex Bidirectional Saturation (H2D + D2H Concurrent)")
    print(" Target: AMD Renoir Root Complex <--> NVIDIA TU117 via PCIe 3.0 x8 | WDDM 2.7")
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

    # Streams for concurrent duplex transfers
    stream_h2d = ctypes.c_void_p()
    stream_d2h = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream_h2d), 0x01) # NON_BLOCKING
    cuda.cuStreamCreate(ctypes.byref(stream_d2h), 0x01) # NON_BLOCKING

    ev_start = ctypes.c_void_p()
    ev_stop = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(ev_start), 0)
    cuda.cuEventCreate(ctypes.byref(ev_stop), 0)

    # Set argtypes
    cuda.cuMemcpyHtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    cuda.cuMemcpyDtoHAsync_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]

    MAX_BYTES = 128 * 1024 * 1024

    # Host pinned buffers
    h_src = ctypes.c_void_p()
    h_dst = ctypes.c_void_p()
    cuda.cuMemHostAlloc(ctypes.byref(h_src), MAX_BYTES, 0)
    cuda.cuMemHostAlloc(ctypes.byref(h_dst), MAX_BYTES, 0)

    # Device VRAM buffers
    d_buf1 = ctypes.c_uint64()
    d_buf2 = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_buf1), MAX_BYTES)
    cuda.cuMemAlloc_v2(ctypes.byref(d_buf2), MAX_BYTES)

    SIZES_MB = [1, 4, 16, 64, 128]
    results = []

    print(f"{'Size (MB)':<10} | {'Simplex H2D (GB/s)':<20} | {'Simplex D2H (GB/s)':<20} | {'Duplex Agg (GB/s)':<20} | {'Duplex Efficiency'}")
    print("-" * 95)

    for sz_mb in SIZES_MB:
        bytes_count = sz_mb * 1024 * 1024
        iters = 20 if sz_mb <= 16 else 5

        # --- 1. Simplex H2D ---
        cuda.cuStreamSynchronize(stream_h2d)
        t0 = time.perf_counter()
        for _ in range(iters):
            cuda.cuMemcpyHtoDAsync_v2(d_buf1.value, h_src, bytes_count, stream_h2d)
        cuda.cuStreamSynchronize(stream_h2d)
        t1 = time.perf_counter()
        h2d_bw = ((bytes_count * iters) / 1e9) / (t1 - t0)

        # --- 2. Simplex D2H ---
        cuda.cuStreamSynchronize(stream_d2h)
        t0 = time.perf_counter()
        for _ in range(iters):
            cuda.cuMemcpyDtoHAsync_v2(h_dst, d_buf2.value, bytes_count, stream_d2h)
        cuda.cuStreamSynchronize(stream_d2h)
        t1 = time.perf_counter()
        d2h_bw = ((bytes_count * iters) / 1e9) / (t1 - t0)

        # --- 3. Full-Duplex (Concurrent H2D + D2H) ---
        cuda.cuStreamSynchronize(stream_h2d)
        cuda.cuStreamSynchronize(stream_d2h)
        t0 = time.perf_counter()
        for _ in range(iters):
            cuda.cuMemcpyHtoDAsync_v2(d_buf1.value, h_src, bytes_count, stream_h2d)
            cuda.cuMemcpyDtoHAsync_v2(h_dst, d_buf2.value, bytes_count, stream_d2h)
        cuda.cuStreamSynchronize(stream_h2d)
        cuda.cuStreamSynchronize(stream_d2h)
        t1 = time.perf_counter()
        total_duplex_bytes = bytes_count * 2 * iters
        duplex_agg_bw = (total_duplex_bytes / 1e9) / (t1 - t0)

        efficiency = duplex_agg_bw / (h2d_bw + d2h_bw)

        print(f"{sz_mb:<10} | {h2d_bw:<20.2f} | {d2h_bw:<20.2f} | {duplex_agg_bw:<20.2f} | {efficiency * 100.0:.1f}%")

        results.append({
            "size_mb": sz_mb,
            "simplex_h2d_gbps": round(h2d_bw, 2),
            "simplex_d2h_gbps": round(d2h_bw, 2),
            "duplex_aggregate_gbps": round(duplex_agg_bw, 2),
            "duplex_efficiency_pct": round(efficiency * 100.0, 1)
        })

    output = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117) on PCIe 3.0 x8",
        "host": "AMD Ryzen 7 4800HS",
        "results": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 95)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFreeHost(h_src)
    cuda.cuMemFreeHost(h_dst)
    cuda.cuMemFree_v2(d_buf1)
    cuda.cuMemFree_v2(d_buf2)
    cuda.cuStreamDestroy_v2(stream_h2d)
    cuda.cuStreamDestroy_v2(stream_d2h)
    cuda.cuModuleUnload(mod) if 'mod' in locals() else None


if __name__ == "__main__":
    run_experiment()
