#!/usr/bin/env python3
"""
experiments/007_pcie_transfer.py
PCIe Host-to-Device (H2D) and Device-to-Host (D2H) Transfer Characterization

Empirically compares Pageable Host Memory vs. Page-Locked (Pinned) Memory
across buffer sizes (1 MB to 128 MB) over the physical PCIe bus between the
AMD Ryzen 7 4800HS CPU and NVIDIA GeForce GTX 1650 GPU.
"""

import ctypes
import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "007_pcie_transfer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    cuda = ctypes.WinDLL("nvcuda.dll")
except Exception as e:
    print(f"Error loading nvcuda.dll: {e}")
    sys.exit(1)

def check(res, op):
    if res != 0:
        raise RuntimeError(f"{op} failed with error code: {res}")

check(cuda.cuInit(0), "cuInit")

device = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

context = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")

start_evt = ctypes.c_void_p()
stop_evt = ctypes.c_void_p()
check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate start")
check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate stop")

sizes_mb = [1, 4, 16, 64, 128]
results_data = []

print("=" * 75)
print(" PCIe Host <-> Device Transfer Characterization (GTX 1650)")
print(" Pageable vs Pinned Memory Throughput (GB/s)")
print("=" * 75)
print(f"{'Size (MB)':<12} | {'Direction':<8} | {'Type':<10} | {'Time (ms)':<12} | {'Throughput (GB/s)':<18}")
print("-" * 75)

for size_mb in sizes_mb:
    num_bytes = size_mb * 1024 * 1024

    # Allocate Device Memory
    d_mem = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), f"cuMemAlloc {size_mb}MB")

    # 1. Pageable Host Memory
    h_pageable = (ctypes.c_byte * num_bytes)()

    # Warmup
    check(cuda.cuMemcpyHtoD_v2(d_mem, h_pageable, num_bytes), "Warmup Pageable H2D")

    # H2D Pageable
    check(cuda.cuEventRecord(start_evt, None), "Record start")
    check(cuda.cuMemcpyHtoD_v2(d_mem, h_pageable, num_bytes), "Pageable H2D")
    check(cuda.cuEventRecord(stop_evt, None), "Record stop")
    check(cuda.cuEventSynchronize(stop_evt), "Sync")
    t_h2d_pageable = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_h2d_pageable), start_evt, stop_evt), "Elapsed")
    gb_s_h2d_pageable = (num_bytes / (1024**3)) / (t_h2d_pageable.value / 1000.0)

    # D2H Pageable
    check(cuda.cuEventRecord(start_evt, None), "Record start")
    check(cuda.cuMemcpyDtoH_v2(h_pageable, d_mem, num_bytes), "Pageable D2H")
    check(cuda.cuEventRecord(stop_evt, None), "Record stop")
    check(cuda.cuEventSynchronize(stop_evt), "Sync")
    t_d2h_pageable = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_d2h_pageable), start_evt, stop_evt), "Elapsed")
    gb_s_d2h_pageable = (num_bytes / (1024**3)) / (t_d2h_pageable.value / 1000.0)

    # 2. Pinned (Page-Locked) Host Memory
    h_pinned = ctypes.c_void_p()
    check(cuda.cuMemAllocHost_v2(ctypes.byref(h_pinned), num_bytes), "cuMemAllocHost")

    # Warmup
    check(cuda.cuMemcpyHtoD_v2(d_mem, h_pinned, num_bytes), "Warmup Pinned H2D")

    # H2D Pinned
    check(cuda.cuEventRecord(start_evt, None), "Record start")
    check(cuda.cuMemcpyHtoD_v2(d_mem, h_pinned, num_bytes), "Pinned H2D")
    check(cuda.cuEventRecord(stop_evt, None), "Record stop")
    check(cuda.cuEventSynchronize(stop_evt), "Sync")
    t_h2d_pinned = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_h2d_pinned), start_evt, stop_evt), "Elapsed")
    gb_s_h2d_pinned = (num_bytes / (1024**3)) / (t_h2d_pinned.value / 1000.0)

    # D2H Pinned
    check(cuda.cuEventRecord(start_evt, None), "Record start")
    check(cuda.cuMemcpyDtoH_v2(h_pinned, d_mem, num_bytes), "Pinned D2H")
    check(cuda.cuEventRecord(stop_evt, None), "Record stop")
    check(cuda.cuEventSynchronize(stop_evt), "Sync")
    t_d2h_pinned = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_d2h_pinned), start_evt, stop_evt), "Elapsed")
    gb_s_d2h_pinned = (num_bytes / (1024**3)) / (t_d2h_pinned.value / 1000.0)

    print(f"{size_mb:<12} | {'H2D':<8} | {'Pageable':<10} | {t_h2d_pageable.value:<12.3f} | {gb_s_h2d_pageable:<18.2f}")
    print(f"{size_mb:<12} | {'H2D':<8} | {'Pinned':<10} | {t_h2d_pinned.value:<12.3f} | {gb_s_h2d_pinned:<18.2f}")
    print(f"{size_mb:<12} | {'D2H':<8} | {'Pageable':<10} | {t_d2h_pageable.value:<12.3f} | {gb_s_d2h_pageable:<18.2f}")
    print(f"{size_mb:<12} | {'D2H':<8} | {'Pinned':<10} | {t_d2h_pinned.value:<12.3f} | {gb_s_d2h_pinned:<18.2f}")
    print("-" * 75)

    results_data.append({
        "size_mb": size_mb,
        "h2d_pageable": {"time_ms": round(t_h2d_pageable.value, 3), "gb_s": round(gb_s_h2d_pageable, 2)},
        "h2d_pinned": {"time_ms": round(t_h2d_pinned.value, 3), "gb_s": round(gb_s_h2d_pinned, 2)},
        "d2h_pageable": {"time_ms": round(t_d2h_pageable.value, 3), "gb_s": round(gb_s_d2h_pageable, 2)},
        "d2h_pinned": {"time_ms": round(t_d2h_pinned.value, 3), "gb_s": round(gb_s_d2h_pinned, 2)}
    })

    cuda.cuMemFree_v2(d_mem)
    cuda.cuMemFreeHost(h_pinned)

out_file = RESULTS_DIR / "pcie_bandwidth.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)

print("=" * 75)
print(f"[SUCCESS] Saved PCIe measurements to: {out_file}\n")

cuda.cuEventDestroy(start_evt)
cuda.cuEventDestroy(stop_evt)
cuda.cuCtxDestroy_v2(context)
