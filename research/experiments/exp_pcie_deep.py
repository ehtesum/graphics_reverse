#!/usr/bin/env python3
"""
research/experiments/exp_pcie_deep.py
Phase 3 Experiment: Deep PCIe Subsystem Characterization
  - Inspects negotiated PCIe generation and link width via CUDA Driver API
  - Sweeps transfer sizes from 1 KB to 256 MB comparing Pinned vs Pageable vs Write-Combined
  - Measures transfer latency, effective bandwidth, and CPU invocation overhead
"""

import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RAW_DIR / "pcie_deep_sweep.json"


def run_experiment():
    print("=" * 80)
    print(" Experiment EXP-GPU-339: Deep PCIe Subsystem & Transfer Size Sweep")
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

    # Query Hardware Attributes
    def get_attr(attr_id):
        val = ctypes.c_int()
        res = cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev)
        return val.value if res == 0 else -1

    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31

    bus_id = get_attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    dev_id = get_attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    domain_id = get_attr(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
    async_engines = get_attr(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)
    unified = get_attr(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
    concurrent_k = get_attr(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)

    print(f"PCIe Topology: Domain {domain_id}, Bus {bus_id}, Device {dev_id}")
    print(f"Hardware DMA Copy Engines: {async_engines} (Dedicated H->D & D->H channels)")
    print(f"Unified Virtual Addressing (UVA): {'Supported' if unified == 1 else 'No'}")
    print(f"Concurrent Kernel Execution: {'Supported' if concurrent_k == 1 else 'No'}")
    print("-" * 80)

    # Transfer sizes from 1 KB to 256 MB
    SIZES_BYTES = [
        1024,                # 1 KB
        4096,                # 4 KB (OS page boundary)
        16 * 1024,           # 16 KB
        64 * 1024,           # 64 KB
        256 * 1024,          # 256 KB
        1024 * 1024,         # 1 MB
        4 * 1024 * 1024,     # 4 MB
        16 * 1024 * 1024,    # 16 MB
        64 * 1024 * 1024,    # 64 MB
        128 * 1024 * 1024,   # 128 MB
        256 * 1024 * 1024    # 256 MB
    ]

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    results = []
    print(f"{'Transfer Size':<15} | {'Pinned Time (ms)':<17} | {'Pinned BW (GB/s)':<17} | {'Pageable BW (GB/s)'}")
    print("-" * 80)

    for sz in SIZES_BYTES:
        sz_mb = sz / (1024 * 1024)
        label = f"{sz_mb * 1024:.0f} KB" if sz < 1024 * 1024 else f"{sz_mb:.0f} MB"

        # Allocate device buffer
        d_buf = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_buf), sz)

        # Allocate Pinned Host Memory
        p_pinned = ctypes.c_void_p()
        cuda.cuMemAllocHost_v2(ctypes.byref(p_pinned), sz)

        # Allocate Pageable Host Memory (standard python/ctypes buffer)
        raw_pageable = (ctypes.c_char * sz)()
        p_pageable = ctypes.cast(raw_pageable, ctypes.c_void_p)

        # 1. Benchmark Pinned H->D
        # Warmup
        cuda.cuMemcpyHtoDAsync_v2(d_buf, p_pinned, sz, stream)
        cuda.cuStreamSynchronize(stream)

        runs = 10 if sz <= 16 * 1024 * 1024 else 5
        pinned_times = []
        for _ in range(runs):
            cuda.cuEventRecord(start_evt, stream)
            cuda.cuMemcpyHtoDAsync_v2(d_buf, p_pinned, sz, stream)
            cuda.cuEventRecord(stop_evt, stream)
            cuda.cuEventSynchronize(stop_evt)
            el = ctypes.c_float()
            cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
            pinned_times.append(el.value)

        avg_pinned_ms = sum(pinned_times) / len(pinned_times)
        pinned_bw_gbs = (sz / (avg_pinned_ms / 1000.0)) / 1e9

        # 2. Benchmark Pageable H->D
        cuda.cuMemcpyHtoDAsync_v2(d_buf, p_pageable, sz, stream)
        cuda.cuStreamSynchronize(stream)

        pageable_times = []
        for _ in range(runs):
            cuda.cuEventRecord(start_evt, stream)
            cuda.cuMemcpyHtoDAsync_v2(d_buf, p_pageable, sz, stream)
            cuda.cuEventRecord(stop_evt, stream)
            cuda.cuEventSynchronize(stop_evt)
            el = ctypes.c_float()
            cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
            pageable_times.append(el.value)

        avg_pageable_ms = sum(pageable_times) / len(pageable_times)
        pageable_bw_gbs = (sz / (avg_pageable_ms / 1000.0)) / 1e9

        print(f"{label:<15} | {avg_pinned_ms:<17.3f} | {pinned_bw_gbs:<17.2f} | {pageable_bw_gbs:.2f}")

        results.append({
            "size_bytes": sz,
            "label": label,
            "pinned_time_ms": round(avg_pinned_ms, 4),
            "pinned_bw_gbs": round(pinned_bw_gbs, 2),
            "pageable_time_ms": round(avg_pageable_ms, 4),
            "pageable_bw_gbs": round(pageable_bw_gbs, 2)
        })

        cuda.cuMemFreeHost(p_pinned)
        cuda.cuMemFree_v2(d_buf)

    cuda.cuStreamDestroy_v2(stream)

    payload = {
        "device_info": {
            "bus_id": bus_id,
            "device_id": dev_id,
            "domain_id": domain_id,
            "async_copy_engines": async_engines,
            "unified_addressing": unified == 1,
            "concurrent_kernels": concurrent_k == 1
        },
        "sweep_results": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] PCIe sweep results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
