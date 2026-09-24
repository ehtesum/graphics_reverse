#!/usr/bin/env python3
"""
research/experiments/exp_multistream_concurrency.py
Forensic Characterization: Corner H3 - Multi-Stream Hardware Queue Concurrency Limits
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5 Turing, 14 SMs), Windows 11 WDDM 2.7

Methodology:
1. Create K independent CUDA streams (K in [1, 2, 4, 8, 12, 14, 16, 24, 32]).
2. Launch a 1-block spin kernel (lasting exactly ~2.0 ms on 1 SM) into each stream simultaneously.
   Each kernel executes on 1 SM (block_dim = 32 threads).
3. If the hardware scheduler can dispatch K streams concurrently across the 14 SMs, total wall
   time will remain ~2.0 ms for K <= 14.
4. Detect the exact hardware queue limit and WDDM serialization threshold.
"""

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "multistream_concurrency_results.json"


def compile_spin_kernel():
    ptx = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry SpinKernel(
    .param .u64 spin_cycles,
    .param .u64 d_out
) {
    .reg .u64 %cycles, %out_ptr, %t0, %t1, %dt;
    .reg .pred %p0;

    ld.param.u64 %cycles, [spin_cycles];
    ld.param.u64 %out_ptr, [d_out];

    mov.u64 %t0, %clock64;

    $spin_loop:
    mov.u64 %t1, %clock64;
    sub.u64 %dt, %t1, %t0;
    setp.lt.u64 %p0, %dt, %cycles;
    @%p0 bra $spin_loop;

    .reg .pred %p_first;
    .reg .u32 %tidx;
    mov.u32 %tidx, %tid.x;
    setp.eq.u32 %p_first, %tidx, 0;
    @%p_first st.global.u64 [%out_ptr], %dt;
    ret;
}
"""
    p_file = RAW_DIR / "spin_kernel.ptx"
    c_file = RAW_DIR / "spin_kernel.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER H3: Multi-Stream Hardware Queue Concurrency Limits")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117 Turing) | 14 SMs | WDDM 2.7")
    print("=" * 80)

    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"[ERROR] Failed to load nvcuda.dll: {e}")
        sys.exit(1)

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)

    # Check concurrent kernels attribute (attr 15: CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    concurrent_attr = ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(concurrent_attr), 15, dev)
    print(f"[*] Hardware Concurrent Kernel Execution Support: {bool(concurrent_attr.value)}")

    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    c_file = compile_spin_kernel()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"SpinKernel")

    # Spin duration: ~3,000,000 GPU clock cycles (~2.0 ms at 1500 MHz)
    SPIN_CYCLES = 3000000
    p_cycles = ctypes.c_uint64(SPIN_CYCLES)

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8)
    p_out = ctypes.c_uint64(d_out.value)

    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_cycles), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
    )

    STREAM_COUNTS = [1, 2, 4, 7, 14, 16, 24, 32]
    results = []

    print(f"\n{'Streams (K)':<14} | {'Total Wall Time (ms)':<22} | {'Expected If Serial (ms)':<24} | {'Concurrency Speedup'}")
    print("-" * 85)

    base_time_ms = 1.0

    for K in STREAM_COUNTS:
        # Create K non-blocking streams
        streams = []
        for _ in range(K):
            s = ctypes.c_void_p()
            # CU_STREAM_NON_BLOCKING = 0x01
            cuda.cuStreamCreate(ctypes.byref(s), 0x01)
            streams.append(s)

        # Warmup
        for s in streams:
            cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, s, args, None)
        for s in streams:
            cuda.cuStreamSynchronize(s)

        # Timed execution: launch 1 kernel into each of the K streams
        t0 = time.perf_counter()
        for s in streams:
            cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, s, args, None)
        for s in streams:
            cuda.cuStreamSynchronize(s)
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000.0
        if K == 1:
            base_time_ms = wall_ms

        expected_serial_ms = base_time_ms * K
        speedup = expected_serial_ms / wall_ms

        print(f"{K:<14} | {wall_ms:<22.3f} | {expected_serial_ms:<24.3f} | {speedup:.2f}x")

        results.append({
            "streams_K": K,
            "wall_time_ms": round(wall_ms, 3),
            "expected_serial_ms": round(expected_serial_ms, 3),
            "speedup_vs_serial": round(speedup, 2)
        })

        for s in streams:
            cuda.cuStreamDestroy_v2(s)

    output = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "total_sms": 14,
        "results": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 85)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
