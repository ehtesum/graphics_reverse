#!/usr/bin/env python3
"""
experiments/009_engine_concurrency.py
Frontier 3: GPU Hardware Engines & Asynchronous Concurrency (GTX 1650)

Investigates whether the NVIDIA GeForce GTX 1650 hardware scheduler supports
true concurrency between the SM Compute Engine and the DMA Copy Engine.
Compares:
  - Sequential Execution: Copy then Compute (Stream 0)
  - Overlapped Concurrent Execution: Simultaneous Copy and Compute on dual streams
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "009_engine_concurrency"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM

def build_compute_kernel():
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry HeavyComputeKernel(
    .param .u64 param_arr,
    .param .u32 param_iters
)
{
    .reg .b32 %r<10>;
    .reg .f32 %f<10>;
    .reg .b64 %rd<5>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u32 %r1, [param_iters];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;

    ld.global.f32 %f1, [%rd3];
    mov.u32 %r6, 0;

LOOP:
    // Heavy ALU computation loop
    fma.rn.f32 %f1, %f1, 1.000001, 0.000001;
    fma.rn.f32 %f1, %f1, 0.999999, 0.000001;
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p1, %r6, %r1;
    @%p1 bra LOOP;

    st.global.f32 [%rd3], %f1;
    ret;
}
"""
    ptx_file = RESULTS_DIR / "heavy_compute.ptx"
    cubin_file = RESULTS_DIR / "heavy_compute.cubin"
    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)
    return cubin_file


def run_concurrency_test():
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"Error loading nvcuda.dll: {e}")
        sys.exit(1)

    def check(res, op):
        if res != 0:
            raise RuntimeError(f"{op} failed with code {res}")

    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    # Check concurrent copy/compute engine support attribute
    async_engine_count = ctypes.c_int()
    check(cuda.cuDeviceGetAttribute(
        ctypes.byref(async_engine_count),
        21, # CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
        dev
    ), "cuDeviceGetAttribute ASYNC_ENGINE_COUNT")

    cubin_file = build_compute_kernel()
    module = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(module), str(cubin_file.resolve()).encode("utf-8")), "cuModuleLoad")
    kernel_func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(kernel_func), module, b"HeavyComputeKernel"), "cuModuleGetFunction")

    # Memory allocation:
    # 1. Compute buffer: 1M floats = 4 MB
    COMPUTE_N = 1024 * 1024
    compute_bytes = COMPUTE_N * 4
    d_compute = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_compute), compute_bytes), "cuMemAlloc compute")

    # 2. DMA Copy buffer: 64 MB pinned host memory and device memory
    COPY_BYTES = 64 * 1024 * 1024
    d_copy = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_copy), COPY_BYTES), "cuMemAlloc copy")
    h_pinned = ctypes.c_void_p()
    check(cuda.cuMemAllocHost_v2(ctypes.byref(h_pinned), COPY_BYTES), "cuMemAllocHost")

    # Create Streams
    stream_compute = ctypes.c_void_p()
    stream_copy = ctypes.c_void_p()
    check(cuda.cuStreamCreate(ctypes.byref(stream_compute), 1), "cuStreamCreate compute") # non-blocking
    check(cuda.cuStreamCreate(ctypes.byref(stream_copy), 1), "cuStreamCreate copy")       # non-blocking

    # Setup Kernel Args
    iterations = ctypes.c_uint32(2000) # tuned for ~30-50ms of pure compute
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(d_compute), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iterations), ctypes.c_void_p)
    )

    # 1. Measure Individual Isolated Times
    evt_start = ctypes.c_void_p()
    evt_stop = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(evt_start), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(evt_stop), 0), "cuEventCreate")

    # Compute-only duration
    check(cuda.cuEventRecord(evt_start, stream_compute), "Start compute")
    check(cuda.cuLaunchKernel(kernel_func, 4096, 1, 1, 256, 1, 1, 0, stream_compute, k_args, None), "Launch compute")
    check(cuda.cuEventRecord(evt_stop, stream_compute), "Stop compute")
    check(cuda.cuEventSynchronize(evt_stop), "Sync compute")
    t_compute_only = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_compute_only), evt_start, evt_stop), "Elapsed compute")

    # DMA Copy-only duration (64 MB H2D)
    check(cuda.cuEventRecord(evt_start, stream_copy), "Start copy")
    check(cuda.cuMemcpyHtoDAsync_v2(d_copy, h_pinned, COPY_BYTES, stream_copy), "Async Copy")
    check(cuda.cuEventRecord(evt_stop, stream_copy), "Stop copy")
    check(cuda.cuEventSynchronize(evt_stop), "Sync copy")
    t_copy_only = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_copy_only), evt_start, evt_stop), "Elapsed copy")

    theoretical_sequential = t_compute_only.value + t_copy_only.value

    # 2. Overlapped Concurrent Launch
    evt_concurrent_start = ctypes.c_void_p()
    evt_concurrent_stop = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(evt_concurrent_start), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(evt_concurrent_stop), 0), "cuEventCreate")

    # Ensure device is idle
    check(cuda.cuCtxSynchronize(), "Pre-sync")

    check(cuda.cuEventRecord(evt_concurrent_start, None), "Start concurrent")
    # Launch Compute on Stream 1
    check(cuda.cuLaunchKernel(kernel_func, 4096, 1, 1, 256, 1, 1, 0, stream_compute, k_args, None), "Launch stream1")
    # Concurrently launch DMA Copy on Stream 2
    check(cuda.cuMemcpyHtoDAsync_v2(d_copy, h_pinned, COPY_BYTES, stream_copy), "Launch stream2 async")
    # Wait for both streams
    check(cuda.cuStreamSynchronize(stream_compute), "Sync stream1")
    check(cuda.cuStreamSynchronize(stream_copy), "Sync stream2")
    check(cuda.cuEventRecord(evt_concurrent_stop, None), "Stop concurrent")
    check(cuda.cuEventSynchronize(evt_concurrent_stop), "Sync concurrent")

    t_concurrent = ctypes.c_float()
    check(cuda.cuEventElapsedTime(ctypes.byref(t_concurrent), evt_concurrent_start, evt_concurrent_stop), "Elapsed concurrent")

    overlap_savings_ms = theoretical_sequential - t_concurrent.value
    overlap_efficiency = (overlap_savings_ms / min(t_compute_only.value, t_copy_only.value)) * 100.0

    print("=" * 75)
    print(" Frontier 3: GPU Hardware Engine Concurrency (GTX 1650)")
    print(" DMA Copy Engine vs SM Compute Engine Overlap Benchmark")
    print("=" * 75)
    print(f"Physical Hardware Async Engines Reported: {async_engine_count.value}")
    print("-" * 75)
    print(f"Isolated Compute Engine Time (Stream 1):  {t_compute_only.value:.2f} ms")
    print(f"Isolated DMA Copy Engine Time (Stream 2): {t_copy_only.value:.2f} ms (64 MB Pinned H2D)")
    print(f"Theoretical Sequential Total:             {theoretical_sequential:.2f} ms")
    print("-" * 75)
    print(f"Observed Concurrent Overlapped Time:      {t_concurrent.value:.2f} ms")
    print(f"Hardware Overlap Time Savings:            {overlap_savings_ms:.2f} ms")
    print(f"Engine Concurrency Overlap Efficiency:    {overlap_efficiency:.1f}%")
    print("=" * 75)

    res = {
        "device": "NVIDIA GeForce GTX 1650",
        "async_engine_count": async_engine_count.value,
        "compute_only_ms": round(t_compute_only.value, 2),
        "copy_only_ms": round(t_copy_only.value, 2),
        "theoretical_sequential_ms": round(theoretical_sequential, 2),
        "concurrent_ms": round(t_concurrent.value, 2),
        "savings_ms": round(overlap_savings_ms, 2),
        "concurrency_efficiency_pct": round(overlap_efficiency, 1)
    }

    out_file = RESULTS_DIR / "concurrency.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[SUCCESS] Saved engine concurrency results to: {out_file}\n")

    # Cleanup
    cuda.cuMemFree_v2(d_compute)
    cuda.cuMemFree_v2(d_copy)
    cuda.cuMemFreeHost(h_pinned)
    cuda.cuStreamDestroy_v2(stream_compute)
    cuda.cuStreamDestroy_v2(stream_copy)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    run_concurrency_test()
