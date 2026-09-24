#!/usr/bin/env python3
"""
research/experiments/exp_constant_memory.py
Corner 1 Forensic Experiment: GPU Constant Memory Warp Broadcast vs Serialization
Target: NVIDIA GeForce GTX 1650 4 GB (Turing TU117, SM 7.5)

Microarchitectural Hypotheses:
  1. Constant memory (c[0x0][...]) is served by an 8 KB L1 Constant Cache per SM.
  2. When all 32 threads in a warp access the same constant address (threadIdx.x & 0),
     the value is broadcast in a single clock cycle.
  3. When threads in a warp access K distinct constant addresses, the constant cache
     serializes execution, scaling latency linearly by K (up to 32x for K=32).
  4. Accesses outside the 8 KB L1 constant cache miss to L2, introducing additional latency.
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
OUT_FILE = RAW_DIR / "constant_memory_results.json"


def generate_constant_ptx():
    """Generates PTX with a 16 KB constant array and a benchmark kernel."""
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        "",
        "// Allocate 16 KB constant memory buffer (4,096 uint32 words)",
        ".const .u32 const_table[4096];",
        "",
        ".visible .entry ConstMemBench(",
        "    .param .u64 param_out,",
        "    .param .u32 param_mask,",
        "    .param .u32 param_stride,",
        "    .param .u32 param_iters",
        ")",
        "{",
        "    .reg .b32 %r<20>;",
        "    .reg .b64 %rd<10>;",
        "    .reg .pred %p<2>;",
        "",
        "    ld.param.u64 %rd1, [param_out];",
        "    ld.param.u32 %r1, [param_mask];",
        "    ld.param.u32 %r2, [param_stride];",
        "    ld.param.u32 %r3, [param_iters];",
        "",
        "    mov.u32 %r4, %tid.x;",
        "    and.b32 %r5, %r4, %r1; // Apply divergence mask (0 to 31)",
        "    mul.lo.u32 %r6, %r5, %r2; // Stride offset",
        "    and.b32 %r6, %r6, 4095; // Wrap within 16 KB table",
        "",
        "    mov.u32 %r10, 0;",
        "    mov.u32 %r11, 0;",
        "",
        "LOOP:",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    ld.const.u32 %r12, const_table[%r6];",
        "    add.u32 %r10, %r10, %r12;",
        "    xor.b32 %r6, %r6, %r12;",
        "    and.b32 %r6, %r6, 4095;",
        "",
        "    add.u32 %r11, %r11, 1;",
        "    setp.lt.u32 %p1, %r11, %r3;",
        "    @%p1 bra LOOP;",
        "",
        "    // Write output to prevent DCE",
        "    mul.wide.u32 %rd2, %r4, 4;",
        "    add.u64 %rd3, %rd1, %rd2;",
        "    st.global.u32 [%rd3], %r10;",
        "    ret;",
        "}"
    ]
    return "\n".join(ptx)


def run_experiment():
    print("=" * 90)
    print(" Corner 1: GPU Constant Memory Warp Broadcast vs Serialization Forensic Audit")
    print(" Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)")
    print("=" * 90)

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

    # Compile PTX
    ptx_src = generate_constant_ptx()
    p_file = RAW_DIR / "const_bench.ptx"
    c_file = RAW_DIR / "const_bench.cubin"
    with open(p_file, "w") as f:
        f.write(ptx_src)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"ConstMemBench")

    # Populate constant memory table
    const_ptr = ctypes.c_uint64()
    const_bytes = ctypes.c_size_t()
    cuda.cuModuleGetGlobal_v2(ctypes.byref(const_ptr), ctypes.byref(const_bytes), mod, b"const_table")
    init_data = (ctypes.c_uint32 * 4096)(*[i + 1 for i in range(4096)])
    cuda.cuMemcpyHtoD_v2(const_ptr, init_data, 4096 * 4)

    # Allocate device output buffer
    BLOCKS = 64
    THREADS = 1024 # 65,536 threads saturating all 14 SMs
    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), BLOCKS * THREADS * 4)

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    LOOP_ITERS = 10000
    TOTAL_LOADS_PER_THREAD = LOOP_ITERS * 8

    # Test cases: Divergent address count per warp
    # mask = 0 (1 addr / broadcast), mask = 1 (2 addrs), mask = 3 (4 addrs), mask = 7 (8 addrs),
    # mask = 15 (16 addrs), mask = 31 (32 addrs, complete serialization)
    TESTS = [
        ("1 Address (Uniform Broadcast)", 0, 1, "Single-cycle broadcast to 32 threads"),
        ("2 Distinct Addresses", 1, 1, "2-way warp serialization"),
        ("4 Distinct Addresses", 3, 1, "4-way warp serialization"),
        ("8 Distinct Addresses", 7, 1, "8-way warp serialization"),
        ("16 Distinct Addresses", 15, 1, "16-way warp serialization"),
        ("32 Distinct Addresses (Full Divergence)", 31, 1, "32-way serialized cache reads"),
        ("32 Addresses (L1 Cache Stride 64B)", 31, 16, "Accessing across multiple 64B cache lines"),
        ("32 Addresses (L1 Cache Stride 256B)", 31, 64, "Exceeding L1 constant cache lines")
    ]

    print(f"{'Access Pattern':<40} | {'Kernel Time (ms)':<18} | {'Throughput (GLoads/s)':<22} | {'Relative Latency'}")
    print("-" * 96)

    baseline_ms = 0.0
    results = []

    p_out = ctypes.c_uint64(d_out.value)
    iters_arg = ctypes.c_uint32(LOOP_ITERS)

    for label, mask_val, stride_val, desc in TESTS:
        mask_arg = ctypes.c_uint32(mask_val)
        stride_arg = ctypes.c_uint32(stride_val)

        k_args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(mask_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(stride_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(iters_arg), ctypes.c_void_p)
        )

        # Warmup
        cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()

        # Measurement
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
        total_loads = BLOCKS * THREADS * TOTAL_LOADS_PER_THREAD
        gloads_sec = (total_loads / (avg_ms / 1000.0)) / 1e9

        if mask_val == 0 and stride_val == 1:
            baseline_ms = avg_ms
            rel_str = "1.00x (Baseline Broadcast)"
        else:
            rel_str = f"{avg_ms / baseline_ms:.2f}x"

        print(f"{label:<40} | {avg_ms:<18.3f} | {gloads_sec:<22.1f} | {rel_str}")

        results.append({
            "label": label,
            "mask": mask_val,
            "stride": stride_val,
            "description": desc,
            "time_ms": round(avg_ms, 3),
            "throughput_gloads_sec": round(gloads_sec, 1),
            "relative_latency": round(avg_ms / baseline_ms, 3)
        })

    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 96)
    print(f"[SUCCESS] Constant memory results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
