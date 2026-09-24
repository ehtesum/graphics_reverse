#!/usr/bin/env python3
"""
research/experiments/exp_heterogeneous_pingpong.py
Step 6 Discovery: Heterogeneous Activation Tensor Ping-Pong Latency
Target: AMD Ryzen 7 4800HS + NVIDIA GeForce GTX 1650 4 GB (PCIe 3.0 x8, WDDM 2.7)

Microarchitectural Hypotheses:
  1. In split-layer heterogeneous LLM execution or speculative decoding, CPU and GPU must
     exchange intermediate hidden states (activation tensors, typically 2 KB - 32 KB).
  2. The total turnaround latency is governed by:
       T_roundtrip = T_dma_init + (Bytes / BW_pcie) + T_wddm_launch + T_kernel + T_dma_d2h + T_wddm_sync
  3. Pinned (page-locked) host memory avoids OS double-buffering, significantly reducing ping-pong jitter.
  4. At small tensor sizes (<= 8 KB), latency is 100% dominated by WDDM driver submission and DMA initiation
     overheads (~25-35 µs total round-trip), not PCIe transfer bandwidth.
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
OUT_FILE = RAW_DIR / "heterogeneous_pingpong_results.json"


def generate_activation_ptx():
    """Generates an in-place FP32 vector transformation kernel (simulating activation layer)."""
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry ActivationKernel(",
        "    .param .u64 param_buf,",
        "    .param .u32 param_n",
        ")",
        "{",
        "    .reg .b32 %r<10>;",
        "    .reg .b64 %rd<10>;",
        "    .reg .f32 %f<5>;",
        "    .reg .pred %p<2>;",
        "",
        "    ld.param.u64 %rd1, [param_buf];",
        "    ld.param.u32 %r1, [param_n];",
        "",
        "    mov.u32 %r2, %ctaid.x;",
        "    mov.u32 %r3, %ntid.x;",
        "    mov.u32 %r4, %tid.x;",
        "    mad.lo.u32 %r5, %r2, %r3, %r4;",
        "",
        "    setp.ge.u32 %p1, %r5, %r1;",
        "    @%p1 bra EXIT;",
        "",
        "    mul.wide.u32 %rd2, %r5, 4;",
        "    add.u64 %rd3, %rd1, %rd2;",
        "    ld.global.f32 %f1, [%rd3];",
        "    mul.f32 %f2, %f1, 1.0001;",
        "    st.global.f32 [%rd3], %f2;",
        "",
        "EXIT:",
        "    ret;",
        "}"
    ]
    return "\n".join(ptx)


def run_experiment():
    print("=" * 88)
    print(" Step 6: Heterogeneous Activation Tensor Ping-Pong Latency Benchmark")
    print(" Target: AMD Ryzen 7 4800HS <-> NVIDIA GeForce GTX 1650 (PCIe 3.0 x8, WDDM 2.7)")
    print("=" * 88)

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

    # Compile kernel
    ptx_src = generate_activation_ptx()
    p_file = RAW_DIR / "activation.ptx"
    c_file = RAW_DIR / "activation.cubin"
    with open(p_file, "w") as f:
        f.write(ptx_src)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"ActivationKernel")

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    # Tensor sizes: from 512 B (small prompt token) to 128 KB (batch activation state)
    SIZES = [
        ("512 B", 512),
        ("1 KB", 1024),
        ("2 KB", 2048),
        ("4 KB", 4096),
        ("8 KB (1x Hidden-4096 FP16)", 8192),
        ("16 KB", 16384),
        ("32 KB (4x Hidden-4096 FP16)", 32768),
        ("64 KB", 65536),
        ("128 KB", 131072)
    ]

    ITERS = 1000

    print(f"{'Tensor Size':<28} | {'Pinned Mean (µs)':<18} | {'Pinned Min (µs)':<16} | {'Pageable Mean (µs)':<20} | {'Pinned Speedup'}")
    print("-" * 96)

    results = []

    for label, size_bytes in SIZES:
        num_floats = size_bytes // 4
        threads_per_block = 256
        blocks = (num_floats + threads_per_block - 1) // threads_per_block

        # Device buffer
        d_buf = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_buf), size_bytes)
        p_dev = ctypes.c_uint64(d_buf.value)
        n_elems = ctypes.c_uint32(num_floats)
        k_args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(p_dev), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_elems), ctypes.c_void_p)
        )

        # 1. Pinned Host Memory
        h_pin_in = ctypes.c_void_p()
        h_pin_out = ctypes.c_void_p()
        cuda.cuMemAllocHost_v2(ctypes.byref(h_pin_in), size_bytes)
        cuda.cuMemAllocHost_v2(ctypes.byref(h_pin_out), size_bytes)
        ctypes.memset(h_pin_in, 0x3F, size_bytes)

        # Warmup
        for _ in range(50):
            cuda.cuMemcpyHtoDAsync_v2(d_buf, h_pin_in, size_bytes, stream)
            cuda.cuLaunchKernel(func, blocks, 1, 1, threads_per_block, 1, 1, 0, stream, k_args, None)
            cuda.cuMemcpyDtoHAsync_v2(h_pin_out, d_buf, size_bytes, stream)
            cuda.cuStreamSynchronize(stream)

        # Benchmark Pinned
        pinned_times = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            cuda.cuMemcpyHtoDAsync_v2(d_buf, h_pin_in, size_bytes, stream)
            cuda.cuLaunchKernel(func, blocks, 1, 1, threads_per_block, 1, 1, 0, stream, k_args, None)
            cuda.cuMemcpyDtoHAsync_v2(h_pin_out, d_buf, size_bytes, stream)
            cuda.cuStreamSynchronize(stream)
            t1 = time.perf_counter_ns()
            pinned_times.append((t1 - t0) / 1000.0) # µs

        # 2. Pageable Host Memory
        h_page_in = (ctypes.c_byte * size_bytes)()
        h_page_out = (ctypes.c_byte * size_bytes)()
        ctypes.memset(ctypes.byref(h_page_in), 0x3F, size_bytes)

        # Warmup Pageable
        for _ in range(50):
            cuda.cuMemcpyHtoDAsync_v2(d_buf, ctypes.byref(h_page_in), size_bytes, stream)
            cuda.cuLaunchKernel(func, blocks, 1, 1, threads_per_block, 1, 1, 0, stream, k_args, None)
            cuda.cuMemcpyDtoHAsync_v2(ctypes.byref(h_page_out), d_buf, size_bytes, stream)
            cuda.cuStreamSynchronize(stream)

        # Benchmark Pageable
        pageable_times = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            cuda.cuMemcpyHtoDAsync_v2(d_buf, ctypes.byref(h_page_in), size_bytes, stream)
            cuda.cuLaunchKernel(func, blocks, 1, 1, threads_per_block, 1, 1, 0, stream, k_args, None)
            cuda.cuMemcpyDtoHAsync_v2(ctypes.byref(h_page_out), d_buf, size_bytes, stream)
            cuda.cuStreamSynchronize(stream)
            t1 = time.perf_counter_ns()
            pageable_times.append((t1 - t0) / 1000.0) # µs

        mean_pin = sum(pinned_times) / len(pinned_times)
        min_pin = min(pinned_times)
        mean_page = sum(pageable_times) / len(pageable_times)
        speedup = mean_page / mean_pin

        print(f"{label:<28} | {mean_pin:<18.2f} | {min_pin:<16.2f} | {mean_page:<20.2f} | {speedup:.2f}x")

        results.append({
            "label": label,
            "size_bytes": size_bytes,
            "pinned_mean_us": round(mean_pin, 2),
            "pinned_min_us": round(min_pin, 2),
            "pageable_mean_us": round(mean_page, 2),
            "pinned_speedup": round(speedup, 2)
        })

        cuda.cuMemFreeHost(h_pin_in)
        cuda.cuMemFreeHost(h_pin_out)
        cuda.cuMemFree_v2(d_buf)

    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 96)
    print(f"[SUCCESS] Heterogeneous ping-pong telemetry saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
