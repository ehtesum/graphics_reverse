#!/usr/bin/env python3
"""
research/experiments/exp_fp16_vector_throughput.py
Forensic Characterization: Corner G3 - FP16 Half2 SIMD Packing vs FP32 Peak Throughput
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5 Turing, 896 CUDA Cores, Non-Tensor)

Methodology:
1. Compile two dense math-bound microbenchmarks:
   a) Pure FP32: Unrolled chains of 100 dependent/independent fma.rn.f32 across 8 accumulators.
   b) Pure FP16x2: Unrolled chains of 100 fma.rn.f16x2 (half2 SIMD packing) across 8 accumulators.
2. Launch across all 14 SMs with 112 blocks of 256 threads (28,672 active threads) to saturate
   warp schedulers.
3. Measure execution time with CUDA events and calculate physical TFLOP/s.
4. Verify whether Turing TU117 achieves the theoretical 2.0x arithmetic rate for packed FP16.
"""

import ctypes
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "fp16_vector_throughput_results.json"


def generate_kernels():
    # 1. FP32 Kernel
    ptx_fp32 = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        "",
        ".visible .entry BenchFP32(",
        "    .param .u64 d_out,",
        "    .param .u32 iters",
        ") {",
        "    .reg .u64 %out_ptr;",
        "    .reg .u32 %tidx, %k, %limit;",
        "    .reg .f32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;",
        "    .reg .f32 %mul, %add;",
        "    .reg .pred %p_loop, %p0;",
        "",
        "    ld.param.u64 %out_ptr, [d_out];",
        "    ld.param.u32 %limit, [iters];",
        "    mov.u32 %tidx, %tid.x;",
        "",
        "    mov.f32 %a0, 1.0; mov.f32 %a1, 1.1; mov.f32 %a2, 1.2; mov.f32 %a3, 1.3;",
        "    mov.f32 %a4, 1.4; mov.f32 %a5, 1.5; mov.f32 %a6, 1.6; mov.f32 %a7, 1.7;",
        "    mov.f32 %mul, 1.0001; mov.f32 %add, 0.0001;",
        "    mov.u32 %k, 0;",
        "",
        "    $loop_f32:",
    ]
    # 100 FMAs across 8 accumulators (each FMA = 2 FLOPs) -> 800 FLOPs per iteration
    for _ in range(12):
        ptx_fp32.append("    fma.rn.f32 %a0, %a0, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a1, %a1, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a2, %a2, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a3, %a3, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a4, %a4, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a5, %a5, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a6, %a6, %mul, %add;")
        ptx_fp32.append("    fma.rn.f32 %a7, %a7, %mul, %add;")

    ptx_fp32.extend([
        "    add.u32 %k, %k, 1;",
        "    setp.lt.u32 %p_loop, %k, %limit;",
        "    @%p_loop bra $loop_f32;",
        "",
        "    add.f32 %a0, %a0, %a1; add.f32 %a2, %a2, %a3;",
        "    add.f32 %a4, %a4, %a5; add.f32 %a6, %a6, %a7;",
        "    add.f32 %a0, %a0, %a2; add.f32 %a4, %a4, %a6;",
        "    add.f32 %a0, %a0, %a4;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.f32 [%out_ptr], %a0;",
        "    ret;",
        "}"
    ])

    # 2. FP16x2 Kernel
    ptx_fp16 = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        "",
        ".visible .entry BenchFP16(",
        "    .param .u64 d_out,",
        "    .param .u32 iters",
        ") {",
        "    .reg .u64 %out_ptr;",
        "    .reg .u32 %tidx, %k, %limit;",
        "    .reg .b32 %h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7;",
        "    .reg .b32 %mul, %add;",
        "    .reg .pred %p_loop, %p0;",
        "",
        "    ld.param.u64 %out_ptr, [d_out];",
        "    ld.param.u32 %limit, [iters];",
        "    mov.u32 %tidx, %tid.x;",
        "",
        "    mov.b32 %h0, 0x3c003c00; mov.b32 %h1, 0x3c003c00; mov.b32 %h2, 0x3c003c00; mov.b32 %h3, 0x3c003c00;",
        "    mov.b32 %h4, 0x3c003c00; mov.b32 %h5, 0x3c003c00; mov.b32 %h6, 0x3c003c00; mov.b32 %h7, 0x3c003c00;",
        "    mov.b32 %mul, 0x3c003c00; mov.b32 %add, 0x04000400;",
        "    mov.u32 %k, 0;",
        "",
        "    $loop_f16:",
    ]
    # 96 half2 FMAs across 8 accumulators (each half2 FMA = 4 FLOPs) -> 384 FLOPs per 8-instruction group
    for _ in range(12):
        ptx_fp16.append("    fma.rn.f16x2 %h0, %h0, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h1, %h1, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h2, %h2, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h3, %h3, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h4, %h4, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h5, %h5, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h6, %h6, %mul, %add;")
        ptx_fp16.append("    fma.rn.f16x2 %h7, %h7, %mul, %add;")

    ptx_fp16.extend([
        "    add.u32 %k, %k, 1;",
        "    setp.lt.u32 %p_loop, %k, %limit;",
        "    @%p_loop bra $loop_f16;",
        "",
        "    add.f16x2 %h0, %h0, %h1; add.f16x2 %h2, %h2, %h3;",
        "    add.f16x2 %h4, %h4, %h5; add.f16x2 %h6, %h6, %h7;",
        "    add.f16x2 %h0, %h0, %h2; add.f16x2 %h4, %h4, %h6;",
        "    add.f16x2 %h0, %h0, %h4;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u32 [%out_ptr], %h0;",
        "    ret;",
        "}"
    ])

    p_f32 = RAW_DIR / "bench_fp32.ptx"
    c_f32 = RAW_DIR / "bench_fp32.cubin"
    with open(p_f32, "w") as f: f.write("\n".join(ptx_fp32))
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_f32), "-o", str(c_f32)], check=True)

    p_f16 = RAW_DIR / "bench_fp16.ptx"
    c_f16 = RAW_DIR / "bench_fp16.cubin"
    with open(p_f16, "w") as f: f.write("\n".join(ptx_fp16))
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_f16), "-o", str(c_f16)], check=True)

    return c_f32, c_f16


def run_experiment():
    print("=" * 80)
    print(" CORNER G3: FP16 Half2 SIMD Packing vs FP32 Peak Throughput")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117 Turing) | Non-Tensor CUDA Cores")
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

    c_f32, c_f16 = generate_kernels()

    mod_f32 = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod_f32), str(c_f32.resolve()).encode("utf-8"))
    func_f32 = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_f32), mod_f32, b"BenchFP32")

    mod_f16 = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod_f16), str(c_f16.resolve()).encode("utf-8"))
    func_f16 = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_f16), mod_f16, b"BenchFP16")

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8)

    ev_start = ctypes.c_void_p()
    ev_stop = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(ev_start), 0)
    cuda.cuEventCreate(ctypes.byref(ev_stop), 0)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    BLOCK_SIZE = 256
    NUM_BLOCKS = 112 # 14 SMs * 8 blocks/SM = 28,672 threads
    TOTAL_THREADS = NUM_BLOCKS * BLOCK_SIZE
    LOOP_ITERS = 50000

    p_out = ctypes.c_uint64(d_out.value)
    p_iters = ctypes.c_uint32(LOOP_ITERS)

    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_iters), ctypes.c_void_p),
    )

    # --- 1. Benchmark Pure FP32 ---
    cuda.cuLaunchKernel(func_f32, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args, None)
    cuda.cuStreamSynchronize(stream)

    cuda.cuEventRecord(ev_start, stream)
    BENCH_REPEATS = 5
    for _ in range(BENCH_REPEATS):
        cuda.cuLaunchKernel(func_f32, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args, None)
    cuda.cuEventRecord(ev_stop, stream)
    cuda.cuEventSynchronize(ev_stop)

    ms_f32 = ctypes.c_float()
    cuda.cuEventElapsedTime(ctypes.byref(ms_f32), ev_start, ev_stop)
    t_f32_ms = ms_f32.value / BENCH_REPEATS

    # 12 groups * 8 FMAs * 2 FLOPs = 192 FLOPs per thread per iter
    flops_f32 = float(TOTAL_THREADS) * LOOP_ITERS * 192.0
    tflops_f32 = (flops_f32 / (t_f32_ms / 1000.0)) / 1e12

    # --- 2. Benchmark Pure FP16x2 ---
    cuda.cuLaunchKernel(func_f16, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args, None)
    cuda.cuStreamSynchronize(stream)

    cuda.cuEventRecord(ev_start, stream)
    for _ in range(BENCH_REPEATS):
        cuda.cuLaunchKernel(func_f16, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args, None)
    cuda.cuEventRecord(ev_stop, stream)
    cuda.cuEventSynchronize(ev_stop)

    ms_f16 = ctypes.c_float()
    cuda.cuEventElapsedTime(ctypes.byref(ms_f16), ev_start, ev_stop)
    t_f16_ms = ms_f16.value / BENCH_REPEATS

    # 12 groups * 8 half2 FMAs * 4 FLOPs = 384 FLOPs per thread per iter
    flops_f16 = float(TOTAL_THREADS) * LOOP_ITERS * 384.0
    tflops_f16 = (flops_f16 / (t_f16_ms / 1000.0)) / 1e12

    speedup = tflops_f16 / tflops_f32

    print(f"{'Precision':<16} | {'Execution Time (ms)':<22} | {'Total TeraFLOPs':<18} | {'Measured TFLOP/s'}")
    print("-" * 80)
    print(f"{'FP32 FMA':<16} | {t_f32_ms:<22.3f} | {flops_f32 / 1e12:<18.3f} | {tflops_f32:.3f} TFLOPS")
    print(f"{'FP16x2 FMA':<16} | {t_f16_ms:<22.3f} | {flops_f16 / 1e12:<18.3f} | {tflops_f16:.3f} TFLOPS")
    print("-" * 80)
    print(f"[*] FP16x2 vs FP32 Arithmetic Throughput Ratio: {speedup:.2f}x (Theoretical = 2.00x)\n")

    results = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "total_cuda_cores": 896,
        "active_threads": TOTAL_THREADS,
        "fp32": {
            "time_ms": round(t_f32_ms, 3),
            "sustained_tflops": round(tflops_f32, 3)
        },
        "fp16x2": {
            "time_ms": round(t_f16_ms, 3),
            "sustained_tflops": round(tflops_f16, 3)
        },
        "fp16_speedup_ratio": round(speedup, 2)
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFree_v2(d_out)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod_f32)
    cuda.cuModuleUnload(mod_f16)


if __name__ == "__main__":
    run_experiment()
