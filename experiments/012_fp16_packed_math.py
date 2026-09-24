#!/usr/bin/env python3
"""
experiments/012_fp16_packed_math.py
Step 3: Turing FP16 Rapid Packed Math (HFMA2 / HADD2) vs FP32 Single Precision

Measures the compute throughput of the NVIDIA GeForce GTX 1650 (Turing sm_75 TU117)
comparing:
  1. FP32 Single Precision FMA (fma.rn.f32 -> SASS FFMA)
  2. FP16x2 Half Precision Rapid Packed FMA (fma.rn.f16x2 -> SASS HFMA2)

Performs 2,000 unrolled FMA operations per thread over 262,144 threads (1024 blocks x 256 threads),
measuring raw computational throughput and quantifying the 2x dual-issue packing factor.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "012_fp16_packed_math"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_fp32_ptx(unroll_count=2000):
    lines = []
    lines.append(".version 7.5")
    lines.append(".target sm_75")
    lines.append(".address_size 64")
    lines.append(".visible .entry BenchmarkFP32(")
    lines.append("    .param .u64 param_arr,")
    lines.append("    .param .u32 param_iters")
    lines.append(")")
    lines.append("{")
    lines.append("    .reg .b32 %r<10>;")
    lines.append("    .reg .f32 %f<20>;")
    lines.append("    .reg .b64 %rd<10>;")
    lines.append("    .reg .pred %p<4>;")
    lines.append("")
    lines.append("    ld.param.u64 %rd1, [param_arr];")
    lines.append("    ld.param.u32 %r1, [param_iters];")
    lines.append("    mov.u32 %r2, %ctaid.x;")
    lines.append("    mov.u32 %r3, %ntid.x;")
    lines.append("    mov.u32 %r4, %tid.x;")
    lines.append("    mad.lo.s32 %r5, %r2, %r3, %r4;")
    lines.append("")
    lines.append("    mul.wide.s32 %rd2, %r5, 4;")
    lines.append("    add.s64 %rd3, %rd1, %rd2;")
    lines.append("    ld.global.f32 %f1, [%rd3];")
    lines.append("    mov.f32 %f2, 0f3F800000; // 1.0f")
    lines.append("    mov.f32 %f3, 0f3F000000; // 0.5f")
    lines.append("    mov.f32 %f4, %f1;")
    lines.append("    mov.f32 %f5, %f1;")
    lines.append("")
    lines.append("    mov.u32 %r6, 0;")
    lines.append("LOOP:")
    lines.append("    setp.ge.u32 %p1, %r6, %r1;")
    lines.append("    @%p1 bra DONE;")
    lines.append("")
    # Unroll FMAs
    for i in range(unroll_count):
        # fma.rn.f32 d, a, b, c -> d = a * b + c
        lines.append(f"    fma.rn.f32 %f4, %f4, %f2, %f3;")
        lines.append(f"    fma.rn.f32 %f5, %f5, %f2, %f3;")
    lines.append("")
    lines.append("    add.u32 %r6, %r6, 1;")
    lines.append("    bra LOOP;")
    lines.append("DONE:")
    lines.append("    add.f32 %f1, %f4, %f5;")
    lines.append("    st.global.f32 [%rd3], %f1;")
    lines.append("    ret;")
    lines.append("}")
    return "\n".join(lines)


def generate_fp16_ptx(unroll_count=2000):
    lines = []
    lines.append(".version 7.5")
    lines.append(".target sm_75")
    lines.append(".address_size 64")
    lines.append(".visible .entry BenchmarkFP16(")
    lines.append("    .param .u64 param_arr,")
    lines.append("    .param .u32 param_iters")
    lines.append(")")
    lines.append("{")
    lines.append("    .reg .b32 %r<10>;")
    lines.append("    .reg .b32 %h<20>;")
    lines.append("    .reg .b64 %rd<10>;")
    lines.append("    .reg .pred %p<4>;")
    lines.append("")
    lines.append("    ld.param.u64 %rd1, [param_arr];")
    lines.append("    ld.param.u32 %r1, [param_iters];")
    lines.append("    mov.u32 %r2, %ctaid.x;")
    lines.append("    mov.u32 %r3, %ntid.x;")
    lines.append("    mov.u32 %r4, %tid.x;")
    lines.append("    mad.lo.s32 %r5, %r2, %r3, %r4;")
    lines.append("")
    lines.append("    mul.wide.s32 %rd2, %r5, 4;")
    lines.append("    add.s64 %rd3, %rd1, %rd2;")
    lines.append("    ld.global.u32 %h1, [%rd3];")
    lines.append("    // 0x3C00 is 1.0 in FP16; packed two 1.0f16 is 0x3C003C00")
    lines.append("    mov.b32 %h2, 0x3C003C00;")
    lines.append("    // 0x3800 is 0.5 in FP16; packed two 0.5f16 is 0x38003800")
    lines.append("    mov.b32 %h3, 0x38003800;")
    lines.append("    mov.b32 %h4, %h1;")
    lines.append("    mov.b32 %h5, %h1;")
    lines.append("")
    lines.append("    mov.u32 %r6, 0;")
    lines.append("LOOP:")
    lines.append("    setp.ge.u32 %p1, %r6, %r1;")
    lines.append("    @%p1 bra DONE;")
    lines.append("")
    # Unroll packed FP16x2 FMAs
    for i in range(unroll_count):
        # fma.rn.f16x2 d, a, b, c -> 2 half ops per instruction
        lines.append(f"    fma.rn.f16x2 %h4, %h4, %h2, %h3;")
        lines.append(f"    fma.rn.f16x2 %h5, %h5, %h2, %h3;")
    lines.append("")
    lines.append("    add.u32 %r6, %r6, 1;")
    lines.append("    bra LOOP;")
    lines.append("DONE:")
    lines.append("    add.f16x2 %h1, %h4, %h5;")
    lines.append("    st.global.u32 [%rd3], %h1;")
    lines.append("    ret;")
    lines.append("}")
    return "\n".join(lines)


def build_and_disassemble(name, ptx_code):
    ptx_file = RESULTS_DIR / f"{name}.ptx"
    cubin_file = RESULTS_DIR / f"{name}.cubin"
    sass_file = RESULTS_DIR / f"{name}_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
    with open(sass_file, "w") as f:
        f.write(sass_out)

    return cubin_file, sass_file


def run_benchmark():
    UNROLL = 500
    ITERS = 100
    TOTAL_FMAS_PER_THREAD = UNROLL * 2 * ITERS  # 100,000 FMA instructions per thread

    print("=" * 75)
    print(" Step 3: Turing FP16 Rapid Packed Math vs FP32 (TU117 / GTX 1650)")
    print(f" Arithmetic Stress: {TOTAL_FMAS_PER_THREAD:,} FMA instructions per thread")
    print(" 262,144 Threads (1024 Blocks x 256 Threads)")
    print("=" * 75)

    print("[*] Generating and compiling PTX -> SASS...")
    ptx_fp32 = generate_fp32_ptx(UNROLL)
    cubin_fp32, sass_fp32 = build_and_disassemble("fp32_fma", ptx_fp32)

    ptx_fp16 = generate_fp16_ptx(UNROLL)
    cubin_fp16, sass_fp16 = build_and_disassemble("fp16_hfma2", ptx_fp16)

    # Check SASS instructions
    with open(sass_fp32, "r") as f:
        sass32_text = f.read()
    with open(sass_fp16, "r") as f:
        sass16_text = f.read()

    ffma_count = sass32_text.count("FFMA")
    hfma2_count = sass16_text.count("HFMA2")
    print(f"[*] SASS Disassembly verification:")
    print(f"    - FP32 Kernel uses: {ffma_count} FFMA instructions")
    print(f"    - FP16 Kernel uses: {hfma2_count} HFMA2 (Packed 2-way) instructions")

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

    BLOCKS = 1024
    THREADS = 256
    TOTAL_THREADS = BLOCKS * THREADS  # 262,144
    num_bytes = TOTAL_THREADS * 4

    d_mem = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), "cuMemAlloc")

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    p_mem = ctypes.c_uint64(d_mem.value)
    p_iters = ctypes.c_uint32(ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_iters), ctypes.c_void_p)
    )

    configs = [
        ("FP32 (fma.rn.f32 -> FFMA)", cubin_fp32, "BenchmarkFP32", 1),
        ("FP16x2 (fma.rn.f16x2 -> HFMA2)", cubin_fp16, "BenchmarkFP16", 2)
    ]

    results = []
    print("-" * 75)
    print(f"{'Precision Mode':<32} | {'Time (ms)':<10} | {'Throughput (TFLOPS)':<20} | {'Scaling':<10}")
    print("-" * 75)

    base_time = 0.0

    for label, cubin, func_name, ops_per_inst in configs:
        mod = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(mod), str(cubin.resolve()).encode("utf-8")), "ModuleLoad")
        func = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, func_name.encode("utf-8")), "GetFunction")

        # Warmup
        check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Warmup")
        check(cuda.cuCtxSynchronize(), "Sync")

        times = []
        for _ in range(5):
            check(cuda.cuEventRecord(start_evt, None), "Start")
            check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Launch")
            check(cuda.cuEventRecord(stop_evt, None), "Stop")
            check(cuda.cuEventSynchronize(stop_evt), "Sync")
            el = ctypes.c_float()
            check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
            times.append(el.value)

        avg_ms = sum(times) / len(times)

        # 1 FMA = 2 floating point operations (multiply + add)
        # ops_per_inst: 1 for FP32, 2 for FP16x2
        total_flops = TOTAL_THREADS * TOTAL_FMAS_PER_THREAD * 2 * ops_per_inst
        tflops = (total_flops / (avg_ms / 1000.0)) / 1e12

        if ops_per_inst == 1:
            base_time = avg_ms
            scaling = "1.00x (Base)"
        else:
            speedup = base_time / avg_ms
            scaling = f"{speedup:.2f}x"

        print(f"{label:<32} | {avg_ms:<10.3f} | {tflops:<20.3f} | {scaling:<10}")

        results.append({
            "mode": label,
            "kernel_time_ms": round(avg_ms, 4),
            "tflops": round(tflops, 3),
            "speedup": round(base_time / avg_ms, 2) if base_time > 0 else 1.0,
            "ops_per_instruction": ops_per_inst
        })

    print("=" * 75)
    out_file = RESULTS_DIR / "fp16_packed_math.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved results to: {out_file}\n")


if __name__ == "__main__":
    run_benchmark()
