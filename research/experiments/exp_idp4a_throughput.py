#!/usr/bin/env python3
"""
research/experiments/exp_idp4a_throughput.py
Step 2 Discovery Experiment: TU117 Non-Tensor-Core INT8 IDP4A Peak Throughput

Microarchitectural Hypotheses:
  1. TU117 lacks Tensor Cores, but features native hardware IDP4A (4-element 8-bit dot product).
  2. Each IDP4A executes 4 INT8 multiplies + 4 INT32 adds = 8 integer arithmetic ops per instruction.
  3. If IDP4A runs at 1 instruction/cycle per datapath, peak INT8 throughput will reach ~10-12 TOP/s,
     exceeding Rapid Packed Math (5.39 TFLOPS FP16) by >2x and FP32 (2.44 TFLOPS) by >4.5x.
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
CUOBJDUMP = CUDA_BIN / "cuobjdump.exe"
OUT_FILE = RAW_DIR / "idp4a_throughput.json"


def generate_idp4a_ptx(precision: str) -> str:
    """
    precision in ("fp32", "fp16", "idp4a")
    """
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry ArithmeticEngineKernel(.param .u64 param_arr, .param .u32 param_iters) {{",
        "  .reg .b32 %r<30>;",
        "  .reg .f32 %f<30>;",
        "  .reg .b64 %rd<10>;",
        "  .reg .pred %p<2>;",
        "  ld.param.u64 %rd1, [param_arr];",
        "  ld.param.u32 %r1, [param_iters];",
        "  mov.u32 %r2, %tid.x;",
        "  mul.wide.u32 %rd2, %r2, 4;",
        "  add.s64 %rd3, %rd1, %rd2;",
        "  ld.global.u32 %r3, [%rd3];",
        "  mov.f32 %f0, 1.0;",
        "  mov.b32 %r4, 0x01010101;",
        "  mov.b32 %r5, 0x3C003C00;",
        "  mov.b32 %r6, 0x38003800;",
        "  mov.f32 %f1, 1.0001;",
        "  mov.f32 %f2, 0.0001;",
        "  mov.u32 %r10, 0;",
        "LOOP:",
    ]

    UNROLL = 20
    if precision == "fp32":
        # 1 FMA = 2 ops
        for _ in range(UNROLL):
            lines.append("  fma.rn.f32 %f0, %f0, %f1, %f2;")
    elif precision == "fp16":
        # 1 HFMA2 = 4 ops (2 FP16 FMAs packed)
        for _ in range(UNROLL):
            lines.append("  fma.rn.f16x2 %r3, %r3, %r5, %r6;")
    elif precision == "idp4a":
        # 1 IDP4A = 8 ops (4 INT8 muls + 4 INT32 adds)
        for _ in range(UNROLL):
            lines.append("  dp4a.s32.s32 %r3, %r3, %r4, %r3;")

    lines.extend([
        "  add.u32 %r10, %r10, 1;",
        "  setp.lt.u32 %p1, %r10, %r1;",
        "  @%p1 bra LOOP;",
    ])

    if precision == "fp32":
        lines.append("  cvt.rzi.u32.f32 %r3, %f0;")

    lines.extend([
        "  st.global.u32 [%rd3], %r3;",
        "  ret;",
        "}"
    ])
    return "\n".join(lines)


def run_experiment():
    print("=" * 80)
    print(" Experiment 2: TU117 Non-Tensor-Core INT8 IDP4A Peak Throughput")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5)")
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

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    BLOCKS = 64
    THREADS = 1024
    d_mem = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_mem), BLOCKS * THREADS * 4)

    p_mem = ctypes.c_uint64(d_mem.value)
    LOOP_ITERS = 10000
    iters = ctypes.c_uint32(LOOP_ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters), ctypes.c_void_p)
    )

    ENGINES = [
        ("fp32", "IEEE FP32 FMA", 2),
        ("fp16", "Rapid Packed Math FP16 (HFMA2)", 4),
        ("idp4a", "INT8 Dot Product (IDP4A.S32)", 8)
    ]

    results = []
    print(f"{'Arithmetic Engine':<32} | {'Kernel Time (ms)':<18} | {'Effective TFLOPS / TOP/s':<26} | {'Speedup over FP32'}")
    print("-" * 92)

    base_time = 0.0

    for prec_id, desc, ops_per_inst in ENGINES:
        ptx_src = generate_idp4a_ptx(prec_id)
        p_file = RAW_DIR / f"engine_{prec_id}.ptx"
        c_file = RAW_DIR / f"engine_{prec_id}.cubin"
        with open(p_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"ArithmeticEngineKernel")

        # Warmup
        cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()

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
        total_arithmetic_ops = BLOCKS * THREADS * LOOP_ITERS * 20 * ops_per_inst
        tera_ops = (total_arithmetic_ops / (avg_ms / 1000.0)) / 1e12

        if prec_id == "fp32":
            base_time = avg_ms
            speedup = "1.00x (Baseline)"
        else:
            speedup = f"{base_time / avg_ms:.2f}x"

        print(f"{desc:<32} | {avg_ms:<18.3f} | {tera_ops:<26.3f} | {speedup}")

        results.append({
            "engine": prec_id,
            "description": desc,
            "ops_per_instruction": ops_per_inst,
            "kernel_time_ms": round(avg_ms, 3),
            "tera_ops_sec": round(tera_ops, 3),
            "speedup_factor": round(base_time / avg_ms, 2)
        })

        cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_mem)

    # Dump SASS for IDP4A
    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    idp4a_cubin = RAW_DIR / "engine_idp4a.cubin"
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(idp4a_cubin)], env=custom_env, text=True)
    sass_file = RAW_DIR / "engine_idp4a_sass.txt"
    with open(sass_file, "w") as f:
        f.write(sass_out)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 92)
    print(f"[SUCCESS] IDP4A throughput results saved to: {OUT_FILE}")
    print(f"[SUCCESS] SASS disassembly saved to: {sass_file}\n")


if __name__ == "__main__":
    run_experiment()
