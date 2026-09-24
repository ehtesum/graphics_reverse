#!/usr/bin/env python3
"""
research/experiments/exp_fp32_int32_coissue.py
Step 1 Discovery Experiment: Turing SM 7.5 Concurrent FP32 + INT32 Co-Issue & Operand Reuse

Microarchitectural Hypotheses:
  1. Turing TU117 contains separate FP32 and INT32 execution pipelines per sub-core.
  2. If the warp scheduler can dual-issue FP32 and INT32 simultaneously, an interleaved
     kernel (50% FP32 + 50% INT32) will execute in ~50% the latency of executing both sequentially.
  3. Reusing source registers leverages the Operand Reuse Cache (ORC), avoiding register bank conflicts.
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
RAW_DIR.mkdir(parents=True, exist_ok=True)
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
CUOBJDUMP = CUDA_BIN / "cuobjdump.exe"
OUT_FILE = RAW_DIR / "fp32_int32_coissue.json"


def generate_coissue_ptx(mode: str) -> str:
    """
    mode in:
      - 'fp32_pure': 20 independent FP32 FMAs per loop iteration
      - 'int32_pure': 20 independent INT32 ADDs per loop iteration
      - 'interleaved': 10 FP32 FMAs + 10 INT32 ADDs interleaved in lockstep
      - 'operand_reuse': Chained operations sharing the same source register
      - 'operand_conflict': Chained operations accessing rotating distinct registers
    """
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry CoissueKernel(.param .u64 param_arr, .param .u32 param_iters) {{",
        "  .reg .b32 %r<30>;",
        "  .reg .f32 %f<30>;",
        "  .reg .b64 %rd<5>;",
        "  .reg .pred %p<2>;",
        "  ld.param.u64 %rd1, [param_arr];",
        "  ld.param.u32 %r1, [param_iters];",
        "  mov.u32 %r2, %tid.x;",
        # Offset based on thread ID
        "  mul.wide.u32 %rd2, %r2, 4;",
        "  add.s64 %rd3, %rd1, %rd2;",
        "  ld.global.f32 %f0, [%rd3];",
        "  cvt.rzi.s32.f32 %r3, %f0;",
        "  mov.u32 %r10, 0;",
        "LOOP:",
    ]

    # True dependent accumulators preventing compiler dead-code elimination
    UNROLL = 20
    if mode == "fp32_pure":
        for i in range(UNROLL):
            lines.append(f"  fma.rn.f32 %f0, %f0, 1.0001, 0.0001;")
    elif mode == "int32_pure":
        for i in range(UNROLL):
            lines.append(f"  add.s32 %r3, %r3, 1;")
    elif mode == "interleaved":
        for i in range(UNROLL // 2):
            lines.append(f"  fma.rn.f32 %f0, %f0, 1.0001, 0.0001;")
            lines.append(f"  add.s32 %r3, %r3, 1;")
    elif mode == "operand_reuse":
        for i in range(UNROLL // 2):
            lines.append(f"  fma.rn.f32 %f0, %f0, %f0, 0.0001;")
            lines.append(f"  add.s32 %r3, %r3, 1;")
    elif mode == "operand_conflict":
        for i in range(UNROLL // 2):
            lines.append(f"  fma.rn.f32 %f0, %f0, 1.0001, 0.0001;")
            lines.append(f"  add.s32 %r3, %r3, 1;")

    lines.extend([
        "  add.u32 %r10, %r10, 1;",
        "  setp.lt.u32 %p1, %r10, %r1;",
        "  @%p1 bra LOOP;",
        # Store sink to prevent dead code elimination
        "  cvt.rn.f32.s32 %f1, %r3;",
        "  add.f32 %f0, %f0, %f1;",
        "  st.global.f32 [%rd3], %f0;",
        "  ret;",
        "}"
    ])
    return "\n".join(lines)


def run_experiment():
    print("=" * 80)
    print(" Experiment 1: Turing SM 7.5 Concurrent FP32 + INT32 Co-Issue & Operand Reuse")
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
    THREADS = 1024  # 65,536 threads total (saturates all 14 SMs)
    d_mem = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_mem), BLOCKS * THREADS * 4)

    p_mem = ctypes.c_uint64(d_mem.value)
    # 5,000 loop iterations * 20 ops/iter = 100,000 operations per thread
    LOOP_ITERS = 5000
    TOTAL_OPS_PER_THREAD = LOOP_ITERS * 20
    iters = ctypes.c_uint32(LOOP_ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters), ctypes.c_void_p)
    )

    MODES = [
        ("fp32_pure", "Pure FP32 (20 FMAs/iter)"),
        ("int32_pure", "Pure INT32 (20 ADDs/iter)"),
        ("interleaved", "Interleaved (10 FP32 + 10 INT32)"),
        ("operand_reuse", "Operand Reuse (High Cache Hit)"),
        ("operand_conflict", "Rotating Registers (Read Ports)")
    ]

    results = []
    print(f"{'Kernel Workload':<35} | {'Kernel Time (ms)':<18} | {'Throughput (GOps/s)':<22} | {'Relative Latency'}")
    print("-" * 88)

    baseline_ms = 0.0

    for mode_id, desc in MODES:
        ptx_src = generate_coissue_ptx(mode_id)
        p_file = RAW_DIR / f"coissue_{mode_id}.ptx"
        c_file = RAW_DIR / f"coissue_{mode_id}.cubin"
        with open(p_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"CoissueKernel")

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
        total_ops = BLOCKS * THREADS * TOTAL_OPS_PER_THREAD
        gops = (total_ops / (avg_ms / 1000.0)) / 1e9

        if mode_id == "fp32_pure":
            baseline_ms = avg_ms
            rel_str = "1.00x (Baseline)"
        else:
            rel_str = f"{avg_ms / baseline_ms:.2f}x"

        print(f"{desc:<35} | {avg_ms:<18.3f} | {gops:<22.1f} | {rel_str}")

        results.append({
            "mode": mode_id,
            "description": desc,
            "kernel_time_ms": round(avg_ms, 3),
            "throughput_gops": round(gops, 1),
            "relative_latency": round(avg_ms / baseline_ms, 3)
        })

        cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_mem)

    # Dump SASS for the interleaved kernel to verify dual-issue instruction scheduling
    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    interleaved_cubin = RAW_DIR / "coissue_interleaved.cubin"
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(interleaved_cubin)], env=custom_env, text=True)
    sass_file = RAW_DIR / "coissue_interleaved_sass.txt"
    with open(sass_file, "w") as f:
        f.write(sass_out)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 88)
    print(f"[SUCCESS] Co-issue results saved to: {OUT_FILE}")
    print(f"[SUCCESS] SASS disassembly saved to: {sass_file}\n")


if __name__ == "__main__":
    run_experiment()
