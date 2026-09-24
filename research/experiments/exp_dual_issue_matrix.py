#!/usr/bin/env python3
"""
research/experiments/exp_dual_issue_matrix.py
Corner 3 Forensic Experiment: Complete Instruction Dual-Issue Co-Execution Matrix
Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)

Microarchitectural Hypotheses:
  1. Turing sub-cores feature dual warp schedulers that can issue 2 independent instructions
     per warp per cycle if they target disjoint execution pipelines.
  2. FP32 math + INT32 math: Disjoint datapaths -> Near 100% co-issue concurrency.
  3. FP32 math + LDS (Shared Memory): Disjoint pipelines (ALU vs LSU) -> Arithmetic can hide LDS latency.
  4. FP32 math + SFU (Special Function): Disjoint pipelines (ALU vs SFU) -> Co-issue possible.
  5. SASS inspection reveals instruction grouping and stall count annotations.
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
OUT_FILE = RAW_DIR / "dual_issue_matrix_results.json"


def generate_pipeline_ptx(mode):
    """Generates specialized kernels combining FP32 with different execution pipelines."""
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry DualIssue_{mode}(",
        "    .param .u64 param_out,",
        "    .param .u32 param_iters",
        ")",
        "{",
        "    .reg .b32 %r<30>;",
        "    .reg .b64 %rd<10>;",
        "    .reg .f32 %f<30>;",
        "    .reg .pred %p<2>;",
        "    .shared .f32 smem[1024];",
        "",
        "    ld.param.u64 %rd1, [param_out];",
        "    ld.param.u32 %r1, [param_iters];",
        "    mov.u32 %r2, %tid.x;",
        "",
        "    // Init registers",
        "    mov.f32 %f0, 1.0001;",
        "    mov.f32 %f1, 1.0002;",
        "    mov.f32 %f2, 1.0003;",
        "    mov.f32 %f3, 1.0004;",
        "    mov.u32 %r10, 1;",
        "    mov.u32 %r11, 2;",
        "",
        "    // Init smem",
        "    and.b32 %r3, %r2, 1023;",
        "    mul.wide.u32 %rd4, %r3, 4;",
        "    mov.u64 %rd5, smem;",
        "    add.u64 %rd6, %rd5, %rd4;",
        "    st.shared.f32 [%rd6], %f0;",
        "    bar.sync 0;",
        "",
        "    mov.u32 %r20, 0;",
        "LOOP:"
    ]

    if mode == "fp32_pure":
        for i in range(20):
            lines.append("    fma.rn.f32 %f0, %f0, %f1, %f2;")
    elif mode == "fp32_int32":
        for i in range(10):
            lines.append("    fma.rn.f32 %f0, %f0, %f1, %f2;")
            lines.append("    add.u32 %r10, %r10, %r11;")
    elif mode == "fp32_lds":
        for i in range(10):
            lines.append("    fma.rn.f32 %f0, %f0, %f1, %f2;")
            lines.append(f"    ld.shared.f32 %f5, [smem+{i*4}];")
            lines.append("    add.f32 %f2, %f2, %f5;")
    elif mode == "fp32_sfu":
        for i in range(10):
            lines.append("    fma.rn.f32 %f0, %f0, %f1, %f2;")
            lines.append("    sin.approx.f32 %f6, %f1;")
            lines.append("    add.f32 %f2, %f2, %f6;")
    elif mode == "int32_lds":
        for i in range(10):
            lines.append("    add.u32 %r10, %r10, %r11;")
            lines.append(f"    ld.shared.u32 %r15, [smem+{i*4}];")
            lines.append("    add.u32 %r11, %r11, %r15;")

    lines.extend([
        "    add.u32 %r20, %r20, 1;",
        "    setp.lt.u32 %p1, %r20, %r1;",
        "    @%p1 bra LOOP;",
        "",
        "    mul.wide.u32 %rd2, %r2, 4;",
        "    add.u64 %rd3, %rd1, %rd2;",
        "    st.global.f32 [%rd3], %f0;",
        "    ret;",
        "}"
    ])
    return "\n".join(lines)


def run_experiment():
    print("=" * 96)
    print(" Corner 3: Complete Instruction Dual-Issue Co-Execution Matrix Forensic Audit")
    print(" Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)")
    print("=" * 96)

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

    MODES = [
        ("fp32_pure", "Pure FP32 FMA Baseline (20 FMAs/iter)"),
        ("fp32_int32", "Interleaved FP32 (10) + INT32 ADD (10)"),
        ("fp32_lds", "Interleaved FP32 (10) + Shared Load LDS (10)"),
        ("fp32_sfu", "Interleaved FP32 (10) + Transcendental SFU (10)"),
        ("int32_lds", "Interleaved INT32 (10) + Shared Load LDS (10)")
    ]

    BLOCKS = 64
    THREADS = 1024 # 65,536 threads total
    LOOP_ITERS = 10000

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), BLOCKS * THREADS * 4)

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_evt), 0)
    cuda.cuEventCreate(ctypes.byref(stop_evt), 0)

    p_out = ctypes.c_uint64(d_out.value)
    iters_arg = ctypes.c_uint32(LOOP_ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters_arg), ctypes.c_void_p)
    )

    print(f"{'Pipeline Combination':<44} | {'Kernel Time (ms)':<18} | {'Relative Latency':<18} | {'Dual-Issue Overlap'}")
    print("-" * 100)

    baseline_ms = 0.0
    results = []

    for mode_id, desc in MODES:
        ptx_src = generate_pipeline_ptx(mode_id)
        p_file = RAW_DIR / f"dual_{mode_id}.ptx"
        c_file = RAW_DIR / f"dual_{mode_id}.cubin"
        with open(p_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, f"DualIssue_{mode_id}".encode("utf-8"))

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

        if mode_id == "fp32_pure":
            baseline_ms = avg_ms
            rel_str = "1.00x (Baseline)"
            overlap_diag = "N/A (Single Pipeline)"
        else:
            rel_ratio = avg_ms / baseline_ms
            rel_str = f"{rel_ratio:.2f}x"
            if rel_ratio < 1.05:
                overlap_diag = "100% Concurrency (Zero Overhead)"
            elif rel_ratio < 1.30:
                overlap_diag = "High Concurrency (Partially Hidden)"
            else:
                overlap_diag = "Serial Execution / Pipeline Bottleneck"

        print(f"{desc:<44} | {avg_ms:<18.3f} | {rel_str:<18} | {overlap_diag}")

        results.append({
            "mode": mode_id,
            "description": desc,
            "kernel_time_ms": round(avg_ms, 3),
            "relative_ratio": round(avg_ms / baseline_ms, 3),
            "overlap_diagnosis": overlap_diag
        })

        cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_out)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 100)
    print(f"[SUCCESS] Dual-issue matrix results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
