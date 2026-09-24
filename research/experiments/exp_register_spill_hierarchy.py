#!/usr/bin/env python3
"""
research/experiments/exp_register_spill_hierarchy.py
Corner 2 Forensic Experiment: GPU Register Spilling & Local Memory Hierarchy
Target: NVIDIA GeForce GTX 1650 4 GB (Turing SM 7.5, TU117)

Microarchitectural Hypotheses:
  1. Turing SM 7.5 features 65,536 32-bit registers per SM (total 917,504 registers across 14 SMs).
  2. Restricting max registers per thread via compiler flags forces the register allocator
     to spill live values into thread-private Local Memory (.local).
  3. Local memory accesses are cached in the unified L1/TEX cache (~44 cycles latency).
  4. SASS disassembly will reveal replacement of register-to-register moves with LDL/STL instructions.
  5. The execution slowdown directly reflects the memory bandwidth pressure of L1 local spills.
"""

import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
CUOBJDUMP = CUDA_BIN / "cuobjdump.exe"
OUT_FILE = RAW_DIR / "register_spill_results.json"


def generate_spill_ptx():
    """Generates a PTX kernel that maintains 32 independent live accumulators."""
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry RegisterStressKernel(",
        "    .param .u64 param_out,",
        "    .param .u32 param_iters",
        ")",
        "{",
        "    .reg .b32 %r<50>;",
        "    .reg .b64 %rd<5>;",
        "    .reg .pred %p<2>;",
        "",
        "    ld.param.u64 %rd1, [param_out];",
        "    ld.param.u32 %r1, [param_iters];",
        "    mov.u32 %r2, %tid.x;",
        "",
        "    // Initialize 32 distinct live values"
    ]
    for i in range(32):
        lines.append(f"    mov.u32 %r{10 + i}, %r2;")

    lines.extend([
        "    mov.u32 %r45, 0;",
        "LOOP:",
        "    // Perform dependent updates across all 32 accumulators to prevent coalescing"
    ])
    for i in range(32):
        next_r = 10 + ((i + 1) % 32)
        lines.append(f"    mad.lo.u32 %r{10 + i}, %r{10 + i}, 3, %r{next_r};")

    lines.extend([
        "    add.u32 %r45, %r45, 1;",
        "    setp.lt.u32 %p1, %r45, %r1;",
        "    @%p1 bra LOOP;",
        "",
        "    // Reduce 32 accumulators to a single sum and write out",
        "    mov.u32 %r46, 0;"
    ])
    for i in range(32):
        lines.append(f"    add.u32 %r46, %r46, %r{10 + i};")

    lines.extend([
        "    mul.wide.u32 %rd2, %r2, 4;",
        "    add.u64 %rd3, %rd1, %rd2;",
        "    st.global.u32 [%rd3], %r46;",
        "    ret;",
        "}"
    ])
    return "\n".join(lines)


def compile_and_audit(ptx_file, max_regs):
    """Compiles PTX with a specific maxrregcount and parses ptxas telemetry."""
    cubin_file = RAW_DIR / f"spill_r{max_regs}.cubin"
    cmd = [
        str(PTXAS),
        "-arch=sm_75",
        f"-maxrregcount={max_regs}",
        "-v",
        str(ptx_file),
        "-o",
        str(cubin_file)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stderr = res.stderr

    # Parse ptxas info: e.g. "Used 24 registers, 32 bytes spill stores, 32 bytes spill loads"
    used_regs = max_regs
    spill_stores = 0
    spill_loads = 0

    m_used = re.search(r"Used (\d+) registers", stderr)
    if m_used:
        used_regs = int(m_used.group(1))

    m_spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", stderr)
    if m_spill:
        spill_stores = int(m_spill.group(1))
        spill_loads = int(m_spill.group(2))

    # Inspect SASS for exact LDL and STL instructions
    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
    ldl_count = len(re.findall(r"LDL\.LU", sass_out))
    stl_count = len(re.findall(r"STL\.LU", sass_out))

    return cubin_file, used_regs, spill_stores, spill_loads, ldl_count, stl_count


def run_experiment():
    print("=" * 96)
    print(" Corner 2: GPU Register Spilling & Local Memory Hierarchy Forensic Audit")
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

    ptx_src = generate_spill_ptx()
    ptx_file = RAW_DIR / "spill_test.ptx"
    with open(ptx_file, "w") as f:
        f.write(ptx_src)

    REG_LIMITS = [64, 32, 24, 20, 16, 12]
    LOOP_ITERS = 20000

    BLOCKS = 64
    THREADS = 1024 # 65,536 threads total
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

    print(f"{'Target Regs':<12} | {'Used Regs':<10} | {'Spill Bytes (Store/Load)':<26} | {'SASS LDL/STL':<14} | {'Kernel Time (ms)':<16} | {'Slowdown'}")
    print("-" * 100)

    baseline_ms = 0.0
    results = []

    for r_lim in REG_LIMITS:
        cubin_file, used_r, s_store, s_load, ldl, stl = compile_and_audit(ptx_file, r_lim)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(cubin_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"RegisterStressKernel")

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

        if r_lim == 64:
            baseline_ms = avg_ms
            rel_str = "1.00x (Baseline)"
        else:
            rel_str = f"{avg_ms / baseline_ms:.2f}x"

        spill_str = f"{s_store} B / {s_load} B"
        sass_str = f"{ldl} LDL / {stl} STL"
        print(f"{r_lim:<12} | {used_r:<10} | {spill_str:<26} | {sass_str:<14} | {avg_ms:<16.3f} | {rel_str}")

        results.append({
            "maxrregcount": r_lim,
            "used_registers": used_r,
            "spill_stores_bytes": s_store,
            "spill_loads_bytes": s_load,
            "sass_ldl_count": ldl,
            "sass_stl_count": stl,
            "kernel_time_ms": round(avg_ms, 3),
            "slowdown": round(avg_ms / baseline_ms, 3)
        })

        cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_out)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 100)
    print(f"[SUCCESS] Register spill telemetry saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
