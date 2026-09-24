#!/usr/bin/env python3
"""
research/experiments/exp_smem_deep.py
Phase 3 Experiment: Comprehensive Shared Memory Reverse Engineering
  - Sweeps strides from 1 to 128
  - Compares access widths: 8-bit (u8), 16-bit (u16), 32-bit (u32), 64-bit (u64)
  - Compares access modes: Read-only, Write-only, Read-Modify-Write (RMW)
  - Evaluates broadcast and bank mapping function: Bank = f(address, width, stride)
"""

import ctypes
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "smem_deep_sweep.json"


def generate_ptx_kernel(stride: int, width_bytes: int, mode: str) -> str:
    """
    Generates PTX for testing shared memory access.
    mode in ("read", "write", "rmw")
    width_bytes in (1, 2, 4, 8)
    """
    type_map = {1: ".u8", 2: ".u16", 4: ".f32", 8: ".u64"}
    type_str = type_map[width_bytes]
    reg_type = "%r" if width_bytes <= 4 else "%rd"
    load_op = f"ld.shared{type_str}"
    store_op = f"st.shared{type_str}"

    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry SmemSweepKernel(.param .u64 param_arr, .param .u32 param_iters) {{",
        f"  .shared .align {max(4, width_bytes)} .b8 smem[16384];",
        "  .reg .b32 %r<20>;",
        "  .reg .b64 %rd<20>;",
        "  .reg .f32 %f<10>;",
        "  .reg .pred %p<2>;",
        "  ld.param.u64 %rd1, [param_arr];",
        "  ld.param.u32 %r1, [param_iters];",
        "  mov.u32 %r2, %tid.x;",
        # Offset calculation: (tid.x * stride * width_bytes) & 16383
        f"  mul.lo.u32 %r3, %r2, {stride * width_bytes};",
        "  and.b32 %r4, %r3, 16383;",
        "  mul.wide.u32 %rd2, %r4, 1;",
        "  mov.u64 %rd3, smem;",
        "  add.s64 %rd4, %rd3, %rd2;",
        "  mov.u32 %r5, 0;",
        "LOOP:",
    ]

    reg = "%r6" if width_bytes <= 4 else "%rd6"
    acc = "%r7" if width_bytes <= 4 else "%rd7"

    if mode == "read":
        # Unrolled loads
        for _ in range(4):
            lines.append(f"  {load_op} {reg}, [%rd4];")
            if width_bytes <= 4:
                lines.append(f"  add.u32 {acc}, {acc}, {reg};")
            else:
                lines.append(f"  add.u64 {acc}, {acc}, {reg};")
    elif mode == "write":
        for _ in range(4):
            lines.append(f"  {store_op} [%rd4], {reg};")
    elif mode == "rmw":
        for _ in range(2):
            lines.append(f"  {load_op} {reg}, [%rd4];")
            if width_bytes <= 4:
                lines.append(f"  add.u32 {reg}, {reg}, 1;")
            else:
                lines.append(f"  add.u64 {reg}, {reg}, 1;")
            lines.append(f"  {store_op} [%rd4], {reg};")

    lines.extend([
        "  add.u32 %r5, %r5, 1;",
        "  setp.lt.u32 %p1, %r5, %r1;",
        "  @%p1 bra LOOP;",
        "  bar.sync 0;",
        "  mul.wide.u32 %rd5, %r2, 4;",
        "  add.s64 %rd6, %rd1, %rd5;",
        "  st.global.u32 [%rd6], %r7;",
        "  ret;",
        "}"
    ])
    return "\n".join(lines)


def run_experiment():
    print("=" * 80)
    print(" Experiment EXP-GPU-219: Shared Memory Reverse Engineering (Strides 1-128 & Widths)")
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
    iters = ctypes.c_uint32(2000)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters), ctypes.c_void_p)
    )

    # 1. Broad Stride Sweep (32-bit RMW)
    STRIDES = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 33, 48, 64, 65, 96, 128]
    print("\n--- Part A: 32-Bit (4-Byte) Stride Sweep (Strides 1 to 128) ---")
    print(f"{'Stride':<8} | {'gcd(S,32)':<10} | {'Expected Conflict':<20} | {'Kernel Time (ms)':<18} | {'Slowdown'}")
    print("-" * 75)

    stride_results = []
    base_time = 0.0

    for s in STRIDES:
        ptx_src = generate_ptx_kernel(s, 4, "rmw")
        p_file = RAW_DIR / f"smem_s{s}.ptx"
        c_file = RAW_DIR / f"smem_s{s}.cubin"
        with open(p_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"SmemSweepKernel")

        # Warmup
        cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()

        times = []
        for _ in range(5):
            cuda.cuEventRecord(start_evt, None)
            cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
            cuda.cuEventRecord(stop_evt, None)
            cuda.cuEventSynchronize(stop_evt)
            el = ctypes.c_float()
            cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
            times.append(el.value)

        avg_ms = sum(times) / len(times)
        gcd_val = math.gcd(s, 32)
        exp_conflict = f"{gcd_val}-way conflict" if gcd_val > 1 else "1-way (Conflict-Free)"

        if s == 1:
            base_time = avg_ms
            slowdown = "1.00x"
        else:
            slowdown = f"{avg_ms / base_time:.2f}x"

        print(f"{s:<8} | {gcd_val:<10} | {exp_conflict:<20} | {avg_ms:<18.3f} | {slowdown}")
        stride_results.append({
            "stride": s,
            "gcd": gcd_val,
            "time_ms": round(avg_ms, 3),
            "slowdown": round(avg_ms / base_time, 2)
        })
        cuda.cuModuleUnload(mod)

    # 2. Access Width Sweep (Stride 1 vs Stride 2 at 8, 16, 32, 64-bit)
    print("\n--- Part B: Access Width & Bank Alignment Sweep ---")
    print(f"{'Width (Bits)':<14} | {'Stride':<8} | {'Access Mode':<12} | {'Kernel Time (ms)':<18} | {'Relative to 32b S1'}")
    print("-" * 75)

    width_results = []
    for w in [1, 2, 4, 8]:
        for s in [1, 2]:
            for mode in ["read", "write", "rmw"]:
                ptx_src = generate_ptx_kernel(s, w, mode)
                p_file = RAW_DIR / f"smem_w{w}_s{s}_{mode}.ptx"
                c_file = RAW_DIR / f"smem_w{w}_s{s}_{mode}.cubin"
                with open(p_file, "w") as f:
                    f.write(ptx_src)

                subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

                mod = ctypes.c_void_p()
                cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
                func = ctypes.c_void_p()
                cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"SmemSweepKernel")

                cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
                cuda.cuCtxSynchronize()

                times = []
                for _ in range(5):
                    cuda.cuEventRecord(start_evt, None)
                    cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None)
                    cuda.cuEventRecord(stop_evt, None)
                    cuda.cuEventSynchronize(stop_evt)
                    el = ctypes.c_float()
                    cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt)
                    times.append(el.value)

                avg_ms = sum(times) / len(times)
                print(f"{w*8:<14} | {s:<8} | {mode:<12} | {avg_ms:<18.3f} | {avg_ms / base_time:.2f}x")
                width_results.append({
                    "width_bits": w * 8,
                    "stride": s,
                    "mode": mode,
                    "time_ms": round(avg_ms, 3),
                    "ratio": round(avg_ms / base_time, 2)
                })
                cuda.cuModuleUnload(mod)

    cuda.cuMemFree_v2(d_mem)

    payload = {
        "stride_sweep": stride_results,
        "width_sweep": width_results
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] Shared memory reverse engineering results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
