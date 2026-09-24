#!/usr/bin/env python3
"""
research/experiments/exp_atomics_topology.py
Phase 3 Experiment: GPU Atomic Contention Scaling & Topology Mapping
  - Sweeps target address contention: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 distinct addresses
  - Compares operations: atomicAdd (RED), atomicExch (ATOM), atomicCAS (CAS), atomicMin (MIN)
  - Identifies bottleneck transitions from serialized SM register writeback to in-situ L2 crossbar routing
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
OUT_FILE = RAW_DIR / "atomics_topology_sweep.json"


def generate_ptx_atomic(op_name: str, num_addrs: int) -> str:
    """Generates PTX kernel contending over `num_addrs` target memory locations."""
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        f".visible .entry AtomicContentionKernel(.param .u64 param_arr, .param .u32 param_iters) {{",
        "  .reg .b32 %r<15>;",
        "  .reg .b64 %rd<10>;",
        "  .reg .pred %p<2>;",
        "  ld.param.u64 %rd1, [param_arr];",
        "  ld.param.u32 %r1, [param_iters];",
        "  mov.u32 %r2, %tid.x;",
        # Address index = tid.x % num_addrs = tid.x & (num_addrs - 1)
        f"  and.b32 %r3, %r2, {num_addrs - 1};",
        "  mul.wide.u32 %rd2, %r3, 4;",
        "  add.s64 %rd3, %rd1, %rd2;",
        "  mov.u32 %r4, 0;",
        "LOOP:",
    ]

    for _ in range(4):
        if op_name == "add":
            lines.append("  red.global.add.u32 [%rd3], 1;")
        elif op_name == "exch":
            lines.append("  atom.global.exch.b32 %r5, [%rd3], %r4;")
        elif op_name == "cas":
            lines.append("  atom.global.cas.b32 %r5, [%rd3], 0, 1;")
        elif op_name == "min":
            lines.append("  red.global.min.u32 [%rd3], %r4;")

    lines.extend([
        "  add.u32 %r4, %r4, 1;",
        "  setp.lt.u32 %p1, %r4, %r1;",
        "  @%p1 bra LOOP;",
        "  ret;",
        "}"
    ])
    return "\n".join(lines)


def run_experiment():
    print("=" * 80)
    print(" Experiment EXP-GPU-305: Atomic Contention Scaling & Topology Mapping")
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
    THREADS = 1024  # 65,536 threads total
    MAX_ADDRS = 1024
    d_mem = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_mem), MAX_ADDRS * 4)

    p_mem = ctypes.c_uint64(d_mem.value)
    iters = ctypes.c_uint32(500)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(iters), ctypes.c_void_p)
    )

    ADDR_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    OPERATIONS = ["add", "min", "exch", "cas"]

    print(f"{'Target Addrs':<14} | {'atomicAdd (ms)':<16} | {'atomicMin (ms)':<16} | {'atomicExch (ms)':<17} | {'atomicCAS (ms)'}")
    print("-" * 80)

    matrix_results = {}

    for n_addrs in ADDR_COUNTS:
        row = {}
        for op in OPERATIONS:
            ptx_src = generate_ptx_atomic(op, n_addrs)
            p_file = RAW_DIR / f"atom_{op}_a{n_addrs}.ptx"
            c_file = RAW_DIR / f"atom_{op}_a{n_addrs}.cubin"
            with open(p_file, "w") as f:
                f.write(ptx_src)

            subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

            mod = ctypes.c_void_p()
            cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
            func = ctypes.c_void_p()
            cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"AtomicContentionKernel")

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
            row[op] = round(avg_ms, 3)
            cuda.cuModuleUnload(mod)

        print(f"{n_addrs:<14} | {row['add']:<16.3f} | {row['min']:<16.3f} | {row['exch']:<17.3f} | {row['cas']:.3f}")
        matrix_results[n_addrs] = row

    cuda.cuMemFree_v2(d_mem)

    with open(OUT_FILE, "w") as f:
        json.dump(matrix_results, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] Atomic contention topology saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
