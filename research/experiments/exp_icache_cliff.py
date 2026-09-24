#!/usr/bin/env python3
"""
research/experiments/exp_icache_cliff.py
Forensic Characterization: Corner G2 - GPU Instruction Cache (L1I) Footprint Cliff
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5 Turing)

Methodology:
1. Generate kernels with unrolled instruction blocks of varying sizes:
   128, 512, 1024, 2048, 4096, 8192 SASS instructions (approx 2 KB to 128 KB code footprint).
2. Execute a calibrated total of 10,000,000 instructions in a loop.
3. Measure execution clock cycles using %clock64() inside a single warp.
4. Detect the exact code-size knee where L1 instruction cache thrashing degrades throughput.
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
OUT_FILE = RAW_DIR / "icache_cliff_results.json"


def generate_icache_kernel(num_unrolled_instr, iters):
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        "",
        f".visible .entry ICacheBench_{num_unrolled_instr}(",
        "    .param .u64 d_out_cycles,",
        "    .param .u64 d_out_val",
        ") {",
        "    .reg .u64 %ptr_out, %ptr_val;",
        "    .reg .u32 %tidx, %r0, %r1, %r2, %r3, %k, %limit;",
        "    .reg .u64 %t0, %t1, %dt;",
        "    .reg .pred %p0, %p_loop;",
        "",
        "    ld.param.u64 %ptr_out, [d_out_cycles];",
        "    ld.param.u64 %ptr_val, [d_out_val];",
        "    mov.u32 %tidx, %tid.x;",
        "    mov.u32 %r0, %tidx;",
        "    mov.u32 %r1, 0x12345678;",
        "    mov.u32 %r2, 0x9abcdef0;",
        "    mov.u32 %r3, 0x0fedcba9;",
        "    mov.u32 %k, 0;",
        f"    mov.u32 %limit, {iters};",
        "",
        "    bar.sync 0;",
        "    mov.u64 %t0, %clock64;",
        "",
        "    $loop:",
    ]

    # Generate unrolled non-foldable rotating XOR instructions
    for i in range(num_unrolled_instr):
        r_dest = i % 4
        r_src = (i + 1) % 4
        ptx.append(f"    xor.b32 %r{r_dest}, %r{r_dest}, %r{r_src};")

    ptx.extend([
        "    add.u32 %k, %k, 1;",
        "    setp.lt.u32 %p_loop, %k, %limit;",
        "    @%p_loop bra $loop;",
        "",
        "    mov.u64 %t1, %clock64;",
        "    sub.u64 %dt, %t1, %t0;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u64 [%ptr_out], %dt;",
        "    add.u32 %r0, %r0, %r1;",
        "    add.u32 %r0, %r0, %r2;",
        "    add.u32 %r0, %r0, %r3;",
        "    @%p0 st.global.u32 [%ptr_val], %r0;",
        "    ret;",
        "}"
    ])

    p_file = RAW_DIR / f"icache_{num_unrolled_instr}.ptx"
    c_file = RAW_DIR / f"icache_{num_unrolled_instr}.cubin"
    with open(p_file, "w") as f:
        f.write("\n".join(ptx))
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER G2: GPU Instruction Cache (L1I) Footprint Cliff")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117 Turing) | SM 7.5")
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

    d_out_cycles = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out_cycles), 8)
    d_out_val = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out_val), 4)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    # Sweep unrolled instructions: 128 to 8192
    # Each SASS instruction is 16 bytes on Turing SM 7.5
    # 128 instrs = ~2 KB
    # 512 instrs = ~8 KB
    # 1024 instrs = ~16 KB
    # 2048 instrs = ~32 KB
    # 4096 instrs = ~64 KB
    # 8192 instrs = ~128 KB
    UNROLL_SIZES = [128, 512, 1024, 2048, 4096, 8192]
    TOTAL_OPS = 2000000 # 2 million operations

    results = []

    print(f"{'Unroll (Instrs)':<16} | {'SASS Footprint':<16} | {'Cycles/Instr':<16} | {'Total Cycles':<16} | {'Slowdown'}")
    print("-" * 85)

    base_cpi = 1.0

    for idx, unroll in enumerate(UNROLL_SIZES):
        iters = TOTAL_OPS // unroll
        c_file = generate_icache_kernel(unroll, iters)

        mod = ctypes.c_void_p()
        cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
        func = ctypes.c_void_p()
        func_name = f"ICacheBench_{unroll}".encode("utf-8")
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, func_name)

        p_out = ctypes.c_uint64(d_out_cycles.value)
        p_val = ctypes.c_uint64(d_out_val.value)
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_val), ctypes.c_void_p),
        )

        # Warmup
        cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
        cuda.cuStreamSynchronize(stream)

        # Timed execution
        cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
        cuda.cuStreamSynchronize(stream)

        h_cycles = ctypes.c_uint64()
        cuda.cuMemcpyDtoH_v2(ctypes.byref(h_cycles), d_out_cycles, 8)
        tot_c = h_cycles.value

        actual_ops = unroll * iters
        cpi = tot_c / actual_ops
        if idx == 0: base_cpi = cpi
        slowdown = cpi / base_cpi

        approx_sass_kb = (unroll * 16) / 1024.0

        print(f"{unroll:<16} | {approx_sass_kb:<13.1f} KB | {cpi:<16.3f} | {tot_c:<16} | {slowdown:.2f}x")

        results.append({
            "unroll_instructions": unroll,
            "approx_sass_kb": round(approx_sass_kb, 1),
            "cycles_per_instruction": round(cpi, 3),
            "total_cycles": int(tot_c),
            "slowdown": round(slowdown, 2)
        })

        cuda.cuModuleUnload(mod)

    output = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "architecture": "Turing SM 7.5",
        "icache_sweep": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 85)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFree_v2(d_out_cycles)
    cuda.cuMemFree_v2(d_out_val)
    cuda.cuStreamDestroy_v2(stream)


if __name__ == "__main__":
    run_experiment()
