#!/usr/bin/env python3
"""
research/experiments/exp_warp_shuffle_latency.py
Forensic Characterization: Corner G1 - GPU Warp Shuffle (__shfl_sync) Latency vs Stride
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5), Turing Architecture

Methodology:
1. Construct dependent chains of 100 back-to-back warp shuffles (shfl.sync.bfly, shfl.sync.up, shfl.sync.down)
   across lane deltas (1, 2, 4, 8, 16).
2. Measure physical GPU clock cycles using %clock64() directly inside the SM.
3. Sink the final computed value to global memory to strictly prevent PTXAS dead-code elimination.
4. Compare against a dependent 100-iteration Shared Memory ping-pong (STS -> LDS).
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
OUT_FILE = RAW_DIR / "warp_shuffle_latency_results.json"


def generate_shuffle_ptx():
    ptx = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        "",
        "// Pure XOR Shuffle Kernel",
        ".visible .entry ShuffleXOR(",
        "    .param .u64 d_out_cycles,",
        "    .param .u64 d_out_val,",
        "    .param .u32 delta",
        ") {",
        "    .reg .u64 %ptr_out, %ptr_val;",
        "    .reg .u32 %tidx, %delta, %val, %mask, %w_size;",
        "    .reg .u64 %t0, %t1, %dt;",
        "    .reg .pred %p_dummy, %p0;",
        "",
        "    ld.param.u64 %ptr_out, [d_out_cycles];",
        "    ld.param.u64 %ptr_val, [d_out_val];",
        "    ld.param.u32 %delta, [delta];",
        "    mov.u32 %tidx, %tid.x;",
        "    mov.u32 %mask, 0xffffffff;",
        "    mov.u32 %w_size, 31;",
        "    mov.u32 %val, %tidx;",
        "",
        "    bar.sync 0;",
        "    mov.u64 %t0, %clock64;",
        ""
    ]
    for _ in range(100):
        ptx.append("    shfl.sync.bfly.b32 %val|%p_dummy, %val, %delta, %w_size, %mask;")
    ptx.extend([
        "    mov.u64 %t1, %clock64;",
        "    sub.u64 %dt, %t1, %t0;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u64 [%ptr_out], %dt;",
        "    @%p0 st.global.u32 [%ptr_val], %val;",
        "    ret;",
        "}",
        "",
        "// Pure DOWN Shuffle Kernel",
        ".visible .entry ShuffleDOWN(",
        "    .param .u64 d_out_cycles,",
        "    .param .u64 d_out_val,",
        "    .param .u32 delta",
        ") {",
        "    .reg .u64 %ptr_out, %ptr_val;",
        "    .reg .u32 %tidx, %delta, %val, %mask, %w_size;",
        "    .reg .u64 %t0, %t1, %dt;",
        "    .reg .pred %p_dummy, %p0;",
        "",
        "    ld.param.u64 %ptr_out, [d_out_cycles];",
        "    ld.param.u64 %ptr_val, [d_out_val];",
        "    ld.param.u32 %delta, [delta];",
        "    mov.u32 %tidx, %tid.x;",
        "    mov.u32 %mask, 0xffffffff;",
        "    mov.u32 %w_size, 31;",
        "    mov.u32 %val, %tidx;",
        "",
        "    bar.sync 0;",
        "    mov.u64 %t0, %clock64;",
        ""
    ])
    for _ in range(100):
        ptx.append("    shfl.sync.down.b32 %val|%p_dummy, %val, %delta, %w_size, %mask;")
    ptx.extend([
        "    mov.u64 %t1, %clock64;",
        "    sub.u64 %dt, %t1, %t0;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u64 [%ptr_out], %dt;",
        "    @%p0 st.global.u32 [%ptr_val], %val;",
        "    ret;",
        "}",
        "",
        "// Pure UP Shuffle Kernel",
        ".visible .entry ShuffleUP(",
        "    .param .u64 d_out_cycles,",
        "    .param .u64 d_out_val,",
        "    .param .u32 delta",
        ") {",
        "    .reg .u64 %ptr_out, %ptr_val;",
        "    .reg .u32 %tidx, %delta, %val, %mask, %w_size;",
        "    .reg .u64 %t0, %t1, %dt;",
        "    .reg .pred %p_dummy, %p0;",
        "",
        "    ld.param.u64 %ptr_out, [d_out_cycles];",
        "    ld.param.u64 %ptr_val, [d_out_val];",
        "    ld.param.u32 %delta, [delta];",
        "    mov.u32 %tidx, %tid.x;",
        "    mov.u32 %mask, 0xffffffff;",
        "    mov.u32 %w_size, 31;",
        "    mov.u32 %val, %tidx;",
        "",
        "    bar.sync 0;",
        "    mov.u64 %t0, %clock64;",
        ""
    ])
    for _ in range(100):
        ptx.append("    shfl.sync.up.b32 %val|%p_dummy, %val, %delta, 0, %mask;")
    ptx.extend([
        "    mov.u64 %t1, %clock64;",
        "    sub.u64 %dt, %t1, %t0;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u64 [%ptr_out], %dt;",
        "    @%p0 st.global.u32 [%ptr_val], %val;",
        "    ret;",
        "}",
        "",
        "// Shared Memory Ping-Pong Baseline Kernel",
        ".visible .entry SmemPingPongBench(",
        "    .param .u64 d_out_cycles,",
        "    .param .u64 d_out_val",
        ") {",
        "    .shared .align 4 .u32 s_data[32];",
        "    .reg .u64 %ptr_out, %ptr_val;",
        "    .reg .u32 %tidx, %val, %s_offset;",
        "    .reg .u64 %t0, %t1, %dt;",
        "    .reg .pred %p0;",
        "",
        "    ld.param.u64 %ptr_out, [d_out_cycles];",
        "    ld.param.u64 %ptr_val, [d_out_val];",
        "    mov.u32 %tidx, %tid.x;",
        "    shl.b32 %s_offset, %tidx, 2;",
        "    mov.u32 %val, %tidx;",
        "",
        "    bar.sync 0;",
        "    mov.u64 %t0, %clock64;",
        ""
    ])
    for _ in range(100):
        ptx.append("    st.shared.u32 [s_data], %val;")
        ptx.append("    ld.shared.u32 %val, [s_data];")
    ptx.extend([
        "    mov.u64 %t1, %clock64;",
        "    sub.u64 %dt, %t1, %t0;",
        "    setp.eq.u32 %p0, %tidx, 0;",
        "    @%p0 st.global.u64 [%ptr_out], %dt;",
        "    @%p0 st.global.u32 [%ptr_val], %val;",
        "    ret;",
        "}"
    ])

    p_file = RAW_DIR / "warp_shuffle.ptx"
    c_file = RAW_DIR / "warp_shuffle.cubin"
    with open(p_file, "w") as f:
        f.write("\n".join(ptx))
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER G1: GPU Warp Shuffle (__shfl_sync) Latency vs Stride")
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

    c_file = generate_shuffle_ptx()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))

    func_xor = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_xor), mod, b"ShuffleXOR")
    func_down = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_down), mod, b"ShuffleDOWN")
    func_up = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_up), mod, b"ShuffleUP")
    func_smem = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_smem), mod, b"SmemPingPongBench")

    d_out_cycles = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out_cycles), 8)
    d_out_val = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out_val), 4)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    DELTAS = [1, 2, 4, 8, 16]
    MODES = [("XOR (Butterfly)", func_xor), ("UP", func_up), ("DOWN", func_down)]
    results = []

    print(f"{'Mode':<18} | {'Delta':<8} | {'100 Shuffles (Cycles)':<24} | {'Cycles / Shuffle'}")
    print("-" * 75)

    for mode_name, func_ptr in MODES:
        for delta in DELTAS:
            # Args: d_out_cycles, d_out_val, delta
            p_out = ctypes.c_uint64(d_out_cycles.value)
            p_val = ctypes.c_uint64(d_out_val.value)
            p_delta = ctypes.c_uint32(delta)

            args = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(p_val), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(p_delta), ctypes.c_void_p),
            )

            # Warmup
            cuda.cuLaunchKernel(func_ptr, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
            cuda.cuStreamSynchronize(stream)

            cuda.cuLaunchKernel(func_ptr, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
            cuda.cuStreamSynchronize(stream)

            h_cycles = ctypes.c_uint64()
            cuda.cuMemcpyDtoH_v2(ctypes.byref(h_cycles), d_out_cycles, 8)

            total_c = h_cycles.value
            c_per_op = total_c / 100.0

            print(f"{mode_name:<18} | {delta:<8} | {total_c:<24} | {c_per_op:.2f}")

            results.append({
                "mode": mode_name,
                "delta": delta,
                "total_cycles_100": int(total_c),
                "cycles_per_shuffle": round(c_per_op, 2)
            })

    # Benchmark SMEM baseline
    p_out = ctypes.c_uint64(d_out_cycles.value)
    p_val = ctypes.c_uint64(d_out_val.value)
    args_smem = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_val), ctypes.c_void_p),
    )
    cuda.cuLaunchKernel(func_smem, 1, 1, 1, 32, 1, 1, 0, stream, args_smem, None)
    cuda.cuStreamSynchronize(stream)

    h_smem_cycles = ctypes.c_uint64()
    cuda.cuMemcpyDtoH_v2(ctypes.byref(h_smem_cycles), d_out_cycles, 8)
    smem_c_per_op = h_smem_cycles.value / 100.0

    print("-" * 75)
    print(f"Shared Memory (STS->LDS Ping-Pong) 100 iterations: {h_smem_cycles.value} cycles ({smem_c_per_op:.2f} cycles/op)")

    output = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "shuffle_results": results,
        "smem_pingpong_cycles_per_op": round(smem_c_per_op, 2),
        "smem_total_cycles": int(h_smem_cycles.value)
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[SUCCESS] Telemetry recorded in: {OUT_FILE}")
    cuda.cuMemFree_v2(d_out_cycles)
    cuda.cuMemFree_v2(d_out_val)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
