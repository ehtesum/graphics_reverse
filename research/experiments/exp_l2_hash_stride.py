#!/usr/bin/env python3
"""
research/experiments/exp_l2_hash_stride.py
Step 3 Discovery Experiment: GPU L2 Cache Slice Hashing & Set Aliasing Reverse Engineering
Target: NVIDIA GeForce GTX 1650 4 GB (1 MB L2 Cache, 128-bit GDDR6)
"""

import ctypes
import json
import os
import subprocess
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "l2_hash_stride_results.json"


def generate_pointer_chase_ptx():
    return """
.version 7.5
.target sm_75
.address_size 64

.visible .entry PointerChaseKernel_ca(
    .param .u64 param_arr,
    .param .u32 param_iters,
    .param .u64 param_cycles
)
{
    .reg .u32 %r1, %r2;
    .reg .u64 %rd1, %rd2, %rd4, %rd5, %rd6, %rd10;
    .reg .pred %p1;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u32 %r1, [param_iters];
    ld.param.u64 %rd10, [param_cycles];

    mov.u64 %rd2, %rd1;
    mov.u32 %r2, 0;

    bar.sync 0;
    mov.u64 %rd4, %clock64;

LOOP_CA:
    ld.global.ca.u64 %rd2, [%rd2];
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra LOOP_CA;

    mov.u64 %rd5, %clock64;
    sub.u64 %rd6, %rd5, %rd4;
    st.global.u64 [%rd10], %rd6;
    st.global.u64 [%rd1], %rd2;
    ret;
}

.visible .entry PointerChaseKernel_cg(
    .param .u64 param_arr,
    .param .u32 param_iters,
    .param .u64 param_cycles
)
{
    .reg .u32 %r1, %r2;
    .reg .u64 %rd1, %rd2, %rd4, %rd5, %rd6, %rd10;
    .reg .pred %p1;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u32 %r1, [param_iters];
    ld.param.u64 %rd10, [param_cycles];

    mov.u64 %rd2, %rd1;
    mov.u32 %r2, 0;

    bar.sync 0;
    mov.u64 %rd4, %clock64;

LOOP_CG:
    ld.global.cg.u64 %rd2, [%rd2];
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra LOOP_CG;

    mov.u64 %rd5, %clock64;
    sub.u64 %rd6, %rd5, %rd4;
    st.global.u64 [%rd10], %rd6;
    st.global.u64 [%rd1], %rd2;
    ret;
}
"""


def run_experiment():
    print("=" * 88)
    print(" Experiment: GPU L2 Cache Slice Hashing & Set Aliasing Reverse Engineering")
    print(" Target: NVIDIA GeForce GTX 1650 4 GB (1 MB L2 Cache, 128-bit GDDR6)")
    print("=" * 88)

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

    ptx_src = generate_pointer_chase_ptx()
    p_file = RAW_DIR / "l2_chase.ptx"
    c_file = RAW_DIR / "l2_chase.cubin"
    with open(p_file, "w") as f:
        f.write(ptx_src)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)

    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))
    func_ca = ctypes.c_void_p()
    func_cg = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_ca), mod, b"PointerChaseKernel_ca")
    cuda.cuModuleGetFunction(ctypes.byref(func_cg), mod, b"PointerChaseKernel_cg")

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8)
    p_out = ctypes.c_uint64(d_out.value)

    CHASE_STEPS = 10000

    # =========================================================================
    # PART A: WORKING SET CAPACITY HIERARCHY (Sattolo Random Permutation)
    # =========================================================================
    print("\n--- PART A: Validated Random-Chasing Memory Hierarchy Latency ---")
    print(f"{'Footprint':<14} | {'Subsystem':<16} | {'L1+L2 (ca) (c)':<18} | {'L2-Only (cg) (c)':<18}")
    print("-" * 75)

    test_sizes_kb = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192]
    part_a_results = []

    def create_random_cycle(num_elements):
        indices = list(range(num_elements))
        for i in range(num_elements - 1, 0, -1):
            j = random.randint(0, i - 1)
            indices[i], indices[j] = indices[j], indices[i]
        return indices

    for size_kb in test_sizes_kb:
        num_elements = (size_kb * 1024) // 8
        indices = create_random_cycle(num_elements)

        d_arr = ctypes.c_uint64()
        num_bytes = num_elements * 8
        cuda.cuMemAlloc_v2(ctypes.byref(d_arr), num_bytes)

        # Correct pointer calculation: each element contains the physical pointer to next element
        base_ptr = d_arr.value
        h_arr = [base_ptr + indices[(i + 1) % num_elements] * 8 for i in range(num_elements)]
        c_arr = (ctypes.c_uint64 * num_elements)(*h_arr)
        cuda.cuMemcpyHtoD_v2(d_arr, c_arr, num_bytes)

        hops_arg = ctypes.c_uint32(CHASE_STEPS)
        p_arr = ctypes.c_uint64(d_arr.value)
        k_args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_arr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(hops_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p)
        )

        # ca (L1 + L2)
        cuda.cuLaunchKernel(func_ca, 1, 1, 1, 1, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()
        res_ca = ctypes.c_uint64()
        cuda.cuMemcpyDtoH_v2(ctypes.byref(res_ca), d_out, 8)
        cyc_ca = res_ca.value / CHASE_STEPS

        # cg (L2 only)
        cuda.cuLaunchKernel(func_cg, 1, 1, 1, 1, 1, 1, 0, None, k_args, None)
        cuda.cuCtxSynchronize()
        res_cg = ctypes.c_uint64()
        cuda.cuMemcpyDtoH_v2(ctypes.byref(res_cg), d_out, 8)
        cyc_cg = res_cg.value / CHASE_STEPS

        tier = "L1 Cache" if size_kb <= 32 else ("L2 Unified Cache" if size_kb <= 1024 else "GDDR6 VRAM")
        kb_str = f"{size_kb} KB" if size_kb < 1024 else f"{size_kb // 1024} MB"

        print(f"{kb_str:<14} | {tier:<16} | {cyc_ca:<18.2f} | {cyc_cg:<18.2f}")
        part_a_results.append({
            "size_kb": size_kb,
            "tier": tier,
            "cycles_ca": round(cyc_ca, 2),
            "cycles_cg": round(cyc_cg, 2)
        })

        cuda.cuMemFree_v2(d_arr)

    # =========================================================================
    # PART B: L2 SET ALIASING & SLICE HASHING REVERSE ENGINEERING
    # Test power-of-2 strides across 64 nodes (> 16-way L2 associativity)
    # =========================================================================
    print("\n--- PART B: GPU L2 Cache Stride Aliasing (Turing Set Indexing) ---")
    print(f"{'Stride':<12} | {'Bytes':<8} | {'Nodes':<8} | {'cg Latency (c)':<16} | {'Diagnosis'}")
    print("-" * 75)

    NODES_B = 64
    STRIDES_B = [
        ("128 B", 128),
        ("256 B", 256),
        ("512 B", 512),
        ("1 KB", 1024),
        ("4 KB", 4096),
        ("16 KB", 16384),
        ("32 KB", 32768),
        ("64 KB", 65536),
        ("65664 B", 65536 + 128),
        ("128 KB", 131072),
        ("131200 B", 131072 + 128),
        ("256 KB", 262144),
        ("512 KB", 524288)
    ]

    part_b_results = []
    hops_arg = ctypes.c_uint32(CHASE_STEPS)

    for label, stride in STRIDES_B:
        req_bytes = NODES_B * stride + 256
        d_arr_b = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_arr_b), req_bytes)

        # Build pointer chain: node i at byte offset (i * stride) points to next node at (next_i * stride)
        base_b = d_arr_b.value
        host_arr = (ctypes.c_uint64 * (req_bytes // 8))()
        for i in range(NODES_B):
            next_i = (i + 1) % NODES_B
            curr_elem = (i * stride) // 8
            next_ptr = base_b + (next_i * stride)
            host_arr[curr_elem] = next_ptr

        cuda.cuMemcpyHtoD_v2(d_arr_b, host_arr, req_bytes)

        p_arr_b = ctypes.c_uint64(d_arr_b.value)
        k_args_b = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_arr_b), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(hops_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p)
        )

        cuda.cuLaunchKernel(func_cg, 1, 1, 1, 1, 1, 1, 0, None, k_args_b, None)
        cuda.cuCtxSynchronize()
        res_cg_b = ctypes.c_uint64()
        cuda.cuMemcpyDtoH_v2(ctypes.byref(res_cg_b), d_out, 8)
        lat_b = res_cg_b.value / CHASE_STEPS

        if lat_b > 350.0:
            diag = "Thrashing (DRAM Spilled)"
        elif lat_b > 240.0:
            diag = "Partial Set Contention"
        else:
            diag = "Clean L2 Hit (Hash Defeated Aliasing)"

        print(f"{label:<12} | {stride:<8} | {NODES_B:<8} | {lat_b:<16.2f} | {diag}")
        part_b_results.append({
            "label": label,
            "stride": stride,
            "nodes": NODES_B,
            "lat_cg": round(lat_b, 2),
            "diagnosis": diag
        })

        cuda.cuMemFree_v2(d_arr_b)

    results = {
        "memory_hierarchy": part_a_results,
        "l2_set_aliasing": part_b_results
    }

    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 90)
    print(f"[SUCCESS] Physical L2 cache set aliasing results saved to: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
