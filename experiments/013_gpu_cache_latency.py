#!/usr/bin/env python3
"""
experiments/013_gpu_cache_latency.py
Frontier 2: NVIDIA Turing (GTX 1650 sm_75) GPU L1/L2 Cache Latency via clock64()

Reverse engineers the GPU memory hierarchy latency in physical GPU clock cycles using
pointer-chasing loops executed by a single thread with the hardware clock64() counter:
  - L1 Cache: 4 KB – 32 KB (Turing unified L1 Data / Shared Memory)
  - L2 Unified Cache: 64 KB – 1024 KB (TU117 1MB L2 cache)
  - GDDR6 VRAM: 4 MB – 64 MB (Off-chip DRAM boundary)
Compares ld.global.ca (L1+L2 caching) vs ld.global.cg (L1 bypass, L2 caching only).
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "013_gpu_cache_latency"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_pointer_chase_ptx(caching_modifier="ca"):
    """
    Generates PTX for pointer-chase latency using hardware %clock64.
    caching_modifier: 'ca' (cache at all levels, L1+L2) or 'cg' (cache at global level, bypass L1)
    """
    return f"""
.version 7.5
.target sm_75
.address_size 64

.visible .entry GpuCacheLatency(
    .param .u64 param_arr,
    .param .u32 param_hops,
    .param .u64 param_out_cycles
)
{{
    .reg .b32 %r<10>;
    .reg .b64 %rd<15>;
    .reg .pred %p<4>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u32 %r1, [param_hops];
    ld.param.u64 %rd2, [param_out_cycles];

    // Only Thread 0 of Block 0 executes the pointer chase
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    or.b32 %r4, %r2, %r3;
    setp.ne.u32 %p1, %r4, 0;
    @%p1 bra EXIT_L;

    // Warmup traversal (1,000 hops to populate cache)
    mov.u64 %rd3, 0;
    mov.u32 %r5, 0;
WARMUP_L:
    setp.ge.u32 %p2, %r5, 1000;
    @%p2 bra BENCH_L;
    shl.b64 %rd4, %rd3, 3; // 64-bit indices (8 bytes)
    add.s64 %rd5, %rd1, %rd4;
    ld.global.{caching_modifier}.u64 %rd3, [%rd5];
    add.u32 %r5, %r5, 1;
    bra WARMUP_L;

BENCH_L:
    // Memory fence and read clock64
    membar.gl;
    mov.u64 %rd6, %clock64;

    mov.u32 %r6, 0;
LOOP_L:
    setp.ge.u32 %p3, %r6, %r1;
    @%p3 bra DONE_L;
    shl.b64 %rd7, %rd3, 3;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.{caching_modifier}.u64 %rd3, [%rd8];
    add.u32 %r6, %r6, 1;
    bra LOOP_L;

DONE_L:
    membar.gl;
    mov.u64 %rd9, %clock64;
    sub.u64 %rd10, %rd9, %rd6;

    // Store elapsed clock cycles and final pointer value
    st.global.u64 [%rd2], %rd10;

    // Prevent dead code elimination of rd3
    shl.b64 %rd11, %rd3, 3;
    add.s64 %rd12, %rd1, %rd11;
    st.global.u64 [%rd12], %rd3;

EXIT_L:
    ret;
}}
"""


def build_kernel(modifier):
    ptx_code = generate_pointer_chase_ptx(modifier)
    ptx_file = RESULTS_DIR / f"cache_latency_{modifier}.ptx"
    cubin_file = RESULTS_DIR / f"cache_latency_{modifier}.cubin"
    sass_file = RESULTS_DIR / f"cache_latency_{modifier}_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
    with open(sass_file, "w") as f:
        f.write(sass_out)

    return cubin_file


def create_random_cycle(num_elements):
    """Creates a randomized cyclic permutation of indices 0..num_elements-1."""
    indices = list(range(num_elements))
    # Sattolo's algorithm for single cyclic permutation
    for i in range(num_elements - 1, 0, -1):
        j = random.randint(0, i - 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def run_benchmark():
    print("=" * 75)
    print(" Frontier 2: NVIDIA Turing (GTX 1650 sm_75) GPU L1/L2 Cache Latency")
    print(" Pointer-Chasing with hardware %clock64 (Single Thread)")
    print("=" * 75)

    print("[*] Compiling PTX pointer-chasing kernels (ca = Cache All, cg = Cache L2 only)...")
    cubin_ca = build_kernel("ca")
    cubin_cg = build_kernel("cg")

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

    mod_ca = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(mod_ca), str(cubin_ca.resolve()).encode("utf-8")), "Load CA")
    func_ca = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func_ca), mod_ca, b"GpuCacheLatency"), "GetFunc CA")

    mod_cg = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(mod_cg), str(cubin_cg.resolve()).encode("utf-8")), "Load CG")
    func_cg = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func_cg), mod_cg, b"GpuCacheLatency"), "GetFunc CG")

    # Working set sizes: 4 KB up to 32 MB
    test_sizes_kb = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 32768]
    HOPS = 100_000

    d_out = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8), "cuMemAlloc out")

    print("\n" + "-" * 75)
    print(f"{'Working Set':<14} | {'Tier / Location':<18} | {'L1+L2 (Cycles)':<18} | {'L2 Only (Cycles)':<18}")
    print("-" * 75)

    results = []

    for size_kb in test_sizes_kb:
        num_elements = (size_kb * 1024) // 8 # 8 bytes per uint64 pointer
        h_arr = create_random_cycle(num_elements)
        c_arr = (ctypes.c_uint64 * num_elements)(*h_arr)

        d_arr = ctypes.c_uint64()
        num_bytes = num_elements * 8
        check(cuda.cuMemAlloc_v2(ctypes.byref(d_arr), num_bytes), f"Alloc {size_kb}KB")
        check(cuda.cuMemcpyHtoD_v2(d_arr, c_arr, num_bytes), f"Copy {size_kb}KB")

        hops_arg = ctypes.c_uint32(HOPS)
        p_arr = ctypes.c_uint64(d_arr.value)
        p_out = ctypes.c_uint64(d_out.value)
        k_args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_arr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(hops_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p)
        )

        # Test CA (L1+L2)
        check(cuda.cuLaunchKernel(func_ca, 1, 1, 1, 32, 1, 1, 0, None, k_args, None), "Launch CA")
        check(cuda.cuCtxSynchronize(), "Sync CA")
        res_cycles_ca = ctypes.c_uint64()
        check(cuda.cuMemcpyDtoH_v2(ctypes.byref(res_cycles_ca), d_out, 8), "D2H CA")
        cyc_ca = res_cycles_ca.value / HOPS

        # Test CG (L2 Only)
        check(cuda.cuLaunchKernel(func_cg, 1, 1, 1, 32, 1, 1, 0, None, k_args, None), "Launch CG")
        check(cuda.cuCtxSynchronize(), "Sync CG")
        res_cycles_cg = ctypes.c_uint64()
        check(cuda.cuMemcpyDtoH_v2(ctypes.byref(res_cycles_cg), d_out, 8), "D2H CG")
        cyc_cg = res_cycles_cg.value / HOPS

        tier = "L1 Cache" if size_kb <= 32 else ("L2 Cache" if size_kb <= 1024 else "GDDR6 VRAM")
        kb_str = f"{size_kb} KB" if size_kb < 1024 else f"{size_kb // 1024} MB"

        print(f"{kb_str:<14} | {tier:<18} | {cyc_ca:<18.1f} | {cyc_cg:<18.1f}")

        results.append({
            "size_kb": size_kb,
            "tier": tier,
            "cycles_ca_l1_l2": round(cyc_ca, 2),
            "cycles_cg_l2_only": round(cyc_cg, 2)
        })

        cuda.cuMemFree_v2(d_arr)

    print("=" * 75)
    out_json = RESULTS_DIR / "gpu_cache_latency.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved GPU cache latency telemetry to: {out_json}\n")


if __name__ == "__main__":
    run_benchmark()
