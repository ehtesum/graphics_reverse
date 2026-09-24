#!/usr/bin/env python3
"""
experiments/011_warp_shuffle.py
Step 2: Warp Register Shuffle Units (SHFL.BFLY) vs Shared Memory vs Global Atomics

Compares three intra-warp reduction paradigms on the NVIDIA GeForce GTX 1650:
  1. Global Memory Atomics (High contention DRAM transactions)
  2. Shared Memory Reduction (SRAM load/store with bar.sync barriers)
  3. Register Butterfly Shuffle (Zero-memory SM register routing: shfl.sync.bfly)
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "011_warp_shuffle"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def build_ptx_kernels():
    # 1. Global Atomics
    ptx_atomic = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry ReductionAtomic(
    .param .u64 param_arr,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .b32 %r<10>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r1, [param_n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    // Every thread contends on single global accumulator
    atom.global.add.f32 %f2, [%rd2], %f1;
    ret;
}
"""

    # 2. Shared Memory Reduction (Full 256-thread binary tree)
    ptx_smem = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry ReductionShared(
    .param .u64 param_arr,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .shared .f32 smem[256];
    .reg .b32 %r<15>;
    .reg .f32 %f<15>;
    .reg .b64 %rd<15>;
    .reg .pred %p<10>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u64 %rd2, [param_out];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    mul.wide.u32 %rd5, %r4, 4;
    mov.u64 %rd6, smem;
    add.s64 %rd7, %rd6, %rd5;
    st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 128
    setp.lt.u32 %p1, %r4, 128;
    @%p1 add.u32 %r6, %r4, 128;
    @%p1 mul.wide.u32 %rd8, %r6, 4;
    @%p1 add.s64 %rd9, %rd6, %rd8;
    @%p1 ld.shared.f32 %f2, [%rd9];
    @%p1 add.f32 %f1, %f1, %f2;
    @%p1 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 64
    setp.lt.u32 %p2, %r4, 64;
    @%p2 add.u32 %r6, %r4, 64;
    @%p2 mul.wide.u32 %rd8, %r6, 4;
    @%p2 add.s64 %rd9, %rd6, %rd8;
    @%p2 ld.shared.f32 %f2, [%rd9];
    @%p2 add.f32 %f1, %f1, %f2;
    @%p2 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 32
    setp.lt.u32 %p3, %r4, 32;
    @%p3 add.u32 %r6, %r4, 32;
    @%p3 mul.wide.u32 %rd8, %r6, 4;
    @%p3 add.s64 %rd9, %rd6, %rd8;
    @%p3 ld.shared.f32 %f2, [%rd9];
    @%p3 add.f32 %f1, %f1, %f2;
    @%p3 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 16
    setp.lt.u32 %p4, %r4, 16;
    @%p4 add.u32 %r6, %r4, 16;
    @%p4 mul.wide.u32 %rd8, %r6, 4;
    @%p4 add.s64 %rd9, %rd6, %rd8;
    @%p4 ld.shared.f32 %f2, [%rd9];
    @%p4 add.f32 %f1, %f1, %f2;
    @%p4 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 8
    setp.lt.u32 %p5, %r4, 8;
    @%p5 add.u32 %r6, %r4, 8;
    @%p5 mul.wide.u32 %rd8, %r6, 4;
    @%p5 add.s64 %rd9, %rd6, %rd8;
    @%p5 ld.shared.f32 %f2, [%rd9];
    @%p5 add.f32 %f1, %f1, %f2;
    @%p5 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 4
    setp.lt.u32 %p6, %r4, 4;
    @%p6 add.u32 %r6, %r4, 4;
    @%p6 mul.wide.u32 %rd8, %r6, 4;
    @%p6 add.s64 %rd9, %rd6, %rd8;
    @%p6 ld.shared.f32 %f2, [%rd9];
    @%p6 add.f32 %f1, %f1, %f2;
    @%p6 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 2
    setp.lt.u32 %p7, %r4, 2;
    @%p7 add.u32 %r6, %r4, 2;
    @%p7 mul.wide.u32 %rd8, %r6, 4;
    @%p7 add.s64 %rd9, %rd6, %rd8;
    @%p7 ld.shared.f32 %f2, [%rd9];
    @%p7 add.f32 %f1, %f1, %f2;
    @%p7 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Step 1
    setp.lt.u32 %p8, %r4, 1;
    @%p8 add.u32 %r6, %r4, 1;
    @%p8 mul.wide.u32 %rd8, %r6, 4;
    @%p8 add.s64 %rd9, %rd6, %rd8;
    @%p8 ld.shared.f32 %f2, [%rd9];
    @%p8 add.f32 %f1, %f1, %f2;
    @%p8 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    setp.eq.u32 %p9, %r4, 0;
    @%p9 atom.global.add.f32 %f3, [%rd2], %f1;
    ret;
}
"""

    # 3. Register Warp Shuffle (shfl.sync.bfly) + 8-warp smem reduction
    ptx_shuffle = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry ReductionShuffle(
    .param .u64 param_arr,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .shared .f32 warp_sums[8];
    .reg .b32 %r<15>;
    .reg .f32 %f<15>;
    .reg .b64 %rd<15>;
    .reg .pred %p<5>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u64 %rd2, [param_out];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    // Full mask 0xFFFFFFFF for 32 threads
    mov.u32 %r6, 0xFFFFFFFF;

    // Butterfly Warp Shuffles: offset 16, 8, 4, 2, 1
    shfl.sync.bfly.b32 %f2, %f1, 16, 0x1F, %r6;
    add.f32 %f1, %f1, %f2;

    shfl.sync.bfly.b32 %f2, %f1, 8, 0x1F, %r6;
    add.f32 %f1, %f1, %f2;

    shfl.sync.bfly.b32 %f2, %f1, 4, 0x1F, %r6;
    add.f32 %f1, %f1, %f2;

    shfl.sync.bfly.b32 %f2, %f1, 2, 0x1F, %r6;
    add.f32 %f1, %f1, %f2;

    shfl.sync.bfly.b32 %f2, %f1, 1, 0x1F, %r6;
    add.f32 %f1, %f1, %f2;

    // Lane 0 of each warp (8 warps total) writes warp sum to shared memory
    and.b32 %r7, %r4, 31;
    shr.u32 %r8, %r4, 5;
    setp.eq.u32 %p1, %r7, 0;
    @%p1 mul.wide.u32 %rd5, %r8, 4;
    @%p1 mov.u64 %rd6, warp_sums;
    @%p1 add.s64 %rd7, %rd6, %rd5;
    @%p1 st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // First 8 threads load the 8 warp sums and reduce them via shuffle
    setp.lt.u32 %p2, %r4, 8;
    @%p2 mul.wide.u32 %rd8, %r4, 4;
    @%p2 mov.u64 %rd9, warp_sums;
    @%p2 add.s64 %rd10, %rd9, %rd8;
    @%p2 ld.shared.f32 %f4, [%rd10];
    @!%p2 mov.f32 %f4, 0f00000000;

    shfl.sync.bfly.b32 %f5, %f4, 4, 0x1F, %r6;
    add.f32 %f4, %f4, %f5;

    shfl.sync.bfly.b32 %f5, %f4, 2, 0x1F, %r6;
    add.f32 %f4, %f4, %f5;

    shfl.sync.bfly.b32 %f5, %f4, 1, 0x1F, %r6;
    add.f32 %f4, %f4, %f5;

    // Lane 0 of warp 0 adds block sum to global accumulator (1 atomic per block!)
    setp.eq.u32 %p3, %r4, 0;
    @%p3 atom.global.add.f32 %f6, [%rd2], %f4;
    ret;
}
"""
    kernels = {
        "atomic": ptx_atomic,
        "shared": ptx_smem,
        "shuffle": ptx_shuffle
    }

    cubins = {}
    for name, code in kernels.items():
        p_file = RESULTS_DIR / f"{name}.ptx"
        c_file = RESULTS_DIR / f"{name}.cubin"
        s_file = RESULTS_DIR / f"{name}_sass.txt"
        with open(p_file, "w") as f:
            f.write(code)
        subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
        custom_env = os.environ.copy()
        custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
        sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(c_file)], env=custom_env, text=True)
        with open(s_file, "w") as f:
            f.write(sass_out)
        cubins[name] = c_file

    return cubins


def run_benchmark():
    cubins = build_ptx_kernels()

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

    N = 1024 * 1024
    num_bytes = N * 4
    d_in = ctypes.c_uint64()
    d_out = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), num_bytes), "cuMemAlloc in")
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4), "cuMemAlloc out")

    h_in = (ctypes.c_float * N)(*([1.0] * N))
    check(cuda.cuMemcpyHtoD_v2(d_in, h_in, num_bytes), "cuMemcpyHtoD in")

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    BLOCKS = 4096
    THREADS = 256
    p_in = ctypes.c_uint64(d_in.value)
    p_out = ctypes.c_uint64(d_out.value)
    n_arg = ctypes.c_uint32(N)
    k_args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.byref(p_in), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p)
    )

    tests = [
        ("Global Atomics (DRAM)", "atomic", "ReductionAtomic"),
        ("Shared Memory Tree (SRAM)", "shared", "ReductionShared"),
        ("Warp Shuffle (Registers)", "shuffle", "ReductionShuffle")
    ]

    print("=" * 75)
    print(" Step 2: Warp Register Shuffle (SHFL.BFLY) vs Shared Memory vs Atomics")
    print(" 1,048,576 Floats Parallel Reduction (GTX 1650)")
    print("=" * 75)
    print(f"{'Reduction Method':<30} | {'Kernel Time (ms)':<18} | {'Speedup vs Atomic':<18}")
    print("-" * 75)

    results = []
    atomic_time = 0.0

    for label, key, func_name in tests:
        mod = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(mod), str(cubins[key].resolve()).encode("utf-8")), f"cuModuleLoad {key}")
        func = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, func_name.encode("utf-8")), f"GetFunc {func_name}")

        # Clear output
        zero = (ctypes.c_float * 1)(0.0)
        check(cuda.cuMemcpyHtoD_v2(d_out, zero, 4), "Zero out")

        # Warmup
        check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Warmup")
        check(cuda.cuCtxSynchronize(), "Sync")

        times = []
        for _ in range(5):
            check(cuda.cuMemcpyHtoD_v2(d_out, zero, 4), "Zero out")
            check(cuda.cuEventRecord(start_evt, None), "Start")
            check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Launch")
            check(cuda.cuEventRecord(stop_evt, None), "Stop")
            check(cuda.cuEventSynchronize(stop_evt), "Sync")
            el = ctypes.c_float()
            check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
            times.append(el.value)

        avg_ms = sum(times) / len(times)

        # Verify output
        h_res = (ctypes.c_float * 1)()
        check(cuda.cuMemcpyDtoH_v2(h_res, d_out, 4), "Memcpy D2H")

        if key == "atomic":
            atomic_time = avg_ms
            speedup = "1.00x (Baseline)"
        else:
            rel = atomic_time / avg_ms if avg_ms > 0 else 1.0
            speedup = f"{rel:.2f}x faster"

        print(f"{label:<30} | {avg_ms:<18.4f} | {speedup:<18}")

        results.append({
            "method": label,
            "kernel_time_ms": round(avg_ms, 4),
            "speedup": round(atomic_time / avg_ms, 2) if avg_ms > 0 else 1.0,
            "result_sum": round(h_res[0], 1)
        })

        cuda.cuModuleUnload(mod)

    print("=" * 75)
    out_json = RESULTS_DIR / "warp_shuffle.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved warp shuffle reduction results to: {out_json}\n")

    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    cuda.cuEventDestroy(start_evt)
    cuda.cuEventDestroy(stop_evt)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    run_benchmark()
