#!/usr/bin/env python3
"""
experiments/010_bank_conflicts.py
Step 1: GPU Shared Memory 32-Bank Conflict Serialization (GTX 1650)

Empirically benchmarks on-chip SRAM bank conflicts across access strides:
  - Stride 1:  1-way (Conflict-free, 32 distinct banks hit)
  - Stride 2:  2-way bank conflict (2 passes)
  - Stride 4:  4-way bank conflict (4 passes)
  - Stride 8:  8-way bank conflict (8 passes)
  - Stride 16: 16-way bank conflict (16 passes)
  - Stride 32: 32-way maximum bank conflict (all 32 threads hit Bank 0)
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "010_bank_conflicts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_bank_conflict_ptx(stride: int) -> str:
    """
    Generates PTX with an unrolled shared memory loop accessing with given stride.
    Shared memory size: 4096 floats (16 KB).
    """
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry BankConflictKernel(.param .u64 param_arr, .param .u32 param_iters) {",
        "  .shared .f32 smem[4096];",
        "  .reg .b32 %r<15>;",
        "  .reg .f32 %f<10>;",
        "  .reg .b64 %rd<5>;",
        "  .reg .pred %p<2>;",
        "  ld.param.u64 %rd1, [param_arr];",
        "  ld.param.u32 %r1, [param_iters];",
        "  mov.u32 %r2, %tid.x;", # thread within block
        # Calculate shared memory index: (tid.x * stride) % 4096 = (tid.x * stride) & 4095
        f"  mul.lo.u32 %r3, %r2, {stride};",
        "  and.b32 %r4, %r3, 4095;",
        "  mul.wide.u32 %rd2, %r4, 4;",
        "  mov.u64 %rd3, smem;",
        "  add.s64 %rd4, %rd3, %rd2;",
        # Initialize smem element
        "  mov.f32 %f1, 1.0;",
        "  st.shared.f32 [%rd4], %f1;",
        "  bar.sync 0;",
        # Unrolled shared memory read-modify-write loop
        "  mov.u32 %r5, 0;",
        "LOOP:",
        "  ld.shared.f32 %f2, [%rd4];",
        "  add.f32 %f2, %f2, 0.001;",
        "  st.shared.f32 [%rd4], %f2;",
        "  ld.shared.f32 %f2, [%rd4];",
        "  add.f32 %f2, %f2, 0.001;",
        "  st.shared.f32 [%rd4], %f2;",
        "  add.u32 %r5, %r5, 1;",
        "  setp.lt.u32 %p1, %r5, %r1;",
        "  @%p1 bra LOOP;",
        "  bar.sync 0;",
        # Store result back to global memory for verification
        "  mul.wide.u32 %rd2, %r2, 4;",
        "  add.s64 %rd3, %rd1, %rd2;",
        "  st.global.f32 [%rd3], %f2;",
        "  ret;",
        "}"
    ]
    return "\n".join(lines)


def run_benchmark():
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

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    # 1024 threads per block, 64 blocks = 65,536 threads total
    THREADS = 1024
    BLOCKS = 64
    d_mem = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), THREADS * BLOCKS * 4), "cuMemAlloc")

    strides = [1, 2, 4, 8, 16, 32]
    results = []

    print("=" * 75)
    print(" Step 1: GPU Shared Memory 32-Bank Conflict Serialization (GTX 1650)")
    print(" 64 Blocks x 1024 Threads (20,000 unrolled SRAM transactions per thread)")
    print("=" * 75)
    print(f"{'Access Stride':<15} | {'Bank Conflict Tier':<22} | {'Kernel Time (ms)':<18} | {'Slowdown Factor':<15}")
    print("-" * 75)

    baseline_ms = 0.0

    for stride in strides:
        ptx_src = generate_bank_conflict_ptx(stride)
        ptx_file = RESULTS_DIR / f"stride_{stride}.ptx"
        cubin_file = RESULTS_DIR / f"stride_{stride}.cubin"
        sass_file = RESULTS_DIR / f"stride_{stride}_sass.txt"

        with open(ptx_file, "w") as f:
            f.write(ptx_src)

        subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

        custom_env = os.environ.copy()
        custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
        sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
        with open(sass_file, "w") as f:
            f.write(sass_out)

        module = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(module), str(cubin_file.resolve()).encode("utf-8")), "cuModuleLoad")
        func = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"BankConflictKernel"), "cuModuleGetFunction")

        iters = ctypes.c_uint32(5000)
        param_ptr = ctypes.c_uint64(d_mem.value)
        k_args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(param_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(iters), ctypes.c_void_p)
        )

        # Warmup
        check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Warmup")
        check(cuda.cuCtxSynchronize(), "Sync")

        times = []
        for _ in range(5):
            check(cuda.cuEventRecord(start_evt, None), "Start")
            check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Launch")
            check(cuda.cuEventRecord(stop_evt, None), "Stop")
            check(cuda.cuEventSynchronize(stop_evt), "Sync")
            el = ctypes.c_float()
            check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
            times.append(el.value)

        avg_ms = sum(times) / len(times)

        tier_name = "Conflict-Free (1-way)" if stride == 1 else f"{stride}-Way Conflict"
        if stride == 1:
            baseline_ms = avg_ms
            slowdown = "1.00x (Baseline)"
        else:
            rel = avg_ms / baseline_ms if baseline_ms > 0 else 1.0
            slowdown = f"{rel:.2f}x"

        print(f"Stride {stride:<8} | {tier_name:<22} | {avg_ms:<18.4f} | {slowdown:<15}")

        results.append({
            "stride": stride,
            "tier": tier_name,
            "kernel_time_ms": round(avg_ms, 4),
            "slowdown": round(avg_ms / baseline_ms, 2) if baseline_ms > 0 else 1.0
        })

        cuda.cuModuleUnload(module)

    print("=" * 75)
    out_json = RESULTS_DIR / "bank_conflicts.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved shared memory bank conflict results to: {out_json}\n")

    cuda.cuMemFree_v2(d_mem)
    cuda.cuEventDestroy(start_evt)
    cuda.cuEventDestroy(stop_evt)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    run_benchmark()
