#!/usr/bin/env python3
"""
experiments/016_atomic_contention.py
Frontier 5: GPU Lock-Free Synchronization & Atomic Contention Scaling (GTX 1650)

Measures atomic transaction serialization latency across hardware levels:
  - 1 Thread (No Contention)
  - 32 Threads (Intra-Warp Lockstep Contention)
  - 256 Threads (Single SM Intra-Block Contention)
  - 4,096 Threads (Multi-SM Inter-Block Contention)
  - 65,536 Threads (L2 Cache Controller Contention)
  - 262,144 Threads (Massive Silicon-Wide DRAM/L2 Saturation)
Compares:
  - atom.global.add
  - atom.global.exch
  - atom.global.cas (Compare-And-Swap)
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "016_atomic_contention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_atomic_ptx(op_type):
    # op_type in ['add', 'exch', 'cas']
    lines = []
    lines.append(".version 7.5")
    lines.append(".target sm_75")
    lines.append(".address_size 64")
    lines.append(f".visible .entry AtomicContention_{op_type}(")
    lines.append("    .param .u64 param_target,")
    lines.append("    .param .u32 param_iters")
    lines.append(")")
    lines.append("{")
    lines.append("    .reg .b32 %r<10>;")
    lines.append("    .reg .b64 %rd<10>;")
    lines.append("    .reg .pred %p<2>;")
    lines.append("")
    lines.append("    ld.param.u64 %rd1, [param_target];")
    lines.append("    ld.param.u32 %r1, [param_iters];")
    lines.append("    mov.u32 %r2, 0;")
    lines.append("")
    lines.append("LOOP:")
    lines.append("    setp.ge.u32 %p1, %r2, %r1;")
    lines.append("    @%p1 bra DONE;")
    lines.append("")

    if op_type == "add":
        lines.append("    atom.global.add.u32 %r3, [%rd1], 1;")
    elif op_type == "exch":
        lines.append("    atom.global.exch.b32 %r3, [%rd1], %r2;")
    elif op_type == "cas":
        # CAS: atom.global.cas.b32 d, [a], compare, val
        # If value is %r3, replace with %r2
        lines.append("    atom.global.cas.b32 %r3, [%rd1], %r3, %r2;")

    lines.append("    add.u32 %r2, %r2, 1;")
    lines.append("    bra LOOP;")
    lines.append("DONE:")
    lines.append("    ret;")
    lines.append("}")
    return "\n".join(lines)


def build_kernel(op_type):
    ptx_code = generate_atomic_ptx(op_type)
    ptx_file = RESULTS_DIR / f"atomic_{op_type}.ptx"
    cubin_file = RESULTS_DIR / f"atomic_{op_type}.cubin"
    sass_file = RESULTS_DIR / f"atomic_{op_type}_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
    with open(sass_file, "w") as f:
        f.write(sass_out)

    return cubin_file


def run_benchmark():
    print("=" * 75)
    print(" Frontier 5: GPU Lock-Free Synchronization & Atomic Contention Scaling")
    print(" NVIDIA GeForce GTX 1650 (Turing sm_75 TU117)")
    print("=" * 75)

    ops = ["add", "exch", "cas"]
    cubins = {op: build_kernel(op) for op in ops}

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

    # Allocate target accumulator in global memory
    d_target = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_target), 4), "cuMemAlloc")

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    # Thread configurations: (total_threads, blocks, threads_per_block, tier_label)
    thread_configs = [
        (1, 1, 1, "1 Thread (Zero Contention)"),
        (32, 1, 32, "32 Threads (1 Warp Lockstep)"),
        (256, 1, 256, "256 Threads (1 SM Intra-Block)"),
        (4096, 16, 256, "4,096 Threads (Multi-SM Inter-Block)"),
        (65536, 256, 256, "65,536 Threads (L2 Cache Bound)"),
        (262144, 1024, 256, "262,144 Threads (Silicon-Wide Contention)")
    ]

    ITERS = 100 # 100 atomic operations per thread
    all_results = {}

    for op in ops:
        mod = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(mod), str(cubins[op].resolve()).encode("utf-8")), f"Load {op}")
        func = ctypes.c_void_p()
        func_name = f"AtomicContention_{op}"
        check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, func_name.encode("utf-8")), f"GetFunc {func_name}")

        print(f"\n[*] Benchmarking Operation: atom.global.{op.upper()}...")
        print(f"{'Concurrency Tier':<40} | {'Kernel Time':<12} | {'Per-Op Latency':<16} | {'Slowdown'}")
        print("-" * 75)

        op_results = []
        baseline_per_op_ns = 0.0

        for total_t, blocks, tpb, tier_label in thread_configs:
            zero = (ctypes.c_uint32 * 1)(0)
            check(cuda.cuMemcpyHtoD_v2(d_target, zero, 4), "Reset")

            p_target = ctypes.c_uint64(d_target.value)
            iters_arg = ctypes.c_uint32(ITERS)
            k_args = (ctypes.c_void_p * 2)(
                ctypes.cast(ctypes.byref(p_target), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(iters_arg), ctypes.c_void_p)
            )

            # Warmup
            check(cuda.cuLaunchKernel(func, blocks, 1, 1, tpb, 1, 1, 0, None, k_args, None), "Warmup")
            check(cuda.cuCtxSynchronize(), "Sync")

            times = []
            for _ in range(5):
                check(cuda.cuEventRecord(start_evt, None), "Start")
                check(cuda.cuLaunchKernel(func, blocks, 1, 1, tpb, 1, 1, 0, None, k_args, None), "Launch")
                check(cuda.cuEventRecord(stop_evt, None), "Stop")
                check(cuda.cuEventSynchronize(stop_evt), "Sync")
                el = ctypes.c_float()
                check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
                times.append(el.value)

            avg_ms = sum(times) / len(times)
            total_atomic_ops = total_t * ITERS
            per_op_ns = (avg_ms * 1e6) / total_atomic_ops

            if total_t == 1:
                baseline_per_op_ns = per_op_ns
                slowdown_str = "1.00x (Baseline)"
            else:
                ratio = per_op_ns / baseline_per_op_ns
                slowdown_str = f"{ratio:.2f}x"

            print(f"{tier_label:<40} | {avg_ms:<10.3f} ms | {per_op_ns:<12.3f} ns | {slowdown_str}")

            op_results.append({
                "concurrency_tier": tier_label,
                "total_threads": total_t,
                "kernel_time_ms": round(avg_ms, 4),
                "per_op_latency_ns": round(per_op_ns, 3),
                "slowdown_factor": round(per_op_ns / baseline_per_op_ns, 2) if baseline_per_op_ns > 0 else 1.0
            })

        all_results[op] = op_results

    out_file = RESULTS_DIR / "atomic_contention.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Saved atomic contention telemetry to: {out_file}\n")


if __name__ == "__main__":
    run_benchmark()
