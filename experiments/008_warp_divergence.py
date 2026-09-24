#!/usr/bin/env python3
"""
experiments/008_warp_divergence.py
Frontier 1: GPU Warp Divergence & Turing Independent Thread Scheduling (ITS)

Generates and benchmarks 1-way (uniform), 2-way, 4-way, 8-way, 16-way, and 32-way
branch divergence on the NVIDIA GeForce GTX 1650 across 1,048,576 threads.
Measures the physical hardware execution penalty as the SM warp scheduler serializes paths.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "008_warp_divergence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM

def generate_divergence_ptx(ways: int) -> str:
    """Generates PTX with N divergent paths per 32-thread warp."""
    lines = [
        ".version 7.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry _Z4testPf(.param .u64 _Z4testPf_param_0) {",
        "  .reg .b32 %r<10>;",
        "  .reg .f32 %f<10>;",
        "  .reg .b64 %rd<5>;",
        "  .reg .pred %p<40>;",
        "  ld.param.u64 %rd1, [_Z4testPf_param_0];",
        "  mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;",
        "  mad.lo.s32 %r4, %r1, %r2, %r3;",
        "  mul.wide.s32 %rd2, %r4, 4;",
        "  add.s64 %rd3, %rd1, %rd2;",
        "  ld.global.f32 %f0, [%rd3];",
        f"  and.b32 %r5, %r3, {ways - 1};" # tid.x % ways (since ways is power of 2)
    ]

    # Generate branches for each way
    for w in range(ways):
        lines.append(f"  setp.eq.u32 %p{w}, %r5, {w};")
        lines.append(f"  @%p{w} bra PATH_{w};")

    lines.append("  bra MERGE;")

    for w in range(ways):
        lines.append(f"PATH_{w}:")
        # Exact IEEE-754 32-bit float hex
        import struct
        h_val = f"{struct.unpack('<I', struct.pack('<f', float(w + 1.0)))[0]:08X}"
        lines.append(f"  add.f32 %f1, %f0, 0f{h_val};")
        lines.append(f"  bra MERGE;")

    lines.append("MERGE:")
    lines.append("  st.global.f32 [%rd3], %f1;")
    lines.append("  ret;")
    lines.append("}")
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

    N = 1024 * 1024
    num_bytes = N * 4
    d_mem = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), "cuMemAlloc")
    h_init = (ctypes.c_float * N)(*([1.0] * N))

    way_configs = [1, 2, 4, 8, 16, 32]
    results = []

    print("=" * 70)
    print(" Frontier 1: GPU Warp Divergence Benchmark (NVIDIA GTX 1650)")
    print(" 1,048,576 Threads (4096 Blocks x 256 Threads, 8 Warps/Block)")
    print("=" * 70)
    print(f"{'Divergence Level':<20} | {'Kernel Time (ms)':<18} | {'Relative Penalty':<18}")
    print("-" * 70)

    baseline_time = 0.0

    for ways in way_configs:
        ptx_src = generate_divergence_ptx(ways)
        ptx_file = RESULTS_DIR / f"divergence_{ways}way.ptx"
        cubin_file = RESULTS_DIR / f"divergence_{ways}way.cubin"
        sass_file = RESULTS_DIR / f"divergence_{ways}way_sass.txt"

        with open(ptx_file, "w") as f:
            f.write(ptx_src)

        # Assemble with ptxas
        subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

        # Disassemble SASS
        custom_env = os.environ.copy()
        custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
        sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
        with open(sass_file, "w") as f:
            f.write(sass_out)

        # Load into GPU
        module = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(module), str(cubin_file.resolve()).encode("utf-8")), "cuModuleLoad")
        func = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"_Z4testPf"), "cuModuleGetFunction")

        # Reset memory
        check(cuda.cuMemcpyHtoD_v2(d_mem, h_init, num_bytes), "cuMemcpyHtoD")

        # Warmup
        param_ptr = ctypes.c_uint64(d_mem.value)
        k_args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(param_ptr), ctypes.c_void_p))
        check(cuda.cuLaunchKernel(func, 4096, 1, 1, 256, 1, 1, 0, None, k_args, None), "Launch warmup")
        check(cuda.cuCtxSynchronize(), "Sync")

        # Timed launches (average of 5 runs)
        times = []
        for _ in range(5):
            check(cuda.cuEventRecord(start_evt, None), "Record start")
            check(cuda.cuLaunchKernel(func, 4096, 1, 1, 256, 1, 1, 0, None, k_args, None), "Launch")
            check(cuda.cuEventRecord(stop_evt, None), "Record stop")
            check(cuda.cuEventSynchronize(stop_evt), "Sync")
            el = ctypes.c_float()
            check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
            times.append(el.value)

        avg_ms = sum(times) / len(times)
        if ways == 1:
            baseline_time = avg_ms
            penalty = "1.00x (Baseline)"
        else:
            rel = avg_ms / baseline_time if baseline_time > 0 else 1.0
            penalty = f"{rel:.2f}x"

        label = f"{ways}-way Divergence" if ways > 1 else "1-way (Uniform)"
        print(f"{label:<20} | {avg_ms:<18.4f} | {penalty:<18}")

        results.append({
            "ways": ways,
            "label": label,
            "kernel_time_ms": round(avg_ms, 4),
            "relative_penalty": round(avg_ms / baseline_time, 2) if baseline_time > 0 else 1.0
        })

        cuda.cuModuleUnload(module)

    print("=" * 70)
    out_json = RESULTS_DIR / "warp_divergence.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved warp divergence metrics to: {out_json}\n")

    cuda.cuMemFree_v2(d_mem)
    cuda.cuEventDestroy(start_evt)
    cuda.cuEventDestroy(stop_evt)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    run_benchmark()
