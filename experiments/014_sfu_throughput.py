#!/usr/bin/env python3
"""
experiments/014_sfu_throughput.py
Frontier 3: Turing Special Function Units (SFU) vs ALU Throughput (TU117 / GTX 1650)

Measures the raw instruction throughput and execution pipe latency of:
  - ALU: FFMA (fma.rn.f32), FADD (add.f32), FMUL (mul.f32)
  - SFU (Special Function Units / MUFU):
      - MUFU.RSQ (rsqrt.approx.f32)
      - MUFU.SIN (sin.approx.f32)
      - MUFU.COS (cos.approx.f32)
      - MUFU.LG2 (lg2.approx.f32)
      - MUFU.EX2 (ex2.approx.f32)
Computes the exact physical ALU:SFU silicon execution ratio.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "014_sfu_throughput"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_throughput_ptx(op_type, unroll_count=1000):
    lines = []
    lines.append(".version 7.5")
    lines.append(".target sm_75")
    lines.append(".address_size 64")
    lines.append(f".visible .entry Benchmark_{op_type}(")
    lines.append("    .param .u64 param_arr,")
    lines.append("    .param .u32 param_iters")
    lines.append(")")
    lines.append("{")
    lines.append("    .reg .b32 %r<10>;")
    lines.append("    .reg .f32 %f<15>;")
    lines.append("    .reg .b64 %rd<10>;")
    lines.append("    .reg .pred %p<4>;")
    lines.append("")
    lines.append("    ld.param.u64 %rd1, [param_arr];")
    lines.append("    ld.param.u32 %r1, [param_iters];")
    lines.append("    mov.u32 %r2, %ctaid.x;")
    lines.append("    mov.u32 %r3, %ntid.x;")
    lines.append("    mov.u32 %r4, %tid.x;")
    lines.append("    mad.lo.s32 %r5, %r2, %r3, %r4;")
    lines.append("")
    lines.append("    mul.wide.s32 %rd2, %r5, 4;")
    lines.append("    add.s64 %rd3, %rd1, %rd2;")
    lines.append("    ld.global.f32 %f1, [%rd3];")
    lines.append("    mov.f32 %f2, 0f3F800010; // 1.0000019f (non-identity)")
    lines.append("    mov.f32 %f3, 0f3F000000; // 0.5f")
    lines.append("    mov.f32 %f4, %f1;")
    lines.append("    mov.f32 %f5, %f1;")
    lines.append("")
    lines.append("    mov.u32 %r6, 0;")
    lines.append("LOOP:")
    lines.append("    setp.ge.u32 %p1, %r6, %r1;")
    lines.append("    @%p1 bra DONE;")
    lines.append("")

    # Generate unrolled instructions
    for _ in range(unroll_count):
        if op_type == "fma":
            lines.append("    fma.rn.f32 %f4, %f4, %f2, %f3;")
            lines.append("    fma.rn.f32 %f5, %f5, %f2, %f3;")
        elif op_type == "fadd":
            lines.append("    add.f32 %f4, %f4, %f3;")
            lines.append("    add.f32 %f5, %f5, %f3;")
        elif op_type == "fmul":
            lines.append("    mul.f32 %f4, %f4, %f2;")
            lines.append("    mul.f32 %f5, %f5, %f2;")
        elif op_type == "rsqrt":
            lines.append("    rsqrt.approx.f32 %f4, %f4;")
            lines.append("    rsqrt.approx.f32 %f5, %f5;")
        elif op_type == "sin":
            lines.append("    sin.approx.f32 %f4, %f4;")
            lines.append("    sin.approx.f32 %f5, %f5;")
        elif op_type == "cos":
            lines.append("    cos.approx.f32 %f4, %f4;")
            lines.append("    cos.approx.f32 %f5, %f5;")
        elif op_type == "lg2":
            lines.append("    lg2.approx.f32 %f4, %f4;")
            lines.append("    lg2.approx.f32 %f5, %f5;")
        elif op_type == "ex2":
            lines.append("    ex2.approx.f32 %f4, %f4;")
            lines.append("    ex2.approx.f32 %f5, %f5;")

    lines.append("")
    lines.append("    add.u32 %r6, %r6, 1;")
    lines.append("    bra LOOP;")
    lines.append("DONE:")
    lines.append("    add.f32 %f1, %f4, %f5;")
    lines.append("    st.global.f32 [%rd3], %f1;")
    lines.append("    ret;")
    lines.append("}")
    return "\n".join(lines)


def build_and_disassemble(op_type):
    ptx_code = generate_throughput_ptx(op_type, 500)
    ptx_file = RESULTS_DIR / f"{op_type}.ptx"
    cubin_file = RESULTS_DIR / f"{op_type}.cubin"
    sass_file = RESULTS_DIR / f"{op_type}_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    subprocess.run([str(PTXAS), "-arch=sm_75", str(ptx_file), "-o", str(cubin_file)], check=True)

    custom_env = os.environ.copy()
    custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
    sass_out = subprocess.check_output([str(CUOBJDUMP), "--dump-sass", str(cubin_file)], env=custom_env, text=True)
    with open(sass_file, "w") as f:
        f.write(sass_out)

    return cubin_file, sass_file


def run_benchmark():
    UNROLL = 500
    ITERS = 100
    TOTAL_OPS = UNROLL * 2 * ITERS # 100,000 instructions per thread

    ops = [
        ("fma", "ALU (FFMA)", "fma.rn.f32"),
        ("fadd", "ALU (FADD)", "add.f32"),
        ("fmul", "ALU (FMUL)", "mul.f32"),
        ("rsqrt", "SFU (MUFU.RSQ)", "rsqrt.approx.f32"),
        ("sin", "SFU (MUFU.SIN)", "sin.approx.f32"),
        ("cos", "SFU (MUFU.COS)", "cos.approx.f32"),
        ("lg2", "SFU (MUFU.LG2)", "lg2.approx.f32"),
        ("ex2", "SFU (MUFU.EX2)", "ex2.approx.f32")
    ]

    print("=" * 75)
    print(" Frontier 3: Turing Special Function Units (SFU) vs ALU Throughput")
    print(f" 262,144 Threads x {TOTAL_OPS:,} Instructions (TU117 / GTX 1650)")
    print("=" * 75)

    cubins = {}
    for key, label, _ in ops:
        c, s = build_and_disassemble(key)
        cubins[key] = c

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

    BLOCKS = 1024
    THREADS = 256
    TOTAL_THREADS = BLOCKS * THREADS
    num_bytes = TOTAL_THREADS * 4

    d_mem = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), "cuMemAlloc")

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    p_mem = ctypes.c_uint64(d_mem.value)
    p_iters = ctypes.c_uint32(ITERS)
    k_args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(p_mem), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_iters), ctypes.c_void_p)
    )

    print(f"{'Instruction / Unit':<24} | {'Time (ms)':<10} | {'Throughput (GInst/s)':<22} | {'Relative to FMA':<15}")
    print("-" * 75)

    results = []
    base_time = 0.0

    for key, label, ptx_inst in ops:
        mod = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(mod), str(cubins[key].resolve()).encode("utf-8")), f"Load {key}")
        func = ctypes.c_void_p()
        func_name = f"Benchmark_{key}"
        check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, func_name.encode("utf-8")), f"GetFunc {func_name}")

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
        total_instructions = TOTAL_THREADS * TOTAL_OPS
        ginst_s = (total_instructions / (avg_ms / 1000.0)) / 1e9

        if key == "fma":
            base_time = avg_ms
            rel_str = "1.00x (Baseline)"
        else:
            ratio = avg_ms / base_time
            rel_str = f"{ratio:.2f}x slower"

        print(f"{label:<24} | {avg_ms:<10.3f} | {ginst_s:<22.2f} | {rel_str:<15}")

        results.append({
            "instruction": label,
            "ptx": ptx_inst,
            "kernel_time_ms": round(avg_ms, 3),
            "throughput_ginst_s": round(ginst_s, 2),
            "slowdown_vs_fma": round(avg_ms / base_time, 2)
        })

    print("=" * 75)
    out_file = RESULTS_DIR / "sfu_throughput.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved SFU throughput benchmark results to: {out_file}\n")


if __name__ == "__main__":
    run_benchmark()
