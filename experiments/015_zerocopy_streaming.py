#!/usr/bin/env python3
"""
experiments/015_zerocopy_streaming.py
Frontier 4: Zero-Copy Host Memory vs Explicit PCIe DMA Streaming (GTX 1650)

Investigates direct GPU accesses to host CPU RAM across PCIe 3.0 via
CU_MEMHOSTALLOC_DEVICEMAP vs local GDDR6 VRAM execution:
  1. Local GDDR6 Execution (Baseline VRAM)
  2. Zero-Copy Host Coalesced Streaming (Sequential PCIe read transactions)
  3. Zero-Copy Host Strided Streaming (Uncoalesced PCIe TLP read overhead)
  4. Explicit Staged DMA Copy + Kernel Compute
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "015_zerocopy_streaming"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from cuda_env import PTXAS, CUOBJDUMP, NVDISASM


def generate_access_ptx():
    return """
.version 7.5
.target sm_75
.address_size 64

.visible .entry ProcessStream(
    .param .u64 param_arr,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_stride
)
{
    .reg .b32 %r<15>;
    .reg .f32 %f<10>;
    .reg .b64 %rd<15>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [param_arr];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r1, [param_n];
    ld.param.u32 %r2, [param_stride];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r6, %r1;
    @%p1 ret;

    // Apply stride to test coalesced (stride=1) vs uncoalesced (stride=32)
    mul.lo.s32 %r7, %r6, %r2;
    rem.u32 %r8, %r7, %r1;

    mul.wide.s32 %rd3, %r8, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    // Simple arithmetic processing
    fma.rn.f32 %f2, %f1, 0f40000000, 0f3F800000; // f2 = f1 * 2.0 + 1.0

    mul.wide.s32 %rd5, %r6, 4;
    add.s64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f2;
    ret;
}
"""


def build_kernel():
    ptx_code = generate_access_ptx()
    ptx_file = RESULTS_DIR / "stream.ptx"
    cubin_file = RESULTS_DIR / "stream.cubin"
    sass_file = RESULTS_DIR / "stream_sass.txt"

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
    print(" Frontier 4: Zero-Copy Host Memory vs Explicit PCIe DMA Streaming")
    print(" 16,777,216 Floats (64 MB Buffer) across PCIe 3.0 Link")
    print("=" * 75)

    cubin_file = build_kernel()

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
    # Enable host mapping flag: CU_CTX_MAP_HOST = 0x08
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0x08, dev), "cuCtxCreate")

    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(mod), str(cubin_file.resolve()).encode("utf-8")), "Load Module")
    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"ProcessStream"), "GetFunc")

    start_evt = ctypes.c_void_p()
    stop_evt = ctypes.c_void_p()
    check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate")
    check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate")

    N = 16 * 1024 * 1024 # 16M floats = 64 MB
    num_bytes = N * 4
    BLOCKS = (N + 255) // 256
    THREADS = 256

    # 1. Allocate Local VRAM buffers
    d_vram_in = ctypes.c_uint64()
    d_vram_out = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_vram_in), num_bytes), "cuMemAlloc in")
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_vram_out), num_bytes), "cuMemAlloc out")

    # 2. Allocate Zero-Copy Host Mapped Buffer
    # CU_MEMHOSTALLOC_DEVICEMAP = 0x02, CU_MEMHOSTALLOC_PORTABLE = 0x01
    h_pinned_ptr = ctypes.c_void_p()
    check(cuda.cuMemHostAlloc(ctypes.byref(h_pinned_ptr), num_bytes, 0x02), "cuMemHostAlloc")
    
    # Get device pointer for host buffer
    d_zerocopy_ptr = ctypes.c_uint64()
    check(cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(d_zerocopy_ptr), h_pinned_ptr, 0), "GetDevicePointer")

    # Initialize host memory with 1.0f
    ctypes.memset(h_pinned_ptr, 0x3F, num_bytes) # approx 1.0f
    check(cuda.cuMemcpyHtoD_v2(d_vram_in, h_pinned_ptr, num_bytes), "Populate VRAM")

    modes = [
        ("1. Pure GDDR6 VRAM (Baseline)", d_vram_in.value, 1, False),
        ("2. Zero-Copy Host (Sequential PCIe)", d_zerocopy_ptr.value, 1, False),
        ("3. Zero-Copy Host (Strided PCIe 32x)", d_zerocopy_ptr.value, 32, False),
        ("4. Staged PCIe DMA Copy + VRAM Compute", d_vram_in.value, 1, True)
    ]

    print(f"{'Access Pattern':<42} | {'Time (ms)':<10} | {'Bandwidth (GB/s)':<16} | {'Slowdown vs VRAM':<15}")
    print("-" * 75)

    results = []
    base_time = 0.0

    for label, in_ptr_val, stride, is_staged in modes:
        p_in = ctypes.c_uint64(in_ptr_val)
        p_out = ctypes.c_uint64(d_vram_out.value)
        n_arg = ctypes.c_uint32(N)
        stride_arg = ctypes.c_uint32(stride)

        k_args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(p_in), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(stride_arg), ctypes.c_void_p)
        )

        times = []
        for _ in range(5):
            check(cuda.cuEventRecord(start_evt, None), "Start")
            if is_staged:
                # Stage explicit DMA HtoD
                check(cuda.cuMemcpyHtoDAsync_v2(d_vram_in, h_pinned_ptr, num_bytes, None), "DMA HtoD")
            check(cuda.cuLaunchKernel(func, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, k_args, None), "Launch")
            check(cuda.cuEventRecord(stop_evt, None), "Stop")
            check(cuda.cuEventSynchronize(stop_evt), "Sync")
            el = ctypes.c_float()
            check(cuda.cuEventElapsedTime(ctypes.byref(el), start_evt, stop_evt), "Elapsed")
            times.append(el.value)

        avg_ms = sum(times) / len(times)
        # Total data: 64 MB read + 64 MB written = 128 MB
        effective_gb_s = ((num_bytes * 2) / (avg_ms / 1000.0)) / 1e9

        if "Baseline" in label:
            base_time = avg_ms
            slowdown = "1.00x (Baseline)"
        else:
            ratio = avg_ms / base_time
            slowdown = f"{ratio:.2f}x slower"

        print(f"{label:<42} | {avg_ms:<10.3f} | {effective_gb_s:<16.2f} | {slowdown:<15}")

        results.append({
            "mode": label,
            "time_ms": round(avg_ms, 3),
            "effective_bandwidth_gb_s": round(effective_gb_s, 2),
            "slowdown_factor": round(avg_ms / base_time, 2)
        })

    print("=" * 75)
    out_file = RESULTS_DIR / "zerocopy_streaming.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved zero-copy streaming results to: {out_file}\n")


if __name__ == "__main__":
    run_benchmark()
