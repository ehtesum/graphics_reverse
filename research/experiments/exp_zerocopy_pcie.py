#!/usr/bin/env python3
"""
research/experiments/exp_zerocopy_pcie.py
Forensic Characterization: Corner H2 - Zero-Copy Host-Mapped Memory (cudaHostAllocMapped) over PCIe 3.0 x8
Hardware Target: AMD Ryzen 7 4800HS + NVIDIA GeForce GTX 1650 (TU117, SM 7.5), Windows 11 WDDM 2.7

Methodology:
1. Allocate page-locked, device-mapped system RAM using cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP.
2. Obtain device pointer via cuMemHostGetDevicePointer_v2 to bypass DMA staging buffers.
3. Benchmark direct GPU vectorized load (LDG.E.128) and store (STG.E.128) throughput across PCIe 3.0 x8.
4. Compare against local GDDR6 DRAM throughput and explicit DMA cudaMemcpyAsync bandwidth.
"""

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "zerocopy_pcie_results.json"

CU_MEMHOSTALLOC_DEVICEMAP = 0x02


def compile_zerocopy_kernels():
    ptx = """
.version 7.5
.target sm_75
.address_size 64

// Vectorized 128-bit streaming read over PCIe
.visible .entry ZeroCopyRead(
    .param .u64 d_src,
    .param .u64 d_sink,
    .param .u32 num_elements_128
) {
    .reg .u64 %src, %sink, %src_addr, %offset64;
    .reg .u32 %n, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride, %r_offset, %accum;
    .reg .b32 %v0, %v1, %v2, %v3;
    .reg .pred %p0, %p_first;

    ld.param.u64 %src, [d_src];
    ld.param.u64 %sink, [d_sink];
    ld.param.u32 %n, [num_elements_128];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;
    mov.u32 %accum, 0;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;

    $loop:
    setp.ge.u32 %p0, %gidx, %n;
    @%p0 bra $exit;

    // 16 bytes (128-bit) per thread
    shl.b32 %r_offset, %gidx, 4;
    cvt.u64.u32 %offset64, %r_offset;
    add.u64 %src_addr, %src, %offset64;

    ld.global.v4.b32 {%v0, %v1, %v2, %v3}, [%src_addr];
    add.u32 %accum, %accum, %v0;

    add.u32 %gidx, %gidx, %stride;
    bra $loop;

    $exit:
    // Dummy sink write to keep loads live
    setp.eq.u32 %p_first, %tidx, 0;
    @%p_first st.global.u32 [%sink], %accum;
    ret;
}

// Vectorized 128-bit streaming write over PCIe
.visible .entry ZeroCopyWrite(
    .param .u64 d_dst,
    .param .u32 num_elements_128
) {
    .reg .u64 %dst, %dst_addr, %offset64;
    .reg .u32 %n, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride, %r_offset;
    .reg .b32 %v0, %v1, %v2, %v3;
    .reg .pred %p0;

    ld.param.u64 %dst, [d_dst];
    ld.param.u32 %n, [num_elements_128];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;

    mov.b32 %v0, 0x12345678;
    mov.b32 %v1, 0x23456789;
    mov.b32 %v2, 0x3456789a;
    mov.b32 %v3, 0x456789ab;

    $w_loop:
    setp.ge.u32 %p0, %gidx, %n;
    @%p0 bra $w_exit;

    shl.b32 %r_offset, %gidx, 4;
    cvt.u64.u32 %offset64, %r_offset;
    add.u64 %dst_addr, %dst, %offset64;

    st.global.v4.b32 [%dst_addr], {%v0, %v1, %v2, %v3};

    add.u32 %gidx, %gidx, %stride;
    bra $w_loop;

    $w_exit:
    ret;
}
"""
    p_file = RAW_DIR / "zerocopy_pcie.ptx"
    c_file = RAW_DIR / "zerocopy_pcie.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER H2: Zero-Copy Host Memory (cudaHostAllocMapped) over PCIe 3.0 x8")
    print(" Target: AMD Renoir Host RAM <--> NVIDIA TU117 via PCIe 3.0 x8")
    print("=" * 80)

    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"[ERROR] Failed to load nvcuda.dll: {e}")
        sys.exit(1)

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)

    # Check device mapped host memory support
    can_map = ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(can_map), 19, dev) # 19 = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
    print(f"[*] Hardware Host Memory Mapping Supported: {bool(can_map.value)}")

    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    c_file = compile_zerocopy_kernels()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))

    func_read = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_read), mod, b"ZeroCopyRead")
    func_write = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_write), mod, b"ZeroCopyWrite")

    # Allocate 64 MB host pinned mapped memory
    BUFFER_SIZE = 64 * 1024 * 1024
    h_ptr = ctypes.c_void_p()
    cuda.cuMemHostAlloc(ctypes.byref(h_ptr), BUFFER_SIZE, CU_MEMHOSTALLOC_DEVICEMAP)

    # Obtain GPU device pointer for host memory
    d_host_ptr = ctypes.c_uint64()
    cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(d_host_ptr), h_ptr, 0)

    # Allocate a small sink buffer in VRAM
    d_sink = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_sink), 16)

    # Create timing events
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(start_event), 0)
    cuda.cuEventCreate(ctypes.byref(stop_event), 0)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    # Test buffer sizes from 1 MB to 64 MB
    TEST_SIZES_MB = [1, 2, 4, 8, 16, 32, 64]
    results = []

    print(f"{'Size (MB)':<10} | {'Direct Read (ms)':<18} | {'Read BW (GB/s)':<16} | {'Direct Write (ms)':<18} | {'Write BW (GB/s)'}")
    print("-" * 85)

    BLOCK_SIZE = 256
    NUM_BLOCKS = 112 # 14 SMs * 8 blocks/SM

    for sz_mb in TEST_SIZES_MB:
        byte_count = sz_mb * 1024 * 1024
        num_vecs = byte_count // 16

        # --- Benchmark Direct Read over PCIe ---
        p_src = ctypes.c_uint64(d_host_ptr.value)
        p_sink = ctypes.c_uint64(d_sink.value)
        p_num = ctypes.c_uint32(num_vecs)

        args_read = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_src), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_sink), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_num), ctypes.c_void_p),
        )

        # Warmup
        cuda.cuLaunchKernel(func_read, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_read, None)
        cuda.cuStreamSynchronize(stream)

        cuda.cuEventRecord(start_event, stream)
        cuda.cuLaunchKernel(func_read, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_read, None)
        cuda.cuEventRecord(stop_event, stream)
        cuda.cuEventSynchronize(stop_event)

        ms_read = ctypes.c_float()
        cuda.cuEventElapsedTime(ctypes.byref(ms_read), start_event, stop_event)
        read_time_ms = ms_read.value
        read_bw_gbps = (byte_count / 1e9) / (read_time_ms / 1000.0)

        # --- Benchmark Direct Write over PCIe ---
        p_dst = ctypes.c_uint64(d_host_ptr.value)
        args_write = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(p_dst), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_num), ctypes.c_void_p),
        )

        cuda.cuLaunchKernel(func_write, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_write, None)
        cuda.cuStreamSynchronize(stream)

        cuda.cuEventRecord(start_event, stream)
        cuda.cuLaunchKernel(func_write, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_write, None)
        cuda.cuEventRecord(stop_event, stream)
        cuda.cuEventSynchronize(stop_event)

        ms_write = ctypes.c_float()
        cuda.cuEventElapsedTime(ctypes.byref(ms_write), start_event, stop_event)
        write_time_ms = ms_write.value
        write_bw_gbps = (byte_count / 1e9) / (write_time_ms / 1000.0)

        print(f"{sz_mb:<10} | {read_time_ms:<18.3f} | {read_bw_gbps:<16.3f} | {write_time_ms:<18.3f} | {write_bw_gbps:.3f}")

        results.append({
            "size_mb": sz_mb,
            "read_time_ms": round(read_time_ms, 4),
            "read_bw_gbps": round(read_bw_gbps, 4),
            "write_time_ms": round(write_time_ms, 4),
            "write_bw_gbps": round(write_bw_gbps, 4)
        })

    # Summary comparison
    output = {
        "hardware": {
            "gpu": "NVIDIA GeForce GTX 1650 (TU117)",
            "bus": "PCIe 3.0 x8",
            "host_cpu": "AMD Ryzen 7 4800HS",
            "os": "Windows 11 WDDM 2.7"
        },
        "zerocopy_results": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 85)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFreeHost(h_ptr)
    cuda.cuMemFree_v2(d_sink)
    cuda.cuEventDestroy(start_event)
    cuda.cuEventDestroy(stop_event)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
