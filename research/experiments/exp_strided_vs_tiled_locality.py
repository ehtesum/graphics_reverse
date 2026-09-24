import ctypes
import os
import json
import time

# Corner G5: 2D Spatial Locality, Memory Coalescing & Tiled Transposition Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing, 14 SMs, 128-bit GDDR6)

def run_spatial_locality_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernels:
    # Kernel 1: Strided Global Load (measuring the coalescing cliff)
    # Each thread in warp i reads data with stride S
    # Kernel 2: 2D Shared Memory Tiled Load (eliminating strided uncoalesced traffic)
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry strided_load_kernel(
    .param .u64 d_in,
    .param .u64 d_out,
    .param .u32 stride,
    .param .u32 num_elements
)
{
    .reg .u64 %in_ptr, %out_ptr, %addr;
    .reg .u32 %s, %n, %idx, %tidx, %bidx, %bdim;
    .reg .f32 %val, %accum;
    .reg .pred %p_in;

    ld.param.u64 %in_ptr, [d_in];
    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %s, [stride];
    ld.param.u32 %n, [num_elements];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %bdim, %ntid.x;

    // idx = bidx * bdim + tidx
    mad.lo.u32 %idx, %bidx, %bdim, %tidx;

    setp.ge.u32 %p_in, %idx, %n;
    @%p_in bra DONE;

    // Strided memory address: addr = in_ptr + (idx * stride * 4) % (total_bytes)
    mul.lo.u32 %idx, %idx, %s;
    and.b32 %idx, %idx, 0x00FFFFFF; // Wrap within 64 MB
    mul.wide.u32 %addr, %idx, 4;
    add.u64 %addr, %in_ptr, %addr;

    ld.global.f32 %val, [%addr];

    // Store to out to prevent dead code
    mad.lo.u32 %idx, %bidx, %bdim, %tidx;
    mul.wide.u32 %addr, %idx, 4;
    add.u64 %addr, %out_ptr, %addr;
    st.global.f32 [%addr], %val;

DONE:
    ret;
}

// 2D 32x32 Tiled Shared Memory Transposition Kernel
// Reads 32x32 block coalesced, stores to shared memory, reads back transposed coalesced
.visible .entry tiled_2d_transpose_kernel(
    .param .u64 d_in,
    .param .u64 d_out,
    .param .u32 width,
    .param .u32 height
)
{
    .shared .align 4 .f32 smem_tile[1056]; // 33 floats per row to avoid bank conflicts

    .reg .u64 %in_ptr, %out_ptr, %addr;
    .reg .u32 %w, %h, %tx, %ty, %bx, %by;
    .reg .u32 %x_in, %y_in, %in_idx, %smem_idx;
    .reg .u32 %x_out, %y_out, %out_idx;
    .reg .f32 %val;

    ld.param.u64 %in_ptr, [d_in];
    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];

    mov.u32 %tx, %tid.x;
    mov.u32 %ty, %tid.y;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %by, %ctaid.y;

    // x_in = bx * 32 + tx; y_in = by * 32 + ty;
    mad.lo.u32 %x_in, %bx, 32, %tx;
    mad.lo.u32 %y_in, %by, 32, %ty;

    // in_idx = y_in * w + x_in; (Coalesced read!)
    mad.lo.u32 %in_idx, %y_in, %w, %x_in;
    mul.wide.u32 %addr, %in_idx, 4;
    add.u64 %addr, %in_ptr, %addr;
    ld.global.f32 %val, [%addr];

    // Store into shared memory: smem_tile[ty][tx]
    mad.lo.u32 %smem_idx, %ty, 33, %tx;
    mul.wide.u32 %addr, %smem_idx, 4;
    st.shared.f32 [%addr], %val;

    bar.sync 0;

    // Transposed block coordinates
    mad.lo.u32 %x_out, %by, 32, %tx;
    mad.lo.u32 %y_out, %bx, 32, %ty;

    // Read from shared memory transposed: smem_tile[tx][ty]
    mad.lo.u32 %smem_idx, %tx, 33, %ty;
    mul.wide.u32 %addr, %smem_idx, 4;
    ld.shared.f32 %val, [%addr];

    // out_idx = y_out * h + x_out; (Coalesced write!)
    mad.lo.u32 %out_idx, %y_out, %h, %x_out;
    mul.wide.u32 %addr, %out_idx, 4;
    add.u64 %addr, %out_ptr, %addr;
    st.global.f32 [%addr], %val;

    ret;
}
"""
    with open("research/raw/spatial_locality.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\spatial_locality.ptx -o research\raw\spatial_locality.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/spatial_locality.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    k_strided = ctypes.c_void_p()
    k_tiled = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(k_strided), module, b"strided_load_kernel")
    nvcuda.cuModuleGetFunction(ctypes.byref(k_tiled), module, b"tiled_2d_transpose_kernel")

    # Allocate 64 MB buffer (16M floats)
    N_ELEMENTS = 16 * 1024 * 1024
    BUFFER_BYTES = N_ELEMENTS * 4

    d_in = ctypes.c_void_p()
    d_out = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_in), BUFFER_BYTES)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out), BUFFER_BYTES)

    # Initialize buffer with ones
    init_data = (ctypes.c_float * 1024)(*([1.0] * 1024))
    nvcuda.cuMemsetD32_v2(d_in, 0x3f800000, N_ELEMENTS) # 1.0f in hex

    # Create CUDA Events
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_event), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_event), 0)

    # Sweep Strides: 1 (perfect coalescing), 2, 4, 8, 16, 32 (worst case: each thread hits separate line), 64, 128
    strides = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    stride_results = []

    print("Benchmarking Global Strided Load Coalescing Cliff...")
    threads_per_block = 256
    num_blocks = N_ELEMENTS // threads_per_block

    for stride in strides:
        c_din = ctypes.c_uint64(d_in.value)
        c_dout = ctypes.c_uint64(d_out.value)
        c_stride = ctypes.c_uint32(stride)
        c_n = ctypes.c_uint32(N_ELEMENTS)

        args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(c_din), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_stride), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_n), ctypes.c_void_p)
        )

        # Warmup
        nvcuda.cuLaunchKernel(k_strided, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        nvcuda.cuEventRecord(start_event, None)
        for _ in range(10):
            nvcuda.cuLaunchKernel(k_strided, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
        nvcuda.cuEventRecord(stop_event, None)
        nvcuda.cuEventSynchronize(stop_event)

        ms = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_event, stop_event)
        avg_ms = ms.value / 10.0
        # 16M elements * 4 bytes read + 16M elements * 4 bytes written = 128 MB total traffic
        gb_s = (128.0 / (1024.0)) / (avg_ms / 1000.0)

        print(f"Stride: {stride:3d} | Latency: {avg_ms:6.3f} ms | Effective Throughput: {gb_s:6.2f} GB/s")
        stride_results.append({
            "stride": stride,
            "latency_ms": avg_ms,
            "effective_throughput_gb_s": gb_s
        })

    # Benchmark 2D 4096 x 4096 Tiled Shared Memory Transpose
    DIM = 4096
    print(f"\nBenchmarking 2D Tiled Transpose on {DIM}x{DIM} Matrix (16M elements)...")
    c_w = ctypes.c_uint32(DIM)
    c_h = ctypes.c_uint32(DIM)
    tiled_args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(c_din), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_h), ctypes.c_void_p)
    )

    grid_x = DIM // 32
    grid_y = DIM // 32

    # Warmup
    nvcuda.cuLaunchKernel(k_tiled, grid_x, grid_y, 1, 32, 32, 1, 0, None, tiled_args, None)
    nvcuda.cuCtxSynchronize()

    nvcuda.cuEventRecord(start_event, None)
    for _ in range(10):
        nvcuda.cuLaunchKernel(k_tiled, grid_x, grid_y, 1, 32, 32, 1, 0, None, tiled_args, None)
    nvcuda.cuEventRecord(stop_event, None)
    nvcuda.cuEventSynchronize(stop_event)

    nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_event, stop_event)
    avg_tiled_ms = ms.value / 10.0
    tiled_gb_s = (128.0 / 1024.0) / (avg_tiled_ms / 1000.0)
    print(f"2D Tiled Shared Memory Transpose: {avg_tiled_ms:6.3f} ms | Throughput: {tiled_gb_s:6.2f} GB/s")

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "memory": "128-bit GDDR6 @ 192 GB/s peak",
        "stride_coalescing_sweep": stride_results,
        "tiled_2d_transpose": {
            "matrix_dim": f"{DIM}x{DIM}",
            "latency_ms": avg_tiled_ms,
            "throughput_gb_s": tiled_gb_s,
            "speedup_vs_stride32": (stride_results[5]["latency_ms"] / avg_tiled_ms)
        }
    }

    with open("research/raw/spatial_locality_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/spatial_locality_results.json")
    nvcuda.cuMemFree_v2(d_in)
    nvcuda.cuMemFree_v2(d_out)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_spatial_locality_benchmark()
