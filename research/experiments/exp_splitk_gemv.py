import ctypes
import os
import json
import time

# Corner A5: Split-K Batched GEMV for Small Batch LLM Decoding
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 14 SMs)

def run_splitk_gemv_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernel: Split-K GEMV
    # Matrix W is M x K (FP16 or FP32). Here we test FP32 (M=4096, K=4096 = 64 MB).
    # Grid dimensions: (M, num_splits)
    # Each block (row m, split s) accumulates dot product over K/num_splits elements
    # and atomically adds to out[m].
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry splitk_gemv_kernel(
    .param .u64 d_W,
    .param .u64 d_x,
    .param .u64 d_y,
    .param .u32 M,
    .param .u32 K,
    .param .u32 num_splits
)
{
    .reg .u64 %w_ptr, %x_ptr, %y_ptr, %w_addr, %x_addr, %y_addr;
    .reg .u32 %m_dim, %k_dim, %n_splits;
    .reg .u32 %tidx, %row, %split_id, %chunk_k, %k_start, %k_end, %k, %w_idx;
    .reg .f32 %w_val, %x_val, %partial_sum, %shfl_val;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %w_ptr, [d_W];
    ld.param.u64 %x_ptr, [d_x];
    ld.param.u64 %y_ptr, [d_y];
    ld.param.u32 %m_dim, [M];
    ld.param.u32 %k_dim, [K];
    ld.param.u32 %n_splits, [num_splits];

    mov.u32 %tidx, %tid.x;
    mov.u32 %row, %ctaid.x;
    mov.u32 %split_id, %ctaid.y;

    // chunk_k = K / num_splits
    div.u32 %chunk_k, %k_dim, %n_splits;
    mul.lo.u32 %k_start, %split_id, %chunk_k;
    add.u32 %k_end, %k_start, %chunk_k;

    // Strided accumulation by 32 (warp size)
    add.u32 %k, %k_start, %tidx;
    mov.f32 %partial_sum, 0.0;

LOOP_K:
    setp.ge.u32 %p_loop, %k, %k_end;
    @%p_loop bra DONE_K;

    // Address of W[row, k]: (row * K + k) * 4
    mad.lo.u32 %w_idx, %row, %k_dim, %k;
    mul.wide.u32 %w_addr, %w_idx, 4;
    add.u64 %w_addr, %w_ptr, %w_addr;
    ld.global.f32 %w_val, [%w_addr];

    // Address of x[k]: k * 4
    mul.wide.u32 %x_addr, %k, 4;
    add.u64 %x_addr, %x_ptr, %x_addr;
    ld.global.f32 %x_val, [%x_addr];

    fma.rn.f32 %partial_sum, %w_val, %x_val, %partial_sum;

    add.u32 %k, %k, 32;
    bra LOOP_K;

DONE_K:
    // Intra-warp butterfly reduction of partial_sum
    shfl.sync.bfly.b32 %shfl_val, %partial_sum, 16, 0x1f, 0xffffffff;
    add.f32 %partial_sum, %partial_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %partial_sum, 8, 0x1f, 0xffffffff;
    add.f32 %partial_sum, %partial_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %partial_sum, 4, 0x1f, 0xffffffff;
    add.f32 %partial_sum, %partial_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %partial_sum, 2, 0x1f, 0xffffffff;
    add.f32 %partial_sum, %partial_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %partial_sum, 1, 0x1f, 0xffffffff;
    add.f32 %partial_sum, %partial_sum, %shfl_val;

    // Thread 0 atomically adds to y[row]
    setp.eq.u32 %p_lead, %tidx, 0;
    @!%p_lead bra EXIT_SPLITK;

    mul.wide.u32 %y_addr, %row, 4;
    add.u64 %y_addr, %y_ptr, %y_addr;
    atom.global.add.f32 %shfl_val, [%y_addr], %partial_sum;

EXIT_SPLITK:
    ret;
}
"""
    with open("research/raw/splitk_gemv.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\splitk_gemv.ptx -o research\raw\splitk_gemv.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/splitk_gemv.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    k_splitk = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(k_splitk), module, b"splitk_gemv_kernel")

    # Dimensions: M=4096, K=4096 (Standard 7B/8B model attention/FFN projection)
    M = 4096
    K = 4096
    W_BYTES = M * K * 4 # 64 MB
    X_BYTES = K * 4     # 16 KB
    Y_BYTES = M * 4     # 16 KB

    d_W = ctypes.c_void_p()
    d_x = ctypes.c_void_p()
    d_y = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_W), W_BYTES)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_x), X_BYTES)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_y), Y_BYTES)

    nvcuda.cuMemsetD32_v2(d_W, 0x3f800000, M * K)
    nvcuda.cuMemsetD32_v2(d_x, 0x3f800000, K)

    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_ev), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_ev), 0)

    # Test Split-K factors: S=1 (standard GEMV), 2, 4, 8, 16
    split_factors = [1, 2, 4, 8, 16]
    results = []

    print("Benchmarking Split-K GEMV Scaling on 14 SMs (Matrix 4096 x 4096)...")
    print(f"{'Split-K Factor':<16} | {'Grid (M x S)':<16} | {'Latency (ms)':<14} | {'Throughput (GB/s)':<18} | {'GFLOPS'}")
    print("-" * 78)

    for S in split_factors:
        # Reset output y to 0
        nvcuda.cuMemsetD32_v2(d_y, 0, M)

        c_dW = ctypes.c_uint64(d_W.value)
        c_dx = ctypes.c_uint64(d_x.value)
        c_dy = ctypes.c_uint64(d_y.value)
        c_M = ctypes.c_uint32(M)
        c_K = ctypes.c_uint32(K)
        c_S = ctypes.c_uint32(S)

        args = (ctypes.c_void_p * 6)(
            ctypes.cast(ctypes.byref(c_dW), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dx), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dy), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_S), ctypes.c_void_p)
        )

        threads = 32
        grid_x = M
        grid_y = S

        # Warmup
        nvcuda.cuLaunchKernel(k_splitk, grid_x, grid_y, 1, threads, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        nvcuda.cuEventRecord(start_ev, None)
        for _ in range(10):
            nvcuda.cuLaunchKernel(k_splitk, grid_x, grid_y, 1, threads, 1, 1, 0, None, args, None)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)

        ms = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
        avg_ms = ms.value / 10.0

        # Memory traffic: W is read once (64 MB)
        gb_s = (W_BYTES / 1e9) / (avg_ms / 1000.0)
        # FLOPs: 2 * M * K
        gflops = (2.0 * M * K / 1e9) / (avg_ms / 1000.0)

        grid_str = f"{grid_x} x {grid_y}"
        print(f"S = {S:<12d} | {grid_str:<16} | {avg_ms:6.3f} ms      | {gb_s:6.2f} GB/s         | {gflops:6.1f}")

        results.append({
            "split_k_factor": S,
            "grid_m": grid_x,
            "grid_s": grid_y,
            "latency_ms": avg_ms,
            "bandwidth_gb_s": gb_s,
            "gflops": gflops
        })

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 14 SMs)",
        "matrix": "4096 x 4096 (FP32, 64 MB)",
        "results": results
    }

    with open("research/raw/splitk_gemv_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/splitk_gemv_results.json")
    nvcuda.cuMemFree_v2(d_W)
    nvcuda.cuMemFree_v2(d_x)
    nvcuda.cuMemFree_v2(d_y)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_splitk_gemv_benchmark()
