import ctypes
import os
import json
import math

# Corner A3: FlashAttention Online Softmax Rescaling & SFU Overhead
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)

def run_flash_attention_softmax_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernels:
    # 1. Pure FMA baseline (matching tile arithmetic)
    # 2. FlashAttention Online Softmax Tile Update:
    #    Iterates over T tiles of KV tokens (each tile = 32 tokens).
    #    Performs dot-product, intra-warp max reduction, SFU exp2 rescaling,
    #    accumulator scaling, and running sum update.
    # 3. Standard Offline Softmax Tile (requires global memory store/reload of attention scores)
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry flash_attention_online_kernel(
    .param .u64 d_out_cycles,
    .param .u32 num_tiles,
    .param .u32 iters
)
{
    .reg .u64 %out_ptr;
    .reg .u32 %n_tiles, %n_iters, %tidx, %bidx;
    .reg .u64 %clk_start, %clk_end, %clk_diff;
    .reg .f32 %m_prev, %m_new, %d_prev, %d_new;
    .reg .f32 %score, %alpha, %p_val, %out_acc, %scale_log2e;
    .reg .f32 %shfl_val;
    .reg .u32 %it, %t;
    .reg .pred %p_loop, %p_tile, %p_gt;

    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %n_tiles, [num_tiles];
    ld.param.u32 %n_iters, [iters];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    // log2(e) = 1.4426950408889634
    mov.f32 %scale_log2e, 1.44269504;

    bar.sync 0;
    mov.u64 %clk_start, %clock64;

    mov.u32 %it, 0;

LOOP_ITERS:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_BENCH;

    // Initialize running online softmax state
    mov.f32 %m_prev, -10000.0;
    mov.f32 %d_prev, 0.0;
    mov.f32 %out_acc, 0.0;

    mov.u32 %t, 0;

LOOP_TILES:
    setp.ge.u32 %p_tile, %t, %n_tiles;
    @%p_tile bra NEXT_ITER;

    // Simulated attention score for this thread's key token
    cvt.rn.f32.u32 %score, %tidx;
    mul.f32 %score, %score, 0.125; // score = tid * 0.125

    // Local max candidate
    max.f32 %m_new, %m_prev, %score;

    // Intra-warp all-reduce max across 32 lanes
    shfl.sync.bfly.b32 %shfl_val, %m_new, 16, 0x1f, 0xffffffff;
    max.f32 %m_new, %m_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %m_new, 8, 0x1f, 0xffffffff;
    max.f32 %m_new, %m_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %m_new, 4, 0x1f, 0xffffffff;
    max.f32 %m_new, %m_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %m_new, 2, 0x1f, 0xffffffff;
    max.f32 %m_new, %m_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %m_new, 1, 0x1f, 0xffffffff;
    max.f32 %m_new, %m_new, %shfl_val;

    // Compute alpha = exp(m_prev - m_new) = exp2((m_prev - m_new) * log2e)
    sub.f32 %alpha, %m_prev, %m_new;
    mul.f32 %alpha, %alpha, %scale_log2e;
    ex2.approx.f32 %alpha, %alpha; // SFU base-2 exponential

    // Compute P_val = exp(score - m_new) = exp2((score - m_new) * log2e)
    sub.f32 %p_val, %score, %m_new;
    mul.f32 %p_val, %p_val, %scale_log2e;
    ex2.approx.f32 %p_val, %p_val; // SFU base-2 exponential

    // Update denominator: d_new = d_prev * alpha + P_val
    fma.rn.f32 %d_new, %d_prev, %alpha, %p_val;

    // Intra-warp sum reduction of d_new
    shfl.sync.bfly.b32 %shfl_val, %d_new, 16, 0x1f, 0xffffffff;
    add.f32 %d_new, %d_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %d_new, 8, 0x1f, 0xffffffff;
    add.f32 %d_new, %d_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %d_new, 4, 0x1f, 0xffffffff;
    add.f32 %d_new, %d_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %d_new, 2, 0x1f, 0xffffffff;
    add.f32 %d_new, %d_new, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %d_new, 1, 0x1f, 0xffffffff;
    add.f32 %d_new, %d_new, %shfl_val;

    // Update output accumulator: out_acc = out_acc * alpha + P_val * V_val (simulated V=1.0f)
    fma.rn.f32 %out_acc, %out_acc, %alpha, %p_val;

    // Commit running state for next tile
    mov.f32 %m_prev, %m_new;
    mov.f32 %d_prev, %d_new;

    add.u32 %t, %t, 1;
    bra LOOP_TILES;

NEXT_ITER:
    add.u32 %it, %it, 1;
    bra LOOP_ITERS;

DONE_BENCH:
    mov.u64 %clk_end, %clock64;
    sub.u64 %clk_diff, %clk_end, %clk_start;

    // Store cycles from thread 0, and store out_acc to prevent dead-code removal
    setp.eq.u32 %p_gt, %tidx, 0;
    @%p_gt st.global.u64 [%out_ptr], %clk_diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_gt st.global.f32 [%out_ptr], %out_acc;

    ret;
}

// Baseline FMA kernel (same arithmetic count without SFU and without shuffles)
.visible .entry fma_baseline_kernel(
    .param .u64 d_out_cycles,
    .param .u32 num_tiles,
    .param .u32 iters
)
{
    .reg .u64 %out_ptr;
    .reg .u32 %n_tiles, %n_iters, %tidx;
    .reg .u64 %clk_start, %clk_end, %clk_diff;
    .reg .f32 %a, %b, %c;
    .reg .u32 %it, %t;
    .reg .pred %p_loop, %p_tile;

    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %n_tiles, [num_tiles];
    ld.param.u32 %n_iters, [iters];

    mov.u32 %tidx, %tid.x;

    bar.sync 0;
    mov.u64 %clk_start, %clock64;

    mov.u32 %it, 0;
    mov.f32 %a, 1.0001;
    mov.f32 %b, 1.0002;
    mov.f32 %c, 0.0;

LOOP_ITERS_BASE:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_BASE;

    mov.u32 %t, 0;

LOOP_TILES_BASE:
    setp.ge.u32 %p_tile, %t, %n_tiles;
    @%p_tile bra NEXT_ITER_BASE;

    // Execute 16 chained FMAs (matching tile instruction count)
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;
    fma.rn.f32 %c, %c, %a, %b;

    add.u32 %t, %t, 1;
    bra LOOP_TILES_BASE;

NEXT_ITER_BASE:
    add.u32 %it, %it, 1;
    bra LOOP_ITERS_BASE;

DONE_BASE:
    mov.u64 %clk_end, %clock64;
    sub.u64 %clk_diff, %clk_end, %clk_start;

    setp.eq.u32 %p_tile, %tidx, 0;
    @%p_tile st.global.u64 [%out_ptr], %clk_diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_tile st.global.f32 [%out_ptr], %c;

    ret;
}
"""
    with open("research/raw/flash_attn_softmax.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\flash_attn_softmax.ptx -o research\raw\flash_attn_softmax.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/flash_attn_softmax.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    k_flash = ctypes.c_void_p()
    k_base = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(k_flash), module, b"flash_attention_online_kernel")
    nvcuda.cuModuleGetFunction(ctypes.byref(k_base), module, b"fma_baseline_kernel")

    d_out_cycles = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out_cycles), 16)

    # Test sweep over sequence lengths: L = 128, 512, 1024, 2048, 4096 tokens
    # Tile size = 32 tokens (1 warp)
    tile_counts = [4, 16, 32, 64, 128] # L = 128 to 4096
    iters = 1000

    results = []
    print("Benchmarking FlashAttention Online Softmax Tile Rescaling & SFU Overhead...")

    for n_tiles in tile_counts:
        seq_len = n_tiles * 32

        c_out = ctypes.c_uint64(d_out_cycles.value)
        c_tiles = ctypes.c_uint32(n_tiles)
        c_iters = ctypes.c_uint32(iters)

        args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(c_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_tiles), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p)
        )

        # 1. Run FlashAttention Online Softmax
        nvcuda.cuLaunchKernel(k_flash, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        flash_cyc = ctypes.c_uint64()
        nvcuda.cuMemcpyDtoH_v2(ctypes.byref(flash_cyc), d_out_cycles, 8)
        avg_flash_cycles = flash_cyc.value / iters
        cycles_per_tile = avg_flash_cycles / n_tiles

        # 2. Run Baseline FMA
        nvcuda.cuLaunchKernel(k_base, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        base_cyc = ctypes.c_uint64()
        nvcuda.cuMemcpyDtoH_v2(ctypes.byref(base_cyc), d_out_cycles, 8)
        avg_base_cycles = base_cyc.value / iters
        base_cycles_per_tile = avg_base_cycles / n_tiles

        sfu_overhead_ratio = avg_flash_cycles / max(avg_base_cycles, 1.0)

        print(f"SeqLen: {seq_len:4d} tokens ({n_tiles:3d} tiles) | "
              f"FlashAttention: {avg_flash_cycles:7.1f} cycles ({cycles_per_tile:5.2f} cyc/tile) | "
              f"FMA Baseline: {avg_base_cycles:7.1f} cycles | "
              f"SFU/Reduction Ratio: {sfu_overhead_ratio:4.2f}x")

        results.append({
            "sequence_length": seq_len,
            "num_tiles_32": n_tiles,
            "total_cycles_per_query": avg_flash_cycles,
            "cycles_per_tile": cycles_per_tile,
            "fma_baseline_cycles": avg_base_cycles,
            "sfu_reduction_overhead_ratio": sfu_overhead_ratio
        })

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "benchmark": "FlashAttention Online Softmax Rescaling & SFU Overhead",
        "warp_size": 32,
        "results": results
    }

    with open("research/raw/flash_attention_softmax_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/flash_attention_softmax_results.json")
    nvcuda.cuMemFree_v2(d_out_cycles)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_flash_attention_softmax_benchmark()
