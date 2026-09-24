#!/usr/bin/env python3
"""
research/experiments/exp_paged_attention_kv.py
Forensic Characterization: Corner A2 - PagedAttention KV-Cache Strided Traversal Overhead
Hardware Target: NVIDIA GeForce GTX 1650 (TU117, SM 7.5 Turing, 4 GB GDDR6)

Methodology:
1. Simulate KV-cache attention reading during LLM autoregressive decode:
   - Head count = 16, Head dimension = 64 (FP16 elements -> 128 bytes per head vector).
   - Sequence lengths L in [128, 512, 1024, 2048, 4096] tokens.
2. Compare:
   a) Contiguous KV-Cache: Linear physical memory layout (ideal sequential GDDR6 burst)
   b) Paged KV-Cache: Non-contiguous block-allocated pages (16 tokens per block) with
      block table indirection and randomized physical block mapping.
3. Quantify effective memory bandwidth (GB/s), TLB/cache miss degradation, and indirection tax.
"""

import ctypes
import json
import subprocess
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CUDA_BIN = ROOT / "cuda_toolkit" / "bin"
PTXAS = CUDA_BIN / "ptxas.exe"
OUT_FILE = RAW_DIR / "paged_attention_kv_results.json"


def compile_kv_kernels():
    ptx = """
.version 7.5
.target sm_75
.address_size 64

// 1. Contiguous KV Cache Traversal Kernel
.visible .entry ContiguousKVRead(
    .param .u64 d_kv,
    .param .u64 d_out,
    .param .u32 total_vectors_128b
) {
    .reg .u64 %kv, %out, %kv_addr;
    .reg .u32 %n, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride, %r_offset;
    .reg .u64 %offset64;
    .reg .b32 %v0, %v1, %v2, %v3, %accum;
    .reg .pred %p0;

    ld.param.u64 %kv, [d_kv];
    ld.param.u64 %out, [d_out];
    ld.param.u32 %n, [total_vectors_128b];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;
    mov.b32 %accum, 0;

    $loop_c:
    setp.ge.u32 %p0, %gidx, %n;
    @%p0 bra $exit_c;

    shl.b32 %r_offset, %gidx, 4; // 16 bytes per vector (128-bit)
    cvt.u64.u32 %offset64, %r_offset;
    add.u64 %kv_addr, %kv, %offset64;

    ld.global.v4.b32 {%v0, %v1, %v2, %v3}, [%kv_addr];
    add.u32 %accum, %accum, %v0;

    add.u32 %gidx, %gidx, %stride;
    bra $loop_c;

    $exit_c:
    setp.eq.u32 %p0, %tidx, 0;
    @%p0 st.global.b32 [%out], %accum;
    ret;
}

// 2. Paged KV Cache Traversal Kernel (Block table indirection)
.visible .entry PagedKVRead(
    .param .u64 d_pool,
    .param .u64 d_block_table,
    .param .u64 d_out,
    .param .u32 total_vectors_128b,
    .param .u32 vecs_per_block
) {
    .reg .u64 %pool, %table, %out, %tbl_addr, %phys_addr, %offset64;
    .reg .u32 %n, %v_per_blk, %tidx, %ntidx, %bidx, %nbidx, %gidx, %stride;
    .reg .u32 %blk_idx, %blk_offset, %phys_blk_id, %r_offset;
    .reg .b32 %v0, %v1, %v2, %v3, %accum;
    .reg .pred %p0;

    ld.param.u64 %pool, [d_pool];
    ld.param.u64 %table, [d_block_table];
    ld.param.u64 %out, [d_out];
    ld.param.u32 %n, [total_vectors_128b];
    ld.param.u32 %v_per_blk, [vecs_per_block];

    mov.u32 %tidx, %tid.x;
    mov.u32 %ntidx, %ntid.x;
    mov.u32 %bidx, %ctaid.x;
    mov.u32 %nbidx, %nctaid.x;

    mad.lo.u32 %gidx, %bidx, %ntidx, %tidx;
    mul.lo.u32 %stride, %nbidx, %ntidx;
    mov.b32 %accum, 0;

    $loop_p:
    setp.ge.u32 %p0, %gidx, %n;
    @%p0 bra $exit_p;

    // Logical token vector -> block index & offset
    div.u32 %blk_idx, %gidx, %v_per_blk;
    rem.u32 %blk_offset, %gidx, %v_per_blk;

    // Indirection: read physical block ID from block table
    shl.b32 %r_offset, %blk_idx, 2; // 4 bytes per table entry
    cvt.u64.u32 %offset64, %r_offset;
    add.u64 %tbl_addr, %table, %offset64;
    ld.global.u32 %phys_blk_id, [%tbl_addr];

    // Compute physical address in pool:
    // byte_offset = (phys_blk_id * vecs_per_block + blk_offset) * 16 bytes
    mad.lo.u32 %r_offset, %phys_blk_id, %v_per_blk, %blk_offset;
    shl.b32 %r_offset, %r_offset, 4;
    cvt.u64.u32 %offset64, %r_offset;
    add.u64 %phys_addr, %pool, %offset64;

    ld.global.v4.b32 {%v0, %v1, %v2, %v3}, [%phys_addr];
    add.u32 %accum, %accum, %v0;

    add.u32 %gidx, %gidx, %stride;
    bra $loop_p;

    $exit_p:
    setp.eq.u32 %p0, %tidx, 0;
    @%p0 st.global.b32 [%out], %accum;
    ret;
}
"""
    p_file = RAW_DIR / "paged_kv.ptx"
    c_file = RAW_DIR / "paged_kv.cubin"
    with open(p_file, "w") as f:
        f.write(ptx)
    subprocess.run([str(PTXAS), "-arch=sm_75", str(p_file), "-o", str(c_file)], check=True)
    return c_file


def run_experiment():
    print("=" * 80)
    print(" CORNER A2: PagedAttention KV-Cache Strided Memory Traversal Overhead")
    print(" Target: NVIDIA GeForce GTX 1650 (TU117 Turing) | 4 GB GDDR6")
    print("=" * 80)

    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"[ERROR] Failed to load nvcuda.dll: {e}")
        sys.exit(1)

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    c_file = compile_kv_kernels()
    mod = ctypes.c_void_p()
    cuda.cuModuleLoad(ctypes.byref(mod), str(c_file.resolve()).encode("utf-8"))

    func_contig = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_contig), mod, b"ContiguousKVRead")
    func_paged = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func_paged), mod, b"PagedKVRead")

    # Sequence lengths: 256, 512, 1024, 2048, 4096, 8192 tokens
    # Each token has 16 heads * 64 FP16 elements = 2,048 bytes (128 vectors of 128-bit)
    SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
    VECS_PER_TOKEN = 128 # 2048 bytes / 16 bytes
    TOKENS_PER_BLOCK = 16 # Standard vLLM block size
    VECS_PER_BLOCK = TOKENS_PER_BLOCK * VECS_PER_TOKEN # 2048 vectors (32 KB per block)

    # Maximum allocation for L=8192: 8192 * 2048 bytes = 16.78 MB
    MAX_BYTES = 8192 * 2048
    MAX_VECS = MAX_BYTES // 16

    d_kv = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_kv), MAX_BYTES)

    d_pool = ctypes.c_uint64()
    # Allocate a 2x oversized pool to enable randomized block fragmentation
    POOL_BYTES = MAX_BYTES * 2
    cuda.cuMemAlloc_v2(ctypes.byref(d_pool), POOL_BYTES)

    # Max blocks = 8192 // 16 = 512 blocks
    MAX_BLOCKS = 8192 // TOKENS_PER_BLOCK
    d_table = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_table), MAX_BLOCKS * 4)

    # Create randomized block mapping
    np.random.seed(42)
    pool_total_blocks = POOL_BYTES // (TOKENS_PER_BLOCK * 2048)
    shuffled_blocks = np.random.choice(pool_total_blocks, MAX_BLOCKS, replace=False).astype(np.uint32)
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2(d_table.value, ctypes.c_void_p(shuffled_blocks.ctypes.data), MAX_BLOCKS * 4)

    d_out = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)

    ev_start = ctypes.c_void_p()
    ev_stop = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(ev_start), 0)
    cuda.cuEventCreate(ctypes.byref(ev_stop), 0)

    stream = ctypes.c_void_p()
    cuda.cuStreamCreate(ctypes.byref(stream), 0)

    BLOCK_SIZE = 256
    NUM_BLOCKS = 112

    results = []

    print(f"{'Seq Length':<12} | {'KV Size (MB)':<14} | {'Contig Time (us)':<18} | {'Contig BW (GB/s)':<18} | {'Paged Time (us)':<16} | {'Paged BW (GB/s)':<16} | {'Tax'}")
    print("-" * 110)

    for seq_len in SEQ_LENGTHS:
        total_vecs = seq_len * VECS_PER_TOKEN
        total_bytes = seq_len * 2048
        mb = total_bytes / 1e6

        # --- Benchmark Contiguous ---
        p_kv = ctypes.c_uint64(d_kv.value)
        p_out = ctypes.c_uint64(d_out.value)
        p_n = ctypes.c_uint32(total_vecs)

        args_c = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(p_kv), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_n), ctypes.c_void_p),
        )

        cuda.cuLaunchKernel(func_contig, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_c, None)
        cuda.cuStreamSynchronize(stream)

        ITERS = 50
        cuda.cuEventRecord(ev_start, stream)
        for _ in range(ITERS):
            cuda.cuLaunchKernel(func_contig, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_c, None)
        cuda.cuEventRecord(ev_stop, stream)
        cuda.cuEventSynchronize(ev_stop)

        ms_c = ctypes.c_float()
        cuda.cuEventElapsedTime(ctypes.byref(ms_c), ev_start, ev_stop)
        us_c = (ms_c.value / ITERS) * 1000.0
        bw_c = (total_bytes / 1e9) / (us_c / 1e6)

        # --- Benchmark Paged ---
        p_pool = ctypes.c_uint64(d_pool.value)
        p_tbl = ctypes.c_uint64(d_table.value)
        p_vpb = ctypes.c_uint32(VECS_PER_BLOCK)

        args_p = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(p_pool), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_tbl), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_n), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_vpb), ctypes.c_void_p),
        )

        cuda.cuLaunchKernel(func_paged, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_p, None)
        cuda.cuStreamSynchronize(stream)

        cuda.cuEventRecord(ev_start, stream)
        for _ in range(ITERS):
            cuda.cuLaunchKernel(func_paged, NUM_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args_p, None)
        cuda.cuEventRecord(ev_stop, stream)
        cuda.cuEventSynchronize(ev_stop)

        ms_p = ctypes.c_float()
        cuda.cuEventElapsedTime(ctypes.byref(ms_p), ev_start, ev_stop)
        us_p = (ms_p.value / ITERS) * 1000.0
        bw_p = (total_bytes / 1e9) / (us_p / 1e6)

        tax = us_p / us_c

        print(f"{seq_len:<12} | {mb:<14.2f} | {us_c:<18.2f} | {bw_c:<18.2f} | {us_p:<16.2f} | {bw_p:<16.2f} | {tax:.2f}x")

        results.append({
            "seq_length": seq_len,
            "kv_bytes": total_bytes,
            "contiguous_us": round(us_c, 2),
            "contiguous_bw_gbps": round(bw_c, 2),
            "paged_us": round(us_p, 2),
            "paged_bw_gbps": round(bw_p, 2),
            "indirection_tax_ratio": round(tax, 2)
        })

    output = {
        "hardware": "NVIDIA GeForce GTX 1650 (TU117)",
        "tokens_per_block": TOKENS_PER_BLOCK,
        "results": results
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 110)
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")

    cuda.cuMemFree_v2(d_kv)
    cuda.cuMemFree_v2(d_pool)
    cuda.cuMemFree_v2(d_table)
    cuda.cuMemFree_v2(d_out)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(mod)


if __name__ == "__main__":
    run_experiment()
