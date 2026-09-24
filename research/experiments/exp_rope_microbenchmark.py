import ctypes
import os
import json
import math

# Corner A4: Rotary Position Embedding (RoPE) Fused Register vs Memory Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 128-bit GDDR6)

def run_rope_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernels:
    # Kernel 1: Unfused Memory RoPE
    # Loads [x0, x1] from DRAM, loads [cos, sin], applies rotation, stores back to DRAM
    # Kernel 2: Fused In-Register RoPE (simulates fused QK projection + RoPE)
    # Operates directly in registers, bypassing DRAM write-back and reload
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry unfused_rope_kernel(
    .param .u64 d_q,
    .param .u64 d_cos_sin,
    .param .u32 num_tokens,
    .param .u32 head_dim
)
{
    .reg .u64 %q_ptr, %cs_ptr, %q_addr, %cs_addr;
    .reg .u32 %n_tok, %h_dim, %tidx, %bidx, %idx, %pair_idx;
    .reg .f32 %x0, %x1, %c, %s, %negs, %out0, %out1;
    .reg .pred %p_valid;

    ld.param.u64 %q_ptr, [d_q];
    ld.param.u64 %cs_ptr, [d_cos_sin];
    ld.param.u32 %n_tok, [num_tokens];
    ld.param.u32 %h_dim, [head_dim];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    // Each thread processes one pair of dimensions (e.g. 64 dim = 32 pairs)
    // idx = bidx * 32 + tidx;
    mad.lo.u32 %idx, %bidx, 32, %tidx;
    shl.b32 %pair_idx, %idx, 1; // 2 * idx

    mul.lo.u32 %n_tok, %n_tok, %h_dim;
    setp.ge.u32 %p_valid, %pair_idx, %n_tok;
    @%p_valid bra DONE_UNFUSED;

    // Load x0, x1
    mul.wide.u32 %q_addr, %pair_idx, 4;
    add.u64 %q_addr, %q_ptr, %q_addr;
    ld.global.f32 %x0, [%q_addr];
    ld.global.f32 %x1, [%q_addr + 4];

    // Load cos, sin
    mul.wide.u32 %cs_addr, %pair_idx, 4;
    add.u64 %cs_addr, %cs_ptr, %cs_addr;
    ld.global.f32 %c, [%cs_addr];
    ld.global.f32 %s, [%cs_addr + 4];

    // out0 = x0 * cos - x1 * sin
    // out1 = x0 * sin + x1 * cos
    neg.f32 %negs, %s;
    mul.f32 %out0, %x0, %c;
    fma.rn.f32 %out0, %x1, %negs, %out0;

    mul.f32 %out1, %x0, %s;
    fma.rn.f32 %out1, %x1, %c, %out1;

    // Store back to DRAM
    st.global.f32 [%q_addr], %out0;
    st.global.f32 [%q_addr + 4], %out1;

DONE_UNFUSED:
    ret;
}

.visible .entry fused_register_rope_kernel(
    .param .u64 d_q,
    .param .u64 d_cos_sin,
    .param .u32 num_tokens,
    .param .u32 head_dim
)
{
    .reg .u64 %q_ptr, %cs_ptr, %q_addr, %cs_addr;
    .reg .u32 %n_tok, %h_dim, %tidx, %bidx, %idx, %pair_idx;
    .reg .f32 %x0, %x1, %c, %s, %negs, %out0, %out1;
    .reg .pred %p_valid;

    ld.param.u64 %q_ptr, [d_q];
    ld.param.u64 %cs_ptr, [d_cos_sin];
    ld.param.u32 %n_tok, [num_tokens];
    ld.param.u32 %h_dim, [head_dim];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    mad.lo.u32 %idx, %bidx, 32, %tidx;
    shl.b32 %pair_idx, %idx, 1;

    mul.lo.u32 %n_tok, %n_tok, %h_dim;
    setp.ge.u32 %p_valid, %pair_idx, %n_tok;
    @%p_valid bra DONE_FUSED;

    // Simulate in-register data (already present in registers from GEMV)
    cvt.rn.f32.u32 %x0, %tidx;
    cvt.rn.f32.u32 %x1, %bidx;

    // Cos and Sin can be held in shared memory or registers
    mov.f32 %c, 0.9998;
    mov.f32 %s, 0.0174;

    // RoPE rotation directly in registers
    neg.f32 %negs, %s;
    mul.f32 %out0, %x0, %c;
    fma.rn.f32 %out0, %x1, %negs, %out0;

    mul.f32 %out1, %x0, %s;
    fma.rn.f32 %out1, %x1, %c, %out1;

    // Only final attention consumer touches it (no extra DRAM roundtrip)
    // We store to prevent dead code removal
    mul.wide.u32 %q_addr, %pair_idx, 4;
    add.u64 %q_addr, %q_ptr, %q_addr;
    st.global.f32 [%q_addr], %out0;
    st.global.f32 [%q_addr + 4], %out1;

DONE_FUSED:
    ret;
}
"""
    with open("research/raw/rope_bench.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\rope_bench.ptx -o research\raw\rope_bench.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/rope_bench.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    k_unfused = ctypes.c_void_p()
    k_fused = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(k_unfused), module, b"unfused_rope_kernel")
    nvcuda.cuModuleGetFunction(ctypes.byref(k_fused), module, b"fused_register_rope_kernel")

    # Sequence lengths: 512, 1024, 2048, 4096, 8192 tokens
    # Model config: 32 attention heads, D=64 (Head Dim), Total Dim per token = 2048 floats = 8192 bytes
    NUM_HEADS = 32
    HEAD_DIM = 64
    seq_lens = [512, 1024, 2048, 4096, 8192]

    # Create CUDA Events
    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_ev), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_ev), 0)

    results = []
    print("Benchmarking RoPE (Rotary Position Embedding): Unfused Memory vs Fused Register...")

    for seq_len in seq_lens:
        total_tokens = seq_len * NUM_HEADS # e.g. 512 * 32 = 16,384 vectors
        num_floats = total_tokens * HEAD_DIM
        total_bytes = num_floats * 4

        d_q = ctypes.c_void_p()
        d_cs = ctypes.c_void_p()
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_q), total_bytes)
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_cs), total_bytes)

        nvcuda.cuMemsetD32_v2(d_q, 0x3f800000, num_floats)
        nvcuda.cuMemsetD32_v2(d_cs, 0x3f800000, num_floats)

        c_dq = ctypes.c_uint64(d_q.value)
        c_dcs = ctypes.c_uint64(d_cs.value)
        c_ntok = ctypes.c_uint32(total_tokens)
        c_hdim = ctypes.c_uint32(HEAD_DIM)

        args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(c_dq), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dcs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ntok), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_hdim), ctypes.c_void_p)
        )

        # 32 pairs per head -> 32 threads per block
        threads_per_block = 32
        num_blocks = total_tokens

        # Warmup
        nvcuda.cuLaunchKernel(k_unfused, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        # 1. Unfused Memory RoPE
        nvcuda.cuEventRecord(start_ev, None)
        for _ in range(20):
            nvcuda.cuLaunchKernel(k_unfused, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)

        ms = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
        unfused_ms = ms.value / 20.0

        # 2. Fused In-Register RoPE
        nvcuda.cuEventRecord(start_ev, None)
        for _ in range(20):
            nvcuda.cuLaunchKernel(k_fused, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)

        nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
        fused_ms = ms.value / 20.0

        speedup = unfused_ms / fused_ms
        # Unfused traffic: Read Q + Read CosSin + Write Q = 3 * total_bytes
        # Fused traffic: Write Q only = 1 * total_bytes
        unfused_bw = (3.0 * total_bytes / 1e9) / (unfused_ms / 1000.0)
        fused_bw = (1.0 * total_bytes / 1e9) / (fused_ms / 1000.0)

        print(f"SeqLen: {seq_len:4d} | Unfused: {unfused_ms*1000:6.1f} µs ({unfused_bw:5.1f} GB/s) | "
              f"Fused: {fused_ms*1000:6.1f} µs ({fused_bw:5.1f} GB/s) | Speedup: {speedup:4.2f}x")

        results.append({
            "seq_len": seq_len,
            "total_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "unfused_ms": unfused_ms,
            "unfused_bandwidth_gb_s": unfused_bw,
            "fused_ms": fused_ms,
            "fused_bandwidth_gb_s": fused_bw,
            "speedup": speedup
        })

        nvcuda.cuMemFree_v2(d_q)
        nvcuda.cuMemFree_v2(d_cs)

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "benchmark": "Rotary Position Embedding (RoPE): Unfused Memory vs Fused In-Register",
        "results": results
    }

    with open("research/raw/rope_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/rope_results.json")
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_rope_benchmark()
