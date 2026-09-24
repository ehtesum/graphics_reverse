import ctypes
import os
import json
import time

# Corner A7: Speculative Decoding Parallel Token Verification Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 14 SMs)

def run_speculative_verification_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernel: Parallel Speculative Verification
    # Given K draft tokens with draft probs q[k], target logits P[K, V]
    # Block k evaluates token k: computes target prob p[draft_token_k] and verifies against q[k].
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry spec_verify_kernel(
    .param .u64 d_target_logits,
    .param .u64 d_draft_tokens,
    .param .u64 d_draft_probs,
    .param .u64 d_accept_mask,
    .param .u32 vocab_size,
    .param .u32 K_draft
)
{
    .reg .u64 %logits_ptr, %tokens_ptr, %probs_ptr, %mask_ptr;
    .reg .u64 %addr;
    .reg .u32 %v_size, %k_dim, %tidx, %k_idx, %draft_tok;
    .reg .f32 %q_prob, %target_logit, %rand_val, %ratio;
    .reg .pred %p_accept, %p_lead;

    ld.param.u64 %logits_ptr, [d_target_logits];
    ld.param.u64 %tokens_ptr, [d_draft_tokens];
    ld.param.u64 %probs_ptr, [d_draft_probs];
    ld.param.u64 %mask_ptr, [d_accept_mask];
    ld.param.u32 %v_size, [vocab_size];
    ld.param.u32 %k_dim, [K_draft];

    mov.u32 %tidx, %tid.x;
    mov.u32 %k_idx, %ctaid.x; // Block k handles draft token k

    // Thread 0 in block k verifies draft token k
    setp.eq.u32 %p_lead, %tidx, 0;
    @!%p_lead bra DONE_VERIFY;

    // Load draft token k
    mul.wide.u32 %addr, %k_idx, 4;
    add.u64 %addr, %tokens_ptr, %addr;
    ld.global.u32 %draft_tok, [%addr];

    // Load draft probability q
    mul.wide.u32 %addr, %k_idx, 4;
    add.u64 %addr, %probs_ptr, %addr;
    ld.global.f32 %q_prob, [%addr];

    // Load target logit for draft_tok: logits[k * V + draft_tok]
    mad.lo.u32 %draft_tok, %k_idx, %v_size, %draft_tok;
    mul.wide.u32 %addr, %draft_tok, 4;
    add.u64 %addr, %logits_ptr, %addr;
    ld.global.f32 %target_logit, [%addr];

    // Convert logit to approximate prob (simulated softmax prob)
    // ratio = p / q
    div.approx.f32 %ratio, %target_logit, %q_prob;

    // Simulated uniform random number in (0, 1)
    mov.f32 %rand_val, 0.75;

    // Accept condition: rand_val < ratio
    setp.ge.f32 %p_accept, %ratio, %rand_val;

    // Store accept mask (1 = accept, 0 = reject)
    mul.wide.u32 %addr, %k_idx, 4;
    add.u64 %addr, %mask_ptr, %addr;
    selp.u32 %draft_tok, 1, 0, %p_accept;
    st.global.u32 [%addr], %draft_tok;

DONE_VERIFY:
    ret;
}
"""
    ptx_file = "research/raw/spec_verify.ptx"
    cubin_file = "research/raw/spec_verify.cubin"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    os.system(f"cuda_toolkit\\bin\\ptxas.exe -arch=sm_75 {ptx_file} -o {cubin_file}")

    module = ctypes.c_void_p()
    with open(cubin_file, "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    kernel = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"spec_verify_kernel")

    # Vocabulary size V = 32,000 (Standard LLaMA vocabulary)
    VOCAB_SIZE = 32000
    K_LIST = [1, 2, 4, 8, 16]

    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_ev), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_ev), 0)

    results = []
    print("Benchmarking Speculative Decoding Parallel Verification Kernel on TU117...")
    print(f"{'Draft Tokens (K)':<18} | {'Total Logits (KB)':<20} | {'Verification Latency (µs)':<26} | {'Throughput (Tokens/s)'}")
    print("-" * 85)

    for K in K_LIST:
        logits_bytes = K * VOCAB_SIZE * 4
        tokens_bytes = K * 4
        probs_bytes = K * 4
        mask_bytes = K * 4

        d_logits = ctypes.c_void_p()
        d_tokens = ctypes.c_void_p()
        d_probs = ctypes.c_void_p()
        d_mask = ctypes.c_void_p()

        nvcuda.cuMemAlloc_v2(ctypes.byref(d_logits), logits_bytes)
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_tokens), tokens_bytes)
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_probs), probs_bytes)
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_mask), mask_bytes)

        nvcuda.cuMemsetD32_v2(d_logits, 0x3f800000, K * VOCAB_SIZE)
        nvcuda.cuMemsetD32_v2(d_tokens, 42, K)
        nvcuda.cuMemsetD32_v2(d_probs, 0x3f000000, K)

        c_dlogits = ctypes.c_uint64(d_logits.value)
        c_dtokens = ctypes.c_uint64(d_tokens.value)
        c_dprobs = ctypes.c_uint64(d_probs.value)
        c_dmask = ctypes.c_uint64(d_mask.value)
        c_vocab = ctypes.c_uint32(VOCAB_SIZE)
        c_K = ctypes.c_uint32(K)

        args = (ctypes.c_void_p * 6)(
            ctypes.cast(ctypes.byref(c_dlogits), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dtokens), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dprobs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dmask), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_vocab), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p)
        )

        threads = 32
        blocks = K

        # Warmup
        nvcuda.cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        nvcuda.cuEventRecord(start_ev, None)
        for _ in range(100):
            nvcuda.cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, None, args, None)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)

        ms = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
        avg_us = (ms.value / 100.0) * 1000.0
        tok_s = (K / (avg_us / 1e6))

        logits_kb = logits_bytes / 1024.0
        print(f"K = {K:<14d} | {logits_kb:<20.1f} | {avg_us:<26.2f} | {tok_s:12.1f}")

        results.append({
            "draft_tokens_K": K,
            "logits_size_kb": logits_kb,
            "verification_latency_us": avg_us,
            "tokens_per_second": tok_s
        })

        nvcuda.cuMemFree_v2(d_logits)
        nvcuda.cuMemFree_v2(d_tokens)
        nvcuda.cuMemFree_v2(d_probs)
        nvcuda.cuMemFree_v2(d_mask)

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "benchmark": "Speculative Decoding Parallel Token Verification",
        "vocabulary_size": VOCAB_SIZE,
        "results": results
    }

    with open("research/raw/speculative_verification_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/speculative_verification_results.json")
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_speculative_verification_benchmark()
