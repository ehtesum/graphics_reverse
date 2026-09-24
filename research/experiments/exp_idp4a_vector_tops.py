import ctypes
import os
import json

# Corner A6: Quantized INT8 DP4A Peak TOPS on Non-Tensor Turing Silicon
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 896 CUDA Cores)

def run_idp4a_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernel: Unrolled IDP4A instruction loop across 16 accumulators
    # dp4a.s32.s32 d, a, b, c;
    # Computes 4 MACs = 8 INT8 operations per instruction
    def generate_dp4a_block(unroll_count):
        lines = []
        for i in range(unroll_count):
            lines.append(f"    dp4a.s32.s32 %acc{i%16}, %a{i%8}, %b{i%8}, %acc{i%16};")
        return "\n".join(lines)

    ptx_code = f"""
.version 7.5
.target sm_75
.address_size 64

.visible .entry idp4a_peak_kernel(
    .param .u64 d_out,
    .param .u32 iters
)
{{
    .reg .u64 %out_ptr;
    .reg .u32 %n_iters, %it, %tidx, %bidx;
    .reg .u32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;
    .reg .u32 %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7;
    .reg .u32 %acc0, %acc1, %acc2, %acc3, %acc4, %acc5, %acc6, %acc7;
    .reg .u32 %acc8, %acc9, %acc10, %acc11, %acc12, %acc13, %acc14, %acc15;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_iters, [iters];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    // Packed INT8 vectors (4 signed 8-bit values per 32-bit register)
    mov.u32 %a0, 0x01010101; mov.u32 %a1, 0x02020202;
    mov.u32 %a2, 0x01020102; mov.u32 %a3, 0x02010201;
    mov.u32 %a4, 0x01010202; mov.u32 %a5, 0x02020101;
    mov.u32 %a6, 0x01020201; mov.u32 %a7, 0x02010102;

    mov.u32 %b0, 0x01010101; mov.u32 %b1, 0x02020202;
    mov.u32 %b2, 0x01020102; mov.u32 %b3, 0x02010201;
    mov.u32 %b4, 0x01010202; mov.u32 %b5, 0x02020101;
    mov.u32 %b6, 0x01020201; mov.u32 %b7, 0x02010102;

    mov.u32 %acc0, 0; mov.u32 %acc1, 0; mov.u32 %acc2, 0; mov.u32 %acc3, 0;
    mov.u32 %acc4, 0; mov.u32 %acc5, 0; mov.u32 %acc6, 0; mov.u32 %acc7, 0;
    mov.u32 %acc8, 0; mov.u32 %acc9, 0; mov.u32 %acc10, 0; mov.u32 %acc11, 0;
    mov.u32 %acc12, 0; mov.u32 %acc13, 0; mov.u32 %acc14, 0; mov.u32 %acc15, 0;

    mov.u32 %it, 0;

LOOP_DP4A:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_DP4A;

{generate_dp4a_block(512)}

    add.u32 %it, %it, 1;
    bra LOOP_DP4A;

DONE_DP4A:
    // Accumulate all registers to prevent dead code elimination
    add.u32 %acc0, %acc0, %acc1;
    add.u32 %acc0, %acc0, %acc2;
    add.u32 %acc0, %acc0, %acc3;
    add.u32 %acc0, %acc0, %acc4;
    add.u32 %acc0, %acc0, %acc5;
    add.u32 %acc0, %acc0, %acc6;
    add.u32 %acc0, %acc0, %acc7;
    add.u32 %acc0, %acc0, %acc8;
    add.u32 %acc0, %acc0, %acc9;
    add.u32 %acc0, %acc0, %acc10;
    add.u32 %acc0, %acc0, %acc11;
    add.u32 %acc0, %acc0, %acc12;
    add.u32 %acc0, %acc0, %acc13;
    add.u32 %acc0, %acc0, %acc14;
    add.u32 %acc0, %acc0, %acc15;

    // Store result
    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u32 [%out_ptr], %acc0;
    ret;
}}
"""
    ptx_file = "research/raw/idp4a_peak.ptx"
    cubin_file = "research/raw/idp4a_peak.cubin"
    sass_file = "research/raw/idp4a_peak_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    os.system(f"cuda_toolkit\\bin\\ptxas.exe -arch=sm_75 {ptx_file} -o {cubin_file}")
    os.system(f"cuda_toolkit\\bin\\cuobjdump.exe -sass {cubin_file} > {sass_file}")

    with open(sass_file, "r") as f:
        sass_txt = f.read()
    has_idp4a = "IDP.4A" in sass_txt or "IDP4A" in sass_txt
    idp_count = sass_txt.count("IDP.4A")
    print(f"SASS Verified: IDP4A instruction emitted: {has_idp4a} (Count in SASS: {idp_count})")

    module = ctypes.c_void_p()
    with open(cubin_file, "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    kernel = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"idp4a_peak_kernel")

    d_out = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out), 16)

    # Launch configuration: Full device saturation (112 blocks * 256 threads = 28,672 threads)
    # Exactly matches our FP32 and FP16 benchmark configurations
    threads_per_block = 256
    num_blocks = 112 # 8 blocks per SM * 14 SMs
    total_threads = threads_per_block * num_blocks # 28,672
    unroll_count = 512
    iters = 1000

    c_dout = ctypes.c_uint64(d_out.value)
    c_iters = ctypes.c_uint32(iters)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p)
    )

    # CUDA Events
    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_ev), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_ev), 0)

    # Warmup
    nvcuda.cuLaunchKernel(kernel, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
    nvcuda.cuCtxSynchronize()

    # Timing
    nvcuda.cuEventRecord(start_ev, None)
    for _ in range(10):
        nvcuda.cuLaunchKernel(kernel, num_blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, None)
    nvcuda.cuEventRecord(stop_ev, None)
    nvcuda.cuEventSynchronize(stop_ev)

    ms = ctypes.c_float()
    nvcuda.cuEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
    avg_ms = ms.value / 10.0

    # Total INT8 operations:
    # 28,672 threads * 512 instructions * 1000 iters * 8 ops per instruction
    total_ops = total_threads * unroll_count * iters * 8.0 # 8 ops (4 MACs) per IDP4A
    sustained_tops = (total_ops / 1e12) / (avg_ms / 1000.0)

    print("\n--- Physical INT8 DP4A Benchmark on Non-Tensor Silicon ---")
    print(f"Execution Time: {avg_ms:.3f} ms")
    print(f"Total Operations: {total_ops / 1e9:.2f} G-Operations")
    print(f"Sustained INT8 Compute: {sustained_tops:.3f} TOPS")
    print(f"Throughput Ratio vs FP32 (3.134 TFLOPS): {sustained_tops / 3.134:.2f}x")
    print(f"Throughput Ratio vs FP16 (6.318 TFLOPS): {sustained_tops / 6.318:.2f}x")

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 896 CUDA Cores)",
        "benchmark": "Quantized INT8 DP4A Vector Dot Product Peak TOPS",
        "total_threads": total_threads,
        "instructions_per_thread": unroll_count * iters,
        "ops_per_instruction": 8,
        "results": {
            "execution_time_ms": avg_ms,
            "sustained_tops": sustained_tops,
            "speedup_vs_fp32": sustained_tops / 3.134,
            "speedup_vs_fp16": sustained_tops / 6.318
        }
    }

    with open("research/raw/idp4a_vector_tops_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/idp4a_vector_tops_results.json")
    nvcuda.cuMemFree_v2(d_out)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_idp4a_benchmark()
