import ctypes
import os
import json

# Corner G7: Special Function Unit (SFU) Throughput & Issue Port Arbitration Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 14 SMs, 1785 MHz boost)

def run_sfu_contention_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernels:
    # 1. Pure FP32 FMA (1024 unrolled FMAs)
    # 2. Pure SFU RSQRT (1024 unrolled rsqrt.approx.f32)
    # 3. Pure SFU EXP2 (1024 unrolled ex2.approx.f32)
    # 4. Interleaved FP32 FMA + SFU RSQRT (512 FMAs + 512 RSQRTs)
    # Loop over K iterations, timed with %clock64
    def build_body(op_type, count):
        lines = []
        if op_type == "fma":
            for i in range(count):
                lines.append(f"    fma.rn.f32 %r{i%8}, %r{i%8}, %c1, %c2;")
        elif op_type == "rsqrt":
            for i in range(count):
                lines.append(f"    rsqrt.approx.f32 %r{i%8}, %r{i%8};")
        elif op_type == "exp2":
            for i in range(count):
                lines.append(f"    ex2.approx.f32 %r{i%8}, %r{i%8};")
        elif op_type == "interleaved":
            for i in range(count // 2):
                lines.append(f"    fma.rn.f32 %r{i%8}, %r{i%8}, %c1, %c2;")
                lines.append(f"    rsqrt.approx.f32 %s{i%8}, %s{i%8};")
        return "\n".join(lines)

    ptx_code = f"""
.version 7.5
.target sm_75
.address_size 64

.visible .entry sfu_bench_fma(.param .u64 d_out, .param .u32 iters)
{{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %it, %tidx;
    .reg .f32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %c1, %c2;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.f32 %c1, 1.0001;
    mov.f32 %c2, 0.0001;
    mov.f32 %r0, 1.0; mov.f32 %r1, 1.1; mov.f32 %r2, 1.2; mov.f32 %r3, 1.3;
    mov.f32 %r4, 1.4; mov.f32 %r5, 1.5; mov.f32 %r6, 1.6; mov.f32 %r7, 1.7;

    bar.sync 0;
    mov.u64 %t0, %clock64;
    mov.u32 %it, 0;

LOOP_FMA:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_FMA;

{build_body("fma", 1024)}

    add.u32 %it, %it, 1;
    bra LOOP_FMA;

DONE_FMA:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_lead st.global.f32 [%out_ptr], %r0;
    ret;
}}

.visible .entry sfu_bench_rsqrt(.param .u64 d_out, .param .u32 iters)
{{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %it, %tidx;
    .reg .f32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.f32 %r0, 1.0; mov.f32 %r1, 1.1; mov.f32 %r2, 1.2; mov.f32 %r3, 1.3;
    mov.f32 %r4, 1.4; mov.f32 %r5, 1.5; mov.f32 %r6, 1.6; mov.f32 %r7, 1.7;

    bar.sync 0;
    mov.u64 %t0, %clock64;
    mov.u32 %it, 0;

LOOP_RSQRT:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_RSQRT;

{build_body("rsqrt", 1024)}

    add.u32 %it, %it, 1;
    bra LOOP_RSQRT;

DONE_RSQRT:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_lead st.global.f32 [%out_ptr], %r0;
    ret;
}}

.visible .entry sfu_bench_exp2(.param .u64 d_out, .param .u32 iters)
{{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %it, %tidx;
    .reg .f32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.f32 %r0, 0.1; mov.f32 %r1, 0.2; mov.f32 %r2, 0.3; mov.f32 %r3, 0.4;
    mov.f32 %r4, 0.5; mov.f32 %r5, 0.6; mov.f32 %r6, 0.7; mov.f32 %r7, 0.8;

    bar.sync 0;
    mov.u64 %t0, %clock64;
    mov.u32 %it, 0;

LOOP_EXP2:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_EXP2;

{build_body("exp2", 1024)}

    add.u32 %it, %it, 1;
    bra LOOP_EXP2;

DONE_EXP2:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_lead st.global.f32 [%out_ptr], %r0;
    ret;
}}

.visible .entry sfu_bench_interleaved(.param .u64 d_out, .param .u32 iters)
{{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %it, %tidx;
    .reg .f32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %c1, %c2;
    .reg .f32 %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.f32 %c1, 1.0001;
    mov.f32 %c2, 0.0001;
    mov.f32 %r0, 1.0; mov.f32 %r1, 1.1; mov.f32 %r2, 1.2; mov.f32 %r3, 1.3;
    mov.f32 %r4, 1.4; mov.f32 %r5, 1.5; mov.f32 %r6, 1.6; mov.f32 %r7, 1.7;
    mov.f32 %s0, 1.0; mov.f32 %s1, 1.1; mov.f32 %s2, 1.2; mov.f32 %s3, 1.3;
    mov.f32 %s4, 1.4; mov.f32 %s5, 1.5; mov.f32 %s6, 1.6; mov.f32 %s7, 1.7;

    bar.sync 0;
    mov.u64 %t0, %clock64;
    mov.u32 %it, 0;

LOOP_INTER:
    setp.ge.u32 %p_loop, %it, %n_iters;
    @%p_loop bra DONE_INTER;

{build_body("interleaved", 1024)}

    add.u32 %it, %it, 1;
    bra LOOP_INTER;

DONE_INTER:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    add.f32 %r0, %r0, %s0;
    @%p_lead st.global.f32 [%out_ptr], %r0;
    ret;
}}
"""
    with open("research/raw/sfu_contention.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\sfu_contention.ptx -o research\raw\sfu_contention.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/sfu_contention.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    funcs = {}
    for name in ["fma", "rsqrt", "exp2", "interleaved"]:
        func = ctypes.c_void_p()
        nvcuda.cuModuleGetFunction(ctypes.byref(func), module, f"sfu_bench_{name}".encode("utf-8"))
        funcs[name] = func

    d_out = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out), 16)

    iters = 1000
    total_instructions = 1024 * iters

    # Launch configuration: 1 block with 32 threads (1 warp) to isolate single warp issue rate
    threads = 32
    blocks = 1

    c_dout = ctypes.c_uint64(d_out.value)
    c_iters = ctypes.c_uint32(iters)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p)
    )

    results = {}
    print("Benchmarking Special Function Unit (SFU) vs FP32 Throughput & Contention...")

    for name in ["fma", "rsqrt", "exp2", "interleaved"]:
        nvcuda.cuLaunchKernel(funcs[name], blocks, 1, 1, threads, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        cyc = ctypes.c_uint64()
        nvcuda.cuMemcpyDtoH_v2(ctypes.byref(cyc), d_out, 8)

        total_cyc = cyc.value
        cycles_per_inst = total_cyc / (1024.0 * iters)
        inst_per_cycle = 1.0 / max(cycles_per_inst, 1e-9)

        print(f"Kernel: {name:<12} | Cycles/Inst (Warp): {cycles_per_inst:5.2f} cycles | Issue Rate: {inst_per_cycle:5.2f} warp-inst/cycle")
        results[name] = {
            "total_cycles": total_cyc,
            "cycles_per_instruction": cycles_per_inst,
            "instructions_per_cycle": inst_per_cycle
        }

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "benchmark": "Special Function Unit (SFU) vs FP32 Issue Port Contention",
        "results": results
    }

    with open("research/raw/sfu_contention_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/sfu_contention_results.json")
    nvcuda.cuMemFree_v2(d_out)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_sfu_contention_benchmark()
