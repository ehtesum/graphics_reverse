import ctypes
import os
import subprocess
import json

# Corner G8: Turing Uniform Datapath & Uniform Register File (URF) Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)

def run_uniform_datapath_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernels:
    # Kernel 1: Uniform Loop Counter & Scalar Predicate (Compiler promotes to Uniform Datapath UR)
    # Kernel 2: Vectorized/Thread-Indexed Loop Counter (Forced onto Vector Register File R)
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry uniform_loop_kernel(
    .param .u64 d_out_cycles,
    .param .u32 iters
)
{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %u_count, %tidx, %accum;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.u32 %accum, 0;
    bar.sync 0;
    mov.u64 %t0, %clock64;

    mov.u32 %u_count, 0;

LOOP_UNIFORM:
    setp.ge.u32 %p_loop, %u_count, %n_iters;
    @%p_loop bra DONE_UNIFORM;

    // Uniform scalar addition (promoted to UADD3 in SASS)
    add.u32 %u_count, %u_count, 1;
    add.u32 %accum, %accum, %u_count;
    bra LOOP_UNIFORM;

DONE_UNIFORM:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_lead st.global.u32 [%out_ptr], %accum;
    ret;
}

.visible .entry vector_loop_kernel(
    .param .u64 d_out_cycles,
    .param .u32 iters
)
{
    .reg .u64 %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %v_count, %tidx, %accum;
    .reg .pred %p_loop, %p_lead;

    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %n_iters, [iters];
    mov.u32 %tidx, %tid.x;

    mov.u32 %accum, 0;
    bar.sync 0;
    mov.u64 %t0, %clock64;

    // Force vector register file usage by making count dependent on %tid.x
    mov.u32 %v_count, %tidx;
    and.b32 %v_count, %v_count, 0; // v_count is 0, but compiler cannot prove it uniform

LOOP_VECTOR:
    setp.ge.u32 %p_loop, %v_count, %n_iters;
    @%p_loop bra DONE_VECTOR;

    // Vector register addition
    add.u32 %v_count, %v_count, 1;
    add.u32 %accum, %accum, %v_count;
    bra LOOP_VECTOR;

DONE_VECTOR:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    setp.eq.u32 %p_lead, %tidx, 0;
    @%p_lead st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    @%p_lead st.global.u32 [%out_ptr], %accum;
    ret;
}
"""
    ptx_file = "research/raw/uniform_datapath.ptx"
    cubin_file = "research/raw/uniform_datapath.cubin"
    sass_file = "research/raw/uniform_datapath_sass.txt"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    os.system(f"cuda_toolkit\\bin\\ptxas.exe -arch=sm_75 {ptx_file} -o {cubin_file}")
    os.system(f"cuda_toolkit\\bin\\cuobjdump.exe -sass {cubin_file} > {sass_file}")

    # Inspect SASS for Uniform instructions
    with open(sass_file, "r") as f:
        sass_content = f.read()

    has_uadd = "UADD" in sass_content
    has_usel = "USEL" in sass_content or "UPRED" in sass_content
    has_ur = "UR" in sass_content

    print("--- SASS Uniform Datapath Inspection ---")
    print(f"Uniform Registers (UR) detected: {has_ur}")
    print(f"Uniform Add instructions detected: {has_uadd}")

    module = ctypes.c_void_p()
    with open(cubin_file, "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    k_uniform = ctypes.c_void_p()
    k_vector = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(k_uniform), module, b"uniform_loop_kernel")
    nvcuda.cuModuleGetFunction(ctypes.byref(k_vector), module, b"vector_loop_kernel")

    d_out = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out), 16)

    iters = 100000
    c_dout = ctypes.c_uint64(d_out.value)
    c_iters = ctypes.c_uint32(iters)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p)
    )

    # 1. Run Uniform Loop Kernel
    nvcuda.cuLaunchKernel(k_uniform, 1, 1, 1, 32, 1, 1, 0, None, args, None)
    nvcuda.cuCtxSynchronize()
    cyc_u = ctypes.c_uint64()
    nvcuda.cuMemcpyDtoH_v2(ctypes.byref(cyc_u), d_out, 8)
    lat_u = cyc_u.value / iters

    # 2. Run Vector Loop Kernel
    nvcuda.cuLaunchKernel(k_vector, 1, 1, 1, 32, 1, 1, 0, None, args, None)
    nvcuda.cuCtxSynchronize()
    cyc_v = ctypes.c_uint64()
    nvcuda.cuMemcpyDtoH_v2(ctypes.byref(cyc_v), d_out, 8)
    lat_v = cyc_v.value / iters

    speedup = lat_v / lat_u
    print(f"\nUniform Loop: {lat_u:.2f} cycles / iteration")
    print(f"Vector Loop:  {lat_v:.2f} cycles / iteration")
    print(f"Uniform Datapath Speedup: {speedup:.2f}x")

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "benchmark": "Turing Uniform Datapath & Uniform Register File (URF)",
        "sass_telemetry": {
            "has_uniform_registers": has_ur,
            "has_uniform_add": has_uadd
        },
        "results": {
            "uniform_cycles_per_iteration": lat_u,
            "vector_cycles_per_iteration": lat_v,
            "speedup_ratio": speedup
        }
    }

    with open("research/raw/uniform_datapath_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/uniform_datapath_results.json")
    nvcuda.cuMemFree_v2(d_out)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_uniform_datapath_benchmark()
