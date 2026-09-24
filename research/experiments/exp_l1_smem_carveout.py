import ctypes
import os
import json

# Corner G9: Dynamic Shared Memory vs L1 Data Cache Partitioning Cliff
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5, 96 KB Unified SRAM/SM)

def run_carveout_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # PTX Kernel: Pointer chase through a 48 KB buffer using ld.global.ca (Cache at All levels)
    # If L1 is 64 KB -> All 48 KB fits in L1D -> High speed hit (~35c)
    # If L1 is 32 KB -> 48 KB overflows L1D -> L2 spill (~180-200c)
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry carveout_chase_kernel(
    .param .u64 d_ptr,
    .param .u32 iters,
    .param .u64 d_out_cycles
)
{
    .reg .u64 %ptr, %out_ptr, %t0, %t1, %diff;
    .reg .u32 %n_iters, %i;
    .reg .pred %p_loop;

    ld.param.u64 %ptr, [d_ptr];
    ld.param.u32 %n_iters, [iters];
    ld.param.u64 %out_ptr, [d_out_cycles];

    bar.sync 0;
    mov.u64 %t0, %clock64;
    mov.u32 %i, 0;

LOOP_CHASE:
    setp.ge.u32 %p_loop, %i, %n_iters;
    @%p_loop bra DONE_CHASE;

    ld.global.ca.u64 %ptr, [%ptr];
    add.u32 %i, %i, 1;
    bra LOOP_CHASE;

DONE_CHASE:
    mov.u64 %t1, %clock64;
    sub.u64 %diff, %t1, %t0;

    st.global.u64 [%out_ptr], %diff;
    add.u64 %out_ptr, %out_ptr, 8;
    st.global.u64 [%out_ptr], %ptr;
    ret;
}
"""
    ptx_file = "research/raw/carveout_bench.ptx"
    cubin_file = "research/raw/carveout_bench.cubin"

    with open(ptx_file, "w") as f:
        f.write(ptx_code)

    os.system(f"cuda_toolkit\\bin\\ptxas.exe -arch=sm_75 {ptx_file} -o {cubin_file}")

    module = ctypes.c_void_p()
    with open(cubin_file, "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    kernel = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"carveout_chase_kernel")

    # Allocate 48 KB working set (6144 uint64 nodes)
    # Fits inside 64 KB L1D, but exceeds 32 KB L1D!
    SIZE_KB = 48
    NUM_NODES = (SIZE_KB * 1024) // 8
    BYTE_SIZE = NUM_NODES * 8

    d_buf = ctypes.c_void_p()
    d_out = ctypes.c_void_p()
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_buf), BYTE_SIZE)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out), 16)

    # Build cyclic random pointer chase
    import random
    indices = list(range(NUM_NODES))
    for i in range(NUM_NODES - 1, 0, -1):
        j = random.randint(0, i - 1)
        indices[i], indices[j] = indices[j], indices[i]

    base_ptr = d_buf.value
    host_buf = (ctypes.c_uint64 * NUM_NODES)(*[base_ptr + indices[(i + 1) % NUM_NODES] * 8 for i in range(NUM_NODES)])
    nvcuda.cuMemcpyHtoD_v2(d_buf, host_buf, BYTE_SIZE)

    CHASE_ITERS = 20000
    c_dbuf = ctypes.c_uint64(d_buf.value)
    c_iters = ctypes.c_uint32(CHASE_ITERS)
    c_dout = ctypes.c_uint64(d_out.value)

    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.byref(c_dbuf), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_dout), ctypes.c_void_p)
    )

    # Attribute for Carveout: CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    # Values: 0 (0% SMEM, max L1), 33 (33% SMEM), 66 (66% SMEM), 100 (100% SMEM, min L1)
    carveouts = [
        ("0% SMEM (Max 64 KB L1)", 0),
        ("33% SMEM (64 KB L1)", 33),
        ("66% SMEM (32 KB L1)", 66),
        ("100% SMEM (Min 16 KB L1)", 100)
    ]

    results = []
    print("Benchmarking Dynamic Shared Memory vs L1 Data Cache Carveout on 48 KB Working Set...")
    print(f"{'Configuration':<28} | {'Carveout %':<12} | {'Latency (cycles/access)':<24} | {'Cache Hit State'}")
    print("-" * 85)

    for label, pct in carveouts:
        nvcuda.cuFuncSetAttribute(kernel, 9, pct)

        # Warmup cache
        nvcuda.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        # Measure
        nvcuda.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        nvcuda.cuCtxSynchronize()

        cyc = ctypes.c_uint64()
        nvcuda.cuMemcpyDtoH_v2(ctypes.byref(cyc), d_out, 8)
        lat = cyc.value / CHASE_ITERS

        state = "L1 Hit (48 KB fits in 64 KB L1)" if lat < 100.0 else "L1 Miss -> L2 Spill (48 KB > 32 KB L1)"
        print(f"{label:<28} | {pct:<12d} | {lat:<24.2f} | {state}")

        results.append({
            "configuration": label,
            "carveout_percent": pct,
            "latency_cycles": lat,
            "state": state
        })

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "sram_per_sm": "96 KB Unified Shared Memory + L1D",
        "working_set_kb": SIZE_KB,
        "results": results
    }

    with open("research/raw/l1_smem_carveout_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/l1_smem_carveout_results.json")
    nvcuda.cuMemFree_v2(d_buf)
    nvcuda.cuMemFree_v2(d_out)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_carveout_benchmark()
