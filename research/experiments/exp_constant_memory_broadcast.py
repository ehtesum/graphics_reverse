import ctypes
import os
import subprocess
import json

cuda = ctypes.WinDLL("nvcuda.dll")

def check(err):
    if err != 0:
        raise RuntimeError(f"CUDA Driver Error: {err}")

check(cuda.cuInit(0))
cu_device = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(cu_device), 0))
cu_context = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(cu_context), 0, cu_device))

ptx_code = r"""
.version 7.5
.target sm_75
.address_size 64

.const .u32 c_table[1024];

.visible .entry const_broadcast_bench(
    .param .u64 d_out_cycles,
    .param .u32 divergence_stride,
    .param .u32 iterations
)
{
    .reg .u64 %out_ptr;
    .reg .u32 %stride, %iters, %tid, %offset, %val, %idx;
    .reg .u64 %t0, %t1, %dt;
    .reg .pred %p;

    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %stride, [divergence_stride];
    ld.param.u32 %iters, [iterations];

    mov.u32 %tid, %tid.x;
    
    // offset = (tid * stride) % 1024
    mul.lo.u32 %offset, %tid, %stride;
    and.b32 %offset, %offset, 1023;
    shl.b32 %offset, %offset, 2; // bytes

    bar.sync 0;
    mov.u64 %t0, %clock64;

    mov.u32 %val, 0;
    mov.u32 %idx, 0;

$LOOP:
    // Read from constant memory with thread-dependent offset
    ld.const.u32 %val, c_table[%offset];
    add.u32 %idx, %idx, 1;
    setp.lt.u32 %p, %idx, %iters;
    @%p bra $LOOP;

    bar.sync 0;
    mov.u64 %t1, %clock64;
    sub.u64 %dt, %t1, %t0;

    // Thread 0 writes out cycles
    setp.eq.u32 %p, %tid, 0;
    @%p st.global.u64 [%out_ptr], %dt;

    ret;
}
"""

with open("const_bench.ptx", "w") as f:
    f.write(ptx_code)

ptxas_bin = r"cuda_toolkit\bin\ptxas.exe"
cubin_file = "const_bench.cubin"
cmd = [ptxas_bin, "-arch=sm_75", "const_bench.ptx", "-o", cubin_file]
subprocess.run(cmd, check=True)

with open(cubin_file, "rb") as f:
    cubin_data = f.read()

cu_module = ctypes.c_void_p()
check(cuda.cuModuleLoadData(ctypes.byref(cu_module), cubin_data))

# Get constant memory symbol pointer and populate it
c_symbol = ctypes.c_void_p()
c_bytes = ctypes.c_size_t()
check(cuda.cuModuleGetGlobal_v2(ctypes.byref(c_symbol), ctypes.byref(c_bytes), cu_module, b"c_table"))

# Fill constant table with values
table_data = (ctypes.c_uint32 * 1024)(*[i * 11 for i in range(1024)])
check(cuda.cuMemcpyHtoD_v2(c_symbol, ctypes.byref(table_data), ctypes.sizeof(table_data)))

kernel = ctypes.c_void_p()
check(cuda.cuModuleGetFunction(ctypes.byref(kernel), cu_module, b"const_broadcast_bench"))

d_cycles = ctypes.c_void_p()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_cycles), 8))

# Test varying unique address counts per warp
# Stride 0 -> All 32 threads read offset 0 (1 unique address = pure broadcast)
# Stride 1 -> 32 threads read 32 unique offsets (32-way serial replay)
test_strides = [
    (0, "Pure Broadcast (1 Unique Address / Warp)"),
    (1, "Full Divergence (32 Unique Addresses / Warp)"),
    (2, "16 Unique Addresses / Warp (2 threads/address)"),
    (4, "8 Unique Addresses / Warp (4 threads/address)"),
    (8, "4 Unique Addresses / Warp (8 threads/address)"),
    (16, "2 Unique Addresses / Warp (16 threads/address)"),
    (32, "1 Unique Address / Warp (Bank-Aliased Broadcast)")
]

iterations = 100000

print("=== NVIDIA Turing TU117 Constant Memory Broadcast vs Serialization Dissection ===")
print(f"{'Access Pattern':<48} | {'Cycles/Iteration':<18} | {'Slowdown Ratio':<15} | {'State'}")
print("-" * 105)

results = []
baseline_cycles = 0

for stride, label in test_strides:
    c_stride = ctypes.c_uint32(stride)
    c_iters = ctypes.c_uint32(iterations)
    params = [
        ctypes.byref(d_cycles),
        ctypes.byref(c_stride),
        ctypes.byref(c_iters)
    ]
    param_ptrs = (ctypes.c_void_p * len(params))(*[ctypes.cast(p, ctypes.c_void_p) for p in params])
    
    # Warmup
    check(cuda.cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, None, param_ptrs, None))
    check(cuda.cuCtxSynchronize())

    # Measure
    check(cuda.cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, None, param_ptrs, None))
    check(cuda.cuCtxSynchronize())

    h_cycles = ctypes.c_uint64()
    check(cuda.cuMemcpyDtoH_v2(ctypes.byref(h_cycles), d_cycles, 8))

    cycles_per_iter = h_cycles.value / iterations
    if stride == 0:
        baseline_cycles = cycles_per_iter

    slowdown = cycles_per_iter / baseline_cycles
    state = "Single-Cycle Broadcast" if stride == 0 or stride == 32 else f"Constant Cache Multi-Cycle Replay ({slowdown:.1f}x)"

    print(f"{label:<48} | {cycles_per_iter:<18.3f} | {slowdown:<15.2f}x | {state}")

    results.append({
        "stride": stride,
        "label": label,
        "cycles_per_iteration": cycles_per_iter,
        "slowdown_ratio": slowdown,
        "state": state
    })

output_data = {
    "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
    "benchmark": "Constant Memory Warp Broadcast vs Serialization Replay",
    "iterations": iterations,
    "baseline_broadcast_cycles": baseline_cycles,
    "results": results
}

with open("research/raw/constant_memory_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\nResults successfully written to research/raw/constant_memory_results.json")

# SASS inspection
cmd_sass = [r"cuda_toolkit\bin\cuobjdump.exe", "-sass", cubin_file]
sass_res = subprocess.run(cmd_sass, capture_output=True, text=True)
with open("research/raw/constant_memory_sass.txt", "w") as f:
    f.write(sass_res.stdout)
print("SASS disassembly written to research/raw/constant_memory_sass.txt")

for f in ["const_bench.ptx", "const_bench.cubin"]:
    if os.path.exists(f): os.remove(f)

check(cuda.cuMemFree_v2(d_cycles))
check(cuda.cuModuleUnload(cu_module))
check(cuda.cuCtxDestroy_v2(cu_context))
