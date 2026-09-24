import ctypes
import os
import json
import time

# Corner G4: Cooperative Device-Wide Grid Synchronization Microbenchmark
# Target: NVIDIA GeForce GTX 1650 (TU117 Turing, 14 SMs)

def run_cooperative_grid_sync():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)
    
    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)
    
    # Query Cooperative Launch Attribute
    coop_supported = ctypes.c_int()
    nvcuda.cuDeviceGetAttribute(ctypes.byref(coop_supported), 86, dev.value) # 86 = CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
    
    sm_count = ctypes.c_int()
    nvcuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, dev.value) # 16 = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    
    print(f"Device SM Count: {sm_count.value}")
    print(f"Hardware Cooperative Launch Supported (WDDM 2.7): {coop_supported.value}")

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)
    
    # PTX Kernel: Sense-reversing grid barrier across concurrent resident blocks
    # Each block elects thread 0 to update atomic arrival count.
    # When arrival count reaches num_blocks, the last arriving block resets count and flips phase.
    # Other blocks spin-wait on phase change.
    # Each thread records start and end clock64() across K barrier rounds.
    ptx_code = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry grid_sync_kernel(
    .param .u64 d_barrier_count,
    .param .u64 d_barrier_phase,
    .param .u64 d_out_cycles,
    .param .u32 num_blocks,
    .param .u32 iters
)
{
    .reg .u64 %bc_ptr, %bp_ptr, %out_ptr;
    .reg .u32 %n_blocks, %n_iters;
    .reg .u32 %tidx, %bidx;
    .reg .u64 %clk_start, %clk_end, %clk_diff;
    .reg .u32 %i, %ticket, %expected, %phase, %cur_phase;
    .reg .pred %p_leader, %p_last, %p_loop, %p_spin;

    ld.param.u64 %bc_ptr, [d_barrier_count];
    ld.param.u64 %bp_ptr, [d_barrier_phase];
    ld.param.u64 %out_ptr, [d_out_cycles];
    ld.param.u32 %n_blocks, [num_blocks];
    ld.param.u32 %n_iters, [iters];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    setp.eq.u32 %p_leader, %tidx, 0;

    // Block-level barrier before starting measurement
    bar.sync 0;

    // Record start clock
    mov.u64 %clk_start, %clock64;

    mov.u32 %i, 0;
    mov.u32 %phase, 0;

LOOP_BARRIER:
    setp.ge.u32 %p_loop, %i, %n_iters;
    @%p_loop bra DONE_BARRIER;

    // Invert expected phase for this round
    xor.b32 %expected, %phase, 1;

    // Thread 0 in each block coordinates grid arrival
    @!%p_leader bra WAIT_PHASE;

    membar.gl;
    atom.global.add.u32 %ticket, [%bc_ptr], 1;

    // Check if we are the last block
    sub.u32 %ticket, %ticket, 0;
    sub.u32 %n_blocks, %n_blocks, 1;
    setp.eq.u32 %p_last, %ticket, %n_blocks;
    add.u32 %n_blocks, %n_blocks, 1;

    @!%p_last bra SPIN_LEADER;

    // Last arriving block: reset counter to 0 and advance phase
    st.global.u32 [%bc_ptr], 0;
    membar.gl;
    st.global.u32 [%bp_ptr], %expected;
    bra ADVANCE_PHASE;

SPIN_LEADER:
    // Leader spins until phase matches expected
    ld.global.volatile.u32 %cur_phase, [%bp_ptr];
    setp.ne.u32 %p_spin, %cur_phase, %expected;
    @%p_spin bra SPIN_LEADER;

ADVANCE_PHASE:
    mov.u32 %phase, %expected;

WAIT_PHASE:
    // Synchronize all threads within the block after grid barrier
    bar.sync 0;

    add.u32 %i, %i, 1;
    bra LOOP_BARRIER;

DONE_BARRIER:
    mov.u64 %clk_end, %clock64;
    sub.u64 %clk_diff, %clk_end, %clk_start;

    // Store total cycles to out_ptr[bidx]
    @%p_leader bra STORE_RES;
    bra EXIT_KERNEL;

STORE_RES:
    mul.wide.u32 %clk_start, %bidx, 8;
    add.u64 %out_ptr, %out_ptr, %clk_start;
    st.global.u64 [%out_ptr], %clk_diff;

EXIT_KERNEL:
    ret;
}
"""
    with open("research/raw/grid_sync.ptx", "w") as f:
        f.write(ptx_code)

    os.system(r"cuda_toolkit\bin\ptxas.exe -arch=sm_75 research\raw\grid_sync.ptx -o research\raw\grid_sync.cubin")

    module = ctypes.c_void_p()
    with open("research/raw/grid_sync.cubin", "rb") as f:
        cubin_data = f.read()
    nvcuda.cuModuleLoadData(ctypes.byref(module), cubin_data)

    kernel = ctypes.c_void_p()
    nvcuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"grid_sync_kernel")

    # Allocate barrier control variables and results on device
    d_barrier_count = ctypes.c_void_p()
    d_barrier_phase = ctypes.c_void_p()
    d_out_cycles = ctypes.c_void_p()

    nvcuda.cuMemAlloc_v2(ctypes.byref(d_barrier_count), 4)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_barrier_phase), 4)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_out_cycles), 14 * 8)

    # Test sweep of block counts up to SM count (14 SMs)
    block_counts = [1, 2, 4, 7, 14]
    iters = 1000
    threads_per_block = 128

    results = []
    print(f"Sweeping Device-Wide Grid Synchronization Latency across {iters} barrier iterations...")

    for num_blocks in block_counts:
        # Reset barrier state
        zero = (ctypes.c_uint32 * 1)(0)
        nvcuda.cuMemcpyHtoD_v2(d_barrier_count, zero, 4)
        nvcuda.cuMemcpyHtoD_v2(d_barrier_phase, zero, 4)

        c_d_bc = ctypes.c_uint64(d_barrier_count.value)
        c_d_bp = ctypes.c_uint64(d_barrier_phase.value)
        c_d_out = ctypes.c_uint64(d_out_cycles.value)
        c_num_b = ctypes.c_uint32(num_blocks)
        c_iters = ctypes.c_uint32(iters)

        args = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_d_bc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_d_bp), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_d_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_num_b), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_iters), ctypes.c_void_p)
        )

        nvcuda.cuLaunchKernel(
            kernel,
            num_blocks, 1, 1,
            threads_per_block, 1, 1,
            0, None, args, None
        )
        nvcuda.cuCtxSynchronize()

        host_cycles = (ctypes.c_uint64 * num_blocks)()
        nvcuda.cuMemcpyDtoH_v2(host_cycles, d_out_cycles, num_blocks * 8)

        avg_cycles_per_barrier = (sum(host_cycles) / len(host_cycles)) / iters
        print(f"Blocks: {num_blocks:2d} (SMs: {min(num_blocks, 14)}) | "
              f"Cycles per device-wide barrier: {avg_cycles_per_barrier:.1f} cycles | "
              f"Latency: {(avg_cycles_per_barrier / 1.785):.2f} ns (at 1785 MHz)")

        results.append({
            "num_blocks": num_blocks,
            "threads_per_block": threads_per_block,
            "barrier_iterations": iters,
            "avg_cycles_per_barrier": avg_cycles_per_barrier,
            "latency_ns_at_1785mhz": avg_cycles_per_barrier / 1.785
        })

    payload = {
        "gpu": "NVIDIA GeForce GTX 1650 (TU117 Turing SM 7.5)",
        "wddm_version": "WDDM 2.7",
        "cooperative_launch_supported": coop_supported.value,
        "sm_count": sm_count.value,
        "results": results
    }

    with open("research/raw/cooperative_grid_sync_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/cooperative_grid_sync_results.json")
    nvcuda.cuMemFree_v2(d_barrier_count)
    nvcuda.cuMemFree_v2(d_barrier_phase)
    nvcuda.cuMemFree_v2(d_out_cycles)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_cooperative_grid_sync()
