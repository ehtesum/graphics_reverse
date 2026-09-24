import ctypes
import os
import json
import time

# Corner H5: Host Pinned Memory Allocation vs On-the-Fly Registration (cudaHostRegister)
# Target: AMD Ryzen 7 4800HS Host DDR4 <-> PCIe 3.0 x8 <-> NVIDIA GTX 1650

def run_pinned_registration_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # 64-bit Argtypes
    nvcuda.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    nvcuda.cuMemHostRegister_v2.restype = ctypes.c_int
    nvcuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    nvcuda.cuMemHostUnregister.restype = ctypes.c_int
    nvcuda.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    nvcuda.cuMemHostAlloc.restype = ctypes.c_int
    nvcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    nvcuda.cuMemAlloc_v2.restype = ctypes.c_int
    nvcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    nvcuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    nvcuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
    nvcuda.cuMemFree_v2.restype = ctypes.c_int
    nvcuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    nvcuda.cuMemFreeHost.restype = ctypes.c_int

    # CUDA Constants
    CU_MEMHOSTALLOC_PORTABLE = 0x01
    CU_MEMHOSTREGISTER_PORTABLE = 0x01

    # Buffer sizes: 64 KB, 1 MB, 4 MB, 16 MB, 64 MB, 128 MB
    sizes_kb = [64, 1024, 4096, 16384, 65536, 131072]
    results = []

    print("Benchmarking Host Memory Registration (cudaHostRegister) vs Pre-allocated Pinned (cudaMemHostAlloc)...")
    print(f"{'Size':<10} | {'cuMemHostAlloc (ms)':<20} | {'cuMemHostRegister (ms)':<22} | {'H2D Pinned BW (GB/s)':<22} | {'H2D Reg BW (GB/s)'}")
    print("-" * 95)

    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    nvcuda.cuEventCreate(ctypes.byref(start_ev), 0)
    nvcuda.cuEventCreate(ctypes.byref(stop_ev), 0)

    for sz_kb in sizes_kb:
        n_bytes = sz_kb * 1024
        sz_str = f"{sz_kb} KB" if sz_kb < 1024 else f"{sz_kb // 1024} MB"

        # 1. Measure cuMemHostAlloc latency
        t0 = time.perf_counter()
        h_pinned = ctypes.c_void_p()
        nvcuda.cuMemHostAlloc(ctypes.byref(h_pinned), n_bytes, 0)
        t1 = time.perf_counter()
        alloc_ms = (t1 - t0) * 1000.0

        # 2. Measure VirtualAlloc + cuMemHostRegister latency
        kernel32 = ctypes.WinDLL("kernel32.dll")
        MEM_COMMIT = 0x1000
        MEM_RESERVE = 0x2000
        PAGE_READWRITE = 0x04
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        kernel32.VirtualFree.restype = ctypes.c_int

        h_virt = kernel32.VirtualAlloc(None, n_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        # Touch pages to commit physical RAM
        ctypes.memset(h_virt, 0x55, n_bytes)

        t2 = time.perf_counter()
        nvcuda.cuMemHostRegister_v2(h_virt, n_bytes, 0)
        t3 = time.perf_counter()
        reg_ms = (t3 - t2) * 1000.0

        # 3. Measure PCIe H2D Bandwidth for both
        d_mem = ctypes.c_void_p()
        nvcuda.cuMemAlloc_v2(ctypes.byref(d_mem), n_bytes)

        # H2D from Pinned
        nvcuda.cuEventRecord(start_ev, None)
        nvcuda.cuMemcpyHtoD_v2(d_mem, h_pinned, n_bytes)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)
        ms_pinned = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms_pinned), start_ev, stop_ev)
        bw_pinned = (n_bytes / 1e9) / (ms_pinned.value / 1000.0)

        # H2D from Registered
        nvcuda.cuEventRecord(start_ev, None)
        nvcuda.cuMemcpyHtoD_v2(d_mem, h_virt, n_bytes)
        nvcuda.cuEventRecord(stop_ev, None)
        nvcuda.cuEventSynchronize(stop_ev)
        ms_reg = ctypes.c_float()
        nvcuda.cuEventElapsedTime(ctypes.byref(ms_reg), start_ev, stop_ev)
        bw_reg = (n_bytes / 1e9) / (ms_reg.value / 1000.0)

        print(f"{sz_str:<10} | {alloc_ms:<20.3f} | {reg_ms:<22.3f} | {bw_pinned:<22.2f} | {bw_reg:.2f}")

        results.append({
            "size_kb": sz_kb,
            "alloc_host_latency_ms": alloc_ms,
            "register_host_latency_ms": reg_ms,
            "pinned_h2d_bandwidth_gb_s": bw_pinned,
            "registered_h2d_bandwidth_gb_s": bw_reg
        })

        nvcuda.cuMemFree_v2(d_mem)
        nvcuda.cuMemHostUnregister(h_virt)
        kernel32.VirtualFree(h_virt, 0, 0x8000) # MEM_RELEASE
        nvcuda.cuMemFreeHost(h_pinned)

    payload = {
        "host_cpu": "AMD Ryzen 7 4800HS",
        "gpu": "NVIDIA GeForce GTX 1650",
        "interconnect": "PCIe 3.0 x8",
        "os": "Windows 11 Build 22631 (WDDM 2.7)",
        "results": results
    }

    with open("research/raw/host_pinned_registration_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/host_pinned_registration_results.json")
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_pinned_registration_benchmark()
