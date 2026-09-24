import ctypes
import os
import json
import time

# Corner H6: PCIe Small-Packet Round-Trip Latency & Framing Knee
# Target: AMD Ryzen 7 4800HS <-> PCIe 3.0 x8 <-> NVIDIA GTX 1650

def run_pcie_small_packet_benchmark():
    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit(0)

    dev = ctypes.c_int()
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)

    ctx = ctypes.c_void_p()
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    # 64-bit Argtypes
    nvcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    nvcuda.cuMemAlloc_v2.restype = ctypes.c_int
    nvcuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
    nvcuda.cuMemFree_v2.restype = ctypes.c_int
    nvcuda.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    nvcuda.cuMemHostAlloc.restype = ctypes.c_int
    nvcuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    nvcuda.cuMemFreeHost.restype = ctypes.c_int
    nvcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    nvcuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    nvcuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    nvcuda.cuMemcpyDtoH_v2.restype = ctypes.c_int

    # Packet sizes: 4 B, 16 B, 64 B, 256 B, 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB
    packet_sizes = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    results = []

    print("Benchmarking PCIe 3.0 x8 Small-Packet Ping-Pong Latency & Bandwidth Knee...")
    print(f"{'Packet Size':<14} | {'Bytes':<10} | {'Round-Trip (µs)':<18} | {'One-Way (µs)':<16} | {'Effective BW (MB/s)'}")
    print("-" * 80)

    # Allocate host pinned and device memory
    MAX_SIZE = 1048576
    h_buf = ctypes.c_void_p()
    d_buf = ctypes.c_void_p()
    nvcuda.cuMemHostAlloc(ctypes.byref(h_buf), MAX_SIZE, 0)
    nvcuda.cuMemAlloc_v2(ctypes.byref(d_buf), MAX_SIZE)

    # Initialize host memory
    ctypes.memset(h_buf, 0x5A, MAX_SIZE)

    ITERS = 1000

    for sz in packet_sizes:
        sz_str = f"{sz} B" if sz < 1024 else (f"{sz // 1024} KB" if sz < 1048576 else "1 MB")

        # Warmup
        for _ in range(20):
            nvcuda.cuMemcpyHtoD_v2(d_buf, h_buf, sz)
            nvcuda.cuMemcpyDtoH_v2(h_buf, d_buf, sz)

        # High-resolution CPU timer (QPC)
        t0 = time.perf_counter_ns()
        for _ in range(ITERS):
            nvcuda.cuMemcpyHtoD_v2(d_buf, h_buf, sz)
            nvcuda.cuMemcpyDtoH_v2(h_buf, d_buf, sz)
        t1 = time.perf_counter_ns()

        round_trip_us = ((t1 - t0) / ITERS) / 1000.0
        one_way_us = round_trip_us / 2.0

        # Effective bandwidth = (2 * sz bytes) / round_trip_time
        effective_mb_s = (2.0 * sz / 1e6) / (round_trip_us / 1e6)

        print(f"{sz_str:<14} | {sz:<10d} | {round_trip_us:<18.2f} | {one_way_us:<16.2f} | {effective_mb_s:8.2f}")

        results.append({
            "packet_label": sz_str,
            "bytes": sz,
            "round_trip_latency_us": round_trip_us,
            "one_way_latency_us": one_way_us,
            "effective_bandwidth_mb_s": effective_mb_s
        })

    payload = {
        "interconnect": "PCIe 3.0 x8",
        "cpu": "AMD Ryzen 7 4800HS",
        "gpu": "NVIDIA GeForce GTX 1650",
        "os": "Windows 11 Build 22631 (WDDM 2.7)",
        "results": results
    }

    with open("research/raw/pcie_small_packet_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to research/raw/pcie_small_packet_results.json")
    nvcuda.cuMemFree_v2(d_buf)
    nvcuda.cuMemFreeHost(h_buf)
    nvcuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    run_pcie_small_packet_benchmark()
