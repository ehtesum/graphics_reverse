#!/usr/bin/env python3
"""
research/experiments/exp_soc_fabric_contention.py
Forensic Characterization: Corner C1 - Monolithic SoC Infinity Fabric Contention
Hardware Target: AMD Ryzen 7 4800HS APU (Zen 2 CPU + Vega 7 iGPU gfx90c on monolithic die)

Methodology:
1. Use OpenCL to run a high-intensity memory traffic kernel on the on-die AMD Radeon Vega 7 iGPU (gfx90c).
2. Measure CPU memory bandwidth (AVX2 streaming read/write) and DRAM pointer-chasing latency
   under two conditions:
     a) IDLE iGPU (Zero graphics fabric traffic)
     b) STRESSED iGPU (Vega 7 saturating the shared Infinity Fabric / DDR4-3200 memory controller)
3. Quantify the exact arbitration penalty, bandwidth drop, and latency inflation imposed by iGPU co-execution.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "research" / "raw"
CPU_BIN = ROOT / "cpu" / "benchmarks"
OUT_FILE = RAW_DIR / "soc_fabric_contention_results.json"

# OpenCL constants
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_NAME = 0x102B
CL_DEVICE_TYPE_GPU = (1 << 2)
CL_MEM_READ_WRITE = (1 << 0)
CL_MEM_COPY_HOST_PTR = (1 << 5)


def get_amd_opencl():
    try:
        ocl = ctypes.WinDLL("OpenCL.dll")
    except Exception as e:
        print(f"[ERROR] Cannot load OpenCL.dll: {e}")
        return None, None, None

    num_plat = ctypes.c_uint32()
    ocl.clGetPlatformIDs(0, None, ctypes.byref(num_plat))
    platforms = (ctypes.c_void_p * num_plat.value)()
    ocl.clGetPlatformIDs(num_plat.value, platforms, None)

    ocl.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    ocl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
    ocl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    amd_plat = None
    amd_dev = None

    for p in platforms:
        pname = (ctypes.c_char * 256)()
        ocl.clGetPlatformInfo(p, CL_PLATFORM_NAME, 256, pname, None)
        if b"AMD" in pname.value:
            amd_plat = p
            num_dev = ctypes.c_uint32()
            ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(num_dev))
            if num_dev.value > 0:
                devs = (ctypes.c_void_p * num_dev.value)()
                ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_dev.value, devs, None)
                amd_dev = devs[0]
                break

    return ocl, amd_plat, amd_dev


def build_cpu_stressor():
    # Compile a self-contained CPU benchmark that measures DRAM bandwidth and latency
    cpp_code = """
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <x86intrin.h>
#include <windows.h>

void pin(int core) {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << core));
}

int main(int argc, char** argv) {
    pin(0); // Pin to Core 0

    const size_t SZ = 64 * 1024 * 1024; // 64 MB
    char* buf = (char*)_aligned_malloc(SZ, 64);
    memset(buf, 0x5A, SZ);

    // 1. Measure AVX2 Read Bandwidth
    const int READ_ITERS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    __m256i sum = _mm256_setzero_si256();
    for (int iter = 0; iter < READ_ITERS; ++iter) {
        __m256i* ptr = (__m256i*)buf;
        for (size_t i = 0; i < SZ / 32; i += 4) {
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 1));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 2));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 3));
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double read_sec = std::chrono::duration<double>(t1 - t0).count();
    double read_gbps = ((double)SZ * READ_ITERS / 1e9) / read_sec;
    volatile int dummy = _mm256_extract_epi32(sum, 0);
    (void)dummy;

    // 2. Measure AVX2 Streaming Write Bandwidth (NT stores)
    const int WRITE_ITERS = 10;
    __m256i val = _mm256_set1_epi32(0x12345678);
    auto tw0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < WRITE_ITERS; ++iter) {
        __m256i* ptr = (__m256i*)buf;
        for (size_t i = 0; i < SZ / 32; i += 4) {
            _mm256_stream_si256(ptr + i, val);
            _mm256_stream_si256(ptr + i + 1, val);
            _mm256_stream_si256(ptr + i + 2, val);
            _mm256_stream_si256(ptr + i + 3, val);
        }
        _mm_sfence();
    }
    auto tw1 = std::chrono::high_resolution_clock::now();
    double write_sec = std::chrono::duration<double>(tw1 - tw0).count();
    double write_gbps = ((double)SZ * WRITE_ITERS / 1e9) / write_sec;

    // 3. Measure DRAM Pointer Chasing Latency (32 MB footprint, 64-byte stride, randomized)
    const size_t CHASE_SZ = 32 * 1024 * 1024;
    const size_t NUM_NODES = CHASE_SZ / sizeof(void*);
    void** chase_buf = (void**)_aligned_malloc(CHASE_SZ, 64);
    std::vector<size_t> indices(NUM_NODES / 8);
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i * 8; // 64-byte stride
    std::mt19937_64 rng(1337);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        chase_buf[indices[i]] = &chase_buf[indices[i + 1]];
    }
    chase_buf[indices.back()] = &chase_buf[indices[0]];

    // Chase 2,000,000 hops
    const uint64_t HOPS = 20000000;
    void** curr = (void**)chase_buf[indices[0]];
    uint64_t c0 = __rdtsc();
    for (uint64_t k = 0; k < HOPS; ++k) {
        curr = (void**)*curr;
        asm volatile("" : "+r"(curr));
    }
    uint64_t c1 = __rdtsc();
    double cycles_per_hop = (double)(c1 - c0) / (double)HOPS;

    _aligned_free(buf);
    _aligned_free(chase_buf);

    std::cout << read_gbps << " " << write_gbps << " " << cycles_per_hop << std::endl;
    return 0;
}
"""
    src_path = CPU_BIN / "bench_cpu_fabric.cpp"
    exe_path = CPU_BIN / "bench_cpu_fabric.exe"
    with open(src_path, "w") as f:
        f.write(cpp_code)
    subprocess.run(["C:\\tools\\mingw64\\bin\\g++.exe", "-O3", "-mavx2", str(src_path), "-o", str(exe_path)], check=True)
    return exe_path


def run_experiment():
    print("=" * 80)
    print(" CORNER C1: Monolithic SoC Infinity Fabric Contention (Vega 7 vs CPU)")
    print(" Hardware: AMD Ryzen 7 4800HS (Renoir) | Shared Data Fabric + DDR4-3200 UMC")
    print("=" * 80)

    ocl, plat, dev = get_amd_opencl()
    if not ocl or not plat or not dev:
        print("[ERROR] Failed to initialize AMD OpenCL on Vega 7 iGPU.")
        sys.exit(1)

    dname = (ctypes.c_char * 256)()
    ocl.clGetDeviceInfo(dev, CL_DEVICE_NAME, 256, dname, None)
    print(f"[*] Identified Integrated GPU: {dname.value.decode()}")

    # Compile CPU stressor executable
    print("[*] Compiling CPU memory and latency harness...")
    cpu_exe = build_cpu_stressor()

    # Create OpenCL Context, Queue, and Kernel on Vega 7
    ctx_err = ctypes.c_int32()
    ocl.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    ocl.clCreateContext.restype = ctypes.c_void_p
    dev_arr = (ctypes.c_void_p * 1)(dev)
    ctx = ocl.clCreateContext(None, 1, dev_arr, None, None, ctypes.byref(ctx_err))

    ocl.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32)]
    ocl.clCreateCommandQueue.restype = ctypes.c_void_p
    q_err = ctypes.c_int32()
    queue = ocl.clCreateCommandQueue(ctx, dev, 0, ctypes.byref(q_err))

    # Kernel that aggressively streams 128 MB DRAM read/write on the Vega 7 CUs
    kernel_src = """
    __kernel void fabric_stress(__global uint4* src, __global uint4* dst, uint count) {
        uint gid = get_global_id(0);
        uint stride = get_global_size(0);
        for (uint i = gid; i < count; i += stride) {
            uint4 val = src[i];
            dst[i] = val + (uint4)(1, 2, 3, 4);
        }
    }
    """
    c_src = ctypes.c_char_p(kernel_src.encode("utf-8"))
    c_len = ctypes.c_size_t(len(kernel_src))
    b_err = ctypes.c_int32()
    ocl.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int32)]
    ocl.clCreateProgramWithSource.restype = ctypes.c_void_p
    prog = ocl.clCreateProgramWithSource(ctx, 1, ctypes.byref(c_src), ctypes.byref(c_len), ctypes.byref(b_err))

    ocl.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
    ocl.clBuildProgram(prog, 1, dev_arr, b"-cl-fast-relaxed-math", None, None)

    k_err = ctypes.c_int32()
    ocl.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
    ocl.clCreateKernel.restype = ctypes.c_void_p
    kernel = ocl.clCreateKernel(prog, b"fabric_stress", ctypes.byref(k_err))

    # Allocate 128 MB on host-visible shared memory for Vega 7
    IGPU_MEM_SZ = 128 * 1024 * 1024
    m1_err = ctypes.c_int32()
    m2_err = ctypes.c_int32()
    ocl.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    ocl.clCreateBuffer.restype = ctypes.c_void_p
    buf_src = ocl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, IGPU_MEM_SZ, None, ctypes.byref(m1_err))
    buf_dst = ocl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, IGPU_MEM_SZ, None, ctypes.byref(m2_err))

    p_src = ctypes.c_void_p(buf_src)
    p_dst = ctypes.c_void_p(buf_dst)
    num_vecs = ctypes.c_uint32(IGPU_MEM_SZ // 16)
    ocl.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p]
    ocl.clSetKernelArg(kernel, 0, ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p)), ctypes.byref(p_src))
    ocl.clSetKernelArg(kernel, 1, ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p)), ctypes.byref(p_dst))
    ocl.clSetKernelArg(kernel, 2, ctypes.c_size_t(4), ctypes.byref(num_vecs))

    # --- Step 1: Benchmark Baseline (Idle Vega 7 iGPU) ---
    print("\n[1/2] Benchmarking CPU Memory Performance with IDLE Vega 7 iGPU...")
    out_idle = subprocess.run([str(cpu_exe)], capture_output=True, text=True, check=True).stdout.strip().split()
    idle_read_gbps = float(out_idle[0])
    idle_write_gbps = float(out_idle[1])
    idle_chase_cycles = float(out_idle[2])

    print(f"  - CPU Read Bandwidth    : {idle_read_gbps:.2f} GB/s")
    print(f"  - CPU NT Write Bandwidth: {idle_write_gbps:.2f} GB/s")
    print(f"  - CPU DRAM Chase Latency: {idle_chase_cycles:.2f} cycles/hop")

    # --- Step 2: Benchmark Contention (Vega 7 Aggressive Fabric Flooding) ---
    print("\n[2/2] Launching continuous OpenCL memory flooding on Vega 7 iGPU...")
    stop_igpu = threading.Event()
    ocl.clFinish.argtypes = [ctypes.c_void_p]

    def igpu_flooder():
        global_work = ctypes.c_size_t(65536) # 64K threads
        local_work = ctypes.c_size_t(256)
        ocl.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        while not stop_igpu.is_set():
            ocl.clEnqueueNDRangeKernel(queue, kernel, 1, None, ctypes.byref(global_work), ctypes.byref(local_work), 0, None, None)
            ocl.clFinish(queue)

    flooder_thread = threading.Thread(target=igpu_flooder)
    flooder_thread.daemon = True
    flooder_thread.start()

    time.sleep(0.5) # Let fabric stress stabilize

    out_stress = subprocess.run([str(cpu_exe)], capture_output=True, text=True, check=True).stdout.strip().split()
    stress_read_gbps = float(out_stress[0])
    stress_write_gbps = float(out_stress[1])
    stress_chase_cycles = float(out_stress[2])

    stop_igpu.set()
    flooder_thread.join(timeout=2.0)

    print(f"  - CPU Read Bandwidth (Under iGPU Stress)    : {stress_read_gbps:.2f} GB/s ({(stress_read_gbps / idle_read_gbps) * 100.0:.1f}% of baseline)")
    print(f"  - CPU NT Write Bandwidth (Under iGPU Stress): {stress_write_gbps:.2f} GB/s ({(stress_write_gbps / idle_write_gbps) * 100.0:.1f}% of baseline)")
    print(f"  - CPU DRAM Chase Latency (Under iGPU Stress): {stress_chase_cycles:.2f} cycles/hop ({(stress_chase_cycles / idle_chase_cycles):.2f}x inflation)")

    read_drop_pct = (1.0 - (stress_read_gbps / idle_read_gbps)) * 100.0
    write_drop_pct = (1.0 - (stress_write_gbps / idle_write_gbps)) * 100.0
    latency_inflation = stress_chase_cycles / idle_chase_cycles

    results = {
        "hardware": {
            "soc": "AMD Ryzen 7 4800HS (Renoir)",
            "cpu_arch": "Zen 2 (8C/16T, 2x CCX)",
            "igpu_arch": "Radeon RX Vega 7 (gfx90c, 7 CUs)",
            "interconnect": "AMD Infinity Fabric (Data Fabric)",
            "memory": "Dual-Channel DDR4-3200"
        },
        "baseline_idle_igpu": {
            "cpu_read_bandwidth_gbps": round(idle_read_gbps, 2),
            "cpu_nt_write_bandwidth_gbps": round(idle_write_gbps, 2),
            "cpu_dram_latency_cycles": round(idle_chase_cycles, 2)
        },
        "under_igpu_fabric_stress": {
            "cpu_read_bandwidth_gbps": round(stress_read_gbps, 2),
            "cpu_nt_write_bandwidth_gbps": round(stress_write_gbps, 2),
            "cpu_dram_latency_cycles": round(stress_chase_cycles, 2)
        },
        "impact_metrics": {
            "read_bandwidth_drop_percent": round(read_drop_pct, 2),
            "write_bandwidth_drop_percent": round(write_drop_pct, 2),
            "latency_inflation_multiplier": round(latency_inflation, 2)
        }
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUMMARY] SoC Fabric Contention Impact:")
    print(f"  - Read Bandwidth Degradation : -{read_drop_pct:.1f}%")
    print(f"  - Write Bandwidth Degradation: -{write_drop_pct:.1f}%")
    print(f"  - Memory Latency Inflation   : +{(latency_inflation - 1.0) * 100.0:.1f}% ({idle_chase_cycles:.1f}c -> {stress_chase_cycles:.1f}c)")
    print(f"[SUCCESS] Telemetry recorded in: {OUT_FILE}\n")


if __name__ == "__main__":
    run_experiment()
