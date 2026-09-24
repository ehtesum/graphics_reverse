#!/usr/bin/env python3
"""
tools/test_gpu_driver.py
Directly launches compiled .cubin kernels on the physical GTX 1650 GPU
using the native Windows NVIDIA driver API (nvcuda.dll).
"""

import ctypes
import os
import sys

# Load NVIDIA Driver API
try:
    cuda = ctypes.WinDLL("nvcuda.dll")
except Exception as e:
    print(f"Failed to load nvcuda.dll: {e}")
    sys.exit(1)

# Check Driver Version
def check(res, name="CUDA call"):
    if res != 0:
        raise RuntimeError(f"{name} failed with CUDA error code: {res}")

check(cuda.cuInit(0), "cuInit")

count = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
print(f"CUDA Devices Detected: {count.value}")

device = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

name_buf = ctypes.create_string_buffer(256)
check(cuda.cuDeviceGetName(name_buf, 256, device), "cuDeviceGetName")
dev_name = name_buf.value.decode("utf-8")
print(f"Active Device: {dev_name}")

# Create Context
context = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")

# Load CUBIN Module
module = ctypes.c_void_p()
cubin_path = os.path.abspath("test_001.cubin").encode("utf-8")
check(cuda.cuModuleLoad(ctypes.byref(module), cubin_path), "cuModuleLoad")

# Get Function
function = ctypes.c_void_p()
check(cuda.cuModuleGetFunction(ctypes.byref(function), module, b"_Z4testPf"), "cuModuleGetFunction")

# Allocate GPU Memory
N = 1024 * 1024
num_bytes = N * 4  # 4 MiB for 1M floats

d_mem = ctypes.c_uint64()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), "cuMemAlloc")

# Prepare Host Data (all 1.0f)
h_data = (ctypes.c_float * N)(*([1.0] * N))
check(cuda.cuMemcpyHtoD_v2(d_mem, h_data, num_bytes), "cuMemcpyHtoD")

# Events for timing
start_evt = ctypes.c_void_p()
stop_evt = ctypes.c_void_p()
check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate start")
check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate stop")

# Kernel Arguments
# Pointer to d_mem
param_ptr = ctypes.c_uint64(d_mem.value)
kernel_args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(param_ptr), ctypes.c_void_p))

# Launch Kernel (4096 blocks, 256 threads)
check(cuda.cuEventRecord(start_evt, None), "cuEventRecord start")
check(cuda.cuLaunchKernel(
    function,
    4096, 1, 1,    # Grid: 4096 blocks
    256, 1, 1,     # Block: 256 threads
    0, None,       # Shared memory, Stream
    kernel_args, None
), "cuLaunchKernel")
check(cuda.cuEventRecord(stop_evt, None), "cuEventRecord stop")
check(cuda.cuEventSynchronize(stop_evt), "cuEventSynchronize")

elapsed_ms = ctypes.c_float()
check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start_evt, stop_evt), "cuEventElapsedTime")

# Copy Back & Verify
out_data = (ctypes.c_float * N)()
check(cuda.cuMemcpyDtoH_v2(out_data, d_mem, num_bytes), "cuMemcpyDtoH")

print(f"\n[EXECUTION SUCCESS on {dev_name}]")
print(f"Kernel Execution Time: {elapsed_ms.value:.6f} ms")
print(f"Elements Processed:    {N:,}")
print(f"First 5 Values:        {[round(out_data[i], 2) for i in range(5)]}")
print(f"Expected:              [2.0, 2.0, 2.0, 2.0, 2.0]")
verified = all(abs(out_data[i] - 2.0) < 1e-4 for i in range(1024))
print(f"Verified:              {verified}")

# Cleanup
cuda.cuMemFree_v2(d_mem)
cuda.cuCtxDestroy_v2(context)
