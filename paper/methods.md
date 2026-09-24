# Empirical Methodology & Instrumentation Guide

This document details the experimental methodology, timing primitives, compilation toolchains, and statistical procedures used to characterize the NVIDIA TU117 and AMD Zen 2 microarchitectures.

---

## 1. Timing Primitives & Calibration

### 1.1 CPU Cycle-Accurate Timing (`RDTSC` with Pipeline Serialization)
To measure CPU instruction latencies and memory hierarchies without interference from out-of-order execution, all CPU microbenchmarks employ the x86-64 `RDTSC` instruction bracketed by full memory fences:

```cpp
_mm_mfence();
uint64_t start = __rdtsc();
// ... kernel loop ...
_mm_mfence();
uint64_t stop = __rdtsc();
uint64_t cycles = stop - start;
```

* **Overhead Calibration**: The intrinsic latency of the `_mm_mfence() + __rdtsc()` sequence was measured across $1,000,000$ iterations and determined to be **18.2 cycles**. This baseline overhead is subtracted from all differential measurements.
* **Affinity Pinning**: CPU benchmarks are bound to physical cores via `SetThreadAffinityMask(GetCurrentThread(), mask)` to eliminate OS scheduler thread migrations across CCXs.

### 1.2 GPU Timing Primitives
Two complementary timing mechanisms are employed for GPU execution:
1. **CUDA Events (`cuEventRecord`, `cuEventElapsedTime`)**:
   Resolution: ~0.5 µs. Used for kernel execution duration and asynchronous PCIe transfer timings.
2. **In-Shader Hardware Clock Register (`%clock64`)**:
   Resolution: 1 GPU core cycle (~0.6 ns at 1665 MHz). Used in pointer-chasing loops (`experiments/013_gpu_cache_latency.py`) to measure cycle latency directly on the streaming multiprocessor without driver or PCIe measurement noise.

### 1.3 Host High-Resolution Timer (`QueryPerformanceCounter`)
Host-side driver overhead and WDDM submission latencies are profiled using Windows `QueryPerformanceCounter` (QPC) via Python's `time.perf_counter()`. QPC frequency on this platform is calibrated at **10.0 MHz** (100 ns nominal resolution).

---

## 2. Compilation Toolchains & Flags

### 2.1 CUDA PTX & CUBIN Toolchain
To bypass compiler optimizations and control instruction generation at the SASS level, benchmarks are authored in PTX 7.5 targeting `sm_75`:

```bash
ptxas.exe -arch=sm_75 kernel.ptx -o kernel.cubin
cuobjdump.exe --dump-sass kernel.cubin > kernel_sass.txt
```

* **PTX Assembler**: NVIDIA `ptxas` Version 12.5 (from `cuda_toolkit/bin/`).
* **SASS Verification**: Generated CUBINs are disassembled via `cuobjdump` to verify instruction sequences, register allocation, and memory transaction encodings.

### 2.2 Host C++ Toolchain
CPU benchmarks are compiled using MinGW-w64 GCC 13.2.0:

```bash
g++.exe -O3 -mavx2 -mfma -march=znver2 benchmark.cpp -o benchmark.exe
```

---

## 3. Statistical Rigor & Outlier Filtering

Every quantitative result in this study adheres to the following protocol:
1. **Warmup Phase**: Every benchmark executes at least 5 unmeasured warmup iterations to warm CPU/GPU caches, establish power boost clocks, and stabilize TLB state.
2. **Repetitions**: Microbenchmarks execute a minimum of 50 independent trials (CPU) or 20 trials (GPU).
3. **Statistical Metrics Reported**:
   - Arithmetic Mean ($\mu$)
   - Median ($\tilde{x}$)
   - Standard Deviation ($\sigma$)
   - Coefficient of Variation ($CV = \sigma / \mu$)
   - 95% Confidence Interval ($CI_{95} = \mu \pm 1.96 \frac{\sigma}{\sqrt{N}}$)
4. **Outlier Filtering**: Modified Z-score filtering ($M_i > 3.5$) is applied to discard samples corrupted by Windows background thread interrupts.
