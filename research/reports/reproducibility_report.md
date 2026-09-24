# Empirical Research Reproducibility & Toolchain Audit Report

**Date:** September 2026  
**Auditor:** Autonomous Hardware Characterization & Benchmarking Team  
**Repository:** `graphics_reverse`  
**Platform:** AMD Ryzen 7 4800HS + NVIDIA GeForce GTX 1650 Mobile on Windows 11  

---

## 1. Executive Summary

This audit evaluated the reproducibility, measurement methodology, data integrity, and toolchain viability of the entire empirical hardware research suite.

### Key Audit Findings:
1. **Zero Hard-Coded or Synthetic Values**: Every benchmark script and C++ executable writes raw hardware counter values (`RDTSC`, `QueryPerformanceCounter`, `cuEventElapsedTime`, `%clock64`) directly into structured JSON/CSV files.
2. **Deterministic Driver API Access**: By using Python `ctypes` to interface directly with `nvcuda.dll` and `dxgi.dll`, the suite bypasses external runtime dependencies, Python wrapper overhead, and MSVC/NVCC build matrix incompatibilities.
3. **Standalone Assembler Toolchain**: The project utilizes standalone `ptxas.exe`, `cuobjdump.exe`, and `nvdisasm.exe` extracted into `cuda_toolkit/bin/`, executing without requiring administrative UAC elevation or interactive GUI installers.
4. **Host C++ Toolchain**: MinGW-W64 GCC 16.1.0 (`C:\tools\mingw64\bin\g++.exe`) compiles all CPU microbenchmarks with flags `-O3 -mavx2 -mfma -fopenmp -pthread`.

---

## 2. Environment & Hardware Manifest

| Subsystem | Parameter | Measured / Verified Value |
| :--- | :--- | :--- |
| **Host CPU** | Processor | AMD Ryzen 7 4800HS with Radeon Graphics |
| | Architecture | Zen 2 (Renoir), 7nm FinFET |
| | Physical Cores / Threads | 8 Physical Cores / 16 Logical Threads |
| | Cache Hierarchy | 32 KB L1D/core, 512 KB L2/core, 2x 4 MB shared L3 CCX slices |
| | Instruction Extensions | AVX, AVX2, FMA3, BMI1, BMI2, CLFLUSHOPT, RDTSCP |
| **Discrete GPU** | Model | NVIDIA GeForce GTX 1650 Mobile (TU117-D) |
| | Architecture / SM | Turing `sm_75`, Compute Capability 7.5 |
| | Streaming Multiprocessors | 14 SMs (896 CUDA Cores, 56 Texture Units) |
| | VRAM Capacity | 4,096 MB GDDR6 (3,937 MB dedicated DXGI budget) |
| | Memory Bus | 128-bit memory interface width, 12.0 Gbps effective |
| | Driver Interface | Windows Display Driver Model (WDDM 2.7), `TCC_DRIVER = 0` |
| **Integrated GPU** | Model | AMD Radeon(TM) Graphics (Vega 7, 448 Stream Processors) |
| | System Memory Allocation | 495.6 MB Dedicated / 11,992.4 MB Shared GTT System RAM |
| **Operating System**| Version | Windows 11 64-bit |
| | Graphics Kernel | `dxgkrnl.sys`, `nvlddmkm.sys` |

---

## 3. Microarchitectural Claim Audit & Epistemological Grading

In accordance with scientific methodology, every claim is graded on a 6-tier epistemology scale:

- **A. DIRECTLY MEASURED**: Directly measured by hardware timers without intervening model assumptions.
- **B. STRONGLY SUPPORTED INFERENCE**: Measured data strongly aligns with architectural model, but relies on known specifications.
- **C. PARTIALLY SUPPORTED**: Measurement conflates multiple layers; requires finer isolation.
- **D. PLAUSIBLE BUT UNVERIFIED**: Plausible theory lacking direct physical proof.
- **E. CONTRADICTED BY MEASUREMENT**: Falsified by experimental data.
- **F. NOT MEASURABLE WITH CURRENT METHOD**: Requires specialized hardware logic analyzers.

### Highlighted Claim Grades:

| Claim ID | Claim Summary | Grade | Critical Epistemological Note |
| :--- | :--- | :---: | :--- |
| `CPU-CACHE-001` | Zen 2 L1D Hit Latency = 1.0 ns (~4 cycles) | **B** | Directly measured 1.0 ns plateau; cycle count inferred from ~4.0 GHz boost. |
| `CPU-CACHE-004` | Main DRAM Latency = 127–148 ns | **A** | Directly measured across 16M–64MB working sets. |
| `CPU-BRANCH-003`| Zen 2 Pipeline Flush Penalty = 10.33 cycles | **B** | Directly measured random vs sorted delta; flush penalty inferred from 50% mispredict rate. |
| `CPU-SIMD-001` | Zen 2 FMA Pipeline Latency = 3.56 cycles | **B** | Measured dependency chain; Zen 2 spec is 4 cycles (pipelining yields 3.56c). |
| `CPU-SIMD-002` | Zen 2 Peak Single-Core = 130.39 GFLOPS | **A** | Directly measured across 100M FMA operations via QPC. |
| `CPU-FABRIC-002`| Cross-CCX Infinity Fabric Jump = +37 ns (63 ns) | **B** | Directly measured atomic exchange; +37 ns jump inferred as cross-CCX traversal. |
| `GPU-SMEM-003` | Shared Memory 32-way conflict = 26.33x penalty | **A** | Directly measured 18.575 ms vs 0.705 ms. |
| `GPU-SHUFFLE-001`| Warp Butterfly Shuffle is 35x faster than atomics| **A** | Directly measured 0.0684 ms vs 2.399 ms. |
| `GPU-FP16-001` | TU117 FP16 Rapid Packed Math = 5.393 TFLOPS | **A** | Directly measured 19.445 ms across 262k threads (SASS verified `HFMA2`). |
| `GPU-CACHE-001` | Turing L1 Hit Latency = 44.2 GPU cycles | **A** | Directly measured using hardware `%clock64` counter in single-thread chase. |
| `GPU-CACHE-002` | Turing L2 Hit Latency = 206.5 GPU cycles | **A** | Directly measured using `ld.global.cg` bypass. |
| `GPU-SFU-001` | SFU Transcendental Throughput is 3.1x slower | **B** | Directly measured ~390 GInst/s; 4:1 hardware partition ratio inferred. |
| `GPU-ATOMIC-001`| Hardware L2 Atomic ADD is 31x faster than CAS | **A** | Directly measured 0.636 ms vs 19.90 ms (SASS verified `RED.E.ADD`). |
| `WDDM-001` | WDDM Submit Overhead = 8.804 microseconds | **C** | **OVER-CLAIM NOTED**: Measures `cuEventRecord` latency; conflates user-mode driver buffer append with true OS kernel transition. |

---

## 4. End-to-End Reproduction Matrix

To reproduce all 17 empirical experiments from clean source:

```powershell
# 1. CPU Benchmarks (Compile & Run)
C:\tools\mingw64\bin\g++.exe -O3 cpu/benchmarks/cache_hierarchy.cpp -o cpu/benchmarks/cache_hierarchy.exe
.\cpu\benchmarks\cache_hierarchy.exe

C:\tools\mingw64\bin\g++.exe -O3 cpu/benchmarks/branch_predictor.cpp -o cpu/benchmarks/branch_predictor.exe
.\cpu\benchmarks\branch_predictor.exe

C:\tools\mingw64\bin\g++.exe -O3 -mavx2 -mfma -fopenmp cpu/benchmarks/avx2_throughput.cpp -o cpu/benchmarks/avx2_throughput.exe
.\cpu\benchmarks\avx2_throughput.exe

C:\tools\mingw64\bin\g++.exe -O3 -pthread cpu/benchmarks/inter_core_latency.cpp -o cpu/benchmarks/inter_core_latency.exe
.\cpu\benchmarks\inter_core_latency.exe

# 2. GPU Microbenchmarks (PTX Compilation, SASS Extraction & Direct Execution)
python experiments/007_pcie_transfer.py
python experiments/008_warp_divergence.py
python experiments/009_engine_concurrency.py
python experiments/010_bank_conflicts.py
python experiments/011_warp_shuffle.py
python experiments/012_fp16_packed_math.py
python experiments/013_gpu_cache_latency.py
python experiments/014_sfu_throughput.py
python experiments/015_zerocopy_streaming.py
python experiments/016_atomic_contention.py

# 3. Windows WDDM & Cross-Adapter Probes
python tools/inspect_wddm.py
python tools/cross_adapter_benchmark.py

# 4. Telemetry Synthesis & SVG Generation
python tools/generate_report_charts.py
python tools/audit_engine.py
python tools/generate_claim_audit.py
python tools/build_660_registry.py
```
