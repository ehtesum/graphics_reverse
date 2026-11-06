# Silicon Micro-Architectural Benchmarking Suite (`graphics_reverse`)

An empirical evaluation framework for characterizing micro-architectural execution bottlenecks, hardware memory hierarchies, and instruction-level parallelism across modern GPU and CPU architectures.

[![Language: CUDA / C++17](https://img.shields.io/badge/Language-CUDA%20%7C%20C%2B%2B17-green.svg)](gpu/)
[![Target: NVIDIA Turing / AMD Zen 2](https://img.shields.io/badge/Hardware-Turing%20%7C%20Zen%202-blue.svg)](gpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Modern hardware architectures employ complex abstractions (SIMT execution models, hierarchical caches, banked SRAM, out-of-order execution) that frequently deviate from theoretical peak specifications under contentious workloads.

This repository provides bare-metal, low-level micro-benchmarks designed to empirically isolate and quantify hardware latency, warp divergence penalties, bank serialization, and cache tier transitions on:
* **GPU Target:** NVIDIA GeForce GTX 1650 (Turing Architecture, TU117, Compute Capability 7.5)
* **CPU Target:** AMD Ryzen 7 4800HS (Zen 2 Architecture, 8C/16T, Renoir)

---

## Empirical Micro-Benchmark Suite

| Experiment ID | Kernel / Benchmark Focus | Micro-Architectural Target | Measured Performance & Empirical Findings |
| :--- | :--- | :--- | :--- |
| **`EXP_008`** | **Warp Divergence Penalty** | SIMT branch convergence | Profiles execution serialization when threads within a 32-lane warp evaluate divergent control paths (`if`/`else`). Quantifies re-convergence latency. |
| **`EXP_010`** | **Shared Memory Bank Conflicts** | 32-bank SRAM crossbar | Evaluates 32-way shared memory conflict penalties. Accessing shared memory with stride 32 triggers a **25.6x serialization slowdown** (7.37 ms vs. 0.29 ms unconflicted). |
| **`EXP_011`** | **Intra-Warp Shuffle Register Exchange** | SM register file / `__shfl_sync` | Measures direct register-to-register data exchange avoiding intermediate SRAM/DRAM staging; profiles bandwidth vs. shared memory barriers. |
| **`EXP_012`** | **Packed FP16 Arithmetic Throughput** | Dual-issue tensor / FP16 ALUs | Evaluates SIMD packed half-precision (`half2`) vector instructions yielding up to **1.94x peak arithmetic throughput** over FP32 baselines. |
| **`EXP_013`** | **Memory Hierarchy Cache Latency** | Pointer-chasing latency tiers | Measures exact clock cycles required across memory tiers: **L1 Cache (~45 cycles)** $\rightarrow$ **L2 Cache (~209 cycles)** $\rightarrow$ **Device Global DRAM (~390 cycles)**. |
| **`EXP_014`** | **Special Function Unit (SFU) Saturation** | Transcendental ALUs (`sin`, `rsqrt`) | Characterizes throughput limits and instruction issue rates for non-IEEE transcendental function approximations. |
| **`EXP_016`** | **Atomic Memory Contention** | L2 / ROP atomic crossbar | Contrasts compare-and-swap (CAS) spin-loops against hardware-native atomic operations under high thread contention. |

---

## Repository Structure

```
graphics_reverse/
├── run_benchmarks.py    # Main CLI benchmark orchestrator & telemetry harness
├── gpu/                 # CUDA C++ kernels, PTX inline assembly, and driver wrappers
├── cpu/                 # C++17 CPU benchmarks (AVX2 256-bit SIMD, cache line stride tests)
├── paper/               # Micro-architectural findings, latency plots, and technical writeups
└── results/             # Raw measurement logs, clock-cycle distributions, and CSV outputs
```

---

## Compilation & Execution

### Prerequisites
* NVIDIA CUDA Toolkit 11.8+ with `nvcc` compiler
* Host C++17 compiler (MSVC 2019+ or GCC 9+)
* Python 3.9+ with `matplotlib` and `numpy` (for telemetry plotting)

### Running Benchmarks
Execute the automated harness via CLI:

```bash
# List all registered micro-benchmarks
python run_benchmarks.py --list

# Execute shared memory bank conflict benchmark (EXP_010)
python run_benchmarks.py --exp 010

# Execute warp divergence benchmark (EXP_008)
python run_benchmarks.py --exp 008
```

---

## Technical Considerations & Measurement Controls

* **Micro-Architecture Specificity**: Kernel calibration and bank stride mappings target NVIDIA Turing (SM 7.5); newer architectures (Ampere, Ada Lovelace, Hopper) employ unified L1 data cache/shared memory architectures that shift contention thresholds.
* **Driver Dispatch Overhead**: Under Windows Display Driver Model (WDDM), runtime kernel launches incur ~5–10 $\mu\text{s}$ scheduling latency compared to native Linux TCC (Tesla Compute Cluster) runtimes.
* **Clock Jitter Mitigation**: Automated GPU Boost frequency scaling is stabilized across benchmark iterations using fixed-iteration warmup passes and core pinning.
