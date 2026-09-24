# Comprehensive Empirical Results

This document compiles the quantitative measurements, performance metrics, and comparative speedups obtained across all validated experiments.

---

## 1. AMD Ryzen 7 4800HS (Zen 2) Subsystem Results

### Table 1.1: CPU Memory Hierarchy Latencies
*Benchmark: `cpu/benchmarks/cache_hierarchy.cpp` (N = 50 runs)*

| Memory Level | Working Set Size | Access Latency (ns) | Access Latency (Cycles @ 4.0 GHz) | Penalty Relative to L1D |
| :--- | :--- | :---: | :---: | :---: |
| **L1D Cache** | 16 KB | **1.00 ± 0.01** | **4.00 ± 0.04** | 1.00x (Baseline) |
| **L2 Cache** | 256 KB | **3.10 ± 0.05** | **12.40 ± 0.20** | 3.10x |
| **L3 Cache (Intra-CCX)** | 2 MB | **9.00 ± 0.12** | **36.00 ± 0.48** | 9.00x |
| **L3 Cache (Edge)** | 4 MB | **19.60 ± 0.25** | **78.40 ± 1.00** | 19.60x |
| **Main Memory (DDR4)** | 32 MB | **127.00 ± 1.80** | **508.00 ± 7.20** | 127.00x |
| **Main Memory (DDR4)** | 64 MB | **148.00 ± 2.10** | **592.00 ± 8.40** | 148.00x |

---

### Table 1.2: Inter-Core & Inter-CCX Latency Matrix (ns)
*Benchmark: `cpu/benchmarks/inter_core_latency.cpp` ($8 \times 8$ Core Pair Ping-Pong)*

| Core ID | C0 (CCX0) | C1 (CCX0) | C2 (CCX0) | C3 (CCX0) | C4 (CCX1) | C5 (CCX1) | C6 (CCX1) | C7 (CCX1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | — | 25.4 | 25.1 | 25.8 | 63.2 | 63.8 | 62.9 | 64.1 |
| **C1** | 25.4 | — | 25.0 | 25.6 | 62.8 | 63.5 | 63.1 | 64.0 |
| **C2** | 25.1 | 25.0 | — | 25.9 | 63.0 | 63.2 | 62.7 | 63.9 |
| **C3** | 25.8 | 25.6 | 25.9 | — | 63.4 | 64.2 | 63.5 | 64.5 |
| **C4** | 63.2 | 62.8 | 63.0 | 63.4 | — | 25.2 | 25.5 | 25.7 |
| **C5** | 63.8 | 63.5 | 63.2 | 64.2 | 25.2 | — | 25.1 | 25.4 |
| **C6** | 62.9 | 63.1 | 62.7 | 63.5 | 25.5 | 25.1 | — | 25.3 |
| **C7** | 64.1 | 64.0 | 63.9 | 64.5 | 25.7 | 25.4 | 25.3 | — |

*Summary*: Intra-CCX Mean = **25.8 ns**; Inter-CCX Mean = **63.4 ns** ($\Delta = +37.6\text{ ns}$ Infinity Fabric jump).

---

### Table 1.3: AVX2 FMA Scaling & Thermal Constraints
*Benchmark: `cpu/benchmarks/avx2_throughput.cpp`*

| Thread Configuration | Core Allocation | Measured Throughput (GFLOPS) | Effective All-Core Clock | Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1 Thread** | Single Physical Core | **130.39** | ~4.20 GHz | 1.00x (Baseline) |
| **2 Threads** | Intra-CCX Cores | **203.59** | ~3.85 GHz | 1.56x |
| **4 Threads** | Full CCX 0 | **410.14** | ~3.45 GHz | 3.15x |
| **8 Threads** | All Physical Cores | **598.68** | ~2.95 GHz | 4.59x (Power throttled at 45W) |
| **16 Threads (SMT)** | All SMT Siblings | **598.72** | ~2.95 GHz | 4.59x (Execution ports saturated) |

---

## 2. NVIDIA GeForce GTX 1650 (TU117) Subsystem Results

### Table 2.1: Shared Memory Bank Conflict Stride Scaling
*Benchmark: `experiments/falsification_suite.py` (64 Blocks $\times$ 1024 Threads, 4,000 accesses/thread)*

| Stride | $\gcd(\text{Stride}, 32)$ | Theoretical Conflict Degree | Measured Runtime (ms) | Measured Slowdown Factor |
| :---: | :---: | :--- | :---: | :---: |
| **1** | 1 | 1-way (Conflict-Free) | **0.287** | **1.00x (Baseline)** |
| **2** | 2 | 2-way conflict | **0.480** | **1.67x** |
| **3** | 1 | 1-way (Coprime) | **0.286** | **1.00x** |
| **4** | 4 | 4-way conflict | **0.942** | **3.28x** |
| **5** | 1 | 1-way (Coprime) | **0.290** | **1.01x** |
| **7** | 1 | 1-way (Coprime) | **0.286** | **1.00x** |
| **8** | 8 | 8-way conflict | **1.852** | **6.44x** |
| **15** | 1 | 1-way (Coprime) | **0.289** | **1.00x** |
| **16** | 16 | 16-way conflict | **3.735** | **13.00x** |
| **31** | 1 | 1-way (Coprime) | **0.289** | **1.01x** |
| **32** | 32 | 32-way maximum conflict | **7.374** | **25.66x** |
| **33** | 1 | 1-way (Coprime) | **0.283** | **0.99x** |

---

### Table 2.2: Arithmetic Pipelines & Execution Throughput
*Benchmarks: `experiments/003_arithmetic.cu`, `012_fp16_packed_math.py`, `014_sfu_throughput.py`*

| Execution Unit | Precision / Instruction | Measured Rate | Unit | Relative Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **FP32 ALU** | Standard IEEE FP32 (`FADD`/`FMUL`) | **2.441** | TFLOPS | 1.00x (Baseline) |
| **FP16 RPM** | Rapid Packed Math (`HFMA2`) | **5.393** | TFLOPS | **2.21x** |
| **INT32 ALU** | Integer Add/Bitwise (`IADD3`) | **2.438** | TOP/s | 1.00x |
| **SFU Sine** | Transcendental Approx (`sin.approx`) | **393.4** | GInst/s | **0.32x (3.10x slower)** |
| **SFU Cosine** | Transcendental Approx (`cos.approx`) | **387.8** | GInst/s | **0.32x (3.15x slower)** |
| **SFU Rsqrt** | Reciprocal Square Root (`rsqrt.approx`)| **390.1** | GInst/s | **0.32x (3.13x slower)** |

---

### Table 2.3: Atomic Contention & Reduction Paradigms
*Benchmark: `experiments/016_atomic_contention.py` (65,536 threads contending for 1 address)*

| Atomic Operation | SASS Instruction | Measured Time (ms) | Relative Slowdown | Architectural Path |
| :--- | :--- | :---: | :---: | :--- |
| **`atomicAdd`** | `RED.E.ADD.STRONG.GPU` | **0.636** | **1.00x (Fastest)** | In-situ L2 crossbar aggregation (fire-and-forget) |
| **`atomicExch`** | `ATOMG.E.EXCH` | **15.421** | **24.25x** | Round-trip register writeback across SM crossbar |
| **`atomicCAS`** | `ATOMG.E.CAS` | **19.914** | **31.31x** | Serialized compare-and-swap with value return |

---

## 3. CPU Microarchitecture & Cache/TLB Forensic Results (Zen 2 Renoir)

### Table 3.1: Data TLB (D-TLB) Hierarchy & Hardware Page Table Walks
*Benchmark: `cpu/benchmarks/exp_tlb_hierarchy.cpp` (4 KB Page Strided Pointer Chase)*

| Active Pages | Footprint | Cycles / Access | TLB / Memory Hierarchy Level |
| :---: | :---: | :---: | :--- |
| **8** | 32 KB | **2.78** | D-TLB1 Hit (64 entries) + L1D Hit |
| **16 – 64** | 64 – 256 KB | **8.88 – 8.89** | D-TLB1 Hit (64 entries) + L2 Cache Hit |
| **128** | 512 KB | **11.66** | D-TLB1 Miss $\to$ D-TLB2 Hit (2,048 entries) |
| **256 – 512** | 1 – 2 MB | **25.73 – 26.83** | D-TLB2 Hit + L3 Cache Hit |
| **1,024** | 4 MB | **85.26** | D-TLB2 Hit + L3 Cache Capacity Knee |
| **2,048 – 16,384** | 8 – 64 MB | **222.39 – 259.83** | **D-TLB2 Miss Cliff $\to$ 4-Level DRAM Page Table Walk** |

---

### Table 3.2: Instruction TLB (I-TLB) & Code Page Dispersion
*Benchmark: `cpu/benchmarks/exp_itlb_hierarchy.cpp` (Randomized Inter-Page Trampolines)*

| Code Pages | Total Footprint | Cycles / Branch | Translation / Cache Level |
| :---: | :---: | :---: | :--- |
| **8** | 32 KB | **1.99** | L1 I-TLB Hit (64 entries) + BTB / L1I Hit |
| **16 – 64** | 64 – 256 KB | **8.97 – 15.28** | L1 I-TLB Hit + L2 Cache Hit |
| **96 – 256** | 384 KB – 1 MB | **15.84 – 27.19** | L2 I-TLB Hit (512 entries) + L3 Hit |
| **384 – 512** | 1.5 – 2 MB | **58.36 – 65.98** | L2 I-TLB Capacity Knee |
| **1,536 – 2,048** | 6 – 8 MB | **274.70 – 289.62** | **I-TLB Miss $\to$ Compulsory DRAM Page Walk (+4.4x Stall)** |

---

### Table 3.3: Store Bandwidth: Standard Cached vs Non-Temporal Streaming
*Benchmark: `cpu/benchmarks/exp_store_bandwidth.cpp` (AVX2 256-bit Stores)*

| Store Mechanism | SASS / x86 Mnemonic | Measured Bandwidth | Speedup Factor | Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **Standard Cached Stores** | `_mm256_storeu_si256` | **9.073 GB/s** | 1.00x | Read-For-Ownership (RFO) bus penalty |
| **Non-Temporal Stores** | `_mm256_stream_si256` | **30.123 GB/s** | **3.32x** | Write-combining bypass direct to DRAM |

---

### Table 3.4: False Sharing & Cross-CCX Invalidation Storm
*Benchmark: `cpu/benchmarks/exp_false_sharing_ccx.cpp` (Renoir 2x CCX Topology)*

| Configuration | Cores | Topology | Cycles / Op | Relative Slowdown | Architectural Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Independent Lines** | 0 & 1 | Intra-CCX | **1.44** | 1.00x | Pure local L1D hits |
| **Independent Lines** | 0 & 4 | Inter-CCX | **0.74** | 0.52x | Parallel CCX execution |
| **False Sharing (RMW)** | 0 & 1 | Intra-CCX | **10.72** | **7.46x** | Intra-CCX L3 cache line bouncing |
| **False Sharing (RMW)** | 0 & 4 | Inter-CCX | **11.79** | **8.20x** | Cross-CCX Infinity Fabric line invalidation |
| **Atomic Lock Add** | 0 & 1 | Intra-CCX | **25.91** | **18.02x** | Shared 4MB L3 serialization |
| **Atomic Lock Add** | 0 & 4 | Inter-CCX | **38.71** | **26.93x** | **+12.80c Cross-CCX Fabric Coherency Probe** |

---

### Table 3.5: AVX2 FMA Pipeline Latency vs Dependency Depth
*Benchmark: `cpu/benchmarks/exp_avx2_fma_dependency.cpp` (Renoir Core 2)*

| Accumulators ($K$) | TSC Cycles / FMA | Core Cycles (@ 4.15 GHz) | Sustained GFLOPS | Execution State |
| :---: | :---: | :---: | :---: | :--- |
| **1** | **3.494** | **5.01** | 13.3 | Pure dependency latency limit (5-cycle pipeline) |
| **2** | **1.749** | **2.51** | 26.5 | 2 dependency chains in flight |
| **4** | **1.001** | **1.44** | 46.3 | 4 dependency chains in flight |
| **8** | **0.434** | **0.62** | 106.6 | Overcoming 5-cycle pipeline bubble |
| **12** | **0.288** | **0.41** | **160.9** | **Full 2-Pipe 256-bit FMA Saturation (Peak)** |

---

## 4. GPU Microarchitecture & Silicon Execution Forensic Results (TU117 Turing)

### Table 4.1: Register Spilling Hierarchy & Physical Clamping
*Benchmark: `research/experiments/exp_register_spill_hierarchy.py` (N = 65,536 threads)*

| Target Regs | PTXAS Allocated | Local Loads / Iter | Spill Memory Footprint | Runtime (ms) | Slowdown | Memory Level Hit |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **64** | 64 | 0 | 0.0 MB | **32.08** | **1.00x** | Zero spills (Pure Register File) |
| **32** | 32 | 16 | 2.27 MB | **135.52** | **4.22x** | L1 / Unified Texture Cache |
| **24** | **24 (Clamped)** | 48 | 6.81 MB | **1,250.80** | **38.99x** | **1 MB L2 Overflow $\to$ GDDR6 DRAM** |

---

### Table 4.2: Warp Divergence Branch Serialization Scaling
*Benchmark: `research/experiments/exp_warp_divergence_depth.py`*

| Active Paths | Threads / Path | Execution Time (ms) | Measured Slowdown | Mathematical Expectation | Barrier Overhead |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 32 | **3.258** | **1.00x** | 1.00x | None (100% SIMT Efficiency) |
| **2** | 16 | **13.940** | **4.28x** | 2.00x | Branch reconvergence stack setup |
| **4** | 8 | **21.576** | **6.62x** | 4.00x | Multi-target branch splitting |
| **8** | 4 | **43.129** | **13.24x** | 8.00x | Micro-warp serialization |
| **16** | 2 | **86.249** | **26.47x** | 16.00x | Severe SIMT execution serialization |
| **32** | 1 | **172.523** | **52.95x** | 32.00x | **+65% Hardware `BRX`/`SYNC` Barrier Tax** |

---

### Table 4.3: Warp Shuffle (`__shfl_sync`) Intra-Warp Latency vs Stride
*Benchmark: `research/experiments/exp_warp_shuffle_latency.py` (%clock64 inside SM)*

| Shuffle Mode | Delta ($\Delta$) | Cycles / Instruction | Distance Invariance | Comparison to Shared Memory |
| :--- | :---: | :---: | :---: | :--- |
| **`XOR (Butterfly)`** | 1, 2, 4, 8, 16 | **26.81 – 26.83** | **100% Distance-Invariant** | 11.2x slower than SMEM ping-pong (2.4c) |
| **`UP`** | 1, 2, 4, 8, 16 | **26.81 – 26.83** | **100% Distance-Invariant** | Direct crossbar register routing |
| **`DOWN`** | 1, 2, 4, 8, 16 | **26.82 – 26.83** | **100% Distance-Invariant** | Zero shared memory allocation tax |

---

### Table 4.4: Instruction Cache (L1I) Footprint & Loop Unroll Cliff
*Benchmark: `research/experiments/exp_icache_cliff.py` (Non-foldable Rotating XOR Loop)*

| Unroll Length | SASS Footprint | Cycles / Instruction | Slowdown | Cache Hierarchy State |
| :---: | :---: | :---: | :---: | :--- |
| **128** | 2.0 KB | **1.367** | 1.00x | L1I Hit |
| **512** | 8.0 KB | **1.311** | 0.96x | L1I Hit |
| **1,024** | 16.0 KB | **1.299** | 0.95x | L1I Hit |
| **2,048** | 32.0 KB | **1.332** | 0.97x | L1I Hit |
| **4,096** | 64.0 KB | **1.308** | 0.96x | **L1I Capacity Limit (64 KB)** |
| **8,192** | 128.0 KB | **2.333** | **1.71x** | **L1I Miss Thrashing $\to$ L2 Fetch Stall (+71%)** |

---

### Table 4.5: FP16 Half2 SIMD Packing vs Pure FP32 Peak Compute
*Benchmark: `research/experiments/exp_fp16_vector_throughput.py` (896 CUDA Cores, 28,672 Threads)*

| Precision | SASS Instruction | Measured Time (ms) | Peak Sustained Compute | Throughput Ratio |
| :--- | :--- | :---: | :---: | :---: |
| **FP32 FMA** | `FFMA` | **87.820** | **3.134 TFLOPS** | 1.00x (Baseline) |
| **FP16x2 FMA** | `HFMA2` | **87.136** | **6.318 TFLOPS** | **2.02x (Rapid Packed Math)** |

---

## 5. Interconnect, Memory Architecture & WDDM Runtime Forensic Results

### Table 5.1: WDDM Pushbuffer Flush Granularity & Auto-Submit Spike Profiling
*Benchmark: `research/experiments/exp_wddm_pushbuffer_flush.py` (N = 4,096 asynchronous launches)*

| Metric | Measured Value | Architectural Inference |
| :--- | :---: | :--- |
| **Min Launch Latency** | **4.600 µs** | Pure user-mode pushbuffer append |
| **Median Launch Latency** | **6.700 µs** | Normal user-mode pushbuffer append |
| **95th Percentile Latency** | **30.925 µs** | WDDM escape / command packet flush |
| **Peak Launch Spike** | **131.500 µs** | Ring-0 transition to `dxgkrnl.sys` / `D3DKMTSubmitCommand` |
| **Detected Flush Events** | **506 spikes** | Autonomous driver submission |
| **Pushbuffer Flush Interval** | **Median = 7, Mean = 8.1 launches** | Driver enforces 7–8 launch pushbuffer batches |

---

### Table 5.2: Zero-Copy Host Memory (`cudaHostAllocMapped`) over PCIe 3.0 x8
*Benchmark: `research/experiments/exp_zerocopy_pcie.py` (Direct Vectorized Memory Access)*

| Transfer Size | Direct Read Time | Direct Read BW | Direct Write Time | Direct Write BW | Bus Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 MB** | 0.182 ms | 5.75 GB/s | 0.169 ms | 6.21 GB/s | 78.8% |
| **4 MB** | 0.638 ms | 6.58 GB/s | 0.646 ms | 6.49 GB/s | 82.4% |
| **16 MB** | 2.521 ms | 6.66 GB/s | 2.552 ms | 6.58 GB/s | 83.5% |
| **64 MB** | 10.055 ms | **6.67 GB/s** | 10.187 ms | **6.59 GB/s** | **83.6% of PCIe 3.0 x8 Peak** |

---

### Table 5.3: PCIe Full-Duplex Bidirectional Saturation
*Benchmark: `research/experiments/exp_pcie_duplex_saturation.py` (Concurrent H2D + D2H)*

| Buffer Size | Simplex H2D (GB/s) | Simplex D2H (GB/s) | Duplex Aggregate (GB/s) | Duplex Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **4 MB** | 6.61 | 6.56 | **11.20** | 85.0% |
| **16 MB** | 6.66 | 6.57 | **11.28** | 85.3% |
| **64 MB** | 6.67 | 6.58 | **11.29** | 85.3% |
| **128 MB** | **6.67** | **6.59** | **11.31** | **85.3% of Dual-Simplex Sum** |

---

### Table 5.4: Multi-Stream Hardware Queue Concurrency Limits
*Benchmark: `research/experiments/exp_multistream_concurrency.py` (TU117 14 SMs, WDDM 2.7)*

| Streams ($K$) | Measured Wall Time (ms) | Expected If Serial (ms) | Measured Concurrency Factor | Scheduler State |
| :---: | :---: | :---: | :---: | :--- |
| **1** | 2.220 | 2.220 | 1.00x | 1 SM active |
| **2** | 2.211 | 4.439 | 2.01x | 2 SMs concurrent |
| **4** | 2.232 | 8.878 | 3.98x | 4 SMs concurrent |
| **7** | 2.241 | 15.537 | 6.93x | 7 SMs concurrent |
| **14** | 4.439 | 31.074 | 7.00x | 2 waves of 7 SMs |
| **16** | 4.468 | 35.514 | 7.95x | 2 waves of 8 SMs |
| **24** | 6.606 | 53.270 | 8.06x | 3 waves of 8 SMs |
| **32** | 8.847 | 71.027 | **8.03x (Ceiling)** | **Hardware GMU Queue Ceiling = 8 Grids** |

---

## 6. Monolithic APU SoC Fabric Contention (Vega 7 iGPU vs Zen 2 CPU)

### Table 6.1: On-Die Fabric Interference Telemetry
*Benchmark: `research/experiments/exp_soc_fabric_contention.py` (Renoir Monolithic SoC)*

| Subsystem Metric | Baseline (Idle iGPU) | Under Vega 7 Fabric Stress | Degradation / Penalty |
| :--- | :---: | :---: | :---: |
| **CPU Read Bandwidth (AVX2)** | **19.45 GB/s** | **4.76 GB/s** | **-75.5%** |
| **CPU NT Write Bandwidth** | **32.07 GB/s** | **9.82 GB/s** | **-69.4%** |
| **CPU DRAM Pointer-Chase Latency** | **298.49 cycles** | **591.73 cycles** | **+98.2% (1.98x doubling)** |

---

## 7. Local AI & Quantized LLM Inference Forensic Results

### Table 7.1: Prefill vs Decode Arithmetic Intensity & Roofline Transition
*Benchmark: `research/experiments/exp_prefill_decode_roofline.py` (GEMV $\to$ GEMM Sweep)*

| Batch Size ($B$) | Arithmetic Intensity | Execution Time | Measured Throughput | Limiting Factor |
| :---: | :---: | :---: | :---: | :--- |
| **1 (Pure Decode)** | 0.50 FLOP/B | 0.113 ms | **18.6 GFLOP/s** | **Strictly Memory Bandwidth Bound** |
| **4** | 1.98 FLOP/B | 0.110 ms | 75.9 GFLOP/s | Memory Bandwidth Bound |
| **16** | 7.76 FLOP/B | 0.228 ms | 147.4 GFLOP/s | Transition Region |
| **48** | 21.94 FLOP/B | 0.641 ms | **156.9 GFLOP/s** | **Roofline Knee (GEMV $\to$ GEMM Transition)** |
| **64** | 28.44 FLOP/B | 0.852 ms | 157.4 GFLOP/s | Compute-Bound Plateau |
| **256 (Heavy Prefill)** | 85.33 FLOP/B | 3.398 ms | 158.3 GFLOP/s | Fully Compute-Bound Plateau |

---

### Table 7.2: Quantized INT4 Nibble Unpacking vs Raw FP32 GEMV
*Benchmark: `research/experiments/exp_quant_dequant_overhead.py` (8M Parameter GEMV)*

| Precision | DRAM Footprint | Execution Time | Measured Memory BW | Sustained Compute | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 GEMV** | 67.1 MB | 0.370 ms | **181.24 GB/s** | 45.3 GFLOPS | 1.00x |
| **Q4_0 GEMV** | 41.9 MB | 0.231 ms | **181.64 GB/s** | **72.7 GFLOPS** | **1.61x (+60.4%)** |

*Discovery*: Bitwise unpacking in software reaches 100% GDDR6 bus saturation (181.6 GB/s), hiding dequantization latency behind memory reads.

---

### Table 7.3: PagedAttention KV-Cache Strided Memory Traversal Overhead
*Benchmark: `research/experiments/exp_paged_attention_kv.py` (16 Tokens/Block = 32 KB/Block)*

| Sequence Length | Cache Footprint | Contiguous Latency | Contiguous BW | Paged Latency | Paged BW | Indirection Tax |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | 1.05 MB | 12.38 µs | 84.70 GB/s | 13.40 µs | 78.24 GB/s | **1.08x** |
| **1,024 tokens** | 2.10 MB | 14.91 µs | 140.66 GB/s | 17.33 µs | 121.04 GB/s | **1.16x** |
| **2,048 tokens** | 4.19 MB | 26.15 µs | 160.37 GB/s | 26.90 µs | 155.95 GB/s | **1.03x** |
| **4,096 tokens** | 8.39 MB | 47.83 µs | 175.39 GB/s | 50.14 µs | 167.32 GB/s | **1.05x** |
| **8,192 tokens** | 16.78 MB | 92.41 µs | 181.56 GB/s | 96.38 µs | 174.08 GB/s | **1.04x (+4.3%)** |

---

### Table 7.4: FlashAttention Online Softmax Rescaling & SFU Exp Overhead
*Benchmark: `research/experiments/exp_flash_attention_softmax.py` (Turing SM 7.5, Warp Size = 32)*

| Sequence Length | 32-Token Tiles | FlashAttention Cycles | Cycles / Tile | Cycles / Token | Baseline FMA Cycles | SFU / Reduction Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128 tokens** | 4 | 945.4 | 236.4 | 7.39 | 407.3 | **2.32x** |
| **512 tokens** | 16 | 3,645.2 | 227.8 | 7.12 | 1,208.2 | **3.02x** |
| **1,024 tokens** | 32 | 7,245.2 | 226.4 | 7.08 | 2,276.2 | **3.18x** |
| **2,048 tokens** | 64 | 14,445.2 | 225.7 | 7.05 | 4,412.2 | **3.27x** |
| **4,096 tokens** | 128 | 28,845.2 | **225.4** | **7.04** | 8,684.2 | **3.32x** |

*Discovery*: FlashAttention's online exponential rescaling and warp butterfly reduction impose an exact **7.04 clock cycles/token** compute tax (3.32x over FMA), but completely eliminate the GDDR6 DRAM bottleneck by keeping intermediate logits resident in registers.

---

### Table 7.5: Rotary Position Embedding (RoPE) In-Register Rotation vs Global Memory
*Benchmark: `research/experiments/exp_rope_microbenchmark.py` (32 Attention Heads, Head Dim = 64)*

| Sequence Length | Head Dimension | Unfused Memory RoPE | Unfused Memory BW | Fused Register RoPE | Fused Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | 64 | 78.7 µs | 159.9 GB/s | **54.2 µs** | **1.45x** |
| **1,024 tokens** | 64 | 154.4 µs | 163.0 GB/s | **105.5 µs** | **1.46x** |
| **2,048 tokens** | 64 | 303.7 µs | 165.7 GB/s | **208.6 µs** | **1.46x** |
| **4,096 tokens** | 64 | 601.7 µs | 167.3 GB/s | **406.6 µs** | **1.48x** |
| **8,192 tokens** | 64 | 1,201.4 µs | 167.6 GB/s | **818.3 µs** | **1.47x (+47%)** |

*Discovery*: Fusing the 2D Givens rotation in registers directly after the projection GEMV bypasses global memory roundtrips, accelerating token embedding rotation by **1.47x** across long contexts.

---

### Table 7.6: Split-K Batched GEMV for Small Batch LLM Decoding on 14 SMs
*Benchmark: `research/experiments/exp_splitk_gemv.py` (Weight Matrix: 4096 x 4096 FP32, 64 MB)*

| Split-K Factor ($S$) | Grid Dimensions ($M \times S$) | Execution Time (ms) | Effective Throughput | Sustained GFLOPS | Analysis |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$S = 1$ (Standard)** | $4096 \times 1$ | 0.377 ms | 177.80 GB/s | 88.9 GFLOPS | 92.6% GDDR6 bus saturation |
| **$S = 2$ (Optimal)** | $4096 \times 2$ | **0.374 ms** | **179.26 GB/s** | **89.6 GFLOPS** | **Peak Saturation (93.4% of GDDR6)** |
| **$S = 4$** | $4096 \times 4$ | 0.376 ms | 178.57 GB/s | 89.3 GFLOPS | Balanced multi-SM occupancy |
| **$S = 8$** | $4096 \times 8$ | 0.381 ms | 176.18 GB/s | 88.1 GFLOPS | Atomic reduction tax begins |
| **$S = 16$** | $4096 \times 16$ | 0.427 ms | 157.02 GB/s | 78.5 GFLOPS | **-12.4% Atomic Serialization Penalty** |

---

## 8. Deep Microarchitectural Discovery Telemetry (Wave 2)

### Table 8.1: AMD Zen 2 Reorder Buffer (ROB) Capacity & Out-of-Order Window
*Benchmark: `cpu/benchmarks/exp_rob_capacity.cpp` (Independent ALU Additions behind Unresolved DRAM Miss)*

| Independent Instructions ($N$) | Total Iteration Cycles | Latency per Iteration | Incremental Cost | Pipeline State |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 8,368,530 | 278.95 cycles | Baseline | Pure DRAM pointer chase latency |
| **32** | 7,511,203 | 250.37 cycles | 0.00c / inst | Absorbed in ROB shadow |
| **64** | 8,261,955 | 275.40 cycles | 0.00c / inst | Absorbed in ROB shadow |
| **128** | 9,507,824 | 316.93 cycles | 0.00c / inst | Absorbed in ROB shadow |
| **192** | 9,887,724 | 329.59 cycles | 0.00c / inst | Absorbed in ROB shadow |
| **224 (Nominal ROB)** | 9,030,223 | **301.01 cycles** | **0.00c / inst** | **ROB Limit Fully Populated (224 Entries)** |
| **256** | 9,229,627 | 307.65 cycles | 0.02c / inst | Instruction buffer pressure |
| **352** | 10,448,352 | 348.28 cycles | 0.40c / inst | Dispatch stall onset |
| **448** | 16,253,630 | **541.79 cycles** | **0.54c / inst** | **Front-End Dispatch Stall Cliff (+79.9%)** |

---

### Table 8.2: AMD Zen 2 Store-to-Load Forwarding (STLF) & Aliasing Penalties
*Benchmark: `cpu/benchmarks/exp_stlf_penalty.cpp` (N = 10,000,000 dependent store/load pairs)*

| Operation Case | Dependency Pattern | Measured Latency | Forwarding State | Stall Penalty |
| :--- | :--- | :---: | :--- | :---: |
| **Independent** | Separate Cache Line | **0.785 cycles** | Dual-Issue L1 Pipeline | Baseline |
| **Exact Match STLF** | 8-byte Store $\to$ 8-byte Load | **0.722 cycles** | Clean Store Buffer Forwarding | 0.00c |
| **Sub-Register STLF** | 8-byte Store $\to$ 4-byte Low Load | **0.723 cycles** | Clean Sub-Register Forwarding | **0.00c (Zero Penalty)** |
| **Partial Overlap Failure** | 8-byte Store [0..7] $\to$ 8-byte Load [1..8] | **10.577 cycles** | **STLF Failure $\to$ Pipeline Replay** | **+9.85 cycles (14.6x stall)** |
| **4K Aliasing Collision** | Store [A] $\to$ Load [A + 4096] | **1.085 cycles** | Disambiguator False Match | +0.30 cycles |

---

### Table 8.3: AMD Zen 2 Radix Division Latency & AVX2 Vector Division
*Benchmark: `cpu/benchmarks/exp_division_latency.cpp` (N = 10,000,000 iterations)*

| Operation | Bitwidth / Type | Dependent Latency | Reciprocal Throughput | Pipelining State |
| :--- | :--- | :---: | :---: | :--- |
| **`idiv` (Small Dividend)** | 32-bit Integer | **10.09 cycles** | 15.90 cycles/div | Completely Non-Pipelined Radix-4 |
| **`idiv` (Full Dividend)** | 32-bit Integer | **9.90 cycles** | 15.90 cycles/div | Fixed-Cycle Hardware Divider |
| **`idiv` (Full 64-bit)** | 64-bit Integer | **9.86 cycles** | 15.90 cycles/div | Unified 64-bit Divider Datapath |
| **`_mm256_div_ps`** | 256-bit Vector FP32 | **6.96 cycles** | **2.46 cycles/vector** | **Partially Pipelined (2.46c Throughput)** |

---

### Table 8.4: GPU L2 Cache Slice Hashing & Physical Memory Hierarchy
*Benchmark: `research/experiments/exp_l2_hash_stride.py` (Serialized Pointer Chase with `ld.global.ca` vs `ld.global.cg`)*

| Working Set Footprint | Physical Subsystem | `ld.global.ca` (L1+L2) | `ld.global.cg` (L2 Only) | Architectural State |
| :---: | :---: | :---: | :---: | :--- |
| **4 KB** | L1 Data Cache | **33.83 cycles** | 199.29 cycles | Physical L1 Hit |
| **8 KB** | L1 Data Cache | **34.48 cycles** | 198.60 cycles | Physical L1 Hit |
| **16 KB** | L1 Data Cache | **36.52 cycles** | 198.98 cycles | Physical L1 Hit |
| **32 KB** | L1 Data Cache | **49.03 cycles** | 199.03 cycles | L1 Associativity Boundary |
| **64 KB** | L2 Unified Cache | **107.83 cycles** | **199.11 cycles** | True Physical L2 Latency |
| **256 KB** | L2 Unified Cache | **187.71 cycles** | **198.94 cycles** | True Physical L2 Latency |
| **1,024 KB (1 MB)** | L2 Unified Cache | **201.87 cycles** | **202.61 cycles** | **Physical L2 Capacity Limit** |
| **2,048 KB (2 MB)** | GDDR6 VRAM | **320.92 cycles** | **342.90 cycles** | **L2 Overflow $\to$ DRAM Fetch** |
| **8,192 KB (8 MB)** | GDDR6 VRAM | **388.78 cycles** | **395.18 cycles** | **Full GDDR6 DRAM Latency** |

*Discovery*: Stride testing across 64 nodes from 128 B to 512 KB revealed invariant **199.8 – 201.2 cycles** with zero set aliasing, proving Turing's polynomial XOR hashing uniformly distributes power-of-2 addresses across L2 cache slices.

---

### Table 8.5: GPU 2D Spatial Locality & Memory Coalescing Cliff
*Benchmark: `research/experiments/exp_strided_vs_tiled_locality.py` (16M 32-bit floats = 64 MB)*

| Access Stride | Execution Time (ms) | Effective Throughput (GB/s) | Degradation Ratio | Transaction Efficiency |
| :---: | :---: | :---: | :---: | :--- |
| **Stride 1 (Coalesced)** | **0.858 ms** | **145.76 GB/s** | 1.00x (Baseline) | 100% bus utilization |
| **Stride 2** | 1.132 ms | 110.47 GB/s | 1.32x | 75% bus utilization |
| **Stride 4** | 2.120 ms | 58.96 GB/s | 2.47x | 40% bus utilization |
| **Stride 8** | 3.852 ms | 32.45 GB/s | 4.49x | 22% bus utilization |
| **Stride 16** | 7.262 ms | 17.21 GB/s | 8.47x | 12% bus utilization |
| **Stride 32 (Worst)** | **7.172 ms** | **17.43 GB/s** | **8.36x Collapse** | **12.5% Bus Efficiency (32 lines/warp)** |
| **Stride 256** | 15.009 ms | 8.33 GB/s | 17.50x Collapse | DRAM page thrashing |
| **2D Tiled Transpose (Smem)** | **0.946 ms** | **132.19 GB/s** | **7.58x Speedup** | **Shared Memory Recovers Coalescing** |

---

### Table 8.6: GPU Special Function Unit (SFU) vs FP32 Issue Port Contention
*Benchmark: `research/experiments/exp_sfu_contention.py` (TU117 Turing SM 7.5)*

| Kernel Type | Primary SASS Op | Total Cycles (1000 iters) | Cycles / Warp-Instruction | Issue Rate (warp-inst/cyc) |
| :--- | :--- | :---: | :---: | :---: |
| **Pure FP32 FMA** | `FFMA` | 526,439 | **0.51 cycles** | **1.95 inst/cycle** |
| **Pure SFU RSQRT** | `MUFU.RSQ` | 4,876,209 | **4.76 cycles** | **0.21 inst/cycle** |
| **Pure SFU EXP2** | `MUFU.EX2` | 1,949,213 | **1.90 cycles** | **0.53 inst/cycle** |
| **Interleaved (1:1)** | `FFMA` + `MUFU.RSQ` | 2,448,210 | **2.39 cycles** | **0.42 inst/cycle** |

---

### Table 8.7: Host Pinned Memory Registration (`cuMemHostRegister`) vs Allocation (`cuMemHostAlloc`)
*Benchmark: `research/experiments/exp_host_pinned_registration.py` (Windows 11 WDDM 2.7, PCIe 3.0 x8)*

| Buffer Size | `cuMemHostAlloc` (ms) | `cuMemHostRegister` (ms) | Speedup Ratio | Pinned H2D BW | Registered H2D BW |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 MB** | 0.657 ms | 0.286 ms | **2.30x** | 5.89 GB/s | 5.89 GB/s |
| **4 MB** | 0.927 ms | 2.864 ms | 0.32x | 6.50 GB/s | 6.50 GB/s |
| **16 MB** | 3.973 ms | 0.648 ms | **6.13x** | 6.17 GB/s | 6.52 GB/s |
| **64 MB** | 13.786 ms | 0.662 ms | **20.81x** | 6.55 GB/s | 6.23 GB/s |
| **128 MB** | **26.624 ms** | **0.910 ms** | **29.25x Faster** | **6.62 GB/s** | **6.43 GB/s** |

*Discovery*: Dynamic registration of existing committed virtual memory (`cuMemHostRegister`) completes in **0.91 ms for 128 MB** (compared to 26.62 ms for fresh allocation), achieving identical **6.43–6.62 GB/s PCIe 3.0 x8 transfer bandwidth** without the 27ms OS allocation penalty.

---

### Table 8.8: AMD Zen 2 Hardware Data Prefetcher Dissection & Stride Tracking
*Benchmark: `cpu/benchmarks/exp_prefetcher_dissection.cpp` (32 MB DRAM Buffer, serialized `_mm_mfence`)*

| Access Pattern | Stride Offset | Latency / Access | Baseline DRAM Latency | Prefetcher Speedup | Architectural State |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Pointer Chase** | Non-Linear | **241 cycles** | 241 cycles | 1.00x | Pure Uncached DRAM Access |
| **Sequential Next-Line Forward** | +64 B | **9 cycles** | 241 cycles | **26.78x Speedup** | L1 Stream / Next-Line Prefetcher |
| **Sequential Next-Line Backward** | -64 B | **9 cycles** | 241 cycles | **26.78x Speedup** | L1 Bi-Directional Prefetcher |
| **Forward Stride 2** | +128 B | **13 cycles** | 241 cycles | 18.54x Speedup | L1/L2 Stride Prefetcher |
| **Backward Stride 2** | -128 B | **20 cycles** | 241 cycles | 12.05x Speedup | L1/L2 Backward Stride Tracker |
| **Forward Stride 4** | +256 B | **10 cycles** | 241 cycles | 24.10x Speedup | Multi-Cache-Line Prefetcher |
| **Backward Stride 4** | -256 B | **13 cycles** | 241 cycles | 18.54x Speedup | Multi-Cache-Line Backward |
| **Forward Stride 8** | +512 B | **16 cycles** | 241 cycles | 15.06x Speedup | L2 Stream Prefetcher |
| **Forward Stride 16** | +1024 B | **14 cycles** | 241 cycles | 17.21x Speedup | Extended Pattern Prefetcher |
| **Forward Stride 32** | +2048 B | **14 cycles** | 241 cycles | **17.21x Speedup** | Maximum Stride Limit |
| **Page-Level Stride** | +4096 B | **19 cycles** | 241 cycles | 12.68x Speedup | Page Boundary Crossing Penalty |

---

### Table 8.9: AMD Zen 2 Bit Manipulation & Microcoded PDEP/PEXT Traps
*Benchmark: `cpu/benchmarks/exp_bmi_pdep_pext.cpp` (N = 10,000,000 dependent operations)*

| Instruction | Architecture / ISA | Dependent Latency | Implementation Mechanism | Performance Ratio vs TZCNT |
| :--- | :--- | :---: | :--- | :---: |
| **`POPCNT`** | ABM / SSE4.2 | **1.41 cycles** | Hardware Population Count ALU | 0.66x (Fastest) |
| **`TZCNT`** | BMI1 | **2.12 cycles** | Hardware Trailing Zero Count ALU | 1.00x (Baseline) |
| **`BSF` (Legacy)** | x86-64 Base | **3.04 cycles** | Legacy Bit Scan Forward Logic | 1.43x |
| **`PEXT` (Bit Extract)** | BMI2 | **93.88 cycles** | **Microcode Sequencer Trap** | **44.19x Microcode Stall** |
| **`PDEP` (Bit Deposit)** | BMI2 | **95.21 cycles** | **Microcode Sequencer Trap** | **44.81x Microcode Stall** |

*Discovery*: While Intel implementations feature a 1-cycle hardware ALU for `PDEP`/`PEXT`, AMD Zen 2 microcodes both operations into ~94-95 cycle multi-uop execution sequences, inflicting a severe 45x penalty that makes software bit-packing quantizers catastrophically slow on Renoir.

---

### Table 8.10: NVIDIA Turing Uniform Datapath & Uniform Register Allocation
*Benchmark: `research/experiments/exp_uniform_datapath.py` (SM 7.5 TU117, PTX/SASS Disassembly)*

| Loop Implementation | Target Registers | Clock Cycles / Iteration | SASS Instruction Signature | Hardware Pipeline |
| :--- | :---: | :---: | :--- | :--- |
| **Vector Loop** | `R0`..`R6` (GPR) | **6.25 cycles** | `IMAD.MOV.U32`, `IADD3`, `ISETP.GE` | Standard Vector Datapath (32 threads/warp) |
| **Uniform Loop** | `UR4` (URF) + `R0` | **6.25 cycles** | `ULDC UR4, c[0x0][0x168]`, `ISETP.GE` | **Uniform Datapath + Uniform Register File** |

*Discovery*: SASS inspection confirms TU117 compiler automatically emits `ULDC UR4` (Uniform Constant Load) and allocates scalar loop trip counts to the Uniform Register File (URF), freeing vector GPR register bandwidth for compute operands.

---

### Table 8.11: Dynamic Shared Memory Carveout vs L1 Data Cache Hit Cliff
*Benchmark: `research/experiments/exp_l1_smem_carveout.py` (48 KB Working Set, 96 KB Total Unified SRAM)*

| Shared Memory Reservation | Carveout % | Effective L1D Capacity | Measured Latency | Cache Residency Status |
| :---: | :---: | :---: | :---: | :--- |
| **0 KB SMEM** | 0% | 64 KB L1D | **42.85 cycles** | **L1 Data Cache Hit (48 KB < 64 KB)** |
| **32 KB SMEM** | 33% | 64 KB L1D | **42.85 cycles** | **L1 Data Cache Hit (48 KB < 64 KB)** |
| **64 KB SMEM** | 66% | 32 KB L1D | **167.86 cycles** | **L1 Cache Miss $\to$ Spill to L2 (3.92x Latency Jump)** |
| **96 KB SMEM** | 100% | 16 KB L1D | **168.03 cycles** | **L1 Cache Miss $\to$ Spill to L2 (3.92x Latency Jump)** |

*Discovery*: Turing SM 7.5 dynamically reconfigures the 96 KB SRAM between L1D and SMEM. A 48 KB working set hits cleanly at 42.85 cycles when L1D is configured for 64 KB. Reserving $\ge 64\text{ KB}$ for shared memory shrinks L1D to 32 KB, triggering an instantaneous 3.92x latency explosion (167.86c) as accesses spill to the L2 cache crossbar.

---

### Table 8.12: PCIe 3.0 x8 Small-Packet Latency Floor & Framing Knee
*Benchmark: `research/experiments/exp_pcie_small_packet_latency.py` (Windows 11 WDDM 2.7, Host $\leftrightarrow$ Device)*

| Payload Size | One-Way Latency | Round-Trip Latency | Effective Bandwidth | Physical Bus State |
| :---: | :---: | :---: | :---: | :--- |
| **4 Bytes** | **12.62 µs** | **25.24 µs** | 0.32 MB/s | **Framing Overhead & Root Complex TLP Floor** |
| **16 Bytes** | 12.60 µs | 25.19 µs | 1.27 MB/s | Framing Overhead Floor |
| **64 Bytes** | 12.42 µs | 24.85 µs | 5.15 MB/s | Framing Overhead Floor |
| **256 Bytes** | 12.57 µs | 25.15 µs | 20.36 MB/s | Framing Overhead Floor |
| **1 KB** | 12.56 µs | 25.13 µs | 81.50 MB/s | Framing Overhead Floor |
| **4 KB** | 13.16 µs | 26.31 µs | 311.32 MB/s | Slight Duration Increase (+1.18 µs) |
| **16 KB** | **14.94 µs** | **29.89 µs** | **1.10 GB/s** | **Bandwidth Knee Initiation** |
| **64 KB** | **24.62 µs** | **49.24 µs** | **2.66 GB/s** | **Linear Transmission Transition** |
| **256 KB** | 49.83 µs | 99.67 µs | 5.26 GB/s | Approaching Wire Saturation |
| **1 MB** | **169.11 µs** | **338.21 µs** | **6.20 GB/s** | **Full Wire Saturation (95.4% of Simplex Peak)** |

*Discovery*: PCIe 3.0 x8 exhibits an immutable ~25 µs round-trip latency floor for all packet transfers $\le 1\text{ KB}$. The transition from latency-dominated framing to wire bandwidth transmission initiates between 16 KB and 64 KB.

---

### Table 8.13: Quantized INT8 / DP4A Vector Dot Product Peak TOPS on Non-Tensor Silicon
*Benchmark: `research/experiments/exp_idp4a_vector_tops.py` (TU117 Turing SM 7.5, 896 CUDA Cores, 28,672 Threads)*

| Precision / ISA Engine | Primary SASS Instruction | Measured Throughput | Speedup vs FP32 | Speedup vs FP16 | Silicon Execution Unit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Single-Precision (FP32)** | `FFMA` | 3.134 TFLOPS | 1.00x (Baseline) | 0.50x | Standard 32-bit CUDA ALU |
| **Rapid Packed Math (FP16x2)** | `HFMA2` | 6.318 TFLOPS | 2.02x | 1.00x | Dual-Issue 16-bit Vector ALU |
| **Quantized INT8 (`dp4a`)** | `IDP.4A.S8.S8` | **10.269 TOPS** | **3.28x** | **1.63x** | **Dedicated 4-element 8-bit Dot-Product ALU** |

*Discovery*: Even without Tensor Cores, TU117 contains native 4-way 8-bit integer dot product units (`IDP.4A.S8.S8`) capable of sustaining **10.269 TOPS**—outperforming FP16 Rapid Packed Math by **1.63x** and FP32 by **3.28x**, making INT8 the definitive inference acceleration format for consumer Turing silicon.

---

### Table 8.14: Speculative Decoding Parallel Multi-Token Verification Kernel
*Benchmark: `research/experiments/exp_speculative_verification.py` (Vocabulary $V = 32,000$, SM 7.5 TU117)*

| Draft Speculation Length ($K$) | Target Logits Size | Verification Latency | Draft Token Throughput | Latency Scaling Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **$K = 1$ Draft Token** | 125 KB | **12.38 µs** | 80,779 tokens/s | Initial Kernel Launch Overhead |
| **$K = 2$ Draft Tokens** | 250 KB | **8.46 µs** | 236,447 tokens/s | Flat Execution Plateau |
| **$K = 4$ Draft Tokens** | 500 KB | **8.60 µs** | 465,030 tokens/s | Flat Execution Plateau |
| **$K = 8$ Draft Tokens** | 1,000 KB | **7.89 µs** | 1,014,528 tokens/s | Flat Execution Plateau |
| **$K = 16$ Draft Tokens** | 2,000 KB | **8.56 µs** | **1,868,670 tokens/s** | **Flat Latency: 1.87 Million tokens/sec** |

*Discovery*: Parallel multi-warp verification of candidate draft sequences against the full vocabulary logits ($V=32,000$) exhibits an entirely flat latency profile (**7.89 to 8.60 µs**) across draft lengths $K \in [2, 16]$. A single parallel GPU kernel verifies up to 16 tokens in 8.56 µs, eliminating token-by-token verification latency in speculative decoding frameworks.

