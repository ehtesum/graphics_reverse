# Paper Epistemological Corrections & De-Biasing

This document tracks every sentence in the technical whitepaper requiring qualification, strengthening, or revision to prevent overclaiming.

### Claim `CPU-L1-001` (CPU)
* **Original Statement**: "L1D cache access latency is 1.00 ns (4.00 clock cycles at 4.0 GHz)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly measured via serialized pointer chasing.

### Claim `CPU-L2-001` (CPU)
* **Original Statement**: "L2 cache access latency is 3.10 to 4.30 ns (12.4 to 17.2 clock cycles)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Supported by step-function transition at 32 KB.

### Claim `CPU-L3-001` (CPU)
* **Original Statement**: "L3 cache access latency is 9.00 to 19.60 ns (36.0 to 78.4 clock cycles)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Reflects Renoir 4 MB victim cache architecture.

### Claim `CPU-DRAM-001` (CPU)
* **Original Statement**: "DRAM reaches 127.0 to 148.0 ns, representing a 127x to 148x latency penalty relative to L1D."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Note that standard 4 KB page walks contribute ~20 ns to the tail.

### Claim `CPU-BRANCH-001` (CPU)
* **Original Statement**: "The true branch resteer penalty across Zen 2's 14-stage integer pipeline is 9.82 cycles."
* **Status**: Category B (STRONGLY SUPPORTED INFERENCE)
* **Audit Verdict**: Retain. Qualify that array-based benchmarks include 0.51 cycles of memory replay.

### Claim `CPU-FMA-001` (CPU)
* **Original Statement**: "Dependent 256-bit FMA chains execute with a latency of 3.56 cycles."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly measured on physical core.

### Claim `CPU-AVX2-001` (CPU)
* **Original Statement**: "8 cores / 16 threads achieve 598.68 GFLOPS (a 4.59x scaling efficiency relative to 1T)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Attribute scaling shortfall to 45W PPT budget rather than software inefficiency.

### Claim `CPU-FABRIC-001` (CPU)
* **Original Statement**: "Intra-CCX latency is 25.8 ns; cross-CCX latency jumps to 63.4 ns (+37.6 ns / 2.45x penalty)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly reveals internal dual-CCX layout of Renoir APU.

### Claim `GPU-SMEM-001` (GPU)
* **Original Statement**: "Proves conclusively that TU117 SM 7.5 shared memory is partitioned into exactly 32 independent 4-byte wide physical banks."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Weakened qualification: 'Empirical coprime testing confirms that shared memory bank arbitration matches a 32-bank modulo model.'

### Claim `GPU-SHFL-001` (GPU)
* **Original Statement**: "Warp shuffle is 35.06x faster than global atomics and 41% faster than shared memory reduction."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly measured across identical reductions.

### Claim `GPU-FP16-001` (GPU)
* **Original Statement**: "FP16 packed math delivers 5.393 TFLOPS, achieving a 2.21x throughput multiplier over FP32."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Note that the >2.0x factor arises from dynamic thermal boost differences.

### Claim `GPU-SFU-001` (GPU)
* **Original Statement**: "Special Function Units sustain 387.8 to 393.4 GInst/s (3.12x slower than FP32 ALU)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly measures SFU pipeline throughput.

### Claim `GPU-ATOMIC-001` (GPU)
* **Original Statement**: "Turing offloads atomic reduction directly into L2 memory controllers via non-returning fire-and-forget operations."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Confirmed via SASS disassembly.

### Claim `GPU-PCIE-001` (GPU)
* **Original Statement**: "Pinned memory sustains 6.08 GB/s (saturating the x8 link)."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Qualified wording: 'Pinned host-to-device transfers sustain 6.08 GB/s, reflecting practical bandwidth utilization over the negotiated PCIe 3.0 x8 interface.'

### Claim `GPU-ASYNC-001` (GPU)
* **Original Statement**: "Overlapping H->D and D->H transfers across two streams achieves 98.9% efficiency."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Confirmed physical presence of two independent copy engines.

### Claim `WDDM-001` (WDDM)
* **Original Statement**: "WDDM user-mode driver command packet generation requires 9.23 µs, while true ring-0 OS transition and hardware completion requires 14.35 µs."
* **Status**: Category C (PARTIALLY SUPPORTED - OVERCLAIM REVISED)
* **Audit Verdict**: Corrected wording: 'The ~9 µs latency represents user-mode command pushbuffer generation within nvcuda.dll; synchronous wait across the OS kernel boundary costs 14.35 µs.'

### Claim `AI-VRAM-001` (AI)
* **Original Statement**: "Desktop Window Manager reserves ~500-650 MB of VRAM, leaving ~3,446 MB usable for inference."
* **Status**: Category A (DIRECTLY MEASURED)
* **Audit Verdict**: Retain. Directly measured via CUDA driver memory query.

