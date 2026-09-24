# CPU Track: AMD Ryzen 7 4800HS (Zen 2) Cache Hierarchy Findings

Empirical pointer-chasing microbenchmark measuring physical memory access latency across working-set sizes (4 KB to 64 MB) on the AMD Ryzen 7 4800HS.

## 1. Measured Cache Latency Profile

| Working Set Size | Cache Tier | Latency (ns) | Cycles (~4.0 GHz) | Inferred Hardware Status |
| :--- | :--- | :--- | :--- | :--- |
| **4 KB** | **L1 Data Cache** | **1.06 ns** | ~4.2 cycles | L1 Hit |
| **8 KB** | **L1 Data Cache** | **1.03 ns** | ~4.1 cycles | L1 Hit |
| **16 KB** | **L1 Data Cache** | **0.99 ns** | ~4.0 cycles | L1 Hit |
| **24 KB** | **L1 Data Cache** | **1.15 ns** | ~4.6 cycles | L1 Hit |
| **32 KB** | **L1 Data Cache Limit** | **1.78 ns** | ~7.1 cycles | L1 Capacity Threshold |
| **48 KB** | **L2 Unified Cache** | **3.27 ns** | ~13.1 cycles | L2 Hit |
| **64 KB** | **L2 Unified Cache** | **2.95 ns** | ~11.8 cycles | L2 Hit |
| **128 KB** | **L2 Unified Cache** | **3.87 ns** | ~15.5 cycles | L2 Hit |
| **256 KB** | **L2 Unified Cache** | **3.11 ns** | ~12.4 cycles | L2 Hit |
| **512 KB** | **L2 Unified Limit** | **4.31 ns** | ~17.2 cycles | L2 Capacity Threshold |
| **768 KB** | **L3 Shared Cache** | **8.99 ns** | ~36.0 cycles | L3 CCX Hit |
| **1 MB** | **L3 Shared Cache** | **11.94 ns** | ~47.7 cycles | L3 CCX Hit |
| **2 MB** | **L3 Shared Cache** | **19.66 ns** | ~78.6 cycles | L3 CCX Inter-slice |
| **4 MB** | **L3 Shared Cache** | **91.74 ns** | ~367 cycles | L3 Miss / CCX Boundary |
| **8 MB** | **L3 Shared Cache Limit** | **96.39 ns** | ~385 cycles | L3 Capacity Threshold |
| **16 MB** | **DRAM (Main RAM)** | **127.03 ns** | ~508 cycles | DDR4 Memory Fetch |
| **32 MB** | **DRAM (Main RAM)** | **148.37 ns** | ~593 cycles | DDR4 Memory Fetch |
| **64 MB** | **DRAM (Main RAM)** | **129.94 ns** | ~520 cycles | DDR4 Memory Fetch |

---

## 2. Key Microarchitectural Takeaways
1. **L1 Cache Line Latency**: Sustained $\approx 1.0\text{ ns}$ (4 cycles) across Core 0, perfectly adhering to Zen 2 L1 specifications.
2. **L2 Transition Plateau**: Drops from L1 at 32 KB, stabilizing at $\approx 3.1\text{ ns}$ (12–14 cycles) up to the 512 KB per-core boundary.
3. **The RAM Cliff**: Once working set exceeds the CCX cache slices, access latency surges from $\approx 10\text{ ns}$ to over **$120\text{ ns}$**—a **$120\times$ penalty** compared to L1 cache hits.
