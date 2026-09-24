# Experiment 007: PCIe Host-to-Device Memory Transfer Characterization

Empirical characterization of the PCIe 3.0 bus connecting the AMD Ryzen 7 4800HS CPU and NVIDIA GeForce GTX 1650 GPU on your ASUS laptop.

## 1. Measured Throughput Comparison

| Buffer Size | Direction | Pageable Memory Throughput | Pinned (Page-Locked) Throughput | Pinned Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **1 MB** | Host $\to$ Device (H2D) | 4.76 GB/s (0.205 ms) | **5.60 GB/s (0.175 ms)** | **+17.6%** |
| **1 MB** | Device $\to$ Host (D2H) | 3.11 GB/s (0.314 ms) | **5.45 GB/s (0.179 ms)** | **+75.2%** |
| **4 MB** | Host $\to$ Device (H2D) | 1.96 GB/s (1.993 ms) | **5.88 GB/s (0.664 ms)** | **+200% (3x faster)** |
| **4 MB** | Device $\to$ Host (D2H) | 3.82 GB/s (1.023 ms) | **5.94 GB/s (0.658 ms)** | **+55.5%** |
| **16 MB** | Host $\to$ Device (H2D) | 4.72 GB/s (3.312 ms) | **5.47 GB/s (2.855 ms)** | **+15.9%** |
| **16 MB** | Device $\to$ Host (D2H) | 4.51 GB/s (3.464 ms) | **5.28 GB/s (2.960 ms)** | **+17.1%** |
| **64 MB** | Host $\to$ Device (H2D) | 4.46 GB/s (14.028 ms) | **6.11 GB/s (10.232 ms)** | **+37.0%** |
| **64 MB** | Device $\to$ Host (D2H) | 4.73 GB/s (13.202 ms) | **5.64 GB/s (11.080 ms)** | **+19.2%** |
| **128 MB** | Host $\to$ Device (H2D) | 4.73 GB/s (26.414 ms) | **5.89 GB/s (21.214 ms)** | **+24.5%** |
| **128 MB** | Device $\to$ Host (D2H) | 4.80 GB/s (26.057 ms) | **5.84 GB/s (21.390 ms)** | **+21.7%** |

---

## 2. Microarchitectural Interpretations
1. **PCIe 3.0 Link Saturation**: Pinned memory achieves a sustained **~6.0 GB/s** transfer rate, directly utilizing the physical PCIe 3.0 link without OS buffer intervention.
2. **Pageable Staging Overhead**: Pageable transfers drop as low as **1.96 GB/s** due to the Windows kernel staging copies between user pages and DMA-accessible system pages.
3. **Bi-Directional Balance**: H2D and D2H show nearly identical maximum bandwidth plateaus (~6.1 GB/s H2D, ~5.9 GB/s D2H).
