# Experimental Limitations & Environmental Scope

Scientific rigor demands transparent documentation of the boundary conditions, constraints, and sources of uncertainty that bound these experimental results.

---

## 1. Operating System & Driver Abstraction Layers

### 1.1 Windows 11 WDDM Command Batching
All GPU microbenchmarks were executed under the Windows Display Driver Model (WDDM 2.7) rather than Linux with the NVIDIA open kernel module or a headless TCC (Tesla Compute Cluster) environment. WDDM introduces specific artifacts:
* **User-Mode Queueing**: Calls to `cuLaunchKernel` and `cuEventRecord` are queued into the user-mode display driver pushbuffer (`nvcuda.dll`), introducing ~9.2 µs of submission latency.
* **DWM Compositing Contention**: The Desktop Window Manager (DWM) actively reserves ~500–650 MB of VRAM for desktop framebuffers, reducing usable physical VRAM from 4,096 MB to ~3,446 MB.
* **TCC Mode Inavailability**: The GeForce GTX 1650 (consumer Turing SKU) cannot be switched into TCC mode, precluding bypass of the WDDM scheduling pipeline.

### 1.2 Unprivileged User-Mode Profiling
Measurements were conducted entirely in Windows user mode without ring-0 kernel driver modifications. Consequently:
* Hardware Performance Monitor Counters (PMCs) requiring elevated administrative permissions or kernel-mode MSR access could not be queried continuously at cycle boundaries.
* OS thread scheduling interrupts, DPC/ISR spikes, and Windows Update background threads can introduce transient latency spikes (mitigated by repeating runs and filtering outliers).

---

## 2. Hardware-Specific Thermal & Power Envelope

### 2.1 Monolithic Renoir APU Package PPT Limits
The AMD Ryzen 7 4800HS is a 35W/45W mobile processor. While single-core workloads boost to 4.2 GHz, sustained 8-core AVX2 workloads draw excess current, triggering hardware power package tracking (PPT) limits. This forces all-core clocks to degrade to ~2.95 GHz, limiting AVX2 multi-threaded scaling to 4.59x instead of the theoretical 8.0x.

### 2.2 Turing GPU Boost Clocks
The GTX 1650 features a base clock of 1485 MHz and a boost clock of 1665 MHz. During sustained compute runs, clock frequencies fluctuate depending on VRM temperature and silicon junction limits. GPU cycle counts measured via `%clock64` capture SM clock cycles at current frequency, but absolute wall-clock durations can vary by ±3% across extended runs due to clock stepping.

---

## 3. Local AI Inference Modeling Scope

The analytical roofline predictions in Section 5 of the whitepaper reflect memory bandwidth, arithmetic peak capabilities, and PCIe transfer limits under optimized tensor kernels (e.g. FlashAttention / AWQ / ExLlamaV2). Actual runtime performance in specific inference runtimes (such as `llama.cpp` or TensorRT-LLM) may experience additional software overhead from tokenization, sampler logic, and CUDA graph instantiation.
