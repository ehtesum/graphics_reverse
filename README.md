# 🔬 Hardware Exploration Lab (`graphics_reverse`)

## 💡 What is this in simple words?

Normally when you write software, you write in Python, C++, or JavaScript, and the operating system handles the computer chip for you.

This project is a **hands-on hardware testing lab** where we talk directly to the raw silicon chips—specifically:
- **NVIDIA GPU** (GTX 1650 / Turing architecture)
- **AMD CPU** (Ryzen 7 4800HS / Zen 2 architecture)

Instead of just believing computer architecture textbooks, this lab runs small benchmark experiments to **measure how computer chips actually behave in real life**:
- What happens when 32 cores try to grab memory from the same spot at the exact same nanosecond? *(Spoiler: they wait in line and run 25x slower!)*
- How much faster is 16-bit half-precision math compared to standard 32-bit math?
- How many clock cycles does it take to fetch data from L1 cache vs L2 cache vs main RAM?

---

## 🧪 The Experiments & What Each One Tests

| Experiment | Name | What it tests in simple words | Real-world result |
| :--- | :--- | :--- | :--- |
| **`008`** | **Warp Divergence** | What happens when half the GPU cores want to do `if` and the other half want to do `else`? | The GPU has to run one branch, freeze the other threads, and then run the other branch. |
| **`010`** | **Bank Conflicts** | Tests GPU fast shared memory. If all 32 threads in a group access memory with bad spacing (stride 32), they hit the same bank. | **25.6x slowdown!** (7.37 ms vs 0.29 ms). |
| **`011`** | **Warp Shuffle** | Tests passing data directly between core registers using `__shfl_sync` instead of writing to RAM. | Much faster than writing to memory. |
| **`012`** | **Packed Math (FP16)** | Tests running two 16-bit numbers packed together in one clock tick. | Up to 2x arithmetic throughput for AI workloads. |
| **`013`** | **Cache Latency** | Measures exact clock ticks required to get data: L1 cache (~45 cycles) → L2 cache (~209 cycles) → Main GPU RAM (~390 cycles). | Proves clear memory step tiers. |
| **`014`** | **Special Functions (SFU)**| Tests math functions like `sin()`, `cos()`, and square root (`rsqrt`). | Runs on special dedicated units inside the GPU. |
| **`016`** | **Atomic Contention** | What happens when many cores try to increment the exact same counter at once? | Compares CAS (compare-and-swap) loops vs hardware atomic addition. |

---

## 📁 What does each file & folder do?

- **`run_benchmarks.py`**: **The main tool!** Python script that lets you choose and run GPU benchmarks from your terminal.
- **`gpu/`**: Contains the raw CUDA / PTX assembly code for the GPU micro-benchmarks.
- **`cpu/`**: Contains C++ benchmarks testing AVX2 256-bit vector math and CPU cache latencies.
- **`paper/`**: Written research notes and whitepaper summarizing the measured numbers and graphs.
- **`results/`**: Log files and raw measurements produced by the benchmark runs.

---

## 🚀 How to Run the GPU Experiments

1. Open PowerShell and go to this directory:
   ```powershell
   cd c:\Users\user\Desktop\projects\graphics_reverse
   ```

2. **List all available benchmarks:**
   ```powershell
   python run_benchmarks.py --list
   ```

3. **Run a specific experiment (e.g. Bank Conflicts #010):**
   ```powershell
   python run_benchmarks.py --exp 010
   ```

4. **Run Warp Divergence (#008):**
   ```powershell
   python run_benchmarks.py --exp 008
   ```

It will run the test on your NVIDIA GPU, measure the time in microseconds, and display the output directly in the console.

---

## Technical Limitations

- **Microarchitecture Specificity**: Kernel calibration and bank stride mappings target NVIDIA Turing (SM 7.5); newer architectures (Ampere, Ada, Hopper) possess unified L1 caches and asynchronous tensor pipelines that shift contention thresholds.
- **WDDM Driver Launch Overhead**: Running under the Windows Display Driver Model (WDDM) adds ~5–10 μs of kernel launch latency overhead compared to native Linux TCC (Tesla Compute Cluster) runtimes.
- **Dynamic Clock Frequency Jitter**: Automated GPU Boost clock adjustments during thermal ramping introduce minor runtime measurement variance unless core and memory clocks are pinned via `nvidia-smi --lock-gpu-clocks`.

---

## Future Roadmap

- [ ] **Tensor Core PTX Micro-benchmarks**: Implement raw inline PTX `wmma.mma.sync` instruction kernels to profile tensor core throughput limits.
- [ ] **Hardware Performance Counter Profiling**: Direct integration with NVIDIA CUPTI / PerfWorks API for automated hardware counter sampling (warp stall reasons, cache hit ratios).
- [ ] **Empirical Roofline Generator**: Automated script generating empirical Roofline model plots comparing arithmetic intensity against measured DRAM and SRAM bandwidth.
- [ ] **Cross-Vendor Vulkan Compute Port**: Port micro-benchmarks to Vulkan Compute shaders to allow comparative analysis against AMD RDNA and Apple Silicon architectures.

