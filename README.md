# Hardware Exploration Lab (`graphics_reverse`)

A personal hobby project and hands-on learning lab exploring how modern GPUs (NVIDIA Turing / GTX 1650) and CPUs (AMD Zen 2 / Ryzen 7 4800HS) execute code below high-level APIs.

## Motivation & Purpose

This is an independent exploration project built to see how hardware operates in physical silicon rather than just reading textbook diagrams. By writing focused micro-benchmarks and inspecting compiler assembly (PTX and SASS), this lab measures the real-world performance impact of warp divergence, shared memory bank conflicts, and memory hierarchy latencies.

## Experimental Scope

### 1. GPU Microbenchmarks (NVIDIA Turing SM 7.5 / GTX 1650)
* **Warp Divergence (`008`)**: Observes execution serialization across 1-way to 32-way branch splits.
* **Shared Memory Bank Conflicts (`010`)**: Demonstrates the 32-bank modulo arbitration penalty across access strides.
* **Warp Shuffle & Reductions (`011`)**: Compares `__shfl_sync` register exchange against shared memory staging.
* **Packed Arithmetic (`012`)**: Benchmarks Rapid Packed Math (FP16 `HFMA2` vs FP32 `FMA`).
* **Cache Latency (`013`)**: Uses `%clock64` pointer chasing to observe L1, L2, and GDDR6 DRAM access boundaries.
* **Special Function Unit (`014`)**: Measures transcendental throughput (`sin`, `cos`, `rsqrt`) relative to standard ALUs.
* **Atomic Contention (`016`)**: Evaluates throughput scaling of atomic operations.

### 2. CPU Benchmarks (AMD Zen 2 / Ryzen 7 4800HS)
* **Vector Throughput**: AVX2 256-bit FMA dependency chains.
* **Instruction Latency**: Measures BMI2 `PDEP` / `PEXT` microcode latency.
* **Cache & Memory**: Serialized `__rdtsc` cycle profiling of cache levels and inter-core latency.

## Key Observations

| Experiment | Observed Behavior | Takeaway |
| :--- | :---: | :--- |
| **Shared Memory Stride 32** | 7.37 ms vs 0.29 ms (Stride 31) | **25.6x slowdown** when all 32 threads in a warp access the same memory bank. |
| **Memory Latency Tiers** | L1 (~45c) -> L2 (~209c) -> DRAM (~390c) | Hardware `%clock64` shows clear latency steps across the memory hierarchy. |
| **L2 Atomic Reductions** | `RED.E.ADD` is 31.3x faster than `ATOMG.E.CAS` | Fire-and-forget reductions avoid round-trip return stall. |
| **Zen 2 `PDEP`/`PEXT`** | 95 cycles (vs 2 cycles for `TZCNT`) | Confirms hardware traps these specific instructions into microcode. |

## Quick Start

### Run GPU Experiments

```bash
# List all available benchmarks
python run_benchmarks.py --list

# Run a specific experiment (e.g. 010 Shared Memory Bank Conflicts)
python run_benchmarks.py --exp 010
```

### Build & Run CPU Benchmarks

```bash
cmake -B build
cmake --build build --config Release

# Run AVX2 FMA throughput benchmark
./build/cpu/benchmarks/avx2_throughput
```

## Documentation

Full detailed notes, methodology, and measurement logs are documented in the [`paper/`](paper/) directory:
* [`paper/revised_whitepaper.md`](paper/revised_whitepaper.md): Extended measurement notes and telemetry logs.
* [`paper/methods.md`](paper/methods.md): Epistemological isolation principles and timer calibration notes.
