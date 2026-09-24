# Experiment 001: Baseline Single Addition

## 1. Hypothesis
A single scalar addition to an array of single-precision floating point numbers (`x[i] += 1.0f`) will produce:
- A single global memory load (`LDG`)
- A single 32-bit floating-point addition (`FADD`)
- A single global memory store (`STG`)
- Baseline register allocation (minimal live state per thread)
- Full warp occupancy potential on Turing SM 7.5

## 2. Independent Variable
Single floating-point addition operation (`+ 1.0f`).

## 3. Controlled Variables
- **Target Architecture**: NVIDIA Turing (`sm_75`, GTX 1650 4 GB)
- **Workload Size**: $N = 1,048,576$ elements (4 MiB float array)
- **Thread Block Configuration**: 256 threads/block, 4096 blocks
- **Host OS & Driver**: Windows 11, NVIDIA Driver 555.97

## 4. Method
1. Allocate host buffer with all values initialized to `1.0f`.
2. Allocate device buffer via `cudaMalloc` and transfer host data to device (`cudaMemcpyHostToDevice`).
3. Launch kernel `test<<<4096, 256>>>(d_data)` and measure execution duration using `cudaEventRecord`.
4. Copy result buffer back to host (`cudaMemcpyDeviceToHost`) and verify that first element and subsequent sample elements equal `2.0f`.
5. Compile with `nvcc -O2 -arch=sm_75` to generate PTX and CUBIN.
6. Disassemble CUBIN with `cuobjdump --dump-sass` and run static analysis via `tools/analyze_sass.py`.

## 5. Expected Observation
- **PTX**: Contains `ld.global.f32`, `add.f32`, and `st.global.f32`.
- **SASS**: Contains `LDG.E.SYS`, `FADD`, and `STG.E.SYS`.
- **Output**: First element verified as `2.0000`.

## 6. Actual Observation
- **Physical Device**: NVIDIA GeForce GTX 1650 (Turing Compute 7.5, WDDM)
- **Kernel Time**: 0.19408 ms (1,048,576 elements)
- **Output Verification**: `true` (First element output: `2.0000`)
- **Total SASS Instructions**: 11
  - Loads: 1 (`LDG.E.SYS`)
  - Stores: 1 (`STG.E.SYS`)
  - Arithmetic: 3 (`IMAD`, `IMAD.WIDE`, `FADD`)
  - Control: 2 (`EXIT`, `BRA`)
  - Movement: 2 (`MOV`, `S2R`)
- **Registers Used**: 5 unique registers (`R0`, `R1`, `R2`, `R3`, `R5`)
- **Status**: PASSED (Reproducible)

## 7. Epistemic Classification
- **OBSERVED**: Direct instructions and timing from disassembler and CUDA events.
- **INFERRED**: Continuous memory coalescing across 32 threads in a warp.
- **UNKNOWN**: Internal SM warp issue cycle stalls.

## 8. Next Experiment
Experiment 002: Dual sequential additions (`x[i] += 1.0f; x[i] += 2.0f;`) to test compiler instruction fusion / constant folding.
