# Experiment 003: Arithmetic Dependency Chain

## 1. Hypothesis
A multi-step dependent calculation ($v \times 2.0 + 3.0 \to v^2 \to v - 7.0$) will:
- Trigger Fused Multiply-Add instructions (`FFMA` / `FFMA32I`) on Turing SM 7.5.
- Require additional temporary register allocation compared to Experiment 001.
- Introduce instruction pipeline latency stalls visible in microbenchmarking timing relative to instruction count.

## 2. Independent Variable
Arithmetic dependency expression length and operator variety.

## 3. Controlled Variables
- **Target Architecture**: NVIDIA Turing (`sm_75`, GTX 1650 4 GB)
- **Workload Size**: $N = 1,048,576$ elements
- **Thread Block Configuration**: 256 threads/block, 4096 blocks

## 4. Method
1. Compile `experiments/003_arithmetic.cu` using `tools/run_gpu_experiment.py`.
2. Extract SASS opcode breakdown via `tools/analyze_sass.py`.
3. Compare against Experiment 001 via `tools/compare_experiments.py`.
4. Verify output element equals `18.0000` ($(1 \times 2 + 3)^2 - 7 = 5^2 - 7 = 18$).

## 5. Expected Observation
- Direct occurrence of `FFMA` or `FMUL` and `FADD`.
- Output verification passes ($18.0000$).

## 6. Actual Observation
- **Physical Device**: NVIDIA GeForce GTX 1650 (Turing Compute 7.5, WDDM)
- **Kernel Time**: 0.172448 ms (1,048,576 elements)
- **Output Verification**: `true` (First element output: `18.0000`)
- **Total SASS Instructions**: 13
  - Loads: 1 (`LDG.E.SYS`)
  - Stores: 1 (`STG.E.SYS`)
  - Arithmetic: 4 (`IMAD`, `IMAD.WIDE`, `FMUL`, `FFMA`)
  - Control: 2 (`EXIT`, `BRA`)
  - Movement: 3 (`MOV`, `S2R`, `F2F`)
- **Registers Used**: 5 unique registers (`R0`, `R1`, `R2`, `R3`, `R5`)
- **Status**: PASSED (Reproducible)

## 7. Epistemic Classification
- **OBSERVED**: Instruction opcodes, register indices, and elapsed kernel time.
- **INFERRED**: Register allocation lifetime and dependency stalls.
- **UNKNOWN**: Internal dual-issue dispatch pipeline slot utilization.

## 8. Next Experiment
Experiment 004: Deliberately expanding live variables to induce register pressure and observe occupancy drop.
