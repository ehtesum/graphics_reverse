# Experiment 002: Dual Additions (Optimization & Instruction Fusion)

## 1. Hypothesis
Depending on compiler optimization (-O2), NVCC will either:
- **Optimization Path A (Constant Folding)**: Fold `+ 1.0f` and `+ 2.0f` into a single `+ 3.0f` FADD instruction, producing identical SASS instruction count to Experiment 001.
- **Optimization Path B (Strict Sequence)**: Emit two distinct `FADD` instructions sequentially sharing the accumulator register.

## 2. Independent Variable
Dual sequential additions (`x[i] += 1.0f; x[i] += 2.0f;`).

## 3. Controlled Variables
- **Target Architecture**: NVIDIA Turing (`sm_75`, GTX 1650 4 GB)
- **Workload Size**: $N = 1,048,576$ elements (4 MiB float array)
- **Thread Block Configuration**: 256 threads/block, 4096 blocks
- **Optimization Flag**: `-O2`

## 4. Method
1. Compile `experiments/002_double_add.cu` using `tools/run_gpu_experiment.py`.
2. Compare SASS diff against Experiment 001 using `tools/compare_experiments.py results/001_add_one results/002_double_add`.
3. Verify output element equals `4.0000` ($1.0 + 1.0 + 2.0$).

## 5. Expected Observation
- If constant-folded: Exactly 1 `FADD` with immediate `0x40400000` (3.0f).
- If not folded: Exactly 2 `FADD` instructions.
- Verification status: `true`.

## 6. Actual Observation
- **Physical Device**: NVIDIA GeForce GTX 1650 (Turing Compute 7.5, WDDM)
- **Kernel Time**: 0.133056 ms (1,048,576 elements)
- **Output Verification**: `true` (First element output: `4.0000`)
- **Total SASS Instructions**: 12 (+1 instruction compared to 001)
  - Loads: 1 (`LDG.E.SYS`)
  - Stores: 1 (`STG.E.SYS`)
  - Arithmetic: 4 (`IMAD`, `IMAD.WIDE`, `FADD`, `FADD`)
  - Control: 2 (`EXIT`, `BRA`)
  - Movement: 2 (`MOV`, `S2R`)
- **Registers Used**: 5 unique registers (`R0`, `R1`, `R2`, `R3`, `R5`)
- **Status**: PASSED (Reproducible)

## 7. Epistemic Classification
- **OBSERVED**: SASS instruction count difference vs Experiment 001.
- **INFERRED**: Compiler AST optimization pass behavior.
- **UNKNOWN**: Non-observable intermediate AST transforms within NVVM.

## 8. Next Experiment
Experiment 003: Arithmetic dependency chain (`* 2.0f + 3.0f, square, - 7.0f`) to force FMA instruction generation and register pressure.
