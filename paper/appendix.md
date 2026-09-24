# Appendix: Artifacts, Disassembly & Reproduction Commands

This appendix provides the compilation toolchains, SASS disassembly listings, PTX templates, and reproduction instructions necessary for independent verification.

---

## 1. Toolchain & Environment Manifest

| Component | Identifier / Version | Path in Repository |
| :--- | :--- | :--- |
| **GPU Assembler** | NVIDIA `ptxas` 12.5.82 | `cuda_toolkit/bin/ptxas.exe` |
| **GPU Disassembler** | NVIDIA `cuobjdump` 12.5.82 | `cuda_toolkit/bin/cuobjdump.exe` |
| **Host C++ Compiler** | MinGW-w64 GCC 13.2.0 | `C:/tools/mingw64/bin/g++.exe` |
| **GPU Driver Runtime** | NVIDIA CUDA Driver 555.97 | `C:/Windows/System32/nvcuda.dll` |
| **DirectX Graphics Interface** | Microsoft DXGI 1.6 | `C:/Windows/System32/dxgi.dll` |
| **Python Interpreter** | Python 3.12 64-bit | Local Environment |

---

## 2. Key PTX Kernel Listings

### 2.1 Shared Memory Bank Stride Kernel (`BankFalsify`)
```ptx
.version 7.5
.target sm_75
.address_size 64
.visible .entry BankFalsify(.param .u64 param_arr, .param .u32 param_iters) {
  .shared .f32 smem[4096];
  .reg .b32 %r<15>;
  .reg .f32 %f<10>;
  .reg .b64 %rd<5>;
  .reg .pred %p<2>;
  ld.param.u64 %rd1, [param_arr];
  ld.param.u32 %r1, [param_iters];
  mov.u32 %r2, %tid.x;
  mul.lo.u32 %r3, %r2, STRIDE_VAL;
  and.b32 %r4, %r3, 4095;
  mul.wide.u32 %rd2, %r4, 4;
  mov.u64 %rd3, smem;
  add.s64 %rd4, %rd3, %rd2;
  mov.f32 %f1, 1.0;
  st.shared.f32 [%rd4], %f1;
  bar.sync 0;
  mov.u32 %r5, 0;
LOOP:
  ld.shared.f32 %f2, [%rd4];
  add.f32 %f2, %f2, 0.001;
  st.shared.f32 [%rd4], %f2;
  ld.shared.f32 %f2, [%rd4];
  add.f32 %f2, %f2, 0.001;
  st.shared.f32 [%rd4], %f2;
  add.u32 %r5, %r5, 1;
  setp.lt.u32 %p1, %r5, %r1;
  @%p1 bra LOOP;
  bar.sync 0;
  mul.wide.u32 %rd2, %r2, 4;
  add.s64 %rd3, %rd1, %rd2;
  st.global.f32 [%rd3], %f2;
  ret;
}
```

### 2.2 In-Situ L2 Atomic Reduction SASS vs CAS SASS
Disassembly produced via `cuobjdump --dump-sass`:

```sass
// Fire-and-Forget atomicAdd path:
MOV R4, 0x1;
RED.E.ADD.STRONG.GPU [R2], R4;     // Handed off directly to L2 crossbar

// Compare-and-Swap path:
MOV R4, 0x1;
MOV R3, 0x0;
ATOMG.E.CAS R5, [R2], R4, R3;      // Must stall SM pipeline until R5 is returned
```

---

## 3. Reproduction Commands

To reproduce the complete empirical suite from clean source:

```powershell
# 1. Run Repository Audit & Registry Validation
python tools/audit_engine.py
python tools/generate_claim_audit.py
python tools/build_660_registry.py
python tools/build_dependencies_dag.py

# 2. Compile and Execute CPU Benchmarks
C:\tools\mingw64\bin\g++.exe -O3 cpu/benchmarks/cache_hierarchy.cpp -o cpu/benchmarks/cache_hierarchy.exe
.\cpu\benchmarks\cache_hierarchy.exe

C:\tools\mingw64\bin\g++.exe -O3 cpu/benchmarks/inter_core_latency.cpp -o cpu/benchmarks/inter_core_latency.exe
.\cpu\benchmarks\inter_core_latency.exe

C:\tools\mingw64\bin\g++.exe -O3 -mavx2 -mfma cpu/benchmarks/avx2_throughput.cpp -o cpu/benchmarks/avx2_throughput.exe
.\cpu\benchmarks\avx2_throughput.exe

C:\tools\mingw64\bin\g++.exe -O3 cpu/benchmarks/falsify_branch.cpp -o cpu/benchmarks/falsify_branch.exe
.\cpu\benchmarks\falsify_branch.exe

# 3. Execute GPU Microbenchmarks
python experiments/007_pcie_transfer.py
python experiments/008_warp_divergence.py
python experiments/009_engine_concurrency.py
python experiments/010_bank_conflicts.py
python experiments/011_warp_shuffle.py
python experiments/012_fp16_packed_math.py
python experiments/013_gpu_cache_latency.py
python experiments/014_sfu_throughput.py
python experiments/015_zerocopy_streaming.py
python experiments/016_atomic_contention.py

# 4. Run Adversarial Falsification Suite
python experiments/falsification_suite.py

# 5. Run Resource-Aware Local AI Engine & Simulation
python tools/resource_aware_ai.py
```
