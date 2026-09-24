#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <x86intrin.h>
#include <fstream>
#include <iomanip>
#include <windows.h>

// Corner C2: Zen 2 AVX2 FMA Pipeline Latency vs Dependency Chain Depth
// Isolating 5-cycle physical execution latency vs 0.5-cycle reciprocal throughput (2 FMAs/cycle).

#define ITERS 50000000ULL

void pin_to_core(int core_id) {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(thread, mask);
}

struct FMAResult {
    int accumulators;
    double cycles_per_fma;
    double gflops;
    double elapsed_ms;
};

// Benchmarking functions with specific accumulator counts
__attribute__((noinline)) uint64_t bench_fma_1() {
    __m256 a0 = _mm256_set1_ps(1.0001f);
    __m256 mul = _mm256_set1_ps(1.000001f);
    __m256 add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(a0);
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_2() {
    __m256 a0 = _mm256_set1_ps(1.0001f), a1 = _mm256_set1_ps(1.0002f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 2; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a0, a1));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_3() {
    __m256 a0 = _mm256_set1_ps(1.0001f), a1 = _mm256_set1_ps(1.0002f), a2 = _mm256_set1_ps(1.0003f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 3; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a0, a1));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_4() {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = _mm256_set1_ps(1.0f), a2 = _mm256_set1_ps(1.0f), a3 = _mm256_set1_ps(1.0f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 4; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
        a3 = _mm256_fmadd_ps(a3, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a0, a1));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_5() {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = _mm256_set1_ps(1.0f), a2 = _mm256_set1_ps(1.0f), a3 = _mm256_set1_ps(1.0f), a4 = _mm256_set1_ps(1.0f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 5; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
        a3 = _mm256_fmadd_ps(a3, mul, add);
        a4 = _mm256_fmadd_ps(a4, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a0, a1));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_8() {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = _mm256_set1_ps(1.0f), a2 = _mm256_set1_ps(1.0f), a3 = _mm256_set1_ps(1.0f);
    __m256 a4 = _mm256_set1_ps(1.0f), a5 = _mm256_set1_ps(1.0f), a6 = _mm256_set1_ps(1.0f), a7 = _mm256_set1_ps(1.0f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 8; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
        a3 = _mm256_fmadd_ps(a3, mul, add);
        a4 = _mm256_fmadd_ps(a4, mul, add);
        a5 = _mm256_fmadd_ps(a5, mul, add);
        a6 = _mm256_fmadd_ps(a6, mul, add);
        a7 = _mm256_fmadd_ps(a7, mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a0, a1));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_10() {
    __m256 a[10];
    for (int k = 0; k < 10; ++k) a[k] = _mm256_set1_ps(1.0f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 10; ++i) {
        a[0] = _mm256_fmadd_ps(a[0], mul, add);
        a[1] = _mm256_fmadd_ps(a[1], mul, add);
        a[2] = _mm256_fmadd_ps(a[2], mul, add);
        a[3] = _mm256_fmadd_ps(a[3], mul, add);
        a[4] = _mm256_fmadd_ps(a[4], mul, add);
        a[5] = _mm256_fmadd_ps(a[5], mul, add);
        a[6] = _mm256_fmadd_ps(a[6], mul, add);
        a[7] = _mm256_fmadd_ps(a[7], mul, add);
        a[8] = _mm256_fmadd_ps(a[8], mul, add);
        a[9] = _mm256_fmadd_ps(a[9], mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a[0], a[1]));
    (void)sink;
    return (t1 - t0);
}

__attribute__((noinline)) uint64_t bench_fma_12() {
    __m256 a[12];
    for (int k = 0; k < 12; ++k) a[k] = _mm256_set1_ps(1.0f);
    __m256 mul = _mm256_set1_ps(1.000001f), add = _mm256_set1_ps(0.000001f);

    uint64_t t0 = __rdtsc();
    for (uint64_t i = 0; i < ITERS / 12; ++i) {
        a[0] = _mm256_fmadd_ps(a[0], mul, add);
        a[1] = _mm256_fmadd_ps(a[1], mul, add);
        a[2] = _mm256_fmadd_ps(a[2], mul, add);
        a[3] = _mm256_fmadd_ps(a[3], mul, add);
        a[4] = _mm256_fmadd_ps(a[4], mul, add);
        a[5] = _mm256_fmadd_ps(a[5], mul, add);
        a[6] = _mm256_fmadd_ps(a[6], mul, add);
        a[7] = _mm256_fmadd_ps(a[7], mul, add);
        a[8] = _mm256_fmadd_ps(a[8], mul, add);
        a[9] = _mm256_fmadd_ps(a[9], mul, add);
        a[10] = _mm256_fmadd_ps(a[10], mul, add);
        a[11] = _mm256_fmadd_ps(a[11], mul, add);
    }
    uint64_t t1 = __rdtsc();
    volatile float sink = _mm256_cvtss_f32(_mm256_add_ps(a[0], a[1]));
    (void)sink;
    return (t1 - t0);
}

int main() {
    pin_to_core(2); // Pin to Core 2 on primary CCX
    std::cout << "================================================================================" << std::endl;
    std::cout << " CORNER C2: Zen 2 AVX2 FMA Execution Pipeline Latency vs Dependency Chain Depth" << std::endl;
    std::cout << " Target: AMD Ryzen 7 4800HS (Zen 2 Renoir) | AVX2 256-bit Dual FMA Units" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Calibrate TSC frequency
    auto w0 = std::chrono::high_resolution_clock::now();
    uint64_t c0 = __rdtsc();
    Sleep(50);
    auto w1 = std::chrono::high_resolution_clock::now();
    uint64_t c1 = __rdtsc();
    double tsc_freq = (double)(c1 - c0) / std::chrono::duration<double>(w1 - w0).count();
    std::cout << "[*] Calibrated TSC Frequency: " << (tsc_freq / 1e9) << " GHz" << std::endl;

    std::vector<int> test_accs = {1, 2, 3, 4, 5, 8, 10, 12};
    std::vector<FMAResult> results;

    std::cout << "\n" << std::setw(15) << "Accumulators" 
              << " | " << std::setw(18) << "Cycles / FMA" 
              << " | " << std::setw(16) << "Instr / Cycle"
              << " | " << std::setw(16) << "Throughput GFLOPS" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (int acc : test_accs) {
        uint64_t total_fmas = 0;
        uint64_t cycles = 0;

        // Warmup
        bench_fma_1();

        if (acc == 1) {
            total_fmas = ITERS;
            cycles = bench_fma_1();
        } else if (acc == 2) {
            total_fmas = (ITERS / 2) * 2;
            cycles = bench_fma_2();
        } else if (acc == 3) {
            total_fmas = (ITERS / 3) * 3;
            cycles = bench_fma_3();
        } else if (acc == 4) {
            total_fmas = (ITERS / 4) * 4;
            cycles = bench_fma_4();
        } else if (acc == 5) {
            total_fmas = (ITERS / 5) * 5;
            cycles = bench_fma_5();
        } else if (acc == 8) {
            total_fmas = (ITERS / 8) * 8;
            cycles = bench_fma_8();
        } else if (acc == 10) {
            total_fmas = (ITERS / 10) * 10;
            cycles = bench_fma_10();
        } else if (acc == 12) {
            total_fmas = (ITERS / 12) * 12;
            cycles = bench_fma_12();
        }

        double c_per_fma = (double)cycles / (double)total_fmas;
        double fma_per_c = 1.0 / c_per_fma;
        // Each AVX2 256-bit float FMA = 8 operations * 2 FLOPs = 16 FLOPs
        double flops = (double)total_fmas * 16.0;
        double elapsed_sec = (double)cycles / tsc_freq;
        double gflops = (flops / elapsed_sec) / 1e9;

        results.push_back({acc, c_per_fma, gflops, elapsed_sec * 1000.0});

        std::cout << std::setw(15) << acc 
                  << " | " << std::setw(18) << std::fixed << std::setprecision(3) << c_per_fma 
                  << " | " << std::setw(16) << std::fixed << std::setprecision(3) << fma_per_c
                  << " | " << std::setw(16) << std::fixed << std::setprecision(1) << gflops << std::endl;
    }

    // Export to JSON
    std::ofstream ofs("research/raw/avx2_fma_dependency_results.json");
    ofs << "{\n";
    ofs << "  \"cpu\": \"AMD Ryzen 7 4800HS\",\n";
    ofs << "  \"microarchitecture\": \"Zen 2 (Renoir)\",\n";
    ofs << "  \"tsc_ghz\": " << std::fixed << std::setprecision(3) << (tsc_freq / 1e9) << ",\n";
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        ofs << "    {\n";
        ofs << "      \"accumulators\": " << results[i].accumulators << ",\n";
        ofs << "      \"cycles_per_fma\": " << std::fixed << std::setprecision(4) << results[i].cycles_per_fma << ",\n";
        ofs << "      \"instructions_per_cycle\": " << std::fixed << std::setprecision(4) << (1.0 / results[i].cycles_per_fma) << ",\n";
        ofs << "      \"gflops\": " << std::fixed << std::setprecision(2) << results[i].gflops << ",\n";
        ofs << "      \"elapsed_ms\": " << std::fixed << std::setprecision(2) << results[i].elapsed_ms << "\n";
        ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "\n[SUCCESS] Raw telemetry written to research/raw/avx2_fma_dependency_results.json" << std::endl;
    return 0;
}
