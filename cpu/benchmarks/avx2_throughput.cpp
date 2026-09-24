/**
 * cpu/benchmarks/avx2_throughput.cpp
 * Step 4: Zen 2 (AMD Ryzen 7 4800HS) AVX2 FMA Vector Pipeline Throughput
 *
 * Measures:
 *   1. Latency bound (1 accumulator): Serial dependency chain measuring FMA instruction latency (~4 cycles).
 *   2. Peak Single-Core Throughput (12 accumulators): Dual 256-bit FMA port saturation (Port 0 + Port 1).
 *   3. Multi-Core Scaling (1, 2, 4, 8 threads): Measuring aggregate multi-core AVX2 GFLOPS on 8 Zen 2 cores.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <immintrin.h>
#include <omp.h>

const uint64_t ITERS = 100'000'000ULL; // 100 million iterations

#include "common.hpp"
using cpu_bench::measure_cpu_ghz;
using cpu_bench::pin_to_core;
using cpu_bench::rdtsc_fence;

// 1. Dependency Chain: 1 accumulator
// Computes 2 FLOPs per FMA * 8 floats = 16 FLOPs per iteration
double benchmark_latency(int iters) {
    __m256 a = _mm256_set1_ps(1.00001f);
    __m256 b = _mm256_set1_ps(0.99999f);
    __m256 c0 = _mm256_set1_ps(1.0f);

    _mm_mfence();
    uint64_t start = __rdtsc();

    for (int i = 0; i < iters; ++i) {
        c0 = _mm256_fmadd_ps(c0, a, b);
        c0 = _mm256_fmadd_ps(c0, a, b);
        c0 = _mm256_fmadd_ps(c0, a, b);
        c0 = _mm256_fmadd_ps(c0, a, b);
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();

    // Prevent compiler dead-code elimination
    volatile float dummy;
    float res[8];
    _mm256_storeu_ps(res, c0);
    dummy = res[0];
    (void)dummy;

    uint64_t total_fmas = (uint64_t)iters * 4;
    return (double)(stop - start) / total_fmas; // Cycles per instruction (Latency)
}

// 2. Peak Throughput: 12 independent accumulators unrolled
// 12 FMAs per inner step * 8 SP floats * 2 FLOPs = 192 FLOPs per unrolled step
double benchmark_peak_single_core(uint64_t iterations, double& out_gflops, double& out_ipc) {
    __m256 a = _mm256_set1_ps(1.000001f);
    __m256 b = _mm256_set1_ps(0.000001f);

    __m256 v0 = _mm256_set1_ps(1.0f);
    __m256 v1 = _mm256_set1_ps(2.0f);
    __m256 v2 = _mm256_set1_ps(3.0f);
    __m256 v3 = _mm256_set1_ps(4.0f);
    __m256 v4 = _mm256_set1_ps(5.0f);
    __m256 v5 = _mm256_set1_ps(6.0f);
    __m256 v6 = _mm256_set1_ps(7.0f);
    __m256 v7 = _mm256_set1_ps(8.0f);
    __m256 v8 = _mm256_set1_ps(9.0f);
    __m256 v9 = _mm256_set1_ps(10.0f);
    __m256 v10 = _mm256_set1_ps(11.0f);
    __m256 v11 = _mm256_set1_ps(12.0f);

    LARGE_INTEGER freq, qpc_start, qpc_end;
    QueryPerformanceFrequency(&freq);

    _mm_mfence();
    QueryPerformanceCounter(&qpc_start);
    uint64_t tsc_start = __rdtsc();

    for (uint64_t i = 0; i < iterations; ++i) {
        v0  = _mm256_fmadd_ps(v0,  a, b);
        v1  = _mm256_fmadd_ps(v1,  a, b);
        v2  = _mm256_fmadd_ps(v2,  a, b);
        v3  = _mm256_fmadd_ps(v3,  a, b);
        v4  = _mm256_fmadd_ps(v4,  a, b);
        v5  = _mm256_fmadd_ps(v5,  a, b);
        v6  = _mm256_fmadd_ps(v6,  a, b);
        v7  = _mm256_fmadd_ps(v7,  a, b);
        v8  = _mm256_fmadd_ps(v8,  a, b);
        v9  = _mm256_fmadd_ps(v9,  a, b);
        v10 = _mm256_fmadd_ps(v10, a, b);
        v11 = _mm256_fmadd_ps(v11, a, b);
    }

    _mm_mfence();
    QueryPerformanceCounter(&qpc_end);
    uint64_t tsc_end = __rdtsc();

    // Prevent DCE: sum all 12 accumulators
    float sum_arr[8];
    __m256 s0 = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
    __m256 s1 = _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7));
    __m256 s2 = _mm256_add_ps(_mm256_add_ps(v8, v9), _mm256_add_ps(v10, v11));
    __m256 total = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);
    _mm256_storeu_ps(sum_arr, total);
    volatile float dummy = sum_arr[0];
    (void)dummy;

    double elapsed_sec = (double)(qpc_end.QuadPart - qpc_start.QuadPart) / (double)freq.QuadPart;
    uint64_t total_cycles = tsc_end - tsc_start;
    uint64_t total_fmas = iterations * 12;
    uint64_t total_flops = total_fmas * 16; // 8 floats * 2 ops

    out_gflops = (total_flops / elapsed_sec) / 1e9;
    out_ipc = (double)total_fmas / (double)total_cycles; // Vector instructions per cycle (max 2.0 on Zen 2)

    return elapsed_sec;
}

// 3. Multi-Core Benchmark
double benchmark_multicore(int num_threads, uint64_t iters_per_thread) {
    LARGE_INTEGER freq, qpc_start, qpc_end;
    QueryPerformanceFrequency(&freq);

    std::vector<float> thread_results(num_threads, 0.0f);

    QueryPerformanceCounter(&qpc_start);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        __m256 a = _mm256_set1_ps(1.000001f);
        __m256 b = _mm256_set1_ps(0.000001f);

        __m256 v0 = _mm256_set1_ps(1.0f);
        __m256 v1 = _mm256_set1_ps(2.0f);
        __m256 v2 = _mm256_set1_ps(3.0f);
        __m256 v3 = _mm256_set1_ps(4.0f);
        __m256 v4 = _mm256_set1_ps(5.0f);
        __m256 v5 = _mm256_set1_ps(6.0f);
        __m256 v6 = _mm256_set1_ps(7.0f);
        __m256 v7 = _mm256_set1_ps(8.0f);

        for (uint64_t i = 0; i < iters_per_thread; ++i) {
            v0 = _mm256_fmadd_ps(v0, a, b);
            v1 = _mm256_fmadd_ps(v1, a, b);
            v2 = _mm256_fmadd_ps(v2, a, b);
            v3 = _mm256_fmadd_ps(v3, a, b);
            v4 = _mm256_fmadd_ps(v4, a, b);
            v5 = _mm256_fmadd_ps(v5, a, b);
            v6 = _mm256_fmadd_ps(v6, a, b);
            v7 = _mm256_fmadd_ps(v7, a, b);
        }

        float dummy_arr[8];
        __m256 tot = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
        tot = _mm256_add_ps(tot, _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
        _mm256_storeu_ps(dummy_arr, tot);
        thread_results[tid] = dummy_arr[0];
    }

    QueryPerformanceCounter(&qpc_end);
    double elapsed = (double)(qpc_end.QuadPart - qpc_start.QuadPart) / (double)freq.QuadPart;

    volatile float sink = 0.0f;
    for (float r : thread_results) sink += r;
    (void)sink;

    uint64_t total_flops = (uint64_t)num_threads * iters_per_thread * 8 * 16;
    return (total_flops / elapsed) / 1e9;
}

int main() {
    std::cout << "===========================================================================\n";
    std::cout << " Step 4: Zen 2 (Ryzen 7 4800HS) AVX2 FMA Vector Pipeline Microbenchmark\n";
    std::cout << "===========================================================================\n";

    double ghz = measure_cpu_ghz();
    std::cout << "[*] Measured Base CPU Clock Frequency: " << std::fixed << std::setprecision(2) << ghz << " GHz\n";

    // Pin main thread to physical Core 0
    pin_to_core(0);

    std::cout << "[*] Benchmarking FMA Instruction Latency (Serial Dependency Chain)...\n";
    double latency_cycles = benchmark_latency(20'000'000);
    std::cout << "    - FMA Dependency Latency: " << std::setprecision(2) << latency_cycles << " cycles/inst ("
              << std::setprecision(2) << (latency_cycles / ghz) << " ns)\n";

    std::cout << "[*] Benchmarking Peak Single-Core AVX2 Throughput (12 Accumulators Unrolled)...\n";
    double single_gflops = 0.0, single_ipc = 0.0;
    double sec = benchmark_peak_single_core(ITERS, single_gflops, single_ipc);
    std::cout << "    - Execution Time: " << std::setprecision(3) << sec << " s\n";
    std::cout << "    - Retired 256-bit FMAs/Cycle (IPC): " << std::setprecision(3) << single_ipc << " / cycle (Max: 2.00)\n";
    std::cout << "    - Single-Core Peak Throughput: " << std::setprecision(2) << single_gflops << " GFLOPS\n\n";

    std::cout << "---------------------------------------------------------------------------\n";
    std::cout << " Multi-Core AVX2 Scaling Across Physical Cores (OpenMP)\n";
    std::cout << "---------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(12) << "Threads"
              << std::setw(18) << "Aggregate GFLOPS"
              << std::setw(18) << "Speedup vs 1T"
              << std::setw(18) << "Efficiency" << "\n";
    std::cout << "---------------------------------------------------------------------------\n";

    std::vector<int> thread_counts = {1, 2, 4, 8};
    double base_multi_gflops = 0.0;

    std::ofstream json_out("results/cpu_benchmarks/avx2_throughput.json");
    json_out << "{\n";
    json_out << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2)\",\n";
    json_out << "  \"fma_latency_cycles\": " << latency_cycles << ",\n";
    json_out << "  \"single_core_peak_gflops\": " << single_gflops << ",\n";
    json_out << "  \"single_core_fma_ipc\": " << single_ipc << ",\n";
    json_out << "  \"scaling\": [\n";

    for (size_t idx = 0; idx < thread_counts.size(); ++idx) {
        int t = thread_counts[idx];
        double gflops = benchmark_multicore(t, 50'000'000ULL);
        if (t == 1) base_multi_gflops = gflops;

        double speedup = gflops / base_multi_gflops;
        double eff = (speedup / t) * 100.0;

        std::cout << std::left << std::setw(12) << (std::to_string(t) + " Threads")
                  << std::setw(18) << std::fixed << std::setprecision(2) << gflops
                  << std::setw(18) << (std::to_string(speedup).substr(0, 4) + "x")
                  << std::setw(18) << (std::to_string(eff).substr(0, 4) + "%") << "\n";

        json_out << "    {\n";
        json_out << "      \"threads\": " << t << ",\n";
        json_out << "      \"gflops\": " << gflops << ",\n";
        json_out << "      \"speedup\": " << speedup << ",\n";
        json_out << "      \"efficiency_pct\": " << eff << "\n";
        json_out << (idx == thread_counts.size() - 1 ? "    }\n" : "    },\n");
    }

    json_out << "  ]\n}\n";
    json_out.close();

    std::cout << "===========================================================================\n";
    std::cout << "[SUCCESS] Saved results to: results/cpu_benchmarks/avx2_throughput.json\n\n";

    return 0;
}
