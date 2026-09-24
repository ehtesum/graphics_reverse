#include <windows.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>

void pin_thread(uint64_t core_mask) {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = static_cast<DWORD_PTR>(core_mask);
    SetThreadAffinityMask(thread, mask);
}

// 1. Throughput-Saturated AVX2 FMA (16 independent accumulators = 32 unrolled FMAs)
// This completely saturates both 256-bit Vector FMA ports on Zen 2.
void avx2_fma_peak_kernel(size_t iterations, double* duration_ms, double* gflops) {
    __m256 a = _mm256_set1_ps(1.000001f);
    __m256 b = _mm256_set1_ps(1.000002f);
    __m256 c0 = _mm256_set1_ps(0.1f), c1 = _mm256_set1_ps(0.2f);
    __m256 c2 = _mm256_set1_ps(0.3f), c3 = _mm256_set1_ps(0.4f);
    __m256 c4 = _mm256_set1_ps(0.5f), c5 = _mm256_set1_ps(0.6f);
    __m256 c6 = _mm256_set1_ps(0.7f), c7 = _mm256_set1_ps(0.8f);
    __m256 c8 = _mm256_set1_ps(0.9f), c9 = _mm256_set1_ps(1.0f);
    __m256 c10 = _mm256_set1_ps(1.1f), c11 = _mm256_set1_ps(1.2f);
    __m256 c12 = _mm256_set1_ps(1.3f), c13 = _mm256_set1_ps(1.4f);
    __m256 c14 = _mm256_set1_ps(1.5f), c15 = _mm256_set1_ps(1.6f);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        c0 = _mm256_fmadd_ps(a, b, c0);
        c1 = _mm256_fmadd_ps(a, b, c1);
        c2 = _mm256_fmadd_ps(a, b, c2);
        c3 = _mm256_fmadd_ps(a, b, c3);
        c4 = _mm256_fmadd_ps(a, b, c4);
        c5 = _mm256_fmadd_ps(a, b, c5);
        c6 = _mm256_fmadd_ps(a, b, c6);
        c7 = _mm256_fmadd_ps(a, b, c7);
        c8 = _mm256_fmadd_ps(a, b, c8);
        c9 = _mm256_fmadd_ps(a, b, c9);
        c10 = _mm256_fmadd_ps(a, b, c10);
        c11 = _mm256_fmadd_ps(a, b, c11);
        c12 = _mm256_fmadd_ps(a, b, c12);
        c13 = _mm256_fmadd_ps(a, b, c13);
        c14 = _mm256_fmadd_ps(a, b, c14);
        c15 = _mm256_fmadd_ps(a, b, c15);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float out[8];
    __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3)),
                               _mm256_add_ps(_mm256_add_ps(c4, c5), _mm256_add_ps(c6, c7)));
    _mm256_storeu_ps(out, sum);
    volatile float sink = out[0];
    (void)sink;

    std::chrono::duration<double, std::milli> elapsed = end - start;
    *duration_ms = elapsed.count();
    // 16 FMAs * 8 floats * 2 ops = 256 FLOPs per iteration
    double total_flops = static_cast<double>(iterations) * 256.0;
    *gflops = (total_flops / (elapsed.count() / 1000.0)) / 1e9;
}

// 2. Integer ALU Dependent Chain (cannot be closed-formed by optimizer)
void int_alu_dependent_kernel(size_t iterations, double* duration_ms) {
    uint64_t val = 0x123456789ABCDEF0ULL;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        // Dependent ALU chain
        asm volatile(
            "add $1, %0\n\t"
            "xor $3, %0\n\t"
            "add $5, %0\n\t"
            "xor $7, %0\n\t"
            : "+r"(val)
            :
            : "cc"
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    volatile uint64_t sink = val;
    (void)sink;
    std::chrono::duration<double, std::milli> elapsed = end - start;
    *duration_ms = elapsed.count();
}

int main() {
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    std::cout << "=== AMD Zen 2 SMT Core Resource Contention Benchmark ===" << std::endl;

    const size_t fma_iters = 100000000; // 100M iters * 256 FLOP = 25.6 GFLOP
    const size_t int_iters = 500000000; // 500M iters * 4 ALU ops = 2B ALU ops

    // 1. Single Thread Peak FMA Baseline (Core 0, Logical Thread 0)
    std::cout << "\n[1] Peak FMA Baseline (Single Thread, Core 0)..." << std::endl;
    double t_base_ms = 0, gflops_base = 0;
    {
        std::thread t1([&]() {
            pin_thread(1ULL << 0);
            avx2_fma_peak_kernel(fma_iters, &t_base_ms, &gflops_base);
        });
        t1.join();
    }
    std::cout << "  Single Thread Peak FMA: " << t_base_ms << " ms | " << gflops_base << " GFLOPS" << std::endl;

    // 2. Two Threads on Independent Physical Cores (Core 0 & Core 1)
    std::cout << "\n[2] Peak FMA on Independent Physical Cores (Core 0 & Core 1)..." << std::endl;
    double t_indep_0_ms = 0, gflops_indep_0 = 0;
    double t_indep_1_ms = 0, gflops_indep_1 = 0;
    {
        std::thread t0([&]() {
            pin_thread(1ULL << 0);
            avx2_fma_peak_kernel(fma_iters, &t_indep_0_ms, &gflops_indep_0);
        });
        std::thread t1([&]() {
            pin_thread(1ULL << 2);
            avx2_fma_peak_kernel(fma_iters, &t_indep_1_ms, &gflops_indep_1);
        });
        t0.join();
        t1.join();
    }
    std::cout << "  Core 0: " << t_indep_0_ms << " ms | " << gflops_indep_0 << " GFLOPS" << std::endl;
    std::cout << "  Core 1: " << t_indep_1_ms << " ms | " << gflops_indep_1 << " GFLOPS" << std::endl;
    double agg_gflops_indep = gflops_indep_0 + gflops_indep_1;
    std::cout << "  Aggregate Independent Throughput: " << agg_gflops_indep << " GFLOPS" << std::endl;

    // 3. Two Threads on SMT Siblings (Core 0, Logical Thread 0 & Thread 1)
    std::cout << "\n[3] Peak FMA on SMT Siblings of Same Physical Core (Core 0, Threads 0 & 1)..." << std::endl;
    double t_smt_0_ms = 0, gflops_smt_0 = 0;
    double t_smt_1_ms = 0, gflops_smt_1 = 0;
    {
        std::thread t0([&]() {
            pin_thread(1ULL << 0);
            avx2_fma_peak_kernel(fma_iters, &t_smt_0_ms, &gflops_smt_0);
        });
        std::thread t1([&]() {
            pin_thread(1ULL << 1);
            avx2_fma_peak_kernel(fma_iters, &t_smt_1_ms, &gflops_smt_1);
        });
        t0.join();
        t1.join();
    }
    std::cout << "  SMT Thread 0: " << t_smt_0_ms << " ms | " << gflops_smt_0 << " GFLOPS" << std::endl;
    std::cout << "  SMT Thread 1: " << t_smt_1_ms << " ms | " << gflops_smt_1 << " GFLOPS" << std::endl;
    double agg_gflops_smt = gflops_smt_0 + gflops_smt_1;
    std::cout << "  Aggregate SMT Throughput: " << agg_gflops_smt << " GFLOPS" << std::endl;
    double smt_fma_slowdown = t_smt_0_ms / t_base_ms;
    std::cout << "  Per-Thread SMT FMA Slowdown: " << smt_fma_slowdown << "x (" << ((smt_fma_slowdown - 1.0) * 100.0) << "% penalty)" << std::endl;
    std::cout << "  Core Utilization vs Single Thread Peak: " << (agg_gflops_smt / gflops_base) * 100.0 << "%" << std::endl;

    // 4. Integer ALU Contention
    std::cout << "\n[4] Integer ALU SMT Contention..." << std::endl;
    double t_int_base_ms = 0;
    {
        std::thread t0([&]() {
            pin_thread(1ULL << 0);
            int_alu_dependent_kernel(int_iters, &t_int_base_ms);
        });
        t0.join();
    }
    std::cout << "  Single Thread Integer ALU: " << t_int_base_ms << " ms" << std::endl;

    double t_int_smt0_ms = 0, t_int_smt1_ms = 0;
    {
        std::thread t0([&]() {
            pin_thread(1ULL << 0);
            int_alu_dependent_kernel(int_iters, &t_int_smt0_ms);
        });
        std::thread t1([&]() {
            pin_thread(1ULL << 1);
            int_alu_dependent_kernel(int_iters, &t_int_smt1_ms);
        });
        t0.join();
        t1.join();
    }
    std::cout << "  SMT Thread 0 Integer ALU: " << t_int_smt0_ms << " ms" << std::endl;
    std::cout << "  SMT Thread 1 Integer ALU: " << t_int_smt1_ms << " ms" << std::endl;
    double int_smt_slowdown = t_int_smt0_ms / t_int_base_ms;
    std::cout << "  Per-Thread SMT Integer ALU Slowdown: " << int_smt_slowdown << "x (" << ((int_smt_slowdown - 1.0) * 100.0) << "% penalty)" << std::endl;

    // Save JSON
    std::ofstream json("research/raw/smt_contention_results.json");
    json << "{\n";
    json << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n";
    json << "  \"benchmark\": \"Zen 2 SMT Core Execution Unit Contention & Port Sharing\",\n";
    json << "  \"results\": {\n";
    json << "    \"single_thread_peak_fma_gflops\": " << gflops_base << ",\n";
    json << "    \"independent_cores_fma_aggregate_gflops\": " << agg_gflops_indep << ",\n";
    json << "    \"smt_siblings_fma_per_thread_gflops\": " << gflops_smt_0 << ",\n";
    json << "    \"smt_siblings_fma_aggregate_gflops\": " << agg_gflops_smt << ",\n";
    json << "    \"smt_fma_per_thread_slowdown_ratio\": " << smt_fma_slowdown << ",\n";
    json << "    \"single_thread_int_alu_ms\": " << t_int_base_ms << ",\n";
    json << "    \"smt_siblings_int_alu_ms\": " << t_int_smt0_ms << ",\n";
    json << "    \"smt_int_alu_per_thread_slowdown_ratio\": " << int_smt_slowdown << "\n";
    json << "  }\n";
    json << "}\n";
    json.close();

    std::cout << "\nResults successfully written to research/raw/smt_contention_results.json" << std::endl;
    return 0;
}
