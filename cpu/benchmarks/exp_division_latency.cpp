#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <windows.h>
#include <x86intrin.h>
#include <immintrin.h>

// Corner C7: Non-Pipelined Radix Division Microbenchmark
// AMD Ryzen 7 4800HS (Zen 2 Renoir)

#define DIV_ITERS 10000000

// Dependent 32-bit idiv: each division uses the quotient as the new dividend
__attribute__((noinline))
uint64_t bench_idiv32_dependent(uint32_t start_val, uint32_t divisor) {
    uint32_t dividend = start_val;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < DIV_ITERS; ++i) {
        // idiv r32 divides edx:eax by reg, quotient in eax, remainder in edx
        asm volatile(
            "xor %%edx, %%edx\n\t"
            "mov %1, %%eax\n\t"
            "divl %2\n\t"
            "add $1, %%eax\n\t"
            "mov %%eax, %0\n\t"
            : "=r"(dividend)
            : "r"(dividend), "r"(divisor)
            : "eax", "edx", "cc"
        );
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    return t1 - t0;
}

// Dependent 64-bit idiv
__attribute__((noinline))
uint64_t bench_idiv64_dependent(uint64_t start_val, uint64_t divisor) {
    uint64_t dividend = start_val;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < DIV_ITERS; ++i) {
        asm volatile(
            "xor %%rdx, %%rdx\n\t"
            "mov %1, %%rax\n\t"
            "divq %2\n\t"
            "add $1, %%rax\n\t"
            "mov %%rax, %0\n\t"
            : "=r"(dividend)
            : "r"(dividend), "r"(divisor)
            : "rax", "rdx", "cc"
        );
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    return t1 - t0;
}

// Independent 32-bit idiv (Throughput / non-pipelined issue rate)
__attribute__((noinline))
uint64_t bench_idiv32_throughput(uint32_t divisor) {
    uint32_t d0 = 1000000, d1 = 2000000, d2 = 3000000, d3 = 4000000;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < DIV_ITERS / 4; ++i) {
        asm volatile(
            "xor %%edx, %%edx\n\t"
            "mov %0, %%eax\n\t"
            "divl %4\n\t"
            "xor %%edx, %%edx\n\t"
            "mov %1, %%eax\n\t"
            "divl %4\n\t"
            "xor %%edx, %%edx\n\t"
            "mov %2, %%eax\n\t"
            "divl %4\n\t"
            "xor %%edx, %%edx\n\t"
            "mov %3, %%eax\n\t"
            "divl %4\n\t"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(divisor)
            : "eax", "edx", "cc"
        );
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    return t1 - t0;
}

// Dependent AVX2 256-bit Vector Division (_mm256_div_ps)
__attribute__((noinline))
uint64_t bench_avx2_div_dependent() {
    __m256 v = _mm256_set1_ps(1.00001f);
    __m256 d = _mm256_set1_ps(1.0000001f);

    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < DIV_ITERS; ++i) {
        v = _mm256_div_ps(v, d);
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();

    float res[8];
    _mm256_storeu_ps(res, v);
    asm volatile("" : "+m"(res));
    return t1 - t0;
}

// Independent AVX2 256-bit Vector Division Throughput (4 independent accumulators)
__attribute__((noinline))
uint64_t bench_avx2_div_throughput() {
    __m256 v0 = _mm256_set1_ps(1.00001f);
    __m256 v1 = _mm256_set1_ps(2.00001f);
    __m256 v2 = _mm256_set1_ps(3.00001f);
    __m256 v3 = _mm256_set1_ps(4.00001f);
    __m256 d  = _mm256_set1_ps(1.0000001f);

    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < DIV_ITERS / 4; ++i) {
        v0 = _mm256_div_ps(v0, d);
        v1 = _mm256_div_ps(v1, d);
        v2 = _mm256_div_ps(v2, d);
        v3 = _mm256_div_ps(v3, d);
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();

    float res[8];
    _mm256_storeu_ps(res, _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)));
    asm volatile("" : "+m"(res));
    return t1 - t0;
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << 2)); // Core 2
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    std::cout << "Starting Zen 2 Radix Division Microbenchmark..." << std::endl;

    // Test operand distributions
    uint64_t c_idiv32_small = bench_idiv32_dependent(0x000000FF, 3);
    uint64_t c_idiv32_large = bench_idiv32_dependent(0xFFFFFFFF, 3);
    uint64_t c_idiv64_large = bench_idiv64_dependent(0xFFFFFFFFFFFFFFFFULL, 3ULL);
    uint64_t c_idiv32_tp    = bench_idiv32_throughput(7);
    uint64_t c_avx_div_dep  = bench_avx2_div_dependent();
    uint64_t c_avx_div_tp   = bench_avx2_div_throughput();

    double lat_idiv32_small = (double)c_idiv32_small / (double)DIV_ITERS;
    double lat_idiv32_large = (double)c_idiv32_large / (double)DIV_ITERS;
    double lat_idiv64_large = (double)c_idiv64_large / (double)DIV_ITERS;
    double tp_idiv32        = (double)c_idiv32_tp / (double)DIV_ITERS;
    double lat_avx_div      = (double)c_avx_div_dep / (double)DIV_ITERS;
    double tp_avx_div       = (double)c_avx_div_tp / (double)DIV_ITERS;

    std::cout << "32-bit idiv latency (small dividend): " << lat_idiv32_small << " cycles" << std::endl;
    std::cout << "32-bit idiv latency (large dividend): " << lat_idiv32_large << " cycles" << std::endl;
    std::cout << "64-bit idiv latency (64-bit full):     " << lat_idiv64_large << " cycles" << std::endl;
    std::cout << "32-bit idiv reciprocal throughput:    " << tp_idiv32 << " cycles/div" << std::endl;
    std::cout << "AVX2 256b vector div latency:         " << lat_avx_div << " cycles" << std::endl;
    std::cout << "AVX2 256b vector div throughput:      " << tp_avx_div << " cycles/vector_div" << std::endl;

    std::ofstream out("research/raw/division_latency_results.json");
    out << "{\n"
        << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n"
        << "  \"benchmark\": \"Non-Pipelined Radix Division Latency & Throughput\",\n"
        << "  \"iterations\": " << DIV_ITERS << ",\n"
        << "  \"results\": {\n"
        << "    \"idiv32_latency_small_cycles\": " << lat_idiv32_small << ",\n"
        << "    \"idiv32_latency_large_cycles\": " << lat_idiv32_large << ",\n"
        << "    \"idiv64_latency_cycles\": " << lat_idiv64_large << ",\n"
        << "    \"idiv32_reciprocal_throughput_cycles\": " << tp_idiv32 << ",\n"
        << "    \"avx2_div_ps_latency_cycles\": " << lat_avx_div << ",\n"
        << "    \"avx2_div_ps_reciprocal_throughput_cycles\": " << tp_avx_div << "\n"
        << "  }\n"
        << "}\n";
    out.close();

    std::cout << "Results saved to research/raw/division_latency_results.json" << std::endl;
    return 0;
}
