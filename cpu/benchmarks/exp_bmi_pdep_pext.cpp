#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <windows.h>
#include <x86intrin.h>
#include <immintrin.h>

// Corner C9: Zen 2 Bit Manipulation & Microcoded PDEP/PEXT Microbenchmark
// AMD Ryzen 7 4800HS (Zen 2 Renoir)

#define ITERS 10000000

// Dependent TZCNT
__attribute__((noinline))
uint64_t bench_tzcnt_dependent() {
    uint64_t val = 0x8000000000000000ULL;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        asm volatile("tzcntq %1, %0" : "=r"(val) : "r"(val + 1));
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Dependent POPCNT
__attribute__((noinline))
uint64_t bench_popcnt_dependent() {
    uint64_t val = 0x123456789ABCDEF0ULL;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        asm volatile("popcntq %1, %0" : "=r"(val) : "r"(val + 1));
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Dependent Legacy BSF
__attribute__((noinline))
uint64_t bench_bsf_dependent() {
    uint64_t val = 0x8000000000000000ULL;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        uint64_t res;
        asm volatile("bsfq %1, %0" : "=r"(res) : "r"(val | 1ULL));
        val = res;
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Dependent PDEP (Parallel Bits Deposit - Zen 2 Microcode)
__attribute__((noinline))
uint64_t bench_pdep_dependent() {
    uint64_t val = 0x5555555555555555ULL;
    uint64_t mask = 0xAAAAAAAAAAAAAAAAULL;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        val = _pdep_u64(val, mask);
        val ^= 0x1ULL; // Ensure changing bits
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Dependent PEXT (Parallel Bits Extract - Zen 2 Microcode)
__attribute__((noinline))
uint64_t bench_pext_dependent() {
    uint64_t val = 0x5555555555555555ULL;
    uint64_t mask = 0xAAAAAAAAAAAAAAAAULL;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        val = _pext_u64(val, mask);
        val += 0x1ULL;
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << 2)); // Core 2
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    std::cout << "Starting Zen 2 Bit Manipulation & Microcoded PDEP/PEXT Microbenchmark..." << std::endl;

    uint64_t c_tzcnt = bench_tzcnt_dependent();
    uint64_t c_popcnt = bench_popcnt_dependent();
    uint64_t c_bsf = bench_bsf_dependent();
    uint64_t c_pdep = bench_pdep_dependent();
    uint64_t c_pext = bench_pext_dependent();

    double lat_tzcnt = (double)c_tzcnt / (double)ITERS;
    double lat_popcnt = (double)c_popcnt / (double)ITERS;
    double lat_bsf = (double)c_bsf / (double)ITERS;
    double lat_pdep = (double)c_pdep / (double)ITERS;
    double lat_pext = (double)c_pext / (double)ITERS;

    std::cout << "TZCNT Latency: " << lat_tzcnt << " cycles" << std::endl;
    std::cout << "POPCNT Latency: " << lat_popcnt << " cycles" << std::endl;
    std::cout << "Legacy BSF Latency: " << lat_bsf << " cycles" << std::endl;
    std::cout << "PDEP Latency (Microcoded): " << lat_pdep << " cycles (Slowdown: " << (lat_pdep / lat_tzcnt) << "x)" << std::endl;
    std::cout << "PEXT Latency (Microcoded): " << lat_pext << " cycles (Slowdown: " << (lat_pext / lat_tzcnt) << "x)" << std::endl;

    std::ofstream out("research/raw/bmi_pdep_pext_results.json");
    out << "{\n"
        << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n"
        << "  \"benchmark\": \"Bit Manipulation & Microcoded PDEP/PEXT\",\n"
        << "  \"iterations\": " << ITERS << ",\n"
        << "  \"results\": {\n"
        << "    \"tzcnt_latency_cycles\": " << lat_tzcnt << ",\n"
        << "    \"popcnt_latency_cycles\": " << lat_popcnt << ",\n"
        << "    \"bsf_legacy_latency_cycles\": " << lat_bsf << ",\n"
        << "    \"pdep_microcode_latency_cycles\": " << lat_pdep << ",\n"
        << "    \"pext_microcode_latency_cycles\": " << lat_pext << ",\n"
        << "    \"pdep_slowdown_vs_tzcnt\": " << (lat_pdep / lat_tzcnt) << "\n"
        << "  }\n"
        << "}\n";
    out.close();

    std::cout << "Results saved to research/raw/bmi_pdep_pext_results.json" << std::endl;
    return 0;
}
