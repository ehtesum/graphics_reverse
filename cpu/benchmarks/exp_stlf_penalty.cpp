#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <windows.h>
#include <x86intrin.h>

// Corner C6: Store-to-Load Forwarding (STLF) Microbenchmark
// AMD Ryzen 7 4800HS (Zen 2 Renoir)

#define ITERS 10000000

// Case 1: Independent Load & Store (Baseline L1 throughput)
__attribute__((noinline))
uint64_t bench_independent(volatile uint64_t* buf) {
    uint64_t val = 0;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        buf[0] = i;
        val += buf[2]; // Different cache line offset, no dependency
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Case 2: Exact Match STLF (8-byte Store -> 8-byte Load dependent chain)
__attribute__((noinline))
uint64_t bench_stlf_exact(volatile uint64_t* buf) {
    uint64_t val = 1;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        buf[0] = val + 1;
        val = buf[0]; // Exact same address and size: fast STLF
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Case 3: Sub-register STLF (8-byte Store -> 4-byte Load from same base)
__attribute__((noinline))
uint64_t bench_stlf_subregister(volatile uint64_t* buf) {
    volatile uint32_t* buf32 = reinterpret_cast<volatile uint32_t*>(buf);
    uint32_t val = 1;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        buf[0] = (uint64_t)val + 1;
        val = buf32[0]; // Low 32 bits from 64-bit store
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Case 4: STLF Failure: Partial Overlap (8-byte Store at 0, 8-byte Load at offset 1)
// Load spans partially across the store boundary
__attribute__((noinline))
uint64_t bench_stlf_partial_overlap(volatile uint8_t* raw_buf) {
    volatile uint64_t* store_ptr = reinterpret_cast<volatile uint64_t*>(raw_buf);
    volatile uint64_t* load_ptr = reinterpret_cast<volatile uint64_t*>(raw_buf + 1);
    uint64_t val = 1;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        *store_ptr = val + 1;
        val = *load_ptr; // Partial overlap -> STLF stall & replay
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    asm volatile("" : "+r"(val));
    return t1 - t0;
}

// Case 5: 4K Aliasing (Store to addr, Load from addr + 4096)
__attribute__((noinline))
uint64_t bench_stlf_4k_alias(volatile uint64_t* buf) {
    volatile uint64_t* ptr_a = buf;
    volatile uint64_t* ptr_b = buf + (4096 / sizeof(uint64_t)); // Exactly 4KB apart
    uint64_t val = 1;
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < ITERS; ++i) {
        *ptr_a = val + 1;
        val = *ptr_b; // False 4K alias collision in load/store disambiguation
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

    alignas(4096) uint8_t buffer[16384] = {0};
    volatile uint64_t* buf64 = reinterpret_cast<volatile uint64_t*>(buffer);

    std::cout << "Starting Store-to-Load Forwarding (STLF) Microbenchmark on Zen 2..." << std::endl;

    // Warmup runs
    bench_independent(buf64);
    bench_stlf_exact(buf64);

    uint64_t c_indep = bench_independent(buf64);
    uint64_t c_exact = bench_stlf_exact(buf64);
    uint64_t c_subreg = bench_stlf_subregister(buf64);
    uint64_t c_overlap = bench_stlf_partial_overlap(buffer);
    uint64_t c_alias = bench_stlf_4k_alias(buf64);

    double cyc_indep = (double)c_indep / (double)ITERS;
    double cyc_exact = (double)c_exact / (double)ITERS;
    double cyc_subreg = (double)c_subreg / (double)ITERS;
    double cyc_overlap = (double)c_overlap / (double)ITERS;
    double cyc_alias = (double)c_alias / (double)ITERS;

    std::cout << "Independent Store/Load:     " << cyc_indep << " cycles/op" << std::endl;
    std::cout << "Exact Match STLF (8B->8B):  " << cyc_exact << " cycles/op" << std::endl;
    std::cout << "Sub-register STLF (8B->4B): " << cyc_subreg << " cycles/op" << std::endl;
    std::cout << "STLF Failure (Overlap):     " << cyc_overlap << " cycles/op (Penalty: +" << (cyc_overlap - cyc_exact) << "c)" << std::endl;
    std::cout << "4K Aliasing Collision:      " << cyc_alias << " cycles/op (Penalty: +" << (cyc_alias - cyc_indep) << "c)" << std::endl;

    std::ofstream out("research/raw/stlf_penalty_results.json");
    out << "{\n"
        << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n"
        << "  \"benchmark\": \"Store-to-Load Forwarding (STLF) & Aliasing Penalties\",\n"
        << "  \"iterations\": " << ITERS << ",\n"
        << "  \"results\": {\n"
        << "    \"independent_store_load_cycles\": " << cyc_indep << ",\n"
        << "    \"exact_stlf_cycles\": " << cyc_exact << ",\n"
        << "    \"subregister_stlf_cycles\": " << cyc_subreg << ",\n"
        << "    \"stlf_failure_overlap_cycles\": " << cyc_overlap << ",\n"
        << "    \"stlf_failure_penalty_cycles\": " << (cyc_overlap - cyc_exact) << ",\n"
        << "    \"alias_4k_cycles\": " << cyc_alias << ",\n"
        << "    \"alias_4k_penalty_cycles\": " << (cyc_alias - cyc_indep) << "\n"
        << "  }\n"
        << "}\n";
    out.close();

    std::cout << "Results saved to research/raw/stlf_penalty_results.json" << std::endl;
    return 0;
}
