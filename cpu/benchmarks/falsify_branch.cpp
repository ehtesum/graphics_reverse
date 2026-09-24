/**
 * cpu/benchmarks/falsify_branch.cpp
 * Falsification Test 3: Zen 2 In-Register Branch Misprediction (Zero-Memory)
 *
 * Compares branch prediction penalties using:
 *   1. Pure in-register Xorshift32 bit generator (0 memory loads)
 *   2. Sequential predictable patterns (all 1s, alternating 1/0)
 *   3. L1-resident array loads
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <windows.h>
#include <x86intrin.h>

const int ITERS = 50'000'000;

// Test 1: In-register predictable (always taken / not taken)
__attribute__((noinline))
uint64_t bench_reg_predictable(int iters, volatile int& out) {
    int sum = 0;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (int i = 0; i < iters; ++i) {
        if ((i & 1) == 0) {
            asm volatile ("add $1, %0" : "+r"(sum));
        } else {
            asm volatile ("add $1, %0" : "+r"(sum));
        }
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    out = sum;
    return stop - start;
}

// Test 2: In-register pseudo-random (Xorshift32, zero memory)
__attribute__((noinline))
uint64_t bench_reg_unpredictable(int iters, volatile int& out) {
    int sum = 0;
    uint32_t state = 2463534242;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (int i = 0; i < iters; ++i) {
        // Xorshift32: 3 ALU ops
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        if (state & 1) {
            asm volatile ("add $1, %0" : "+r"(sum));
        } else {
            asm volatile ("sub $1, %0" : "+r"(sum));
        }
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    out = sum;
    return stop - start;
}

// Test 3: Xorshift32 baseline without branch (ALU overhead baseline)
__attribute__((noinline))
uint64_t bench_reg_xorshift_baseline(int iters, volatile int& out) {
    int sum = 0;
    uint32_t state = 2463534242;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (int i = 0; i < iters; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        asm volatile ("add $1, %0" : "+r"(sum));
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    out = sum;
    return stop - start;
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);

    volatile int dummy = 0;
    // Warmup
    bench_reg_predictable(1'000'000, dummy);

    uint64_t c_pred = bench_reg_predictable(ITERS, dummy);
    uint64_t c_unpred = bench_reg_unpredictable(ITERS, dummy);
    uint64_t c_base = bench_reg_xorshift_baseline(ITERS, dummy);

    double cyc_pred = (double)c_pred / ITERS;
    double cyc_unpred = (double)c_unpred / ITERS;
    double cyc_base = (double)c_base / ITERS;

    // The pure branch misprediction penalty = (unpred - base) or (unpred - pred)
    double net_mispredict_penalty = cyc_unpred - cyc_base;

    std::cout << "=================================================================\n";
    std::cout << " Falsification Test 3: Zen 2 In-Register Zero-Memory Branch Benchmark\n";
    std::cout << " 50,000,000 Iterations pinned to Core 0\n";
    std::cout << "=================================================================\n";
    std::cout << "1. Predictable Branch (Trivial Pattern):      " << std::fixed << std::setprecision(2) << cyc_pred << " cycles / branch\n";
    std::cout << "2. Xorshift32 ALU Baseline (No Branch):        " << std::fixed << std::setprecision(2) << cyc_base << " cycles / iter\n";
    std::cout << "3. Unpredictable Branch + Xorshift32:          " << std::fixed << std::setprecision(2) << cyc_unpred << " cycles / iter\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Net Pipeline Flush Penalty (Unpredict - Base):  " << std::fixed << std::setprecision(2) << net_mispredict_penalty << " cycles\n";
    std::cout << "Historical Whitepaper Penalty (with L3 Array):  10.33 cycles\n";
    std::cout << "=================================================================\n";

    return 0;
}
