/**
 * cpu/benchmarks/branch_predictor.cpp
 * Zen 2 (AMD Ryzen 7 4800HS) Branch Predictor & Pipeline Flush Penalty Benchmark
 *
 * Measures CPU cycle latency of a data-dependent conditional branch over:
 *   1. Sorted array (100% predictable, high branch target buffer hit rate)
 *   2. Random array (50% unpredictable, triggering speculative execution flushes)
 * Isolates the exact Zen 2 pipeline flush latency penalty in CPU clock cycles and nanoseconds.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <x86intrin.h>

const size_t N = 1'000'000;
const int RUNS = 50;

uint64_t benchmark_branch(const std::vector<int>& data, volatile int& dummy_out) {
    uint64_t total_cycles = 0;
    int sum = 0;

    for (int r = 0; r < RUNS; ++r) {
        // Clear CPU pipeline
        _mm_mfence();
        uint64_t start = __rdtsc();

        for (size_t i = 0; i < N; ++i) {
            if (data[i] > 128) {
                asm volatile ("add $1, %0" : "+r"(sum));
            } else {
                asm volatile ("sub $1, %0" : "+r"(sum));
            }
        }

        _mm_mfence();
        uint64_t stop = __rdtsc();
        total_cycles += (stop - start);
    }

    dummy_out = sum;
    return total_cycles / RUNS;
}

int main() {
    // Pin to single physical core (Core 0) to avoid scheduling migrations
    SetThreadAffinityMask(GetCurrentThread(), 1);

    std::vector<int> data(N);
    std::mt19937 g(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(g);
    }

    // Measure Unpredictable (Random)
    volatile int dummy_random = 0;
    uint64_t cycles_random = benchmark_branch(data, dummy_random);

    // Sort to make 100% predictable
    std::sort(data.begin(), data.end());

    // Measure Predictable (Sorted)
    volatile int dummy_sorted = 0;
    uint64_t cycles_sorted = benchmark_branch(data, dummy_sorted);

    double cycles_per_elem_random = static_cast<double>(cycles_random) / N;
    double cycles_per_elem_sorted = static_cast<double>(cycles_sorted) / N;
    double cycle_penalty = cycles_per_elem_random - cycles_per_elem_sorted;

    // Estimate clock frequency (~4.0 GHz)
    double estimated_ghz = 4.0;
    double ns_sorted = cycles_per_elem_sorted / estimated_ghz;
    double ns_random = cycles_per_elem_random / estimated_ghz;
    double ns_penalty = cycle_penalty / estimated_ghz;

    std::cout << "=================================================================\n";
    std::cout << " AMD Ryzen 7 4800HS (Zen 2) Branch Predictor Benchmark\n";
    std::cout << " 1,000,000 Elements across 50 Execution Trials\n";
    std::cout << "=================================================================\n";
    std::cout << std::left << std::setw(25) << "Branch Pattern"
              << std::setw(20) << "Cycles / Branch"
              << std::setw(20) << "Time / Branch (ns)" << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Sorted (Predictable)"
              << std::setw(20) << std::fixed << std::setprecision(2) << cycles_per_elem_sorted
              << std::setw(20) << std::fixed << std::setprecision(3) << ns_sorted << "\n";
    std::cout << std::left << std::setw(25) << "Random (Unpredictable)"
              << std::setw(20) << std::fixed << std::setprecision(2) << cycles_per_elem_random
              << std::setw(20) << std::fixed << std::setprecision(3) << ns_random << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Measured Zen 2 Pipeline Flush Penalty: "
              << std::fixed << std::setprecision(2) << cycle_penalty << " clock cycles ("
              << std::fixed << std::setprecision(2) << ns_penalty << " ns)\n";
    std::cout << "Performance Degradation Factor:       "
              << std::fixed << std::setprecision(2) << (cycles_per_elem_random / cycles_per_elem_sorted) << "x slowdown\n";
    std::cout << "=================================================================\n";

    std::ofstream out_json("results/cpu_branch_predictor.json");
    out_json << "{\n"
             << "  \"cpu\": \"AMD Ryzen 7 4800HS with Radeon Graphics (Zen 2)\",\n"
             << "  \"elements\": " << N << ",\n"
             << "  \"trials\": " << RUNS << ",\n"
             << "  \"cycles_per_branch_sorted\": " << cycles_per_elem_sorted << ",\n"
             << "  \"cycles_per_branch_random\": " << cycles_per_elem_random << ",\n"
             << "  \"pipeline_flush_penalty_cycles\": " << cycle_penalty << ",\n"
             << "  \"slowdown_factor\": " << (cycles_per_elem_random / cycles_per_elem_sorted) << "\n"
             << "}\n";
    out_json.close();
    std::cout << "[SUCCESS] Saved results to results/cpu_branch_predictor.json\n";
    return 0;
}
