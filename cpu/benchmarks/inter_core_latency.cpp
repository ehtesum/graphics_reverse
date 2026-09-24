/**
 * cpu/benchmarks/inter_core_latency.cpp
 * Frontier 1: AMD Zen 2 (Ryzen 7 4800HS) Inter-Core & Cross-CCX Infinity Fabric Latency
 *
 * Measures the round-trip ping-pong cache-line transfer latency between all pairs
 * of CPU cores using atomic synchronization (x86 LOCK CMPXCHG / std::atomic).
 *
 * Microarchitectural targets:
 *   1. Intra-Core SMT Sibling Latency (Threads sharing the same physical L1/L2)
 *   2. Intra-CCX Physical Core Latency (Cores within the same 4-core CCX sharing L3)
 *   3. Inter-CCX Physical Core Latency (Cores crossing the AMD Infinity Fabric to the second CCX)
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <x86intrin.h>

const int PING_PONG_ITERS = 100'000;

struct alignas(64) CacheAlignedFlag {
    std::atomic<int> flag{0};
    char pad[64 - sizeof(std::atomic<int>)]; // Avoid false sharing
};

double measure_pair_latency(int core_a, int core_b, double ghz) {
    if (core_a == core_b) return 0.0;

    std::vector<double> samples;
    for (int run = 0; run < 3; ++run) {
        CacheAlignedFlag turn;
        std::atomic<bool> start_signal{false};

        uint64_t total_cycles = 0;

        std::thread thread_b([&]() {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core_b);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

            while (!start_signal.load(std::memory_order_acquire)) {
                _mm_pause();
            }

            for (int i = 0; i < PING_PONG_ITERS; ++i) {
                while (turn.flag.load(std::memory_order_acquire) != 1) {
                    _mm_pause();
                }
                turn.flag.store(0, std::memory_order_release);
            }
        });

        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core_a);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

        turn.flag.store(0, std::memory_order_relaxed);
        start_signal.store(true, std::memory_order_release);

        _mm_mfence();
        uint64_t tsc_start = __rdtsc();

        for (int i = 0; i < PING_PONG_ITERS; ++i) {
            turn.flag.store(1, std::memory_order_release);
            while (turn.flag.load(std::memory_order_acquire) != 0) {
                _mm_pause();
            }
        }

        _mm_mfence();
        uint64_t tsc_end = __rdtsc();

        thread_b.join();

        total_cycles = tsc_end - tsc_start;
        double cycles_per_hop = (double)total_cycles / (2.0 * PING_PONG_ITERS);
        samples.push_back(cycles_per_hop);
    }

    std::sort(samples.begin(), samples.end());
    return samples[0]; // Min latency represents clean hardware interconnect without OS preemption
}

double get_cpu_ghz() {
    LARGE_INTEGER freq, qpc0, qpc1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&qpc0);
    uint64_t t0 = __rdtsc();
    Sleep(50);
    QueryPerformanceCounter(&qpc1);
    uint64_t t1 = __rdtsc();

    double sec = (double)(qpc1.QuadPart - qpc0.QuadPart) / (double)freq.QuadPart;
    return ((double)(t1 - t0) / sec) / 1e9;
}

int main() {
    std::cout << "===========================================================================\n";
    std::cout << " Frontier 1: AMD Zen 2 (Ryzen 7 4800HS) Inter-Core Latency Matrix\n";
    std::cout << " 8 Physical Cores (CCX 0: Cores 0-3 | CCX 1: Cores 4-7)\n";
    std::cout << "===========================================================================\n";

    double ghz = get_cpu_ghz();
    std::cout << "[*] Measured CPU Clock Frequency: " << std::fixed << std::setprecision(2) << ghz << " GHz\n";

    // 8 Physical cores mapped to logical IDs: 0, 2, 4, 6 (CCX 0) and 8, 10, 12, 14 (CCX 1)
    std::vector<int> phys_logical_ids = {0, 2, 4, 6, 8, 10, 12, 14};
    int N = phys_logical_ids.size();

    std::vector<std::vector<double>> latency_cycles(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> latency_ns(N, std::vector<double>(N, 0.0));

    std::cout << "\n[*] Measuring 8 Physical Core Interconnect Latency Matrix (in Nanoseconds)...\n";
    std::cout << "    [Cores 0-3: CCX 0 (Intra-CCX)] | [Cores 4-7: CCX 1 (Cross-Infinity Fabric)]\n\n";

    std::cout << "        ";
    for (int j = 0; j < N; ++j) {
        std::cout << "Core " << j << "   ";
    }
    std::cout << "\n-------";
    for (int j = 0; j < N; ++j) {
        std::cout << "---------";
    }
    std::cout << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout << "Core " << i << " |  ";
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                std::cout << std::setw(6) << std::fixed << std::setprecision(1) << 0.0 << "   ";
                continue;
            }
            double cyc = measure_pair_latency(phys_logical_ids[i], phys_logical_ids[j], ghz);
            double ns = cyc / ghz;
            latency_cycles[i][j] = cyc;
            latency_ns[i][j] = ns;
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << ns << "   ";
        }
        std::cout << "\n";
    }

    // Save JSON results
    std::ofstream json_file("results/cpu_benchmarks/inter_core_latency.json");
    json_file << "{\n";
    json_file << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2)\",\n";
    json_file << "  \"base_ghz\": " << ghz << ",\n";
    json_file << "  \"num_cores_tested\": " << N << ",\n";
    json_file << "  \"matrix_ns\": [\n";
    for (int i = 0; i < N; ++i) {
        json_file << "    [";
        for (int j = 0; j < N; ++j) {
            json_file << std::fixed << std::setprecision(2) << latency_ns[i][j] << (j == N - 1 ? "" : ", ");
        }
        json_file << (i == N - 1 ? "]\n" : "],\n");
    }
    json_file << "  ],\n";
    json_file << "  \"matrix_cycles\": [\n";
    for (int i = 0; i < N; ++i) {
        json_file << "    [";
        for (int j = 0; j < N; ++j) {
            json_file << std::fixed << std::setprecision(1) << latency_cycles[i][j] << (j == N - 1 ? "" : ", ");
        }
        json_file << (i == N - 1 ? "]\n" : "],\n");
    }
    json_file << "  ]\n}\n";
    json_file.close();

    std::cout << "\n===========================================================================\n";
    std::cout << "[SUCCESS] Saved inter-core latency matrix to: results/cpu_benchmarks/inter_core_latency.json\n\n";

    return 0;
}
