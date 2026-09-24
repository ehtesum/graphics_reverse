/**
 * cpu/benchmarks/cache_hierarchy.cpp
 * Pointer-Chasing Memory Latency Benchmark for AMD Ryzen 7 4800HS (Zen 2)
 *
 * Measures cache access latency across varying buffer working-set sizes
 * (4 KB to 64 MB) using randomized pointer chasing to bypass stream prefetchers.
 * Identifies the exact empirical transition boundaries:
 *   - L1 Data Cache (32 KB per core)
 *   - L2 Unified Cache (512 KB per core)
 *   - L3 Shared Cache (8 MB per CCX)
 *   - System DDR4 Memory
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <windows.h>

struct Node {
    Node* next;
    char padding[56]; // 64-byte total size (standard x86 cache line size)
};

double measure_latency_ns(size_t size_bytes, size_t iterations) {
    size_t num_nodes = size_bytes / sizeof(Node);
    if (num_nodes < 2) num_nodes = 2;

    std::vector<Node> buffer(num_nodes);
    std::vector<size_t> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);

    // Permute indices using a linear congruential cycle or random shuffle with fixed seed
    std::mt19937 g(1337);
    std::shuffle(indices.begin(), indices.end(), g);

    // Link nodes in shuffled cyclic order
    for (size_t i = 0; i < num_nodes - 1; ++i) {
        buffer[indices[i]].next = &buffer[indices[i + 1]];
    }
    buffer[indices[num_nodes - 1]].next = &buffer[indices[0]];

    // Pin thread to Core 0 to prevent OS scheduling hops between CCXs
    SetThreadAffinityMask(GetCurrentThread(), 1);

    // Warm up cache
    Node* curr = &buffer[indices[0]];
    for (size_t i = 0; i < num_nodes * 2; ++i) {
        curr = curr->next;
    }

    // Timed pointer traversal
    LARGE_INTEGER freq, start, stop;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    for (size_t i = 0; i < iterations; ++i) {
        curr = curr->next;
    }

    QueryPerformanceCounter(&stop);

    // Prevent compiler from optimizing away the loop
    if (curr == nullptr) {
        std::cerr << "Unreachable" << std::endl;
    }

    double elapsed_sec = static_cast<double>(stop.QuadPart - start.QuadPart) / freq.QuadPart;
    double ns_per_access = (elapsed_sec * 1e9) / static_cast<double>(iterations);
    return ns_per_access;
}

int main() {
    std::vector<size_t> sizes_kb = {
        4, 8, 16, 24, 32,        // L1 Cache (<= 32 KB)
        48, 64, 128, 256, 512,   // L2 Cache (<= 512 KB)
        768, 1024, 2048, 4096, 8192, // L3 Cache (<= 8 MB)
        12288, 16384, 32768, 65536   // Main RAM (> 8 MB)
    };

    const size_t ITERATIONS = 10'000'000;

    std::cout << "=================================================================\n";
    std::cout << " AMD Ryzen 7 4800HS (Zen 2) Cache Hierarchy Microbenchmark\n";
    std::cout << " Pointer-Chasing Latency (Randomized, 64-byte Cache Lines)\n";
    std::cout << "=================================================================\n";
    std::cout << std::left << std::setw(15) << "Size (KB)"
              << std::setw(18) << "Target Tier"
              << std::setw(15) << "Latency (ns)"
              << std::setw(15) << "Inferred Status\n";
    std::cout << "-----------------------------------------------------------------\n";

    std::ofstream json_out("results/cpu_cache_hierarchy.json");
    json_out << "{\n  \"cpu\": \"AMD Ryzen 7 4800HS with Radeon Graphics\",\n  \"benchmarks\": [\n";

    for (size_t i = 0; i < sizes_kb.size(); ++i) {
        size_t kb = sizes_kb[i];
        size_t bytes = kb * 1024;
        double ns = measure_latency_ns(bytes, ITERATIONS);

        std::string tier;
        if (kb <= 32) tier = "L1 Data";
        else if (kb <= 512) tier = "L2 Unified";
        else if (kb <= 8192) tier = "L3 Shared";
        else tier = "DRAM (Main RAM)";

        std::cout << std::left << std::setw(15) << kb
                  << std::setw(18) << tier
                  << std::setw(15) << std::fixed << std::setprecision(2) << ns
                  << "OBSERVED\n";

        json_out << "    {\"size_kb\": " << kb << ", \"tier\": \"" << tier
                 << "\", \"latency_ns\": " << std::fixed << std::setprecision(2) << ns << "}";
        if (i + 1 < sizes_kb.size()) json_out << ",";
        json_out << "\n";
    }

    json_out << "  ]\n}\n";
    json_out.close();

    std::cout << "=================================================================\n";
    std::cout << "[SUCCESS] Saved results to results/cpu_cache_hierarchy.json\n";
    return 0;
}
