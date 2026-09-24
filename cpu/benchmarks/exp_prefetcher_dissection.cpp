#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstdint>
#include <windows.h>
#include <x86intrin.h>

// Corner C8: Zen 2 Hardware Data Prefetcher Dissection Microbenchmark
// AMD Ryzen 7 4800HS (Zen 2 Renoir)

#define BUFFER_SIZE (32 * 1024 * 1024) // 32 MB (Exceeds 4MB L3 per CCX)

// Benchmark a linear or strided stream read
__attribute__((noinline))
uint64_t benchmark_strided_read(const uint8_t* buffer, size_t size, int64_t stride_bytes) {
    uint64_t sum = 0;
    size_t num_accesses = 0;

    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    if (stride_bytes > 0) {
        for (size_t offset = 0; offset < size; offset += stride_bytes) {
            sum += *reinterpret_cast<const uint64_t*>(buffer + offset);
            num_accesses++;
        }
    } else {
        int64_t step = -stride_bytes;
        for (int64_t offset = size - 64; offset >= 0; offset -= step) {
            sum += *reinterpret_cast<const uint64_t*>(buffer + offset);
            num_accesses++;
        }
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();

    asm volatile("" : "+r"(sum));
    return (t1 - t0) / (num_accesses ? num_accesses : 1);
}

// Benchmark random pointer chasing (Zero Prefetching baseline)
__attribute__((noinline))
uint64_t benchmark_random_chase(void** chain_head, size_t steps) {
    void** p = chain_head;

    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();

    for (size_t i = 0; i < steps; ++i) {
        p = (void**)*p;
    }

    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();

    asm volatile("" : "+r"(p));
    return (t1 - t0) / steps;
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << 2)); // Core 2
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    std::vector<uint8_t> buffer(BUFFER_SIZE, 0xAA);

    // Build random pointer chain for baseline
    const size_t LINE_SIZE = 64;
    const size_t NUM_LINES = BUFFER_SIZE / LINE_SIZE;
    std::vector<size_t> indices(NUM_LINES);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937_64 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    void** base = reinterpret_cast<void**>(buffer.data());
    for (size_t i = 0; i < NUM_LINES - 1; ++i) {
        size_t curr_idx = indices[i] * (LINE_SIZE / sizeof(void*));
        size_t next_idx = indices[i + 1] * (LINE_SIZE / sizeof(void*));
        base[curr_idx] = &base[next_idx];
    }
    size_t last_idx = indices[NUM_LINES - 1] * (LINE_SIZE / sizeof(void*));
    base[last_idx] = &base[indices[0] * (LINE_SIZE / sizeof(void*))];
    void** chain_head = &base[indices[0] * (LINE_SIZE / sizeof(void*))];

    std::cout << "Starting Zen 2 Hardware Data Prefetcher Dissection..." << std::endl;

    // 1. Measure pure random pointer chase latency (no prefetcher help)
    uint64_t random_cpa = benchmark_random_chase(chain_head, 500000);
    std::cout << "Random Pointer Chase (No Prefetching): " << random_cpa << " cycles / access" << std::endl;

    // Strides to test: +64, +128, +256, +512, +1024, +2048, +4096, -64, -128, -256, -512
    struct StrideTest {
        std::string label;
        int64_t stride;
    };

    std::vector<StrideTest> tests = {
        {"+64 B (Next-Line Forward)", 64},
        {"+128 B (Forward Stride 2)", 128},
        {"+256 B (Forward Stride 4)", 256},
        {"+512 B (Forward Stride 8)", 512},
        {"+1024 B (Forward Stride 16)", 1024},
        {"+2048 B (Forward Stride 32)", 2048},
        {"+4096 B (Page Stride)", 4096},
        {"-64 B (Backward Next-Line)", -64},
        {"-128 B (Backward Stride 2)", -128},
        {"-256 B (Backward Stride 4)", -256},
        {"-512 B (Backward Stride 8)", -512}
    };

    std::ofstream out("research/raw/prefetcher_dissection_results.json");
    out << "{\n  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n"
        << "  \"benchmark\": \"Hardware Data Prefetcher Dissection\",\n"
        << "  \"random_dram_latency_cycles\": " << random_cpa << ",\n"
        << "  \"results\": [\n";

    for (size_t i = 0; i < tests.size(); ++i) {
        // Run multiple repetitions to ensure prefetcher is fully trained
        benchmark_strided_read(buffer.data(), BUFFER_SIZE, tests[i].stride); // Warmup
        uint64_t cpa = benchmark_strided_read(buffer.data(), BUFFER_SIZE, tests[i].stride);

        double speedup = (double)random_cpa / (double)cpa;
        std::cout << tests[i].label << ": " << cpa << " cycles / access (Speedup: " << speedup << "x)" << std::endl;

        out << "    {\"label\": \"" << tests[i].label << "\", \"stride_bytes\": " << tests[i].stride
            << ", \"cycles_per_access\": " << cpa << ", \"prefetcher_speedup\": " << speedup << "}"
            << (i + 1 < tests.size() ? ",\n" : "\n");
    }

    out << "  ]\n}\n";
    out.close();

    std::cout << "Results saved to research/raw/prefetcher_dissection_results.json" << std::endl;
    return 0;
}
