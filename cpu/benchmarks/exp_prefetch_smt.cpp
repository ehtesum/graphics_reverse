/**
 * cpu/benchmarks/exp_prefetch_smt.cpp
 * Phase 3 Experiment: Zen 2 Hardware Prefetcher & SMT Sibling Contention
 *
 * Measures:
 *   1. Forward vs Backward sequential streaming (testing directional prefetchers)
 *   2. Multi-stream stride prefetching (1 to 8 concurrent streams)
 *   3. SMT sibling core sharing: Physical core isolation vs SMT hyperthread contention
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <thread>
#include <atomic>
#include <windows.h>
#include <x86intrin.h>

const size_t ARRAY_SIZE = 64 * 1024 * 1024; // 64 MB (DRAM resident)
const int REPS = 5;

// Forward streaming read
uint64_t bench_forward(const std::vector<uint32_t>& data) {
    uint64_t sum = 0;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t i = 0; i < data.size(); i += 16) { // Stride 64 bytes (1 cache line)
        sum += data[i];
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    volatile uint64_t sink = sum;
    return stop - start;
}

// Backward streaming read
uint64_t bench_backward(const std::vector<uint32_t>& data) {
    uint64_t sum = 0;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t i = data.size() - 16; (int64_t)i >= 0; i -= 16) {
        sum += data[i];
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    volatile uint64_t sink = sum;
    return stop - start;
}

// Multi-stream prefetcher test
uint64_t bench_multi_stream(const std::vector<uint32_t>& data, int num_streams) {
    size_t chunk = data.size() / num_streams;
    uint64_t sum = 0;
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t offset = 0; offset < chunk; offset += 16) {
        for (int s = 0; s < num_streams; ++s) {
            sum += data[s * chunk + offset];
        }
    }

    _mm_mfence();
    uint64_t stop = __rdtsc();
    volatile uint64_t sink = sum;
    return stop - start;
}

// SMT contention workload (AVX2 FMA compute intensive)
void fma_workload(std::atomic<bool>& start_flag, std::atomic<bool>& done_flag, int iterations) {
    while (!start_flag.load(std::memory_order_relaxed)) {}

    __m256 a = _mm256_set1_ps(1.001f);
    __m256 b = _mm256_set1_ps(1.002f);
    __m256 c = _mm256_set1_ps(0.0f);

    for (int i = 0; i < iterations; ++i) {
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
    }

    volatile float sink = ((float*)&c)[0];
    done_flag.store(true, std::memory_order_relaxed);
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1); // Pin main thread to Core 0

    std::vector<uint32_t> data(ARRAY_SIZE / 4, 1);

    std::cout << "=================================================================\n";
    std::cout << " Experiment EXP-CPU-056: Zen 2 Prefetcher & SMT Contention\n";
    std::cout << " 64 MB DRAM Array, Core 0 (Physical) vs Core 1 (SMT Sibling)\n";
    std::cout << "=================================================================\n";

    // 1. Forward vs Backward Streaming
    uint64_t c_fwd = bench_forward(data);
    uint64_t c_bwd = bench_backward(data);
    size_t cache_lines = (ARRAY_SIZE / 64);
    double cyc_per_line_fwd = (double)c_fwd / cache_lines;
    double cyc_per_line_bwd = (double)c_bwd / cache_lines;

    std::cout << "--- Directional Stream Prefetching ---\n";
    std::cout << "Forward Stream (Sequential):  " << std::fixed << std::setprecision(2) << cyc_per_line_fwd << " cycles / cache line\n";
    std::cout << "Backward Stream (Reverse):    " << std::fixed << std::setprecision(2) << cyc_per_line_bwd << " cycles / cache line\n";
    std::cout << "Directional Asymmetry:        " << (cyc_per_line_bwd / cyc_per_line_fwd) << "x\n";

    // 2. Multi-Stream Scaling
    std::cout << "\n--- Multi-Stream Prefetcher Scaling ---\n";
    for (int s : {1, 2, 4, 8, 16}) {
        uint64_t c_s = bench_multi_stream(data, s);
        double cyc_line = (double)c_s / cache_lines;
        std::cout << "Concurrent Streams: " << std::setw(2) << s 
                  << " | Latency: " << std::setw(6) << std::fixed << std::setprecision(2) << cyc_line 
                  << " cycles / line\n";
    }

    // 3. SMT Sibling Contention (1T Physical Core vs 2T on same Physical Core SMT siblings)
    std::cout << "\n--- SMT Sibling Resource Contention ---\n";
    const int FMA_ITERS = 50'000'000;

    // Solo run on Core 0
    std::atomic<bool> start_solo(true);
    std::atomic<bool> done_solo(false);
    uint64_t t0 = __rdtsc();
    fma_workload(start_solo, done_solo, FMA_ITERS);
    uint64_t t1 = __rdtsc();
    uint64_t solo_cycles = t1 - t0;

    // Dual run: Thread 1 on Logical Core 0 (mask 0x1), Thread 2 on Logical Core 1 (mask 0x2) - SAME PHYSICAL CORE
    std::atomic<bool> start_dual(false);
    std::atomic<bool> done_t1(false);
    std::atomic<bool> done_t2(false);

    std::thread t_sibling([&]() {
        SetThreadAffinityMask(GetCurrentThread(), 0x2); // SMT sibling mask for Core 0 (Logical 1)
        fma_workload(start_dual, done_t2, FMA_ITERS);
    });

    _mm_mfence();
    t0 = __rdtsc();
    start_dual.store(true, std::memory_order_relaxed);
    fma_workload(start_dual, done_t1, FMA_ITERS);
    t_sibling.join();
    _mm_mfence();
    t1 = __rdtsc();
    uint64_t dual_cycles = t1 - t0;

    std::cout << "Solo Physical Core (1T):            " << (solo_cycles / 1e6) << " M cycles\n";
    std::cout << "Dual SMT Sibling Contention (2T):   " << (dual_cycles / 1e6) << " M cycles\n";
    std::cout << "SMT Execution Throughput Penalty:   " << std::fixed << std::setprecision(2) 
              << ((double)dual_cycles / solo_cycles) << "x slowdown per thread\n";
    std::cout << "Aggregated SMT Speedup:             " << std::fixed << std::setprecision(2) 
              << (2.0 / ((double)dual_cycles / solo_cycles)) << "x aggregate throughput\n";
    std::cout << "=================================================================\n";

    return 0;
}
