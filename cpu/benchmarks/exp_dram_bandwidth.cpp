/**
 * cpu/benchmarks/exp_dram_bandwidth.cpp
 * Step 5 Discovery: Multi-Core Zen 2 AVX2 DRAM Bandwidth & Latency Benchmark
 * Target: AMD Ryzen 7 4800HS (Renoir APU, 8C/16T, DDR4-3200 dual-channel)
 *
 * Designed to measure CPU memory controller bandwidth alone and concurrently
 * with saturated GPU PCIe 3.0 x8 Direct Memory Access (DMA) traffic.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <immintrin.h>

const size_t TOTAL_BUFFER_BYTES = 256 * 1024 * 1024; // 256 MB
const int NUM_THREADS = 8;
const size_t THREAD_CHUNK_BYTES = TOTAL_BUFFER_BYTES / NUM_THREADS; // 32 MB per thread

// Physical core affinity masks for Ryzen 7 4800HS (8 Physical Cores: 0, 2, 4, 6, 8, 10, 12, 14)
const DWORD_PTR CORE_MASKS[8] = {
    0x0001, 0x0004, 0x0010, 0x0040, // CCX 0: Cores 0, 1, 2, 3
    0x0100, 0x0400, 0x1000, 0x4000  // CCX 1: Cores 4, 5, 6, 7
};

std::atomic<bool> g_start_flag{false};
std::atomic<bool> g_stop_flag{false};
std::atomic<uint64_t> g_total_bytes_read{0};

void worker_thread(int thread_id, char* buffer_start, double duration_sec) {
    SetThreadAffinityMask(GetCurrentThread(), CORE_MASKS[thread_id % 8]);

    // Wait for synchronized start
    while (!g_start_flag.load(std::memory_order_relaxed)) {
        _mm_pause();
    }

    char* ptr = buffer_start;
    size_t chunk = THREAD_CHUNK_BYTES;
    uint64_t bytes_read = 0;
    __m256i accum = _mm256_setzero_si256();

    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        for (size_t offset = 0; offset < chunk; offset += 128) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + offset));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + offset + 32));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + offset + 64));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + offset + 96));
            accum = _mm256_add_epi32(accum, _mm256_add_epi32(v0, v1));
            accum = _mm256_add_epi32(accum, _mm256_add_epi32(v2, v3));
        }
        bytes_read += chunk;
    }

    g_total_bytes_read.fetch_add(bytes_read, std::memory_order_relaxed);

    // Prevent compiler DCE
    volatile int dummy = _mm256_extract_epi32(accum, 0);
    (void)dummy;
}

int main(int argc, char* argv[]) {
    double target_duration_sec = 2.0;
    if (argc > 1) {
        target_duration_sec = std::stod(argv[1]);
    }

    // Allocate 256 MB in DRAM with physical commit
    char* big_buf = static_cast<char*>(VirtualAlloc(NULL, TOTAL_BUFFER_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!big_buf) {
        std::cerr << "VirtualAlloc failed" << std::endl;
        return 1;
    }

    // Touch all pages to force physical DRAM allocation
    for (size_t i = 0; i < TOTAL_BUFFER_BYTES; i += 4096) {
        big_buf[i] = static_cast<char>(i & 0xFF);
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker_thread, t, big_buf + t * THREAD_CHUNK_BYTES, target_duration_sec);
    }

    // High resolution timer
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    // Start benchmark
    QueryPerformanceCounter(&t0);
    g_start_flag.store(true, std::memory_order_release);

    // Sleep for test duration
    Sleep(static_cast<DWORD>(target_duration_sec * 1000));

    // Stop benchmark
    g_stop_flag.store(true, std::memory_order_release);
    QueryPerformanceCounter(&t1);

    for (auto& th : threads) {
        th.join();
    }

    double elapsed_sec = static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(freq.QuadPart);
    uint64_t total_bytes = g_total_bytes_read.load();
    double bandwidth_gb_s = (static_cast<double>(total_bytes) / elapsed_sec) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "ELAPSED_SEC: " << elapsed_sec << "\n";
    std::cout << "TOTAL_GB_READ: " << (static_cast<double>(total_bytes) / 1e9) << "\n";
    std::cout << "CPU_DRAM_BANDWIDTH_GB_S: " << bandwidth_gb_s << "\n";

    VirtualFree(big_buf, 0, MEM_RELEASE);
    return 0;
}
