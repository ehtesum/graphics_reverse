/**
 * cpu/benchmarks/exp_store_bandwidth.cpp
 * Corner 6 Forensic Experiment: AVX2 Standard Stores vs Non-Temporal Streaming Stores
 * Target: AMD Ryzen 7 4800HS (Zen 2 Renoir, 8C/16T, Dual-Channel DDR4-3200)
 *
 * Microarchitectural Hypotheses:
 *   1. Standard cached stores (_mm256_storeu_si256) trigger Read-For-Ownership (RFO) bus transactions,
 *      fetching the cache line from memory before writing, halving effective write bus efficiency.
 *   2. Non-temporal streaming stores (_mm256_stream_si256) bypass L1/L2/L3 caches and utilize
 *      Write-Combining (WC) buffers, eliminating RFO reads and boosting DRAM write bandwidth.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <immintrin.h>

const size_t TOTAL_BUFFER_BYTES = 256 * 1024 * 1024; // 256 MB
const int NUM_THREADS = 8;
const size_t THREAD_CHUNK_BYTES = TOTAL_BUFFER_BYTES / NUM_THREADS; // 32 MB per thread

const DWORD_PTR CORE_MASKS[8] = {
    0x0001, 0x0004, 0x0010, 0x0040, // CCX 0: Cores 0, 1, 2, 3
    0x0100, 0x0400, 0x1000, 0x4000  // CCX 1: Cores 4, 5, 6, 7
};

std::atomic<bool> g_start{false};
std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_total_bytes_written{0};

void store_worker(int thread_id, char* buffer_start, bool use_streaming) {
    SetThreadAffinityMask(GetCurrentThread(), CORE_MASKS[thread_id % 8]);

    while (!g_start.load(std::memory_order_relaxed)) {
        _mm_pause();
    }

    char* ptr = buffer_start;
    size_t chunk = THREAD_CHUNK_BYTES;
    uint64_t bytes_written = 0;
    __m256i val = _mm256_set1_epi32(0x5A5A5A5A);

    while (!g_stop.load(std::memory_order_relaxed)) {
        if (use_streaming) {
            for (size_t offset = 0; offset < chunk; offset += 128) {
                _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr + offset), val);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr + offset + 32), val);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr + offset + 64), val);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr + offset + 96), val);
            }
            _mm_sfence(); // Flush write-combining buffers
        } else {
            for (size_t offset = 0; offset < chunk; offset += 128) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + offset), val);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + offset + 32), val);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + offset + 64), val);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + offset + 96), val);
            }
        }
        bytes_written += chunk;
    }

    g_total_bytes_written.fetch_add(bytes_written, std::memory_order_relaxed);
}

double run_store_mode(bool use_streaming, double duration_sec) {
    g_start.store(false);
    g_stop.store(false);
    g_total_bytes_written.store(0);

    char* big_buf = static_cast<char*>(VirtualAlloc(NULL, TOTAL_BUFFER_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!big_buf) return 0.0;

    // Touch all pages
    for (size_t i = 0; i < TOTAL_BUFFER_BYTES; i += 4096) {
        big_buf[i] = static_cast<char>(i & 0xFF);
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(store_worker, t, big_buf + t * THREAD_CHUNK_BYTES, use_streaming);
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&t0);
    g_start.store(true, std::memory_order_release);

    Sleep(static_cast<DWORD>(duration_sec * 1000));

    g_stop.store(true, std::memory_order_release);
    QueryPerformanceCounter(&t1);

    for (auto& th : threads) {
        th.join();
    }

    double elapsed = static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(freq.QuadPart);
    uint64_t total_bytes = g_total_bytes_written.load();
    double bw_gb_s = (static_cast<double>(total_bytes) / elapsed) / 1e9;

    VirtualFree(big_buf, 0, MEM_RELEASE);
    return bw_gb_s;
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " Corner 6: Zen 2 AVX2 Standard Stores vs Non-Temporal Streaming Stores Forensic Audit\n";
    std::cout << " Target: AMD Ryzen 7 4800HS (8 Physical Cores, 256 MB DRAM Buffer)\n";
    std::cout << "========================================================================================\n";

    double DURATION = 2.0;

    std::cout << "[*] Benchmarking Standard Cached Stores (_mm256_storeu_si256 with RFO)...\n";
    double bw_standard = run_store_mode(false, DURATION);
    std::cout << "    Standard Store Bandwidth: " << std::fixed << std::setprecision(3) << bw_standard << " GB/s\n";

    std::cout << "\n[*] Benchmarking Non-Temporal Streaming Stores (_mm256_stream_si256 bypass)...\n";
    double bw_streaming = run_store_mode(true, DURATION);
    std::cout << "    Streaming Store Bandwidth: " << std::fixed << std::setprecision(3) << bw_streaming << " GB/s\n";

    double speedup = bw_streaming / bw_standard;
    std::cout << "\n----------------------------------------------------------------------------------------\n";
    std::cout << " Non-Temporal Streaming Store Advantage: " << std::fixed << std::setprecision(2) << speedup << "x write throughput\n";

    std::ofstream json_out("research/raw/store_bandwidth_results.json");
    json_out << "{\n"
             << "  \"standard_store_gb_s\": " << bw_standard << ",\n"
             << "  \"streaming_store_gb_s\": " << bw_streaming << ",\n"
             << "  \"speedup\": " << speedup << "\n"
             << "}\n";
    json_out.close();

    std::cout << "[SUCCESS] Store bandwidth telemetry saved to: research/raw/store_bandwidth_results.json\n\n";

    return 0;
}
