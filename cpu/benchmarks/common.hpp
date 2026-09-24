#pragma once

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <immintrin.h>
#else
    #include <pthread.h>
    #include <x86intrin.h>
#endif

namespace cpu_bench {

/**
 * Pins the calling thread to a specific logical processor core.
 * Cross-platform: uses SetThreadAffinityMask on Windows and pthread_setaffinity_np on Linux.
 */
inline bool pin_to_core(int core_id = 0) {
#if defined(_WIN32)
    DWORD_PTR mask = (1ULL << core_id);
    DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
    return (prev != 0);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0);
#endif
}

/**
 * Serialized x86-64 cycle counter with full memory fence.
 */
inline uint64_t rdtsc_fence() {
    _mm_mfence();
    return static_cast<uint64_t>(__rdtsc());
}

/**
 * Measures CPU clock frequency in GHz by calibrating RDTSC against a high-resolution reference timer.
 */
inline double measure_cpu_ghz(uint32_t sample_ms = 50) {
#if defined(_WIN32)
    LARGE_INTEGER freq, qpc_start, qpc_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&qpc_start);
    uint64_t tsc_start = __rdtsc();

    Sleep(sample_ms);

    QueryPerformanceCounter(&qpc_end);
    uint64_t tsc_end = __rdtsc();

    double elapsed_sec = static_cast<double>(qpc_end.QuadPart - qpc_start.QuadPart) / static_cast<double>(freq.QuadPart);
    double cycles = static_cast<double>(tsc_end - tsc_start);
    return (cycles / elapsed_sec) / 1e9;
#else
    auto t_start = std::chrono::steady_clock::now();
    uint64_t tsc_start = __rdtsc();

    std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));

    auto t_end = std::chrono::steady_clock::now();
    uint64_t tsc_end = __rdtsc();

    std::chrono::duration<double> elapsed = t_end - t_start;
    double cycles = static_cast<double>(tsc_end - tsc_start);
    return (cycles / elapsed.count()) / 1e9;
#endif
}

} // namespace cpu_bench
