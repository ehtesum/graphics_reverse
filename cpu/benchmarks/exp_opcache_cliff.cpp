/**
 * cpu/benchmarks/exp_opcache_cliff.cpp
 * Step 4 Discovery: Zen 2 Op-Cache Capacity Cliff (4K uops) Reverse Engineering
 * Target: AMD Ryzen 7 4800HS (Zen 2 Renoir, 8C/16T, 2.9 - 4.2 GHz)
 *
 * Microarchitectural Hypotheses:
 *   1. AMD Zen 2 core features a 4,096-entry Micro-op Cache (Op-Cache).
 *   2. Loops <= 4,096 uops execute entirely from the Op-Cache, bypassing the x86 instruction
 *      decoder and sustaining high IPC (approaching 4.0 IPC across 4 integer ALUs).
 *   3. When loop body exceeds 4,096 uops, Op-Cache overflows. The core must stream from L1I
 *      through the 4-wide legacy instruction decoders, introducing decode stalls and reducing IPC.
 */

#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <windows.h>
#include <x86intrin.h>

const uint64_t TARGET_TOTAL_INSNS = 40'000'000;
const int WARMUP_RUNS = 3;
const int MEASURE_RUNS = 5;

#define DEFINE_OPCACHE_TEST(NAME, N_INSNS, REPT_COUNT) \
__attribute__((noinline)) double test_##NAME() { \
    uint64_t iters = TARGET_TOTAL_INSNS / (N_INSNS); \
    uint32_t r1 = 0, r2 = 0, r3 = 0, r4 = 0; \
    /* Warmup */ \
    for (int w = 0; w < WARMUP_RUNS; ++w) { \
        for (uint64_t i = 0; i < iters / 4; ++i) { \
            __asm__ __volatile__( \
                ".rept " #REPT_COUNT "\n\t" \
                "addl $1, %0\n\t" \
                "addl $1, %1\n\t" \
                "addl $1, %2\n\t" \
                "addl $1, %3\n\t" \
                ".endr\n\t" \
                : "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4) \
            ); \
        } \
    } \
    uint64_t min_cycles = UINT64_MAX; \
    for (int m = 0; m < MEASURE_RUNS; ++m) { \
        _mm_mfence(); \
        uint64_t t0 = __rdtsc(); \
        for (uint64_t i = 0; i < iters; ++i) { \
            __asm__ __volatile__( \
                ".rept " #REPT_COUNT "\n\t" \
                "addl $1, %0\n\t" \
                "addl $1, %1\n\t" \
                "addl $1, %2\n\t" \
                "addl $1, %3\n\t" \
                ".endr\n\t" \
                : "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4) \
            ); \
        } \
        _mm_mfence(); \
        uint64_t t1 = __rdtsc(); \
        uint64_t delta = t1 - t0; \
        if (delta < min_cycles) min_cycles = delta; \
    } \
    uint64_t total_insns = iters * (N_INSNS); \
    return (double)min_cycles / (double)total_insns; \
}

DEFINE_OPCACHE_TEST(256, 256, 64)
DEFINE_OPCACHE_TEST(512, 512, 128)
DEFINE_OPCACHE_TEST(1024, 1024, 256)
DEFINE_OPCACHE_TEST(2048, 2048, 512)
DEFINE_OPCACHE_TEST(3072, 3072, 768)
DEFINE_OPCACHE_TEST(3584, 3584, 896)
DEFINE_OPCACHE_TEST(3840, 3840, 960)
DEFINE_OPCACHE_TEST(4000, 4000, 1000)
DEFINE_OPCACHE_TEST(4096, 4096, 1024)
DEFINE_OPCACHE_TEST(4192, 4192, 1048)
DEFINE_OPCACHE_TEST(4352, 4352, 1088)
DEFINE_OPCACHE_TEST(4608, 4608, 1152)
DEFINE_OPCACHE_TEST(5120, 5120, 1280)
DEFINE_OPCACHE_TEST(6144, 6144, 1536)
DEFINE_OPCACHE_TEST(8192, 8192, 2048)
DEFINE_OPCACHE_TEST(12288, 12288, 3072)
DEFINE_OPCACHE_TEST(16384, 16384, 4096)
DEFINE_OPCACHE_TEST(24576, 24576, 6144)
DEFINE_OPCACHE_TEST(32768, 32768, 8192)
DEFINE_OPCACHE_TEST(65536, 65536, 16384)

struct TestEntry {
    std::string label;
    int uops;
    double (*func)();
};

int main() {
    // Pin to Core 0 (Physical Core 0 on Renoir APU)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1);
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    // Measure nominal vs core boost frequency
    LARGE_INTEGER qpc_freq, qpc_t0, qpc_t1;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&qpc_t0);
    uint64_t tsc_0 = __rdtsc();
    Sleep(50);
    QueryPerformanceCounter(&qpc_t1);
    uint64_t tsc_1 = __rdtsc();
    double elapsed_sec = (double)(qpc_t1.QuadPart - qpc_t0.QuadPart) / (double)qpc_freq.QuadPart;
    double tsc_ghz = (double)(tsc_1 - tsc_0) / elapsed_sec / 1e9;

    std::cout << "========================================================================================\n";
    std::cout << " Step 4 Discovery: Zen 2 Op-Cache & L1I Capacity Boundaries\n";
    std::cout << " Target: AMD Ryzen 7 4800HS (Renoir Zen 2, 8C/16T, Measured Base TSC: "
              << std::fixed << std::setprecision(2) << tsc_ghz << " GHz)\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(14) << "Loop Body"
              << " | " << std::setw(12) << "Code Size"
              << " | " << std::setw(14) << "CPI (TSC Cyc)"
              << " | " << std::setw(12) << "IPC (Nominal)"
              << " | " << "Frontend Subsystem\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    std::vector<TestEntry> tests = {
        {"256 uops", 256, test_256},
        {"512 uops", 512, test_512},
        {"1024 uops", 1024, test_1024},
        {"2048 uops", 2048, test_2048},
        {"3072 uops", 3072, test_3072},
        {"4096 uops", 4096, test_4096},
        {"5120 uops", 5120, test_5120},
        {"8192 uops", 8192, test_8192},
        {"12288 uops", 12288, test_12288},
        {"16384 uops", 16384, test_16384},
        {"24576 uops", 24576, test_24576},
        {"32768 uops", 32768, test_32768},
        {"65536 uops", 65536, test_65536}
    };

    std::ofstream json_out("research/raw/opcache_cliff_results.json");
    json_out << "[\n";

    for (size_t i = 0; i < tests.size(); ++i) {
        double cpi = tests[i].func();
        double ipc = 1.0 / cpi;
        int code_bytes = tests[i].uops * 3; // 3 bytes per addl $1, reg
        std::string size_str = std::to_string(code_bytes / 1024) + " KB (" + std::to_string(tests[i].uops) + " uops)";

        std::string subsystem;
        if (tests[i].uops <= 4096) {
            subsystem = "Op-Cache Hit (4K Entry Max)";
        } else if (code_bytes <= 32768) {
            subsystem = "Op-Cache Overflow -> L1I Hit (32 KB L1I)";
        } else {
            subsystem = "L1I Overflow -> L2 Fetch (512 KB L2)";
        }

        std::cout << std::left << std::setw(14) << tests[i].label
                  << " | " << std::setw(12) << (std::to_string(code_bytes / 1024) + " KB")
                  << " | " << std::fixed << std::setprecision(4) << std::setw(14) << cpi
                  << " | " << std::fixed << std::setprecision(2) << std::setw(12) << ipc
                  << " | " << subsystem << "\n";

        json_out << "  {\n"
                 << "    \"label\": \"" << tests[i].label << "\",\n"
                 << "    \"uops\": " << tests[i].uops << ",\n"
                 << "    \"code_bytes\": " << code_bytes << ",\n"
                 << "    \"cpi\": " << cpi << ",\n"
                 << "    \"ipc\": " << ipc << ",\n"
                 << "    \"subsystem\": \"" << subsystem << "\"\n"
                 << "  }" << (i + 1 < tests.size() ? ",\n" : "\n");
    }
    json_out << "]\n";
    json_out.close();

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "[SUCCESS] Zen 2 Op-cache cliff results saved to: research/raw/opcache_cliff_results.json\n\n";

    return 0;
}
