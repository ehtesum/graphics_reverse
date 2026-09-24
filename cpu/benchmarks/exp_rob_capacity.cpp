#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <windows.h>
#include <x86intrin.h>

// Corner C5: Zen 2 Reorder Buffer (ROB) Capacity Microbenchmark
// AMD Ryzen 7 4800HS (Zen 2 Renoir)

#define N_WARMUP 2000
#define N_MEASURE 30000

// Generate independent operations that consume ROB entries.
// We use independent register additions across rax, rbx, rcx, rdx, rsi, rdi.
#define OP1  asm volatile("add $1, %0" : "+r"(v0) :: "cc");
#define OP2  OP1 asm volatile("add $1, %0" : "+r"(v1) :: "cc");
#define OP4  OP2 asm volatile("add $1, %0" : "+r"(v2) :: "cc"); asm volatile("add $1, %0" : "+r"(v3) :: "cc");
#define OP8  OP4 asm volatile("add $1, %0" : "+r"(v4) :: "cc"); asm volatile("add $1, %0" : "+r"(v5) :: "cc"); asm volatile("add $1, %0" : "+r"(v6) :: "cc"); asm volatile("add $1, %0" : "+r"(v7) :: "cc");
#define OP16 OP8 OP8

template<int N>
__attribute__((noinline))
uint64_t benchmark_rob_step(void** chain_head, size_t steps) {
    void** p = chain_head;
    uint64_t v0 = 0, v1 = 1, v2 = 2, v3 = 3, v4 = 4, v5 = 5, v6 = 6, v7 = 7;
    
    // Warmup
    for (size_t i = 0; i < N_WARMUP; ++i) {
        p = (void**)*p;
    }
    
    _mm_mfence();
    uint64_t t0 = __rdtsc();
    _mm_mfence();
    
    for (size_t i = 0; i < steps; ++i) {
        // Unresolved long-latency load from DRAM
        p = (void**)*p;
        
        // Followed by N independent ALU operations
        if constexpr (N >= 8)   { OP8 }
        if constexpr (N >= 16)  { OP8 }
        if constexpr (N >= 32)  { OP16 }
        if constexpr (N >= 48)  { OP16 }
        if constexpr (N >= 64)  { OP16 }
        if constexpr (N >= 80)  { OP16 }
        if constexpr (N >= 96)  { OP16 }
        if constexpr (N >= 112) { OP16 }
        if constexpr (N >= 128) { OP16 }
        if constexpr (N >= 144) { OP16 }
        if constexpr (N >= 160) { OP16 }
        if constexpr (N >= 176) { OP16 }
        if constexpr (N >= 192) { OP16 }
        if constexpr (N >= 208) { OP16 }
        if constexpr (N >= 224) { OP16 }
        if constexpr (N >= 240) { OP16 }
        if constexpr (N >= 256) { OP16 }
        if constexpr (N >= 272) { OP16 }
        if constexpr (N >= 288) { OP16 }
        if constexpr (N >= 304) { OP16 }
        if constexpr (N >= 320) { OP16 }
        if constexpr (N >= 352) { OP16 OP16 }
        if constexpr (N >= 384) { OP16 OP16 }
        if constexpr (N >= 448) { OP16 OP16 OP16 OP16 }
        if constexpr (N >= 512) { OP16 OP16 OP16 OP16 }
    }
    
    _mm_mfence();
    uint64_t t1 = __rdtsc();
    _mm_mfence();
    
    // Prevent dead code elimination
    asm volatile("" : "+r"(p), "+r"(v0), "+r"(v1), "+r"(v2), "+r"(v3), "+r"(v4), "+r"(v5), "+r"(v6), "+r"(v7));
    return t1 - t0;
}

int main() {
    // Pin to Core 2 (intra-CCX0)
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << 2));
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    // Allocate 64 MB buffer (well exceeds Zen 2 L3 of 4MB per CCX)
    const size_t BUFFER_SIZE = 64 * 1024 * 1024;
    const size_t LINE_SIZE = 64;
    const size_t NUM_LINES = BUFFER_SIZE / LINE_SIZE;

    std::vector<uint8_t> memory(BUFFER_SIZE, 0);
    std::vector<size_t> indices(NUM_LINES);
    std::iota(indices.begin(), indices.end(), 0);

    // Randomize indices to defeat hardware stream prefetchers
    std::mt19937_64 rng(1337);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Build linked pointer chain
    void** base = reinterpret_cast<void**>(memory.data());
    for (size_t i = 0; i < NUM_LINES - 1; ++i) {
        size_t curr_idx = indices[i] * (LINE_SIZE / sizeof(void*));
        size_t next_idx = indices[i + 1] * (LINE_SIZE / sizeof(void*));
        base[curr_idx] = &base[next_idx];
    }
    size_t last_idx = indices[NUM_LINES - 1] * (LINE_SIZE / sizeof(void*));
    base[last_idx] = &base[indices[0] * (LINE_SIZE / sizeof(void*))];

    void** chain_head = &base[indices[0] * (LINE_SIZE / sizeof(void*))];

    std::cout << "Starting Zen 2 ROB Capacity Benchmark..." << std::endl;

    // Test instruction counts
    std::vector<int> test_counts = {
        0, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 352, 384, 448, 512
    };

    std::ofstream out("research/raw/rob_capacity_results.json");
    out << "{\n  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n  \"benchmark\": \"Reorder Buffer (ROB) Capacity\",\n  \"sweep\": [\n";

    for (size_t k = 0; k < test_counts.size(); ++k) {
        int N = test_counts[k];
        uint64_t cycles = 0;
        
        switch (N) {
            case 0: cycles = benchmark_rob_step<0>(chain_head, N_MEASURE); break;
            case 8: cycles = benchmark_rob_step<8>(chain_head, N_MEASURE); break;
            case 16: cycles = benchmark_rob_step<16>(chain_head, N_MEASURE); break;
            case 32: cycles = benchmark_rob_step<32>(chain_head, N_MEASURE); break;
            case 48: cycles = benchmark_rob_step<48>(chain_head, N_MEASURE); break;
            case 64: cycles = benchmark_rob_step<64>(chain_head, N_MEASURE); break;
            case 80: cycles = benchmark_rob_step<80>(chain_head, N_MEASURE); break;
            case 96: cycles = benchmark_rob_step<96>(chain_head, N_MEASURE); break;
            case 112: cycles = benchmark_rob_step<112>(chain_head, N_MEASURE); break;
            case 128: cycles = benchmark_rob_step<128>(chain_head, N_MEASURE); break;
            case 144: cycles = benchmark_rob_step<144>(chain_head, N_MEASURE); break;
            case 160: cycles = benchmark_rob_step<160>(chain_head, N_MEASURE); break;
            case 176: cycles = benchmark_rob_step<176>(chain_head, N_MEASURE); break;
            case 192: cycles = benchmark_rob_step<192>(chain_head, N_MEASURE); break;
            case 208: cycles = benchmark_rob_step<208>(chain_head, N_MEASURE); break;
            case 224: cycles = benchmark_rob_step<224>(chain_head, N_MEASURE); break;
            case 240: cycles = benchmark_rob_step<240>(chain_head, N_MEASURE); break;
            case 256: cycles = benchmark_rob_step<256>(chain_head, N_MEASURE); break;
            case 272: cycles = benchmark_rob_step<272>(chain_head, N_MEASURE); break;
            case 288: cycles = benchmark_rob_step<288>(chain_head, N_MEASURE); break;
            case 304: cycles = benchmark_rob_step<304>(chain_head, N_MEASURE); break;
            case 320: cycles = benchmark_rob_step<320>(chain_head, N_MEASURE); break;
            case 352: cycles = benchmark_rob_step<352>(chain_head, N_MEASURE); break;
            case 384: cycles = benchmark_rob_step<384>(chain_head, N_MEASURE); break;
            case 448: cycles = benchmark_rob_step<448>(chain_head, N_MEASURE); break;
            case 512: cycles = benchmark_rob_step<512>(chain_head, N_MEASURE); break;
        }

        double cpi = (double)cycles / (double)N_MEASURE;
        std::cout << "ALU count N=" << N << " | Total cycles: " << cycles 
                  << " | Latency per iteration: " << cpi << " cycles" << std::endl;

        out << "    {\"independent_instructions\": " << N 
            << ", \"total_cycles\": " << cycles 
            << ", \"cycles_per_iteration\": " << cpi << "}" 
            << (k + 1 < test_counts.size() ? ",\n" : "\n");
    }

    out << "  ]\n}\n";
    out.close();
    std::cout << "Done! Results saved to research/raw/rob_capacity_results.json" << std::endl;
    return 0;
}
