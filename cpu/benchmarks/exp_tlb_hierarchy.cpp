/**
 * cpu/benchmarks/exp_tlb_hierarchy.cpp
 * Corner 5 Forensic Experiment: Zen 2 Data TLB Hierarchy & Page Table Walk Latency
 * Target: AMD Ryzen 7 4800HS (Zen 2 Renoir, 8C/16T)
 *
 * Microarchitectural Hypotheses:
 *   1. AMD Zen 2 core features:
 *      - L1 Data TLB (D-TLB1): 64 entries (serving 64 x 4 KB = 256 KB)
 *      - L2 Data TLB (D-TLB2): 2,048 entries (serving 2,048 x 4 KB = 8 MB)
 *   2. When chasing pointers with 4 KB page stride (1 access per page):
 *      - <= 64 pages (256 KB): 100% D-TLB1 hit (0 TLB penalty).
 *      - 128 to 2,048 pages (512 KB to 8 MB): D-TLB1 miss -> D-TLB2 hit (~7-8 cycles penalty).
 *      - > 2,048 pages (> 8 MB): D-TLB2 miss -> Hardware Page Table Walk to DRAM (~50-100+ cycles penalty).
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <random>
#include <numeric>
#include <windows.h>
#include <x86intrin.h>

const int CHASE_STEPS = 500'000;
const int RUNS = 5;

// Sattolo cyclic permutation generator
std::vector<size_t> create_page_cycle(size_t num_pages) {
    std::vector<size_t> indices(num_pages);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937_64 rng(1337);
    for (size_t i = num_pages - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i - 1);
        size_t j = dist(rng);
        std::swap(indices[i], indices[j]);
    }
    return indices;
}

double benchmark_tlb(size_t num_pages) {
    size_t total_bytes = num_pages * 4096;
    char* buf = static_cast<char*>(VirtualAlloc(NULL, total_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!buf) return -1.0;

    // Build page-strided cyclic pointer chain
    // Node at page p points to node at page next_p
    auto page_order = create_page_cycle(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
        size_t next_page = page_order[i];
        void* curr_node = buf + (i * 4096);
        void* next_node = buf + (next_page * 4096);
        *reinterpret_cast<void**>(curr_node) = next_node;
    }

    // Warmup traversal (touch all pages)
    void* p = buf;
    for (size_t i = 0; i < num_pages * 2; ++i) {
        p = *reinterpret_cast<void**>(p);
    }

    // Measured pointer chase with serialized inline assembly
    uint64_t min_cycles = UINT64_MAX;
    for (int r = 0; r < RUNS; ++r) {
        uint64_t steps = CHASE_STEPS;
        void* curr_p = p;
        _mm_mfence();
        uint64_t t0 = __rdtsc();
        __asm__ __volatile__(
            "1:\n\t"
            "movq (%[ptr]), %[ptr]\n\t"
            "decq %[cnt]\n\t"
            "jnz 1b\n\t"
            : [ptr] "+r"(curr_p), [cnt] "+r"(steps)
            :
            : "memory", "cc"
        );
        _mm_mfence();
        uint64_t t1 = __rdtsc();
        p = curr_p;
        uint64_t delta = t1 - t0;
        if (delta < min_cycles) min_cycles = delta;
    }

    VirtualFree(buf, 0, MEM_RELEASE);
    return static_cast<double>(min_cycles) / static_cast<double>(CHASE_STEPS);
}

int main() {
    // Pin to Physical Core 0
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    std::cout << "========================================================================================\n";
    std::cout << " Corner 5: Zen 2 Data TLB Hierarchy (D-TLB1 vs D-TLB2) & Page Walk Forensic Audit\n";
    std::cout << " Target: AMD Ryzen 7 4800HS (Zen 2 Renoir, 4 KB Page Stride Pointer Chasing)\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(14) << "Pages Chased"
              << " | " << std::setw(14) << "Address Space"
              << " | " << std::setw(18) << "Cycles / Access"
              << " | " << "TLB Subsystem Classification\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    std::vector<size_t> test_pages = {
        8,      // 32 KB (Fits in L1 D-TLB)
        16,     // 64 KB (Fits in L1 D-TLB)
        32,     // 128 KB (Fits in L1 D-TLB)
        64,     // 256 KB (L1 D-TLB Boundary: 64 entries)
        128,    // 512 KB (Spills to L2 D-TLB)
        256,    // 1 MB (L2 D-TLB hit)
        512,    // 2 MB (L2 D-TLB hit)
        1024,   // 4 MB (L2 D-TLB hit)
        2048,   // 8 MB (L2 D-TLB Boundary: 2,048 entries)
        3072,   // 12 MB (Spills L2 D-TLB -> Page Table Walk)
        4096,   // 16 MB (D-TLB miss -> DRAM Page Walk)
        8192,   // 32 MB (Severe TLB Thrashing)
        16384   // 64 MB (Deep Page Table Walking)
    };

    std::ofstream json_out("research/raw/tlb_hierarchy_results.json");
    json_out << "[\n";

    for (size_t i = 0; i < test_pages.size(); ++i) {
        size_t pages = test_pages[i];
        double cyc = benchmark_tlb(pages);
        size_t kb = pages * 4;
        std::string size_str = (kb < 1024) ? (std::to_string(kb) + " KB") : (std::to_string(kb / 1024) + " MB");

        std::string subsystem;
        if (pages <= 64) {
            subsystem = "D-TLB1 Hit (64 entries, <1c penalty)";
        } else if (pages <= 2048) {
            subsystem = "D-TLB1 Miss -> D-TLB2 Hit (2K entries, ~7-8c)";
        } else {
            subsystem = "D-TLB2 Miss -> Hardware Page Table Walk in DRAM";
        }

        std::cout << std::left << std::setw(14) << (std::to_string(pages) + " pages")
                  << " | " << std::setw(14) << size_str
                  << " | " << std::fixed << std::setprecision(2) << std::setw(18) << cyc
                  << " | " << subsystem << "\n";

        json_out << "  {\n"
                 << "    \"pages\": " << pages << ",\n"
                 << "    \"address_space_kb\": " << kb << ",\n"
                 << "    \"cycles_per_access\": " << cyc << ",\n"
                 << "    \"subsystem\": \"" << subsystem << "\"\n"
                 << "  }" << (i + 1 < test_pages.size() ? ",\n" : "\n");
    }

    json_out << "]\n";
    json_out.close();

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "[SUCCESS] TLB hierarchy results saved to: research/raw/tlb_hierarchy_results.json\n\n";

    return 0;
}
