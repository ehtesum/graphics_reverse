#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <windows.h>
#include <x86intrin.h>

// Corner C4: AMD Zen 2 Instruction TLB (I-TLB) Hierarchy & Code Page Dispersion
// Isolating L1 I-TLB (64 entries) and L2 I-TLB (512 entries) capacity knees.

void pin_thread(int core) {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << core));
}

struct ITLBResult {
    int num_pages;
    double cycles_per_jump;
    std::string level;
};

int main() {
    pin_thread(2);
    std::cout << "================================================================================" << std::endl;
    std::cout << " CORNER C4: AMD Zen 2 Instruction TLB (I-TLB) Hierarchy & Page Dispersion" << std::endl;
    std::cout << " Target: AMD Ryzen 7 4800HS (Zen 2 Renoir) | L1 I-TLB (64) & L2 I-TLB (512)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const size_t PAGE_SIZE = 4096;
    const std::vector<int> PAGE_COUNTS = {8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};

    std::vector<ITLBResult> results;

    std::cout << "\n" << std::setw(15) << "Code Pages" 
              << " | " << std::setw(15) << "Total Code KB"
              << " | " << std::setw(20) << "Cycles / Jump"
              << " | " << std::setw(22) << "Identified Hierarchy" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (int N : PAGE_COUNTS) {
        size_t total_alloc = (size_t)N * PAGE_SIZE;
        uint8_t* code_mem = (uint8_t*)VirtualAlloc(NULL, total_alloc, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        if (!code_mem) {
            std::cerr << "Failed to allocate executable memory for N=" << N << std::endl;
            continue;
        }

        // Create randomized cyclic permutation of pages
        std::vector<int> pages(N);
        std::iota(pages.begin(), pages.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(pages.begin(), pages.end(), rng);

        // At the start of each page, write an x86-64 near JMP instruction (0xE9 <rel32>)
        // or indirect JMP.
        // Direct relative 32-bit jump: opcode 0xE9, followed by int32 offset.
        for (int i = 0; i < N - 1; ++i) {
            int src_page = pages[i];
            int dst_page = pages[i + 1];
            uint8_t* src_ptr = code_mem + (src_page * PAGE_SIZE);
            uint8_t* dst_ptr = code_mem + (dst_page * PAGE_SIZE);

            // JMP rel32: offset is relative to next instruction (src_ptr + 5)
            int32_t rel_offset = (int32_t)(dst_ptr - (src_ptr + 5));
            src_ptr[0] = 0xE9;
            memcpy(src_ptr + 1, &rel_offset, 4);
        }

        // For the last page, write RET (0xC3) to return to caller
        int last_page = pages.back();
        uint8_t* last_ptr = code_mem + (last_page * PAGE_SIZE);
        last_ptr[0] = 0xC3; // RET

        // Function pointer to first page
        typedef void (*FuncPtr)();
        FuncPtr entry_point = (FuncPtr)(code_mem + (pages[0] * PAGE_SIZE));

        // Warmup I-cache / TLB
        entry_point();

        // Measure time for repeating traversals
        const int REPEATS = 50000 / N + 10;
        uint64_t t0 = __rdtsc();
        for (int r = 0; r < REPEATS; ++r) {
            entry_point();
        }
        uint64_t t1 = __rdtsc();

        uint64_t total_jumps = (uint64_t)N * REPEATS;
        double cycles_per_jump = (double)(t1 - t0) / (double)total_jumps;

        std::string level;
        if (N <= 64) {
            level = "L1 I-TLB Hit (<64)";
        } else if (N <= 512) {
            level = "L2 I-TLB Hit (64-512)";
        } else {
            level = "I-TLB Miss (Page Walk)";
        }

        std::cout << std::setw(15) << N
                  << " | " << std::setw(15) << (N * 4)
                  << " | " << std::setw(20) << std::fixed << std::setprecision(2) << cycles_per_jump
                  << " | " << std::setw(22) << level << std::endl;

        results.push_back({N, cycles_per_jump, level});

        VirtualFree(code_mem, 0, MEM_RELEASE);
    }

    // Write to JSON
    std::ofstream ofs("research/raw/itlb_hierarchy_results.json");
    ofs << "{\n";
    ofs << "  \"cpu\": \"AMD Ryzen 7 4800HS\",\n";
    ofs << "  \"architecture\": \"Zen 2 (Renoir)\",\n";
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        ofs << "    {\n";
        ofs << "      \"pages\": " << results[i].num_pages << ",\n";
        ofs << "      \"code_kb\": " << (results[i].num_pages * 4) << ",\n";
        ofs << "      \"cycles_per_jump\": " << std::fixed << std::setprecision(2) << results[i].cycles_per_jump << ",\n";
        ofs << "      \"level\": \"" << results[i].level << "\"\n";
        ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "\n[SUCCESS] Raw telemetry written to research/raw/itlb_hierarchy_results.json" << std::endl;
    return 0;
}
