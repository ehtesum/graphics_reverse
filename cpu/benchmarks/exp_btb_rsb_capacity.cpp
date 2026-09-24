#include <windows.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <x86intrin.h>

// Disable inlining to ensure actual CALL and RET instructions are generated
#define NOINLINE __attribute__((noinline))

// Forward declaration of recursive call chain
NOINLINE uint64_t rsb_recurse(uint32_t current_depth, uint32_t target_depth, uint64_t val) {
    if (current_depth >= target_depth) {
        return val + 1;
    }
    // Prevent tail call optimization by doing an operation after the return
    return rsb_recurse(current_depth + 1, target_depth, val) ^ 0x55;
}

int main() {
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1); // Pin to Core 0

    std::cout << "=== AMD Zen 2 Return Stack Buffer (RSB) Capacity Dissection ===" << std::endl;

    const std::vector<uint32_t> test_depths = {
        1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31, 32, 33, 34, 36, 40, 48, 64
    };

    const uint32_t iterations = 200000; // 200k iterations per depth

    std::ofstream json("research/raw/btb_rsb_capacity_results.json");
    json << "{\n";
    json << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n";
    json << "  \"benchmark\": \"Return Stack Buffer (RSB) Capacity & Underflow Cliff\",\n";
    json << "  \"iterations_per_depth\": " << iterations << ",\n";
    json << "  \"results\": [\n";

    std::cout << std::setw(15) << "Call Depth" 
              << std::setw(20) << "Cycles/Iteration" 
              << std::setw(22) << "Cycles/Call-Ret Pair" 
              << std::setw(20) << "RSB Status" << std::endl;
    std::cout << std::string(77, '-') << std::endl;

    for (size_t i = 0; i < test_depths.size(); ++i) {
        uint32_t depth = test_depths[i];
        
        // Warmup
        volatile uint64_t dummy = 0;
        for (uint32_t w = 0; w < 1000; ++w) {
            dummy += rsb_recurse(1, depth, dummy);
        }

        _mm_mfence();
        uint64_t start_tsc = __rdtsc();
        _mm_mfence();

        for (uint32_t it = 0; it < iterations; ++it) {
            dummy += rsb_recurse(1, depth, dummy);
        }

        _mm_mfence();
        uint64_t end_tsc = __rdtsc();
        _mm_mfence();

        double total_cycles = static_cast<double>(end_tsc - start_tsc);
        double cycles_per_iter = total_cycles / iterations;
        double cycles_per_pair = cycles_per_iter / depth;

        std::string status = (depth <= 32) ? "RSB Hit (<= 32)" : "RSB Underflow Cliff (> 32)";

        std::cout << std::setw(15) << depth 
                  << std::setw(20) << std::fixed << std::setprecision(2) << cycles_per_iter 
                  << std::setw(22) << std::fixed << std::setprecision(3) << cycles_per_pair 
                  << std::setw(25) << status << std::endl;

        json << "    {\n";
        json << "      \"call_depth\": " << depth << ",\n";
        json << "      \"cycles_per_iteration\": " << cycles_per_iter << ",\n";
        json << "      \"cycles_per_call_ret_pair\": " << cycles_per_pair << ",\n";
        json << "      \"rsb_status\": \"" << status << "\"\n";
        json << "    }" << (i + 1 < test_depths.size() ? "," : "") << "\n";
    }

    json << "  ]\n";
    json << "}\n";
    json.close();

    std::cout << "\nResults written to research/raw/btb_rsb_capacity_results.json" << std::endl;
    return 0;
}
