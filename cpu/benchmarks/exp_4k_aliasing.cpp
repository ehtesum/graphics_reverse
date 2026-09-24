#include <windows.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <x86intrin.h>

int main() {
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1); // Pin to Core 0

    std::cout << "=== AMD Zen 2 Store-to-Load 4K Aliasing Collision Dissection ===" << std::endl;

    // Allocate 1 MB aligned buffer to test aliasing offsets up to 64 KB
    const size_t buf_size = 1024 * 1024;
    std::vector<uint8_t> buffer(buf_size, 0);
    uint8_t* base_ptr = buffer.data();

    const uint64_t iterations = 10000000; // 10M iterations

    struct TestCase {
        std::string label;
        size_t load_offset;
        std::string expectation;
    };

    std::vector<TestCase> cases = {
        {"Independent Line (Offset +64 B)", 64, "Clean Independent L1 (No STLF, No Alias)"},
        {"Exact Store-Load Match (Offset 0)", 0, "Clean Store-to-Load Forwarding (STLF)"},
        {"Non-Aliased (Offset +4160 B [4K+64])", 4096 + 64, "Clean Independent L1 (Different Page Offset)"},
        {"4K Aliasing Collision (Offset +4096 B)", 4096, "False Match on Bits [11:0] -> Pipeline Replay"},
        {"8K Aliasing Collision (Offset +8192 B)", 8192, "False Match on Bits [11:0] -> Pipeline Replay"},
        {"16K Aliasing Collision (Offset +16384 B)", 16384, "False Match on Bits [11:0] -> Pipeline Replay"},
        {"32K Aliasing Collision (Offset +32768 B)", 32768, "False Match on Bits [11:0] -> Pipeline Replay"},
        {"64K Aliasing Collision (Offset +65536 B)", 65536, "False Match on Bits [11:0] -> Pipeline Replay"}
    };

    std::ofstream json("research/raw/4k_aliasing_results.json");
    json << "{\n";
    json << "  \"cpu\": \"AMD Ryzen 7 4800HS (Zen 2 Renoir)\",\n";
    json << "  \"benchmark\": \"Store-to-Load 4K Aliasing Disambiguation Penalty\",\n";
    json << "  \"iterations\": " << iterations << ",\n";
    json << "  \"results\": [\n";

    std::cout << std::left << std::setw(42) << "Test Case"
              << std::setw(20) << "Cycles / Op Pair"
              << std::setw(20) << "Penalty vs Clean"
              << "State" << std::endl;
    std::cout << std::string(105, '-') << std::endl;

    double clean_cycles = 0;

    for (size_t i = 0; i < cases.size(); ++i) {
        auto& tc = cases[i];
        volatile uint64_t* store_addr = reinterpret_cast<volatile uint64_t*>(base_ptr);
        volatile uint64_t* load_addr = reinterpret_cast<volatile uint64_t*>(base_ptr + tc.load_offset);

        // Warmup
        for (int w = 0; w < 10000; ++w) {
            *store_addr = w;
            volatile uint64_t v = *load_addr;
            (void)v;
        }

        _mm_mfence();
        uint64_t start = __rdtsc();
        _mm_mfence();

        for (uint64_t it = 0; it < iterations; ++it) {
            *store_addr = it;
            volatile uint64_t v = *load_addr;
            (void)v;
        }

        _mm_mfence();
        uint64_t end = __rdtsc();
        _mm_mfence();

        double cycles_per_pair = static_cast<double>(end - start) / iterations;
        if (i == 0) clean_cycles = cycles_per_pair;
        double penalty = cycles_per_pair - clean_cycles;

        std::cout << std::left << std::setw(42) << tc.label
                  << std::setw(20) << std::fixed << std::setprecision(3) << cycles_per_pair
                  << std::setw(20) << (penalty > 0 ? "+" + std::to_string(penalty).substr(0, 5) + "c" : "0.00c")
                  << tc.expectation << std::endl;

        json << "    {\n";
        json << "      \"label\": \"" << tc.label << "\",\n";
        json << "      \"load_offset_bytes\": " << tc.load_offset << ",\n";
        json << "      \"cycles_per_pair\": " << cycles_per_pair << ",\n";
        json << "      \"penalty_cycles\": " << penalty << ",\n";
        json << "      \"state\": \"" << tc.expectation << "\"\n";
        json << "    }" << (i + 1 < cases.size() ? "," : "") << "\n";
    }

    json << "  ]\n";
    json << "}\n";
    json.close();

    std::cout << "\nResults written to research/raw/4k_aliasing_results.json" << std::endl;
    return 0;
}
