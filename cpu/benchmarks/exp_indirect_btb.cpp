/**
 * cpu/benchmarks/exp_indirect_btb.cpp
 * Corner 7 Forensic Experiment: Zen 2 Branch Target Buffer & Indirect Jump Prediction
 * Target: AMD Ryzen 7 4800HS (Zen 2 Renoir, 8C/16T)
 *
 * Microarchitectural Hypotheses:
 *   1. Indirect jumps (jmp *reg, virtual functions) depend on the Branch Target Buffer (BTB).
 *   2. When cycling through M distinct indirect targets:
 *      - M = 1: 100% predictable target (~1-2 cycles per call).
 *      - M <= 32: Predictable via Zen 2 path-history indirect predictor.
 *      - M >= 64: Predictor collapses; every indirect jump suffers a full pipeline resteer (~14-16 cycles).
 */

#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <x86intrin.h>

const uint64_t TOTAL_CALLS = 2'000'000;
const int RUNS = 5;

// Define 64 distinct target functions
#define DEFINE_FUNC(N) \
__attribute__((noinline)) uint32_t target_func_##N(uint32_t x) { \
    return x + N; \
}

DEFINE_FUNC(0) DEFINE_FUNC(1) DEFINE_FUNC(2) DEFINE_FUNC(3)
DEFINE_FUNC(4) DEFINE_FUNC(5) DEFINE_FUNC(6) DEFINE_FUNC(7)
DEFINE_FUNC(8) DEFINE_FUNC(9) DEFINE_FUNC(10) DEFINE_FUNC(11)
DEFINE_FUNC(12) DEFINE_FUNC(13) DEFINE_FUNC(14) DEFINE_FUNC(15)
DEFINE_FUNC(16) DEFINE_FUNC(17) DEFINE_FUNC(18) DEFINE_FUNC(19)
DEFINE_FUNC(20) DEFINE_FUNC(21) DEFINE_FUNC(22) DEFINE_FUNC(23)
DEFINE_FUNC(24) DEFINE_FUNC(25) DEFINE_FUNC(26) DEFINE_FUNC(27)
DEFINE_FUNC(28) DEFINE_FUNC(29) DEFINE_FUNC(30) DEFINE_FUNC(31)
DEFINE_FUNC(32) DEFINE_FUNC(33) DEFINE_FUNC(34) DEFINE_FUNC(35)
DEFINE_FUNC(36) DEFINE_FUNC(37) DEFINE_FUNC(38) DEFINE_FUNC(39)
DEFINE_FUNC(40) DEFINE_FUNC(41) DEFINE_FUNC(42) DEFINE_FUNC(43)
DEFINE_FUNC(44) DEFINE_FUNC(45) DEFINE_FUNC(46) DEFINE_FUNC(47)
DEFINE_FUNC(48) DEFINE_FUNC(49) DEFINE_FUNC(50) DEFINE_FUNC(51)
DEFINE_FUNC(52) DEFINE_FUNC(53) DEFINE_FUNC(54) DEFINE_FUNC(55)
DEFINE_FUNC(56) DEFINE_FUNC(57) DEFINE_FUNC(58) DEFINE_FUNC(59)
DEFINE_FUNC(60) DEFINE_FUNC(61) DEFINE_FUNC(62) DEFINE_FUNC(63)

typedef uint32_t (*FuncPtr)(uint32_t);

FuncPtr ALL_FUNCS[64] = {
    target_func_0, target_func_1, target_func_2, target_func_3,
    target_func_4, target_func_5, target_func_6, target_func_7,
    target_func_8, target_func_9, target_func_10, target_func_11,
    target_func_12, target_func_13, target_func_14, target_func_15,
    target_func_16, target_func_17, target_func_18, target_func_19,
    target_func_20, target_func_21, target_func_22, target_func_23,
    target_func_24, target_func_25, target_func_26, target_func_27,
    target_func_28, target_func_29, target_func_30, target_func_31,
    target_func_32, target_func_33, target_func_34, target_func_35,
    target_func_36, target_func_37, target_func_38, target_func_39,
    target_func_40, target_func_41, target_func_42, target_func_43,
    target_func_44, target_func_45, target_func_46, target_func_47,
    target_func_48, target_func_49, target_func_50, target_func_51,
    target_func_52, target_func_53, target_func_54, target_func_55,
    target_func_56, target_func_57, target_func_58, target_func_59,
    target_func_60, target_func_61, target_func_62, target_func_63
};

double benchmark_indirect(int num_targets) {
    std::vector<FuncPtr> table(num_targets);
    for (int i = 0; i < num_targets; ++i) {
        table[i] = ALL_FUNCS[i % 64];
    }

    uint64_t min_cycles = UINT64_MAX;
    uint32_t val = 1;

    for (int r = 0; r < RUNS; ++r) {
        _mm_mfence();
        uint64_t t0 = __rdtsc();
        for (uint64_t i = 0; i < TOTAL_CALLS; ++i) {
            FuncPtr f = table[i % num_targets];
            val = f(val);
        }
        _mm_mfence();
        uint64_t t1 = __rdtsc();
        uint64_t delta = t1 - t0;
        if (delta < min_cycles) min_cycles = delta;
    }

    // Sinking val to prevent DCE
    volatile uint32_t sink = val;
    (void)sink;

    return static_cast<double>(min_cycles) / static_cast<double>(TOTAL_CALLS);
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    std::cout << "========================================================================================\n";
    std::cout << " Corner 7: Zen 2 Indirect Branch Predictor & BTB Capacity Forensic Audit\n";
    std::cout << " Target: AMD Ryzen 7 4800HS (Polymorphic Function Pointer Calls)\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(18) << "Target Count"
              << " | " << std::setw(18) << "Cycles / Call"
              << " | " << std::setw(18) << "Mispredict Penalty"
              << " | " << "Predictor Behavior\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    std::vector<int> target_counts = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64};
    std::vector<std::pair<int, double>> results;

    double baseline_cyc = benchmark_indirect(1);

    for (int targets : target_counts) {
        double cyc = benchmark_indirect(targets);
        double penalty = cyc - baseline_cyc;

        std::string behavior;
        if (penalty < 1.0) {
            behavior = "100% BTB Hit (Zero Mispredicts)";
        } else if (penalty < 6.0) {
            behavior = "Partial History Tracking";
        } else {
            behavior = "Predictor Collapse -> Pipeline Flush (~14-16c)";
        }

        std::cout << std::left << std::setw(18) << (std::to_string(targets) + " targets")
                  << " | " << std::fixed << std::setprecision(2) << std::setw(18) << cyc
                  << " | " << std::fixed << std::setprecision(2) << std::setw(18) << penalty
                  << " | " << behavior << "\n";

        results.push_back({targets, cyc});
    }

    std::ofstream json_out("research/raw/indirect_btb_results.json");
    json_out << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json_out << "  {\n"
                 << "    \"targets\": " << results[i].first << ",\n"
                 << "    \"cycles_per_call\": " << results[i].second << ",\n"
                 << "    \"penalty_vs_single_target\": " << (results[i].second - baseline_cyc) << "\n"
                 << "  }" << (i + 1 < results.size() ? ",\n" : "\n");
    }
    json_out << "]\n";
    json_out.close();

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "[SUCCESS] Indirect branch telemetry saved to: research/raw/indirect_btb_results.json\n\n";

    return 0;
}
