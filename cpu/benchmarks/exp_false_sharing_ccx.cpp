#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <windows.h>
#include <x86intrin.h>

// Corner C3: Zen 2 False Sharing & Cross-CCX Cache Invalidation Storm
// Renoir 8C/16T: CCX0 = Cores 0,1,2,3 (Shared 4MB L3); CCX1 = Cores 4,5,6,7 (Shared 4MB L3)

#define ITERS 20000000ULL

struct alignas(64) SharedLine {
    volatile uint64_t word0;
    volatile uint64_t word1;
    volatile uint64_t word2;
    volatile uint64_t word3;
    volatile uint64_t word4;
    volatile uint64_t word5;
    volatile uint64_t word6;
    volatile uint64_t word7;
};

struct alignas(64) DisjointLines {
    volatile uint64_t word0;
    char pad0[56];
    volatile uint64_t word1;
    char pad1[56];
};

void pin_thread(HANDLE handle, int core_id) {
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(handle, mask);
}

struct TestResult {
    std::string config_name;
    int core_a;
    int core_b;
    std::string topology;
    double cycles_per_write_a;
    double cycles_per_write_b;
    double elapsed_ms;
};

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << " CORNER C3: Zen 2 False Sharing & Cross-CCX Cache Invalidation Storm" << std::endl;
    std::cout << " Target: AMD Ryzen 7 4800HS (Renoir 2x CCX) | MESI/MOESI Invalidation Penalty" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Calibrate TSC
    auto w0 = std::chrono::high_resolution_clock::now();
    uint64_t c0 = __rdtsc();
    Sleep(50);
    auto w1 = std::chrono::high_resolution_clock::now();
    uint64_t c1 = __rdtsc();
    double tsc_freq = (double)(c1 - c0) / std::chrono::duration<double>(w1 - w0).count();

    std::vector<TestResult> results;

    struct Config {
        std::string name;
        int core_a;
        int core_b;
        std::string topology;
        int mode; // 0: Disjoint store, 1: False-share store, 2: False-share RMW, 3: False-share Atomic
    };

    std::vector<Config> configs = {
        {"Independent Lines (Intra-CCX)", 0, 1, "Same CCX (CCX0)", 0},
        {"Independent Lines (Inter-CCX)", 0, 4, "Cross-CCX (CCX0 <-> CCX1)", 0},
        {"False Sharing Store (Intra-CCX)", 0, 1, "Same CCX (CCX0)", 1},
        {"False Sharing Store (Inter-CCX)", 0, 4, "Cross-CCX (CCX0 <-> CCX1)", 1},
        {"False Sharing RMW (Intra-CCX)",   0, 1, "Same CCX (CCX0)", 2},
        {"False Sharing RMW (Inter-CCX)",   0, 4, "Cross-CCX (CCX0 <-> CCX1)", 2},
        {"Atomic Lock Add (Intra-CCX)",     0, 1, "Same CCX (CCX0)", 3},
        {"Atomic Lock Add (Inter-CCX)",     0, 4, "Cross-CCX (CCX0 <-> CCX1)", 3}
    };

    SharedLine sl;
    DisjointLines dl;

    std::cout << "\n" << std::setw(34) << "Configuration"
              << " | " << std::setw(8) << "Cores"
              << " | " << std::setw(20) << "Topology"
              << " | " << std::setw(15) << "Cycles / Op"
              << " | " << std::setw(12) << "Slowdown" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    double baseline_cycles = 1.0;

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& cfg = configs[i];
        std::atomic<bool> start_flag(false);
        uint64_t cycles_a = 0, cycles_b = 0;
        uint64_t n_iters = (cfg.mode >= 2) ? 5000000ULL : ITERS;

        std::thread th_a([&]() {
            pin_thread(GetCurrentThread(), cfg.core_a);
            while (!start_flag.load(std::memory_order_acquire));
            uint64_t t0 = __rdtsc();
            if (cfg.mode == 0) {
                for (uint64_t k = 0; k < n_iters; ++k) dl.word0 = k;
            } else if (cfg.mode == 1) {
                for (uint64_t k = 0; k < n_iters; ++k) sl.word0 = k;
            } else if (cfg.mode == 2) {
                for (uint64_t k = 0; k < n_iters; ++k) sl.word0 = sl.word0 + 1;
            } else if (cfg.mode == 3) {
                for (uint64_t k = 0; k < n_iters; ++k) {
                    _InterlockedIncrement64((volatile long long*)&sl.word0);
                }
            }
            uint64_t t1 = __rdtsc();
            cycles_a = (t1 - t0);
        });

        std::thread th_b([&]() {
            pin_thread(GetCurrentThread(), cfg.core_b);
            while (!start_flag.load(std::memory_order_acquire));
            uint64_t t0 = __rdtsc();
            if (cfg.mode == 0) {
                for (uint64_t k = 0; k < n_iters; ++k) dl.word1 = k;
            } else if (cfg.mode == 1) {
                for (uint64_t k = 0; k < n_iters; ++k) sl.word1 = k;
            } else if (cfg.mode == 2) {
                for (uint64_t k = 0; k < n_iters; ++k) sl.word1 = sl.word1 + 1;
            } else if (cfg.mode == 3) {
                for (uint64_t k = 0; k < n_iters; ++k) {
                    _InterlockedIncrement64((volatile long long*)&sl.word1);
                }
            }
            uint64_t t1 = __rdtsc();
            cycles_b = (t1 - t0);
        });

        Sleep(10);
        auto wall_start = std::chrono::high_resolution_clock::now();
        start_flag.store(true, std::memory_order_release);

        th_a.join();
        th_b.join();
        auto wall_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

        double c_per_op_a = (double)cycles_a / (double)n_iters;
        double c_per_op_b = (double)cycles_b / (double)n_iters;
        double avg_c_op = (c_per_op_a + c_per_op_b) / 2.0;

        if (i == 0) baseline_cycles = avg_c_op;
        double slowdown = avg_c_op / baseline_cycles;

        std::string cores_str = std::to_string(cfg.core_a) + " & " + std::to_string(cfg.core_b);

        std::cout << std::setw(34) << cfg.name
                  << " | " << std::setw(8) << cores_str
                  << " | " << std::setw(20) << cfg.topology
                  << " | " << std::setw(15) << std::fixed << std::setprecision(2) << avg_c_op
                  << " | " << std::setw(11) << std::fixed << std::setprecision(2) << slowdown << "x" << std::endl;

        results.push_back({cfg.name, cfg.core_a, cfg.core_b, cfg.topology, c_per_op_a, c_per_op_b, elapsed_ms});
    }

    // Write to JSON
    std::ofstream ofs("research/raw/false_sharing_ccx_results.json");
    ofs << "{\n";
    ofs << "  \"cpu\": \"AMD Ryzen 7 4800HS\",\n";
    ofs << "  \"tsc_ghz\": " << std::fixed << std::setprecision(3) << (tsc_freq / 1e9) << ",\n";
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        ofs << "    {\n";
        ofs << "      \"config\": \"" << results[i].config_name << "\",\n";
        ofs << "      \"cores\": [" << results[i].core_a << ", " << results[i].core_b << "],\n";
        ofs << "      \"topology\": \"" << results[i].topology << "\",\n";
        ofs << "      \"cycles_per_store_a\": " << std::fixed << std::setprecision(2) << results[i].cycles_per_write_a << ",\n";
        ofs << "      \"cycles_per_store_b\": " << std::fixed << std::setprecision(2) << results[i].cycles_per_write_b << ",\n";
        ofs << "      \"elapsed_ms\": " << std::fixed << std::setprecision(2) << results[i].elapsed_ms << "\n";
        ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "\n[SUCCESS] Raw telemetry written to research/raw/false_sharing_ccx_results.json" << std::endl;
    return 0;
}
