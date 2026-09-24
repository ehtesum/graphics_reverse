#!/usr/bin/env python3
"""
run_benchmarks.py - Unified experiment runner for graphics_reverse.

Usage:
    python run_benchmarks.py --list                 # List available benchmarks
    python run_benchmarks.py --exp <id>             # Run specific experiment (e.g. 008, 010)
    python run_benchmarks.py --all-gpu              # Run all 10 GPU microbenchmarks
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "experiments"

GPU_EXPERIMENTS = {
    "008": ("008_warp_divergence.py", "Warp divergence across 1-way to 32-way branch splits"),
    "009": ("009_engine_concurrency.py", "Hardware engine concurrency & compute/copy overlap"),
    "010": ("010_bank_conflicts.py", "Shared memory 32-bank modulo arbitration & stride sweep"),
    "011": ("011_warp_shuffle.py", "Warp shuffle exchange vs shared memory and atomic staging"),
    "012": ("012_fp16_packed_math.py", "Turing Rapid Packed Math (FP16 HFMA2 vs FP32 FMA)"),
    "013": ("013_gpu_cache_latency.py", "%clock64 pointer chasing for L1, L2, DRAM latency"),
    "014": ("014_sfu_throughput.py", "Special Function Unit (sin/cos/rsqrt) pipeline throughput"),
    "015": ("015_zerocopy_streaming.py", "Zero-copy pinned host memory streaming performance"),
    "016": ("016_atomic_contention.py", "Atomic reduction (RED.E.ADD) vs Compare-And-Swap scaling"),
    "falsification": ("falsification_suite.py", "Adversarial hypothesis falsification test suite")
}

def list_benchmarks():
    print("=== Available GPU Microbenchmarks ===")
    for exp_id, (script, desc) in GPU_EXPERIMENTS.items():
        print(f"  [{exp_id:>13}] {script:<26} -> {desc}")
    
    cpu_dir = ROOT / "cpu" / "benchmarks"
    if cpu_dir.exists():
        cpps = sorted([f.name for f in cpu_dir.glob("*.cpp") if f.name != "common.hpp"])
        print(f"\n=== Available CPU Benchmarks ({len(cpps)} C++ sources) ===")
        for c in cpps[:6]:
            print(f"  {c}")
        if len(cpps) > 6:
            print(f"  ... and {len(cpps) - 6} more in cpu/benchmarks/")

def run_experiment(exp_id: str):
    matched = None
    for k, (script, _) in GPU_EXPERIMENTS.items():
        if exp_id in (k, script, script.split(".")[0]):
            matched = script
            break
            
    if not matched:
        print(f"[ERROR] Unknown experiment: '{exp_id}'. Run with --list to see options.")
        return 1
        
    script_path = EXP_DIR / matched
    print(f"[INFO] Running {matched}...")
    res = subprocess.run([sys.executable, str(script_path)])
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="Empirical Hardware Research Runner")
    parser.add_argument("--list", action="store_true", help="List all available benchmarks")
    parser.add_argument("--exp", type=str, help="Run a specific experiment by ID (e.g. 008, 010)")
    parser.add_argument("--all-gpu", action="store_true", help="Run all GPU experiments sequentially")
    args = parser.parse_args()

    if args.list or (len(sys.argv) == 1):
        list_benchmarks()
        return 0

    if args.exp:
        return run_experiment(args.exp)

    if args.all_gpu:
        print("[INFO] Running all GPU benchmarks...")
        for exp_id in GPU_EXPERIMENTS:
            ret = run_experiment(exp_id)
            if ret != 0:
                print(f"[WARN] Experiment {exp_id} exited with status {ret}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
