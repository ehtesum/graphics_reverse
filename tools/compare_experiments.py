#!/usr/bin/env python3
"""
tools/compare_experiments.py
Phase 8: Experiment Comparator

Compares two experiment result directories (e.g. results/001_add_one vs results/002_double_add)
and contrasts instruction counts, register pressures, execution times, and assembly diffs.
"""

import argparse
import difflib
import json
import sys
from pathlib import Path


def load_json(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_text(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    return ""


def compare_dirs(dir_a: Path, dir_b: Path) -> dict:
    name_a = dir_a.name
    name_b = dir_b.name

    exec_a = load_json(dir_a / "execution.json")
    exec_b = load_json(dir_b / "execution.json")

    analysis_a = load_json(dir_a / "analysis.json")
    analysis_b = load_json(dir_b / "analysis.json")

    sass_a = load_text(dir_a / "sass.txt")
    sass_b = load_text(dir_b / "sass.txt")

    # Metrics extraction
    time_a = exec_a.get("kernel_time_ms", "N/A")
    time_b = exec_b.get("kernel_time_ms", "N/A")

    instr_a = analysis_a.get("instruction_count", {}).get("total", "N/A")
    instr_b = analysis_b.get("instruction_count", {}).get("total", "N/A")

    regs_a = analysis_a.get("registers", {}).get("count_referenced", "N/A")
    regs_b = analysis_b.get("registers", {}).get("count_referenced", "N/A")

    breakdown_a = analysis_a.get("instruction_count", {}).get("breakdown", {})
    breakdown_b = analysis_b.get("instruction_count", {}).get("breakdown", {})

    # Calculate differences if numbers available
    time_diff = (time_b - time_a) if isinstance(time_a, (int, float)) and isinstance(time_b, (int, float)) else "N/A"
    instr_diff = (instr_b - instr_a) if isinstance(instr_a, int) and isinstance(instr_b, int) else "N/A"
    reg_diff = (regs_b - regs_a) if isinstance(regs_a, int) and isinstance(regs_b, int) else "N/A"

    # Line-by-line diff of SASS
    sass_diff_lines = list(difflib.unified_diff(
        sass_a.splitlines(),
        sass_b.splitlines(),
        fromfile=f"{name_a}/sass.txt",
        tofile=f"{name_b}/sass.txt",
        lineterm=""
    ))

    result = {
        "experiment_a": name_a,
        "experiment_b": name_b,
        "metrics": {
            "kernel_time_ms": {"a": time_a, "b": time_b, "delta": time_diff},
            "total_instructions": {"a": instr_a, "b": instr_b, "delta": instr_diff},
            "registers_used": {"a": regs_a, "b": regs_b, "delta": reg_diff},
            "breakdown_a": breakdown_a,
            "breakdown_b": breakdown_b
        },
        "sass_diff_summary": {
            "lines_added": sum(1 for l in sass_diff_lines if l.startswith("+") and not l.startswith("+++")),
            "lines_removed": sum(1 for l in sass_diff_lines if l.startswith("-") and not l.startswith("---"))
        },
        "sass_unified_diff": sass_diff_lines[:50]  # first 50 lines of diff
    }
    return result


def print_comparison_table(comp: dict):
    print("=" * 65)
    print(f" EXPERIMENT COMPARISON: {comp['experiment_a']} vs {comp['experiment_b']}")
    print("=" * 65)
    print(f"{'Metric':<25} | {comp['experiment_a']:<15} | {comp['experiment_b']:<15} | {'Delta':<10}")
    print("-" * 65)

    m = comp["metrics"]
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    print(f"{'Kernel Time (ms)':<25} | {fmt(m['kernel_time_ms']['a']):<15} | {fmt(m['kernel_time_ms']['b']):<15} | {fmt(m['kernel_time_ms']['delta']):<10}")
    print(f"{'Total SASS Instructions':<25} | {str(m['total_instructions']['a']):<15} | {str(m['total_instructions']['b']):<15} | {str(m['total_instructions']['delta']):<10}")
    print(f"{'Registers Referenced':<25} | {str(m['registers_used']['a']):<15} | {str(m['registers_used']['b']):<15} | {str(m['registers_used']['delta']):<10}")
    print("-" * 65)

    bd_a = m["breakdown_a"]
    bd_b = m["breakdown_b"]
    all_keys = set(bd_a.keys()) | set(bd_b.keys())
    if all_keys:
        print("Instruction Breakdown:")
        for k in sorted(all_keys):
            va = bd_a.get(k, 0)
            vb = bd_b.get(k, 0)
            delta = vb - va if isinstance(va, int) and isinstance(vb, int) else "N/A"
            print(f"  - {k:<21} | {str(va):<15} | {str(vb):<15} | {str(delta):<10}")
        print("-" * 65)

    diff_sum = comp["sass_diff_summary"]
    print(f"SASS Assembly Diff: +{diff_sum['lines_added']} lines, -{diff_sum['lines_removed']} lines")
    print("=" * 65)


def self_test():
    comp = {
        "experiment_a": "001_add_one",
        "experiment_b": "002_double_add",
        "metrics": {
            "kernel_time_ms": {"a": 0.045, "b": 0.046, "delta": 0.001},
            "total_instructions": {"a": 12, "b": 12, "delta": 0},
            "registers_used": {"a": 6, "b": 6, "delta": 0},
            "breakdown_a": {"arithmetic": 1, "loads": 1, "stores": 1},
            "breakdown_b": {"arithmetic": 1, "loads": 1, "stores": 1}
        },
        "sass_diff_summary": {"lines_added": 0, "lines_removed": 0},
        "sass_unified_diff": []
    }
    print_comparison_table(comp)
    print("[PASS] Experiment Comparator self-test succeeded.")


def main():
    parser = argparse.ArgumentParser(description="Compare two experiment results.")
    parser.add_argument("dir_a", nargs="?", help="Path to first experiment result directory.")
    parser.add_argument("dir_b", nargs="?", help="Path to second experiment result directory.")
    parser.add_argument("--test", action="store_true", help="Run internal comparator unit test.")
    parser.add_argument("-o", "--output", help="Save comparison to JSON file.")
    args = parser.parse_args()

    if args.test or not args.dir_a or not args.dir_b:
        self_test()
        if not args.dir_a or not args.dir_b:
            return

    path_a = Path(args.dir_a)
    path_b = Path(args.dir_b)

    if not path_a.exists() or not path_b.exists():
        print("Error: Both experiment directories must exist.", file=sys.stderr)
        sys.exit(1)

    comp = compare_dirs(path_a, path_b)
    print_comparison_table(comp)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        print(f"Saved comparison to: {args.output}")


if __name__ == "__main__":
    main()
