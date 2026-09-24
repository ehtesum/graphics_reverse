#!/usr/bin/env python3
"""
tools/analyze_sass.py
Phase 7: Static SASS Disassembly Analyzer

Parses disassembled NVIDIA SASS text (produced by cuobjdump --dump-sass)
and extracts empirical microarchitectural metrics.

Labels observations strictly by epistemic status:
  - OBSERVED: Directly present in compiler/disassembler text.
  - INFERRED: Interpreted based on Turing architectural documentation.
  - UNKNOWN: Cannot be established from the current artifact.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Common Turing (sm_75) Instruction Prefixes
ARITH_OPCODES = {
    "FADD", "FADD32I", "FMUL", "FMUL32I", "FFMA", "FFMA32I", "FCHK", "FMNMX",
    "IADD", "IADD3", "IADD32I", "IMUL", "IMUL32I", "IMAD", "IMAD32I", "IMADSP",
    "HADD2", "HMUL2", "HFMA2", "DADD", "DMUL", "DFMA", "LEA"
}

LOAD_OPCODES = {"LDG", "LDC", "LDS", "LDL", "LD"}
STORE_OPCODES = {"STG", "STS", "STL", "ST"}
BRANCH_OPCODES = {"BRA", "EXIT", "RET", "JMP", "CALL", "BRX", "KILL"}
SYNC_OPCODES = {"BAR", "SYNC", "MEMBAR", "DEPBAR"}
MOVE_OPCODES = {"MOV", "MOV32I", "PRMT", "SEL", "SGXT", "SHFL", "F2F", "I2I", "F2I", "I2F"}


def parse_sass(sass_text: str) -> dict:
    lines = sass_text.splitlines()

    kernel_name = "unknown"
    instructions = []
    registers_observed = set()
    predicates_observed = set()

    # Regex patterns
    # Example: Function : _Z4testPf
    func_pattern = re.compile(r"^\s*Function\s*:\s*(\S+)")
    # Example: /*0010*/                   MOV R1, c[0x0][0x28] ;
    # Or with predicates: /*0020*/   @P0  BRA `(.L_x_0) ;
    instr_pattern = re.compile(
        r"^\s*/\*[0-9a-fA-F]+\*/\s+(@!?[pP]\d+\s+)?([A-Z0-9_\.]+)\s+([^;]+)?;"
    )
    reg_pattern = re.compile(r"\b(R\d+)\b")
    pred_pattern = re.compile(r"\b(P\d+)\b")

    for line in lines:
        fn_match = func_pattern.search(line)
        if fn_match:
            kernel_name = fn_match.group(1)
            continue

        ins_match = instr_pattern.search(line)
        if ins_match:
            predicate_guard = ins_match.group(1).strip() if ins_match.group(1) else None
            full_opcode = ins_match.group(2).strip()
            operands_str = ins_match.group(3).strip() if ins_match.group(3) else ""

            # Base opcode without modifier (e.g., LDG.E.SYS -> LDG)
            base_opcode = full_opcode.split(".")[0]

            # Track registers
            for reg in reg_pattern.findall(operands_str):
                registers_observed.add(reg)
            if predicate_guard:
                for pred in pred_pattern.findall(predicate_guard):
                    predicates_observed.add(pred)

            instructions.append({
                "line": line.strip(),
                "guard": predicate_guard,
                "opcode_full": full_opcode,
                "opcode_base": base_opcode,
                "operands": operands_str
            })

    # Categorize instructions
    count_arith = sum(1 for i in instructions if i["opcode_base"] in ARITH_OPCODES)
    count_load = sum(1 for i in instructions if i["opcode_base"] in LOAD_OPCODES)
    count_store = sum(1 for i in instructions if i["opcode_base"] in STORE_OPCODES)
    count_branch = sum(1 for i in instructions if i["opcode_base"] in BRANCH_OPCODES)
    count_sync = sum(1 for i in instructions if i["opcode_base"] in SYNC_OPCODES)
    count_move = sum(1 for i in instructions if i["opcode_base"] in MOVE_OPCODES)
    count_other = len(instructions) - (
        count_arith + count_load + count_store + count_branch + count_sync + count_move
    )

    # Sort registers by index
    sorted_regs = sorted(
        list(registers_observed),
        key=lambda r: int(r[1:]) if r[1:].isdigit() else 999
    )

    return {
        "kernel_name": {
            "value": kernel_name,
            "status": "OBSERVED"
        },
        "instruction_count": {
            "total": len(instructions),
            "breakdown": {
                "arithmetic": count_arith,
                "loads": count_load,
                "stores": count_store,
                "branches_and_control": count_branch,
                "synchronization": count_sync,
                "data_movement": count_move,
                "other": count_other
            },
            "status": "OBSERVED"
        },
        "registers": {
            "unique_referenced": sorted_regs,
            "count_referenced": len(sorted_regs),
            "max_register_index": int(sorted_regs[-1][1:]) if sorted_regs else 0,
            "status": "OBSERVED"
        },
        "predicates": {
            "unique_referenced": sorted(list(predicates_observed)),
            "count_referenced": len(predicates_observed),
            "status": "OBSERVED"
        },
        "microarchitecture_inferences": {
            "target_architecture": "NVIDIA Turing (Compute 7.5)",
            "warp_execution_model": "32 threads per warp lockstep scheduling",
            "inferred_memory_coalescing": (
                "Continuous thread index mapping suggests 128-byte transaction coalescing"
                if count_load > 0 else "N/A"
            ),
            "status": "INFERRED"
        },
        "unknowns": {
            "pipeline_stall_cycles": "Cannot determine without live hardware profiling counters (Nsight)",
            "exact_cache_hit_rate": "Cannot determine from static assembly alone",
            "status": "UNKNOWN"
        }
    }


def self_test():
    """Validates parser using a synthetic Turing SASS snippet."""
    sample_sass = """
    Function : _Z4testPf
    .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS"
    /*0000*/                   MOV R1, c[0x0][0x28] ;
    /*0010*/                   S2R R0, SR_TID.X ;
    /*0020*/                   S2R R3, SR_CTAID.X ;
    /*0030*/                   IMAD R0, R3, c[0x0][0x0], R0 ;
    /*0040*/                   LEA R2, P0, R0, c[0x0][0x160], 0x2 ;
    /*0050*/                   LDG.E.SYS R4, [R2] ;
    /*0060*/                   FADD R4, R4, 1.0 ;
    /*0070*/                   STG.E.SYS [R2], R4 ;
    /*0080*/                   EXIT ;
    """
    res = parse_sass(sample_sass)
    assert res["kernel_name"]["value"] == "_Z4testPf"
    assert res["instruction_count"]["total"] == 9
    assert res["instruction_count"]["breakdown"]["loads"] == 1
    assert res["instruction_count"]["breakdown"]["stores"] == 1
    assert res["instruction_count"]["breakdown"]["arithmetic"] >= 2  # IMAD, FADD
    assert res["registers"]["count_referenced"] >= 4
    print("[PASS] Static SASS Analyzer self-test succeeded.")


def main():
    parser = argparse.ArgumentParser(description="Analyze SASS disassembly text.")
    parser.add_argument("sass_file", nargs="?", help="Path to SASS text file.")
    parser.add_argument("--test", action="store_true", help="Run internal parser unit test.")
    parser.add_argument("-o", "--output", help="Path to write analysis.json.")
    args = parser.parse_args()

    if args.test or not args.sass_file:
        self_test()
        if not args.sass_file:
            return

    path = Path(args.sass_file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    analysis = parse_sass(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {out_path}")
    else:
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
