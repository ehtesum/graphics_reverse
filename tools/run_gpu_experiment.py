#!/usr/bin/env python3
"""
tools/run_gpu_experiment.py
Phase 3: Automated GPU Experiment Runner

Takes an experiment source (.ptx or .cu), compiles/assembles for Turing sm_75,
disassembles SASS via cuobjdump/nvdisasm, executes the benchmark directly
on the physical NVIDIA GeForce GTX 1650 via native nvcuda driver,
measures hardware execution time with CUDA events, and records structured artifacts.

Pipeline:
  SOURCE -> CUBIN (ptxas) -> SASS (cuobjdump/nvdisasm) -> HARDWARE EXECUTION -> JSON RESULTS
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_ROOT / "results"
TOOLS_DIR = WORKSPACE_ROOT / "tools"
CUDA_BIN = WORKSPACE_ROOT / "cuda_toolkit" / "bin"


def run_cmd(cmd_list, timeout=60, cwd=None, env=None):
    try:
        proc = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            shell=(sys.platform == "win32")
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def find_ptxas():
    ptxas = CUDA_BIN / "ptxas.exe"
    if ptxas.exists():
        return str(ptxas.resolve())
    which = shutil.which("ptxas")
    return which if which else None


def find_cuobjdump():
    cuobj = CUDA_BIN / "cuobjdump.exe"
    if cuobj.exists():
        return str(cuobj.resolve())
    which = shutil.which("cuobjdump")
    return which if which else None


def execute_cubin_on_gpu(cubin_path: Path, kernel_name="_Z4testPf", expected_val=2.0) -> dict:
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        return {"error": f"Failed to load nvcuda.dll: {e}"}

    def check(res, op):
        if res != 0:
            raise RuntimeError(f"{op} failed with error code: {res}")

    check(cuda.cuInit(0), "cuInit")

    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

    name_buf = ctypes.create_string_buffer(256)
    check(cuda.cuDeviceGetName(name_buf, 256, device), "cuDeviceGetName")
    dev_name = name_buf.value.decode("utf-8")

    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")

    try:
        module = ctypes.c_void_p()
        check(cuda.cuModuleLoad(ctypes.byref(module), str(cubin_path.resolve()).encode("utf-8")), "cuModuleLoad")

        function = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode("utf-8")), "cuModuleGetFunction")

        N = 1024 * 1024
        num_bytes = N * 4

        d_mem = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), num_bytes), "cuMemAlloc")

        h_data = (ctypes.c_float * N)(*([1.0] * N))
        check(cuda.cuMemcpyHtoD_v2(d_mem, h_data, num_bytes), "cuMemcpyHtoD")

        start_evt = ctypes.c_void_p()
        stop_evt = ctypes.c_void_p()
        check(cuda.cuEventCreate(ctypes.byref(start_evt), 0), "cuEventCreate start")
        check(cuda.cuEventCreate(ctypes.byref(stop_evt), 0), "cuEventCreate stop")

        param_ptr = ctypes.c_uint64(d_mem.value)
        kernel_args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(param_ptr), ctypes.c_void_p))

        # Launch
        check(cuda.cuEventRecord(start_evt, None), "cuEventRecord start")
        check(cuda.cuLaunchKernel(
            function,
            4096, 1, 1,
            256, 1, 1,
            0, None,
            kernel_args, None
        ), "cuLaunchKernel")
        check(cuda.cuEventRecord(stop_evt, None), "cuEventRecord stop")
        check(cuda.cuEventSynchronize(stop_evt), "cuEventSynchronize")

        elapsed_ms = ctypes.c_float()
        check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start_evt, stop_evt), "cuEventElapsedTime")

        out_data = (ctypes.c_float * N)()
        check(cuda.cuMemcpyDtoH_v2(out_data, d_mem, num_bytes), "cuMemcpyDtoH")

        first_val = round(out_data[0], 4)
        verified = all(abs(out_data[i] - expected_val) < 1e-3 for i in range(1024))

        cuda.cuMemFree_v2(d_mem)
        cuda.cuEventDestroy(start_evt)
        cuda.cuEventDestroy(stop_evt)

        return {
            "device": dev_name,
            "kernel": kernel_name,
            "elements": N,
            "threads_per_block": 256,
            "blocks_per_grid": 4096,
            "kernel_time_ms": round(elapsed_ms.value, 6),
            "first_element_output": first_val,
            "expected_element_output": expected_val,
            "verified": verified
        }
    finally:
        cuda.cuCtxDestroy_v2(context)


def run_experiment(source_file: Path, arch="sm_75", force=False):
    if not source_file.exists():
        # Check if .ptx sibling exists
        ptx_cand = source_file.with_suffix(".ptx")
        if ptx_cand.exists():
            source_file = ptx_cand
        else:
            print(f"Error: Source file does not exist: {source_file}", file=sys.stderr)
            sys.exit(1)

    exp_name = source_file.stem
    exp_dir = RESULTS_DIR / exp_name

    if exp_dir.exists() and not force:
        print(f"Experiment directory already exists: {exp_dir}")
        print("Use --force to overwrite previous run results.")
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, exp_dir / source_file.name)

    ptxas = find_ptxas()
    cuobjdump = find_cuobjdump()

    metadata = {
        "experiment": exp_name,
        "source_file": str(source_file.resolve()),
        "arch": arch,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ptxas_found": bool(ptxas),
        "cuobjdump_found": bool(cuobjdump),
        "compilation_success": False,
        "execution_success": False
    }

    if not ptxas:
        print("Error: ptxas assembler not found.", file=sys.stderr)
        return

    print("=" * 65)
    print(f" EXPERIMENT EXECUTION: {exp_name} ({arch})")
    print("=" * 65)

    # 1. Assemble to CUBIN
    cubin_file = exp_dir / "program.cubin"
    print(f"[1/4] Assembling {source_file.suffix.upper()} to CUBIN with ptxas (arch={arch})...")
    code, out, err = run_cmd([ptxas, "-v", f"-arch={arch}", str(source_file), "-o", str(cubin_file)])
    if code != 0:
        print(f"[-] Assembly failed: {err}", file=sys.stderr)
        return
    metadata["compilation_success"] = True
    print(f"      Saved: {cubin_file}")
    if err:
        print(f"      Properties: {err}")

    # 2. Extract SASS
    sass_file = exp_dir / "sass.txt"
    if cuobjdump:
        print("[2/4] Disassembling SASS with cuobjdump/nvdisasm...")
        custom_env = os.environ.copy()
        custom_env["PATH"] = str(CUDA_BIN) + ";" + custom_env.get("PATH", "")
        code, out, err = run_cmd([cuobjdump, "--dump-sass", str(cubin_file)], env=custom_env)
        if code == 0 and out:
            with open(sass_file, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"      Saved: {sass_file}")
        else:
            print(f"[-] cuobjdump failed: {err}", file=sys.stderr)

    # 3. Static SASS Analysis
    if sass_file.exists():
        print("[3/4] Performing static SASS architecture analysis...")
        analyze_script = TOOLS_DIR / "analyze_sass.py"
        analysis_json = exp_dir / "analysis.json"
        code, out, err = run_cmd([sys.executable, str(analyze_script), str(sass_file), "-o", str(analysis_json)])
        if code == 0:
            print(f"      Saved: {analysis_json}")

    # 4. Hardware Execution on Physical GPU
    print("[4/4] Executing on Physical Silicon via NVIDIA Driver API...")
    expected_map = {
        "001_add_one": 2.0,
        "002_double_add": 4.0,
        "003_arithmetic": 18.0,
        "004_register_pressure": 152.0,
        "005_register_spill": 72.0,
        "006_memory_strided": 2.0
    }
    exp_val = expected_map.get(exp_name, 2.0)
    exec_data = execute_cubin_on_gpu(cubin_file, expected_val=exp_val)

    if "error" not in exec_data:
        exec_json = exp_dir / "execution.json"
        with open(exec_json, "w", encoding="utf-8") as f:
            json.dump(exec_data, f, indent=2)
        metadata["execution_success"] = True
        print(f"      GPU Device:   {exec_data.get('device')}")
        print(f"      Kernel Time:  {exec_data.get('kernel_time_ms')} ms")
        print(f"      First Value:  {exec_data.get('first_element_output')} (Expected: {exp_val})")
        print(f"      Verification: {exec_data.get('verified')}")
    else:
        print(f"[-] Execution error: {exec_data['error']}", file=sys.stderr)

    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 65)
    print(f"[SUCCESS] Experiment Complete. Artifacts in: {exp_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Automated GPU Experiment Runner")
    parser.add_argument("source", help="Path to experiment source (.ptx or .cu)")
    parser.add_argument("--arch", default="sm_75", help="Target GPU architecture (default: sm_75)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing experiment results")
    args = parser.parse_args()

    run_experiment(Path(args.source), arch=args.arch, force=args.force)


if __name__ == "__main__":
    main()
