#!/usr/bin/env python3
"""
tools/setup_cuda_toolkit.py
Consolidates extracted CUDA packages into a clean, standalone cuda_toolkit/ directory.
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "cuda_toolkit"


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        dest_dir = dst / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_file = Path(root) / f
            dst_file = dest_dir / f
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)


def main():
    print("[*] Consolidating CUDA Toolkit components into:", TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)

    sources = [
        ROOT / "cuda_nvcc" / "nvcc",
        ROOT / "cuda_cudart" / "cudart",
        ROOT / "cuda_cuobjdump" / "cuobjdump",
        ROOT / "cuda_cccl" / "thrust"
    ]

    for src in sources:
        if src.exists():
            print(f" -> Merging {src.name} from {src.parent.name}...")
            copy_tree(src, TARGET)

    # Clean up extraction directories to save space
    temp_dirs = ["cuda_nvcc", "cuda_cudart", "cuda_cuobjdump", "cuda_cccl"]
    for td in temp_dirs:
        p = ROOT / td
        if p.exists():
            print(f" -> Removing temporary directory {td}...")
            shutil.rmtree(p, ignore_errors=True)

    print("\n[✓] CUDA Toolkit assembled successfully at:", TARGET)
    nvcc = TARGET / "bin" / "nvcc.exe"
    cuobjdump = TARGET / "bin" / "cuobjdump.exe"
    cudart = TARGET / "bin" / "cudart64_12.dll"
    print(f"    nvcc:       {nvcc.exists()} ({nvcc})")
    print(f"    cuobjdump:  {cuobjdump.exists()} ({cuobjdump})")
    print(f"    cudart dll: {cudart.exists()} ({cudart})")


if __name__ == "__main__":
    main()
