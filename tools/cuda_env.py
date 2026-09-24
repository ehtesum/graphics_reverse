"""
tools/cuda_env.py - Centralized CUDA toolchain discovery.
Dynamically resolves ptxas, cuobjdump, and nvdisasm across Windows and Linux.
"""
import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def find_cuda_binary(name: str) -> Path:
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    
    # 1. Check system PATH
    sys_path = shutil.which(exe_name) or shutil.which(name)
    if sys_path:
        return Path(sys_path)
    
    # 2. Check CUDA_PATH or CUDA_HOME environment variables
    for env_var in ["CUDA_PATH", "CUDA_HOME"]:
        env_dir = os.environ.get(env_var)
        if env_dir:
            cand = Path(env_dir) / "bin" / exe_name
            if cand.exists():
                return cand
                
    # 3. Check local cuda_toolkit directory if present
    local_cand = ROOT / "cuda_toolkit" / "bin" / exe_name
    if local_cand.exists():
        return local_cand
        
    # 4. Standard default installation paths
    if sys.platform == "win32":
        win_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if win_base.exists():
            for v_dir in sorted(win_base.iterdir(), reverse=True):
                cand = v_dir / "bin" / exe_name
                if cand.exists():
                    return cand
    else:
        for p in ["/usr/local/cuda/bin", "/opt/cuda/bin"]:
            cand = Path(p) / exe_name
            if cand.exists():
                return cand
                
    return Path(exe_name)

PTXAS = find_cuda_binary("ptxas")
CUOBJDUMP = find_cuda_binary("cuobjdump")
NVDISASM = find_cuda_binary("nvdisasm")
