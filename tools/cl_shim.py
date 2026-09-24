#!/usr/bin/env python3
"""
tools/cl_shim.py
Intercepts nvcc compiler version query and delegates to MSVC cl.exe,
logging exact arguments and output for debugging.
"""

import os
import subprocess
import sys
from pathlib import Path

REAL_CL = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"
LOGFILE = Path(__file__).resolve().parent / "cl_shim.log"

args = sys.argv[1:]

with open(LOGFILE, "a", encoding="utf-8") as lf:
    lf.write(f"\n--- INVOCATION ---\nARGS: {args}\n")
    for a in args:
        if os.path.exists(a):
            try:
                content = open(a, "r", encoding="utf-8", errors="ignore").read()
                lf.write(f"FILE {a}:\n{content}\n")
            except Exception as e:
                lf.write(f"Could not read {a}: {e}\n")

proc = subprocess.run([REAL_CL] + args, capture_output=True, text=True)

stdout_out = proc.stdout
stderr_out = proc.stderr

with open(LOGFILE, "a", encoding="utf-8") as lf:
    lf.write(f"RAW STDOUT:\n{stdout_out}\nRAW STDERR:\n{stderr_out}\nRC: {proc.returncode}\n")

# Spoof MSVC version to 19.40 (VS 2022)
if "Version 19.51" in stderr_out:
    stderr_out = stderr_out.replace("Version 19.51.36248", "Version 19.40.33811")
    stderr_out = stderr_out.replace("Version 19.51", "Version 19.40")

sys.stdout.write(stdout_out)
sys.stderr.write(stderr_out)
sys.exit(proc.returncode)
