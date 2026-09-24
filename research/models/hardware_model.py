#!/usr/bin/env python3
"""
research/models/hardware_model.py
Phase 4: Analytical Hardware Performance Model
Predicts execution latency for heterogeneous CPU/GPU workloads:
  T_total = T_CPU + T_PCIe + T_GPU + T_memory + T_launch + T_sync + T_KV
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class HardwareConstants:
    # Calibrated from empirical measurements:
    pcie_bw_pinned_gbs: float = 6.68        # Large buffer plateau (EXP-GPU-339)
    pcie_bw_small_gbs: float = 1.10         # Small buffer <64KB
    pcie_dma_latency_us: float = 18.2       # Base initiation overhead
    gpu_gddr6_bw_gbs: float = 136.0         # Empirically measured in Exp 013
    gpu_fp16_tflops: float = 5.393           # Exp 012
    gpu_fp32_tflops: float = 2.441           # Exp 003
    gpu_sfu_ginst_s: float = 390.0          # Exp 014
    cpu_avx2_peak_gflops: float = 598.68    # 8-core sustained (Exp CPU-003)
    cpu_ram_bw_gbs: float = 32.0            # DDR4-3200 dual channel
    wddm_async_launch_us: float = 8.5       # Pushbuffer append (EXP-WDDM-404)
    wddm_sync_roundtrip_us: float = 13.5    # Synchronous wait (EXP-WDDM-404)
    cpu_inter_ccx_jump_ns: float = 37.6     # Infinity Fabric penalty (EXP-CPU-024)


@dataclass
class KernelWorkload:
    name: str
    cpu_flops: float = 0.0
    gpu_flops: float = 0.0
    gpu_precision: str = "FP16"             # "FP16" or "FP32"
    pcie_h2d_bytes: float = 0.0
    pcie_d2h_bytes: float = 0.0
    gpu_mem_bytes: float = 0.0
    num_kernel_launches: int = 1
    num_sync_barriers: int = 1
    is_pinned: bool = True


@dataclass
class LatencyBreakdown:
    t_cpu_ms: float
    t_pcie_ms: float
    t_gpu_compute_ms: float
    t_gpu_mem_ms: float
    t_launch_ms: float
    t_sync_ms: float
    t_total_ms: float
    predicted_throughput_ops_sec: float
    primary_bottleneck: str


class CompositeHardwareModel:
    def __init__(self, hw: HardwareConstants = HardwareConstants()):
        self.hw = hw

    def predict_latency(self, w: KernelWorkload) -> LatencyBreakdown:
        # 1. CPU Compute Time
        t_cpu_s = (w.cpu_flops / (self.hw.cpu_avx2_peak_gflops * 1e9)) if w.cpu_flops > 0 else 0.0

        # 2. PCIe Transfer Time
        total_pcie_bytes = w.pcie_h2d_bytes + w.pcie_d2h_bytes
        if total_pcie_bytes > 0:
            bw = self.hw.pcie_bw_pinned_gbs if w.is_pinned else (self.hw.pcie_bw_pinned_gbs * 0.75)
            # Apply small-buffer scaling if < 64 KB
            if total_pcie_bytes < 64 * 1024:
                bw = self.hw.pcie_bw_small_gbs
            t_pcie_s = (total_pcie_bytes / (bw * 1e9)) + (self.hw.pcie_dma_latency_us / 1e6)
        else:
            t_pcie_s = 0.0

        # 3. GPU Compute Time
        gpu_peak_flops = (self.hw.gpu_fp16_tflops * 1e12) if w.gpu_precision == "FP16" else (self.hw.gpu_fp32_tflops * 1e12)
        t_gpu_compute_s = (w.gpu_flops / gpu_peak_flops) if w.gpu_flops > 0 else 0.0

        # 4. GPU Memory Streaming Time
        t_gpu_mem_s = (w.gpu_mem_bytes / (self.hw.gpu_gddr6_bw_gbs * 1e9)) if w.gpu_mem_bytes > 0 else 0.0

        # 5. Launch and Sync Overhead
        t_launch_s = (w.num_kernel_launches * self.hw.wddm_async_launch_us) / 1e6
        t_sync_s = (w.num_sync_barriers * self.hw.wddm_sync_roundtrip_us) / 1e6

        # Effective GPU time (memory vs compute roofline overlap)
        t_gpu_effective_s = max(t_gpu_compute_s, t_gpu_mem_s)

        # Composite total latency
        t_total_s = t_cpu_s + t_pcie_s + t_gpu_effective_s + t_launch_s + t_sync_s

        # Identify primary bottleneck
        durations = {
            "CPU Compute": t_cpu_s,
            "PCIe Interconnect": t_pcie_s,
            "GPU Compute": t_gpu_compute_s,
            "GPU GDDR6 Memory": t_gpu_mem_s,
            "WDDM Submission / Sync": t_launch_s + t_sync_s
        }
        primary = max(durations.items(), key=lambda x: x[1])[0]

        total_ops = w.cpu_flops + w.gpu_flops
        throughput = (total_ops / t_total_s) if t_total_s > 0 else 0.0

        return LatencyBreakdown(
            t_cpu_ms=round(t_cpu_s * 1000.0, 4),
            t_pcie_ms=round(t_pcie_s * 1000.0, 4),
            t_gpu_compute_ms=round(t_gpu_compute_s * 1000.0, 4),
            t_gpu_mem_ms=round(t_gpu_mem_s * 1000.0, 4),
            t_launch_ms=round(t_launch_s * 1000.0, 4),
            t_sync_ms=round(t_sync_s * 1000.0, 4),
            t_total_ms=round(t_total_s * 1000.0, 4),
            predicted_throughput_ops_sec=round(throughput, 2),
            primary_bottleneck=primary
        )


def main():
    model = CompositeHardwareModel()

    print("=" * 80)
    print(" Composite Hardware Performance Model Validation")
    print(" Grounded on Physical AMD Zen 2 + NVIDIA TU117 Calibration Constants")
    print("=" * 80)

    # Test Workload 1: Small Transformer Layer GEMM (Batch 1, Dim 4096)
    # Flops: 2 * 4096 * 4096 = 33.55 MFLOPs
    w1 = KernelWorkload(
        name="Small Transformer Linear Layer",
        gpu_flops=33.55e6,
        gpu_precision="FP16",
        gpu_mem_bytes=32.0 * 1024 * 1024, # 32 MB weight matrix
        pcie_h2d_bytes=8192,              # 8 KB activation
        pcie_d2h_bytes=8192,
        num_kernel_launches=1,
        num_sync_barriers=1
    )

    pred1 = model.predict_latency(w1)
    print(f"Workload 1: {w1.name}")
    print(f"  Predicted Total Latency: {pred1.t_total_ms:.3f} ms")
    print(f"  Breakdown -> GPU Mem: {pred1.t_gpu_mem_ms:.3f} ms | PCIe: {pred1.t_pcie_ms:.3f} ms | WDDM Sync: {pred1.t_sync_ms:.3f} ms")
    print(f"  Primary Bottleneck:      {pred1.primary_bottleneck}\n")

    # Test Workload 2: CPU Offloaded Layer GEMM (Ran on Zen 2 AVX2)
    w2 = KernelWorkload(
        name="CPU Offloaded Layer GEMM",
        cpu_flops=33.55e6,
        pcie_h2d_bytes=0,
        pcie_d2h_bytes=0,
        num_kernel_launches=0,
        num_sync_barriers=0
    )
    pred2 = model.predict_latency(w2)
    print(f"Workload 2: {w2.name}")
    print(f"  Predicted Total Latency: {pred2.t_total_ms:.3f} ms")
    print(f"  Primary Bottleneck:      {pred2.primary_bottleneck}\n")

    out_file = ROOT / "research" / "raw" / "hardware_model_sample.json"
    with open(out_file, "w") as f:
        json.dump({"workload_1": asdict(pred1), "workload_2": asdict(pred2)}, f, indent=2)
    print(f"[SUCCESS] Model output saved to: {out_file}\n")


if __name__ == "__main__":
    main()
