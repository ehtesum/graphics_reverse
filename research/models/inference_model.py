#!/usr/bin/env python3
"""
research/models/inference_model.py
Phase 4: LLM Inference Analytical Performance Model & Offload Sweep
  - Architecture-exact KV-cache formulas
  - Memory-bound token generation rate (autoregressive decoding)
  - Compute-bound prompt prefill rate
  - Layer-by-layer offload sweep from 0% GPU to 100% GPU
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LLMArchitecture:
    name: str
    param_count_b: float
    layers: int
    heads: int
    kv_heads: int
    dim: int
    hidden_dim: int

    def kv_bytes_per_token(self) -> float:
        """
        Architecture exact KV cache bytes per token:
        2 (K and V) * num_layers * num_kv_heads * head_dim * 2 bytes (FP16)
        """
        head_dim = self.dim // self.heads
        return 2 * self.layers * self.kv_heads * head_dim * 2.0


MODELS = {
    "Gemma-2-2B": LLMArchitecture("Gemma-2-2B", 2.6, 26, 8, 4, 2048, 8192),
    "Phi-3-Mini-3.8B": LLMArchitecture("Phi-3-Mini-3.8B", 3.8, 32, 32, 32, 3072, 8192),
    "Mistral-7B-v0.3": LLMArchitecture("Mistral-7B-v0.3", 7.2, 32, 32, 8, 4096, 14336),
    "Llama-3-8B": LLMArchitecture("Llama-3-8B", 8.0, 32, 32, 8, 4096, 14336)
}

QUANT_BITS = {
    "FP16": 16.0,
    "INT8": 8.0,
    "Q4_K_M": 4.5,
    "Q2_K": 2.5
}


@dataclass
class LayerOffloadPoint:
    gpu_layers: int
    cpu_layers: int
    gpu_vram_mb: float
    cpu_ram_mb: float
    tok_per_sec: float
    pcie_mb_per_sec: float
    is_vram_feasible: bool


class AnalyticalInferenceModel:
    def __init__(
        self,
        vram_limit_mb: float = 3446.0,
        gpu_bw_gbs: float = 136.0,
        cpu_bw_gbs: float = 32.0,
        pcie_bw_gbs: float = 6.68,
        launch_overhead_us: float = 8.5
    ):
        self.vram_limit_mb = vram_limit_mb
        self.gpu_bw_gbs = gpu_bw_gbs
        self.cpu_bw_gbs = cpu_bw_gbs
        self.pcie_bw_gbs = pcie_bw_gbs
        self.launch_overhead_us = launch_overhead_us

    def evaluate_model(
        self,
        arch: LLMArchitecture,
        quant: str,
        context_len: int
    ) -> Dict:
        bits = QUANT_BITS.get(quant, 4.5)
        bytes_per_param = bits / 8.0
        weights_mb = (arch.param_count_b * 1e9 * bytes_per_param) / (1024 * 1024)
        kv_cache_mb = (arch.kv_bytes_per_token() * context_len) / (1024 * 1024)
        layer_weight_mb = weights_mb / arch.layers

        # Sweep GPU layer count from 0 to total layers
        sweep = []
        for g_layers in range(arch.layers + 1):
            c_layers = arch.layers - g_layers
            req_vram = (g_layers * layer_weight_mb) + (kv_cache_mb if g_layers > 0 else 0)
            req_ram = c_layers * layer_weight_mb
            is_feasible = (req_vram <= self.vram_limit_mb)

            # Memory-bound token generation latency (seconds per token)
            t_gpu_mem = (g_layers * layer_weight_mb * 1e6) / (self.gpu_bw_gbs * 1e9)
            t_cpu_mem = (c_layers * layer_weight_mb * 1e6) / (self.cpu_bw_gbs * 1e9)

            # Inter-layer activation transfer across PCIe (dim * 2 bytes FP16)
            act_bytes = arch.dim * 2.0
            t_pcie = (act_bytes * 2.0) / (self.pcie_bw_gbs * 1e9) if (g_layers > 0 and c_layers > 0) else 0.0
            t_launch = (g_layers * self.launch_overhead_us) / 1e6

            t_token = t_gpu_mem + t_cpu_mem + t_pcie + t_launch
            tok_s = (1.0 / t_token) if t_token > 0 else 0.0
            pcie_traffic_mb_s = (act_bytes * 2.0 * tok_s) / (1024 * 1024) if (g_layers > 0 and c_layers > 0) else 0.0

            sweep.append(LayerOffloadPoint(
                gpu_layers=g_layers,
                cpu_layers=c_layers,
                gpu_vram_mb=round(req_vram, 1),
                cpu_ram_mb=round(req_ram, 1),
                tok_per_sec=round(tok_s, 2),
                pcie_mb_per_sec=round(pcie_traffic_mb_s, 2),
                is_vram_feasible=is_feasible
            ))

        # Optimal feasible configuration
        feasible_points = [p for p in sweep if p.is_vram_feasible]
        best_point = max(feasible_points, key=lambda x: x.tok_per_sec) if feasible_points else sweep[0]

        return {
            "model_name": arch.name,
            "quantization": quant,
            "context_length": context_len,
            "total_weights_mb": round(weights_mb, 1),
            "kv_cache_mb": round(kv_cache_mb, 1),
            "best_feasible_gpu_layers": best_point.gpu_layers,
            "max_feasible_tok_s": best_point.tok_per_sec,
            "sweep_curve": [asdict(p) for p in sweep]
        }


def main():
    model = AnalyticalInferenceModel()

    print("=" * 80)
    print(" Analytical LLM Inference Model & Layer Offload Sweep")
    print(" Grounded on 3,446 MB Usable VRAM, 136 GB/s GDDR6, 6.68 GB/s PCIe 3.0")
    print("=" * 80)

    results = {}
    for m_name, arch in MODELS.items():
        res = model.evaluate_model(arch, "Q4_K_M", 2048)
        results[m_name] = res
        print(f"Model: {m_name:<16} | Weight: {res['total_weights_mb']:<6.0f} MB | KV (2K): {res['kv_cache_mb']:<5.1f} MB | Best GPU Layers: {res['best_feasible_gpu_layers']}/{arch.layers} | Tok/s: {res['max_feasible_tok_s']}")

    out_file = ROOT / "research" / "raw" / "inference_model_sweep.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] Inference model evaluation saved to: {out_file}\n")


if __name__ == "__main__":
    main()
