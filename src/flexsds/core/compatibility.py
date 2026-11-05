"""Compatibility and pre-flight safety guard module for FlexSDS."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompatibilityReport:
    """Pre-flight evaluation report for target and draft models."""
    target_model: str
    draft_model: str
    draft_method: str
    use_heterogeneous_vocab: bool
    target_vram_est_mb: float
    draft_vram_est_mb: float
    kv_cache_min_mb: float
    total_vram_needed_mb: float
    free_vram_mb: float
    oom_risk: bool
    is_compatible: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "COMPATIBLE" if self.is_compatible and not self.oom_risk else "WARNING/INCOMPATIBLE"
        return (
            f"Compatibility: {status} | Method: {self.draft_method} | "
            f"VRAM Needed: {self.total_vram_needed_mb / 1024:.1f}GB / Free: {self.free_vram_mb / 1024:.1f}GB | "
            f"Hetero Vocab: {self.use_heterogeneous_vocab}"
        )


def estimate_model_vram_mb(model_name: str, default_quant: str = "awq_marlin") -> float:
    """Estimates required VRAM in MB based on model name heuristics."""
    name_lower = model_name.lower()

    # Speculator Head (EAGLE-3 / EAGLE-3.1 head) is a lightweight 1-layer decoder
    if "eagle" in name_lower or "speculator" in name_lower or "head" in name_lower:
        return 450.0

    # Extract parameter count (e.g., 7b, 0.5b, 14b, 72b, 8b)
    param_match = re.search(r"(\d+(?:\.\d+)?)\s*b", name_lower)
    params_billion = float(param_match.group(1)) if param_match else 7.0

    # Determine quantization precision
    if "awq" in name_lower or "int4" in name_lower or "gptq" in name_lower or default_quant == "awq_marlin":
        # 4-bit quantization: ~0.6 GB per billion params + overhead
        return round(params_billion * 600.0 + 500.0, 1)
    elif "fp8" in name_lower:
        # 8-bit quantization: ~1.1 GB per billion params
        return round(params_billion * 1100.0 + 500.0, 1)
    else:
        # Default FP16 / BF16: ~2.1 GB per billion params
        return round(params_billion * 2100.0 + 800.0, 1)


def check_compatibility(
    target_model: str,
    draft_model: str,
    free_vram_mb: float = 23000.0,
    target_quant: str = "awq_marlin",
    max_num_seqs: int = 32,
    max_model_len: int = 4096,
) -> CompatibilityReport:
    """Runs pre-flight compatibility and VRAM estimation checks."""
    warnings: List[str] = []
    recommendations: List[str] = []

    target_lower = target_model.lower()
    draft_lower = draft_model.lower()

    # 1. Determine Draft Method
    is_eagle = "eagle" in draft_lower or "speculator" in draft_lower
    draft_method = "eagle3" if is_eagle else "draft_model"

    # 2. Estimate VRAM Consumption
    target_vram = estimate_model_vram_mb(target_model, default_quant=target_quant)
    draft_vram = estimate_model_vram_mb(draft_model, default_quant="fp16")
    
    # KV cache minimum allocation for serving concurrency
    # Rough estimate: 2 * num_layers * hidden_size * kv_heads * seq_len * num_seqs * dtype_size
    kv_cache_min = min(3500.0, max(1500.0, max_num_seqs * max_model_len * 0.02))
    
    total_needed = target_vram + draft_vram + kv_cache_min

    # 3. Check VRAM headroom and OOM risk
    oom_risk = total_needed > (free_vram_mb * 0.95)
    if oom_risk:
        shortfall = total_needed - free_vram_mb
        warnings.append(
            f"VRAM shortfall detected: Required ~{total_needed/1024:.1f} GB, but only {free_vram_mb/1024:.1f} GB free."
        )
        recommendations.append(
            "Use an AWQ/GPTQ 4-bit quantized target model or reduce --max-num-seqs to free KV cache."
        )

    # 4. Vocabulary Alignment
    # Check if target and draft are from different model families
    target_family = "qwen" if "qwen" in target_lower else ("llama" if "llama" in target_lower else "other")
    draft_family = "qwen" if "qwen" in draft_lower else ("llama" if "llama" in draft_lower else "other")
    
    use_hetero_vocab = False
    if target_family != draft_family or is_eagle or "0.5b" in draft_lower:
        # vLLM requires heterogeneous vocab flag when tokenizers differ in vocabulary sizes
        use_hetero_vocab = True
        recommendations.append(
            "Heterogeneous vocabulary flag enabled (--use-heterogeneous-vocab) for safe draft alignment."
        )

    # 5. Architecture Family compatibility
    is_compatible = True
    if target_family != "other" and draft_family != "other" and target_family != draft_family:
        warnings.append(
            f"Cross-family pairing ({target_family} target with {draft_family} draft) may exhibit low acceptance rate."
        )

    return CompatibilityReport(
        target_model=target_model,
        draft_model=draft_model,
        draft_method=draft_method,
        use_heterogeneous_vocab=use_hetero_vocab,
        target_vram_est_mb=target_vram,
        draft_vram_est_mb=draft_vram,
        kv_cache_min_mb=kv_cache_min,
        total_vram_needed_mb=total_needed,
        free_vram_mb=free_vram_mb,
        oom_risk=oom_risk,
        is_compatible=is_compatible,
        warnings=warnings,
        recommendations=recommendations,
    )
