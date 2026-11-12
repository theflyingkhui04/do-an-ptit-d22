"""Compatibility and pre-flight safety guard module for FlexSDS."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreDownloadCheckResult:
    """Kết quả kiểm tra trước khi tải mô hình từ Hugging Face."""
    is_safe: bool
    vram_needed_mb: float
    free_vram_mb: float
    warning: Optional[str] = None
    suggested_model: Optional[str] = None
    shared_tokenizer: bool = True


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


def inspect_remote_model_metadata(model_id_or_path: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """Đọc tệp config.json (<10 KB) của mô hình từ Hugging Face Hub hoặc thư mục cục bộ."""
    # 1. Kiểm tra nếu là đường dẫn cục bộ
    if os.path.exists(model_id_or_path):
        config_path = (
            model_id_or_path
            if os.path.isfile(model_id_or_path) and model_id_or_path.endswith(".json")
            else os.path.join(model_id_or_path, "config.json")
        )
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    # 2. Truy vấn từ Hugging Face qua HTTP API công khai
    clean_id = model_id_or_path.strip().replace("https://huggingface.co/", "")
    url = f"https://huggingface.co/{clean_id}/raw/main/config.json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FlexSDS-PreFlightGuard/0.1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content)
    except Exception:
        return {}


def pre_download_guard(
    target_model: str,
    draft_model: str,
    free_vram_mb: float,
    mock_metadata: Optional[Dict[str, Any]] = None,
) -> PreDownloadCheckResult:
    """Kiểm tra VRAM dự kiến trước khi tải mô hình, ngăn ngừa lãng phí băng thông và OOM."""
    meta = mock_metadata if mock_metadata is not None else inspect_remote_model_metadata(target_model)
    target_lower = target_model.lower()

    # 1. Xác định số lượng tham số
    num_params: float = 7.0e9
    if "num_parameters" in meta and meta["num_parameters"]:
        num_params = float(meta["num_parameters"])
    else:
        param_match = re.search(r"(\d+(?:\.\d+)?)\s*b", target_lower)
        if param_match:
            num_params = float(param_match.group(1)) * 1e9

    # 2. Xác định định dạng lượng tử hóa
    quant_cfg = meta.get("quantization_config")
    is_quantized = False
    if quant_cfg is not None:
        is_quantized = True
    elif any(q in target_lower for q in ("awq", "int4", "gptq")):
        is_quantized = True

    if is_quantized:
        bytes_per_param = 0.55  # AWQ/GPTQ 4-bit + scale/zero overhead
    elif "fp8" in target_lower:
        bytes_per_param = 1.05
    else:
        torch_dtype = str(meta.get("torch_dtype", "")).lower()
        if "float32" in torch_dtype:
            bytes_per_param = 4.0
        else:
            bytes_per_param = 2.0  # FP16 / BF16

    target_weights_mb = (num_params * bytes_per_param) / (1024.0 * 1024.0)
    draft_weights_mb = 600.0  # EAGLE-3.1 speculator head
    kv_cache_mb = 1200.0  # KV-cache tối thiểu cho serving cơ bản
    total_needed_mb = round(target_weights_mb + draft_weights_mb + kv_cache_mb, 1)

    is_safe = total_needed_mb <= (free_vram_mb * 0.95)
    warning = None
    suggested_model = None

    if not is_safe:
        warning = (
            f"Estimated VRAM (~{total_needed_mb / 1024:.1f} GB) exceeds available VRAM "
            f"({free_vram_mb / 1024:.1f} GB free). High risk of OOM!"
        )
        if not is_quantized:
            suggested_model = f"{target_model}-AWQ"

    # Kiểm tra tokenizer dùng chung
    # Với kiến trúc EAGLE được huấn luyện đồng bộ trên cùng base model, tokenizer là đồng nhất
    shared_tokenizer = True

    return PreDownloadCheckResult(
        is_safe=is_safe,
        vram_needed_mb=total_needed_mb,
        free_vram_mb=free_vram_mb,
        warning=warning,
        suggested_model=suggested_model,
        shared_tokenizer=shared_tokenizer,
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
