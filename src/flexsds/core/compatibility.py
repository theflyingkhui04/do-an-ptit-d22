"""Compatibility and pre-flight safety guard module for FlexSDS."""
from __future__ import annotations

import json
import os
import re
import struct
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_CACHE_DIR: Path = Path.home() / ".flexsds" / "cache" / "models"


class ModelMetadataError(Exception):
    """Ngoại lệ cơ sở cho các lỗi đọc hoặc xử lý metadata của mô hình."""
    pass


class NetworkOfflineError(ModelMetadataError):
    """Ngoại lệ khi không thể kết nối mạng và không tìm thấy cấu hình trong cache."""
    pass


@dataclass
class PreDownloadCheckResult:
    """Kết quả kiểm tra trước khi tải mô hình từ Hugging Face."""
    is_safe: bool
    vram_needed_mb: float
    free_vram_mb: float
    warning: Optional[str] = None
    suggested_model: Optional[str] = None
    shared_tokenizer: bool = False


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

    method_label: str = "Unknown"
    method_family: str = "Unknown"
    identified_method: str = "unknown"
    detection_evidence: str = ""
    method_supported: bool = False
    checkpoint_verified: bool = False
    runtime_policy: str = "Not configured"
    eagle_version: str = "Unknown"
    vocabulary_status: str = "Unknown"
    runtime_overhead_mb: float = 0.0
    target_revision: str = ""
    draft_revision: str = ""
    blocking_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "COMPATIBLE" if self.is_compatible and not self.oom_risk else "WARNING/INCOMPATIBLE"
        memory = (f"{self.total_vram_needed_mb / 1024:.1f} GiB"
                  if self.total_vram_needed_mb else "Not evaluated")
        return (
            f"Compatibility: {status} | Speculation: {self.method_label} | Family: {self.method_family} | "
            f"VRAM estimate: {memory} / Budget: {self.free_vram_mb / 1024:.1f} GiB | "
            f"Vocabulary: {self.vocabulary_status}"
        )


def get_model_config(
    model_id_or_path: str,
    cache_dir: Optional[Path] = None,
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """Đọc tệp config.json của mô hình từ đường dẫn cục bộ, cache hoặc Hugging Face Hub.

    Args:
        model_id_or_path: Đường dẫn thư mục/tệp cục bộ hoặc Hugging Face model ID.
        cache_dir: Thư mục lưu cache cục bộ (mặc định ~/.flexsds/cache/models).
        timeout_sec: Thời gian chờ tối đa khi tải qua HTTP (giây).

    Returns:
        Dict chứa nội dung tệp config.json.

    Raises:
        NetworkOfflineError: Khi không thể kết nối mạng và chưa có dữ liệu trong cache.
        ModelMetadataError: Khi tệp config.json không tồn tại cục bộ hoặc không hợp lệ.
    """
    # 1. Kiểm tra nếu là đường dẫn cục bộ
    expanded_path = Path(os.path.expanduser(model_id_or_path))
    if expanded_path.exists():
        if expanded_path.is_file() and expanded_path.suffix == ".json":
            config_file = expanded_path
        else:
            config_file = expanded_path / "config.json"

        if config_file.exists() and config_file.is_file():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                raise ModelMetadataError(f"Cannot read local config file '{config_file}': {exc}") from exc
        else:
            raise ModelMetadataError(f"Config file config.json not found in local path '{model_id_or_path}'")

    # 2. Chuẩn hóa định danh mô hình từ xa và xác định thư mục cache
    clean_id = model_id_or_path.strip().replace("https://huggingface.co/", "").strip("/")
    effective_cache_dir = Path(cache_dir) if cache_dir is not None else MODEL_CACHE_DIR
    model_dir_name = clean_id.replace("/", "--")
    cached_file = effective_cache_dir / model_dir_name / "config.json"

    # 3. Đọc từ cache cục bộ nếu đã tồn tại
    if cached_file.exists() and cached_file.is_file():
        try:
            with open(cached_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Nếu cache bị hỏng, tiếp tục thử tải từ mạng
            pass

    # 4. Tải từ Hugging Face qua HTTP
    url = f"https://huggingface.co/{clean_id}/raw/main/config.json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FlexSDS-PreFlightGuard/0.1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()
            content = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
            data = json.loads(content)
    except Exception as exc:
        raise NetworkOfflineError(
            f"Cannot fetch metadata for model '{clean_id}' from Hugging Face Hub: {exc}"
        ) from exc

    # 5. Lưu vào cache cục bộ cho các lần kiểm tra sau
    try:
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text(content, encoding="utf-8")
    except Exception:
        # Lỗi lưu cache không cản trở việc trả về dữ liệu đã tải thành công
        pass

    return data


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
    """Legacy rough download estimate; never authorizes serving or vocabulary alignment."""
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
    draft_weights_mb = estimate_model_vram_mb(draft_model, default_quant="fp16")
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

    # This legacy estimate does not inspect token IDs and cannot establish sharing.
    shared_tokenizer = False

    return PreDownloadCheckResult(
        is_safe=is_safe,
        vram_needed_mb=total_needed_mb,
        free_vram_mb=free_vram_mb,
        warning=warning,
        suggested_model=suggested_model,
        shared_tokenizer=shared_tokenizer,
    )



def estimate_model_vram_mb(model_name: str, default_quant: str = "awq_marlin") -> float:
    """Legacy name-based estimate; serving uses inspected checkpoint sizes instead."""
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


def _validate_vocab_mapping(files, tensors, draft_vocab, target_vocab):
    """Native EAGLE d2t stores offsets: target_id = draft_id + d2t[draft_id]."""
    mapping = next((v for k, v in tensors.items() if k in {"d2t", "draft_id_to_target_id"}), None)
    if not mapping or mapping.get("dtype") != "I64" or mapping.get("shape") != [draft_vocab]:
        raise ValueError("Reduced EAGLE vocabulary requires a checkpoint d2t I64 mapping")
    raw = files.tensor_bytes(mapping)
    if len(raw) != draft_vocab * 8:
        raise ValueError("Invalid d2t mapping byte length")
    ids = [i + offset[0] for i, offset in enumerate(struct.iter_unpack("<q", raw))]
    if len(set(ids)) != draft_vocab or any(i < 0 or i >= target_vocab for i in ids):
        raise ValueError("d2t contains duplicate or out-of-range target token IDs")
    mask = tensors.get("t2d")
    if mask:
        if mask.get("dtype") != "BOOL" or mask.get("shape") != [target_vocab]:
            raise ValueError("Invalid t2d vocabulary mask shape or dtype")
        mask_bytes = files.tensor_bytes(mask)
        if len(mask_bytes) != target_vocab or any(b not in (0, 1) for b in mask_bytes):
            raise ValueError("Invalid t2d vocabulary mask bytes")
        selected = {i for i, enabled in enumerate(mask_bytes) if enabled}
        if selected != set(ids):
            raise ValueError("d2t and t2d vocabulary mappings disagree")


def _kv_cache_mb(config, max_num_seqs, max_model_len):
    layers = int(config["num_hidden_layers"])
    heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // int(config["num_attention_heads"]))
    if min(layers, heads, head_dim, max_num_seqs, max_model_len) <= 0:
        raise ValueError("Invalid model dimensions or serving limits")
    # K and V, BF16/FP16, full configured concurrent context capacity (not measured).
    return 2 * layers * heads * head_dim * 2 * max_num_seqs * max_model_len / 1024**2


def check_compatibility(
    target_model: str,
    draft_model: str,
    free_vram_mb: float = 23000.0,
    target_quant: str = "fp16",
    max_num_seqs: int = 32,
    max_model_len: int = 4096,
) -> CompatibilityReport:
    """Inspect real snapshot artifacts; unknown metadata never implies compatibility.

    Weight storage sizes and configured KV capacity inform an estimate. GPU
    allocation, kernels and a successful inference still require runtime checks.
    target_quant is retained for API compatibility; the checkpoint determines size.
    """
    from flexsds.core.model_inspection import ModelFiles
    report = CompatibilityReport(
        target_model, draft_model, "unsupported", False,
        0.0, 0.0, 0.0, 0.0, free_vram_mb, False, False,
    )
    def block(message):
        report.blocking_reasons.append(message)
        report.warnings.append(message)

    try:
        draft = ModelFiles(draft_model)
        draft_cfg = draft.config()
        report.draft_revision = draft.revision
        from flexsds.core.speculative_method import detect_method, needs_checkpoint, method_support
        detection = detect_method(draft_cfg)
        tensors = draft.tensor_headers() if needs_checkpoint(draft_cfg, detection) else {}
        if needs_checkpoint(draft_cfg, detection):
            detection = detect_method(draft_cfg, tensors)
        report.checkpoint_verified = detection.verified
        report.method_label = detection.label
        report.method_family = detection.family_label
        report.identified_method = detection.method
        report.detection_evidence = detection.evidence
        report.eagle_version = detection.label if detection.family == "feature_ar" else "Non-EAGLE"
        supported, reason = method_support(detection, draft_cfg)
        if not supported:
            block(reason)
            return report
        report.method_supported = True
        report.draft_method = detection.backend_method
        report.runtime_policy = "Sequential EAGLE drafting (vLLM method=" + report.draft_method + ")"
        if report.draft_method == "eagle":
            report.warnings.append("EAGLE-1/2 share a checkpoint format. The vLLM eagle path uses sequential "
                                   "drafting; checkpoint recognition does not enable EAGLE-2 confidence-adaptive trees.")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        block(f"Cannot verify draft metadata/checkpoint: {exc}")
        return report

    try:
        target = ModelFiles(target_model)
        target_cfg = target.config()
        report.target_revision = target.revision
        target_bases = target.base_models(target_cfg) | {target_model}
        draft_bases = draft.base_models(draft_cfg)
        if not draft_bases or not (target_bases & draft_bases):
            block("Cannot verify trained target/draft pairing: draft base_model metadata must identify "
                  "the target or the target's declared base model.")
        if target_cfg.get("hidden_size") != (draft_cfg.get("target_hidden_size") or draft_cfg.get("hidden_size")):
            block("Target hidden size does not match the EAGLE feature input dimension")

        target_vocab = int(target_cfg["vocab_size"])
        draft_input_vocab = int(draft_cfg["vocab_size"])
        draft_output_vocab = int(draft_cfg.get("draft_vocab_size", draft_input_vocab))
        if target_vocab != draft_input_vocab:
            block("Not supported yet: different target/draft input vocabularies; "
                  "a heterogeneous-vocab flag cannot align arbitrary tokenizers")
            report.vocabulary_status = "Mismatch"
        else:
            if 'tokenizer.json' in draft.files():
                shared = target.tokenizer_signature() == draft.tokenizer_signature()
                evidence = "tokenizer artifacts verified"
            elif target_bases & draft_bases:
                # Heads commonly omit a tokenizer and inherit the trained target's.
                base_id = next(iter(target_bases & draft_bases))
                base = target if base_id == target_model else ModelFiles(base_id)
                shared = target.tokenizer_signature() == base.tokenizer_signature()
                evidence = "inherited base tokenizer verified"
            else:
                shared, evidence = False, "no tokenizer evidence"
            if not shared:
                report.vocabulary_status = "Mismatch / unverified"
                block("Not supported yet: tokenizer token IDs or processing differ, or cannot be verified. "
                      "Equal vocab_size does not establish a shared vocabulary.")
            elif draft_output_vocab == target_vocab:
                report.vocabulary_status = f"Shared ({evidence})"
            elif 0 < draft_output_vocab < target_vocab:
                if report.draft_method != "eagle3":
                    raise ValueError("Reduced output vocabulary mapping requires the EAGLE-3/3.1 loader")
                _validate_vocab_mapping(draft, tensors, draft_output_vocab, target_vocab)
                report.vocabulary_status = f"Reduced {draft_output_vocab}/{target_vocab} (native d2t verified; {evidence})"
                report.recommendations.append("Use the checkpoint's native EAGLE d2t mapping; no generic tokenizer conversion is needed.")
            else:
                block("Invalid EAGLE draft output vocabulary size")

        report.target_vram_est_mb = target.weight_bytes() / 1024**2
        report.draft_vram_est_mb = draft.weight_bytes() / 1024**2
        report.kv_cache_min_mb = (_kv_cache_mb(target_cfg, max_num_seqs, max_model_len)
                                  + _kv_cache_mb(draft_cfg, max_num_seqs, max_model_len))
        weights = report.target_vram_est_mb + report.draft_vram_est_mb
        report.runtime_overhead_mb = 1024.0 + weights * 0.10
        report.total_vram_needed_mb = weights + report.kv_cache_min_mb + report.runtime_overhead_mb
        report.oom_risk = report.total_vram_needed_mb > free_vram_mb * 0.95
        report.warnings.append("VRAM is an estimate from checkpoint storage + full configured KV capacity "
                               f"(B={max_num_seqs}, context={max_model_len}) + 10% weights/1 GiB runtime reserve; "
                               "actual GPU memory and inference are not verified by preflight.")
        if report.oom_risk:
            report.recommendations.append("Reduce max_num_seqs/max_model_len or use a compatible quantized target checkpoint.")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        block(f"Cannot verify model pairing, vocabulary or VRAM metadata: {exc}")
    report.is_compatible = not report.blocking_reasons
    return report


class ServingBlockedError(RuntimeError):
    """A preflight decision prevented any serving action."""


@dataclass
class ServingPreflight:
    report: CompatibilityReport
    download: PreDownloadCheckResult
    reasons: List[str]

    @property
    def is_safe(self) -> bool:
        return not self.reasons

    def require_safe(self, allow_oom_risk: bool = False) -> None:
        reasons = self.launch_reasons(allow_oom_risk)
        if reasons:
            raise ServingBlockedError("Launch blocked: " + "; ".join(reasons))

    def launch_reasons(self, allow_oom_risk: bool = False) -> List[str]:
        # The memory estimate is the only overrideable preflight condition.
        return [reason for reason in self.reasons
                if not (allow_oom_risk and reason == OOM_ESTIMATE_REASON)]


OOM_ESTIMATE_REASON = "estimated VRAM exceeds 95% of the available memory budget"


def evaluate_serving_preflight(target_model: str, draft_model: str, hw_info,
                               backend: str = "local", target_gpu: str = "L4",
                               max_num_seqs: int = 32, max_model_len: int = 4096) -> ServingPreflight:
    """One metadata preflight policy shared by flag mode and wizard; no full weights.

    Remote GPU budgets are conservative estimates, never local free VRAM or a
    measurement of the eventual container. Runtime memory fit remains unproven.
    """
    if any(type(value) is not int or value <= 0 for value in (max_num_seqs, max_model_len)):
        raise ValueError("Serving batch/context limits must be positive integers")
    budgets = {"T4": 15000.0, "L4": 23000.0, "A10G": 22000.0}
    available = hw_info.free_vram_mb if backend == "local" else budgets.get(target_gpu, 0.0)
    report = check_compatibility(target_model, draft_model, free_vram_mb=available,
                                 max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    # One estimate and one policy; do not run a second name-based VRAM guard.
    download = PreDownloadCheckResult(
        is_safe=report.is_compatible and not report.oom_risk,
        vram_needed_mb=report.total_vram_needed_mb, free_vram_mb=available,
        shared_tokenizer=report.vocabulary_status.startswith(("Shared", "Reduced")),
    )
    reasons = list(report.blocking_reasons)
    if not report.is_compatible and not reasons:
        reasons.append("incompatible target/draft configuration")
    if report.oom_risk:
        reasons.append(OOM_ESTIMATE_REASON)

    if backend == "local" and not hw_info.is_cuda_available:
        reasons.append("CUDA GPU unavailable; system RAM is not serving VRAM")
    return ServingPreflight(report, download, reasons)


def select_report_eagle_version(report, version):
    """Set the displayed version while preserving all pairing and VRAM decisions."""
    from dataclasses import replace
    from flexsds.core.speculative_method import MethodDetection, select_eagle_version
    detection = MethodDetection(report.identified_method, report.method_label, 'feature_ar',
                                report.detection_evidence, verified=report.checkpoint_verified)
    selected = select_eagle_version(detection, version)
    return replace(report, identified_method=selected.method, method_label=selected.label,
                   eagle_version=selected.label, detection_evidence=selected.evidence)
