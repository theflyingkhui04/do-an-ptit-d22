"""Tests for pre-download guard in flexsds.core.compatibility."""
import pytest
from unittest.mock import patch, MagicMock
from flexsds.core.compatibility import (
    pre_download_guard,
    inspect_remote_model_metadata,
    PreDownloadCheckResult,
    check_compatibility,
    NetworkOfflineError,
)


def test_pre_download_guard_detects_oversized_model():
    # Mô phỏng mô hình FP16 15GB trên GPU 8GB
    res = pre_download_guard(
        target_model="Qwen/Qwen2.5-7B-Instruct",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        free_vram_mb=8000.0,
        mock_metadata={"torch_dtype": "float16", "num_parameters": 7.6e9, "quantization_config": None},
    )
    assert res.is_safe is False
    assert res.vram_needed_mb > 14000.0
    assert "AWQ" in (res.suggested_model or "")
    assert res.warning is not None


def test_pre_download_guard_passes_awq():
    res = pre_download_guard(
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        free_vram_mb=8000.0,
        mock_metadata={"torch_dtype": "float16", "num_parameters": 7.6e9, "quantization_config": {"quant_method": "awq"}},
    )
    assert res.is_safe is True
    assert res.vram_needed_mb < 6000.0
    assert res.warning is None


def test_inspect_remote_model_metadata_mock():
    mock_json = b'{"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16", "num_hidden_layers": 28}'
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_json
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        meta = inspect_remote_model_metadata("Qwen/Qwen2.5-7B-Instruct")
        assert meta.get("torch_dtype") == "bfloat16"
        assert meta.get("num_hidden_layers") == 28

