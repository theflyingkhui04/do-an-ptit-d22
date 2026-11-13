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


def test_reject_non_eagle_draft_architecture(tmp_path):
    # Mock config target là Qwen2.5 và draft là Qwen2.5-0.5B (không phải EAGLE)
    target_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}
    draft_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct",
            draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        )
        assert report.is_compatible is False
        assert any("chỉ hỗ trợ kiến trúc đầu nháp EAGLE" in w for w in report.warnings)


def test_accept_eagle_draft_architecture_and_detect_hetero_vocab(tmp_path):
    # Mock config target 152k và draft EAGLE 16k
    target_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}
    draft_cfg = {"architectures": ["EagleModel"], "vocab_size": 16000}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        )
        assert report.is_compatible is True
        assert report.draft_method == "eagle3"
        assert report.use_heterogeneous_vocab is True


def test_shared_vocab_detection_when_same_vocab_size():
    # Giả lập 2 mô hình LLaMA cùng vocab_size và kiến trúc EAGLE-LLaMA
    target_cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 128256}
    draft_cfg = {"architectures": ["EagleForCausalLM"], "vocab_size": 128256}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="meta-llama/Meta-Llama-3-8B",
            draft_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        )
        assert report.is_compatible is True
        assert report.draft_method == "eagle3"
        # Khi cùng vocab_size, không gian từ vựng dùng chung nên cờ dị thể được tắt (TC-CMP-07)
        assert report.use_heterogeneous_vocab is False


def test_check_compatibility_offline_fallback():
    # Kiểm tra trường hợp mất mạng, fallback về heuristic dựa trên tên mô hình
    with patch("flexsds.core.compatibility.get_model_config", side_effect=NetworkOfflineError("Network unreachable")):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        )
        assert report.is_compatible is True
        assert report.draft_method == "eagle3"
        assert report.use_heterogeneous_vocab is True


def test_check_compatibility_offline_fallback_rejects_non_eagle():
    # Kiểm tra trường hợp mất mạng với mô hình nháp không phải EAGLE
    with patch("flexsds.core.compatibility.get_model_config", side_effect=NetworkOfflineError("Network unreachable")):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        )
        assert report.is_compatible is False
        assert any("chỉ hỗ trợ kiến trúc đầu nháp EAGLE" in w for w in report.warnings)

