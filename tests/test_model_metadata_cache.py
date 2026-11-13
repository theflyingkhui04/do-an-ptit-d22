"""Tests for model metadata caching and offline network handling in FlexSDS."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from flexsds.core.compatibility import (
    get_model_config,
    NetworkOfflineError,
    MODEL_CACHE_DIR,
)


def test_get_model_config_downloads_and_caches(tmp_path):
    mock_json = b'{"architectures": ["EagleModel"], "vocab_size": 16000}'
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_json
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        cfg = get_model_config("test-org/test-eagle-model", cache_dir=tmp_path)
        assert cfg.get("vocab_size") == 16000
        assert "EagleModel" in cfg.get("architectures", [])

        # Kiểm tra file đã được lưu vào cache
        cached_file = tmp_path / "test-org--test-eagle-model" / "config.json"
        assert cached_file.exists()


def test_get_model_config_reads_from_cache_without_network(tmp_path):
    # Tạo sẵn cache giả lập
    cached_dir = tmp_path / "test-org--test-eagle-model"
    cached_dir.mkdir(parents=True)
    cached_file = cached_dir / "config.json"
    cached_file.write_text(json.dumps({"architectures": ["EagleModel"], "vocab_size": 16000}))

    # Không gọi mạng, đọc từ cache
    with patch("urllib.request.urlopen", side_effect=Exception("Should not call network")):
        cfg = get_model_config("test-org/test-eagle-model", cache_dir=tmp_path)
        assert cfg.get("vocab_size") == 16000


def test_get_model_config_offline_no_cache_raises_network_error(tmp_path):
    with patch("urllib.request.urlopen", side_effect=Exception("Network unreachable")):
        with pytest.raises(NetworkOfflineError):
            get_model_config("non-cached/model-id", cache_dir=tmp_path)


def test_get_model_config_reads_local_directory(tmp_path):
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()
    config_path = local_model_dir / "config.json"
    config_path.write_text(json.dumps({"architectures": ["EagleModel"], "vocab_size": 32000}))

    with patch("urllib.request.urlopen", side_effect=Exception("Should not call network")):
        cfg = get_model_config(str(local_model_dir))
        assert cfg.get("vocab_size") == 32000
        assert "EagleModel" in cfg.get("architectures", [])


def test_get_model_config_reads_local_file(tmp_path):
    config_file = tmp_path / "custom_config.json"
    config_file.write_text(json.dumps({"architectures": ["EagleForCausalLM"], "vocab_size": 152064}))

    with patch("urllib.request.urlopen", side_effect=Exception("Should not call network")):
        cfg = get_model_config(str(config_file))
        assert cfg.get("vocab_size") == 152064
