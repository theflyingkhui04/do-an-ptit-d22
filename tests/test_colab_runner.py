"""Tests for Google Colab serving runner and Cloudflare tunnel verification."""
import pytest
from flexsds.server.colab_runner import ColabServingRunner


def test_colab_runner_instructions_and_link():
    runner = ColabServingRunner()
    info = runner.get_colab_instructions(target_gpu="T4")
    assert "colab.research.google.com" in info["notebook_url"]
    assert "cloudflared" in info["tunnel_method"]
    assert "T4" in info["recommended_runtime"]


def test_colab_runner_verify_tunnel_health_mock(monkeypatch):
    runner = ColabServingRunner()
    monkeypatch.setattr(runner, "_probe_endpoint", lambda url, **kwargs: True)
    assert runner.verify_tunnel_health("https://mock.trycloudflare.com/v1") is True


def test_colab_runner_verify_tunnel_health_failure(monkeypatch):
    runner = ColabServingRunner()
    monkeypatch.setattr(runner, "_probe_endpoint", lambda url, **kwargs: False)
    assert runner.verify_tunnel_health("https://mock.trycloudflare.com/v1") is False


def test_colab_notebook_file_exists():
    import json
    from pathlib import Path
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "flexsds_colab_server.ipynb"
    assert nb_path.exists(), f"Missing notebook file at {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cells" in data
    assert len(data["cells"]) >= 3


def test_colab_runner_sync_notebook_config(tmp_path):
    import json
    runner = ColabServingRunner()
    fake_nb = tmp_path / "test_nb.ipynb"
    fake_nb.write_text(json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    'TARGET_MODEL = "old"  #@param {type:"string"}\n',
                    'DRAFT_MODEL = "old_draft"  #@param {type:"string"}\n',
                    'MAX_NUM_SEQS = 32\n',
                    'MAX_MODEL_LEN = 4096\n',
                ]
            }
        ]
    }), encoding="utf-8")
    cfg = {
        "target_model": "custom/target-model",
        "draft_model": "custom/draft-model",
        "draft_method": "draft_model",
        "max_num_seqs": 4, "max_model_len": 1024,
    }
    success = runner.sync_notebook_config(cfg, notebook_path=str(fake_nb))
    assert success is True
    updated = json.loads(fake_nb.read_text(encoding="utf-8"))
    cell_code = "".join(updated["cells"][0]["source"])
    assert "custom/target-model" in cell_code
    assert "custom/draft-model" in cell_code

    assert "MAX_NUM_SEQS = 4" in cell_code
    assert "MAX_MODEL_LEN = 1024" in cell_code
