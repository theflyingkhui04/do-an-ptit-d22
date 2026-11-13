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
