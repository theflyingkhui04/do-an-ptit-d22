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
