"""Tests for interactive CLI chat session and performance metrics."""
import pytest
from unittest.mock import MagicMock
from flexsds.client.chat import InteractiveChatSession, PerformanceMetrics
from flexsds.client.client import FlexSDSClient


def test_performance_metrics_calculation():
    metrics = PerformanceMetrics(total_chunks=100, elapsed_seconds=2.0)
    assert metrics.chunks_per_second == 50.0
    assert metrics.ms_per_chunk == 20.0  # 2000 ms / 100 = 20.0 ms per chunk


def test_chat_session_runs_and_exits(monkeypatch):
    mock_client = MagicMock(spec=FlexSDSClient)
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.check_health.return_value = True
    mock_client.get_models.return_value = ["qwen2.5-7b-adaptive"]

    session = InteractiveChatSession(mock_client, model="qwen2.5-7b-adaptive")

    # Giả lập nhập /exit ngay lập tức
    monkeypatch.setattr("builtins.input", lambda prompt: "/exit")
    exit_code = session.start()
    assert exit_code == 0


def test_chat_session_streaming_turn(monkeypatch, capsys):
    mock_client = MagicMock(spec=FlexSDSClient)
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.check_health.return_value = True
    mock_client.get_models.return_value = ["qwen2.5-7b-adaptive"]
    mock_client.stream_chat.return_value = iter(["Xin", " chào", "!"])

    session = InteractiveChatSession(mock_client, model="qwen2.5-7b-adaptive")

    inputs = iter(["Chào bot", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = session.start()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Xin chào!" in captured.out
    assert "Thống kê suy luận" in captured.out


def test_chat_handles_sigint_gracefully(monkeypatch):
    mock_client = MagicMock(spec=FlexSDSClient)
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.check_health.return_value = True
    mock_client.get_models.return_value = ["qwen2.5-7b-adaptive"]

    session = InteractiveChatSession(mock_client, model="qwen2.5-7b-adaptive")

    # Giả lập ấn Ctrl+C (KeyboardInterrupt)
    def raise_sigint(prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raise_sigint)
    exit_code = session.start()
    assert exit_code == 0


def test_main_chat_dispatch(monkeypatch):
    from flexsds.cli.main import main
    called = {}

    def mock_run_chat(endpoint, model, temperature):
        called["endpoint"] = endpoint
        called["model"] = model
        called["temperature"] = temperature
        return 0

    monkeypatch.setattr("flexsds.cli.main.run_interactive_chat", mock_run_chat)
    code = main(["chat", "--endpoint", "http://127.0.0.1:9000/v1", "--model", "custom-model", "--temperature", "0.7"])
    assert code == 0
    assert called["endpoint"] == "http://127.0.0.1:9000/v1"
    assert called["model"] == "custom-model"
    assert called["temperature"] == 0.7
