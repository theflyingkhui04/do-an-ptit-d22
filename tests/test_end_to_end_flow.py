"""End-to-End integration tests for FlexSDS CLI across all 3 platforms with dry-run."""
import json
import os
from unittest.mock import patch
import pytest
from flexsds.cli.main import main


def test_e2e_cli_local_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "local",
        "--port", "8000",
        "--dry-run",
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "vllm serve" in captured.out
    assert "--speculative-config" in captured.out
    assert "AdaptiveAsyncScheduler" in captured.out


def test_e2e_cli_modal_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "modal",
        "--dry-run",
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "modal run" in captured.out
    assert "experiments/modal_lanes.py" in captured.out


def test_e2e_cli_modal_deploy_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "modal",
        "--deploy",
        "--dry-run",
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "modal deploy" in captured.out


def test_e2e_cli_colab_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "colab",
        "--dry-run",
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Colab notebook URL" in captured.out


def test_e2e_cli_detect_command(capsys):
    ret = main(["detect"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Trạng thái phần cứng" in captured.out


def test_e2e_cli_profile_command(capsys):
    ret = main(["profile"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Đã liên kết với hồ sơ phần cứng" in captured.out


def test_e2e_cli_stop_command(capsys):
    ret = main(["stop"])
    assert ret == 0
