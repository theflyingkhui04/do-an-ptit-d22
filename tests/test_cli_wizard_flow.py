"""Tests for interactive wizard and CLI dispatcher flow."""
import pytest
from flexsds.cli.wizard import PLATFORM_CHOICES
from flexsds.cli.main import parse_args


def test_wizard_platform_selection_structure():
    assert "Modal.com" in PLATFORM_CHOICES[0]
    assert "Google Colab" in PLATFORM_CHOICES[1]
    assert "Tự dùng GPU" in PLATFORM_CHOICES[2]


def test_cli_stop_command_parser():
    args = parse_args(["stop"])
    assert args.command == "stop"


def test_cli_serve_flags_parser():
    args = parse_args(["serve", "--port", "8008", "--background", "--deploy", "--dry-run"])
    assert args.command == "serve"
    assert args.port == 8008
    assert args.background is True
    assert args.deploy is True
    assert args.dry_run is True
