"""Presentation regression tests independent of the host terminal's locale."""
import io
import subprocess

import pytest

from flexsds.cli import banner
from flexsds.cli import main as cli
from flexsds.cli import wizard
from flexsds.core.hardware import HardwareInfo


class EncodedStream:
    """Strict sink; metadata can deliberately disagree with its actual codec."""

    def __init__(self, encoding="cp1252", actual=None, tty=False):
        self.encoding = encoding
        self.actual = actual or encoding or "ascii"
        self.tty = tty
        self.parts = []

    def write(self, text):
        text.encode(self.actual)
        self.parts.append(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return self.tty

    def getvalue(self):
        return "".join(self.parts)


@pytest.fixture
def hardware():
    return HardwareInfo("NVIDIA L4", "sm_89", 23000, 22000, "12.4", True)


def test_utf8_truecolor_preserves_original_banner(monkeypatch, hardware):
    sink = EncodedStream("utf-8", tty=True)
    monkeypatch.setattr(banner.sys, "stdout", sink)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    banner.print_banner(hardware)
    assert sink.getvalue() == banner.render_banner(hardware) + "\n"
    assert "\u2584" in sink.getvalue()


@pytest.mark.parametrize("encoding,actual", [
    ("cp1252", "cp1252"), ("ascii", "ascii"), (None, "ascii"),
    ("unknown-test-codec", "ascii"), ("utf-8", "ascii"),
])
def test_banner_falls_back_without_losing_identity(monkeypatch, hardware, encoding, actual):
    sink = EncodedStream(encoding, actual, tty=True)
    monkeypatch.setattr(banner.sys, "stdout", sink)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-direct")
    monkeypatch.setenv("COLORTERM", "truecolor")
    banner.print_banner(hardware)
    result = sink.getvalue()
    assert result.isascii()
    assert "\x1b" not in result
    assert all(word in result for word in ("FlexSDS", "PTIT", "NVIDIA L4", "sm_89", "Not yet configured"))


def test_missing_encoding_and_isatty_are_safe(monkeypatch, hardware):
    class UnknownStream:
        def __init__(self):
            self.text = ""

        def write(self, text):
            text.encode("ascii")
            self.text += text

    sink = UnknownStream()
    monkeypatch.setattr(banner.sys, "stdout", sink)
    banner.print_banner(hardware)
    assert "FlexSDS" in sink.text and sink.text.isascii()


@pytest.mark.parametrize("tty,no_color,term", [
    (False, False, "xterm-direct"), (True, True, "xterm-direct"),
    (True, False, "dumb"), (True, False, "xterm-256color"),
])
def test_no_truecolor_uses_plain_ascii(monkeypatch, hardware, tty, no_color, term):
    sink = EncodedStream("utf-8", tty=tty)
    monkeypatch.setattr(banner.sys, "stdout", sink)
    monkeypatch.setenv("TERM", term)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    banner.print_banner(hardware)
    assert sink.getvalue().isascii()
    assert "\x1b" not in sink.getvalue()


@pytest.mark.parametrize("args,expected", [
    (["detect"], "Hardware:"), (["profile"], "Linked hardware profile:"),
    (["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft",
      "Qwen/Qwen2.5-0.5B-Instruct", "--backend", "local", "--no-adaptive", "--dry-run"],
     "Dry-run execution command:"),
])
def test_command_completes_on_cp1252(monkeypatch, hardware, args, expected):
    sink = EncodedStream()
    monkeypatch.setattr(cli.sys, "stdout", sink)
    monkeypatch.setattr(cli, "detect_hardware", lambda: hardware)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Server launched"))
    assert cli.main(args) == 0
    assert expected in sink.getvalue()
    assert sink.getvalue().isascii()


def test_unrelated_write_errors_propagate(monkeypatch, hardware):
    class BrokenStream(io.StringIO):
        def write(self, text):
            raise BrokenPipeError("closed pipe")

    monkeypatch.setattr(banner.sys, "stdout", BrokenStream())
    with pytest.raises(BrokenPipeError, match="closed pipe"):
        banner.print_banner(hardware)


def test_wizard_detection_output_on_cp1252(monkeypatch, hardware):
    sink = EncodedStream()
    monkeypatch.setattr(wizard.sys, "stdout", sink)
    monkeypatch.setattr(wizard, "prompt_choice", lambda *a, **kw: "Detect hardware")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Server launched"))
    assert wizard.run_interactive_wizard(hardware) == {"action": "detect"}
    assert "Hardware specs:" in sink.getvalue()
    assert sink.getvalue().isascii()


def test_wizard_text_prompt_on_cp1252(monkeypatch):
    class Input(io.StringIO):
        def isatty(self):
            return True

    sink = EncodedStream()
    monkeypatch.setattr(wizard.sys, "stdout", sink)
    monkeypatch.setattr(wizard.sys, "stdin", Input("chosen-model\n"))
    assert wizard.prompt_string("Target model?", "default") == "chosen-model"
    assert "chosen-model" in sink.getvalue()
    assert sink.getvalue().isascii()


def test_wizard_choice_confirmation_on_cp1252(monkeypatch):
    sink = EncodedStream()
    monkeypatch.setattr(wizard.sys, "stdout", sink)
    monkeypatch.setattr(wizard.sys, "stdin", io.StringIO())
    assert wizard.prompt_choice("Platform?", ["Local", "Modal"]) == "Local"
    assert "[OK]" in sink.getvalue()
    assert sink.getvalue().isascii()
