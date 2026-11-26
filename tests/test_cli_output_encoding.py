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
      "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--backend", "local", "--no-adaptive", "--dry-run"],
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
    answers = iter(["Detect hardware"])

    def choose(*a, **kw):
        try:
            return next(answers)
        except StopIteration:  # the menu comes back after detection; leave it
            raise KeyboardInterrupt from None

    monkeypatch.setattr(wizard, "prompt_choice", choose)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Server launched"))
    with pytest.raises(KeyboardInterrupt):
        wizard.run_interactive_wizard(hardware)
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


@pytest.mark.parametrize("terminal,expected", [("legacy", "[OK]"), ("wt", "✔"), ("vscode", "✔")])
def test_windows_status_marker_handles_missing_font_glyph(monkeypatch, terminal, expected):
    from flexsds.cli import output
    sink = EncodedStream("utf-8", tty=True)
    monkeypatch.setattr(output.sys, "platform", "win32")
    monkeypatch.setattr(output, "supports_truecolor", lambda stream: True)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    if terminal == "wt":
        monkeypatch.setenv("WT_SESSION", "test-session")
    elif terminal == "vscode":
        monkeypatch.setenv("TERM_PROGRAM", "vscode")
    output.print_output("\x1b[32m✔ Ready\x1b[0m", file=sink)
    assert sink.getvalue() == f"\x1b[32m{expected} Ready\x1b[0m\n"
