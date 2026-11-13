"""Encoding-safe presentation without reconfiguring caller-owned streams."""
from __future__ import annotations

import os
import re
import sys


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def supports_unicode(stream) -> bool:
    """Unknown codecs/streams conservatively use ASCII."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "\u2584\u2714\u26a0\u00b7".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def supports_truecolor(stream) -> bool:
    """Require positive terminal capability evidence, never send color to pipes."""
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    try:
        if not getattr(stream, "isatty", lambda: False)():
            return False
    except OSError:
        return False
    return (os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
            or os.environ.get("TERM", "").endswith(("-direct", "-truecolor"))
            or bool(os.environ.get("WT_SESSION")))


def ascii_text(text: str) -> str:
    """Keep status meanings; escape other unencodable content rather than omit it."""
    text = _ANSI.sub("", text)
    for symbol, replacement in (("\u2714", "[OK]"), ("\u26a0", "[WARN]"),
                                ("\u00b7", "|"), ("\u2022", "*")):
        text = text.replace(symbol, replacement)
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def print_output(text: str = "", *, file=None, end: str = "\n", fallback=None) -> None:
    """Write one message; only encoding errors may cause an ASCII retry.

    I/O errors such as a closed pipe propagate. The stream's encoding is never
    changed. A supplied fallback preserves a compact banner if metadata lies.
    """
    stream = sys.stdout if file is None else file
    if stream is None:
        return
    message = text + end
    if not supports_truecolor(stream):
        message = _ANSI.sub("", message)
    if not supports_unicode(stream):
        message = ascii_text((fallback if fallback is not None else text) + end)
    try:
        stream.write(message)
    except UnicodeEncodeError:
        stream.write(ascii_text((fallback if fallback is not None else text) + end))


def write_output(text: str) -> None:
    """Encoding-safe incremental output for streamed chat content."""
    print_output(text, end="")
