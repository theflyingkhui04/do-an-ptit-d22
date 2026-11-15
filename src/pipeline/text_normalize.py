"""[M5-01 canonical flow] One fixed rule to remove Vietnamese diacritics.

Applied to OCR text (before Node 1), to user questions (before Node 2) and to
MC-OCR labels when scoring, so the canonical flow can never show a wrong mark.
"""

from __future__ import annotations

import unicodedata

_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})


def strip_vietnamese_marks(text: str) -> str:
    """NFD, drop combining marks, map đ/Đ to d/D, then NFC."""
    decomposed = unicodedata.normalize("NFD", text.translate(_D_MAP))
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", kept)


def has_vietnamese_marks(text: str) -> bool:
    return strip_vietnamese_marks(text) != unicodedata.normalize("NFC", text)
