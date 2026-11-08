"""[M5-01 canonical flow] Library OCR output -> plain text, and label coverage for choosing the engine.

Pure helpers (no OCR library import): the Modal app (experiments/modal_ocr_lib.py)
runs PaddleOCR / VietOCR and hands detected boxes to ``boxes_to_text``.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.pipeline.ocr import OcrError
from src.pipeline.text_normalize import strip_vietnamese_marks

ENGINES = {
    "A": "PaddleOCR text detection + VietOCR vgg_transformer recognition",
    "B": "PaddleOCR text detection + PaddleOCR recognition",
}
LABELS = ("SELLER", "ADDRESS", "TIMESTAMPS", "TOTAL_COST")


@dataclass(frozen=True)
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def centre_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @classmethod
    def from_polygon(cls, text: str, points: Any) -> Box:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return cls(text=text, x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def boxes_to_text(boxes: list[Box]) -> str:
    """Fixed rule: same row when centres differ by < half the median box height; rows top to bottom,
    boxes left to right joined by one space."""
    kept = [b for b in boxes if b.text.strip()]
    if not kept:
        raise OcrError("no text detected")
    threshold = statistics.median(b.height for b in kept) / 2
    rows: list[list[Box]] = []
    for b in sorted(kept, key=lambda b: (b.centre_y, b.x0)):
        if rows and abs(b.centre_y - rows[-1][0].centre_y) < threshold:
            rows[-1].append(b)
        else:
            rows.append([b])
    return "\n".join(" ".join(b.text.strip() for b in sorted(row, key=lambda b: b.x0)) for row in rows)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def label_coverage(ocr_text: str, key_text_pairs: list[dict[str, str]], *, strip_marks: bool) -> dict[str, dict[str, int]]:
    """Per label: how many MC-OCR label lines appear in the OCR text (whitespace collapsed, marks optionally removed)."""
    prepare = (lambda s: _collapse(strip_vietnamese_marks(s))) if strip_marks else _collapse
    text = prepare(ocr_text)
    result: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "found": 0})
    for pair in key_text_pairs:
        line = prepare(pair["text"])
        entry = result[pair["label"]]
        entry["total"] += 1
        entry["found"] += int(bool(line) and line in text)
    return dict(result)


def add_coverage(total: dict[str, dict[str, int]], one: dict[str, dict[str, int]]) -> None:
    for label, counts in one.items():
        entry = total.setdefault(label, {"total": 0, "found": 0})
        entry["total"] += counts["total"]
        entry["found"] += counts["found"]


def coverage_found(coverage: dict[str, dict[str, int]]) -> int:
    return sum(entry["found"] for entry in coverage.values())


def choose_engine(stripped_coverage: dict[str, dict[str, dict[str, int]]], median_latency_ms: dict[str, float]) -> str:
    """Pre-set rule: more label lines found after mark stripping wins; a tie goes to the faster engine."""
    return max(stripped_coverage, key=lambda e: (coverage_found(stripped_coverage[e]), -median_latency_ms[e]))
