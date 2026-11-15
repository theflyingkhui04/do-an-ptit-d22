"""Score Node 1 JSON against MC-OCR labels (pure functions, no file access).

Labels come from ``key_text_pairs`` in the M0-04 dev-set. They stay outside the
model prompt (see src/pipeline/README.md) and are used here for evaluation only.
The labels are raw receipt text, so each field is normalized before comparison:

    SELLER      <-> seller.name
    ADDRESS     <-> seller.address
    TIMESTAMPS  <-> invoice.issued_date (and issued_time when the label has a time)
    TOTAL_COST  <-> totals.total_amount.value

Vietnamese diacritics are kept on purpose: removing them would hide real errors.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

MATCH = "match"
MISMATCH = "mismatch"
BLANK = "blank"  # labelled field, but the model returned nothing
UNSCORED = "unscored"  # the label itself could not be parsed
NO_LABEL = "no_label"  # the sample has no label for this field

FIELDS = ("SELLER", "ADDRESS", "TIMESTAMPS", "TOTAL_COST")
AMOUNT_TOLERANCE = 0.5

_ADDRESS_PREFIX_RE = re.compile(r"^(?:đc|đ/c|địa chỉ)\s*:\s*", re.IGNORECASE)
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)")
_TIME_RE = re.compile(r"(?<![\d.:])(\d{1,2})[:.](\d{2})(?!\d)")
_AMOUNT_RE = re.compile(r"\d+(?:[.,\s]\d{3})*(?:[.,]\d{2})?")


@dataclass(frozen=True)
class FieldScore:
    status: str
    f1: float | None = None
    date_ok: bool | None = None
    time_ok: bool | None = None
    gold: str | None = None
    predicted: Any = None


def normalize_text(text: str | None) -> str:
    """NFC, lower case, punctuation to spaces, single spaces; diacritics kept."""
    folded = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(re.sub(r"[^\w\s]", " ", folded).split())


def token_f1(gold: str | None, predicted: str | None) -> float:
    gold_tokens = normalize_text(gold).split()
    predicted_tokens = normalize_text(predicted).split()
    if not gold_tokens or not predicted_tokens:
        return 0.0
    common = sum((Counter(gold_tokens) & Counter(predicted_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_date(text: str) -> str | None:
    """First real ``dd/mm/yyyy`` (also ``-`` or ``.``) as ``YYYY-MM-DD``."""
    for day, month, year in _DATE_RE.findall(text):
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            continue
    return None


def parse_time(text: str) -> str | None:
    """First real ``HH:MM`` (also ``HH.MM``) outside a date, as ``HH:MM``."""
    for hour, minute in _TIME_RE.findall(_DATE_RE.sub(" ", text)):
        if int(hour) < 24 and int(minute) < 60:
            return f"{int(hour):02d}:{minute}"
    return None


def _amount_value(token: str) -> float:
    integer, decimals = token, "0"
    if len(token) > 3 and token[-3] in ".,":
        integer, decimals = token[:-3], token[-2:]
    digits = re.sub(r"\D", "", integer)
    return float(f"{digits}.{decimals}")


def parse_amounts(text: str) -> list[float]:
    """Every amount in one label line; thousands use ``,`` ``.`` or a space."""
    return [_amount_value(token) for token in _AMOUNT_RE.findall(text)]


def _strip_address_prefix(text: str) -> str:
    return _ADDRESS_PREFIX_RE.sub("", unicodedata.normalize("NFC", text).strip())


def _score_text(lines: list[str], predicted: Any, *, address: bool = False) -> FieldScore:
    if not lines:
        return FieldScore(NO_LABEL)
    gold = " ".join(lines)
    if not isinstance(predicted, str) or not predicted.strip():
        return FieldScore(BLANK, f1=0.0, gold=gold, predicted=predicted)
    gold_text = _strip_address_prefix(gold) if address else gold
    predicted_text = _strip_address_prefix(predicted) if address else predicted
    exact = normalize_text(gold_text) == normalize_text(predicted_text)
    return FieldScore(
        MATCH if exact else MISMATCH,
        f1=token_f1(gold_text, predicted_text),
        gold=gold,
        predicted=predicted,
    )


def _score_timestamps(lines: list[str], issued_date: Any, issued_time: Any) -> FieldScore:
    if not lines:
        return FieldScore(NO_LABEL)
    gold = " ".join(lines)
    gold_date = parse_date(gold)
    if gold_date is None:
        return FieldScore(UNSCORED, gold=gold, predicted=issued_date)
    if not issued_date:
        return FieldScore(BLANK, gold=gold, predicted=issued_date)
    date_ok = issued_date == gold_date
    gold_time = parse_time(gold)
    time_ok = None if gold_time is None else str(issued_time or "").startswith(gold_time)
    return FieldScore(
        MATCH if date_ok and time_ok is not False else MISMATCH,
        date_ok=date_ok,
        time_ok=time_ok,
        gold=gold,
        predicted=f"{issued_date} {issued_time}",
    )


def _score_total(lines: list[str], predicted: Any) -> FieldScore:
    if not lines:
        return FieldScore(NO_LABEL)
    gold = " ".join(lines)
    amounts = [amount for line in lines for amount in parse_amounts(line)]
    if not amounts:
        return FieldScore(UNSCORED, gold=gold, predicted=predicted)
    if predicted is None:
        return FieldScore(BLANK, gold=gold, predicted=predicted)
    matched = any(abs(float(predicted) - amount) <= AMOUNT_TOLERANCE for amount in amounts)
    return FieldScore(MATCH if matched else MISMATCH, gold=gold, predicted=predicted)


def _lines(pairs: list[dict[str, str]], label: str) -> list[str]:
    return [pair["text"] for pair in pairs if pair.get("label") == label and pair.get("text")]


def _strip_strings(value: Any) -> Any:
    from src.pipeline.text_normalize import strip_vietnamese_marks

    if isinstance(value, str):
        return strip_vietnamese_marks(value)
    if isinstance(value, dict):
        return {k: _strip_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_strings(v) for v in value]
    return value


def score_sample(payload: dict[str, Any] | None, pairs: list[dict[str, str]], *,
                 strip_marks: bool = False) -> dict[str, FieldScore]:
    """Score one sample; ``payload=None`` means the run has no valid JSON for it.

    ``strip_marks`` ([M5-01 canonical flow]) removes Vietnamese marks from labels and values first.
    """
    if strip_marks:
        payload = _strip_strings(payload)
        pairs = _strip_strings(pairs)
    data = payload or {}
    seller = data.get("seller") or {}
    invoice = data.get("invoice") or {}
    total = (data.get("totals") or {}).get("total_amount") or {}
    return {
        "SELLER": _score_text(_lines(pairs, "SELLER"), seller.get("name")),
        "ADDRESS": _score_text(_lines(pairs, "ADDRESS"), seller.get("address"), address=True),
        "TIMESTAMPS": _score_timestamps(
            _lines(pairs, "TIMESTAMPS"), invoice.get("issued_date"), invoice.get("issued_time")
        ),
        "TOTAL_COST": _score_total(_lines(pairs, "TOTAL_COST"), total.get("value")),
    }
