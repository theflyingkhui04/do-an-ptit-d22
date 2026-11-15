"""Which extracted text values do not appear in the OCR text (anti-fabrication report).

Checks text fields and the ``raw`` strings of money/quantity values, comparing
case-insensitively with collapsed whitespace. Normalized values (ISO dates,
numeric ``value`` fields) are not compared: they never appear verbatim.
"""

from __future__ import annotations

import re
from typing import Any

TEXT_FIELDS = (
    ("seller", ("name", "address", "tax_code", "phone")),
    ("buyer", ("name", "address", "tax_code", "phone")),
    ("invoice", ("invoice_number", "invoice_series")),
)
MONEY_FIELDS = ("subtotal", "discount", "vat_amount", "total_amount")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def candidate_fields(payload: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for section, keys in TEXT_FIELDS:
        for key in keys:
            value = (payload.get(section) or {}).get(key)
            if isinstance(value, str) and value.strip():
                values[f"{section}.{key}"] = value
    for name in MONEY_FIELDS:
        raw = ((payload.get("totals") or {}).get(name) or {}).get("raw")
        if isinstance(raw, str) and raw.strip():
            values[f"totals.{name}.raw"] = raw
    for index, item in enumerate(payload.get("items") or []):
        for key in ("description", "sku"):
            if isinstance(item.get(key), str) and item[key].strip():
                values[f"items[{index}].{key}"] = item[key]
        for name in ("quantity", "unit_price", "line_total"):
            raw = (item.get(name) or {}).get("raw")
            if isinstance(raw, str) and raw.strip():
                values[f"items[{index}].{name}.raw"] = raw
    return values


def grounding_report(payload: dict[str, Any], ocr_text: str) -> dict[str, Any]:
    text = _norm(ocr_text)
    fields = candidate_fields(payload)
    missing = sorted(path for path, value in fields.items() if _norm(value) not in text)
    return {"checked_fields": len(fields), "grounded_fields": len(fields) - len(missing),
            "ungrounded_fields": missing}
