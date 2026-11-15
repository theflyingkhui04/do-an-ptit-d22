"""[M5-01] Arithmetic checks for Node 1 item rows.

MC-OCR has no item labels, so item quality is checked without labels: each row
must satisfy quantity x unit_price = line_total, and the rows must add up to a
printed total. Item descriptions are not scored.
"""

from __future__ import annotations

from typing import Any

# Weighed goods are rounded on the receipt: allow 0.5% (at least 1 VND).
LINE_RELATIVE_TOLERANCE = 0.005
LINE_ABSOLUTE_TOLERANCE = 1.0
SUM_ABSOLUTE_TOLERANCE = 1.0


def _number(container: object, key: str = "value") -> float | None:
    if not isinstance(container, dict):
        return None
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _expected_line_total(item: dict[str, Any]) -> float | None:
    unit_price = _number(item.get("unit_price"))
    if unit_price is None:
        return None
    quantity = _number(item.get("quantity"))
    return (1.0 if quantity is None else quantity) * unit_price


def line_consistent(item: dict[str, Any]) -> bool | None:
    """True/False when the row can be checked, None when a value is missing."""
    expected = _expected_line_total(item)
    actual = _number(item.get("line_total"))
    if expected is None or actual is None:
        return None
    tolerance = max(LINE_ABSOLUTE_TOLERANCE, LINE_RELATIVE_TOLERANCE * abs(actual))
    return abs(expected - actual) <= tolerance


def invoice_sum_consistent(payload: dict[str, Any]) -> bool | None:
    """Sum of line totals equals total, subtotal or total + discount (±1 VND)."""
    items = payload.get("items") or []
    if not items:
        return None
    line_totals = [_number(row.get("line_total")) if isinstance(row, dict) else None for row in items]
    if any(value is None for value in line_totals):
        return None
    totals = payload.get("totals") or {}
    total = _number(totals.get("total_amount"))
    subtotal = _number(totals.get("subtotal"))
    discount = _number(totals.get("discount"))
    targets = [value for value in (total, subtotal) if value is not None]
    if total is not None and discount is not None:
        targets.append(total + discount)
    if not targets:
        return None
    line_sum = sum(line_totals)
    return any(abs(line_sum - target) <= SUM_ABSOLUTE_TOLERANCE for target in targets)


def summarize_items(outputs: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Item checks over the valid payloads of one run (None = no valid JSON)."""
    summary: dict[str, Any] = {
        "invoices": 0,
        "invoices_with_items": 0,
        "item_rows": 0,
        "lines_checked": 0,
        "lines_consistent": 0,
        "invoices_checked": 0,
        "invoices_consistent": 0,
        "mismatches": [],
    }
    for sample_id, payload in outputs.items():
        if payload is None:
            continue
        summary["invoices"] += 1
        items = [row for row in payload.get("items") or [] if isinstance(row, dict)]
        if not items:
            continue
        summary["invoices_with_items"] += 1
        summary["item_rows"] += len(items)
        for row in items:
            verdict = line_consistent(row)
            if verdict is None:
                continue
            summary["lines_checked"] += 1
            summary["lines_consistent"] += verdict
            if not verdict:
                summary["mismatches"].append({
                    "sample_id": sample_id,
                    "kind": "line",
                    "line_number": row.get("line_number"),
                    "expected": _expected_line_total(row),
                    "actual": _number(row.get("line_total")),
                })
        verdict = invoice_sum_consistent(payload)
        if verdict is None:
            continue
        summary["invoices_checked"] += 1
        summary["invoices_consistent"] += verdict
        if not verdict:
            totals = payload.get("totals") or {}
            summary["mismatches"].append({
                "sample_id": sample_id,
                "kind": "sum",
                "line_number": None,
                "expected": _number(totals.get("total_amount")),
                "actual": sum(_number(row.get("line_total")) or 0.0 for row in items),
            })
    return summary
