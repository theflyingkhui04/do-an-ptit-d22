"""Tests for item arithmetic checks ([M5-01] node1-full-receipt-quality spec)."""

from __future__ import annotations

from src.pipeline.item_consistency import invoice_sum_consistent, line_consistent, summarize_items


def item(line: int, quantity, unit_price, line_total) -> dict:
    return {
        "line_number": line,
        "description": f"item {line}",
        "sku": None,
        "quantity": {"raw": None, "value": quantity, "unit": None},
        "unit_price": {"raw": None, "value": unit_price, "currency": "VND"},
        "line_total": {"raw": None, "value": line_total, "currency": "VND"},
        "vat_rate": None,
    }


def money(value) -> dict:
    return {"raw": None, "value": value, "currency": "VND"}


def payload(items: list[dict], total=None, subtotal=None, discount=None) -> dict:
    return {
        "items": items,
        "totals": {
            "subtotal": money(subtotal),
            "discount": money(discount),
            "vat_amount": money(None),
            "total_amount": money(total),
        },
    }


def test_consistent_receipt():
    receipt = payload([item(1, 2, 10_000, 20_000), item(2, 1, 5_000, 5_000)], total=25_000)

    assert line_consistent(receipt["items"][0]) is True
    summary = summarize_items({"s1": receipt})

    assert summary["item_rows"] == 2
    assert summary["lines_checked"] == 2 and summary["lines_consistent"] == 2
    assert summary["invoices_with_items"] == 1
    assert summary["invoices_checked"] == 1 and summary["invoices_consistent"] == 1
    assert summary["mismatches"] == []


def test_missing_quantity_counts_as_one():
    assert line_consistent(item(1, None, 7_000, 7_000)) is True


def test_weighed_goods_within_half_percent():
    # 0.354 kg x 52,000 = 18,408 printed as 18,400.
    assert line_consistent(item(1, 0.354, 52_000, 18_400)) is True
    assert line_consistent(item(1, 0.354, 52_000, 17_000)) is False


def test_missing_line_total_is_not_checked():
    receipt = payload([item(1, 1, 5_000, None), item(2, 1, 5_000, 5_000)], total=10_000)

    assert line_consistent(receipt["items"][0]) is None
    assert invoice_sum_consistent(receipt) is None
    summary = summarize_items({"s1": receipt})
    assert summary["lines_checked"] == 1
    assert summary["invoices_checked"] == 0


def test_sum_matches_subtotal_or_total_plus_discount():
    assert invoice_sum_consistent(payload([item(1, 1, 100_000, 100_000)], total=110_000, subtotal=100_000)) is True
    assert invoice_sum_consistent(payload([item(1, 1, 50_000, 50_000)], total=45_000, discount=5_000)) is True
    assert invoice_sum_consistent(payload([item(1, 1, 295_008, 295_008)], total=500_000)) is False


def test_mismatches_and_invalid_samples_are_reported():
    bad = payload([item(1, 2, 10_000, 15_000)], total=15_000)
    summary = summarize_items({"s1": bad, "s2": None, "s3": payload([], total=1_000)})

    assert summary["invoices"] == 2
    assert summary["invoices_with_items"] == 1
    assert summary["lines_consistent"] == 0
    assert summary["invoices_consistent"] == 1
    assert summary["mismatches"] == [{"sample_id": "s1", "kind": "line", "line_number": 1,
                                      "expected": 20_000, "actual": 15_000}]
