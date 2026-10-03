"""Tests for the invoice_extraction.v1 Pydantic mirror."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.pipeline.invoice_models import validate_invoice_payload


EXAMPLE_PATH = Path("src/pipeline/examples/node1_extractor_output_m0_04_001.json")


def load_example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


class InvoiceModelsTests(unittest.TestCase):
    def test_accepts_canonical_example(self) -> None:
        payload = load_example()
        model = validate_invoice_payload(payload)

        self.assertEqual(model.schema_version, "invoice_extraction.v1")
        self.assertEqual(model.source.sample_id, "m0_04_001")

    def test_rejects_currency_outside_iso_pattern(self) -> None:
        payload = load_example()
        payload["invoice"]["currency"] = "VNĐ"

        with self.assertRaises(ValidationError):
            validate_invoice_payload(payload)

    def test_rejects_impossible_calendar_date(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_date"] = "2020-02-30"

        with self.assertRaises(ValidationError):
            validate_invoice_payload(payload)

    def test_requires_issued_at_when_date_and_time_exist(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_at"] = None

        with self.assertRaises(ValidationError):
            validate_invoice_payload(payload)

    def test_rejects_duplicate_item_line_numbers(self) -> None:
        payload = load_example()
        item = {
            "line_number": 1,
            "description": "San pham kiem thu",
            "sku": "TEST-001",
            "quantity": {"raw": "2", "value": 2, "unit": "cai"},
            "unit_price": {"raw": "10,000", "value": 10000, "currency": "VND"},
            "line_total": {"raw": "20,000", "value": 20000, "currency": "VND"},
            "vat_rate": 10,
        }
        payload["items"] = [copy.deepcopy(item), copy.deepcopy(item)]

        with self.assertRaises(ValidationError):
            validate_invoice_payload(payload)

    def test_rejects_non_verbatim_field_evidence(self) -> None:
        payload = load_example()
        payload["extraction"]["field_evidence"][0]["raw_text"] = "not in the OCR source"

        with self.assertRaises(ValidationError):
            validate_invoice_payload(payload)


if __name__ == "__main__":
    unittest.main()
