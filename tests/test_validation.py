"""Tests for the shared M1-03 invoice validation stack."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Callable

from src.pipeline.validation import (
    json_schema_validation_errors,
    pydantic_validation_errors,
    semantic_validation_errors,
    validate_payload,
)


EXAMPLE_PATH = Path(
    "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
)
SCHEMA_PATH = Path(
    "src/pipeline/schemas/invoice_extraction.schema.json"
)


def load_example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def add_invalid_item_with_line_number_zero(payload: dict) -> None:
    payload["items"] = [
        {
            "line_number": 0,
            "description": "Test item",
            "sku": None,
            "quantity": {
                "raw": "1",
                "value": 1,
                "unit": "pcs",
            },
            "unit_price": {
                "raw": "1000",
                "value": 1000,
                "currency": "VND",
            },
            "line_total": {
                "raw": "1000",
                "value": 1000,
                "currency": "VND",
            },
            "vat_rate": None,
        }
    ]


class SharedValidationTests(unittest.TestCase):
    def assert_public_contract_parity(
        self,
        payload: dict,
        *,
        expected_invalid: bool,
        case_name: str,
    ) -> None:
        schema_errors = json_schema_validation_errors(
            payload,
            SCHEMA_PATH,
        )
        pydantic_errors = pydantic_validation_errors(payload)

        self.assertEqual(
            bool(schema_errors),
            expected_invalid,
            msg=(
                f"{case_name}: JSON Schema invalid-state mismatch; "
                f"errors={schema_errors}"
            ),
        )
        self.assertEqual(
            bool(pydantic_errors),
            expected_invalid,
            msg=(
                f"{case_name}: Pydantic invalid-state mismatch; "
                f"errors={pydantic_errors}"
            ),
        )
        self.assertEqual(
            bool(schema_errors),
            bool(pydantic_errors),
            msg=(
                f"{case_name}: JSON Schema/Pydantic contract drift; "
                f"schema_errors={schema_errors}; "
                f"pydantic_errors={pydantic_errors}"
            ),
        )

    def test_canonical_example_passes_shared_stack(self) -> None:
        validate_payload(load_example(), SCHEMA_PATH)

    def test_public_contract_parity_matrix(self) -> None:
        cases: list[
            tuple[str, Callable[[dict], None], bool]
        ] = [
            (
                "valid canonical fixture",
                lambda payload: None,
                False,
            ),
            (
                "missing required top-level field",
                lambda payload: payload.pop("document_type"),
                True,
            ),
            (
                "unexpected top-level property",
                lambda payload: payload.__setitem__(
                    "unexpected",
                    True,
                ),
                True,
            ),
            (
                "invalid document_type enum",
                lambda payload: payload.__setitem__(
                    "document_type",
                    "not-valid",
                ),
                True,
            ),
            (
                "negative money value",
                lambda payload: payload["totals"][
                    "total_amount"
                ].__setitem__("value", -1),
                True,
            ),
            (
                "confidence above one",
                lambda payload: payload["extraction"].__setitem__(
                    "confidence",
                    1.1,
                ),
                True,
            ),
            (
                "line_number zero",
                add_invalid_item_with_line_number_zero,
                True,
            ),
            (
                "required source raw_text is null",
                lambda payload: payload["source"].__setitem__(
                    "raw_text",
                    None,
                ),
                True,
            ),
        ]

        for case_name, mutate, expected_invalid in cases:
            with self.subTest(case=case_name):
                payload = load_example()
                mutate(payload)
                self.assert_public_contract_parity(
                    payload,
                    expected_invalid=expected_invalid,
                    case_name=case_name,
                )

    def test_json_schema_rejects_additional_top_level_property(self) -> None:
        payload = load_example()
        payload["unexpected"] = True

        errors = json_schema_validation_errors(
            payload,
            SCHEMA_PATH,
        )

        self.assertTrue(errors)
        self.assertTrue(
            any("Additional properties" in error for error in errors)
        )

    def test_semantic_validation_rejects_impossible_date(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_date"] = "2020-02-30"

        errors = semantic_validation_errors(payload)

        self.assertTrue(
            any("invalid calendar value" in error for error in errors)
        )

    def test_semantic_validation_rejects_date_time_conflict(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_date"] = "2020-08-10"
        payload["invoice"]["issued_time"] = "17:12:00"
        payload["invoice"]["issued_at"] = "2020-08-11T17:12:00"

        errors = semantic_validation_errors(payload)

        self.assertTrue(
            any(
                "date component conflicts" in error
                for error in errors
            )
        )

    def test_semantic_validation_rejects_duplicate_line_numbers(self) -> None:
        payload = load_example()
        item = {
            "line_number": 1,
            "description": "Test",
            "sku": None,
            "quantity": {
                "raw": "1",
                "value": 1,
                "unit": "cai",
            },
            "unit_price": {
                "raw": "1",
                "value": 1,
                "currency": "VND",
            },
            "line_total": {
                "raw": "1",
                "value": 1,
                "currency": "VND",
            },
            "vat_rate": None,
        }
        payload["items"] = [
            copy.deepcopy(item),
            copy.deepcopy(item),
        ]

        errors = semantic_validation_errors(payload)

        self.assertIn(
            "items: line_number values must be unique",
            errors,
        )

    def test_semantic_validation_rejects_non_verbatim_evidence(self) -> None:
        payload = load_example()
        payload["extraction"]["field_evidence"][0]["raw_text"] = (
            "not present in OCR text"
        )

        errors = semantic_validation_errors(payload)

        self.assertTrue(
            any(
                "not a verbatim source substring" in error
                for error in errors
            )
        )

    def test_pydantic_validation_rejects_non_iso_currency(self) -> None:
        payload = load_example()
        payload["invoice"]["currency"] = "VNĐ"

        errors = pydantic_validation_errors(payload)

        self.assertTrue(errors)

    def test_validate_payload_uses_same_stack(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_date"] = "2020-02-30"

        with self.assertRaises(ValueError) as caught:
            validate_payload(payload, SCHEMA_PATH)

        message = str(caught.exception)
        self.assertIn("semantic:", message)
        self.assertIn("pydantic:", message)


if __name__ == "__main__":
    unittest.main()
