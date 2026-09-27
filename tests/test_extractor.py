"""Unit tests for the pure-core M1-03 Node 1 extractor helpers."""

from __future__ import annotations

import copy
import unittest

from src.pipeline.extractor import (
    ExtractorError,
    ExtractorStage,
    FailureCode,
    normalize_canonical_payload,
    parse_assistant_payload,
    validate_source_binding,
)


EXPECTED_SOURCE = {
    "sample_id": "m0_04_001",
    "img_id": "img-001",
    "image_path": "images/001.jpg",
    "raw_text": "SHOP ABC\nTOTAL 10.000",
}


def response_with_content(content: object) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


class ParseAssistantPayloadTests(unittest.TestCase):
    def test_strict_json_object_success_preserves_exact_content(self) -> None:
        content = '{"schema_version":"invoice_extraction.v1","source":{"sample_id":"m0_04_001"}}'
        payload, raw_content = parse_assistant_payload(response_with_content(content))

        self.assertEqual(payload["schema_version"], "invoice_extraction.v1")
        self.assertEqual(raw_content, content)

    def test_markdown_wrapped_json_is_rejected(self) -> None:
        content = '```json\n{"a":1}\n```'

        with self.assertRaises(ExtractorError) as caught:
            parse_assistant_payload(response_with_content(content))

        self.assertEqual(caught.exception.code, FailureCode.JSON_PARSE_ERROR)
        self.assertEqual(caught.exception.stage, ExtractorStage.PARSE)

    def test_json_array_is_rejected(self) -> None:
        with self.assertRaises(ExtractorError) as caught:
            parse_assistant_payload(response_with_content('[{"a":1}]'))

        self.assertEqual(caught.exception.code, FailureCode.JSON_PARSE_ERROR)
        self.assertEqual(caught.exception.stage, ExtractorStage.PARSE)

    def test_non_finite_json_numbers_are_rejected(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(ExtractorError) as caught:
                    parse_assistant_payload(
                        response_with_content(f'{{"value": {constant}}}')
                    )

                self.assertEqual(
                    caught.exception.code,
                    FailureCode.JSON_PARSE_ERROR,
                )
                self.assertEqual(caught.exception.stage, ExtractorStage.PARSE)

    def test_missing_choices_is_response_envelope_error(self) -> None:
        with self.assertRaises(ExtractorError) as caught:
            parse_assistant_payload({})

        self.assertEqual(
            caught.exception.code,
            FailureCode.RESPONSE_ENVELOPE_ERROR,
        )
        self.assertEqual(caught.exception.stage, ExtractorStage.PARSE)

    def test_empty_content_is_response_envelope_error(self) -> None:
        with self.assertRaises(ExtractorError) as caught:
            parse_assistant_payload(response_with_content("   "))

        self.assertEqual(
            caught.exception.code,
            FailureCode.RESPONSE_ENVELOPE_ERROR,
        )


class SourceBindingTests(unittest.TestCase):
    def test_exact_source_passes_without_mutation(self) -> None:
        payload = {"source": copy.deepcopy(EXPECTED_SOURCE)}
        before = copy.deepcopy(payload)

        validate_source_binding(payload, EXPECTED_SOURCE)

        self.assertEqual(payload, before)

    def test_sample_id_mismatch_is_rejected(self) -> None:
        payload = {"source": copy.deepcopy(EXPECTED_SOURCE)}
        payload["source"]["sample_id"] = "wrong"

        with self.assertRaises(ExtractorError) as caught:
            validate_source_binding(payload, EXPECTED_SOURCE)

        self.assertEqual(
            caught.exception.code,
            FailureCode.SOURCE_BINDING_ERROR,
        )
        self.assertIn("sample_id", caught.exception.message)

    def test_raw_text_mismatch_is_rejected(self) -> None:
        payload = {"source": copy.deepcopy(EXPECTED_SOURCE)}
        payload["source"]["raw_text"] = "normalized by model"

        with self.assertRaises(ExtractorError) as caught:
            validate_source_binding(payload, EXPECTED_SOURCE)

        self.assertEqual(
            caught.exception.code,
            FailureCode.SOURCE_BINDING_ERROR,
        )
        self.assertIn("raw_text", caught.exception.message)

    def test_unexpected_source_key_is_rejected(self) -> None:
        payload = {"source": copy.deepcopy(EXPECTED_SOURCE)}
        payload["source"]["model_note"] = "invented"

        with self.assertRaises(ExtractorError) as caught:
            validate_source_binding(payload, EXPECTED_SOURCE)

        self.assertEqual(
            caught.exception.code,
            FailureCode.SOURCE_BINDING_ERROR,
        )
        self.assertIn("unexpected keys", caught.exception.message)
        self.assertIn("model_note", caught.exception.message)

    def test_missing_source_key_is_rejected(self) -> None:
        payload = {"source": copy.deepcopy(EXPECTED_SOURCE)}
        del payload["source"]["img_id"]

        with self.assertRaises(ExtractorError) as caught:
            validate_source_binding(payload, EXPECTED_SOURCE)

        self.assertEqual(
            caught.exception.code,
            FailureCode.SOURCE_BINDING_ERROR,
        )
        self.assertIn("missing keys", caught.exception.message)
        self.assertIn("img_id", caught.exception.message)


class CanonicalizationTests(unittest.TestCase):
    def test_composes_issued_at(self) -> None:
        payload = {
            "invoice": {
                "issued_date": "2020-08-10",
                "issued_time": "17:12",
                "issued_at": None,
            }
        }

        notes = normalize_canonical_payload(payload, EXPECTED_SOURCE)

        self.assertEqual(
            payload["invoice"]["issued_at"],
            "2020-08-10T17:12:00",
        )
        self.assertEqual(len(notes), 1)

    def test_replaces_null_money_containers(self) -> None:
        payload = {
            "totals": {
                "subtotal": None,
                "discount": None,
                "vat_amount": None,
                "total_amount": None,
            }
        }

        notes = normalize_canonical_payload(payload, EXPECTED_SOURCE)

        empty_money = {
            "raw": None,
            "value": None,
            "currency": None,
        }
        self.assertTrue(
            all(value == empty_money for value in payload["totals"].values())
        )
        self.assertEqual(len(notes), 4)

    def test_defaults_source_label_and_removes_unsupported_evidence(self) -> None:
        payload = {
            "extraction": {
                "field_evidence": [
                    {
                        "field_path": "seller.name",
                        "raw_text": "SHOP ABC",
                    },
                    {
                        "field_path": "totals.total_amount",
                        "raw_text": "invented total",
                        "source_label": None,
                    },
                ]
            }
        }

        notes = normalize_canonical_payload(payload, EXPECTED_SOURCE)

        self.assertEqual(
            payload["extraction"]["field_evidence"],
            [
                {
                    "field_path": "seller.name",
                    "raw_text": "SHOP ABC",
                    "source_label": None,
                }
            ],
        )
        self.assertEqual(len(notes), 2)

    def test_business_values_and_source_are_not_changed(self) -> None:
        payload = {
            "source": copy.deepcopy(EXPECTED_SOURCE),
            "seller": {
                "name": " Seller  X ",
                "address": "Addr",
                "tax_code": "TAX-001",
                "phone": None,
            },
            "buyer": {
                "name": "Buyer",
                "address": None,
                "tax_code": None,
                "phone": None,
            },
            "invoice": {
                "invoice_number": "INV-001",
                "invoice_series": None,
                "issued_at": None,
                "issued_date": None,
                "issued_time": None,
                "currency": "VND",
                "payment_method": "Cash",
            },
            "totals": {
                "subtotal": {
                    "raw": "10.000",
                    "value": 10000,
                    "currency": "VND",
                },
                "discount": {
                    "raw": None,
                    "value": None,
                    "currency": None,
                },
                "vat_amount": {
                    "raw": "1.000",
                    "value": 1000,
                    "currency": "VND",
                },
                "total_amount": {
                    "raw": "11.000",
                    "value": 11000,
                    "currency": "VND",
                },
            },
            "taxes": [
                {
                    "tax_name": "VAT",
                    "tax_rate": 10,
                    "taxable_amount": {
                        "raw": "10.000",
                        "value": 10000,
                        "currency": "VND",
                    },
                    "tax_amount": {
                        "raw": "1.000",
                        "value": 1000,
                        "currency": "VND",
                    },
                }
            ],
            "items": [
                {
                    "line_number": 1,
                    "description": "Item A",
                    "sku": "SKU-1",
                    "quantity": {
                        "raw": "1",
                        "value": 1,
                        "unit": "pcs",
                    },
                    "unit_price": {
                        "raw": "10.000",
                        "value": 10000,
                        "currency": "VND",
                    },
                    "line_total": {
                        "raw": "10.000",
                        "value": 10000,
                        "currency": "VND",
                    },
                    "vat_rate": 10,
                }
            ],
            "extraction": {
                "field_evidence": [],
            },
        }

        before = copy.deepcopy(payload)
        normalize_canonical_payload(payload, EXPECTED_SOURCE)

        for field in ("source", "seller", "buyer", "taxes", "items"):
            self.assertEqual(payload[field], before[field])

        self.assertEqual(
            payload["invoice"]["invoice_number"],
            before["invoice"]["invoice_number"],
        )
        self.assertEqual(
            payload["invoice"]["currency"],
            before["invoice"]["currency"],
        )
        self.assertEqual(
            payload["invoice"]["payment_method"],
            before["invoice"]["payment_method"],
        )
        self.assertEqual(payload["totals"], before["totals"])


if __name__ == "__main__":
    unittest.main()
