"""Unit tests for M1-03 Node 1 extractor helpers and runtime core."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.pipeline.extractor import (
    ExtractorError,
    ExtractorStage,
    FailureCode,
    Node1Extractor,
    Node1Failure,
    Node1Success,
    normalize_canonical_payload,
    parse_assistant_payload,
    validate_source_binding,
)
from src.pipeline.openai_compatible import OpenAICompatibleError


EXAMPLE_PATH = Path(
    "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
)
SCHEMA_PATH = Path(
    "src/pipeline/schemas/invoice_extraction.schema.json"
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


class FakeClient:
    def __init__(
        self,
        models: list[str] | None = None,
        response: dict | None = None,
        list_error: Exception | None = None,
        request_error: Exception | None = None,
    ) -> None:
        self.models = (
            ["qwen2.5-7b-vanilla"]
            if models is None
            else list(models)
        )
        self.response = {} if response is None else response
        self.list_error = list_error
        self.request_error = request_error
        self.last_request: dict | None = None

    def list_models(self) -> list[str]:
        if self.list_error is not None:
            raise self.list_error
        return list(self.models)

    def chat_completions(self, payload: dict) -> dict:
        self.last_request = copy.deepcopy(payload)
        if self.request_error is not None:
            raise self.request_error
        return copy.deepcopy(self.response)


def load_example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def make_request_row(payload: dict) -> dict:
    source = copy.deepcopy(payload["source"])
    return {
        "sample_id": source["sample_id"],
        "schema_version": "invoice_extraction.v1",
        "prompt_version": "node1_extractor.v1",
        "expected_source": source,
        "request": {
            "model": "frozen-package-model-name",
            "messages": [
                {"role": "system", "content": "extract"},
                {"role": "user", "content": source["raw_text"]},
            ],
            "temperature": 0.0,
            "seed": 42,
            "response_format": {"type": "json_object"},
        },
    }


class ParseAssistantPayloadTests(unittest.TestCase):
    def test_strict_json_object_success_preserves_exact_content(self) -> None:
        content = (
            '{"schema_version":"invoice_extraction.v1",'
            '"source":{"sample_id":"m0_04_001"}}'
        )
        payload, raw_content = parse_assistant_payload(
            response_with_content(content)
        )

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
                self.assertEqual(
                    caught.exception.stage,
                    ExtractorStage.PARSE,
                )

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
        self.assertEqual(caught.exception.stage, ExtractorStage.PARSE)


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
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.SOURCE_BINDING,
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
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.SOURCE_BINDING,
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
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.SOURCE_BINDING,
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
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.SOURCE_BINDING,
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
            all(
                value == empty_money
                for value in payload["totals"].values()
            )
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


class Node1ExtractorRuntimeTests(unittest.TestCase):
    def make_extractor(
        self,
        response: dict,
    ) -> tuple[Node1Extractor, FakeClient]:
        client = FakeClient(response=response)
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )
        return extractor, client

    def test_preflight_passes_when_model_exists(self) -> None:
        client = FakeClient(models=["qwen2.5-7b-vanilla"])
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )

        extractor.preflight()

    def test_preflight_rejects_missing_model(self) -> None:
        client = FakeClient(models=["other-model"])
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )

        with self.assertRaises(ExtractorError) as caught:
            extractor.preflight()

        self.assertEqual(caught.exception.code, FailureCode.MODEL_NOT_FOUND)
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.PRECONDITION,
        )

    def test_preflight_classifies_unreachable_endpoint(self) -> None:
        client = FakeClient(
            list_error=OpenAICompatibleError(
                "Cannot reach http://127.0.0.1:8001/v1/models: refused"
            )
        )
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )

        with self.assertRaises(ExtractorError) as caught:
            extractor.preflight()

        self.assertEqual(
            caught.exception.code,
            FailureCode.ENDPOINT_UNREACHABLE,
        )
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.PRECONDITION,
        )

    def test_run_request_success_preserves_raw_and_overrides_only_model(self) -> None:
        payload = load_example()
        raw_response = response_with_content(
            json.dumps(payload, ensure_ascii=False)
        )
        extractor, client = self.make_extractor(raw_response)
        row = make_request_row(payload)
        frozen_row = copy.deepcopy(row)

        result = extractor.run_request(row)

        self.assertIsInstance(result, Node1Success)
        assert isinstance(result, Node1Success)
        self.assertEqual(result.sample_id, payload["source"]["sample_id"])
        self.assertEqual(result.raw_response, raw_response)
        self.assertEqual(result.canonical_payload, payload)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(row, frozen_row)

        # Raw evidence and canonical payload must not alias.
        self.assertIsNot(result.raw_response, result.canonical_payload)

        assert client.last_request is not None
        self.assertEqual(
            client.last_request["model"],
            "qwen2.5-7b-vanilla",
        )
        self.assertEqual(
            client.last_request["messages"],
            row["request"]["messages"],
        )
        self.assertEqual(
            client.last_request["temperature"],
            row["request"]["temperature"],
        )
        self.assertEqual(
            client.last_request["seed"],
            row["request"]["seed"],
        )
        self.assertEqual(
            client.last_request["response_format"],
            row["request"]["response_format"],
        )

        expected_request = copy.deepcopy(row["request"])
        expected_request["model"] = "qwen2.5-7b-vanilla"
        self.assertEqual(client.last_request, expected_request)

    def test_run_request_transport_error_is_sample_failure(self) -> None:
        client = FakeClient(
            request_error=OpenAICompatibleError("HTTP 500")
        )
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )
        row = make_request_row(load_example())

        result = extractor.run_request(row)

        self.assertIsInstance(result, Node1Failure)
        assert isinstance(result, Node1Failure)
        self.assertEqual(
            result.error_code,
            FailureCode.TRANSPORT_ERROR,
        )
        self.assertEqual(result.stage, ExtractorStage.REQUEST)
        self.assertIsNone(result.raw_response)
        self.assertIsNotNone(result.latency_ms)

    def test_run_request_source_mismatch_is_sample_failure(self) -> None:
        payload = load_example()
        payload["source"]["raw_text"] = "model changed source"
        extractor, _ = self.make_extractor(
            response_with_content(
                json.dumps(payload, ensure_ascii=False)
            )
        )
        row = make_request_row(load_example())

        result = extractor.run_request(row)

        self.assertIsInstance(result, Node1Failure)
        assert isinstance(result, Node1Failure)
        self.assertEqual(
            result.error_code,
            FailureCode.SOURCE_BINDING_ERROR,
        )
        self.assertEqual(
            result.stage,
            ExtractorStage.SOURCE_BINDING,
        )
        self.assertIsNotNone(result.raw_response)

    def test_run_request_schema_error_is_sample_failure(self) -> None:
        payload = load_example()
        payload["document_type"] = "not-valid"
        extractor, _ = self.make_extractor(
            response_with_content(
                json.dumps(payload, ensure_ascii=False)
            )
        )
        row = make_request_row(load_example())

        result = extractor.run_request(row)

        self.assertIsInstance(result, Node1Failure)
        assert isinstance(result, Node1Failure)
        self.assertEqual(
            result.error_code,
            FailureCode.SCHEMA_VALIDATION_ERROR,
        )
        self.assertEqual(result.stage, ExtractorStage.VALIDATION)

    def test_run_request_semantic_error_is_sample_failure(self) -> None:
        payload = load_example()
        payload["invoice"]["issued_date"] = "2020-02-30"
        payload["invoice"]["issued_at"] = "2020-02-30T17:12:00"
        extractor, _ = self.make_extractor(
            response_with_content(
                json.dumps(payload, ensure_ascii=False)
            )
        )
        row = make_request_row(load_example())

        result = extractor.run_request(row)

        self.assertIsInstance(result, Node1Failure)
        assert isinstance(result, Node1Failure)
        self.assertEqual(
            result.error_code,
            FailureCode.SEMANTIC_VALIDATION_ERROR,
        )
        self.assertEqual(result.stage, ExtractorStage.VALIDATION)

    def test_invalid_request_row_is_run_level_dataset_error(self) -> None:
        client = FakeClient()
        extractor = Node1Extractor(
            client=client,  # type: ignore[arg-type]
            model="qwen2.5-7b-vanilla",
            json_schema_path=SCHEMA_PATH,
        )

        with self.assertRaises(ExtractorError) as caught:
            extractor.run_request({"sample_id": "x"})

        self.assertEqual(
            caught.exception.code,
            FailureCode.DATASET_INVALID,
        )
        self.assertEqual(
            caught.exception.stage,
            ExtractorStage.PRECONDITION,
        )


if __name__ == "__main__":
    unittest.main()
