"""Reusable pure-core helpers for M1-03 Node 1 invoice extraction.

This module intentionally contains only deterministic, offline-testable logic:
response-envelope parsing, strict JSON decoding, safe canonicalization, and
source-binding verification.

HTTP transport, preflight, schema/Pydantic validation, result envelopes, and
SQLite persistence are added in later integration layers.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class FailureCode(str, Enum):
    """Stable machine-readable failure codes for Node 1."""

    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    RESPONSE_ENVELOPE_ERROR = "RESPONSE_ENVELOPE_ERROR"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    SOURCE_BINDING_ERROR = "SOURCE_BINDING_ERROR"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    SEMANTIC_VALIDATION_ERROR = "SEMANTIC_VALIDATION_ERROR"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"


class ExtractorStage(str, Enum):
    """Stable pipeline stage identifiers for Node 1 failures."""

    PRECONDITION = "PRECONDITION"
    REQUEST = "REQUEST"
    PARSE = "PARSE"
    NORMALIZE = "NORMALIZE"
    SOURCE_BINDING = "SOURCE_BINDING"
    VALIDATION = "VALIDATION"
    PERSISTENCE = "PERSISTENCE"


class ExtractorError(RuntimeError):
    """Structured internal error carrying a stable failure code and stage."""

    def __init__(
        self,
        code: FailureCode,
        stage: ExtractorStage,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message


def _reject_non_finite_json_constant(value: str) -> None:
    """Reject NaN/Infinity/-Infinity because they are not valid strict JSON."""

    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def parse_assistant_payload(
    response: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Extract and strictly decode ``choices[0].message.content``.

    The assistant content must itself be exactly one JSON object. Markdown
    fences, surrounding prose, JSON arrays, non-finite numeric constants,
    empty content, and malformed response envelopes are rejected rather than
    repaired.

    Returns:
        ``(payload, assistant_content)`` where ``assistant_content`` is the
        exact string received in the decoded response envelope.

    Raises:
        ExtractorError: with RESPONSE_ENVELOPE_ERROR or JSON_PARSE_ERROR.
    """
    try:
        choices = response["choices"]
        if not isinstance(choices, list) or not choices:
            raise TypeError("choices must be a non-empty list")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise TypeError("choices[0] must be an object")
        message = first_choice["message"]
        if not isinstance(message, dict):
            raise TypeError("choices[0].message must be an object")
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtractorError(
            FailureCode.RESPONSE_ENVELOPE_ERROR,
            ExtractorStage.PARSE,
            "Response does not contain a valid choices[0].message.content string.",
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise ExtractorError(
            FailureCode.RESPONSE_ENVELOPE_ERROR,
            ExtractorStage.PARSE,
            "Assistant content is empty or not a string.",
        )

    try:
        payload = json.loads(
            content,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractorError(
            FailureCode.JSON_PARSE_ERROR,
            ExtractorStage.PARSE,
            f"Assistant content is not strict JSON: {exc}",
        ) from exc

    if not isinstance(payload, dict):
        raise ExtractorError(
            FailureCode.JSON_PARSE_ERROR,
            ExtractorStage.PARSE,
            "Assistant content must decode to exactly one JSON object.",
        )

    return payload, content


def validate_source_binding(
    payload: dict[str, Any],
    expected_source: object,
) -> None:
    """Require model-returned ``source`` to equal the trusted request source.

    This function never overwrites ``payload["source"]`` with trusted data.
    Any missing key, unexpected key, or value mismatch is evidence of a source
    contract violation and is rejected.
    """
    if not isinstance(expected_source, dict):
        raise ExtractorError(
            FailureCode.SOURCE_BINDING_ERROR,
            ExtractorStage.SOURCE_BINDING,
            "Request envelope is missing a valid expected_source object.",
        )

    actual_source = payload.get("source")
    if actual_source == expected_source:
        return

    if not isinstance(actual_source, dict):
        raise ExtractorError(
            FailureCode.SOURCE_BINDING_ERROR,
            ExtractorStage.SOURCE_BINDING,
            "Source binding mismatch: payload.source is missing or not an object.",
        )

    expected_keys = set(expected_source)
    actual_keys = set(actual_source)

    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    mismatched_values = sorted(
        key
        for key in expected_keys & actual_keys
        if actual_source.get(key) != expected_source.get(key)
    )

    details: list[str] = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(missing_keys)}")
    if unexpected_keys:
        details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
    if mismatched_values:
        details.append(f"value mismatch: {', '.join(mismatched_values)}")

    diagnostic = "; ".join(details) or "source objects are not exactly equal"
    raise ExtractorError(
        FailureCode.SOURCE_BINDING_ERROR,
        ExtractorStage.SOURCE_BINDING,
        f"Source binding mismatch: {diagnostic}",
    )


def normalize_canonical_payload(
    payload: dict[str, Any],
    expected_source: object,
) -> list[str]:
    """Apply only the structural repairs already allowed by M1-03.

    Allowed repairs:
    - compose ``invoice.issued_at`` from existing issued_date + issued_time;
    - replace null total money containers with explicit empty Money objects;
    - default missing ``field_evidence[].source_label`` to null;
    - remove field-evidence rows whose raw_text is not a source substring.

    The function MUST NOT modify source provenance or infer/change business
    facts such as seller, buyer, item, amount, tax, identifiers, or labels.
    """
    normalizations: list[str] = []

    invoice = payload.get("invoice")
    if isinstance(invoice, dict):
        issued_date = invoice.get("issued_date")
        issued_time = invoice.get("issued_time")
        if (
            invoice.get("issued_at") is None
            and isinstance(issued_date, str)
            and isinstance(issued_time, str)
        ):
            normalized_time = (
                f"{issued_time}:00" if len(issued_time) == 5 else issued_time
            )
            invoice["issued_at"] = f"{issued_date}T{normalized_time}"
            normalizations.append(
                "invoice.issued_at composed from issued_date and issued_time"
            )

    totals = payload.get("totals")
    if isinstance(totals, dict):
        for field_name in (
            "subtotal",
            "discount",
            "vat_amount",
            "total_amount",
        ):
            if totals.get(field_name) is None:
                totals[field_name] = {
                    "raw": None,
                    "value": None,
                    "currency": None,
                }
                normalizations.append(
                    f"totals.{field_name} replaced null with an empty money object"
                )

    extraction = payload.get("extraction")
    evidence_rows = (
        extraction.get("field_evidence")
        if isinstance(extraction, dict)
        else None
    )
    source_text = (
        expected_source.get("raw_text")
        if isinstance(expected_source, dict)
        else None
    )

    if isinstance(evidence_rows, list):
        retained_evidence: list[Any] = []

        for index, evidence in enumerate(evidence_rows):
            if not isinstance(evidence, dict):
                retained_evidence.append(evidence)
                continue

            if "source_label" not in evidence:
                evidence["source_label"] = None
                normalizations.append(
                    "extraction.field_evidence"
                    f"[{index}].source_label defaulted to null"
                )

            raw_text = evidence.get("raw_text")
            if (
                isinstance(source_text, str)
                and isinstance(raw_text, str)
                and raw_text not in source_text
            ):
                normalizations.append(
                    "extraction.field_evidence"
                    f"[{index}] removed: raw_text is not a source substring"
                )
                continue

            retained_evidence.append(evidence)

        extraction["field_evidence"] = retained_evidence

    return normalizations
