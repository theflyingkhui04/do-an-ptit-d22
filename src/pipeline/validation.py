"""Shared validation stack for the frozen invoice_extraction.v1 contract.

Dependency direction is intentionally ``scripts -> src``. Both reusable runtime
code and CLI/SQLite tooling import this module so JSON Schema, semantic, and
Pydantic validation cannot silently drift into separate implementations.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from src.pipeline.invoice_models import validate_invoice_payload


DEFAULT_JSON_SCHEMA_PATH = Path(
    "src/pipeline/schemas/invoice_extraction.schema.json"
)


def load_json_schema(path: Path = DEFAULT_JSON_SCHEMA_PATH) -> dict[str, Any]:
    """Load and validate the JSON Schema document itself."""
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"Expected JSON Schema object in {path}")
    Draft202012Validator.check_schema(schema)
    return schema


def json_schema_validation_errors(
    payload: object,
    schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
) -> list[str]:
    """Return deterministic JSON Schema validation errors."""
    schema = load_json_schema(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in errors
    ]


def semantic_validation_errors(payload: object) -> list[str]:
    """Return cross-field semantic errors not fully expressed by JSON Schema."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return errors

    invoice_value = payload.get("invoice", {})
    invoice = invoice_value if isinstance(invoice_value, dict) else {}

    parsers = (
        ("invoice.issued_date", invoice.get("issued_date"), date.fromisoformat),
        ("invoice.issued_time", invoice.get("issued_time"), time.fromisoformat),
        ("invoice.issued_at", invoice.get("issued_at"), datetime.fromisoformat),
    )
    parsed: dict[str, object] = {}
    for field_path, value, parser in parsers:
        if value is None:
            continue
        try:
            parsed[field_path] = parser(value)
        except (TypeError, ValueError):
            errors.append(f"{field_path}: invalid calendar value {value!r}")

    issued_at = parsed.get("invoice.issued_at")
    issued_date = parsed.get("invoice.issued_date")
    issued_time = parsed.get("invoice.issued_time")
    if (
        isinstance(issued_at, datetime)
        and isinstance(issued_date, date)
        and issued_at.date() != issued_date
    ):
        errors.append(
            "invoice.issued_at: date component conflicts with invoice.issued_date"
        )
    if (
        isinstance(issued_at, datetime)
        and isinstance(issued_time, time)
        and issued_at.time() != issued_time
    ):
        errors.append(
            "invoice.issued_at: time component conflicts with invoice.issued_time"
        )

    source_value = payload.get("source", {})
    source = source_value if isinstance(source_value, dict) else {}
    source_raw_text = source.get("raw_text")

    extraction_value = payload.get("extraction", {})
    extraction = extraction_value if isinstance(extraction_value, dict) else {}
    evidence_value = extraction.get("field_evidence", [])
    evidence_rows = evidence_value if isinstance(evidence_value, list) else []
    if isinstance(source_raw_text, str):
        for index, evidence in enumerate(evidence_rows):
            if not isinstance(evidence, dict):
                continue
            evidence_text = evidence.get("raw_text")
            if (
                isinstance(evidence_text, str)
                and evidence_text not in source_raw_text
            ):
                errors.append(
                    "extraction.field_evidence"
                    f"[{index}].raw_text: not a verbatim source substring"
                )

    items_value = payload.get("items", [])
    items = items_value if isinstance(items_value, list) else []
    line_numbers = [
        item.get("line_number")
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("line_number"), int)
    ]
    if len(line_numbers) != len(set(line_numbers)):
        errors.append("items: line_number values must be unique")

    return errors


def pydantic_validation_errors(payload: object) -> list[str]:
    """Return Pydantic mirror errors without changing frozen coercion policy."""
    try:
        validate_invoice_payload(payload)
    except ValidationError as exc:
        details: list[str] = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error.get("loc", ())) or "$"
            details.append(f"{location}: {error.get('msg', 'validation error')}")
        return details
    return []


def validate_payload(
    payload: object,
    json_schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
) -> None:
    """Validate one payload through the single shared M1-03 validation stack.

    Order is JSON Schema -> semantic checks -> Pydantic mirror. The Pydantic
    mirror remains subordinate to the frozen JSON Schema contract and does not
    enable a new global strict-mode policy.
    """
    schema_errors = json_schema_validation_errors(payload, json_schema_path)
    semantic_errors = semantic_validation_errors(payload)
    pydantic_errors = pydantic_validation_errors(payload)

    details: list[str] = []
    details.extend(f"json_schema: {error}" for error in schema_errors)
    details.extend(f"semantic: {error}" for error in semantic_errors)
    details.extend(f"pydantic: {error}" for error in pydantic_errors)
    if details:
        raise ValueError(
            "Payload does not match invoice_extraction.v1: "
            + "; ".join(details)
        )
