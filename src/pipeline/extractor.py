"""Reusable runtime core for M1-03 Node 1 invoice extraction."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.pipeline.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleError,
)
from src.pipeline.validation import (
    DEFAULT_JSON_SCHEMA_PATH,
    json_schema_validation_errors,
    pydantic_validation_errors,
    semantic_validation_errors,
)


class FailureCode(str, Enum):
    """Stable machine-readable failure codes for Node 1."""

    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    RESPONSE_ENVELOPE_ERROR = "RESPONSE_ENVELOPE_ERROR"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    SOURCE_BINDING_ERROR = "SOURCE_BINDING_ERROR"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    SEMANTIC_VALIDATION_ERROR = "SEMANTIC_VALIDATION_ERROR"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"

    ENDPOINT_UNREACHABLE = "ENDPOINT_UNREACHABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    BACKEND_CONTRACT_UNSUPPORTED = "BACKEND_CONTRACT_UNSUPPORTED"
    DATASET_INVALID = "DATASET_INVALID"
    PROMPT_PACKAGE_INVALID = "PROMPT_PACKAGE_INVALID"


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
    """Structured run-level/internal error carrying a stable code and stage."""

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


@dataclass(frozen=True)
class Node1Success:
    """Successful per-sample result after full validation."""

    sample_id: str
    latency_ms: float
    raw_response: dict[str, Any]
    assistant_content: str
    canonical_payload: dict[str, Any]
    normalizations: tuple[str, ...]


@dataclass(frozen=True)
class Node1Failure:
    """Stable per-sample failure result after a request was attempted."""

    sample_id: str
    error_code: FailureCode
    stage: ExtractorStage
    message: str
    latency_ms: float | None
    raw_response: dict[str, Any] | None = None
    assistant_content: str | None = None


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def parse_assistant_payload(
    response: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Extract and strictly decode ``choices[0].message.content``."""
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
    """Require model-returned ``source`` to equal the trusted request source."""
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

    raise ExtractorError(
        FailureCode.SOURCE_BINDING_ERROR,
        ExtractorStage.SOURCE_BINDING,
        "Source binding mismatch: "
        + ("; ".join(details) or "source objects are not exactly equal"),
    )


def normalize_canonical_payload(
    payload: dict[str, Any],
    expected_source: object,
) -> list[str]:
    """Apply only structural repairs already allowed by M1-03."""
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


class Node1Extractor:
    """Reusable Node 1 runtime core around an OpenAI-compatible client."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        json_schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.client = client
        self.model = model
        self.json_schema_path = json_schema_path

    def preflight(self) -> None:
        """Fail fast before the sample loop if endpoint/model is unusable."""
        try:
            models = self.client.list_models()
        except OpenAICompatibleError as exc:
            message = str(exc)
            if message.startswith("Cannot reach "):
                code = FailureCode.ENDPOINT_UNREACHABLE
            else:
                code = FailureCode.BACKEND_CONTRACT_UNSUPPORTED
            raise ExtractorError(
                code,
                ExtractorStage.PRECONDITION,
                message,
            ) from exc

        if self.model not in models:
            raise ExtractorError(
                FailureCode.MODEL_NOT_FOUND,
                ExtractorStage.PRECONDITION,
                f"Model {self.model!r} is not available. Available models: {models}",
            )

    def _validate_request_row(
        self,
        request_row: object,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if not isinstance(request_row, dict):
            raise ExtractorError(
                FailureCode.DATASET_INVALID,
                ExtractorStage.PRECONDITION,
                "request_row must be an object.",
            )

        sample_id = request_row.get("sample_id")
        request = request_row.get("request")
        expected_source = request_row.get("expected_source")

        if not isinstance(sample_id, str) or not sample_id:
            raise ExtractorError(
                FailureCode.DATASET_INVALID,
                ExtractorStage.PRECONDITION,
                "request_row.sample_id must be a non-empty string.",
            )
        if not isinstance(request, dict):
            raise ExtractorError(
                FailureCode.DATASET_INVALID,
                ExtractorStage.PRECONDITION,
                f"{sample_id}: request_row.request must be an object.",
            )
        if not isinstance(expected_source, dict):
            raise ExtractorError(
                FailureCode.DATASET_INVALID,
                ExtractorStage.PRECONDITION,
                f"{sample_id}: request_row.expected_source must be an object.",
            )
        if expected_source.get("sample_id") != sample_id:
            raise ExtractorError(
                FailureCode.DATASET_INVALID,
                ExtractorStage.PRECONDITION,
                f"{sample_id}: expected_source.sample_id does not match request_row.sample_id.",
            )
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ExtractorError(
                FailureCode.PROMPT_PACKAGE_INVALID,
                ExtractorStage.PRECONDITION,
                f"{sample_id}: request.messages must be a non-empty list.",
            )
        if request.get("stream") not in (None, False):
            raise ExtractorError(
                FailureCode.BACKEND_CONTRACT_UNSUPPORTED,
                ExtractorStage.PRECONDITION,
                f"{sample_id}: Node1Extractor correctness path requires non-streaming requests.",
            )

        return sample_id, request, expected_source

    @staticmethod
    def _failure(
        sample_id: str,
        error: ExtractorError,
        latency_ms: float | None,
        raw_response: dict[str, Any] | None = None,
        assistant_content: str | None = None,
    ) -> Node1Failure:
        return Node1Failure(
            sample_id=sample_id,
            error_code=error.code,
            stage=error.stage,
            message=error.message,
            latency_ms=latency_ms,
            raw_response=raw_response,
            assistant_content=assistant_content,
        )

    def run_request(
        self,
        request_row: dict[str, Any],
    ) -> Node1Success | Node1Failure:
        """Run one frozen request through parse -> normalize -> validate.

        Request-row/package errors are raised as run-level ExtractorError so
        malformed datasets are not counted as extraction failures. Once the
        server request is attempted, observable model/transport failures are
        returned as Node1Failure and remain in the measured denominator.
        """
        sample_id, request, expected_source = self._validate_request_row(
            request_row
        )

        request_payload = copy.deepcopy(request)
        request_payload["model"] = self.model

        started = time.perf_counter()
        try:
            response = self.client.chat_completions(request_payload)
        except OpenAICompatibleError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            error = ExtractorError(
                FailureCode.TRANSPORT_ERROR,
                ExtractorStage.REQUEST,
                str(exc),
            )
            return self._failure(sample_id, error, latency_ms)

        latency_ms = (time.perf_counter() - started) * 1000
        raw_response = copy.deepcopy(response)
        assistant_content: str | None = None

        try:
            payload, assistant_content = parse_assistant_payload(response)
            normalizations = normalize_canonical_payload(
                payload,
                expected_source,
            )
            validate_source_binding(payload, expected_source)
        except ExtractorError as exc:
            return self._failure(
                sample_id,
                exc,
                latency_ms,
                raw_response=raw_response,
                assistant_content=assistant_content,
            )

        schema_errors = json_schema_validation_errors(
            payload,
            self.json_schema_path,
        )
        if schema_errors:
            error = ExtractorError(
                FailureCode.SCHEMA_VALIDATION_ERROR,
                ExtractorStage.VALIDATION,
                "; ".join(schema_errors),
            )
            return self._failure(
                sample_id,
                error,
                latency_ms,
                raw_response=raw_response,
                assistant_content=assistant_content,
            )

        semantic_errors = semantic_validation_errors(payload)
        if semantic_errors:
            error = ExtractorError(
                FailureCode.SEMANTIC_VALIDATION_ERROR,
                ExtractorStage.VALIDATION,
                "; ".join(semantic_errors),
            )
            return self._failure(
                sample_id,
                error,
                latency_ms,
                raw_response=raw_response,
                assistant_content=assistant_content,
            )

        pydantic_errors = pydantic_validation_errors(payload)
        if pydantic_errors:
            error = ExtractorError(
                FailureCode.SCHEMA_VALIDATION_ERROR,
                ExtractorStage.VALIDATION,
                "Pydantic mirror rejected payload: "
                + "; ".join(pydantic_errors),
            )
            return self._failure(
                sample_id,
                error,
                latency_ms,
                raw_response=raw_response,
                assistant_content=assistant_content,
            )

        return Node1Success(
            sample_id=sample_id,
            latency_ms=latency_ms,
            raw_response=raw_response,
            assistant_content=assistant_content,
            canonical_payload=payload,
            normalizations=tuple(normalizations),
        )
