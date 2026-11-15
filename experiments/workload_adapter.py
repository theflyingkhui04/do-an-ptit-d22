from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


REQUIRED_OUTER_FIELDS = (
    "workload_id",
    "dataset_version",
    "sample_id",
    "prompt_version",
)

REQUIRED_REQUEST_FIELDS = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "seed",
)

# Fail fast if a future frozen package adds request semantics that this adapter
# has not been explicitly reviewed to replay.
ALLOWED_REQUEST_FIELDS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "seed",
    "response_format",
}

OPTIONAL_AUDIT_FIELDS = (
    "schema_version",
    "expected_source",
    "img_id",
)

WORKLOAD_LABEL_TO_ID = {
    "W1": "w1_node1_invoice_extraction",
    # [M5-01] W1 on full-receipt OCR text (experiments/configs/w1_v2).
    "W1_V2": "w1_node1_invoice_extraction",
    # [M5-01 canonical flow] W1 on library OCR text without marks (experiments/configs/w1_v3).
    "W1_V3": "w1_node1_invoice_extraction",
    "W2": "w2_text_to_sql",
    "W3": "w3_free_analysis",
}

MANIFEST_REQUIRED_FIELDS = (
    "workload_id",
    "dataset_version",
    "prompt_version",
    "sample_count",
    "model",
    "temperature",
    "max_tokens",
    "seed",
    "artifacts",
    "file_hashes",
)

MANIFEST_GENERATION_FIELDS = (
    "model",
    "temperature",
    "max_tokens",
    "seed",
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class WorkloadContractError(ValueError):
    """Canonical workload package violates the benchmark adapter contract."""


@dataclass(frozen=True)
class WorkloadIntegrity:
    """Verified file-level identity for one selected frozen workload artifact."""

    manifest_path: Path
    artifact_key: str
    expected_sha256: str
    actual_sha256: str
    manifest_package_sha256: str | None
    manifest: dict


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise WorkloadContractError(
            f"package manifest does not exist: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise WorkloadContractError(
            f"package manifest path is not a file: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkloadContractError(
            f"cannot read package manifest {manifest_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WorkloadContractError(
            f"malformed package manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise WorkloadContractError(
            f"package manifest must be a JSON object: {manifest_path}"
        )

    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise WorkloadContractError(
            f"package manifest missing required fields: {missing}"
        )

    if not isinstance(manifest["artifacts"], dict) or not manifest["artifacts"]:
        raise WorkloadContractError(
            "package manifest artifacts must be a non-empty object"
        )

    if not isinstance(manifest["file_hashes"], dict):
        raise WorkloadContractError(
            "package manifest file_hashes must be an object"
        )

    manifest_package_sha256 = manifest.get("package_sha256")
    if (
        manifest_package_sha256 is not None
        and not _is_sha256(manifest_package_sha256)
    ):
        raise WorkloadContractError(
            "package manifest package_sha256 must be a 64-character SHA-256 hex string"
        )

    return manifest


def _artifact_path_matches(
    workload_path: Path,
    manifest_path: Path,
    artifact_value: object,
) -> bool:
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        return False

    declared = Path(artifact_value)

    if declared.is_absolute():
        candidates = [declared]
    else:
        # Real manifests store repository-relative paths such as
        # experiments/configs/w2/chat_requests.jsonl. Walking ancestors makes
        # resolution independent of the caller's current working directory.
        manifest_dir = manifest_path.parent.resolve(strict=True)
        candidates = [root / declared for root in (manifest_dir, *manifest_dir.parents)]

    for candidate in candidates:
        try:
            if candidate.resolve(strict=False) == workload_path:
                return True
        except OSError:
            continue

    return False


def verify_workload_integrity(path: Path) -> WorkloadIntegrity:
    """Verify the selected workload file against its sibling package manifest.

    This verifies file-level SHA-256 from manifest["file_hashes"]. It does not
    recompute package_sha256 because the repository does not currently pin the
    authoritative package-level canonicalization/construction algorithm here.
    """
    try:
        workload_path = path.resolve(strict=True)
    except OSError as exc:
        raise WorkloadContractError(
            f"cannot resolve workload file {path}: {exc}"
        ) from exc

    if not workload_path.is_file():
        raise WorkloadContractError(
            f"workload path is not a file: {workload_path}"
        )

    manifest_path = workload_path.parent / "package_manifest.json"
    manifest = _read_manifest(manifest_path)

    matching_keys: list[str] = []
    for artifact_key, artifact_value in manifest["artifacts"].items():
        if not isinstance(artifact_key, str) or not artifact_key.strip():
            raise WorkloadContractError(
                "package manifest artifact keys must be non-empty strings"
            )
        if _artifact_path_matches(
            workload_path,
            manifest_path,
            artifact_value,
        ):
            matching_keys.append(artifact_key)

    if len(matching_keys) != 1:
        raise WorkloadContractError(
            "selected workload file must match exactly one manifest artifact; "
            f"matched_keys={matching_keys}, workload_file={workload_path}"
        )

    artifact_key = matching_keys[0]
    expected_hash = manifest["file_hashes"].get(artifact_key)
    if not _is_sha256(expected_hash):
        raise WorkloadContractError(
            "manifest file_hashes entry for selected workload artifact must be "
            f"a 64-character SHA-256 hex string: artifact_key={artifact_key!r}"
        )

    try:
        with workload_path.open("rb") as file:
            actual_hash = hashlib.file_digest(file, "sha256").hexdigest()
    except OSError as exc:
        raise WorkloadContractError(
            f"cannot hash workload file {workload_path}: {exc}"
        ) from exc

    expected_hash = expected_hash.lower()
    if actual_hash != expected_hash:
        raise WorkloadContractError(
            "workload SHA-256 mismatch: "
            f"artifact_key={artifact_key!r}, expected={expected_hash}, "
            f"actual={actual_hash}, file={workload_path}"
        )

    manifest_package_sha256 = manifest.get("package_sha256")
    if isinstance(manifest_package_sha256, str):
        manifest_package_sha256 = manifest_package_sha256.lower()

    return WorkloadIntegrity(
        manifest_path=manifest_path,
        artifact_key=artifact_key,
        expected_sha256=expected_hash,
        actual_sha256=actual_hash,
        manifest_package_sha256=manifest_package_sha256,
        manifest=manifest,
    )


def validate_manifest_consistency(
    integrity: WorkloadIntegrity,
    rows: list[dict],
) -> None:
    """Validate manifest metadata/generation contract against loaded rows."""
    if not rows:
        raise WorkloadContractError(
            "cannot validate manifest consistency against an empty workload"
        )

    manifest = integrity.manifest
    first = rows[0]

    for field in ("workload_id", "dataset_version", "prompt_version"):
        manifest_value = manifest.get(field)
        row_value = first.get(field)
        if manifest_value != row_value:
            raise WorkloadContractError(
                "manifest/row contract mismatch: "
                f"field={field!r}, manifest={manifest_value!r}, row={row_value!r}"
            )

    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise WorkloadContractError(
            "package manifest sample_count must be an integer"
        )
    if sample_count != len(rows):
        raise WorkloadContractError(
            "manifest sample_count mismatch: "
            f"manifest={sample_count}, loaded_rows={len(rows)}"
        )

    if "schema_version" in manifest:
        expected_schema = manifest["schema_version"]
        for row_index, row in enumerate(rows):
            if row.get("schema_version") != expected_schema:
                raise WorkloadContractError(
                    "manifest/row schema_version mismatch: "
                    f"row_index={row_index}, manifest={expected_schema!r}, "
                    f"row={row.get('schema_version')!r}"
                )

    for field in MANIFEST_GENERATION_FIELDS:
        expected_value = manifest.get(field)
        for row_index, row in enumerate(rows):
            actual_value = row["request"].get(field)
            if actual_value != expected_value:
                raise WorkloadContractError(
                    "manifest/request generation mismatch: "
                    f"row_index={row_index}, field={field!r}, "
                    f"manifest={expected_value!r}, request={actual_value!r}"
                )

    manifest_has_response_format = "response_format" in manifest
    for row_index, row in enumerate(rows):
        request = row["request"]
        request_has_response_format = "response_format" in request

        if manifest_has_response_format:
            if request.get("response_format") != manifest["response_format"]:
                raise WorkloadContractError(
                    "manifest/request response_format mismatch: "
                    f"row_index={row_index}, "
                    f"manifest={manifest['response_format']!r}, "
                    f"request={request.get('response_format')!r}"
                )
        elif request_has_response_format:
            raise WorkloadContractError(
                "request contains response_format not declared by manifest: "
                f"row_index={row_index}, request={request['response_format']!r}"
            )


def validate_workload_label(
    workload_label: str,
    package_workload_id: str,
) -> str:
    """Validate CLI W1/W2/W3 label against the frozen package workload_id.

    Returns the normalized canonical label (W1/W2/W3) so cell identity is
    stable even if the user typed a lowercase alias.
    """
    if not isinstance(workload_label, str) or not workload_label.strip():
        raise WorkloadContractError(
            "workload label must be a non-empty string"
        )

    if not isinstance(package_workload_id, str) or not package_workload_id.strip():
        raise WorkloadContractError(
            "package workload_id must be a non-empty string"
        )

    canonical_label = workload_label.strip().upper()
    expected_id = WORKLOAD_LABEL_TO_ID.get(canonical_label)

    if expected_id is None:
        allowed = ", ".join(WORKLOAD_LABEL_TO_ID)
        raise WorkloadContractError(
            f"unsupported canonical workload label {workload_label!r}; "
            f"expected one of: {allowed}"
        )

    if package_workload_id != expected_id:
        raise WorkloadContractError(
            "workload label/package mismatch: "
            f"--workload {canonical_label} expects workload_id={expected_id!r}, "
            f"but package declares {package_workload_id!r}"
        )

    return canonical_label


def _validate_non_empty_string(
    value: object,
    field_name: str,
    *,
    line_no: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadContractError(
            f"line {line_no}: {field_name} must be a non-empty string"
        )
    return value


def _validate_request(
    request: object,
    *,
    line_no: int,
) -> dict:
    if not isinstance(request, dict):
        raise WorkloadContractError(
            f"line {line_no}: request must be a JSON object"
        )

    missing = [
        field
        for field in REQUIRED_REQUEST_FIELDS
        if field not in request
    ]
    if missing:
        raise WorkloadContractError(
            f"line {line_no}: request missing required fields: {missing}"
        )

    unknown = sorted(set(request) - ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise WorkloadContractError(
            "line "
            f"{line_no}: unsupported canonical request fields {unknown}. "
            "Adapter review is required before replaying new request semantics."
        )

    _validate_non_empty_string(
        request["model"],
        "request.model",
        line_no=line_no,
    )

    messages = request["messages"]
    if not isinstance(messages, list) or not messages:
        raise WorkloadContractError(
            f"line {line_no}: request.messages must be a non-empty list"
        )

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise WorkloadContractError(
                f"line {line_no}: messages[{message_index}] must be an object"
            )

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not role.strip():
            raise WorkloadContractError(
                f"line {line_no}: messages[{message_index}].role "
                "must be a non-empty string"
            )

        # Current frozen W1/W2/W3 packages are text-only. If multimodal
        # content is introduced later, review the adapter first.
        if not isinstance(content, str):
            raise WorkloadContractError(
                f"line {line_no}: messages[{message_index}].content "
                "must currently be a string"
            )

    temperature = request["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
    ):
        raise WorkloadContractError(
            f"line {line_no}: request.temperature must be numeric"
        )

    seed = request["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise WorkloadContractError(
            f"line {line_no}: request.seed must be an integer"
        )

    max_tokens = request["max_tokens"]
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise WorkloadContractError(
            f"line {line_no}: request.max_tokens must be an integer >= 1"
        )

    if "response_format" in request:
        response_format = request["response_format"]
        if not isinstance(response_format, dict):
            raise WorkloadContractError(
                f"line {line_no}: request.response_format must be an object"
            )

    return request


def load_workload_rows(path: Path) -> list[dict]:
    """Load and validate one frozen workload chat-request JSONL package."""
    if not path.exists():
        raise WorkloadContractError(
            f"workload file does not exist: {path}"
        )

    if not path.is_file():
        raise WorkloadContractError(
            f"workload path is not a file: {path}"
        )

    rows: list[dict] = []
    sample_ids: set[str] = set()
    reference_contract: tuple[str, str, str] | None = None

    try:
        file = path.open(encoding="utf-8")
    except OSError as exc:
        raise WorkloadContractError(
            f"cannot open workload file {path}: {exc}"
        ) from exc

    with file:
        for line_no, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise WorkloadContractError(
                    f"line {line_no}: malformed JSON: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise WorkloadContractError(
                    f"line {line_no}: workload row must be a JSON object"
                )

            for field in REQUIRED_OUTER_FIELDS:
                if field not in row:
                    raise WorkloadContractError(
                        f"line {line_no}: missing outer field {field!r}"
                    )

                _validate_non_empty_string(
                    row[field],
                    field,
                    line_no=line_no,
                )

            sample_id = row["sample_id"]
            if sample_id in sample_ids:
                raise WorkloadContractError(
                    f"line {line_no}: duplicate sample_id {sample_id!r}"
                )
            sample_ids.add(sample_id)

            _validate_request(
                row.get("request"),
                line_no=line_no,
            )

            row_contract = (
                row["workload_id"],
                row["dataset_version"],
                row["prompt_version"],
            )

            if reference_contract is None:
                reference_contract = row_contract
            elif row_contract != reference_contract:
                raise WorkloadContractError(
                    "line "
                    f"{line_no}: workload/dataset/prompt contract differs "
                    f"from first row: {row_contract!r} != "
                    f"{reference_contract!r}"
                )

            rows.append(row)

    if not rows:
        raise WorkloadContractError(
            f"workload file contains no canonical rows: {path}"
        )

    return rows


def resolve_repeat_offset(
    sample_count: int,
    requests_per_repeat: int,
    repeat_index: int,
    *,
    sample_policy: str,
) -> int:
    """Return the deterministic per-repeat sample offset.

    PM-approved canonical policy ``cycle-rotate`` advances the start position
    by the remainder left after each measured repeat:

        offset = ((repeat_index - 1) *
                  (requests_per_repeat % sample_count)) % sample_count

    ``strict`` and legacy/debug ``cycle`` do not rotate and therefore return 0.
    """
    if sample_count < 1:
        raise WorkloadContractError(
            f"sample_count must be >= 1, got {sample_count}"
        )
    if requests_per_repeat < 1:
        raise WorkloadContractError(
            "requests_per_repeat must be >= 1, "
            f"got {requests_per_repeat}"
        )
    if repeat_index < 1:
        raise WorkloadContractError(
            f"repeat_index must be >= 1, got {repeat_index}"
        )

    if sample_policy in {"strict", "cycle"}:
        return 0
    if sample_policy == "cycle-rotate":
        step = requests_per_repeat % sample_count
        return ((repeat_index - 1) * step) % sample_count

    raise WorkloadContractError(
        f"unsupported sample policy: {sample_policy!r}"
    )


def resolve_sample_index(
    sample_count: int,
    request_index: int,
    *,
    sample_policy: str,
    sample_offset: int = 0,
) -> int:
    """Resolve request_index to zero-based canonical sample_index."""
    if sample_count < 1:
        raise WorkloadContractError(
            f"sample_count must be >= 1, got {sample_count}"
        )

    if request_index < 0:
        raise WorkloadContractError(
            f"request_index must be >= 0, got {request_index}"
        )

    if sample_offset < 0 or sample_offset >= sample_count:
        raise WorkloadContractError(
            "sample_offset must satisfy 0 <= sample_offset < sample_count, "
            f"got sample_offset={sample_offset}, sample_count={sample_count}"
        )

    if sample_policy == "strict":
        if sample_offset != 0:
            raise WorkloadContractError(
                "strict sample policy requires sample_offset=0"
            )
        if request_index >= sample_count:
            raise WorkloadContractError(
                "canonical workload exhausted: "
                f"request_index={request_index}, sample_count={sample_count}. "
                "A sample-reuse/expansion policy is required. "
                "Do not silently cycle or random-sample."
            )
        return request_index

    if sample_policy == "cycle":
        if sample_offset != 0:
            raise WorkloadContractError(
                "legacy cycle policy requires sample_offset=0; "
                "use cycle-rotate for deterministic rotating offsets"
            )
        return request_index % sample_count

    if sample_policy == "cycle-rotate":
        return (request_index + sample_offset) % sample_count

    raise WorkloadContractError(
        f"unsupported sample policy: {sample_policy!r}"
    )


def select_workload_row(
    rows: list[dict],
    request_index: int,
    *,
    sample_policy: str,
    sample_offset: int = 0,
) -> dict:
    """Resolve request_index to a canonical workload row before scheduling."""
    if not rows:
        raise WorkloadContractError(
            "cannot select from an empty workload"
        )

    sample_index = resolve_sample_index(
        len(rows),
        request_index,
        sample_policy=sample_policy,
        sample_offset=sample_offset,
    )

    return rows[sample_index]


def build_workload_metadata(
    row: dict,
    *,
    integrity: WorkloadIntegrity | None = None,
) -> dict:
    """Build traceable raw-telemetry metadata from a resolved canonical row."""
    metadata = {
        "workload_id": row["workload_id"],
        "dataset_version": row["dataset_version"],
        "sample_id": row["sample_id"],
        "prompt_version": row["prompt_version"],
    }

    for field in OPTIONAL_AUDIT_FIELDS:
        if field in row:
            metadata[field] = deepcopy(row[field])

    # Preserve the package-declared model for audit even though runtime model
    # binding is intentionally overridden with the served model for each lane.
    metadata["package_model"] = row["request"].get("model")

    if integrity is not None:
        metadata["manifest_package_sha256"] = (
            integrity.manifest_package_sha256
        )
        metadata["workload_artifact_key"] = integrity.artifact_key
        metadata["workload_file_sha256"] = integrity.actual_sha256

    return metadata


def build_request_payload(
    row: dict,
    *,
    runtime_model: str,
) -> dict:
    """Build one concrete OpenAI-compatible HTTP payload for a resolved row."""
    if not isinstance(runtime_model, str) or not runtime_model.strip():
        raise WorkloadContractError(
            "runtime model must be a non-empty served-model name"
        )

    request = row["request"]

    payload = {
        # Runtime lane binding.
        "model": runtime_model,

        # Canonical workload semantics.
        "messages": deepcopy(request["messages"]),
        "temperature": request["temperature"],
        "seed": request["seed"],

        # Canonical field-name normalization only; numeric semantics unchanged.
        "max_completion_tokens": request["max_tokens"],

        # Benchmark execution/measurement invariants.
        "n": 1,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    if "response_format" in request:
        payload["response_format"] = deepcopy(
            request["response_format"]
        )

    return payload
