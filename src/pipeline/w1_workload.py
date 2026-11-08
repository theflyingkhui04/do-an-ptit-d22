"""Utilities for frozen Workload W1 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


W1_WORKLOAD_ID = "w1_node1_invoice_extraction"
W1_DATASET_VERSION = "w1.m1_03.v1"
# [M5-01] Full-receipt OCR text instead of the four labelled MC-OCR lines.
W1_V2_DATASET_VERSION = "w1.m5_01.v2"
W1_V2_PROMPT_VERSION = "node1_extractor.v2"
W1_DATASET_VERSIONS = (W1_DATASET_VERSION, W1_V2_DATASET_VERSION)
# Every artifact is hashed after normalizing line endings to LF.
HASH_POLICY_LF = "lf-canonical"

# W1 v1 was frozen from artifacts produced/checked out with mixed newline
# representations. Preserve that frozen identity explicitly so the same
# package hashes are reproduced on every operating system.
W1_V1_ARTIFACT_LINE_ENDINGS = {
    "samples_jsonl": "\n",
    "provenance_jsonl": "\n",
    "manifest_csv": "\r\n",
    "node1_inputs_jsonl": "\n",
    "node1_chat_requests_jsonl": "\n",
    "system_prompt": "\n",
    "json_schema": "\r\n",
}


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: object) -> str:
    return sha256_text(canonical_json(data))


def file_sha256(path: Path) -> str:
    """Hash exact file bytes.

    Keep this helper for artifacts whose byte representation itself is the
    identity. W1 v1 package artifacts should use ``w1_v1_artifact_sha256``
    instead so their frozen cross-platform serialization policy is explicit.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_file_sha256(path: Path, line_ending: str = "\n") -> str:
    """Hash UTF-8 text after canonicalizing all line endings.

    ``newline=None`` enables universal-newline reading, translating CRLF, CR,
    and LF to ``\n``. The text is then serialized with the requested canonical
    line ending before hashing.
    """
    if line_ending not in {"\n", "\r\n"}:
        raise ValueError("line_ending must be LF or CRLF")

    with path.open("r", encoding="utf-8", newline=None) as file:
        text = file.read()

    canonical = text.replace("\n", line_ending)
    return sha256_text(canonical)


def w1_v1_artifact_sha256(name: str, path: Path) -> str:
    """Hash one frozen W1 v1 artifact using its canonical serialization.

    W1 v1 predates an explicit hash-policy version and therefore contains a
    mixture of LF- and CRLF-canonical text artifacts. This mapping preserves
    the already-frozen package identity while making verification independent
    of the current checkout OS and Git ``core.autocrlf`` setting.
    """
    try:
        line_ending = W1_V1_ARTIFACT_LINE_ENDINGS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown W1 v1 artifact: {name}") from exc

    return canonical_text_file_sha256(path, line_ending=line_ending)


def artifact_sha256(name: str, path: Path, hash_policy: str | None) -> str:
    """Hash a W1 package artifact with the package's declared policy.

    A manifest without ``hash_policy`` is a frozen W1 v1 package.
    """
    if hash_policy is None:
        return w1_v1_artifact_sha256(name, path)
    if hash_policy == HASH_POLICY_LF:
        return canonical_text_file_sha256(path, line_ending="\n")
    raise ValueError(f"Unknown W1 hash_policy: {hash_policy!r}")


def compose_node1_prompt_v2(v1_prompt: str, full_receipt_rules: str) -> str:
    """Prompt node1_extractor.v2 = the frozen v1 prompt + the full-receipt rules."""
    return v1_prompt.strip() + "\n\n" + full_receipt_rules.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def request_semantic_view(request_row: dict[str, Any]) -> dict[str, Any]:
    request = request_row.get("request")
    if not isinstance(request, dict):
        raise ValueError("request row is missing request object")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("W1 request must contain exactly system and user messages")
    return {
        "sample_id": request_row.get("sample_id"),
        "schema_version": request_row.get("schema_version"),
        "prompt_version": request_row.get("prompt_version"),
        "expected_source_sha256": sha256_json(request_row.get("expected_source")),
        "system_prompt_sha256": sha256_text(str(messages[0].get("content", ""))),
        "user_content_sha256": sha256_text(str(messages[1].get("content", ""))),
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "seed": request.get("seed"),
        "response_format": request.get("response_format"),
    }


def compute_package_sha256(file_hashes: dict[str, str], metadata: dict[str, Any]) -> str:
    return sha256_json({"file_hashes": file_hashes, "metadata": metadata})
