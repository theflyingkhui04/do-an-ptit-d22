"""Utilities for frozen Workload W1 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


W1_WORKLOAD_ID = "w1_node1_invoice_extraction"
W1_DATASET_VERSION = "w1.m1_03.v1"


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: object) -> str:
    return sha256_text(canonical_json(data))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
