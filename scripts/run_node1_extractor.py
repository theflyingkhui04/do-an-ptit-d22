"""Run Node 1 requests against an OpenAI-compatible LLM and load valid JSON into SQLite."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.openai_compatible import OpenAICompatibleClient  # noqa: E402

from init_invoice_sqlite import (  # noqa: E402
    DEFAULT_JSON_SCHEMA_PATH,
    DEFAULT_SCHEMA_PATH,
    connect,
    init_schema,
    insert_extraction,
    validate_payload,
)


DEFAULT_REQUESTS_PATH = Path("data/processed/m0_04_mcocr_devset/node1/node1_chat_requests.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/m0_04_mcocr_devset/node1/ollama_qwen2_5_7b")
DEFAULT_REPORT_PATH = Path("experiments/results/m0_04_node1_ollama_summary.json")
DEFAULT_DB_PATH = Path("data/processed/m0_04_mcocr_devset/invoices_v1.sqlite")
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_RUN_ID = "m0_04_node1_ollama_qwen2_5_7b"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_assistant_payload(response: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Response does not contain choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Assistant content is empty")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Assistant content is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Assistant content must decode to one JSON object")
    return payload, content


def validate_source_binding(payload: dict[str, Any], expected_source: object) -> None:
    if not isinstance(expected_source, dict):
        raise ValueError("Request envelope is missing expected_source")
    actual_source = payload.get("source")
    if actual_source == expected_source:
        return
    mismatched = [
        key
        for key in ("sample_id", "img_id", "image_path", "raw_text")
        if not isinstance(actual_source, dict) or actual_source.get(key) != expected_source.get(key)
    ]
    raise ValueError(f"Source binding mismatch for: {', '.join(mismatched)}")


def normalize_canonical_payload(
    payload: dict[str, Any], expected_source: object
) -> list[str]:
    """Apply auditable structural fixes without inventing business values."""
    normalizations = []
    invoice = payload.get("invoice")
    if isinstance(invoice, dict):
        issued_date = invoice.get("issued_date")
        issued_time = invoice.get("issued_time")
        if (
            invoice.get("issued_at") is None
            and isinstance(issued_date, str)
            and isinstance(issued_time, str)
        ):
            normalized_time = f"{issued_time}:00" if len(issued_time) == 5 else issued_time
            invoice["issued_at"] = f"{issued_date}T{normalized_time}"
            normalizations.append("invoice.issued_at composed from issued_date and issued_time")

    totals = payload.get("totals")
    if isinstance(totals, dict):
        for field_name in ("subtotal", "discount", "vat_amount", "total_amount"):
            if totals.get(field_name) is None:
                totals[field_name] = {"raw": None, "value": None, "currency": None}
                normalizations.append(
                    f"totals.{field_name} replaced null with an empty money object"
                )

    extraction = payload.get("extraction")
    evidence_rows = extraction.get("field_evidence") if isinstance(extraction, dict) else None
    source_text = expected_source.get("raw_text") if isinstance(expected_source, dict) else None
    if isinstance(evidence_rows, list):
        retained_evidence = []
        for index, evidence in enumerate(evidence_rows):
            if not isinstance(evidence, dict):
                retained_evidence.append(evidence)
                continue
            if "source_label" not in evidence:
                evidence["source_label"] = None
                normalizations.append(
                    f"extraction.field_evidence[{index}].source_label defaulted to null"
                )
            raw_text = evidence.get("raw_text")
            if (
                isinstance(source_text, str)
                and isinstance(raw_text, str)
                and raw_text not in source_text
            ):
                normalizations.append(
                    f"extraction.field_evidence[{index}] removed: raw_text is not a source substring"
                )
                continue
            retained_evidence.append(evidence)
        extraction["field_evidence"] = retained_evidence
    return normalizations


def prepare_run(
    conn: Any,
    run_id: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    replace_run: bool,
) -> None:
    exists = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM extraction_runs WHERE run_id = ?)",
        (run_id,),
    ).fetchone()[0]
    if exists and not replace_run:
        raise RuntimeError(f"run_id={run_id!r} already exists; pass --replace-run to rerun it")
    if exists:
        conn.execute("DELETE FROM extraction_runs WHERE run_id = ?", (run_id,))
    conn.execute(
        """
        INSERT INTO extraction_runs (
            run_id, lane, model_name, prompt_version, schema_version,
            dataset_manifest_path, notes
        )
        VALUES (?, 'vanilla', ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            model,
            prompt_version,
            schema_version,
            "experiments/results/m0_04_mcocr_devset_manifest.csv",
            "Local Ollama PoC through the OpenAI-compatible API",
        ),
    )
    conn.commit()


def complete_run(conn: Any, run_id: str) -> None:
    conn.execute(
        """
        UPDATE extraction_runs
        SET completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE run_id = ?
        """,
        (run_id,),
    )
    conn.commit()


def usage_value(response: dict[str, Any], name: str) -> int:
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    value = usage.get(name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--sql-schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--json-schema", type=Path, default=DEFAULT_JSON_SCHEMA_PATH)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PIPELINE_LLM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PIPELINE_LLM_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PIPELINE_LLM_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replace-run", action="store_true")
    parser.add_argument("--no-db", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_rows = load_jsonl(args.requests)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        request_rows = request_rows[: args.limit]
    if not request_rows:
        raise RuntimeError(f"No requests found in {args.requests}")

    client = OpenAICompatibleClient(args.base_url, args.api_key, args.timeout)
    available_models = client.list_models()
    if args.model not in available_models:
        raise RuntimeError(
            f"Model {args.model!r} is not available at {args.base_url}. "
            f"Available models: {available_models}"
        )

    first_row = request_rows[0]
    schema_version = first_row["schema_version"]
    prompt_version = first_row["prompt_version"]
    inconsistent_rows = [
        row.get("sample_id")
        for row in request_rows
        if row.get("schema_version") != schema_version
        or row.get("prompt_version") != prompt_version
    ]
    if inconsistent_rows:
        raise ValueError(f"Mixed schema/prompt versions in request rows: {inconsistent_rows}")

    conn = None
    if not args.no_db:
        conn = connect(args.db)
        try:
            init_schema(conn, args.sql_schema)
            prepare_run(
                conn,
                args.run_id,
                args.model,
                prompt_version,
                schema_version,
                args.replace_run,
            )
        except Exception:
            conn.close()
            raise

    raw_responses = []
    validated_outputs = []
    failures = []
    latencies_ms = []
    prompt_tokens = 0
    completion_tokens = 0
    normalization_count = 0

    try:
        for index, row in enumerate(request_rows, start=1):
            sample_id = row.get("sample_id")
            request_payload = dict(row["request"])
            request_payload["model"] = args.model
            started = time.perf_counter()
            try:
                response = client.chat_completions(request_payload)
                latency_ms = (time.perf_counter() - started) * 1000
                raw_responses.append(
                    {
                        "sample_id": sample_id,
                        "latency_ms": round(latency_ms, 3),
                        "response": response,
                    }
                )
                latencies_ms.append(latency_ms)
                prompt_tokens += usage_value(response, "prompt_tokens")
                completion_tokens += usage_value(response, "completion_tokens")
                payload, content = parse_assistant_payload(response)
                normalizations = normalize_canonical_payload(
                    payload, row.get("expected_source")
                )
                validate_source_binding(payload, row.get("expected_source"))
                if conn is not None:
                    invoice_id = insert_extraction(
                        conn,
                        payload,
                        args.run_id,
                        replace=True,
                        json_schema_path=args.json_schema,
                    )
                else:
                    validate_payload(payload, args.json_schema)
                    invoice_id = None

                validated_outputs.append(
                    {
                        "sample_id": sample_id,
                        "invoice_id": invoice_id,
                        "latency_ms": round(latency_ms, 3),
                        "assistant_content": content,
                        "normalizations": normalizations,
                        "payload": payload,
                    }
                )
                normalization_count += len(normalizations)
                print(f"[{index}/{len(request_rows)}] OK {sample_id} ({latency_ms:.0f} ms)")
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                failures.append(
                    {
                        "sample_id": sample_id,
                        "latency_ms": round(latency_ms, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[{index}/{len(request_rows)}] FAIL {sample_id}: {exc}")
    finally:
        if conn is not None:
            complete_run(conn, args.run_id)
            conn.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "raw_responses.jsonl", raw_responses)
    write_jsonl(args.output_dir / "validated_outputs.jsonl", validated_outputs)
    write_jsonl(args.output_dir / "failures.jsonl", failures)

    summary = {
        "task": "M0-04",
        "step": "7-run-node1-extractor",
        "backend": "openai-compatible",
        "base_url": args.base_url,
        "model": args.model,
        "run_id": args.run_id,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "request_count": len(request_rows),
        "success_count": len(validated_outputs),
        "failure_count": len(failures),
        "success_rate": round(len(validated_outputs) / len(request_rows), 4),
        "normalization_count": normalization_count,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else None,
            "median": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
            "min": round(min(latencies_ms), 3) if latencies_ms else None,
            "max": round(max(latencies_ms), 3) if latencies_ms else None,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "database": None if args.no_db else args.db.as_posix(),
        "outputs": {
            "raw_responses": (args.output_dir / "raw_responses.jsonl").as_posix(),
            "validated_outputs": (args.output_dir / "validated_outputs.jsonl").as_posix(),
            "failures": (args.output_dir / "failures.jsonl").as_posix(),
        },
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
