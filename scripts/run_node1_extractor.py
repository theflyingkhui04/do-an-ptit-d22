"""Run frozen W1 Node 1 requests against an OpenAI-compatible endpoint.

This script is intentionally a thin orchestration layer around
``src.pipeline.extractor.Node1Extractor``. Reusable parsing, canonicalization,
source binding, and validation logic must stay in ``src/pipeline`` rather than
being duplicated here.

Typical target-vLLM run:

    python scripts/run_node1_extractor.py \
      --requests experiments/configs/w1/node1_chat_requests.jsonl \
      --package-manifest experiments/configs/w1/package_manifest.json \
      --base-url http://127.0.0.1:8001/v1 \
      --model qwen2.5-7b-vanilla \
      --run-id m1_03_w1_vanilla \
      --output-dir experiments/runs/m1_03_w1_vanilla \
      --report experiments/results/m1_03_node1_vllm_summary.json \
      --no-db \
      --min-schema-valid-rate 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.extractor import (  # noqa: E402
    ExtractorError,
    ExtractorStage,
    FailureCode,
    Node1Extractor,
    Node1Failure,
    Node1Success,
)
from src.pipeline.openai_compatible import OpenAICompatibleClient  # noqa: E402
from src.pipeline.validation import DEFAULT_JSON_SCHEMA_PATH  # noqa: E402

from init_invoice_sqlite import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    connect,
    init_schema,
    insert_extraction,
)


DEFAULT_REQUESTS_PATH = Path(
    "experiments/configs/w1/node1_chat_requests.jsonl"
)
DEFAULT_PACKAGE_MANIFEST_PATH = Path(
    "experiments/configs/w1/package_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/runs/m1_03_w1_vanilla"
)
DEFAULT_REPORT_PATH = Path(
    "experiments/results/m1_03_node1_vllm_summary.json"
)
DEFAULT_DB_PATH = Path(
    "data/processed/m0_04_mcocr_devset/invoices_v1.sqlite"
)

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_API_KEY = ""
DEFAULT_MODEL = "qwen2.5-7b-vanilla"
DEFAULT_RUN_ID = "m1_03_w1_vanilla"
DEFAULT_LANE = "vanilla"
DEFAULT_ACCEPTANCE_THRESHOLD = 0.95


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file whose non-empty lines must be JSON objects."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_optional_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def usage_value(response: dict[str, Any], name: str) -> int:
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    value = usage.get(name, 0)
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, (int, float)) else 0


def rounded_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def summarize_latency(values_ms: list[float]) -> dict[str, float | None]:
    if not values_ms:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": round(statistics.fmean(values_ms), 3),
        "median": round(statistics.median(values_ms), 3),
        "min": round(min(values_ms), 3),
        "max": round(max(values_ms), 3),
    }


def validate_request_package(
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    """Fail before the measured loop if package-wide metadata is inconsistent."""
    if not rows:
        raise RuntimeError("Request package is empty.")

    first = rows[0]
    schema_version = first.get("schema_version")
    prompt_version = first.get("prompt_version")

    if not isinstance(schema_version, str) or not schema_version:
        raise ExtractorError(
            FailureCode.DATASET_INVALID,
            ExtractorStage.PRECONDITION,
            "First request row has no valid schema_version.",
        )
    if not isinstance(prompt_version, str) or not prompt_version:
        raise ExtractorError(
            FailureCode.PROMPT_PACKAGE_INVALID,
            ExtractorStage.PRECONDITION,
            "First request row has no valid prompt_version.",
        )

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    inconsistent_rows: list[str] = []

    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in seen_ids:
                duplicate_ids.append(sample_id)
            seen_ids.add(sample_id)

        if (
            row.get("schema_version") != schema_version
            or row.get("prompt_version") != prompt_version
        ):
            inconsistent_rows.append(str(sample_id))

    if duplicate_ids:
        raise ExtractorError(
            FailureCode.DATASET_INVALID,
            ExtractorStage.PRECONDITION,
            "Duplicate sample_id values: "
            + ", ".join(sorted(set(duplicate_ids))),
        )

    if inconsistent_rows:
        raise ExtractorError(
            FailureCode.PROMPT_PACKAGE_INVALID,
            ExtractorStage.PRECONDITION,
            "Mixed schema/prompt versions in request rows: "
            + ", ".join(inconsistent_rows),
        )

    return schema_version, prompt_version


def prepare_run(
    conn: Any,
    *,
    run_id: str,
    lane: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    package_manifest_path: Path | None,
    replace_run: bool,
) -> None:
    """Prepare the optional SQLite audit run.

    Persistence remains outside Node1Extractor validity. The extractor validates
    successfully before this function's per-sample insert path is reached.
    """
    exists = conn.execute(
        "SELECT EXISTS("
        "SELECT 1 FROM extraction_runs WHERE run_id = ?"
        ")",
        (run_id,),
    ).fetchone()[0]

    if exists and not replace_run:
        raise RuntimeError(
            f"run_id={run_id!r} already exists; "
            "pass --replace-run to rerun it"
        )

    if exists:
        conn.execute(
            "DELETE FROM extraction_runs WHERE run_id = ?",
            (run_id,),
        )

    dataset_manifest_path = (
        package_manifest_path.as_posix()
        if package_manifest_path is not None
        else None
    )

    conn.execute(
        """
        INSERT INTO extraction_runs (
            run_id,
            lane,
            model_name,
            prompt_version,
            schema_version,
            dataset_manifest_path,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            lane,
            model,
            prompt_version,
            schema_version,
            dataset_manifest_path,
            "M1-03 Node 1 run through reusable Node1Extractor",
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


def failure_record(
    result: Node1Failure,
    *,
    run_id: str,
    lane: str,
    model: str,
) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "run_id": run_id,
        "lane": lane,
        "model": model,
        "error_code": result.error_code.value,
        "stage": result.stage.value,
        "message": result.message,
        "latency_ms": (
            round(result.latency_ms, 3)
            if result.latency_ms is not None
            else None
        ),
        "assistant_content": result.assistant_content,
    }


def raw_response_record(
    *,
    sample_id: str,
    latency_ms: float | None,
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "latency_ms": (
            round(latency_ms, 3)
            if latency_ms is not None
            else None
        ),
        "response": response,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_REQUESTS_PATH,
    )
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=DEFAULT_PACKAGE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "PIPELINE_LLM_BASE_URL",
            DEFAULT_BASE_URL,
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "PIPELINE_LLM_API_KEY",
            DEFAULT_API_KEY,
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "PIPELINE_LLM_MODEL",
            DEFAULT_MODEL,
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=0.0,
        help="Seconds to keep retrying HTTP 503 while a Modal lane cold-starts (0 = no retry).",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
    )
    parser.add_argument(
        "--lane",
        default=DEFAULT_LANE,
    )
    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--json-schema",
        type=Path,
        default=DEFAULT_JSON_SCHEMA_PATH,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument(
        "--sql-schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
    )
    parser.add_argument(
        "--replace-run",
        action="store_true",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
    )

    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=DEFAULT_ACCEPTANCE_THRESHOLD,
        help=(
            "Threshold recorded in the summary. "
            "Default: 0.95."
        ),
    )
    parser.add_argument(
        "--min-schema-valid-rate",
        type=float,
        default=None,
        help=(
            "Optional execution gate. If supplied, return exit code 2 when "
            "schema_valid_rate is below this value. "
            "Use 0.95 for the official M1-03 acceptance run."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    if not 0.0 <= args.acceptance_threshold <= 1.0:
        raise ValueError(
            "--acceptance-threshold must be between 0 and 1"
        )

    if (
        args.min_schema_valid_rate is not None
        and not 0.0 <= args.min_schema_valid_rate <= 1.0
    ):
        raise ValueError(
            "--min-schema-valid-rate must be between 0 and 1"
        )

    request_rows = load_jsonl(args.requests)
    if args.limit is not None:
        request_rows = request_rows[: args.limit]

    schema_version, prompt_version = validate_request_package(
        request_rows
    )

    package_manifest = load_optional_json_object(
        args.package_manifest
    )

    client = OpenAICompatibleClient(
        args.base_url,
        args.api_key,
        args.timeout,
        ready_timeout_seconds=args.ready_timeout,
    )
    extractor = Node1Extractor(
        client=client,
        model=args.model,
        json_schema_path=args.json_schema,
    )

    # Run-level gate: an unavailable endpoint/model must not become N sample
    # failures and must not enter the measured denominator.
    extractor.preflight()

    conn = None
    if not args.no_db:
        conn = connect(args.db)
        try:
            init_schema(conn, args.sql_schema)
            prepare_run(
                conn,
                run_id=args.run_id,
                lane=args.lane,
                model=args.model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                package_manifest_path=args.package_manifest,
                replace_run=args.replace_run,
            )
        except Exception:
            conn.close()
            raise

    raw_responses: list[dict[str, Any]] = []
    validated_outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    failure_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []

    prompt_tokens = 0
    completion_tokens = 0
    normalization_count = 0

    fully_validated_count = 0
    persistence_failure_count = 0

    try:
        for index, row in enumerate(request_rows, start=1):
            result = extractor.run_request(row)

            if result.latency_ms is not None:
                latencies_ms.append(result.latency_ms)

            if isinstance(result, Node1Failure):
                failure_counts[result.error_code.value] += 1

                if result.raw_response is not None:
                    raw_responses.append(
                        raw_response_record(
                            sample_id=result.sample_id,
                            latency_ms=result.latency_ms,
                            response=result.raw_response,
                        )
                    )
                    prompt_tokens += usage_value(
                        result.raw_response,
                        "prompt_tokens",
                    )
                    completion_tokens += usage_value(
                        result.raw_response,
                        "completion_tokens",
                    )

                failures.append(
                    failure_record(
                        result,
                        run_id=args.run_id,
                        lane=args.lane,
                        model=args.model,
                    )
                )

                print(
                    f"[{index}/{len(request_rows)}] "
                    f"FAIL {result.sample_id} "
                    f"{result.error_code.value}: {result.message}"
                )
                continue

            if not isinstance(result, Node1Success):
                raise TypeError(
                    f"Unexpected extractor result: {type(result)!r}"
                )

            fully_validated_count += 1
            normalization_count += len(result.normalizations)

            raw_responses.append(
                raw_response_record(
                    sample_id=result.sample_id,
                    latency_ms=result.latency_ms,
                    response=result.raw_response,
                )
            )
            prompt_tokens += usage_value(
                result.raw_response,
                "prompt_tokens",
            )
            completion_tokens += usage_value(
                result.raw_response,
                "completion_tokens",
            )

            invoice_id: str | None = None
            persistence_error: str | None = None

            if conn is not None:
                try:
                    invoice_id = insert_extraction(
                        conn,
                        result.canonical_payload,
                        args.run_id,
                        replace=True,
                        json_schema_path=args.json_schema,
                    )
                except Exception as exc:
                    persistence_failure_count += 1
                    failure_counts[
                        FailureCode.PERSISTENCE_ERROR.value
                    ] += 1
                    persistence_error = str(exc)
                    failures.append(
                        {
                            "sample_id": result.sample_id,
                            "run_id": args.run_id,
                            "lane": args.lane,
                            "model": args.model,
                            "error_code": (
                                FailureCode.PERSISTENCE_ERROR.value
                            ),
                            "stage": ExtractorStage.PERSISTENCE.value,
                            "message": persistence_error,
                            "latency_ms": round(
                                result.latency_ms,
                                3,
                            ),
                            "assistant_content": (
                                result.assistant_content
                            ),
                        }
                    )

            validated_outputs.append(
                {
                    "sample_id": result.sample_id,
                    "run_id": args.run_id,
                    "lane": args.lane,
                    "model": args.model,
                    "invoice_id": invoice_id,
                    "latency_ms": round(
                        result.latency_ms,
                        3,
                    ),
                    "assistant_content": (
                        result.assistant_content
                    ),
                    "normalizations": list(
                        result.normalizations
                    ),
                    "payload": result.canonical_payload,
                    "persistence_error": persistence_error,
                }
            )

            status_suffix = (
                ""
                if persistence_error is None
                else " (validated; persistence failed)"
            )
            print(
                f"[{index}/{len(request_rows)}] "
                f"OK {result.sample_id} "
                f"({result.latency_ms:.0f} ms)"
                f"{status_suffix}"
            )
    finally:
        if conn is not None:
            complete_run(conn, args.run_id)
            conn.close()

    attempted = len(request_rows)

    transport_failures = failure_counts[
        FailureCode.TRANSPORT_ERROR.value
    ]
    response_envelope_failures = failure_counts[
        FailureCode.RESPONSE_ENVELOPE_ERROR.value
    ]
    json_parse_failures = failure_counts[
        FailureCode.JSON_PARSE_ERROR.value
    ]
    source_binding_failures = failure_counts[
        FailureCode.SOURCE_BINDING_ERROR.value
    ]

    transport_success_count = attempted - transport_failures

    strict_json_count = (
        attempted
        - transport_failures
        - response_envelope_failures
        - json_parse_failures
    )

    source_binding_count = (
        strict_json_count
        - source_binding_failures
    )

    schema_valid_rate = rounded_rate(
        fully_validated_count,
        attempted,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = args.output_dir / "raw_responses.jsonl"
    validated_path = (
        args.output_dir / "validated_outputs.jsonl"
    )
    failures_path = args.output_dir / "failures.jsonl"
    summary_path = args.output_dir / "summary.json"

    write_jsonl(raw_path, raw_responses)
    write_jsonl(
        validated_path,
        validated_outputs,
    )
    write_jsonl(failures_path, failures)

    artifact_hashes = None
    package_sha256 = None
    workload_id = None
    dataset_version = None

    if package_manifest is not None:
        artifact_hashes = package_manifest.get(
            "file_hashes"
        )
        package_sha256 = package_manifest.get(
            "package_sha256"
        )
        workload_id = package_manifest.get(
            "workload_id"
        )
        dataset_version = package_manifest.get(
            "dataset_version"
        )

    summary = {
        "task": "M1-03",
        "step": "run-node1-extractor",
        "backend": "openai-compatible",
        "lane": args.lane,
        "base_url": args.base_url,
        "model": args.model,
        "run_id": args.run_id,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "workload_id": workload_id,
        "dataset_version": dataset_version,
        "w1_package_sha256": package_sha256,
        "artifact_hashes": artifact_hashes,
        "request_count": attempted,
        "fully_validated_count": fully_validated_count,
        "sample_failure_count": (
            attempted - fully_validated_count
        ),
        "persistence_failure_count": (
            persistence_failure_count
        ),
        "failure_counts": dict(
            sorted(failure_counts.items())
        ),
        "rates": {
            "transport_success_rate": rounded_rate(
                transport_success_count,
                attempted,
            ),
            "strict_json_rate": rounded_rate(
                strict_json_count,
                attempted,
            ),
            "source_binding_rate": rounded_rate(
                source_binding_count,
                attempted,
            ),
            "schema_valid_rate": schema_valid_rate,
        },
        "acceptance": {
            "threshold": args.acceptance_threshold,
            "pass": (
                schema_valid_rate
                >= args.acceptance_threshold
            ),
            "formula": (
                "fully_validated_count / "
                "request_count"
            ),
        },
        "normalization_count": normalization_count,
        "latency_ms": summarize_latency(
            latencies_ms
        ),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens
            ),
        },
        "database": (
            None
            if args.no_db
            else args.db.as_posix()
        ),
        "outputs": {
            "raw_responses": raw_path.as_posix(),
            "validated_outputs": (
                validated_path.as_posix()
            ),
            "failures": failures_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
    }

    summary_text = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
        newline="\n",
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        summary_text,
        encoding="utf-8",
        newline="\n",
    )

    print(summary_text, end="")

    if (
        args.min_schema_valid_rate is not None
        and schema_valid_rate
        < args.min_schema_valid_rate
    ):
        return 2

    if persistence_failure_count:
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
