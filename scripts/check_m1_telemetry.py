"""Validate M1 normalized telemetry records for Interface A handoff."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


DEFAULT_SCHEMA = Path("experiments/configs/m1_request_telemetry.schema.json")
DEFAULT_REPORT = Path("experiments/results/m1_01_telemetry_contract_summary.json")
SCHEMA_VERSION = "m1_request_telemetry.v1"
TELEMETRY_TRANSPORT = "vllm_stream_usage_metrics.summary"
STATIC_NUM_SPEC_TOKENS = 3
FLOAT_TOLERANCE = 1e-6


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
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
    value = load_json(path)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"Expected JSON object/list in {path}")


def finite_number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: NaN/Infinity is not allowed")
    return result


def close_enough(actual: object, expected: float, path: str) -> None:
    actual_float = finite_number(actual, path)
    if abs(actual_float - expected) > FLOAT_TOLERANCE:
        raise ValueError(f"{path}: expected {expected}, found {actual_float}")


def extract_native_spec_metrics(response_or_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return native vLLM `metrics.speculative_decoding` when present."""
    if not isinstance(response_or_metrics, dict):
        return None
    metrics = response_or_metrics.get("metrics", response_or_metrics)
    if not isinstance(metrics, dict):
        return None
    spec = metrics.get("speculative_decoding")
    return spec if isinstance(spec, dict) else None


def recompute_alpha(spec: dict[str, Any]) -> float | None:
    draft_tokens = spec.get("num_draft_tokens")
    accepted_tokens = spec.get("num_accepted_draft_tokens")
    if not isinstance(draft_tokens, int) or not isinstance(accepted_tokens, int):
        raise ValueError("speculative_decoding counters must be integers")
    if draft_tokens < 0 or accepted_tokens < 0:
        raise ValueError("speculative_decoding counters must be nonnegative")
    if draft_tokens == 0:
        return None
    return accepted_tokens / draft_tokens


def recompute_tau(spec: dict[str, Any]) -> float | None:
    spec_steps = spec.get("num_spec_steps")
    accepted_tokens = spec.get("num_accepted_draft_tokens")
    if not isinstance(spec_steps, int) or not isinstance(accepted_tokens, int):
        raise ValueError("speculative_decoding counters must be integers")
    if spec_steps < 0 or accepted_tokens < 0:
        raise ValueError("speculative_decoding counters must be nonnegative")
    if spec_steps == 0:
        return None
    return 1 + accepted_tokens / spec_steps


def validate_native_spec_metrics(spec: dict[str, Any]) -> None:
    if spec.get("num_spec_tokens") != STATIC_NUM_SPEC_TOKENS:
        raise ValueError("speculative_decoding.num_spec_tokens must be 3")
    histogram = spec.get("acceptance_histogram")
    if not isinstance(histogram, list) or len(histogram) != STATIC_NUM_SPEC_TOKENS + 1:
        raise ValueError("acceptance_histogram length must be 4 for k=3")
    if not all(isinstance(value, int) and value >= 0 for value in histogram):
        raise ValueError("acceptance_histogram must contain nonnegative integers")
    if sum(histogram) != spec.get("num_spec_steps"):
        raise ValueError("sum(acceptance_histogram) must equal num_spec_steps")

    accepted_from_histogram = sum(index * count for index, count in enumerate(histogram))
    if accepted_from_histogram != spec.get("num_accepted_draft_tokens"):
        raise ValueError(
            "sum(i * acceptance_histogram[i]) must equal num_accepted_draft_tokens"
        )

    alpha = recompute_alpha(spec)
    tau = recompute_tau(spec)
    if alpha is not None:
        close_enough(spec.get("draft_acceptance_rate"), alpha, "draft_acceptance_rate")
    if tau is not None:
        close_enough(spec.get("mean_acceptance_length"), tau, "mean_acceptance_length")


def normalize_record(
    *,
    client_request_id: str,
    server_request_id: str | None,
    lane: str,
    model: str,
    server_metrics: dict[str, Any] | None,
    client_metrics: dict[str, Any],
    gpu_metrics: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    spec = extract_native_spec_metrics(server_metrics)
    if lane == "vanilla":
        spec = None
        alpha = None
        tau = None
    elif lane == "static":
        if spec is None:
            raise ValueError("static lane requires metrics.speculative_decoding")
        validate_native_spec_metrics(spec)
        alpha = finite_number(spec["draft_acceptance_rate"], "alpha")
        tau = finite_number(spec["mean_acceptance_length"], "tau")
    else:
        raise ValueError(f"unsupported lane: {lane}")

    return {
        "schema_version": SCHEMA_VERSION,
        "client_request_id": client_request_id,
        "server_request_id": server_request_id,
        "lane": lane,
        "model": model,
        "telemetry_transport": TELEMETRY_TRANSPORT,
        "speculative_decoding": spec,
        "alpha": alpha,
        "tau": tau,
        "tree_budgets": None,
        "client_metrics": client_metrics,
        "gpu_metrics": gpu_metrics,
        "runtime": runtime,
    }


def schema_errors(record: dict[str, Any], validator: Draft202012Validator) -> list[str]:
    errors = []
    for error in sorted(validator.iter_errors(record), key=lambda err: list(err.path)):
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path
        )
        errors.append(f"{path}: {error.message}")
    return errors


def semantic_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lane = record.get("lane")
    spec = record.get("speculative_decoding")

    for section_name in ("client_metrics", "gpu_metrics"):
        section = record.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                errors.append(f"{section_name}.{key}: NaN/Infinity is not allowed")

    if lane == "vanilla":
        if spec is not None:
            errors.append("vanilla speculative_decoding must be null")
        if record.get("alpha") is not None:
            errors.append("vanilla alpha must be null")
        if record.get("tau") is not None:
            errors.append("vanilla tau must be null")
        return errors

    if lane == "static":
        if not isinstance(spec, dict):
            errors.append("static speculative_decoding must be present")
            return errors
        try:
            validate_native_spec_metrics(spec)
            alpha = recompute_alpha(spec)
            tau = recompute_tau(spec)
            if alpha is not None:
                close_enough(record.get("alpha"), alpha, "alpha")
            if tau is not None:
                close_enough(record.get("tau"), tau, "tau")
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def validate_normalized_record(record: dict[str, Any], validator: Draft202012Validator) -> list[str]:
    return schema_errors(record, validator) + semantic_errors(record)


def summarize(records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    static_records = [record for record in records if record.get("lane") == "static"]
    vanilla_records = [record for record in records if record.get("lane") == "vanilla"]
    static_alpha_values = [record["alpha"] for record in static_records if isinstance(record.get("alpha"), (int, float))]
    static_tau_values = [record["tau"] for record in static_records if isinstance(record.get("tau"), (int, float))]
    vram_values = [
        record.get("gpu_metrics", {}).get("vram_peak_mb")
        for record in records
        if isinstance(record.get("gpu_metrics", {}).get("vram_peak_mb"), (int, float))
    ]
    return {
        "task": "M1-01",
        "step": "telemetry-contract-check",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "record_count": len(records),
        "failure_count": len(failures),
        "vanilla_count": len(vanilla_records),
        "static_count": len(static_records),
        "static_alpha_min": min(static_alpha_values) if static_alpha_values else None,
        "static_alpha_max": max(static_alpha_values) if static_alpha_values else None,
        "static_tau_min": min(static_tau_values) if static_tau_values else None,
        "static_tau_max": max(static_tau_values) if static_tau_values else None,
        "vram_peak_mb_max": max(vram_values) if vram_values else None,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_json(args.schema)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in args.records:
        for index, record in enumerate(load_records(path), start=1):
            records.append(record)
            errors = validate_normalized_record(record, validator)
            if errors:
                failures.append({"path": path.as_posix(), "index": index, "errors": errors})

    summary = summarize(records, failures)
    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
