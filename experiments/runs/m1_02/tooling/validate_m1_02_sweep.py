#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

BENCH_REPO = Path("/kaggle/working/do-an-ptit-d22")
EXPERIMENTS = BENCH_REPO / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from aggregate_results import aggregate_raw_jsonl
from workload_adapter import (
    load_workload_rows,
    resolve_repeat_offset,
    resolve_sample_index,
    validate_manifest_consistency,
    validate_workload_label,
    verify_workload_integrity,
)

FROZEN_SHA = "f307e33f9aea4bbd45499f548480cb7d6f90a4b9"

CONCURRENCIES = (1, 2, 4, 8, 16, 32)
REPEATS = 5
REQUESTS_PER_REPEAT = 32
EXPECTED_RECORDS = len(CONCURRENCIES) * REPEATS * REQUESTS_PER_REPEAT

WORKLOAD_FILES = {
    "W1": BENCH_REPO / "experiments/configs/w1/node1_chat_requests.jsonl",
    "W2": BENCH_REPO / "experiments/configs/w2_v2/chat_requests.jsonl",
    "W3": BENCH_REPO / "experiments/configs/w3_v2/chat_requests.jsonl",
}

LANE_MODELS = {
    "vanilla": "qwen2.5-7b-vanilla",
    "static": "qwen2.5-7b-static-draft",
}

ENV_EXPECTED = {
    "environment_profile": "m1-t4-cu130-v1",
    "gpu": "Tesla T4",
    "gpu_index": 0,
    "driver_version": "580.178.04",
    "torch_version": "2.13.0+cu130",
    "torch_cuda_version": "13.0",
    "vllm_version": "0.30.0",
    "aiohttp_version": "3.13.5",
    "commit": FROZEN_SHA,
}


def fail(message: str) -> None:
    raise ValueError(message)


def finite_number(value, field: str, *, minimum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field}: expected numeric value, got {value!r}")

    result = float(value)

    if not math.isfinite(result):
        fail(f"{field}: NaN/Infinity not allowed")

    if minimum is not None and result < minimum:
        fail(f"{field}: expected >= {minimum}, got {result}")

    return result


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"line {line_no}: invalid JSON: {exc}")

            if not isinstance(row, dict):
                fail(f"line {line_no}: expected JSON object")

            records.append(row)

    return records


def validate_static_spec(record: dict) -> None:
    request_id = record.get("request_id")

    metrics = record.get("server_metrics")
    if not isinstance(metrics, dict):
        fail(f"{request_id}: static server_metrics missing")

    spec = metrics.get("speculative_decoding")
    if not isinstance(spec, dict):
        fail(f"{request_id}: speculative_decoding metrics missing")

    names = (
        "num_spec_steps",
        "num_accepted_draft_tokens",
        "num_draft_tokens",
        "num_spec_tokens",
    )

    values = {}

    for name in names:
        value = spec.get(name)

        if isinstance(value, bool) or not isinstance(value, int):
            fail(f"{request_id}: {name} must be integer, got {value!r}")

        values[name] = value

    steps = values["num_spec_steps"]
    accepted = values["num_accepted_draft_tokens"]
    drafted = values["num_draft_tokens"]
    k = values["num_spec_tokens"]

    if k != 3:
        fail(f"{request_id}: expected num_spec_tokens=3, got {k}")

    if steps <= 0:
        fail(f"{request_id}: num_spec_steps must be >0")

    if drafted <= 0:
        fail(f"{request_id}: num_draft_tokens must be >0")

    if accepted < 0 or accepted > drafted:
        fail(
            f"{request_id}: invalid accepted/drafted "
            f"{accepted}/{drafted}"
        )

    if drafted > k * steps:
        fail(
            f"{request_id}: drafted={drafted} exceeds "
            f"k*steps={k * steps}"
        )

    histogram = spec.get("acceptance_histogram")

    if not isinstance(histogram, list) or len(histogram) != k + 1:
        fail(
            f"{request_id}: histogram must have "
            f"{k + 1} entries, got {histogram!r}"
        )

    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0
        for x in histogram
    ):
        fail(f"{request_id}: invalid histogram {histogram!r}")

    if sum(histogram) != steps:
        fail(
            f"{request_id}: histogram sum={sum(histogram)} "
            f"!= steps={steps}"
        )

    histogram_accepted = sum(
        index * count
        for index, count in enumerate(histogram)
    )

    if histogram_accepted != accepted:
        fail(
            f"{request_id}: histogram accepted="
            f"{histogram_accepted} != counter={accepted}"
        )

    alpha = accepted / drafted
    tau = 1.0 + accepted / steps

    if not (0.0 <= alpha <= 1.0):
        fail(f"{request_id}: alpha out of range: {alpha}")

    if not (1.0 <= tau <= 4.0):
        fail(f"{request_id}: tau out of range: {tau}")

    reported_alpha = spec.get("draft_acceptance_rate")
    if isinstance(reported_alpha, (int, float)):
        if not math.isclose(
            float(reported_alpha),
            alpha,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            fail(
                f"{request_id}: reported alpha mismatch: "
                f"{reported_alpha} vs {alpha}"
            )

    reported_tau = spec.get("mean_acceptance_length")
    if isinstance(reported_tau, (int, float)):
        if not math.isclose(
            float(reported_tau),
            tau,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            fail(
                f"{request_id}: reported tau mismatch: "
                f"{reported_tau} vs {tau}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one complete canonical M1-02 sweep."
    )

    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument(
        "--lane",
        required=True,
        choices=["vanilla", "static"],
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=["W1", "W2", "W3"],
    )
    parser.add_argument("--experiment-id", required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.raw.is_file():
        fail(f"raw file missing: {args.raw}")

    workload_path = WORKLOAD_FILES[args.workload]

    integrity = verify_workload_integrity(workload_path)
    workload_rows = load_workload_rows(workload_path)
    validate_manifest_consistency(integrity, workload_rows)
    validate_workload_label(
        args.workload,
        workload_rows[0]["workload_id"],
    )

    sample_count = len(workload_rows)
    records = load_jsonl(args.raw)

    print("=== CANONICAL SWEEP VALIDATION ===")
    print(f"raw              : {args.raw}")
    print(f"lane             : {args.lane}")
    print(f"workload         : {args.workload}")
    print(f"experiment_id    : {args.experiment_id}")
    print(f"records          : {len(records)}")
    print(f"expected_records : {EXPECTED_RECORDS}")

    if len(records) != EXPECTED_RECORDS:
        fail(
            f"expected exactly {EXPECTED_RECORDS} measured records, "
            f"got {len(records)}"
        )

    first_workload = workload_rows[0]

    expected_common = {
        "phase": "measured",
        "experiment_id": args.experiment_id,
        "lane": args.lane,
        "workload": args.workload,
        "model": LANE_MODELS[args.lane],
        "sample_policy": "cycle-rotate",
        "canonical_sample_count": sample_count,
        "workload_id": first_workload["workload_id"],
        "dataset_version": first_workload["dataset_version"],
        "prompt_version": first_workload["prompt_version"],
        "workload_file_sha256": integrity.actual_sha256,
        "workload_artifact_key": integrity.artifact_key,
        "manifest_package_sha256": integrity.manifest_package_sha256,
    }

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    seen_request_ids: set[str] = set()

    repeat_re = re.compile(r"^r([0-9]{3})$")

    for row_no, record in enumerate(records, start=1):
        prefix = f"row {row_no}"

        for field, expected in ENV_EXPECTED.items():
            if record.get(field) != expected:
                fail(
                    f"{prefix}: {field}: expected "
                    f"{expected!r}, got {record.get(field)!r}"
                )

        for field, expected in expected_common.items():
            if record.get(field) != expected:
                fail(
                    f"{prefix}: {field}: expected "
                    f"{expected!r}, got {record.get(field)!r}"
                )

        if record.get("success") is not True:
            fail(f"{prefix}: success != True")

        if record.get("metrics_complete") is not True:
            fail(f"{prefix}: metrics_complete != True")

        if record.get("http_status") != 200:
            fail(
                f"{prefix}: HTTP status "
                f"{record.get('http_status')!r}"
            )

        concurrency = record.get("concurrency")

        if concurrency not in CONCURRENCIES:
            fail(
                f"{prefix}: invalid concurrency "
                f"{concurrency!r}"
            )

        expected_cell = (
            f"{args.lane}__{args.workload}__c{concurrency}"
        )

        if record.get("cell_id") != expected_cell:
            fail(
                f"{prefix}: cell_id expected "
                f"{expected_cell!r}, got "
                f"{record.get('cell_id')!r}"
            )

        repeat_id = record.get("repeat_id")
        match = (
            repeat_re.fullmatch(repeat_id)
            if isinstance(repeat_id, str)
            else None
        )

        if match is None:
            fail(f"{prefix}: invalid repeat_id {repeat_id!r}")

        repeat_index = int(match.group(1))

        if not (1 <= repeat_index <= REPEATS):
            fail(
                f"{prefix}: repeat index out of range: "
                f"{repeat_index}"
            )

        request_index = record.get("request_index")

        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or not (0 <= request_index < REQUESTS_PER_REPEAT)
        ):
            fail(
                f"{prefix}: invalid request_index "
                f"{request_index!r}"
            )

        expected_offset = resolve_repeat_offset(
            sample_count,
            REQUESTS_PER_REPEAT,
            repeat_index,
            sample_policy="cycle-rotate",
        )

        expected_sample_index = resolve_sample_index(
            sample_count,
            request_index,
            sample_policy="cycle-rotate",
            sample_offset=expected_offset,
        )

        if record.get("sample_offset") != expected_offset:
            fail(
                f"{prefix}: sample_offset expected "
                f"{expected_offset}, got "
                f"{record.get('sample_offset')!r}"
            )

        if record.get("sample_index") != expected_sample_index:
            fail(
                f"{prefix}: sample_index expected "
                f"{expected_sample_index}, got "
                f"{record.get('sample_index')!r}"
            )

        expected_row = workload_rows[expected_sample_index]

        if record.get("sample_id") != expected_row["sample_id"]:
            fail(
                f"{prefix}: sample_id mapping mismatch: "
                f"expected {expected_row['sample_id']!r}, got "
                f"{record.get('sample_id')!r}"
            )

        request_contract = expected_row["request"]

        payload_expectations = {
            "package_model": request_contract["model"],
            "temperature": request_contract["temperature"],
            "seed": request_contract["seed"],
            "max_completion_tokens": request_contract["max_tokens"],
            "response_format": request_contract.get("response_format"),
        }

        for field, expected in payload_expectations.items():
            if record.get(field) != expected:
                fail(
                    f"{prefix}: {field}: expected "
                    f"{expected!r}, got "
                    f"{record.get(field)!r}"
                )

        request_id = record.get("request_id")

        if not isinstance(request_id, str) or not request_id:
            fail(f"{prefix}: invalid request_id")

        if request_id in seen_request_ids:
            fail(f"{prefix}: duplicate request_id {request_id}")

        seen_request_ids.add(request_id)

        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            fail(f"{prefix}: invalid run_id")

        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")

        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens <= 0
        ):
            fail(
                f"{prefix}: invalid input_tokens "
                f"{input_tokens!r}"
            )

        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens <= 1
        ):
            fail(
                f"{prefix}: canonical TPOT requires "
                f"output_tokens > 1, got {output_tokens!r}"
            )

        ttft = finite_number(
            record.get("ttft_ms"),
            f"{prefix}.ttft_ms",
            minimum=0,
        )
        e2e = finite_number(
            record.get("e2e_ms"),
            f"{prefix}.e2e_ms",
            minimum=0,
        )
        tpot = finite_number(
            record.get("tpot_ms"),
            f"{prefix}.tpot_ms",
            minimum=0,
        )

        finite_number(
            record.get("client_queue_ms"),
            f"{prefix}.client_queue_ms",
            minimum=0,
        )
        finite_number(
            record.get("run_duration_s"),
            f"{prefix}.run_duration_s",
            minimum=0,
        )
        finite_number(
            record.get("run_vram_peak_mb"),
            f"{prefix}.run_vram_peak_mb",
            minimum=0,
        )

        if e2e < ttft:
            fail(
                f"{prefix}: e2e_ms={e2e} < ttft_ms={ttft}"
            )

        expected_tpot = round(
            (e2e - ttft) / (output_tokens - 1),
            3,
        )

        if not math.isclose(
            tpot,
            expected_tpot,
            rel_tol=0,
            abs_tol=0.001,
        ):
            fail(
                f"{prefix}: TPOT formula mismatch: "
                f"stored={tpot}, recomputed={expected_tpot}"
            )

        itl = record.get("itl_ms")

        if not isinstance(itl, list):
            fail(f"{prefix}: itl_ms must be list")

        for value in itl:
            finite_number(
                value,
                f"{prefix}.itl_ms",
                minimum=0,
            )

        if args.lane == "static":
            validate_static_spec(record)
        else:
            metrics = record.get("server_metrics")
            if (
                isinstance(metrics, dict)
                and metrics.get("speculative_decoding") is not None
            ):
                fail(
                    f"{prefix}: Vanilla unexpectedly contains "
                    "speculative_decoding metrics"
                )

        groups[(concurrency, repeat_id)].append(record)

    expected_keys = {
        (concurrency, f"r{repeat_index:03d}")
        for concurrency in CONCURRENCIES
        for repeat_index in range(1, REPEATS + 1)
    }

    if set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        extra = sorted(set(groups) - expected_keys)

        fail(
            f"cell/repeat structure mismatch; "
            f"missing={missing}, extra={extra}"
        )

    seen_run_ids: set[str] = set()

    for key in sorted(expected_keys):
        group = groups[key]

        if len(group) != REQUESTS_PER_REPEAT:
            fail(
                f"{key}: expected {REQUESTS_PER_REPEAT} "
                f"requests, got {len(group)}"
            )

        indices = sorted(
            row["request_index"]
            for row in group
        )

        if indices != list(range(REQUESTS_PER_REPEAT)):
            fail(
                f"{key}: request_index coverage invalid: "
                f"{indices}"
            )

        run_ids = {row["run_id"] for row in group}

        if len(run_ids) != 1:
            fail(f"{key}: expected exactly one run_id")

        run_id = next(iter(run_ids))

        if run_id in seen_run_ids:
            fail(
                f"{key}: run_id reused across repeats: "
                f"{run_id}"
            )

        seen_run_ids.add(run_id)

        durations = {
            row["run_duration_s"]
            for row in group
        }

        if len(durations) != 1:
            fail(
                f"{key}: run_duration_s inconsistent "
                "inside repeat"
            )

    # Reuse frozen canonical aggregator as an independent downstream check.
    summaries = aggregate_raw_jsonl(args.raw)

    if len(summaries) != len(CONCURRENCIES):
        fail(
            f"aggregator expected {len(CONCURRENCIES)} cells, "
            f"got {len(summaries)}"
        )

    for summary in summaries:
        if summary["n_runs"] != REPEATS:
            fail(
                f"aggregator cell {summary['concurrency']} "
                f"n_runs={summary['n_runs']} != {REPEATS}"
            )

    digest = hashlib.sha256(args.raw.read_bytes()).hexdigest()

    print()
    print("=== VALIDATION RESULT ===")
    print(f"records          : {len(records)}")
    print(f"cells            : {len(CONCURRENCIES)}")
    print(f"repeats          : {len(seen_run_ids)}")
    print(f"request_ids      : {len(seen_request_ids)}")
    print(f"raw_sha256       : {digest}")
    print("CANONICAL SWEEP VALIDATION: PASS")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"CANONICAL SWEEP VALIDATION: FAIL: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
