"""Diagnostic-only check for W3 v2 output-cap truncation.

This is NOT benchmark evidence. It sends each frozen W3 v2 request once with
``stream=False`` so ``finish_reason`` and usage can be inspected directly.
Restart the serving lane before any canonical benchmark because this diagnostic
changes server/cache history.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp


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
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def build_payload(row: dict[str, Any], runtime_model: str) -> dict[str, Any]:
    request = row.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"{row.get('sample_id')}: missing request object")

    allowed = {"model", "messages", "temperature", "max_tokens", "seed", "response_format"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"{row.get('sample_id')}: unknown request fields {sorted(unknown)}")

    payload: dict[str, Any] = {
        "model": runtime_model,
        "messages": request["messages"],
        "temperature": request["temperature"],
        "seed": request["seed"],
        "max_completion_tokens": request["max_tokens"],
        "n": 1,
        "stream": False,
    }
    if "response_format" in request:
        payload["response_format"] = request["response_format"]
    return payload


async def diagnose(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.workload_file)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    output_rows: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, row in enumerate(rows):
            payload = build_payload(row, args.model)
            cap = int(row["request"]["max_tokens"])
            record: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "request_index": index,
                "dataset_version": row.get("dataset_version"),
                "prompt_version": row.get("prompt_version"),
                "scenario_category": row.get("scenario_category"),
                "max_tokens": cap,
                "http_status": None,
                "output_tokens": None,
                "finish_reason": None,
                "truncation_suspected": None,
                "generated_text": None,
                "error": None,
            }

            try:
                async with session.post(args.endpoint, json=payload) as response:
                    record["http_status"] = response.status
                    body = await response.json(content_type=None)
                    if response.status != 200:
                        record["error"] = json.dumps(body, ensure_ascii=False)[:2000]
                    else:
                        choices = body.get("choices") or []
                        choice = choices[0] if choices else {}
                        message = choice.get("message") or {}
                        usage = body.get("usage") or {}
                        output_tokens = usage.get("completion_tokens")
                        finish_reason = choice.get("finish_reason")
                        record["output_tokens"] = output_tokens
                        record["finish_reason"] = finish_reason
                        record["generated_text"] = message.get("content")
                        record["truncation_suspected"] = (
                            finish_reason == "length"
                            or (isinstance(output_tokens, int) and output_tokens >= cap)
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"

            output_rows.append(record)
            print(
                f"{record['sample_id']}: status={record['http_status']} "
                f"tokens={record['output_tokens']} finish={record['finish_reason']} "
                f"truncated={record['truncation_suspected']}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    errors = [row for row in output_rows if row["error"] is not None]
    truncated = [row for row in output_rows if row["truncation_suspected"] is True]
    completed = [row for row in output_rows if row["output_tokens"] is not None]
    max_observed = max((row["output_tokens"] for row in completed), default=None)

    summary = {
        "mode": "diagnostic-only-not-benchmark-evidence",
        "workload_file": str(args.workload_file),
        "sample_count": len(output_rows),
        "completed": len(completed),
        "errors": len(errors),
        "truncation_suspected_count": len(truncated),
        "truncated_sample_ids": [row["sample_id"] for row in truncated],
        "max_observed_output_tokens": max_observed,
        "raw_jsonl": str(args.out),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        return 3
    if truncated:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload-file",
        type=Path,
        default=Path("experiments/configs/w3_v2/chat_requests.jsonl"),
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/w3_v2_truncation_diagnostic.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(diagnose(parse_args())))


if __name__ == "__main__":
    main()
