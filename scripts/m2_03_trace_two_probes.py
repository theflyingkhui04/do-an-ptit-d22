"""Run EXACTLY two sequential W3-004 requests against diagnostic Lane 3.

Never overwrites an existing probes.jsonl. Must be run only after the traced
server was freshly started and the reference r03 files are preserved.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs/m2_03"
DIAG = RUNS / "diagnostics/k4_steptrace_coldstart_w3_004"
PROBES = DIAG / "probes.jsonl"
TRACE = DIAG / "step_trace.jsonl"
TELEMETRY = DIAG / "telemetry.jsonl"
WORKLOAD_FILE = ROOT / "experiments/configs/w3_v2/chat_requests.jsonl"
REFERENCE_FILE = RUNS / "m2-03-20261013-r03/vanilla.jsonl"
SAMPLE_ID = "w3_business_v2_004"
BASE_URL = "http://127.0.0.1:8003"
MODEL = "qwen2.5-7b-adaptive"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_mismatch(ref: list[int], actual: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(ref, actual)):
        if left != right:
            return idx
    if len(ref) != len(actual):
        return min(len(ref), len(actual))
    return None


def main() -> None:
    if not DIAG.is_dir() or not TRACE.parent.is_dir():
        raise SystemExit("STOP: create diagnostic directory before running probes")
    if PROBES.exists():
        raise SystemExit("STOP: probes.jsonl already exists; never overwrite evidence")
    for path in (TRACE, TELEMETRY):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"STOP: {path} already contains data; fresh start needed")

    workload = [r for r in read_jsonl(WORKLOAD_FILE) if r["sample_id"] == SAMPLE_ID]
    refs = [r for r in read_jsonl(REFERENCE_FILE)
            if r["workload"] == "W3" and r["sample_id"] == SAMPLE_ID]
    if len(workload) != 1 or len(refs) != 1 or not refs[0]["ok"]:
        raise SystemExit("STOP: missing or ambiguous frozen workload/reference")

    payload = copy.deepcopy(workload[0]["request"])
    payload["model"] = MODEL
    payload["return_token_ids"] = True
    if (payload.get("max_tokens"), payload.get("seed"), payload.get("temperature")) != (768, 42, 0.0):
        raise SystemExit("STOP: frozen W3 generation config differs from expected")
    if payload.get("stream", False):
        raise SystemExit("STOP: probes must use non-streaming responses")

    models = requests.get(BASE_URL + "/v1/models", timeout=30)
    models.raise_for_status()
    ids = [entry["id"] for entry in models.json()["data"]]
    if MODEL not in ids:
        raise SystemExit(f"STOP: wrong served model: {ids}")

    ref_ids = refs[0]["token_ids"]
    print("SERVER:", BASE_URL, MODEL)
    print("SAMPLE:", SAMPLE_ID)
    print("INPUT_SHA256:", sha(WORKLOAD_FILE))
    print("VANILLA_SHA256:", sha(REFERENCE_FILE))
    print("GENERATION:", {k: payload.get(k) for k in ("max_tokens", "seed", "temperature")})

    with PROBES.open("x", encoding="utf-8") as output:
        for attempt in (1, 2):
            started = now()
            response = requests.post(
                BASE_URL + "/v1/chat/completions", json=payload, timeout=600
            )
            try:
                body = response.json()
            except ValueError:
                body = {"unparsed_response": response.text}
            choices = body.get("choices") or []
            token_ids = choices[0].get("token_ids") if choices else None
            spec = (body.get("metrics") or {}).get("speculative_decoding")
            record = {
                "attempt": attempt,
                "started_utc": started,
                "finished_utc": now(),
                "sample_id": SAMPLE_ID,
                "http_status": response.status_code,
                "response_id": body.get("id"),
                "token_ids": token_ids,
                "match_vanilla": token_ids == ref_ids,
                "first_mismatch_vanilla": first_mismatch(ref_ids, token_ids) if isinstance(token_ids, list) else None,
                "speculative_metrics": spec,
                "raw_response": body,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print("\nRUN:", attempt)
            print("HTTP:", response.status_code, "ID:", record["response_id"])
            print("TOKEN_COUNT:", len(token_ids) if isinstance(token_ids, list) else None)
            print("MATCH_VANILLA:", record["match_vanilla"])
            print("FIRST_MISMATCH:", record["first_mismatch_vanilla"])
            print("PER_STEP_ACCEPTED:", (spec or {}).get("per_step_accepted"))
            if response.status_code != 200 or not isinstance(token_ids, list):
                raise SystemExit("STOP: incomplete probe; raw evidence preserved")
            if not isinstance(spec, dict) or not isinstance(spec.get("per_step_accepted"), list):
                raise SystemExit("STOP: missing detailed metrics; raw evidence preserved")
    print("\nSAVED:", PROBES)
    print("NEXT: python scripts/analyze_m2_03_trace.py")


if __name__ == "__main__":
    main()
