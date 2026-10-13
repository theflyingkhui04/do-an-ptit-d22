#!/usr/bin/env python3
"""Send exactly two frozen W3 requests, then print A3 mapping/records evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8003"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> int:
    health = requests.get(BASE + "/health", timeout=15)
    health.raise_for_status()
    models = requests.get(BASE + "/v1/models", timeout=15)
    models.raise_for_status()
    ids = [x["id"] for x in models.json()["data"]]
    if "qwen2.5-7b-adaptive" not in ids:
        raise RuntimeError(f"Unexpected models: {ids}")
    print("MODEL_IDENTITY: PASS")

    d = Path("/tmp/m203_a3_adaptive_dir").read_text(encoding="utf-8").strip()
    out = Path(d)
    if not out.is_dir():
        raise RuntimeError(f"Adaptive diagnostic directory missing: {out}")

    cfg = json.loads(Path("/tmp/m203_a3_probe_config.json").read_text(encoding="utf-8"))
    if Path(cfg["output_dir"]).resolve() != (out / "probe").resolve():
        raise RuntimeError("A3 config output_dir does not match current diagnostic directory")

    rows = load_jsonl(ROOT / "experiments/configs/w3_v2/chat_requests.jsonl")
    matches = [x for x in rows if x["sample_id"] == "w3_business_v2_004"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one w3_business_v2_004 row, found {len(matches)}")
    payload = dict(matches[0]["request"])
    payload["model"] = "qwen2.5-7b-adaptive"
    payload["return_token_ids"] = True
    if (payload.get("temperature"), payload.get("seed"), payload.get("max_tokens")) != (0.0, 42, 768):
        raise RuntimeError("Frozen request settings changed")

    vanilla_dir = Path("/tmp/m203_a3_smoke_dir").read_text(encoding="utf-8").strip()
    vanilla = json.loads((Path(vanilla_dir) / "vanilla_response.json").read_text(encoding="utf-8"))
    ref = vanilla["choices"][0]["token_ids"]

    for attempt in (1, 2):
        path = out / f"adaptive_attempt{attempt}.json"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing response: {path}")
        response = requests.post(BASE + "/v1/chat/completions", json=payload, timeout=600)
        print(f"\n=== ATTEMPT {attempt} ===")
        print("HTTP:", response.status_code)
        response.raise_for_status()
        body = response.json()
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        token_ids = body["choices"][0]["token_ids"]
        print("OUTPUT_TOKENS:", len(token_ids))
        print("EXACT_MATCH:", token_ids == ref)
        print("FIRST_MISMATCH:", first_mismatch(ref, token_ids))
        print("TOKEN_AT_200:", token_ids[200] if len(token_ids) > 200 else None)
        print("SAVED:", path)

    print("\n=== A3 MARKERS ===")
    log_path = out / "server.log"
    if log_path.exists():
        lines = [
            line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "M2-03 A3 CONFIG" in line or "M2-03 A3 MAP" in line or "ProbeError" in line
        ]
        for line in lines[-80:]:
            print(line)
    else:
        print("SERVER_LOG_MISSING")

    print("\n=== PROBE RECORDS ===")
    records = out / "probe/records.jsonl"
    if records.exists():
        print(records.read_text(encoding="utf-8"), end="")
    else:
        print("NO_RECORDS")

    print("\nEVIDENCE_DIR:", out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"A3_TWO_REQUEST_RUN_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
