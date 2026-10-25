"""[M3-01] Smoke test and client check of the three Modal lanes via the shared client.

    python -m experiments.modal_smoke client-check --run-id <id>   # chat + stream + logit margin per lane
    python -m experiments.modal_smoke smoke --run-id <id>          # 5 frozen prompts x 3 lanes

Prompts come from the frozen workload packages (same payload builder as the
benchmark harness). Latency is measured from this machine, so it includes the
network round trip to us-east: reference only, not benchmark evidence.
Evidence: experiments/runs/m3_01/<run-id>/{client_check.json,smoke.json}.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments import modal_lanes as lanes
from experiments.workload_adapter import build_request_payload, load_workload_rows, verify_workload_integrity
from src.client.modal_client import LogitMarginUnsupported, ModalClientError, ModalLaneClient


REPO_ROOT = Path(__file__).resolve().parents[1]  # evidence root
WORKLOAD_ROOT = Path(__file__).resolve().parents[1]  # frozen workloads (never patched)
SMOKE_SAMPLES = (("W1", 0), ("W1", 1), ("W2", 0), ("W2", 1), ("W3", 0))
CHECK_MESSAGES = [{"role": "user", "content": "Thủ đô của Việt Nam là gì? Trả lời một câu."}]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

make_client: Callable[[str], Any] = ModalLaneClient


def load_smoke_rows() -> list[tuple[str, dict]]:
    """(label, row) for each smoke sample, read from integrity-checked frozen packages."""
    rows = []
    for workload, index in SMOKE_SAMPLES:
        path = WORKLOAD_ROOT / lanes.BENCH_WORKLOADS[workload]
        verify_workload_integrity(path)
        rows.append((workload, load_workload_rows(path)[index]))
    return rows


def _params(row: dict, model: str) -> tuple[list[dict], dict]:
    payload = build_request_payload(row, runtime_model=model)
    messages = payload.pop("messages")
    for key in ("model", "stream", "stream_options"):
        payload.pop(key, None)
    return messages, payload


def _strip_fence(body: str) -> str:
    """Remove a surrounding ```json ... ``` markdown fence, if any."""
    newline = body.find("\n")
    if body.startswith("```") and body.endswith("```") and newline != -1:
        return body[newline + 1 : -3].strip()
    return body


def _json_valid(text: str) -> bool:
    try:
        json.loads(_strip_fence(text.strip()))
        return True
    except ValueError:
        return False


def run_smoke(lane_names: list[str], client_factory: Callable[[str], Any]) -> list[dict]:
    rows = load_smoke_rows()
    records = []
    for lane in lane_names:
        client = client_factory(lane)
        for workload, row in rows:
            messages, params = _params(row, client.model)
            record = {"lane": lane, "workload": workload, "sample_id": row["sample_id"],
                      "ok": False, "ttft_s": None, "total_s": None, "completion_tokens": None,
                      "json_valid": None, "error": None}
            try:
                final = list(client.stream(messages, **params))[-1]
                record.update(ttft_s=final.ttft_s, total_s=final.total_s,
                              completion_tokens=(final.usage or {}).get("completion_tokens"),
                              ok=bool(final.text.strip()) and final.usage is not None)
                if workload == "W1":
                    record["json_valid"] = _json_valid(final.text)
            except ModalClientError as exc:
                record["error"] = str(exc)
            records.append(record)
    return records


def client_check(lane_names: list[str], client_factory: Callable[[str], Any]) -> list[dict]:
    report = []
    for lane in lane_names:
        client = client_factory(lane)
        entry = {"lane": lane, "chat_ok": False, "stream_usage": None, "stream_metrics": None,
                 "margins": None, "margins_unsupported": None, "error": None}
        try:
            entry["chat_ok"] = bool(client.chat(CHECK_MESSAGES, max_tokens=16, temperature=0).text.strip())
            final = list(client.stream(CHECK_MESSAGES, max_tokens=16, temperature=0))[-1]
            entry["stream_usage"], entry["stream_metrics"] = final.usage, final.metrics
            try:
                entry["margins"] = [{"token": m.token, "logprob": m.logprob, "margin": m.margin}
                                    for m in client.margins(CHECK_MESSAGES, max_tokens=8)]
            except LogitMarginUnsupported as exc:
                entry["margins_unsupported"] = str(exc)
        except ModalClientError as exc:
            entry["error"] = str(exc)
        report.append(entry)
    return report


def evidence_path(run_id: str, name: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"Invalid run-id {run_id!r}: use letters, digits, '.', '_' or '-'")
    root = (REPO_ROOT / "experiments" / "runs" / "m3_01").resolve()
    path = (root / run_id / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid run-id {run_id!r}: evidence must stay under {root}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke / client check of the Modal lanes")
    parser.add_argument("command", choices=("smoke", "client-check"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lanes", default="vanilla,static,adaptive")
    args = parser.parse_args(argv)
    try:
        out = evidence_path(args.run_id, "smoke.json" if args.command == "smoke" else "client_check.json")
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    selected = [lane for lane in args.lanes.split(",") if lane]
    meta = {"commit": _git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "latency measured from the caller's machine; includes network round trip"}
    if args.command == "smoke":
        records = run_smoke(selected, make_client)
        ok = sum(r["ok"] for r in records)
        data = {**meta, "summary": {"total": len(records), "ok": ok}, "records": records}
        status = 0 if ok == len(records) else 1
    else:
        report = client_check(selected, make_client)
        data = {**meta, "lanes": report}
        status = 0 if all(r["chat_ok"] and r["stream_usage"] for r in report) else 1
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
    print(f"Wrote {out}")
    return status


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
