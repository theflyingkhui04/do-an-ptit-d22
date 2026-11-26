"""Run the MC-OCR pipeline end to end: Node 1 -> SQLite -> Node 2 -> Node 3.

Node 1 runs through the existing CLI (scripts/run_node1_extractor.py) as a
subprocess; Node 2 (Text-to-SQL, Q1-Q6) and Node 3 (VAT report) run in-process
against the SQLite database produced for this run. The API key is read from
PIPELINE_LLM_API_KEY (or the Modal proxy token in .env) and is passed to the
child process only through its environment.

Example (Modal vanilla lane):
    python scripts/run_pipeline_e2e.py --lane vanilla --run-id 20261103-m306-e2e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.client.modal_client import LANE_MODELS, lane_base_url, load_token  # noqa: E402
from src.pipeline.node2_questions import QUESTIONS  # noqa: E402
from src.pipeline.openai_compatible import OpenAICompatibleClient  # noqa: E402
from src.pipeline.report_writer import (  # noqa: E402
    NODE3_USER_TEMPLATE_PATH,
    W3_SYSTEM_PROMPT_PATH,
    Node3Result,
    Node3Writer,
    render_report_md,
)
from src.pipeline.text_to_sql import (  # noqa: E402
    W2_SYSTEM_PROMPT_PATH,
    Node2Runner,
    load_system_prompt,
    open_read_only,
)

NODE1_SCRIPT = REPO_ROOT / "scripts/run_node1_extractor.py"
# [M5-01] --node1-package-dir experiments/configs/w1_v2 runs Node 1 on full-receipt OCR text.
NODE1_PACKAGE_DIR = "experiments/configs/w1"
NODE1_REQUESTS_FILE = "node1_chat_requests.jsonl"
NODE1_MANIFEST_FILE = "package_manifest.json"
API_KEY_ENV = "PIPELINE_LLM_API_KEY"
DB_NAME = "invoices.sqlite"
# A run_id becomes a directory name under --output-root, so it must not
# contain path separators or "..".
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lane", default="vanilla", choices=sorted(LANE_MODELS))
    parser.add_argument("--base-url", help="Override the lane URL (e.g. local Ollama http://localhost:11434/v1).")
    parser.add_argument("--model", help="Override the served model name.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hardware", help="GPU/host label recorded in e2e_summary.json (e.g. modal-A10).")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments/runs/m3_06")
    parser.add_argument("--limit", type=int, default=30, help="Number of W1 invoices for Node 1.")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--ready-timeout", type=float, default=1200.0)
    parser.add_argument("--skip-node1", action="store_true", help="Reuse --db instead of running Node 1.")
    parser.add_argument("--db", type=Path, help="Existing SQLite database (required with --skip-node1).")
    parser.add_argument("--node1-package-dir", default=NODE1_PACKAGE_DIR,
                        help="Repo-relative W1 package for Node 1 (default: W1 v1; W1_v2: experiments/configs/w1_v2).")
    args = parser.parse_args(argv)
    package_dir = REPO_ROOT / args.node1_package_dir
    if not all((package_dir / name).is_file() for name in (NODE1_REQUESTS_FILE, NODE1_MANIFEST_FILE)):
        parser.error(f"--node1-package-dir {args.node1_package_dir} has no {NODE1_REQUESTS_FILE}/{NODE1_MANIFEST_FILE}")
    if args.skip_node1 and args.db is None:
        parser.error("--skip-node1 requires --db")
    if not RUN_ID_RE.fullmatch(args.run_id) or ".." in args.run_id:
        parser.error("--run-id must be 1-100 characters of letters, digits, '.', '_' or '-' (no '..')")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def resolve_endpoint(args: argparse.Namespace) -> tuple[str, str]:
    return (args.base_url or lane_base_url(args.lane), args.model or LANE_MODELS[args.lane])


def resolve_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    if not args.base_url:
        return load_token()
    return ""


def run_node1(args: argparse.Namespace, run_dir: Path, base_url: str, model: str, api_key: str) -> dict[str, Any]:
    summary_path = run_dir / "node1_summary.json"
    cmd = [
        sys.executable,
        str(NODE1_SCRIPT),
        "--requests", f"{args.node1_package_dir.rstrip('/')}/{NODE1_REQUESTS_FILE}",
        "--package-manifest", f"{args.node1_package_dir.rstrip('/')}/{NODE1_MANIFEST_FILE}",
        "--base-url", base_url,
        "--model", model,
        "--run-id", args.run_id,
        "--lane", args.lane,
        "--output-dir", str(run_dir / "node1"),
        "--report", str(summary_path),
        "--db", str(run_dir / DB_NAME),
        "--replace-run",
        "--limit", str(args.limit),
        "--timeout", str(args.timeout),
        "--ready-timeout", str(args.ready_timeout),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, API_KEY_ENV: api_key}, check=False)
    if not summary_path.exists():
        return {"returncode": proc.returncode, "error": "node1_summary_missing"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["returncode"] = proc.returncode
    return summary


STALE_NODE1_FILES = (
    "node1_summary.json",
    DB_NAME,
    f"{DB_NAME}-wal",
    f"{DB_NAME}-shm",
    f"{DB_NAME}-journal",
)


def remove_stale_node1_artifacts(run_dir: Path) -> None:
    """Delete Node 1 outputs of an earlier run with the same run_id/lane."""
    for name in STALE_NODE1_FILES:
        (run_dir / name).unlink(missing_ok=True)


def count_invoices(db_path: Path) -> int:
    if not Path(db_path).exists():
        return 0
    conn = open_read_only(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM v_invoice_facts").fetchone()[0])
    finally:
        conn.close()


def latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "sum": None}
    ordered = sorted(values)

    def nearest_rank(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {"p50": nearest_rank(0.50), "p95": nearest_rank(0.95), "sum": round(sum(ordered), 3)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sum_tokens(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[..., Any] = OpenAICompatibleClient,
    node1_runner: Callable[..., dict[str, Any]] = run_node1,
) -> int:
    args = parse_args(argv)
    base_url, model = resolve_endpoint(args)
    api_key = resolve_api_key(args)
    output_root = Path(args.output_root).resolve()
    run_dir = (output_root / f"{args.run_id}-{args.lane}").resolve()
    if run_dir.parent != output_root:
        raise SystemExit(f"run directory {run_dir} escapes --output-root {output_root}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    wall_start = time.perf_counter()

    if args.skip_node1:
        db_path = Path(args.db)
        node1: dict[str, Any] = {"skipped": True, "db": str(db_path)}
    else:
        db_path = run_dir / DB_NAME
        remove_stale_node1_artifacts(run_dir)
        node1 = node1_runner(args, run_dir, base_url, model, api_key)
    # A missing Node 1 summary means Node 1 crashed; its DB may be partial.
    invoice_count = 0 if node1.get("error") else count_invoices(db_path)

    node2_results = []
    node3 = Node3Result(status="skipped_no_data")
    if invoice_count > 0:
        client = client_factory(
            base_url,
            api_key,
            timeout_seconds=args.timeout,
            ready_timeout_seconds=args.ready_timeout,
        )
        runner = Node2Runner(client, model, load_system_prompt(W2_SYSTEM_PROMPT_PATH), db_path)
        node2_results = runner.run_all(QUESTIONS)
        with (run_dir / "node2_results.jsonl").open("w", encoding="utf-8") as fh:
            for r in node2_results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        writer = Node3Writer(client, model, load_system_prompt(W3_SYSTEM_PROMPT_PATH))
        node3 = writer.write(node2_results, invoice_count)
        if node3.status == "ok":
            meta = {"run_id": args.run_id, "lane": args.lane, "model": model}
            (run_dir / "report.md").write_text(render_report_md(node3, node2_results, meta), encoding="utf-8")
    (run_dir / "node3_result.json").write_text(
        json.dumps(node3.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    status_counts = Counter(r.status for r in node2_results)
    summary = {
        "task": "M3-06",
        "run_id": args.run_id,
        "lane": args.lane,
        "model": model,
        "base_url": base_url,
        "hardware": args.hardware,
        "node1_package_dir": args.node1_package_dir,
        "git_sha": git_sha(),
        "started_at": started_at,
        "finished_at": _now(),
        "e2e_wall_ms": round((time.perf_counter() - wall_start) * 1000.0, 3),
        "invoice_count": invoice_count,
        "node1": {
            key: node1.get(key)
            for key in ("returncode", "error", "skipped", "db", "request_count", "fully_validated_count", "rates", "latency_ms")
            if key in node1
        },
        "node2": {
            "requests": len(node2_results),
            "status_counts": dict(status_counts),
            "latency_ms": latency_stats([r.latency_ms for r in node2_results if r.latency_ms is not None]),
            "prompt_tokens": _sum_tokens([r.prompt_tokens for r in node2_results]),
            "completion_tokens": _sum_tokens([r.completion_tokens for r in node2_results]),
        },
        "node3": {
            "status": node3.status,
            "latency_ms": node3.latency_ms,
            "prompt_tokens": node3.prompt_tokens,
            "completion_tokens": node3.completion_tokens,
            "finish_reason": node3.finish_reason,
        },
        "prompt_hashes": {
            "w2_system_prompt": sha256_file(W2_SYSTEM_PROMPT_PATH),
            "w3_system_prompt": sha256_file(W3_SYSTEM_PROMPT_PATH),
            "node3_user_template": sha256_file(NODE3_USER_TEMPLATE_PATH),
        },
    }
    (run_dir / "e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ok_or_empty = status_counts.get("ok", 0) + status_counts.get("empty", 0)
    success = invoice_count >= 1 and ok_or_empty >= 1 and node3.status == "ok"
    print(
        f"[M3-06] invoices={invoice_count} node2={dict(status_counts)} node3={node3.status} "
        f"wall_ms={summary['e2e_wall_ms']} -> {run_dir}"
    )
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
