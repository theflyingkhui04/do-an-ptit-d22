"""[M3-00] Client-side check and startup timing for the deployed Modal lanes.

Talks to the *deployed* app (``modal deploy -m experiments.modal_runner``);
the proxy token is read from MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET
(environment or the gitignored ``.env``) and is never written to evidence.

    python -m experiments.modal_probe check --run-id <id>
    python -m experiments.modal_probe startup --kind cold --repeats 3 --idle-wait 120 --run-id <id>
    python -m experiments.modal_probe startup --kind warm --repeats 3 --run-id <id>

Evidence: experiments/runs/m3_00/<run-id>/{check.json,startup.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments import modal_lanes as lanes


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_DECLARED = "NVIDIA L4"
CHAT_PROMPT = "Xin chào"

Request = Callable[..., tuple[int, bytes]]


def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from exc


def load_token(env_file: Path = REPO_ROOT / ".env") -> str:
    """Return ``<id>.<secret>`` for ``Authorization: Bearer``."""
    values = {}
    if Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    token_id = os.environ.get("MODAL_PROXY_TOKEN_ID") or values.get("MODAL_PROXY_TOKEN_ID")
    secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET") or values.get("MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not secret:
        raise RuntimeError(
            "Set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET (environment or .env)"
        )
    return f"{token_id}.{secret}"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def wait_ready(
    base_url: str,
    headers: dict[str, str],
    *,
    timeout_s: float,
    poll_s: float,
    request: Request = http_request,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll ``/health`` through cold-start 503s and connection errors."""
    start = clock()
    attempts = 0
    first_status = None
    status = None
    while True:
        attempts += 1
        try:
            status, _ = request("GET", f"{base_url}/health", headers=headers)
        except (ConnectionError, OSError):
            status = None
        if first_status is None and status is not None:
            first_status = status
        if status == 200:
            return {
                "status": "ok",
                "seconds_to_ready": clock() - start,
                "attempts": attempts,
                "first_status": first_status,
            }
        if clock() - start >= timeout_s:
            return {"status": "timeout", "attempts": attempts, "first_status": first_status, "last_status": status}
        sleep(poll_s)


def _chat(lane: str, base_url: str, token: str, request: Request, max_tokens: int) -> tuple[int, dict]:
    body = json.dumps(
        {
            "model": lanes.LANES[lane].served_model_name,
            "messages": [{"role": "user", "content": CHAT_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    headers = {**_auth(token), "Content-Type": "application/json"}
    status, raw = request("POST", f"{base_url}/v1/chat/completions", headers=headers, body=body, timeout=300.0)
    try:
        payload = json.loads(raw) if raw else {}
    except ValueError:
        payload = {}
    return status, payload


def check_lane(
    lane: str,
    base_url: str,
    token: str,
    *,
    request: Request = http_request,
    timeout_s: float = 20 * 60,
    poll_s: float = 10,
) -> dict[str, Any]:
    ready = wait_ready(base_url, _auth(token), timeout_s=timeout_s, poll_s=poll_s, request=request)
    health = 200 if ready["status"] == "ok" else ready.get("last_status")

    unauth_status, _ = request("GET", f"{base_url}/v1/models", headers={})
    status, raw = request("GET", f"{base_url}/v1/models", headers=_auth(token))
    models = [m.get("id") for m in json.loads(raw).get("data", [])] if status == 200 else []

    chat_status, payload = _chat(lane, base_url, token, request, max_tokens=16)
    choices = payload.get("choices") or [{}]
    content = (choices[0].get("message") or {}).get("content") or ""

    ok = (
        health == 200
        and unauth_status in (401, 403)
        and models == [lanes.LANES[lane].served_model_name]
        and chat_status == 200
        and bool(content.strip())
    )
    return {
        "lane": lane,
        "ok": ok,
        "health": health,
        "unauthenticated_status": unauth_status,
        "models": models,
        "chat_status": chat_status,
        "chat_content_nonempty": bool(content.strip()),
    }


def measure_startup(
    lane: str,
    base_url: str,
    token: str,
    *,
    kind: str,
    profile: str,
    commit: str,
    request: Request = http_request,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timeout_s: float = 20 * 60,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    """One startup sample: time to ``/health`` 200 and to a 1-token reply."""
    dtype = lanes.DTYPE_PROFILES[profile]
    start = clock()
    ready = wait_ready(
        base_url, _auth(token), timeout_s=timeout_s, poll_s=poll_s, request=request, clock=clock, sleep=sleep
    )
    record: dict[str, Any] = {
        "lane": lane,
        "gpu": GPU_DECLARED,
        "vllm_version": lanes.VLLM_VERSION,
        "dtype": dtype["dtype"],
        "quantization": dtype["quantization"],
        "profile": profile,
        "kind": kind,
        "observed_cold": ready["first_status"] not in (200, None),
        "health_attempts": ready["attempts"],
        "seconds_to_ready": None,
        "first_response_seconds": None,
        "chat_status": None,
        "status": ready["status"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
    }
    if ready["status"] != "ok":
        return record
    record["seconds_to_ready"] = round(ready["seconds_to_ready"], 3)
    chat_status, _ = _chat(lane, base_url, token, request, max_tokens=1)
    record["first_response_seconds"] = round(clock() - start, 3)
    record["chat_status"] = chat_status
    if chat_status != 200:
        record["status"] = "chat_error"
    return record


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "ok":
            groups.setdefault((record["lane"], record["kind"]), []).append(record)
    return [
        {
            "lane": lane,
            "kind": kind,
            "n_ok": len(items),
            "median_seconds_to_ready": statistics.median(r["seconds_to_ready"] for r in items),
            "median_first_response_seconds": statistics.median(r["first_response_seconds"] for r in items),
        }
        for (lane, kind), items in sorted(groups.items())
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def resolve_url(lane: str) -> str:
    import modal

    url = modal.Server.from_name(lanes.APP_NAME, lane).get_url()
    if not url:
        raise RuntimeError(f"Lane {lane!r} of app {lanes.APP_NAME!r} has no URL; is it deployed?")
    return url.rstrip("/")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _evidence_path(run_id: str, name: str) -> Path:
    path = REPO_ROOT / "experiments" / "runs" / "m3_00" / run_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check / time the deployed Modal lane servers")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "startup"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--lanes", default="vanilla,static,adaptive")
        cmd.add_argument("--run-id", required=True)
        cmd.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    startup = sub.choices["startup"]
    startup.add_argument("--kind", choices=("cold", "warm"), required=True)
    startup.add_argument("--repeats", type=int, default=3)
    startup.add_argument("--idle-wait", type=float, default=0.0,
                         help="Seconds to wait before each sample so the lane scales to zero (cold).")
    startup.add_argument("--profile", choices=sorted(lanes.DTYPE_PROFILES), default=lanes.DEFAULT_PROFILE)
    args = parser.parse_args(argv)

    token = load_token(args.env_file)
    selected = [lane for lane in args.lanes.split(",") if lane]
    urls = {lane: resolve_url(lane) for lane in selected}

    if args.command == "check":
        results = [check_lane(lane, urls[lane], token) for lane in selected]
        out = _evidence_path(args.run_id, "check.json")
        out.write_text(json.dumps({"commit": _git_commit(), "lanes": results}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 0 if all(r["ok"] for r in results) else 1

    out = _evidence_path(args.run_id, "startup.json")
    existing = json.loads(out.read_text(encoding="utf-8"))["records"] if out.exists() else []
    commit = _git_commit()
    for lane in selected:
        for _ in range(args.repeats):
            if args.idle_wait:
                time.sleep(args.idle_wait)
            record = measure_startup(lane, urls[lane], token, kind=args.kind, profile=args.profile, commit=commit)
            existing.append(record)
            print(json.dumps(record), flush=True)
            out.write_text(
                json.dumps({"records": existing, "summary": summarize(existing)}, indent=2) + "\n", encoding="utf-8"
            )
    return 0 if all(r["status"] == "ok" for r in existing) else 1


if __name__ == "__main__":
    sys.exit(main())
