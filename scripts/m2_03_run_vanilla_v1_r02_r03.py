#!/usr/bin/env python3
"""M2-03: fresh-server Vanilla V1 cold-cache controls R02/R03."""

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "experiments/runs/m2_03/diagnostics"
PYTHON = ROOT / ".venv/bin/python"
VLLM = ROOT / ".venv/bin/vllm"

RUNNER = (
    ROOT / ".venv/lib/python3.13/site-packages"
    / "vllm/v1/worker/gpu_model_runner.py"
)

EXPECTED_SHA = (
    "87c29d08c0bbf66993d8b984811e7325"
    "e35e6436242a773feb55ec88f16b2c51"
)

MODEL = "qwen2.5-7b-vanilla"
BASE = "http://127.0.0.1:8001"

V2_RESPONSE = (
    DIAG / "a3_vanilla_20261013T125113Z_7nwv"
    / "vanilla_response.json"
)

ADAPTIVE_RESPONSE = (
    DIAG / "a3_v1_v8_adaptive_20261014T005559Z_UbV7"
    / "adaptive_attempt2.json"
)

R01 = (
    DIAG /
    "vanilla_v1_control_20261014T013505Z_8F1s"
)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_new(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")

def mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None

def port_busy(port=8001):
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", port)) == 0

def check_preflight():
    if not PYTHON.is_file() or not VLLM.is_file():
        raise RuntimeError("Pinned vLLM environment missing")

    if sha_file(RUNNER) != EXPECTED_SHA:
        raise RuntimeError("V1 runner SHA mismatch; STOP")

    if port_busy():
        raise RuntimeError(
            "Port 8001 occupied. Stop R01 server first."
        )

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=ROOT, text=True
    ).strip()

    if branch != "feat/m2-03-lossless-verification":
        raise RuntimeError(f"Wrong branch: {branch}")

    for path in (
        V2_RESPONSE,
        ADAPTIVE_RESPONSE,
        R01 / "r01_response.json",
        R01 / "r01_comparison.json",
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing evidence: {path}")

    r01 = json.loads(
        (R01 / "r01_comparison.json").read_text()
    )
    if not (
        r01["exact_vs_vanilla_v2"] is True
        and r01["token_200"] == 128371
    ):
        raise RuntimeError("R01 evidence inconsistent")

    rows = [
        json.loads(line)
        for line in (
            ROOT / "experiments/configs/w3_v2/chat_requests.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]

    matches = [
        row for row in rows
        if row["sample_id"] == "w3_business_v2_004"
    ]

    if len(matches) != 1:
        raise RuntimeError("Frozen sample not unique")

    payload = dict(matches[0]["request"])

    if (
        payload.get("temperature"),
        payload.get("seed"),
        payload.get("max_tokens"),
    ) != (0.0, 42, 768):
        raise RuntimeError("Frozen sampling config changed")

    payload["model"] = MODEL
    payload["return_token_ids"] = True

    v2 = json.loads(
        V2_RESPONSE.read_text()
    )["choices"][0]["token_ids"]

    adaptive = json.loads(
        ADAPTIVE_RESPONSE.read_text()
    )["choices"][0]["token_ids"]

    if len(v2) != 284 or v2[200] != 128371:
        raise RuntimeError("Unexpected Vanilla V2 reference")

    if adaptive[200] != 128585:
        raise RuntimeError("Unexpected Adaptive reference")

    print("PREFLIGHT: PASS", flush=True)
    print("RUNNER: V1 ORIGINAL", flush=True)
    print("BRANCH:", branch, flush=True)

    return payload, v2, adaptive

def stop_owned_server(proc):
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print("WARNING: Server did not stop within 90s.")
        print("Manual inspection required. No automatic SIGKILL.")
        raise RuntimeError("SERVER_SHUTDOWN_TIMEOUT")

def wait_ready(proc, timeout=420):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited during startup: {proc.returncode}"
            )

        try:
            response = requests.get(
                BASE + "/health", timeout=5
            )
            if response.status_code == 200:
                models = requests.get(
                    BASE + "/v1/models", timeout=5
                )
                models.raise_for_status()

                ids = [
                    item["id"]
                    for item in models.json()["data"]
                ]
                if MODEL not in ids:
                    raise RuntimeError(
                        f"Model identity mismatch: {ids}"
                    )
                return
        except requests.RequestException:
            pass

        time.sleep(5)

    raise RuntimeError("Server readiness timeout")

def run_repeat(label, payload, v2, adaptive):
    if port_busy():
        raise RuntimeError(
            f"{label}: port 8001 already occupied"
        )

    d = Path(tempfile.mkdtemp(
        prefix=f"vanilla_v1_control_{label}_",
        dir=DIAG
    ))

    print(f"\n=== {label.upper()} ===", flush=True)
    print("EVIDENCE_DIR:", d, flush=True)

    save_new(d / "manifest.json", {
        "task": "M2-03",
        "result_kind": "CONTROL_DIAGNOSTIC",
        "repeat": label,
        "sample_id": "w3_business_v2_004",
        "cache_condition": "fresh_server_first_request",
        "prefix_cache_config": "ON",
        "runner": "V1",
        "speculative_decoding": False,
        "timestamp_utc": now(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True
        ).strip(),
        "runner_sha256": sha_file(RUNNER),
        "vllm_version": "0.30.0",
    })

    save_new(d / "request.json", payload)

    gpu = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=True
    )
    (d / "gpu_before.txt").write_text(gpu.stdout)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    })

    cmd = [
        str(VLLM), "serve",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--served-model-name", MODEL,
        "--generation-config", "vllm",
        "--quantization", "awq",
        "--dtype", "half",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
    ]

    save_new(d / "launch_config.json", {
        "command": cmd,
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        }
    })

    with (d / "server.log").open("xb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            print("WAITING FOR SERVER...", flush=True)
            wait_ready(proc)
            print("HEALTH: PASS", flush=True)

            text = (d / "server.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if not (
                "Initializing a V1 LLM engine (v0.30.0)"
                in text
                and "speculative_config=None" in text
                and "enable_prefix_caching=True" in text
            ):
                raise RuntimeError(
                    "Runtime config not verified"
                )

            if '"POST /v1/chat/completions' in text:
                raise RuntimeError(
                    "Inference already happened before request"
                )

            print("SENDING FROZEN REQUEST...", flush=True)
            response = requests.post(
                BASE + "/v1/chat/completions",
                json=payload,
                timeout=600
            )

            print("HTTP:", response.status_code, flush=True)

            if response.status_code != 200:
                (d / "http_error.txt").write_text(
                    response.text, encoding="utf-8"
                )
                response.raise_for_status()

            body = response.json()
            save_new(d / "response.json", body)

            ids = body["choices"][0]["token_ids"]

            if not isinstance(ids, list):
                raise RuntimeError("No output token IDs")

            report = {
                "result_kind": "DIAGNOSTIC_NOT_CANONICAL",
                "repeat": label,
                "sample_id": "w3_business_v2_004",
                "timestamp_utc": now(),
                "request_id": body.get("id"),
                "output_tokens": len(ids),
                "token_200": (
                    ids[200] if len(ids) > 200 else None
                ),
                "exact_vs_vanilla_v2": ids == v2,
                "first_mismatch_vs_vanilla_v2":
                    mismatch(ids, v2),
                "exact_vs_adaptive_v1_attempt2":
                    ids == adaptive,
                "first_mismatch_vs_adaptive_v1_attempt2":
                    mismatch(ids, adaptive),
            }

            save_new(d / "comparison.json", report)
            print(json.dumps(report, indent=2), flush=True)

        finally:
            print("STOPPING OWNED SERVER...", flush=True)
            stop_owned_server(proc)

    if port_busy():
        raise RuntimeError(
            f"{label}: port 8001 still occupied after stop"
        )

    text = (d / "server.log").read_text(
        encoding="utf-8", errors="replace"
    )
    rates = re.findall(
        r"Prefix cache hit rate:\s*([0-9.]+)%",
        text
    )

    save_new(d / "cache_observation.json", {
        "repeat": label,
        "observed_prefix_cache_hit_rates_percent":
            [float(x) for x in rates],
        "zero_hit_observed": (
            bool(rates)
            and all(float(x) == 0 for x in rates)
        ),
        "note": (
            "Server-level cache metric, not proof "
            "of KV-cache tensor equivalence."
        )
    })

    print("CACHE_HIT_RATES:", rates, flush=True)
    print(f"{label.upper()}_SAVED: PASS", flush=True)
    return d

def main():
    payload, v2, adaptive = check_preflight()
    paths = []

    for label in ("r02", "r03"):
        paths.append(
            run_repeat(label, payload, v2, adaptive)
        )

    print("\n=== CONTROL SUMMARY ===")
    for path in paths:
        report = json.loads(
            (path / "comparison.json").read_text()
        )
        cache = json.loads(
            (path / "cache_observation.json").read_text()
        )
        print(
            path.name,
            "token200=", report["token_200"],
            "match_v2=", report["exact_vs_vanilla_v2"],
            "cache_zero=", cache["zero_hit_observed"]
        )

    print("R02_R03_COMPLETED: PASS")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("INTERRUPTED: evidence retained", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(
            "CONTROL_FAILED:",
            type(exc).__name__,
            str(exc),
            file=sys.stderr
        )
        sys.exit(2)
