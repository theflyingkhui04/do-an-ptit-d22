"""[M5-01] OCR the M0-04 devset images on Modal with the pinned Qwen2.5-VL model.

Every image is OCRed with ``src/pipeline/ocr.py`` (scale, request, cleaning)
against a ``vllm serve`` process inside the container. Output, one folder per run:

    experiments/runs/m5_01/<run-id>/ocr_texts.jsonl   one row per sample (text + provenance)
    experiments/runs/m5_01/<run-id>/ocr_summary.json  GPU, vLLM, counts, timings

Run from the repository root with the conda env ``doan-d22``:

    modal run -m experiments.modal_ocr --run-id 20261108-m501-ocr-a10g

Switches: PTIT_OCR_GPU (default A10G).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = "/root/project"
HF_HOME = "/root/.cache/huggingface"
VLLM_VERSION = "0.30.0"
GPU = os.environ.get("PTIT_OCR_GPU", "A10G")
PORT = 8000
CONCURRENCY = 8
DEFAULT_SAMPLES = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
DEFAULT_OUTPUT_ROOT = Path("experiments/runs/m5_01")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HOME": HF_HOME})
    .add_local_dir(REPO_ROOT / "src", f"{REMOTE_PROJECT_ROOT}/src", ignore=["**/__pycache__/**"])
)
# Separate from ptit-model-cache, whose weights manifest belongs to the serving lanes.
cache = modal.Volume.from_name("ptit-ocr-cache", create_if_missing=True)
app = modal.App("ptit-ocr", image=image)


def _wait_ready(proc: subprocess.Popen, timeout_s: float = 1500.0) -> float:
    start = time.monotonic()
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            return time.monotonic() - start
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {proc.returncode}")
            if time.monotonic() - start > timeout_s:
                raise RuntimeError("vllm serve did not become ready")
            time.sleep(5)


@app.function(gpu=GPU, timeout=60 * 60, volumes={HF_HOME: cache})
def ocr_batch(jobs: list[dict]) -> dict:
    """OCR ``jobs`` ({sample, image_b64}) and return provenance rows in input order."""
    sys.path.insert(0, REMOTE_PROJECT_ROOT)
    import torch
    import vllm

    from src.pipeline import ocr
    from src.pipeline.openai_compatible import OpenAICompatibleClient

    proc = subprocess.Popen(["vllm", "serve", *ocr.ocr_server_args(port=PORT)])
    try:
        startup_s = _wait_ready(proc)
        cache.commit()
        client = OpenAICompatibleClient(f"http://127.0.0.1:{PORT}/v1", api_key="", timeout_seconds=600.0)

        def one(job: dict) -> dict:
            data = base64.b64decode(job["image_b64"])
            start = time.monotonic()
            try:
                result = ocr.run_ocr(client, data)
            except Exception as exc:  # recorded per sample; the builder rejects error rows
                return ocr.error_row(job["sample"], data, f"{type(exc).__name__}: {exc}"[:500])
            return ocr.provenance_row(job["sample"], data, result, (time.monotonic() - start) * 1000)

        wall_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            rows = list(pool.map(one, jobs))
        wall_s = time.monotonic() - wall_start
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {
        "rows": rows,
        "server": {
            "gpu": torch.cuda.get_device_name(0),
            "vllm_version": vllm.__version__,
            "vllm_args": ocr.ocr_server_args(port=PORT),
            "startup_seconds": round(startup_s, 1),
            "wall_seconds": round(wall_s, 1),
            "concurrency": CONCURRENCY,
        },
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()


@app.local_entrypoint()
def main(run_id: str, samples: str = DEFAULT_SAMPLES.as_posix(), limit: int = 0) -> None:
    from src.pipeline import ocr

    rows_in = [json.loads(line) for line in (REPO_ROOT / samples).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows_in = rows_in[:limit]
    jobs = [
        {
            "sample": {key: row.get(key) for key in ("sample_id", "img_id", "image_path")},
            "image_b64": base64.b64encode((REPO_ROOT / row["image_path"]).read_bytes()).decode("ascii"),
        }
        for row in rows_in
    ]
    out_dir = REPO_ROOT / DEFAULT_OUTPUT_ROOT / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; use a new --run-id")

    started = datetime.now(timezone.utc).isoformat()
    result = ocr_batch.remote(jobs)
    rows = result["rows"]

    out_dir.mkdir(parents=True)
    with (out_dir / "ocr_texts.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "task": "M5-01",
        "step": "ocr-devset",
        "run_id": run_id,
        "started_at_utc": started,
        "samples_jsonl": samples,
        **ocr.summarize_ocr(rows),
        "server": result["server"],
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--", "src/pipeline/ocr.py", "experiments/modal_ocr.py")),
    }
    (out_dir / "ocr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "finish_reasons", "errors")}, ensure_ascii=False))
    print(f"[M5-01] wrote {out_dir}")
