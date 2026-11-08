"""[M5-01 canonical flow] Library OCR on Modal CPU — no vision-language model.

Engine A: PaddleOCR text detection + VietOCR (vgg_transformer) recognition.
Engine B: PaddleOCR text detection + PaddleOCR recognition (Vietnamese/Latin).
Both use the same detector and the fixed line rule of src/pipeline/library_ocr.py.

    modal run -m experiments.modal_ocr_lib --run-id 20261108-m501-ocrlib-choice

writes experiments/runs/m5_01/<run>/ocr_lib_{A,B}.jsonl (raw OCR text, devset images only) and
experiments/results/m5_01_ocr_lib_choice.json (coverage after mark stripping, timings, pre-set rule, choice).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = "/root/project"
CACHE = "/cache"
PADDLEPADDLE_VERSION = "3.3.1"
PADDLEOCR_VERSION = "3.7.0"
VIETOCR_VERSION = "0.3.13"
DET_MODEL = "PP-OCRv5_server_det"
VIETOCR_WEIGHTS_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
DEFAULT_SAMPLES = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
QWEN_BASELINE = Path("experiments/runs/m5_01/20261108-m501-ocr-a10g/ocr_texts.jsonl")
CHOICE_REPORT = Path("experiments/results/m5_01_ocr_lib_choice.json")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(f"paddlepaddle=={PADDLEPADDLE_VERSION}", f"paddleocr=={PADDLEOCR_VERSION}")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(f"vietocr=={VIETOCR_VERSION}", extra_options="--no-deps")
    .pip_install("einops==0.8.0", "gdown==5.2.0", "pyyaml", "requests")
    .env({"PADDLE_PDX_CACHE_HOME": f"{CACHE}/paddlex", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"})
    .add_local_dir(REPO_ROOT / "src", f"{REMOTE_PROJECT_ROOT}/src", ignore=["**/__pycache__/**"])
)
cache = modal.Volume.from_name("ptit-ocr-lib-cache", create_if_missing=True)
app = modal.App("ptit-ocr-lib", image=image)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vietocr_predictor():
    import requests
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    weights = Path(CACHE) / "vietocr" / "vgg_transformer.pth"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(VIETOCR_WEIGHTS_URL, timeout=600)
        response.raise_for_status()
        weights.write_bytes(response.content)
        cache.commit()
    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = str(weights)
    config["device"] = "cpu"
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = False
    return Predictor(config), {"weights_url": VIETOCR_WEIGHTS_URL, "weights_sha256": _sha256(weights)}


def _crop(rgb, box):
    from PIL import Image

    height, width = rgb.shape[:2]
    x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
    x1, y1 = min(width, round(box.x1) + 1), min(height, round(box.y1) + 1)
    return Image.fromarray(rgb[y0:y1, x0:x1])


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes={CACHE: cache})
def ocr_lib_batch(jobs: list[dict]) -> dict:
    """Run engines A and B on every job ({sample, image_b64}); return rows per engine and versions."""
    sys.path.insert(0, REMOTE_PROJECT_ROOT)
    import numpy as np
    from paddleocr import PaddleOCR, TextDetection
    from PIL import Image

    from src.pipeline.library_ocr import Box, boxes_to_text

    # oneDNN breaks PaddlePaddle 3.x CPU inference (ConvertPirAttribute2RuntimeAttribute): plain CPU kernels.
    detector = TextDetection(model_name=DET_MODEL, device="cpu", enable_mkldnn=False)
    paddle = PaddleOCR(lang="vi", text_detection_model_name=DET_MODEL, use_doc_orientation_classify=False,
                       use_doc_unwarping=False, use_textline_orientation=False, device="cpu", enable_mkldnn=False)
    vietocr, vietocr_meta = _vietocr_predictor()
    cache.commit()

    rows: dict[str, list[dict]] = {"A": [], "B": []}
    for job in jobs:
        data = base64.b64decode(job["image_b64"])
        rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        base = {**job["sample"], "image_sha256": hashlib.sha256(data).hexdigest()}

        started = time.perf_counter()
        polys = list(detector.predict(rgb)[0]["dt_polys"])
        empty = [Box.from_polygon("x", p) for p in polys]
        texts = vietocr.predict_batch([_crop(rgb, b) for b in empty]) if empty else []
        boxes = [Box.from_polygon(t, p) for t, p in zip(texts, polys)]
        rows["A"].append(_row(base, boxes, boxes_to_text, (time.perf_counter() - started) * 1000))

        started = time.perf_counter()
        result = paddle.predict(rgb)[0]
        boxes = [Box.from_polygon(t, p) for t, p in zip(result["rec_texts"], result["rec_polys"])]
        rows["B"].append(_row(base, boxes, boxes_to_text, (time.perf_counter() - started) * 1000))

    import paddle as paddle_framework
    import paddleocr
    import torch

    versions = {"paddlepaddle": paddle_framework.__version__, "paddleocr": paddleocr.__version__,
                "vietocr": VIETOCR_VERSION, "torch": torch.__version__, "detector": DET_MODEL,
                "paddle_rec_lang": "vi", "mkldnn": False, **vietocr_meta,
                "paddlex_models": sorted(p.name for p in (Path(CACHE) / "paddlex" / "official_models").glob("*"))}
    return {"rows": rows, "versions": versions}


def _row(base: dict, boxes, to_text, latency_ms: float) -> dict:
    try:
        text, error = to_text(boxes), None
    except Exception as exc:  # noqa: BLE001  (recorded per sample)
        text, error = "", f"{type(exc).__name__}: {exc}"
    return {**base, "boxes": len(boxes), "text": text, "latency_ms": round(latency_ms, 1), "error": error}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()


@app.local_entrypoint()
def main(run_id: str, samples: str = DEFAULT_SAMPLES.as_posix()) -> None:
    from src.pipeline.library_ocr import (
        ENGINES,
        add_coverage,
        choose_engine,
        label_coverage,
    )

    rows_in = [json.loads(line) for line in (REPO_ROOT / samples).read_text(encoding="utf-8").splitlines() if line]
    out_dir = REPO_ROOT / "experiments/runs/m5_01" / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; use a new --run-id")
    jobs = [{"sample": {k: r[k] for k in ("sample_id", "img_id", "image_path")},
             "image_b64": base64.b64encode((REPO_ROOT / r["image_path"]).read_bytes()).decode("ascii")} for r in rows_in]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = ocr_lib_batch.remote(jobs)

    out_dir.mkdir(parents=True)
    pairs = {r["sample_id"]: r["key_text_pairs"] for r in rows_in}
    coverage: dict[str, dict] = {}
    latency: dict[str, float] = {}
    for engine, rows in result["rows"].items():
        with (out_dir / f"ocr_lib_{engine}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        exact, stripped = {}, {}
        for row in rows:
            add_coverage(exact, label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=False))
            add_coverage(stripped, label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=True))
        coverage[engine] = {"exact": exact, "stripped": stripped,
                            "errors": sum(1 for r in rows if r["error"])}
        latency[engine] = statistics.median(r["latency_ms"] for r in rows)

    baseline = {"exact": {}, "stripped": {}}
    for line in (REPO_ROOT / QWEN_BASELINE).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        add_coverage(baseline["exact"], label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=False))
        add_coverage(baseline["stripped"], label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=True))

    chosen = choose_engine({e: c["stripped"] for e, c in coverage.items()}, latency)
    report = {
        "task": "M5-01", "step": "choose-library-ocr", "run_id": run_id, "started_at_utc": started,
        "rule": "more MC-OCR label lines found after mark stripping and whitespace collapsing wins; tie -> lower median latency",
        "engines": ENGINES, "versions": result["versions"], "samples": len(rows_in),
        "coverage": coverage, "median_latency_ms_cpu4": latency,
        "found_stripped": {e: sum(v["found"] for v in c["stripped"].values()) for e, c in coverage.items()},
        "qwen_vl_baseline_not_eligible": baseline, "chosen_engine": chosen,
        "source_commit": _git("rev-parse", "HEAD"),
    }
    (REPO_ROOT / CHOICE_REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("found_stripped", "median_latency_ms_cpu4", "chosen_engine")}))
