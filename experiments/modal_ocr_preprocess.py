"""[M5-01 OCR giữ dấu, 14/11] Image preprocessing ablation for PaddleOCR detection + VietOCR recognition.

Design: openspec change m5-01-canonical-flow §7 (approved by Dương 14/11). Each configuration adds one step:

    P0   the 08/11 engine A (axis-aligned line crop, VietOCR predict_batch = greedy)
    P1   P0 + each line straightened (perspective strip) with a 15% margin above and below
    P2   P1 + VietOCR beam search (predict() per line: predict_batch ignores the beam-search flag)
    P3a  P2 + page levelled by the median slope of the detected lines, then detected again
    P3b  P2 + page unwarped by PaddleOCR UVDoc
    P4a  P3a + CLAHE + mild unsharp mask
    P4b  P3b + CLAHE + mild unsharp mask

The choice runs on 60 labelled MC-OCR train images outside the devset (30 with quality >= 0.8, 30 in [0.5, 0.8),
fixed seed); the 30 devset images run once afterwards, for P0 and the chosen configuration only.

    modal run -m experiments.modal_ocr_preprocess --phase smoke --run-id 20261114-m501-ocr-preprocess-smoke
    modal run -m experiments.modal_ocr_preprocess --phase select --run-id 20261114-m501-ocr-preprocess
    modal run -m experiments.modal_ocr_preprocess --phase devset --run-id 20261114-m501-ocr-preprocess

Outputs: experiments/runs/m5_01/<run>/<phase>/<config>.jsonl (OCR text per image) and
experiments/results/m5_01_ocr_preprocess_choice.json (rule, scores, choice, devset check).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from experiments.modal_ocr_lib import (
    CACHE,
    DET_MODEL,
    REMOTE_PROJECT_ROOT,
    VIETOCR_VERSION,
    VIETOCR_WEIGHTS_URL,
    cache,
)
from experiments.modal_ocr_lib import image as ocr_image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVSET = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
MCOCR_ROOT = Path("data/raw/mc_ocr_2021")
MCOCR_TRAIN_CSV = MCOCR_ROOT / "mcocr_train_df.csv"
REPORT = Path("experiments/results/m5_01_ocr_preprocess_choice.json")
REQUIRED_LABELS = ("SELLER", "ADDRESS", "TIMESTAMPS", "TOTAL_COST")
SELECTION_SEED = 20261114
BANDS = {"q>=0.8": (0.8, 1.01), "0.5<=q<0.8": (0.5, 0.8)}
PER_BAND = 30
CHUNK = 6
UNWARP_MODEL = "UVDoc"
CONFIGS: dict[str, dict[str, Any]] = {
    "P0": {"page": None, "enhance": False, "crop": "axis", "beam": False},
    "P1": {"page": None, "enhance": False, "crop": "strip", "beam": False},
    "P2": {"page": None, "enhance": False, "crop": "strip", "beam": True},
    "P3a": {"page": "level", "enhance": False, "crop": "strip", "beam": True},
    "P3b": {"page": "unwarp", "enhance": False, "crop": "strip", "beam": True},
    "P4a": {"page": "level", "enhance": True, "crop": "strip", "beam": True},
    "P4b": {"page": "unwarp", "enhance": True, "crop": "strip", "beam": True},
}
RULE = ("most MC-OCR label lines found with their marks (whitespace collapsed, case-sensitive: label_coverage of the "
        "08/11 run); tie -> most found after stripping marks; tie -> lower median latency; median latency above "
        "2 x P0 cannot win; a full tie keeps P0")

app = modal.App("ptit-ocr-preprocess", image=ocr_image)


# ---------------------------------------------------------------- local helpers (tested in tests/test_ocr_preprocess_run.py)
def selection_set(rows: list[dict[str, Any]], devset_ids: set[str], seed: int = SELECTION_SEED,
                  per_band: int = PER_BAND) -> list[dict[str, Any]]:
    """``per_band`` images per quality band, outside the devset, with all four label types; ``rows`` are parsed
    MC-OCR train rows {img_id, image_path, quality, key_text_pairs}. Sorted by img_id before sampling."""
    picked = []
    for band, (low, high) in BANDS.items():
        pool = sorted((r for r in rows if r["img_id"] not in devset_ids and low <= r["quality"] < high
                       and set(REQUIRED_LABELS) <= {p["label"] for p in r["key_text_pairs"]}),
                      key=lambda r: r["img_id"])
        chosen = random.Random(f"{seed}:{band}").sample(pool, per_band)
        picked += [{**r, "band": band} for r in sorted(chosen, key=lambda r: r["img_id"])]
    return [{**r, "sample_id": f"sel_{i + 1:03d}"} for i, r in enumerate(picked)]


def score(rows: list[dict[str, Any]], pairs: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Label coverage with and without marks, words outside the Vietnamese alphabet, median latency."""
    from src.api.questions import foreign_words
    from src.pipeline.library_ocr import add_coverage, coverage_found, label_coverage

    exact: dict[str, dict[str, int]] = {}
    stripped: dict[str, dict[str, int]] = {}
    odd: list[str] = []
    for row in rows:
        add_coverage(exact, label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=False))
        add_coverage(stripped, label_coverage(row["text"], pairs[row["sample_id"]], strip_marks=True))
        odd += [w for w in foreign_words(row["text"]) if not w.isascii()]
    return {"exact": coverage_found(exact), "stripped": coverage_found(stripped),
            "total": sum(v["total"] for v in exact.values()), "by_label_exact": exact, "by_label_stripped": stripped,
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
            "foreign_words": len(odd), "foreign_examples": sorted(set(odd))[:12],
            "errors": sum(1 for r in rows if r.get("error"))}


# ---------------------------------------------------------------- Modal (CPU, like the 08/11 run)
def _predictor(beam: bool):
    import requests
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    weights = Path(CACHE) / "vietocr" / "vgg_transformer.pth"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(VIETOCR_WEIGHTS_URL, timeout=600)
        response.raise_for_status()
        weights.write_bytes(response.content)
    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = str(weights)
    config["device"] = "cpu"
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = beam
    return Predictor(config), hashlib.sha256(weights.read_bytes()).hexdigest()


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes={CACHE: cache})
def run_chunk(jobs: list[dict], configs: list[str], debug: bool = False) -> dict:
    """Every configuration on every image of the chunk, one after the other in this container (comparable times)."""
    sys.path.insert(0, REMOTE_PROJECT_ROOT)
    import numpy as np
    from paddleocr import TextDetection, TextImageUnwarping
    from PIL import Image

    from src.pipeline import ocr_preprocess as pre
    from src.pipeline.library_ocr import Box, boxes_to_text

    detector = TextDetection(model_name=DET_MODEL, device="cpu", enable_mkldnn=False)
    unwarper = (TextImageUnwarping(model_name=UNWARP_MODEL, device="cpu", enable_mkldnn=False)
                if any(CONFIGS[c]["page"] == "unwarp" for c in configs) else None)
    greedy, weights_sha = _predictor(beam=False)
    beam, _ = _predictor(beam=True)
    cache.commit()

    def detect(rgb):
        return [np.asarray(p, dtype=float) for p in detector.predict(rgb)[0]["dt_polys"]]

    def unwarp(rgb):
        result = unwarper.predict(rgb)[0]
        out = np.asarray(result["doctr_img"])
        if out.dtype != np.uint8:
            out = np.clip(out * (255.0 if out.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out[:, :, :3])

    def run_config(name: str, rgb):
        step, extra = CONFIGS[name], {}
        work = rgb
        polys = None
        if step["page"] == "unwarp":
            work = unwarp(work)
            extra["page_size"] = list(work.shape[:2])
        elif step["page"] == "level":
            polys = detect(work)
            angle = pre.skew_angle(polys)
            extra["skew_deg"] = round(angle, 2)
            original = Image.fromarray(work)
            levelled = pre.deskew(original, angle)
            if levelled is not original:  # rotated: detect the lines again on the level page
                work, polys = np.asarray(levelled), None
        if step["enhance"]:
            work, polys = pre.enhance(work), None
        if polys is None:
            polys = detect(work)
        page = Image.fromarray(work)
        crops = [pre.axis_crop(page, p) if step["crop"] == "axis" else pre.line_strip(page, p) for p in polys]
        if not crops:
            texts = []
        elif step["beam"]:
            texts = [beam.predict(c) for c in crops]
        else:
            texts = greedy.predict_batch(crops)
        boxes = [Box.from_polygon(t, p) for t, p in zip(texts, polys)]
        if debug:
            extra["debug"] = {"page_png_b64": _png_b64(page, 1400),
                              "strips_png_b64": [_png_b64(c, 900) for c in crops[:8]],
                              "lines": texts[:8]}
        return boxes, extra

    rows: dict[str, list[dict]] = {c: [] for c in configs}
    for job in jobs:
        data = base64.b64decode(job["image_b64"])
        rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        base = {**job["sample"], "image_sha256": hashlib.sha256(data).hexdigest()}
        for name in configs:
            started = time.perf_counter()
            try:
                boxes, extra = run_config(name, rgb)
                text, error = boxes_to_text(boxes), None
            except Exception as exc:  # noqa: BLE001  (recorded per sample and configuration)
                boxes, extra, text, error = [], {}, "", f"{type(exc).__name__}: {exc}"[:300]
            rows[name].append({**base, "config": name, "boxes": len(boxes), "text": text,
                               "latency_ms": round((time.perf_counter() - started) * 1000, 1), "error": error, **extra})

    import cv2
    import paddle as paddle_framework
    import paddleocr
    import torch

    versions = {"paddlepaddle": paddle_framework.__version__, "paddleocr": paddleocr.__version__,
                "vietocr": VIETOCR_VERSION, "torch": torch.__version__, "opencv": cv2.__version__,
                "detector": DET_MODEL, "unwarp_model": UNWARP_MODEL, "mkldnn": False,
                "weights_url": VIETOCR_WEIGHTS_URL, "weights_sha256": weights_sha}
    return {"rows": rows, "versions": versions}


def _png_b64(img, max_side: int) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------- local entrypoint
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def _train_rows() -> list[dict[str, Any]]:
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import prepare_mcocr_devset as prep

    image_index, _ = prep.find_image_index(REPO_ROOT / MCOCR_ROOT)
    rows = []
    for _, row in pd.read_csv(REPO_ROOT / MCOCR_TRAIN_CSV).iterrows():
        img_id = str(row.get("img_id", "")).strip()
        if img_id not in image_index:
            continue
        labels = [prep.normalize_label(x) for x in prep.split_mcocr_field(row.get("anno_labels"))]
        texts = prep.split_mcocr_field(row.get("anno_texts"))
        rows.append({"img_id": img_id, "image_path": image_index[img_id].relative_to(REPO_ROOT).as_posix(),
                     "quality": prep.safe_float(row.get("anno_image_quality")),
                     "key_text_pairs": prep.build_key_text_pairs(labels, texts)})
    return rows


def _jobs(samples: list[dict[str, Any]]) -> list[dict]:
    return [{"sample": {k: s[k] for k in ("sample_id", "img_id", "image_path") if k in s},
             "image_b64": base64.b64encode((REPO_ROOT / s["image_path"]).read_bytes()).decode("ascii")}
            for s in samples]


def _run(samples: list[dict[str, Any]], configs: list[str], out_dir: Path) -> tuple[dict, dict]:
    jobs = _jobs(samples)
    chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
    rows: dict[str, list[dict]] = {c: [] for c in configs}
    versions: dict = {}
    for result in run_chunk.map(chunks, kwargs={"configs": configs}):
        versions = result["versions"]
        for name, part in result["rows"].items():
            rows[name] += part
    out_dir.mkdir(parents=True)
    for name, part in rows.items():
        part.sort(key=lambda r: r["sample_id"])
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pairs = {s["sample_id"]: s["key_text_pairs"] for s in samples}
    return {name: score(part, pairs) for name, part in rows.items()}, versions


@app.local_entrypoint()
def main(phase: str, run_id: str) -> None:
    from src.pipeline.ocr_preprocess import choose_config

    devset = [json.loads(line) for line in (REPO_ROOT / DEVSET).read_text(encoding="utf-8").splitlines() if line]
    selection = selection_set(_train_rows(), {r["img_id"] for r in devset})
    run_dir = REPO_ROOT / "experiments/runs/m5_01" / run_id
    report_path = REPO_ROOT / REPORT
    provenance = {"source_commit": _git("rev-parse", "HEAD"), "worktree_dirty": bool(_git("status", "--porcelain")),
                  "script_sha256": _sha(Path("experiments/modal_ocr_preprocess.py")),
                  "module_sha256": _sha(Path("src/pipeline/ocr_preprocess.py"))}

    if phase == "smoke":  # one selection image, every configuration, page and strip images for a visual check
        result = run_chunk.remote(_jobs(selection[:1]), list(CONFIGS), debug=True)
        run_dir.mkdir(parents=True)
        for name, part in result["rows"].items():
            row = part[0]
            debug = row.pop("debug", {})
            for key, value in (("page", debug.get("page_png_b64")),):
                if value:
                    (run_dir / f"{name}_{key}.png").write_bytes(base64.b64decode(value))
            for i, value in enumerate(debug.get("strips_png_b64", [])):
                (run_dir / f"{name}_strip{i}.png").write_bytes(base64.b64decode(value))
            print(f"{name}: {row['latency_ms']:.0f} ms, {row['boxes']} boxes, error={row['error']}, "
                  f"extra={ {k: v for k, v in row.items() if k in ('skew_deg', 'page_size')} }, lines={debug.get('lines')}")
        print(json.dumps(result["versions"]))
        return

    if phase == "select":
        if report_path.exists():
            raise SystemExit(f"{report_path} exists: the choice runs once")
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        scores, versions = _run(selection, list(CONFIGS), run_dir / "select")
        decision = choose_config({n: s for n, s in scores.items()})
        report = {"task": "M5-01", "step": "ocr-preprocess-choice", "run_id": run_id, "started_at_utc": started,
                  "design": "openspec m5-01-canonical-flow §7 (local), approved by Dương 14/11", "rule": RULE,
                  "configs": CONFIGS, "selection": {"seed": SELECTION_SEED, "bands": BANDS, "per_band": PER_BAND,
                                                    "img_ids": [s["img_id"] for s in selection]},
                  "scores_selection": scores, "decision": decision, "versions": versions, **provenance}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({n: {k: s[k] for k in ("exact", "stripped", "total", "median_latency_ms", "foreign_words")}
                          for n, s in scores.items()}))
        print(json.dumps(decision))
        return

    if phase == "devset":
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "scores_devset" in report:
            raise SystemExit("the devset check already ran")
        configs = sorted({"P0", report["decision"]["chosen"]}, key=list(CONFIGS).index)
        scores, _ = _run(devset, configs, run_dir / "devset")
        report["scores_devset"] = scores
        report["devset_provenance"] = provenance
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({n: {k: s[k] for k in ("exact", "stripped", "total", "median_latency_ms", "foreign_words")}
                          for n, s in scores.items()}))
        return

    raise SystemExit("phase must be smoke, select or devset")
