"""[M5-01 canonical flow] W1_v3 package: library OCR text with Vietnamese marks removed."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_w1_v3_workload as builder
import check_w1_dataset as checker

from src.pipeline import w1_workload
from src.pipeline.text_normalize import has_vietnamese_marks

DEVSET = REPO_ROOT / "data/processed/m0_04_mcocr_devset/samples.jsonl"
CHOICE = {"chosen_engine": "A", "engines": {"A": "PaddleOCR det + VietOCR rec"},
          "versions": {"paddleocr": "3.7.0", "vietocr": "0.3.13", "weights_sha256": "f" * 64}}


def devset() -> list[dict]:
    return [json.loads(line) for line in DEVSET.read_text(encoding="utf-8").splitlines() if line.strip()]


def ocr_rows(**overrides: dict) -> list[dict]:
    rows = []
    for sample in devset():
        row = {"sample_id": sample["sample_id"], "img_id": sample["img_id"], "image_path": sample["image_path"],
               "image_sha256": "a" * 64, "boxes": 12, "latency_ms": 900.0, "error": None,
               "text": f"CỬA HÀNG {sample['sample_id']}\n  Tổng tiền: 42.000đ  "}
        row.update(overrides.get(sample["sample_id"], {}))
        rows.append(row)
    return rows


def build(tmp_path: Path, rows: list[dict]) -> Path:
    ocr = tmp_path / "ocr_lib_A.jsonl"
    ocr.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    choice = tmp_path / "choice.json"
    choice.write_text(json.dumps(CHOICE), encoding="utf-8")
    out = tmp_path / "w1_v3"
    builder.main(["--ocr-jsonl", str(ocr), "--choice-report", str(choice), "--output-dir", str(out),
                  "--report-dir", str(tmp_path / "results")])
    return out


def test_package_has_stripped_library_ocr_text(tmp_path):
    out = build(tmp_path, ocr_rows())
    samples = w1_workload.load_jsonl(out / "samples.jsonl")
    requests = w1_workload.load_jsonl(out / "node1_chat_requests.jsonl")
    manifest = json.loads((out / "package_manifest.json").read_text(encoding="utf-8"))
    first = samples[0]
    assert first["raw_text"] == f"CUA HANG {first['sample_id']}\nTong tien: 42.000d"
    assert first["dataset_version"] == "w1.m5_01.v3"
    assert all(not has_vietnamese_marks(s["raw_text"]) for s in samples)
    assert requests[0]["request"]["max_tokens"] == 3072 and requests[0]["prompt_version"] == "node1_extractor.v2"
    assert requests[0]["expected_source"]["raw_text"] == first["raw_text"]
    assert manifest["dataset_version"] == "w1.m5_01.v3" and manifest["requires_max_model_len"] == 8192
    assert manifest["mark_stripping"].startswith("strip_vietnamese_marks")
    assert manifest["ocr"]["engine"] == "A" and manifest["ocr"]["kind"] == "library"
    assert manifest["ocr"]["versions"]["vietocr"] == "0.3.13"
    assert checker.main(["--w1-dir", str(out), "--no-write-report"]) == 0


@pytest.mark.parametrize("override, message", [
    ({"text": "  "}, "empty OCR text"),
    ({"error": "OcrError: no text detected", "text": ""}, "OCR error"),
    ({"img_id": "other.jpg"}, "img_id"),
])
def test_bad_rows_are_refused(tmp_path, override, message):
    with pytest.raises(SystemExit, match=message):
        build(tmp_path, ocr_rows(m0_04_007=override))
    assert not (tmp_path / "w1_v3").exists()


def test_refuses_frozen_packages(tmp_path):
    ocr = tmp_path / "o.jsonl"
    ocr.write_text("".join(json.dumps(r) + "\n" for r in ocr_rows()), encoding="utf-8")
    choice = tmp_path / "c.json"
    choice.write_text(json.dumps(CHOICE), encoding="utf-8")
    for frozen in ("w1", "w1_v2"):
        with pytest.raises(SystemExit, match="frozen"):
            builder.main(["--ocr-jsonl", str(ocr), "--choice-report", str(choice),
                          "--output-dir", str(REPO_ROOT / "experiments/configs" / frozen), "--report-dir", str(tmp_path)])


def test_paths_outside_the_repository_and_temp_are_refused(tmp_path):
    elsewhere = Path(os.path.abspath(os.sep)) / "ptit-not-allowed"
    with pytest.raises(SystemExit, match="outside the repository"):
        builder.main(["--ocr-jsonl", str(tmp_path / "o.jsonl"), "--output-dir", str(elsewhere / "w1_v3")])
    out = build(tmp_path, ocr_rows())
    with pytest.raises(SystemExit, match="outside the repository"):
        checker.main(["--w1-dir", str(out), "--report", str(elsewhere / "summary.json")])
    assert not elsewhere.exists()


def test_default_report_is_a_fixed_path_per_dataset_version():
    assert checker.default_report_path("w1.m5_01.v3") == Path("experiments/results/m5_01_w1_v3_check_summary.json")
    with pytest.raises(AssertionError, match="unexpected dataset_version"):
        checker.default_report_path("../../elsewhere")


def test_checker_rejects_marks_in_a_stripped_package(tmp_path):
    with pytest.raises(AssertionError, match="Vietnamese marks"):
        checker.assert_marks_stripped([{"sample_id": "s1", "raw_text": "Tổng"}], {"mark_stripping": "strip_vietnamese_marks"})
    checker.assert_marks_stripped([{"sample_id": "s1", "raw_text": "Tổng"}], {})  # older packages keep their marks
