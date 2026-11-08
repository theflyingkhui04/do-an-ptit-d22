"""Tests for the W1_v2 full-receipt workload package ([M5-01] node1-full-receipt-workload spec)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_w1_v2_workload as builder  # noqa: E402
import check_w1_dataset as checker  # noqa: E402
from src.pipeline import w1_workload  # noqa: E402

DEVSET = REPO_ROOT / "data/processed/m0_04_mcocr_devset/samples.jsonl"
V1_PROMPT = REPO_ROOT / "src/pipeline/prompts/node1_extractor_system.md"
RULES = REPO_ROOT / "src/pipeline/prompts/node1_full_receipt_rules.md"
W1_V1_PACKAGE_SHA256 = "e7cba647035f8834920d523ce8f6a70ef68896069ee71d3a862fac2fc42eff66"


def devset_rows() -> list[dict]:
    return [json.loads(line) for line in DEVSET.read_text(encoding="utf-8").splitlines() if line.strip()]


def ocr_rows(**overrides: dict) -> list[dict]:
    rows = []
    for sample in devset_rows():
        row = {
            "sample_id": sample["sample_id"],
            "img_id": sample["img_id"],
            "image_path": sample["image_path"],
            "image_sha256": "a" * 64,
            "scaled_sha256": "b" * 64,
            "scale": 2.0,
            "scaled_size": [1536, 2048],
            "ocr_model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "ocr_revision": "536a35794df8831aa814970ee8f89eff577e7718",
            "ocr_prompt_version": "receipt_ocr_full.v1",
            "ocr_prompt_sha256": "c" * 64,
            "text": f"CỬA HÀNG {sample['sample_id']}\n  Tổng tiền: 42.000  ",
            "finish_reason": "stop",
            "error": None,
        }
        row.update(overrides.get(sample["sample_id"], {}))
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def build(tmp_path: Path, rows: list[dict]) -> Path:
    ocr_path = write_jsonl(tmp_path / "ocr_texts.jsonl", rows)
    out_dir = tmp_path / "w1_v2"
    builder.main(["--ocr-jsonl", str(ocr_path), "--output-dir", str(out_dir),
                  "--report-dir", str(tmp_path / "results")])
    return out_dir


def load(path: Path) -> list[dict]:
    return w1_workload.load_jsonl(path)


def test_prompt_v2_is_v1_plus_rules():
    v1 = V1_PROMPT.read_text(encoding="utf-8")
    rules = RULES.read_text(encoding="utf-8")
    prompt = w1_workload.compose_node1_prompt_v2(v1, rules)
    assert prompt == v1.strip() + "\n\n" + rules.strip()
    assert "Never invent a time" in rules
    assert "totals.discount" in rules


def test_artifact_hash_policy(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"x\r\ny\r\n")
    assert w1_workload.artifact_sha256("samples_jsonl", path, None) == w1_workload.w1_v1_artifact_sha256("samples_jsonl", path)
    assert w1_workload.artifact_sha256("anything", path, "lf-canonical") == w1_workload.sha256_text("x\ny\n")
    with pytest.raises(ValueError):
        w1_workload.artifact_sha256("anything", path, "md5")


def test_build_complete_package(tmp_path):
    out_dir = build(tmp_path, ocr_rows())
    samples = load(out_dir / "samples.jsonl")
    requests = load(out_dir / "node1_chat_requests.jsonl")
    manifest = json.loads((out_dir / "package_manifest.json").read_text(encoding="utf-8"))

    assert len(samples) == len(requests) == 30
    first = samples[0]
    assert first["raw_text"] == f"CỬA HÀNG {first['sample_id']}\nTổng tiền: 42.000"
    assert first["dataset_version"] == "w1.m5_01.v2"
    request = requests[0]
    assert request["dataset_version"] == "w1.m5_01.v2"
    assert request["prompt_version"] == "node1_extractor.v2"
    assert request["expected_source"]["raw_text"] == first["raw_text"]
    assert request["request"]["max_tokens"] == 3072
    assert request["request"]["temperature"] == 0.0 and request["request"]["seed"] == 42
    assert request["request"]["response_format"] == {"type": "json_object"}
    user = request["request"]["messages"][1]["content"]
    assert user.startswith("SOURCE_METADATA:\n") and f"OCR_TEXT:\n{first['raw_text']}\n" in user

    system_prompt = (out_dir / "system_prompt.md").read_text(encoding="utf-8")
    assert system_prompt.startswith(V1_PROMPT.read_text(encoding="utf-8").strip())
    assert system_prompt.rstrip("\n").endswith(RULES.read_text(encoding="utf-8").strip())
    assert request["request"]["messages"][0]["content"] == system_prompt.rstrip("\n")

    assert manifest["workload_id"] == "w1_node1_invoice_extraction"
    assert manifest["dataset_version"] == "w1.m5_01.v2"
    assert manifest["prompt_version"] == "node1_extractor.v2"
    assert manifest["max_tokens"] == 3072
    assert manifest["hash_policy"] == "lf-canonical"
    assert manifest["requires_max_model_len"] == 8192
    assert manifest["ocr"]["model"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert manifest["ocr"]["revision"] == "536a35794df8831aa814970ee8f89eff577e7718"

    provenance = load(out_dir / "provenance.jsonl")
    assert provenance[0]["ocr"]["finish_reason"] == "stop"
    assert sum(provenance[0]["annotation_label_counts"].values()) > 0


def test_package_has_no_label_leakage(tmp_path):
    out_dir = build(tmp_path, ocr_rows())
    for name in ("samples.jsonl", "node1_chat_requests.jsonl", "node1_inputs.jsonl"):
        text = (out_dir / name).read_text(encoding="utf-8")
        assert "key_text_pairs" not in text and '"labels"' not in text


@pytest.mark.parametrize(
    "override, message",
    [
        ({"text": "  \n "}, "empty OCR text"),
        ({"finish_reason": "length"}, "finish_reason"),
        ({"error": "OcrError: boom", "text": ""}, "OCR error"),
    ],
)
def test_bad_ocr_row_is_refused_without_writing(tmp_path, override, message):
    rows = ocr_rows(m0_04_007=override)
    with pytest.raises(SystemExit, match=f"m0_04_007.*{message}|{message}.*m0_04_007"):
        build(tmp_path, rows)
    assert not (tmp_path / "w1_v2").exists()


def test_missing_and_unknown_samples_are_refused(tmp_path):
    rows = [row for row in ocr_rows() if row["sample_id"] != "m0_04_003"]
    with pytest.raises(SystemExit, match="m0_04_003"):
        build(tmp_path, rows)
    rows = ocr_rows() + [{**ocr_rows()[0], "sample_id": "m9_99_999"}]
    with pytest.raises(SystemExit, match="m9_99_999"):
        build(tmp_path, rows)
    assert not (tmp_path / "w1_v2").exists()


def test_refuses_to_overwrite_frozen_w1(tmp_path):
    ocr_path = write_jsonl(tmp_path / "ocr_texts.jsonl", ocr_rows())
    with pytest.raises(SystemExit, match="frozen"):
        builder.main(["--ocr-jsonl", str(ocr_path), "--output-dir", str(REPO_ROOT / "experiments/configs/w1"),
                      "--report-dir", str(tmp_path)])


def test_check_passes_on_built_v2_and_detects_tampering(tmp_path):
    out_dir = build(tmp_path, ocr_rows())
    assert checker.main(["--w1-dir", str(out_dir), "--no-write-report"]) == 0

    # A metadata field that no semantic check reads: only the hash can catch it.
    samples = out_dir / "samples.jsonl"
    rows = load(samples)
    rows[0]["source_csv"] = "tampered.csv"
    w1_workload.write_jsonl(samples, rows)
    with pytest.raises(AssertionError, match="hash mismatch for samples_jsonl"):
        checker.main(["--w1-dir", str(out_dir), "--no-write-report"])


def test_check_rejects_request_max_tokens_mismatch(tmp_path):
    out_dir = build(tmp_path, ocr_rows())
    path = out_dir / "node1_chat_requests.jsonl"
    rows = load(path)
    rows[0]["request"]["max_tokens"] = 1024
    w1_workload.write_jsonl(path, rows)
    with pytest.raises(AssertionError, match="max_tokens"):
        checker.main(["--w1-dir", str(out_dir), "--no-write-report"])


def test_check_frozen_w1_v1_unchanged():
    assert checker.main(["--no-write-report"]) == 0
    manifest = json.loads((REPO_ROOT / "experiments/configs/w1/package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_sha256"] == W1_V1_PACKAGE_SHA256
