"""[M5-01 Node 1 sửa dấu, 14/11] Prompt v3, request arms and mark scoring (design §8 of m5-01-canonical-flow)."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline import node1_marks as marks
from src.pipeline.text_normalize import strip_vietnamese_marks
from src.pipeline.w1_workload import compose_node1_prompt_v2

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = (REPO_ROOT / "src/pipeline/prompts/node1_extractor_system.md").read_text(encoding="utf-8")
RULES = (REPO_ROOT / "src/pipeline/prompts/node1_full_receipt_rules.md").read_text(encoding="utf-8")
W1_V3 = REPO_ROOT / "experiments/configs/w1_v3/node1_chat_requests.jsonl"
OCR_A = REPO_ROOT / "experiments/runs/m5_01/20261108-m501-ocrlib-choice/ocr_lib_A.jsonl"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prompt_v3_is_v2_plus_the_mark_rules():
    v3 = marks.system_prompt("B2")
    assert v3.startswith(compose_node1_prompt_v2(V1, RULES)) and v3 != compose_node1_prompt_v2(V1, RULES)
    assert marks.system_prompt("B0") == marks.system_prompt("B1") == compose_node1_prompt_v2(V1, RULES)
    rules = marks.MARK_RULES.read_text(encoding="utf-8")
    assert "never add, remove or replace letters" in rules and "verbatim" in rules
    for word in ("Lâm", "tiền", "HÀNG", "Thụy", "Phú"):  # words of the label sets: no hint towards the test set
        assert word not in rules


def test_arms_differ_only_in_marks_and_prompt():
    rows = [{"sample_id": "s1", "img_id": "a.jpg", "image_path": "x/a.jpg", "text": "Cám ơn quy khách\n\nTổng: 5,000 "}]
    b0, b1, b2 = (marks.arm_requests(rows, arm, "m")[0] for arm in marks.ARMS)
    assert b0["expected_source"]["raw_text"] == "Cam on quy khach\nTong: 5,000"
    assert b1["expected_source"]["raw_text"] == b2["expected_source"]["raw_text"] == "Cám ơn quy khách\nTổng: 5,000"
    assert (b0["prompt_version"], b1["prompt_version"], b2["prompt_version"]) == (
        "node1_extractor.v2", "node1_extractor.v2", "node1_extractor.v3")
    assert b1["request"]["messages"][1] == b2["request"]["messages"][1]
    assert b0["request"]["messages"][0] == b1["request"]["messages"][0] != b2["request"]["messages"][0]
    assert {b["arm"] for b in (b0, b1, b2)} == {"B0", "B1", "B2"}


def test_arm_b0_on_the_devset_is_exactly_the_w1_v3_package():
    frozen = {r["sample_id"]: r for r in jsonl(W1_V3)}
    built = {r["sample_id"]: r for r in marks.arm_requests(jsonl(OCR_A), "B0", "m")}
    assert built.keys() == frozen.keys()
    for sample_id, row in built.items():
        assert row["request"]["messages"] == frozen[sample_id]["request"]["messages"], sample_id
        assert row["request"]["max_tokens"] == frozen[sample_id]["request"]["max_tokens"]


def payload(name: str | None, address: str | None = None) -> dict:
    return {"seller": {"name": name, "address": address}}


def test_mark_counts_separate_wrong_marks_from_wrong_letters():
    labels = {
        "s1": [{"label": "SELLER", "text": "CỬA HÀNG NĂM OÁNH"}, {"label": "ADDRESS", "text": "Chợ Sủi Gia Lâm"}],
        "s2": [{"label": "SELLER", "text": "MINIMART ANAN"}],
        "s3": [{"label": "SELLER", "text": "VinMart"}],
    }
    outputs = {"s1": payload("CỬA HANG NĂM OÁNH", "Chợ Sủi Gia Lâm"),  # seller: right letters, wrong mark
               "s2": payload("MINIMART ANNA")}                         # wrong letters; s3: no valid JSON
    out = marks.mark_counts(labels, outputs)
    assert out == {"samples": 3, "valid_json": 2, "labelled_fields": 4, "exact": 1, "stripped": 2,
                   "right_letters_wrong_marks": 1}


def test_stripped_output_counts_as_wrong_marks_against_accented_labels():
    labels = {"s1": [{"label": "SELLER", "text": "Cửa hàng"}]}
    out = marks.mark_counts(labels, {"s1": payload(strip_vietnamese_marks("Cửa hàng"))})
    assert (out["exact"], out["stripped"], out["right_letters_wrong_marks"]) == (0, 1, 1)


def score(exact, stripped, valid, wrong):
    return {"exact": exact, "stripped": stripped, "valid_json": valid, "right_letters_wrong_marks": wrong}


def test_choose_b2_only_when_every_condition_holds():
    ok = {"B0": score(10, 80, 60, 70), "B1": score(60, 79, 60, 19), "B2": score(70, 80, 59, 4)}
    assert marks.choose_arm(ok)["chosen"] == "B2"
    cases = {
        "not better than B1": {**ok, "B2": score(60, 80, 60, 4)},
        "letters lost": {**ok, "B2": score(70, 78, 60, 4)},
        "json lost": {**ok, "B2": score(70, 80, 58, 4)},
        "too many wrong marks": {**ok, "B2": score(70, 80, 60, 5)},  # 5 > 5% of 80
    }
    for reason, scores in cases.items():
        out = marks.choose_arm(scores)
        assert out["chosen"] == "B0", reason
        assert out["failed"], reason
