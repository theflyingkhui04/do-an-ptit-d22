"""[M5-01 Node 1 sửa dấu, 14/11] Can Node 1 write correct Vietnamese marks from OCR text with wrong marks?

Design: openspec change m5-01-canonical-flow §8 (approved by Dương 14/11). Three arms on the same OCR text:

    B0  prompt node1_extractor.v2, OCR text with marks stripped (the current flow, = W1_v3 on the devset)
    B1  prompt node1_extractor.v2, OCR text with the OCR's own marks
    B2  prompt node1_extractor.v3 (v2 + node1_marks_rules.md), OCR text with the OCR's own marks

Scored on SELLER and ADDRESS (src/pipeline/entity_eval.py): match with marks, match after stripping marks, and
"right letters, wrong marks" (matches only after stripping) = the error a viewer of an accented database would see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.entity_eval import MATCH, score_sample
from src.pipeline.node1_request import build_chat_request, normalize_text
from src.pipeline.text_normalize import strip_vietnamese_marks
from src.pipeline.w1_workload import W1_V2_PROMPT_VERSION, compose_node1_prompt_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = REPO_ROOT / "src/pipeline/prompts"
V1_PROMPT = PROMPTS / "node1_extractor_system.md"
FULL_RECEIPT_RULES = PROMPTS / "node1_full_receipt_rules.md"
MARK_RULES = PROMPTS / "node1_marks_rules.md"
V3_PROMPT_VERSION = "node1_extractor.v3"
ARMS = ("B0", "B1", "B2")
TEXT_FIELDS = ("SELLER", "ADDRESS")
MAX_TOKENS = 3072  # as W1_v3
WRONG_MARK_SHARE = 0.05  # B2 may show wrong marks on at most 5% of its letter-correct fields (Dương 14/11)


def system_prompt(arm: str) -> str:
    v2 = compose_node1_prompt_v2(V1_PROMPT.read_text(encoding="utf-8"), FULL_RECEIPT_RULES.read_text(encoding="utf-8"))
    return v2 if arm in ("B0", "B1") else v2 + "\n\n" + MARK_RULES.read_text(encoding="utf-8").strip()


def arm_requests(ocr_rows: list[dict[str, Any]], arm: str, model: str) -> list[dict[str, Any]]:
    """One Node 1 request per OCR row {sample_id, img_id, image_path, text}; B0 strips the marks first."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    prompt = system_prompt(arm)
    rows = []
    for ocr in ocr_rows:
        text = strip_vietnamese_marks(ocr["text"]) if arm == "B0" else ocr["text"]
        sample = {"sample_id": ocr["sample_id"], "img_id": ocr["img_id"], "image_path": ocr["image_path"],
                  "raw_text": normalize_text(text)}
        row = build_chat_request(sample, prompt, model, MAX_TOKENS, 0.0, 42)
        row["prompt_version"] = V3_PROMPT_VERSION if arm == "B2" else W1_V2_PROMPT_VERSION
        rows.append({"arm": arm, **row})
    return rows


def mark_counts(labels: dict[str, list[dict[str, str]]], outputs: dict[str, dict[str, Any]]) -> dict[str, int]:
    """SELLER + ADDRESS counts over all labelled samples; a sample without valid JSON scores as blank."""
    counts = {"samples": len(labels), "valid_json": sum(1 for s in labels if outputs.get(s) is not None),
              "labelled_fields": 0, "exact": 0, "stripped": 0, "right_letters_wrong_marks": 0}
    for sample_id, pairs in labels.items():
        payload = outputs.get(sample_id)
        with_marks = score_sample(payload, pairs)
        without = score_sample(payload, pairs, strip_marks=True)
        for field in TEXT_FIELDS:
            if not any(p["label"] == field for p in pairs):
                continue
            counts["labelled_fields"] += 1
            exact, stripped = with_marks[field].status == MATCH, without[field].status == MATCH
            counts["exact"] += exact
            counts["stripped"] += stripped
            counts["right_letters_wrong_marks"] += stripped and not exact
    return counts


def choose_arm(scores: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Rule fixed before the run (design §8): B2 (store marks) only if every condition holds, else B0 (no marks)."""
    b0, b1, b2 = scores["B0"], scores["B1"], scores["B2"]
    checks = {
        "B2 exact > B1 exact": b2["exact"] > b1["exact"],
        "B2 stripped >= B0 stripped - 1": b2["stripped"] >= b0["stripped"] - 1,
        "B2 valid JSON >= B0 valid JSON - 1": b2["valid_json"] >= b0["valid_json"] - 1,
        "B2 wrong marks <= 5% of B2 stripped": b2["right_letters_wrong_marks"] <= WRONG_MARK_SHARE * b2["stripped"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"chosen": "B0" if failed else "B2", "checks": checks, "failed": failed}
