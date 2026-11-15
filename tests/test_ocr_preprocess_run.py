"""[M5-01 OCR giữ dấu, 14/11] Selection set and scoring of experiments/modal_ocr_preprocess.py (no Modal call)."""

from __future__ import annotations

from experiments import modal_ocr_preprocess as run

FULL = [{"label": label, "text": f"{label} text"} for label in run.REQUIRED_LABELS]


def train_rows() -> list[dict]:
    rows = []
    for i in range(200):
        quality = (0.3, 0.55, 0.7, 0.85, 0.95)[i % 5]
        pairs = FULL if i % 7 else FULL[:3]  # every 7th row lacks TOTAL_COST
        rows.append({"img_id": f"img_{i:03d}.jpg", "image_path": f"x/img_{i:03d}.jpg", "quality": quality,
                     "key_text_pairs": pairs})
    return rows


def test_selection_set_is_fixed_outside_the_devset_with_all_labels_and_both_bands():
    devset = {"img_003.jpg", "img_004.jpg"}
    picked = run.selection_set(train_rows(), devset, per_band=10)
    assert picked == run.selection_set(train_rows(), devset, per_band=10)       # same seed, same set
    assert len(picked) == 20 and len({p["img_id"] for p in picked}) == 20
    assert not devset & {p["img_id"] for p in picked}
    assert all({q["label"] for q in p["key_text_pairs"]} >= set(run.REQUIRED_LABELS) for p in picked)
    assert [p["band"] for p in picked] == ["q>=0.8"] * 10 + ["0.5<=q<0.8"] * 10
    assert all(p["quality"] >= 0.8 for p in picked[:10]) and all(0.5 <= p["quality"] < 0.8 for p in picked[10:])
    assert [p["sample_id"] for p in picked[:2]] == ["sel_001", "sel_002"]
    assert picked != run.selection_set(train_rows(), devset, seed=1, per_band=10)


def test_score_counts_lines_with_and_without_marks_and_foreign_letters():
    pairs = {"s1": [{"label": "ADDRESS", "text": "Gia Lâm"}, {"label": "SELLER", "text": "CỬA HÀNG"}]}
    rows = [{"sample_id": "s1", "text": "CỬA HÀNG\nChợ Gia Làm ső", "latency_ms": 1000.0, "error": None}]
    out = run.score(rows, pairs)
    assert (out["exact"], out["stripped"], out["total"]) == (1, 2, 2)
    assert out["foreign_words"] == 1 and out["foreign_examples"] == ["ső"]
    assert out["median_latency_ms"] == 1000.0 and out["errors"] == 0


def test_every_configuration_adds_one_step_to_its_parent():
    parent = {"P1": "P0", "P2": "P1", "P3a": "P2", "P3b": "P2", "P4a": "P3a", "P4b": "P3b"}
    for child, base in parent.items():
        changed = {k for k in run.CONFIGS[child] if run.CONFIGS[child][k] != run.CONFIGS[base][k]}
        assert len(changed) == 1, (child, changed)
