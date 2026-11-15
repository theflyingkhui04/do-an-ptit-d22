"""[M5-01 canonical flow] Mark stripping, fixed line ordering and label coverage for library OCR."""

from __future__ import annotations

import pytest

from src.pipeline.library_ocr import Box, boxes_to_text, label_coverage
from src.pipeline.ocr import OcrError
from src.pipeline.text_normalize import has_vietnamese_marks, strip_vietnamese_marks


def test_strip_marks_removes_every_vietnamese_diacritic():
    assert strip_vietnamese_marks("Tổng tiền: 42.000đ — Đường Trần Phú") == "Tong tien: 42.000d — Duong Tran Phu"
    assert strip_vietnamese_marks("CỬA HÀNG NĂM OÁNH, Sữa tươi, Quầy ưu đãi") == "CUA HANG NAM OANH, Sua tuoi, Quay uu dai"
    assert strip_vietnamese_marks("MINIMART ANAN 42,000") == "MINIMART ANAN 42,000"
    assert not has_vietnamese_marks(strip_vietnamese_marks("ấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵđĐ"))
    assert has_vietnamese_marks("Tổng") and not has_vietnamese_marks("Tong")


def box(text: str, x: float, y: float, w: float = 80, h: float = 20) -> Box:
    return Box(text=text, x0=x, y0=y, x1=x + w, y1=y + h)


def test_boxes_on_one_row_are_joined_left_to_right():
    boxes = [box("42,000", 300, 102), box("Tong tien", 10, 100), box("MINIMART ANAN", 50, 10)]
    assert boxes_to_text(boxes) == "MINIMART ANAN\nTong tien 42,000"


def test_rows_are_split_when_centres_differ_by_half_a_box_height():
    boxes = [box("a", 10, 100), box("b", 10, 109), box("c", 10, 111)]
    # median height 20 -> threshold 10: b (centre 119) joins a (110); c (121) starts a new row
    assert boxes_to_text(boxes) == "a b\nc"


def test_empty_detection_is_an_explicit_error():
    with pytest.raises(OcrError, match="no text detected"):
        boxes_to_text([])
    with pytest.raises(OcrError, match="no text detected"):
        boxes_to_text([box("   ", 0, 0)])


def test_label_coverage_counts_label_lines_with_and_without_marks():
    pairs = [{"label": "SELLER", "text": "CỬA HÀNG NĂM OÁNH"}, {"label": "TOTAL_COST", "text": "Tổng tiền"},
             {"label": "TOTAL_COST", "text": "42,000"}, {"label": "ADDRESS", "text": "Hà  Nội"}]
    text = "CUA HANG NAM OANH\nTong tien 42,000\nHa Noi"
    exact = label_coverage(text, pairs, strip_marks=False)
    plain = label_coverage(text, pairs, strip_marks=True)
    assert exact == {"SELLER": {"total": 1, "found": 0}, "TOTAL_COST": {"total": 2, "found": 1},
                     "ADDRESS": {"total": 1, "found": 0}}
    assert plain == {"SELLER": {"total": 1, "found": 1}, "TOTAL_COST": {"total": 2, "found": 2},
                     "ADDRESS": {"total": 1, "found": 1}}


def test_choice_rule_prefers_coverage_then_speed():
    from src.pipeline.library_ocr import choose_engine

    cov = lambda found: {"SELLER": {"total": 31, "found": found}}
    assert choose_engine({"A": cov(30), "B": cov(29)}, {"A": 9000.0, "B": 1000.0}) == "A"
    assert choose_engine({"A": cov(28), "B": cov(29)}, {"A": 1000.0, "B": 9000.0}) == "B"
    assert choose_engine({"A": cov(29), "B": cov(29)}, {"A": 2000.0, "B": 1500.0}) == "B"


def test_entity_match_can_ignore_marks_on_both_sides():
    from src.pipeline.entity_eval import score_sample

    pairs = [{"label": "SELLER", "text": "CỬA HÀNG NĂM OÁNH"}, {"label": "TOTAL_COST", "text": "840,000"}]
    payload = {"seller": {"name": "CUA HANG NAM OANH"}, "invoice": {}, "totals": {"total_amount": {"value": 840000}}}
    assert score_sample(payload, pairs)["SELLER"].status != "match"
    assert score_sample(payload, pairs, strip_marks=True)["SELLER"].status == "match"
    assert score_sample(payload, pairs, strip_marks=True)["TOTAL_COST"].status == "match"
