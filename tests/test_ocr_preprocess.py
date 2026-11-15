"""[M5-01 OCR giữ dấu, 14/11] Image steps around PaddleOCR detection + VietOCR recognition (design §7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.pipeline import ocr_preprocess as pre


def rotated_quad(cx: float, cy: float, w: float, h: float, deg: float) -> list[list[float]]:
    """Corners of a w x h box centred on (cx, cy), rotated by deg (image coordinates, y down)."""
    a = math.radians(deg)
    ux, uy = math.cos(a), math.sin(a)       # along the text line
    vx, vy = -math.sin(a), math.cos(a)      # towards the bottom of the line
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # TL, TR, BR, BL
        pts.append([cx + sx * w / 2 * ux + sy * h / 2 * vx, cy + sx * w / 2 * uy + sy * h / 2 * vy])
    return pts


def test_order_quad_returns_top_left_top_right_bottom_right_bottom_left():
    tl, tr, br, bl = [10, 20], [110, 25], [108, 45], [8, 40]
    for shuffled in ([br, tl, bl, tr], [bl, br, tr, tl], [tr, bl, tl, br]):
        assert pre.order_quad(shuffled).tolist() == [tl, tr, br, bl]


def test_axis_crop_matches_the_choice_run_of_08_11():
    img = Image.new("RGB", (200, 100), "white")
    crop = pre.axis_crop(img, [[10.4, 20.6], [110.2, 25.0], [108.0, 45.7], [8.9, 40.0]])
    # x0 = int(8.9), y0 = int(20.6), x1 = round(110.2) + 1, y1 = round(45.7) + 1  (experiments/modal_ocr_lib.py _crop)
    assert crop.size == (111 - 8, 47 - 20)


def test_line_strip_straightens_a_tilted_line_and_pads_above_and_below():
    img = Image.new("RGB", (400, 200), "white")
    quad = rotated_quad(200, 100, 200, 30, 8)
    ImageDraw.Draw(img).polygon([tuple(p) for p in quad], fill="black")
    strip = pre.line_strip(img, quad, pad_ratio=0.15)
    assert strip.size == (200, 39)  # 30 * (1 + 2 * 0.15)
    gray = np.asarray(strip.convert("L"))
    assert gray[:3].mean() > 200 and gray[-3:].mean() > 200   # padding rows are background, not cut text
    assert gray[10:29, 5:-5].mean() < 40                      # the straightened line fills the middle band


def test_skew_angle_is_the_median_slope_of_wide_boxes():
    quads = [rotated_quad(200, 40 + 30 * i, 220, 20, 4.0 + 0.2 * (i % 3)) for i in range(5)]
    quads.append(rotated_quad(50, 300, 20, 20, 40))  # square box (a logo) is ignored
    assert pre.skew_angle(quads) == pytest.approx(4.2, abs=0.01)
    assert pre.skew_angle(quads[:2]) == 0.0           # fewer than 3 wide boxes: no estimate


def test_deskew_levels_a_sloping_line():
    img = Image.new("RGB", (600, 400), "white")
    ImageDraw.Draw(img).polygon([tuple(p) for p in rotated_quad(300, 200, 400, 12, 6)], fill="black")
    level = np.asarray(pre.deskew(img, 6.0).convert("L"))
    rows = np.where((level < 100).sum(axis=1) > 50)[0]
    assert rows.max() - rows.min() <= 16              # was ~54 px tall before levelling


def test_deskew_leaves_tiny_or_implausible_angles_alone():
    img = Image.new("RGB", (60, 40), "white")
    assert pre.deskew(img, 0.1) is img
    assert pre.deskew(img, 30.0) is img


def test_choose_config_prefers_more_exact_lines_then_stripped_then_speed():
    scores = {
        "P0": {"exact": 100, "stripped": 120, "median_latency_ms": 10_000},
        "P1": {"exact": 110, "stripped": 121, "median_latency_ms": 11_000},
        "P2": {"exact": 110, "stripped": 124, "median_latency_ms": 19_000},
        "P3a": {"exact": 130, "stripped": 130, "median_latency_ms": 21_000},  # best but > 2 x P0
    }
    out = pre.choose_config(scores)
    assert out == {"chosen": "P2", "too_slow": ["P3a"], "latency_limit_ms": 20_000}


def test_choose_config_keeps_the_baseline_on_a_full_tie():
    scores = {"P0": {"exact": 100, "stripped": 120, "median_latency_ms": 10_000},
              "P1": {"exact": 100, "stripped": 120, "median_latency_ms": 10_000}}
    assert pre.choose_config(scores)["chosen"] == "P0"
