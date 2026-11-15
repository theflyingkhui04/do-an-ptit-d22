"""[M5-01 OCR giữ dấu, 14/11] Image steps around PaddleOCR detection + VietOCR recognition.

Design: openspec change m5-01-canonical-flow §7. The 08/11 choice run (experiments/modal_ocr_lib.py) cut each
detected line as its axis-aligned bounding box, so a tilted line also carried parts of its neighbours and marks
above the letters could be cut. These helpers straighten each line (perspective strip with a margin), level the
whole page, and pick a configuration by the rule fixed before the run.

PIL/numpy only, so they run in the local tests; ``enhance`` needs OpenCV and runs on Modal.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
from PIL import Image

WHITE = (255, 255, 255)
LINE_PAD_RATIO = 0.15      # margin above and below each line, as a share of the line height
MIN_ASPECT = 2.0           # only boxes at least twice as wide as tall vote for the page angle
MIN_VOTES = 3
MIN_ANGLE_DEG = 0.3        # below this the page is left as it is
MAX_ANGLE_DEG = 15.0       # above this the estimate is not trusted (rotated photo, logo, stamp)
MAX_LATENCY_RATIO = 2.0    # a configuration slower than 2 x P0 (median) cannot win (Dương 14/11)


def order_quad(points: Any) -> np.ndarray:
    """Four corners as a float array ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) != 4:  # not a quadrilateral: use its bounding box
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    centre = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def axis_crop(img: Image.Image, quad: Any) -> Image.Image:
    """The 08/11 crop (P0): the axis-aligned box, same rounding as ``experiments/modal_ocr_lib.py:_crop``."""
    pts = np.asarray(quad, dtype=float).reshape(-1, 2)
    width, height = img.size
    x0, y0 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
    x1, y1 = min(width, round(pts[:, 0].max()) + 1), min(height, round(pts[:, 1].max()) + 1)
    return img.crop((x0, y0, x1, y1))


def line_strip(img: Image.Image, quad: Any, pad_ratio: float = LINE_PAD_RATIO) -> Image.Image:
    """The line as a horizontal strip: the quadrilateral mapped to a rectangle, widened above and below by
    ``pad_ratio`` x its height so that marks over or under the letters stay in the crop."""
    tl, tr, br, bl = order_quad(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    down = (bl - tl) + (br - tr)
    norm = np.linalg.norm(down)
    down = down / norm if norm > 0 else np.array([0.0, 1.0])
    pad = pad_ratio * height * down
    size = (max(1, round(width)), max(1, round(height * (1 + 2 * pad_ratio))))
    corners = (*(tl - pad), *(bl + pad), *(br + pad), *(tr - pad))  # PIL QUAD: UL, LL, LR, UR
    return img.transform(size, Image.Transform.QUAD, corners, resample=Image.Resampling.BICUBIC, fillcolor=WHITE)


def skew_angle(quads: list[Any], min_aspect: float = MIN_ASPECT, min_votes: int = MIN_VOTES) -> float:
    """Median slope (degrees, image coordinates: positive = the line goes down to the right) of the top edges of
    wide boxes; 0.0 when fewer than ``min_votes`` boxes qualify."""
    angles = []
    for quad in quads:
        tl, tr, _, bl = order_quad(quad)
        width, height = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
        if height > 0 and width >= min_aspect * height:
            angles.append(math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])))
    return float(statistics.median(angles)) if len(angles) >= min_votes else 0.0


def deskew(img: Image.Image, angle: float) -> Image.Image:
    """Rotate the page so that lines sloping by ``angle`` become level; tiny or implausible angles are ignored."""
    if not MIN_ANGLE_DEG <= abs(angle) <= MAX_ANGLE_DEG:
        return img
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=WHITE)


def enhance(rgb: np.ndarray) -> np.ndarray:
    """Local contrast (CLAHE on the lightness channel) then a mild unsharp mask. Needs OpenCV (Modal image)."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    contrasted = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(contrasted, (0, 0), 1.0)
    return cv2.addWeighted(contrasted, 1.6, blurred, -0.6, 0)


def choose_config(scores: dict[str, dict[str, float]], baseline: str = "P0",
                  max_latency_ratio: float = MAX_LATENCY_RATIO) -> dict[str, Any]:
    """Rule fixed before the run (design §7): most label lines found with their marks, then most found after
    stripping marks, then the lower median latency; slower than ``max_latency_ratio`` x the baseline cannot win;
    a full tie keeps the baseline."""
    limit = max_latency_ratio * scores[baseline]["median_latency_ms"]
    too_slow = [name for name, s in scores.items() if s["median_latency_ms"] > limit]
    eligible = [name for name in scores if name not in too_slow]
    chosen = max(eligible, key=lambda n: (scores[n]["exact"], scores[n]["stripped"],
                                          -scores[n]["median_latency_ms"], n == baseline))
    return {"chosen": chosen, "too_slow": too_slow, "latency_limit_ms": limit}
