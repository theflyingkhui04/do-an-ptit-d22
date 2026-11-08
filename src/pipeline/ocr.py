"""[M5-01] Receipt OCR stage: image -> plain multi-line text for Node 1.

The OCR model only transcribes; it never returns JSON. Node 1 keeps the job of
turning text into ``invoice_extraction.v1`` JSON.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

OCR_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
# Hugging Face commit resolved on 2026-11-08.
OCR_REVISION = "536a35794df8831aa814970ee8f89eff577e7718"
OCR_PROMPT_VERSION = "receipt_ocr_full.v1"
OCR_PROMPT = (
    "Hãy chép lại toàn bộ chữ có trong ảnh hoá đơn này theo đúng thứ tự từ trên xuống dưới, "
    "mỗi dòng chữ trên ảnh là một dòng. Giữ nguyên chính tả, dấu tiếng Việt và các con số như trên ảnh. "
    "Chỉ trả về phần chữ, không giải thích, không dùng Markdown, không dùng JSON."
)

# Images are scaled so that width*height <= MAX_PIXELS, enlarging at most MAX_UPSCALE
# times: low-resolution MC-OCR photos (768x1024) are read at x2.
MAX_PIXELS = 3_300_000
MAX_UPSCALE = 2.0
JPEG_QUALITY = 95
OCR_MAX_TOKENS = 1536
OCR_TEMPERATURE = 0.0
OCR_SEED = 42
OCR_MAX_MODEL_LEN = 8192

_FENCE = re.compile(r"^```[A-Za-z]*\s*$")


class OcrError(RuntimeError):
    """The image cannot be prepared or the OCR model returned no text."""


class ChatClient(Protocol):
    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PreparedImage:
    jpeg: bytes
    original_size: tuple[int, int]
    scaled_size: tuple[int, int]
    scale: float


@dataclass(frozen=True)
class OcrResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    image: PreparedImage


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ocr_prompt_sha256() -> str:
    return sha256_bytes(OCR_PROMPT.encode("utf-8"))


def ocr_scale(width: int, height: int, max_pixels: int = MAX_PIXELS, max_upscale: float = MAX_UPSCALE) -> float:
    if width <= 0 or height <= 0:
        raise OcrError(f"invalid image size {width}x{height}")
    return min(max_upscale, math.sqrt(max_pixels / (width * height)))


def prepare_image(data: bytes) -> PreparedImage:
    """Decode, scale with LANCZOS and re-encode as RGB JPEG."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OcrError(f"cannot read image: {exc}") from exc

    width, height = rgb.size
    scale = ocr_scale(width, height)
    # Floor keeps the scaled image inside the pixel budget.
    scaled = (max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
    if scaled != rgb.size:
        rgb = rgb.resize(scaled, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return PreparedImage(jpeg=buffer.getvalue(), original_size=(width, height), scaled_size=scaled, scale=scale)


def build_ocr_request(jpeg: bytes, model: str = OCR_MODEL) -> dict[str, Any]:
    image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "temperature": OCR_TEMPERATURE,
        "max_tokens": OCR_MAX_TOKENS,
        "seed": OCR_SEED,
    }


def clean_ocr_text(text: str | None) -> str:
    """Drop Markdown fences, strip every line and drop empty lines."""
    lines = []
    for line in (text or "").splitlines():
        cleaned = line.strip()
        if cleaned and not _FENCE.match(cleaned):
            lines.append(cleaned)
    return "\n".join(lines)


def _usage(response: dict[str, Any], name: str) -> int:
    value = (response.get("usage") or {}).get(name, 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def run_ocr(client: ChatClient, data: bytes, model: str = OCR_MODEL) -> OcrResult:
    """OCR one image. A truncated answer is returned; the caller decides what to do."""
    prepared = prepare_image(data)
    response = client.chat_completions(build_ocr_request(prepared.jpeg, model))
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OcrError(f"OCR response has no message content: {exc}") from exc
    text = clean_ocr_text(content)
    finish_reason = choice.get("finish_reason")
    if not text:
        raise OcrError(f"OCR returned empty text (finish_reason={finish_reason})")
    return OcrResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=_usage(response, "prompt_tokens"),
        completion_tokens=_usage(response, "completion_tokens"),
        image=prepared,
    )


def ocr_server_args(port: int = 8000, served_model_name: str = OCR_MODEL) -> list[str]:
    """``vllm serve`` arguments for the OCR model (model first, then flags)."""
    return [
        OCR_MODEL,
        "--revision", OCR_REVISION,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--max-model-len", str(OCR_MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0}),
        "--mm-processor-kwargs", json.dumps({"max_pixels": MAX_PIXELS}),
    ]


def _sample_fields(sample: dict[str, Any], data: bytes) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "img_id": sample.get("img_id"),
        "image_path": sample.get("image_path"),
        "image_sha256": sha256_bytes(data),
        "ocr_model": OCR_MODEL,
        "ocr_revision": OCR_REVISION,
        "ocr_prompt_version": OCR_PROMPT_VERSION,
        "ocr_prompt_sha256": ocr_prompt_sha256(),
    }


def provenance_row(sample: dict[str, Any], data: bytes, result: OcrResult, latency_ms: float) -> dict[str, Any]:
    """One ``ocr_texts.jsonl`` row: OCR text plus everything needed to reproduce it."""
    return {
        **_sample_fields(sample, data),
        "scaled_sha256": sha256_bytes(result.image.jpeg),
        "scale": round(result.image.scale, 6),
        "original_size": list(result.image.original_size),
        "scaled_size": list(result.image.scaled_size),
        "text": result.text,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": round(latency_ms, 1),
        "error": None,
    }


def error_row(sample: dict[str, Any], data: bytes, error: str) -> dict[str, Any]:
    return {**_sample_fields(sample, data), "text": "", "finish_reason": None, "error": error}


def summarize_ocr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finish = Counter(row["finish_reason"] for row in rows if not row.get("error"))
    return {
        "images": len(rows),
        "finish_reasons": dict(sorted(finish.items(), key=lambda item: str(item[0]))),
        "errors": sum(1 for row in rows if row.get("error")),
        "ocr_model": OCR_MODEL,
        "ocr_revision": OCR_REVISION,
        "ocr_prompt_version": OCR_PROMPT_VERSION,
        "ocr_prompt_sha256": ocr_prompt_sha256(),
    }
