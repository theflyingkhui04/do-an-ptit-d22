"""Tests for the receipt OCR stage ([M5-01] receipt-ocr spec)."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from src.pipeline import ocr


def jpeg_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeClient:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.payloads: list[dict] = []

    def chat_completions(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": self.finish_reason}],
            "usage": {"prompt_tokens": 4100, "completion_tokens": 300},
        }


def test_small_mc_ocr_image_is_upscaled_twice():
    assert ocr.ocr_scale(768, 1024) == 2.0
    prepared = ocr.prepare_image(jpeg_bytes(768, 1024))
    assert prepared.scale == 2.0
    assert prepared.original_size == (768, 1024)
    assert prepared.scaled_size == (1536, 2048)
    assert Image.open(io.BytesIO(prepared.jpeg)).size == (1536, 2048)


def test_large_photo_is_downscaled_within_pixel_budget():
    prepared = ocr.prepare_image(jpeg_bytes(3000, 4000))
    width, height = prepared.scaled_size
    assert prepared.scale < 1.0
    assert width * height <= ocr.MAX_PIXELS
    assert Image.open(io.BytesIO(prepared.jpeg)).size == (width, height)


def test_non_rgb_image_is_converted_to_rgb_jpeg():
    buffer = io.BytesIO()
    Image.new("RGBA", (100, 100), (255, 0, 0, 128)).save(buffer, format="PNG")
    prepared = ocr.prepare_image(buffer.getvalue())
    image = Image.open(io.BytesIO(prepared.jpeg))
    assert image.format == "JPEG"
    assert image.mode == "RGB"


def test_unreadable_image_raises_ocr_error():
    with pytest.raises(ocr.OcrError):
        ocr.prepare_image(b"not an image")


def test_request_has_one_image_and_the_prompt_and_no_json_mode():
    request = ocr.build_ocr_request(b"\xff\xd8jpeg")
    assert request["model"] == ocr.OCR_MODEL
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 1536
    assert request["seed"] == 42
    assert "response_format" not in request
    (message,) = request["messages"]
    image_part, text_part = message["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()
    assert text_part == {"type": "text", "text": ocr.OCR_PROMPT}
    assert "JSON" in ocr.OCR_PROMPT  # the prompt forbids JSON explicitly


def test_clean_text_drops_code_fences_and_blank_lines():
    assert ocr.clean_ocr_text("```\nCOOP MART\n\n  Tổng: 42.000  \n```") == "COOP MART\nTổng: 42.000"
    assert ocr.clean_ocr_text("```text\nA\n```") == "A"
    assert ocr.clean_ocr_text(None) == ""


def test_run_ocr_returns_clean_text_and_metadata():
    client = FakeClient("```\nVINMART\nTổng tiền: 50.000\n```")
    result = ocr.run_ocr(client, jpeg_bytes(768, 1024))
    assert result.text == "VINMART\nTổng tiền: 50.000"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 4100
    assert result.completion_tokens == 300
    assert result.image.scale == 2.0
    assert len(client.payloads) == 1


def test_run_ocr_empty_text_raises():
    with pytest.raises(ocr.OcrError, match="empty"):
        ocr.run_ocr(FakeClient("```\n\n```"), jpeg_bytes(64, 64))


def test_run_ocr_keeps_truncated_finish_reason_for_caller():
    result = ocr.run_ocr(FakeClient("A\nB", finish_reason="length"), jpeg_bytes(64, 64))
    assert result.finish_reason == "length"


def test_server_args_pin_model_revision_and_limits():
    args = ocr.ocr_server_args(port=8000)
    assert args[0] == ocr.OCR_MODEL
    flags = {flag: value for flag, value in zip(args, args[1:]) if flag.startswith("--")}
    assert flags["--revision"] == ocr.OCR_REVISION
    assert flags["--max-model-len"] == "8192"
    assert json.loads(flags["--limit-mm-per-prompt"]) == {"image": 1, "video": 0}
    assert json.loads(flags["--mm-processor-kwargs"]) == {"max_pixels": ocr.MAX_PIXELS}
    assert flags["--port"] == "8000"


def test_provenance_row_and_summary():
    data = jpeg_bytes(768, 1024)
    result = ocr.run_ocr(FakeClient("VINMART"), data)
    sample = {"sample_id": "m0_04_001", "img_id": "a.jpg", "image_path": "data/a.jpg"}
    row = ocr.provenance_row(sample, data, result, latency_ms=1234.5)
    assert row["sample_id"] == "m0_04_001"
    assert row["text"] == "VINMART"
    assert row["ocr_model"] == ocr.OCR_MODEL
    assert row["ocr_revision"] == ocr.OCR_REVISION
    assert row["ocr_prompt_version"] == "receipt_ocr_full.v1"
    assert row["ocr_prompt_sha256"] == ocr.ocr_prompt_sha256()
    assert row["scale"] == 2.0 and row["scaled_size"] == [1536, 2048]
    assert len(row["image_sha256"]) == 64 and len(row["scaled_sha256"]) == 64
    assert row["latency_ms"] == 1234.5

    error_row = ocr.error_row(sample, data, "OcrError: empty")
    summary = ocr.summarize_ocr([row, error_row])
    assert summary["images"] == 2
    assert summary["finish_reasons"] == {"stop": 1}
    assert summary["errors"] == 1


# [M5-01 web demo] long-running OCR endpoint (experiments/modal_ocr_serve.py)
def test_ocr_endpoint_url_default_workspace_override_and_url_override():
    assert ocr.ocr_endpoint_url({}) == "https://nguyenduonghb940--ptit-ocr-serve-ocr.us-east.modal.direct/v1"
    assert ocr.ocr_endpoint_url({"PTIT_MODAL_WORKSPACE": "otherws"}).startswith("https://otherws--ptit-ocr-serve-ocr.")
    assert ocr.ocr_endpoint_url({"PTIT_OCR_URL": "https://example.test/v1/"}) == "https://example.test/v1"


def test_ocr_serve_app_reuses_the_batch_server_contract():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "experiments/modal_ocr_serve.py").read_text(encoding="utf-8")
    assert "ocr.ocr_server_args(port=PORT)" in source
    assert f'APP_NAME = "{ocr.OCR_SERVE_APP}"' in source
    assert f'name="{ocr.OCR_SERVE_NAME}"' in source
    assert "unauthenticated=True" not in source  # Modal proxy authentication stays on
    assert '"ptit-ocr-cache"' in source
