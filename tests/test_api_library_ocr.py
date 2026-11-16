"""[M5-01 canonical flow] Backend client of the library OCR endpoint (fake HTTP transport)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from src.api.library_ocr_client import LibraryOcrClient
from src.api.llm import LLMError
from src.pipeline.library_ocr import library_ocr_url
from src.pipeline.ocr import OcrError

TOKEN = "tok-id.tok-secret"


def client(handler, **kwargs) -> LibraryOcrClient:
    return LibraryOcrClient("https://ocr.test/", TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)),
                            clock=lambda: 0.0, sleep=lambda s: None, **kwargs)


def test_read_sends_the_image_and_returns_text_and_metadata():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["image"] = base64.b64decode(json.loads(request.content)["image_b64"])
        return httpx.Response(200, json={"text": "MINIMART ANAN\nTổng tiền 42,000", "boxes": 2, "latency_ms": 900.0,
                                         "engine": "A", "versions": {"vietocr": "0.3.13"}})

    result = client(handler).read(b"\xff\xd8jpeg")
    assert seen == {"url": "https://ocr.test/v1/ocr", "auth": f"Bearer {TOKEN}", "image": b"\xff\xd8jpeg"}
    assert result.text == "MINIMART ANAN\nTổng tiền 42,000" and result.boxes == 2 and result.engine == "A"


def test_no_text_detected_is_an_ocr_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "no text detected"})

    with pytest.raises(OcrError, match="no text detected"):
        client(handler).read(b"x")


def test_cold_start_is_waited_for_and_errors_hide_the_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("reset")
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"text": "a", "boxes": 1, "latency_ms": 1.0, "engine": "A", "versions": {}})

    assert client(handler).read(b"x").text == "a" and len(calls) == 3

    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad {TOKEN}")

    with pytest.raises(LLMError) as info:
        client(unauthorized).read(b"x")
    assert TOKEN not in str(info.value) and "401" in str(info.value)


def test_health_and_url():
    assert client(lambda r: httpx.Response(200, json={"status": "ok"})).health() is True
    assert library_ocr_url({}) == "https://nguyenduonghb940--ptit-ocr-lib-ocr.us-east.modal.direct"
    assert library_ocr_url({"PTIT_LIBRARY_OCR_URL": "https://x.test/"}) == "https://x.test"


def test_server_contract_matches_the_frozen_choice():
    from pathlib import Path

    from src.pipeline.library_ocr import (
        CHOSEN_ENGINE,
        LIBRARY_OCR_APP,
        LIBRARY_OCR_SERVER,
    )

    source = (Path(__file__).resolve().parents[1] / "experiments/modal_ocr_lib.py").read_text(encoding="utf-8")
    choice = json.loads((Path(__file__).resolve().parents[1] / "experiments/results/m5_01_ocr_lib_choice.json")
                        .read_text(encoding="utf-8"))
    assert choice["chosen_engine"] == CHOSEN_ENGINE == "A"
    assert f'modal.App("{LIBRARY_OCR_APP}"' in source and f'@app.server(name="{LIBRARY_OCR_SERVER}"' in source
    assert "unauthenticated=True" not in source
    assert "strip_vietnamese_marks(boxes_to_text(boxes))" in source  # the endpoint returns unaccented text
    assert f'FROZEN_WEIGHTS_SHA256 = "{choice["versions"]["weights_sha256"]}"' in source
    assert f'DET_MODEL = "{choice["versions"]["detector"]}"' in source
    for name in ("paddlepaddle", "paddleocr", "vietocr"):
        assert f'_VERSION = "{choice["versions"][name]}"' in source


def test_progress_reports_each_wait_for_a_cold_ocr_server():
    answers = iter([httpx.Response(503), httpx.Response(503),
                    httpx.Response(200, json={"text": "A", "boxes": 1, "latency_ms": 1.0, "engine": "A"})])
    events: list[dict] = []
    reader = client(lambda request: next(answers))
    reader.progress = events.append
    reader.read(b"x")
    assert [(e["kind"], e["tries"], e["reason"]) for e in events] == [("waiting", 1, "HTTP 503"), ("waiting", 2, "HTTP 503")]
