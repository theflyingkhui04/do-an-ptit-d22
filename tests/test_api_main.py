"""[M5-01 web demo] Production wiring builds without network access and keeps demo lanes separate."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main
from src.api.settings import Settings

ENV = {"DEMO_KEY": "demo-key-123456", "DEMO_DATABASE_URL": "postgresql://demo_writer:pw@127.0.0.1:1/x",
       "MODAL_PROXY_TOKEN_ID": "id-abc", "MODAL_PROXY_TOKEN_SECRET": "secret-abc",
       "DEMO_FRONTEND_ORIGINS": "https://demo.vercel.app"}


def test_build_app_without_network(monkeypatch):
    monkeypatch.setenv("PTIT_APP_NAME", "ptit-vllm-demo")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)  # a stale conda path breaks httpx locally
    monkeypatch.setattr(main, "configure_logging", lambda settings: None)
    app = main.build_app(ENV)
    with TestClient(app) as client:
        body = client.get("/api/status").json()
    assert body["backend"] == "ready" and body["database"] == "error"
    assert set(body["services"]) == {"database", "ocr", "vanilla", "adaptive", "static"}


def test_missing_database_or_token_is_refused():
    import pytest

    with pytest.raises(RuntimeError, match="DEMO_DATABASE_URL"):
        main.build_app({"DEMO_KEY": "demo-key-123456"})


def test_warm_lanes_follow_the_lane_map():
    settings = Settings(demo_key="demo-key-123456")
    assert main.warm_lanes(settings) == ["vanilla", "adaptive", "static"]


def test_ocr_is_the_library_server_never_the_vision_model():
    import httpx

    from src.api.library_ocr_client import LibraryOcrClient

    http = httpx.Client()
    reader = main.library_ocr_reader({}, "tok", http)
    assert isinstance(reader, LibraryOcrClient)
    assert reader.base_url == "https://nguyenduonghb940--ptit-ocr-lib-ocr.us-east.modal.direct"
    override = main.library_ocr_reader({"PTIT_LIBRARY_OCR_URL": "https://ocr.test/"}, "tok", http)
    assert override.base_url == "https://ocr.test"
    from pathlib import Path

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "ocr_endpoint_url" not in source and "OCR_MODEL" not in source
