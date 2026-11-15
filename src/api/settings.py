"""Backend settings, read from environment variables only (secrets never live in files).

Model endpoints are not configured here: people paste them in the web (``src/api/endpoints.py``).
The Modal proxy token is only used for the library OCR server (``ptit-ocr-lib``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

MIN_DEMO_KEY_LENGTH = 8


@dataclass(frozen=True)
class Settings:
    demo_key: str
    frontend_origins: tuple[str, ...] = ("http://localhost:3000",)
    writer_url: str | None = None
    reader_url: str | None = None
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    image_bucket: str = "invoice-images"
    modal_token: str | None = None
    upload_limit: int = 10
    question_limit: int = 30
    endpoint_limit: int = 30
    bench_limit: int = 6
    rate_window_s: float = 600.0
    max_upload_bytes: int = 10 * 1024 * 1024
    max_endpoints: int = 12
    max_run_endpoints: int = 6

    def __post_init__(self) -> None:
        if len(self.demo_key or "") < MIN_DEMO_KEY_LENGTH:
            raise ValueError(f"DEMO_KEY must be set (at least {MIN_DEMO_KEY_LENGTH} characters)")

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        token = None
        if environ.get("MODAL_PROXY_TOKEN_ID") and environ.get("MODAL_PROXY_TOKEN_SECRET"):
            token = f"{environ['MODAL_PROXY_TOKEN_ID']}.{environ['MODAL_PROXY_TOKEN_SECRET']}"
        origins = tuple(o.strip().rstrip("/") for o in environ.get("DEMO_FRONTEND_ORIGINS", "").split(",")
                        if o.strip())
        return cls(
            demo_key=environ.get("DEMO_KEY", ""),
            frontend_origins=origins or ("http://localhost:3000",),
            writer_url=environ.get("DEMO_DATABASE_URL") or None,
            reader_url=environ.get("DEMO_READER_DATABASE_URL") or None,
            supabase_url=environ.get("SUPABASE_URL") or None,
            supabase_service_key=environ.get("SUPABASE_SERVICE_KEY") or None,
            image_bucket=environ.get("DEMO_IMAGE_BUCKET") or "invoice-images",
            modal_token=token,
        )

    def secret_values(self) -> list[str]:
        values = [self.demo_key, self.writer_url, self.reader_url, self.supabase_service_key, self.modal_token]
        if self.modal_token:
            values.extend(self.modal_token.split("."))
        return [v for v in values if v and len(v) >= 6]
