"""FastAPI application factory of the web demo backend (routes, demo key, limits, CORS)."""

from __future__ import annotations

import io
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.api.security import RateLimiter, client_ip, key_matches, redact
from src.api.settings import LANES, Settings
from src.api.store import Store
from src.api.warmup import WarmupManager

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JOB_ID_RE = re.compile(r"up_[0-9a-f]{12}")
MAX_QUESTION_CHARS = 500

log = logging.getLogger("src.api")


@dataclass
class Deps:
    store: Store
    pipeline: Any          # InvoicePipeline: run(job_id, data, ext, compare_lanes)
    warmup: WarmupManager
    storage: Any = None    # SupabaseStorage: signed_url(path)
    questions: Any = None  # QuestionService: answer(question) -> dict
    clock: Callable[[], float] = time.monotonic


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def image_kind(data: bytes) -> str | None:
    """``jpg`` or ``png`` when the bytes are a decodable image of that type, else ``None``."""
    if data.startswith(JPEG_MAGIC):
        kind = "jpg"
    elif data.startswith(PNG_MAGIC):
        kind = "png"
    else:
        return None
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:  # noqa: BLE001  (service boundary: reported, never raised to the user)
        return None
    return kind


def create_app(settings: Settings, deps: Deps) -> FastAPI:
    secrets = settings.secret_values()
    upload_limiter = RateLimiter(settings.upload_limit, settings.rate_window_s, deps.clock)
    question_limiter = RateLimiter(settings.question_limit, settings.rate_window_s, deps.clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            count = deps.store.mark_unfinished_interrupted()
            log.info("marked %s unfinished jobs as interrupted", count)
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            log.error("could not mark unfinished jobs: %s", redact(str(exc), secrets))
        yield

    app = FastAPI(title="PTIT invoice demo API", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.frontend_origins),
                       allow_methods=["GET", "POST"], allow_headers=["X-Demo-Key", "Content-Type"], max_age=600)

    def require_key(request: Request) -> None:
        if not key_matches(request.headers.get("x-demo-key"), settings.demo_key):
            raise HTTPException(401, "Sai mật khẩu demo")

    def limit(limiter: RateLimiter, request: Request) -> None:
        retry_after = limiter.hit(client_ip(request))
        if retry_after is not None:
            raise HTTPException(429, "Quá nhiều yêu cầu, hãy thử lại sau",
                                headers={"Retry-After": str(retry_after)})

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error: %s", redact(f"{type(exc).__name__}: {exc}", secrets))
        return JSONResponse(status_code=500, content={"detail": "Lỗi máy chủ"})

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            database = "ready" if deps.store.ping() else "error"
        except Exception:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            database = "error"
        deps.warmup.mark("database", database == "ready", None if database == "ready" else "ping failed")
        return {"backend": "ready", "database": database, "services": deps.warmup.snapshot(),
                "gpu": settings.gpu_label, "lane_map": settings.lane_map}

    @app.post("/api/warmup", dependencies=[Depends(require_key)])
    def warmup() -> dict[str, Any]:
        return {"services": deps.warmup.request(), "max_seconds": deps.warmup.max_s}

    @app.post("/api/invoices", status_code=202, dependencies=[Depends(require_key)])
    async def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...),  # noqa: B008  (FastAPI parameter declaration)
                     compare: str = Form("vanilla")) -> dict[str, str]:
        limit(upload_limiter, request)
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Ảnh lớn hơn 10 MB")
        kind = image_kind(data)
        if kind is None:
            raise HTTPException(415, "Chỉ nhận ảnh JPEG hoặc PNG")
        if compare not in ("vanilla", "all"):
            raise HTTPException(422, "compare phải là 'vanilla' hoặc 'all'")
        job_id = f"up_{uuid.uuid4().hex[:12]}"
        deps.store.create_job(job_id)
        background.add_task(deps.pipeline.run, job_id, data, kind, LANES if compare == "all" else ("vanilla",))
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_key)])
    def job(job_id: str) -> dict[str, Any]:
        row = deps.store.get_job(job_id) if JOB_ID_RE.fullmatch(job_id) else None
        if row is None:
            raise HTTPException(404, "Không tìm thấy job")
        return row

    @app.get("/api/invoices", dependencies=[Depends(require_key)])
    def invoices(limit: int = 200) -> dict[str, Any]:
        return {"invoices": deps.store.list_invoices(max(1, min(limit, 200)))}

    @app.get("/api/invoices/{invoice_id}", dependencies=[Depends(require_key)])
    def invoice(invoice_id: str) -> dict[str, Any]:
        row = deps.store.get_invoice(invoice_id)
        if row is None:
            raise HTTPException(404, "Không tìm thấy hoá đơn")
        image_url = None
        if deps.storage is not None and row.get("storage_path"):
            try:
                image_url = deps.storage.signed_url(row["storage_path"])
            except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
                log.warning("signed url failed: %s", redact(str(exc), secrets))
        return {**row, "image_url": image_url}

    @app.post("/api/questions", dependencies=[Depends(require_key)])
    def ask(request: Request, body: QuestionIn) -> dict[str, Any]:
        limit(question_limiter, request)
        if deps.questions is None:
            raise HTTPException(503, "Chức năng hỏi dữ liệu chưa sẵn sàng")
        return deps.questions.answer(body.question.strip())

    return app
