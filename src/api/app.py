"""FastAPI application factory of the FlexSDS web console backend (routes, demo key, limits, CORS).

Model endpoints are the ones people paste from `flexsds serve` (``/api/endpoints``); every node and every
measurement runs on endpoints chosen in the request. The backend never starts or stops GPUs.
"""

from __future__ import annotations

import io
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

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

from src.api.endpoints import EndpointError
from src.api.security import RateLimiter, client_ip, key_matches, redact
from src.api.settings import Settings
from src.api.store import Store

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JOB_ID_RE = re.compile(r"up_[0-9a-f]{12}")
ENDPOINT_ID = r"^ep_[0-9a-f]{8}$"
BENCH_ID_RE = re.compile(r"bn_[0-9a-f]{12}")
MAX_QUESTION_CHARS = 500

log = logging.getLogger("src.api")


@dataclass
class Deps:
    store: Store
    endpoints: Any              # EndpointRegistry
    pipeline: Any               # InvoicePipeline: run(job_id, data, ext, endpoints, baseline_id)
    storage: Any = None         # SupabaseStorage: signed_url(path)
    questions: Any = None       # QuestionService: answer(question, caller)
    reports: Any = None         # ReportService: run(caller)
    bench: Any = None           # BenchService: start(...), get(run_id), list()
    clock: Callable[[], float] = time.monotonic


class EndpointsIn(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=10)
    api_key: str | None = Field(default=None, max_length=500)
    serving_config: dict[str, Any] | None = None


class EndpointPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    is_baseline: bool | None = None


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    endpoint_id: str = Field(pattern=ENDPOINT_ID)


class ReportIn(BaseModel):
    endpoint_id: str = Field(pattern=ENDPOINT_ID)


class BenchIn(BaseModel):
    kind: Literal["speed", "chain"]
    endpoint_ids: list[str] = Field(min_length=1, max_length=6)
    workload: Literal["W1", "W2", "W3"] = "W1"
    loads: list[int] = Field(default_factory=lambda: [1, 8], min_length=1, max_length=6)
    requests: int = Field(default=8, ge=1, le=32)
    repeats: int = Field(default=1, ge=1, le=3)
    sample_id: str | None = Field(default=None, max_length=40)


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
    endpoint_limiter = RateLimiter(settings.endpoint_limit, settings.rate_window_s, deps.clock)
    bench_limiter = RateLimiter(settings.bench_limit, settings.rate_window_s, deps.clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for name, mark in (("jobs", deps.store.mark_unfinished_interrupted),
                           ("measurement runs", deps.store.mark_bench_interrupted)):
            try:
                log.info("marked %s unfinished %s as interrupted", mark(), name)
            except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
                log.error("could not mark unfinished %s: %s", name, redact(str(exc), secrets))
        yield

    app = FastAPI(title="FlexSDS console API", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.frontend_origins),
                       allow_methods=["GET", "POST", "PATCH", "DELETE"],
                       allow_headers=["X-Demo-Key", "Content-Type"], max_age=600)

    def require_key(request: Request) -> None:
        if not key_matches(request.headers.get("x-demo-key"), settings.demo_key):
            raise HTTPException(401, "Sai mật khẩu demo")

    def limit(limiter: RateLimiter, request: Request) -> None:
        retry_after = limiter.hit(client_ip(request))
        if retry_after is not None:
            raise HTTPException(429, "Quá nhiều yêu cầu, hãy thử lại sau",
                                headers={"Retry-After": str(retry_after)})

    def endpoint_or_404(endpoint_id: str) -> dict[str, Any]:
        try:
            return deps.endpoints.get(endpoint_id)
        except LookupError:
            raise HTTPException(404, "Không tìm thấy endpoint") from None

    def caller_or_404(endpoint_id: str) -> Any:
        endpoint_or_404(endpoint_id)
        return deps.endpoints.caller(endpoint_id)

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error: %s", redact(f"{type(exc).__name__}: {exc}", secrets))
        return JSONResponse(status_code=500, content={"detail": "Lỗi máy chủ"})

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            database = "ready" if deps.store.ping() else "error"
            endpoints = deps.endpoints.list() if database == "ready" else []
        except Exception:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            database, endpoints = "error", []
        return {"backend": "ready", "database": database,
                "endpoints": {"total": len(endpoints), "ok": sum(e["status"] == "ok" for e in endpoints),
                              "down": sum(e["status"] == "down" for e in endpoints)}}

    # ---------------------------------------------------------------- endpoints
    @app.get("/api/endpoints", dependencies=[Depends(require_key)])
    def list_endpoints() -> dict[str, Any]:
        return {"endpoints": deps.endpoints.list()}

    @app.post("/api/endpoints/detect", dependencies=[Depends(require_key)])
    def detect_endpoints(request: Request, body: EndpointsIn) -> dict[str, Any]:
        limit(endpoint_limiter, request)
        return {"results": deps.endpoints.detect(body.urls, body.api_key)}

    @app.post("/api/endpoints", dependencies=[Depends(require_key)])
    def add_endpoints(request: Request, body: EndpointsIn) -> dict[str, Any]:
        limit(endpoint_limiter, request)
        try:
            return deps.endpoints.add(body.urls, body.api_key, body.serving_config)
        except EndpointError as exc:
            raise HTTPException(422, exc.message) from exc

    @app.patch("/api/endpoints/{endpoint_id}", dependencies=[Depends(require_key)])
    def patch_endpoint(endpoint_id: str, body: EndpointPatch) -> dict[str, Any]:
        endpoint_or_404(endpoint_id)
        if body.name is not None:
            deps.endpoints.rename(endpoint_id, body.name)
        if body.is_baseline is not None:
            current = deps.endpoints.baseline()
            if body.is_baseline:
                deps.endpoints.set_baseline(endpoint_id)
            elif current and current["endpoint_id"] == endpoint_id:
                deps.endpoints.set_baseline(None)
        return deps.endpoints.get(endpoint_id)

    @app.delete("/api/endpoints/{endpoint_id}", dependencies=[Depends(require_key)])
    def delete_endpoint(endpoint_id: str) -> dict[str, Any]:
        if not deps.endpoints.delete(endpoint_id):
            raise HTTPException(404, "Không tìm thấy endpoint")
        return {"deleted": endpoint_id}

    @app.post("/api/endpoints/{endpoint_id}/check", dependencies=[Depends(require_key)])
    def check_endpoint(request: Request, endpoint_id: str) -> dict[str, Any]:
        limit(endpoint_limiter, request)
        endpoint_or_404(endpoint_id)
        return deps.endpoints.check(endpoint_id)

    # ---------------------------------------------------------------- Node 1
    @app.post("/api/invoices", status_code=202, dependencies=[Depends(require_key)])
    async def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...),  # noqa: B008  (FastAPI parameter declaration)
                     endpoint_ids: str = Form(...)) -> dict[str, Any]:
        limit(upload_limiter, request)
        ids = list(dict.fromkeys(i.strip() for i in endpoint_ids.split(",") if i.strip()))
        if not ids or len(ids) > settings.max_run_endpoints:
            raise HTTPException(422, f"Chọn từ 1 đến {settings.max_run_endpoints} endpoint")
        chosen = [endpoint_or_404(i) for i in ids]
        baseline = deps.endpoints.baseline()
        if baseline is not None and baseline["endpoint_id"] not in ids:
            chosen.append(baseline)
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Ảnh lớn hơn 10 MB")
        kind = image_kind(data)
        if kind is None:
            raise HTTPException(415, "Chỉ nhận ảnh JPEG hoặc PNG")
        job_id = f"up_{uuid.uuid4().hex[:12]}"
        deps.store.create_job(job_id)
        background.add_task(deps.pipeline.run, job_id, data, kind, chosen,
                            None if baseline is None else baseline["endpoint_id"])
        return {"job_id": job_id, "status": "queued", "endpoint_ids": [e["endpoint_id"] for e in chosen]}

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

    # ---------------------------------------------------------------- Node 2 and Node 3
    @app.post("/api/questions", dependencies=[Depends(require_key)])
    def ask(request: Request, body: QuestionIn) -> dict[str, Any]:
        limit(question_limiter, request)
        if deps.questions is None:
            raise HTTPException(503, "Chức năng hỏi dữ liệu chưa sẵn sàng")
        endpoint = endpoint_or_404(body.endpoint_id)
        result = deps.questions.answer(body.question.strip(), deps.endpoints.caller(body.endpoint_id))
        return {**result, "endpoint": {"endpoint_id": endpoint["endpoint_id"], "name": endpoint["name"]}}

    @app.post("/api/reports", dependencies=[Depends(require_key)])
    def report(request: Request, body: ReportIn) -> dict[str, Any]:
        limit(question_limiter, request)
        if deps.reports is None:
            raise HTTPException(503, "Chức năng báo cáo chưa sẵn sàng")
        endpoint = endpoint_or_404(body.endpoint_id)
        result = deps.reports.run(deps.endpoints.caller(body.endpoint_id))
        return {**result, "endpoint": {"endpoint_id": endpoint["endpoint_id"], "name": endpoint["name"]}}

    # ---------------------------------------------------------------- measurements
    @app.post("/api/bench", status_code=202, dependencies=[Depends(require_key)])
    def start_bench(request: Request, body: BenchIn) -> dict[str, Any]:
        limit(bench_limiter, request)
        if deps.bench is None:
            raise HTTPException(503, "Chức năng đo chưa sẵn sàng")
        for endpoint_id in body.endpoint_ids:
            endpoint_or_404(endpoint_id)
        if any(load < 1 or load > 32 for load in body.loads):
            raise HTTPException(422, "Số người dùng cùng lúc phải từ 1 đến 32")
        try:
            run_id = deps.bench.start(body.model_dump())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/bench", dependencies=[Depends(require_key)])
    def list_bench() -> dict[str, Any]:
        return {"runs": [] if deps.bench is None else deps.bench.list()}

    @app.get("/api/bench/{run_id}", dependencies=[Depends(require_key)])
    def get_bench(run_id: str) -> dict[str, Any]:
        row = deps.bench.get(run_id) if deps.bench is not None and BENCH_ID_RE.fullmatch(run_id) else None
        if row is None:
            raise HTTPException(404, "Không tìm thấy lần đo")
        return row

    return app
