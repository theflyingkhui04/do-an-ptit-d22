"""Production wiring of the FlexSDS web console backend (Render).

    uvicorn src.api.main:build_app --factory --host 0.0.0.0 --port $PORT

Required environment: DEMO_KEY, DEMO_FRONTEND_ORIGINS, DEMO_DATABASE_URL (demo_writer),
DEMO_READER_DATABASE_URL (demo_reader), MODAL_PROXY_TOKEN_ID, MODAL_PROXY_TOKEN_SECRET (library OCR server).
Optional: SUPABASE_URL + SUPABASE_SERVICE_KEY (images), DEMO_IMAGE_BUCKET, PTIT_MODAL_WORKSPACE,
PTIT_LIBRARY_OCR_URL. Model endpoints are not configured here: people paste them in the web.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping

import httpx
from fastapi import FastAPI

from src.api.app import Deps, create_app
from src.api.bench import BenchService
from src.api.endpoints import EndpointRegistry
from src.api.jobs import InvoicePipeline
from src.api.library_ocr_client import LibraryOcrClient
from src.api.questions import QuestionService
from src.api.reports import ReportService
from src.api.security import RedactingJsonFormatter, redact
from src.api.settings import Settings
from src.api.storage import SupabaseStorage
from src.api.store import PostgresStore
from src.pipeline.library_ocr import library_ocr_url


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingJsonFormatter(settings.secret_values()))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def library_ocr_reader(environ: Mapping[str, str], token: str, http: httpx.Client) -> LibraryOcrClient:
    """OCR of the canonical flow: the library server (engine A), never the vision-language model."""
    return LibraryOcrClient(library_ocr_url(environ), token, client=http)


def build_app(environ: Mapping[str, str] | None = None) -> FastAPI:
    environ = os.environ if environ is None else environ
    settings = Settings.from_env(environ)
    if not settings.writer_url or not settings.modal_token:
        raise RuntimeError("DEMO_DATABASE_URL and MODAL_PROXY_TOKEN_ID/SECRET are required")
    configure_logging(settings)
    secrets = settings.secret_values()

    def scrub(text: str) -> str:
        return redact(text, secrets)

    store = PostgresStore(settings.writer_url, settings.reader_url)
    storage = None
    if settings.supabase_url and settings.supabase_service_key:
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_service_key, settings.image_bucket)
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
    probe_http = httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=False)
    registry = EndpointRegistry(store, probe_http, max_endpoints=settings.max_endpoints, llm_client=http)

    def ocr_reader() -> LibraryOcrClient:
        return library_ocr_reader(environ, settings.modal_token, http)

    pipeline = InvoicePipeline(store, ocr_reader=ocr_reader, caller_for=registry.caller, storage=storage,
                               redact=scrub)
    questions = QuestionService(store, redact=scrub)
    reports = ReportService(store, questions)
    bench = BenchService(store, registry, pipeline=pipeline, reports=reports, redact=scrub)
    return create_app(settings, Deps(store=store, endpoints=registry, pipeline=pipeline, storage=storage,
                                     questions=questions, reports=reports, bench=bench))
