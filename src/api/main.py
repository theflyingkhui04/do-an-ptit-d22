"""Production wiring of the web demo backend (Render).

    uvicorn src.api.main:build_app --factory --host 0.0.0.0 --port $PORT

Required environment: DEMO_KEY, DEMO_FRONTEND_ORIGINS, DEMO_DATABASE_URL (demo_writer),
DEMO_READER_DATABASE_URL (demo_reader), MODAL_PROXY_TOKEN_ID, MODAL_PROXY_TOKEN_SECRET.
Optional: SUPABASE_URL + SUPABASE_SERVICE_KEY (images), DEMO_IMAGE_BUCKET, PTIT_APP_NAME
(default ptit-vllm-demo here), PTIT_MODAL_WORKSPACE, PTIT_LIBRARY_OCR_URL (library OCR server,
Modal app ptit-ocr-lib by default), DEMO_LANE_<STEP>, DEMO_GPU_LABEL.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping

import httpx
from fastapi import FastAPI

from src.api.app import Deps, create_app
from src.api.jobs import InvoicePipeline
from src.api.library_ocr_client import LibraryOcrClient
from src.api.llm import LaneCaller
from src.api.security import RedactingJsonFormatter, redact
from src.api.settings import Settings
from src.api.storage import SupabaseStorage
from src.api.store import PostgresStore
from src.api.warmup import WarmupManager
from src.client.modal_client import LANE_MODELS, lane_base_url
from src.pipeline.library_ocr import library_ocr_url

DEMO_APP_NAME = "ptit-vllm-demo"
PING = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0}


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingJsonFormatter(settings.secret_values()))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def library_ocr_reader(environ: Mapping[str, str], token: str, http: httpx.Client) -> LibraryOcrClient:
    """OCR of the canonical flow: the library server (engine A), never the vision-language model."""
    return LibraryOcrClient(library_ocr_url(environ), token, client=http)


def warm_lanes(settings: Settings) -> list[str]:
    """Lanes kept warm: vanilla (comparison) + the lanes of the frequent steps; the report lane on demand."""
    steps = [settings.lane_map[s] for s in ("node1", "node2", "answer")]
    return list(dict.fromkeys(["vanilla", *steps]))


def build_app(environ: Mapping[str, str] | None = None) -> FastAPI:
    environ = os.environ if environ is None else environ
    settings = Settings.from_env(environ)
    if not settings.writer_url or not settings.modal_token:
        raise RuntimeError("DEMO_DATABASE_URL and MODAL_PROXY_TOKEN_ID/SECRET are required")
    os.environ.setdefault("PTIT_APP_NAME", DEMO_APP_NAME)  # demo lanes, never the benchmark app
    configure_logging(settings)
    secrets = settings.secret_values()

    def scrub(text: str) -> str:
        return redact(text, secrets)

    store = PostgresStore(settings.writer_url, settings.reader_url)
    storage = None
    if settings.supabase_url and settings.supabase_service_key:
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_service_key, settings.image_bucket)
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
    token = settings.modal_token

    def ocr_reader() -> LibraryOcrClient:
        return library_ocr_reader(environ, token, http)

    def lane_caller(lane: str) -> LaneCaller:
        return LaneCaller(lane, lane_base_url(lane), token, LANE_MODELS[lane], client=http)

    probes = {"database": store.ping, "ocr": lambda: ocr_reader().health()}
    for lane in warm_lanes(settings):
        probes[lane] = lambda lane=lane: lane_caller(lane).chat_completions(PING)
    warmup = WarmupManager(probes, redact=scrub)
    pipeline = InvoicePipeline(store, ocr_reader=ocr_reader, lane_caller=lane_caller, lane_map=settings.lane_map,
                               storage=storage, gpu_label=settings.gpu_label, on_service=warmup.mark,
                               redact=scrub)
    questions = None
    try:
        from src.api.questions import QuestionService

        questions = QuestionService(store, lane_caller=lane_caller, lane_map=settings.lane_map,
                                    on_service=warmup.mark, redact=scrub)
    except ImportError:
        pass
    return create_app(settings, Deps(store=store, pipeline=pipeline, warmup=warmup, storage=storage,
                                     questions=questions))
