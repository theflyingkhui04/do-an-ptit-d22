"""In-memory fakes for the web demo API tests (no network, no database)."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.api.app import Deps, create_app
from src.api.jobs import InvoicePipeline
from src.api.llm import CallMetrics, LLMError
from src.api.settings import Settings
from src.api.warmup import WarmupManager

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json")
                     .read_text(encoding="utf-8"))
OCR_TEXT = "MINIMART ANAN\nChợ Sủi Phú Thị Gia Lâm\nNgày: 10/08/2020 - 17:12\nTổng tiền\n42,000"
DEMO_KEY = "demo-key-123456"
SECRET_URL = "postgresql://demo_writer:pw-very-secret@db.example:5432/postgres"


def jpeg_bytes(size: tuple[int, int] = (64, 96)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="JPEG")
    return buf.getvalue()


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.invoices: dict[str, dict] = {}
        self.saved: list[tuple[dict, Any]] = []
        self.ping_ok = True

    def ping(self) -> bool:
        return self.ping_ok

    def create_job(self, job_id: str) -> None:
        self.jobs[job_id] = {"job_id": job_id, "status": "queued", "error": None, "result": {}, "invoice_id": None}

    def update_job(self, job_id, status, *, result=None, error=None, invoice_id=None) -> None:
        job = self.jobs[job_id]
        job["status"] = status
        job["error"] = error
        if result is not None:
            job["result"] = json.loads(json.dumps(result, default=str))
        if invoice_id is not None:
            job["invoice_id"] = invoice_id

    def get_job(self, job_id):
        return copy.deepcopy(self.jobs.get(job_id))

    def mark_unfinished_interrupted(self) -> int:
        count = 0
        for job in self.jobs.values():
            if job["status"] in ("queued", "ocr", "node1", "saving"):
                job.update(status="interrupted", error="backend restarted")
                count += 1
        return count

    def save_invoice(self, payload, run, *, storage_path, latency_ms) -> str:
        invoice_id = f"{run.run_id}:{payload['source']['sample_id']}"
        self.saved.append((payload, run))
        self.invoices[invoice_id] = {"invoice_id": invoice_id, "payload": payload, "source": "upload",
                                     "storage_path": storage_path, "lane": run.lane}
        return invoice_id

    def list_invoices(self, limit=200):
        return [{"invoice_id": k, "source": v["source"]} for k, v in self.invoices.items()][:limit]

    def get_invoice(self, invoice_id):
        return copy.deepcopy(self.invoices.get(invoice_id))

    def categorical_values(self):
        return {"extraction_status": ["partial"]}

    def run_readonly(self, sql):
        raise NotImplementedError


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload(self, path, data, content_type) -> None:
        self.uploads[path] = data

    def signed_url(self, path, expires_in=300) -> str:
        return f"https://storage.test/{path}?token=t"


def node1_answer(request: dict, mutate: Callable[[dict], None] | None = None) -> str:
    """Example payload bound to the request's own SOURCE_METADATA and OCR_TEXT."""
    user = request["messages"][1]["content"]
    metadata = json.loads(user.split("SOURCE_METADATA:\n", 1)[1].split("\n", 1)[0])
    raw_text = user.split("OCR_TEXT:\n", 1)[1].rstrip("\n")
    payload = copy.deepcopy(EXAMPLE)
    payload["source"] = {**metadata, "raw_text": raw_text}
    if mutate:
        mutate(payload)
    return json.dumps(payload, ensure_ascii=False)


class FakeCaller:
    """Stands in for LaneCaller: same ``model``/``last_metrics``/``chat_completions`` surface."""

    def __init__(self, lane: str, *, content: Callable[[dict], str] | str, finish_reason: str = "stop",
                 latency_ms: float = 1000.0, tpot_ms: float = 20.0, error: str | None = None) -> None:
        self.lane = lane
        self.model = f"model-{lane}"
        self._content = content
        self._finish = finish_reason
        self._latency = latency_ms
        self._tpot = tpot_ms
        self._error = error
        self.calls: list[dict] = []
        self.last_metrics: CallMetrics | None = None

    def chat_completions(self, payload: dict) -> dict:
        self.calls.append(payload)
        if self._error:
            raise LLMError(f"{self.lane}: {self._error}")
        text = self._content(payload) if callable(self._content) else self._content
        self.last_metrics = CallMetrics(self.lane, self.model, self._latency, 100.0, self._tpot, 10, 50,
                                        50 / (self._latency / 1000), self._finish)
        return {"choices": [{"message": {"content": text}, "finish_reason": self._finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 50}}


def lane_callers(**overrides: FakeCaller) -> Callable[[str], FakeCaller]:
    def make(lane: str) -> FakeCaller:
        if lane in overrides:
            return overrides[lane]
        return FakeCaller(lane, content=node1_answer, tpot_ms=20.0 if lane == "vanilla" else 14.0)
    return make


def build(*, store: FakeStore | None = None, ocr: FakeCaller | None = None,
          lanes: Callable[[str], FakeCaller] | None = None, storage: FakeStorage | None = None,
          questions: Any = None, clock: Callable[[], float] | None = None,
          settings: Settings | None = None) -> tuple[Any, FakeStore, FakeStorage]:
    store = store or FakeStore()
    storage = storage or FakeStorage()
    settings = settings or Settings(demo_key=DEMO_KEY, frontend_origins=("https://demo.vercel.app",),
                                    writer_url=SECRET_URL)
    ocr_caller = ocr or FakeCaller("ocr", content=OCR_TEXT)
    pipeline = InvoicePipeline(store, ocr_caller=lambda: ocr_caller, lane_caller=lanes or lane_callers(),
                               lane_map=settings.lane_map, storage=storage, gpu_label="A10G")
    warmup = WarmupManager({"database": store.ping}, background=False)
    deps = Deps(store=store, pipeline=pipeline, warmup=warmup, storage=storage, questions=questions,
                clock=clock or (lambda: 0.0))
    return create_app(settings, deps), store, storage


HEADERS = {"X-Demo-Key": DEMO_KEY}
