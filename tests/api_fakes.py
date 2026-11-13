"""In-memory fakes for the web demo API tests (no network, no database)."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any, Callable

import httpx
from PIL import Image

from src.api.app import Deps, create_app
from src.api.endpoints import EndpointRegistry
from src.api.jobs import InvoicePipeline
from src.api.library_ocr_client import LibraryOcrResult
from src.api.llm import CallMetrics, LLMError
from src.api.settings import Settings
from src.pipeline.text_normalize import strip_vietnamese_marks

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json")
                     .read_text(encoding="utf-8"))
OCR_TEXT = "MINIMART ANAN\nChợ Sủi Phú Thị Gia Lâm\nNgày: 10/08/2020 - 17:12\nTổng tiền\n42,000"
OCR_TEXT_PLAIN = strip_vietnamese_marks(OCR_TEXT)  # what Node 1 sees: marks removed first
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
        self.endpoints: dict[str, dict] = {}
        self.bench: dict[str, dict] = {}
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

    # endpoints pasted from flexsds serve (dict keeps insertion order like ORDER BY created_at)
    def list_endpoints(self) -> list[dict]:
        rows = []
        for row in self.endpoints.values():
            public = {k: v for k, v in row.items() if k != "api_key"}
            rows.append(copy.deepcopy({**public, "has_key": bool(row.get("api_key"))}))
        return rows

    def get_endpoint(self, endpoint_id: str, *, with_key: bool = False) -> dict | None:
        row = self.endpoints.get(endpoint_id)
        if row is None:
            return None
        public = {k: v for k, v in row.items() if with_key or k != "api_key"}
        return copy.deepcopy({**public, "has_key": bool(row.get("api_key"))})

    def find_endpoint_by_url(self, base_url: str) -> dict | None:
        return next((self.get_endpoint(e) for e, r in self.endpoints.items() if r["base_url"] == base_url), None)

    def insert_endpoint(self, row: dict) -> None:
        self.endpoints[row["endpoint_id"]] = copy.deepcopy(row)

    def update_endpoint(self, endpoint_id: str, fields: dict) -> None:
        self.endpoints[endpoint_id].update(copy.deepcopy(fields))

    def set_baseline(self, endpoint_id: str | None) -> None:
        for key, row in self.endpoints.items():
            row["is_baseline"] = key == endpoint_id

    def delete_endpoint(self, endpoint_id: str) -> bool:
        return self.endpoints.pop(endpoint_id, None) is not None

    # measurement runs
    def create_bench(self, run_id: str, kind: str, config: dict) -> None:
        self.bench[run_id] = {"run_id": run_id, "kind": kind, "status": "queued", "config": copy.deepcopy(config),
                              "progress": {}, "result": {}, "error": None, "created_at": f"t{len(self.bench)}",
                              "updated_at": None}

    def update_bench(self, run_id, status, *, progress=None, result=None, error=None) -> None:
        run = self.bench[run_id]
        run.update(status=status, error=error)
        if progress is not None:
            run["progress"] = json.loads(json.dumps(progress, default=str))
        if result is not None:
            run["result"] = json.loads(json.dumps(result, default=str))

    def get_bench(self, run_id: str) -> dict | None:
        return copy.deepcopy(self.bench.get(run_id))

    def list_bench(self, limit: int = 20) -> list[dict]:
        return [copy.deepcopy(r) for r in reversed(list(self.bench.values()))][:limit]

    def mark_bench_interrupted(self) -> int:
        count = 0
        for run in self.bench.values():
            if run["status"] in ("queued", "running"):
                run.update(status="interrupted", error="backend restarted")
                count += 1
        return count


def endpoint(endpoint_id: str, name: str, *, model_id: str | None = None, is_baseline: bool = False,
             status: str = "ok", platform: str = "modal") -> dict:
    """A saved endpoint row as the registry returns it (no key)."""
    return {"endpoint_id": endpoint_id, "name": name, "base_url": f"https://{endpoint_id.replace('_', '-')}.modal.run/v1",
            "platform": platform, "model_id": model_id or f"model-{name}", "max_model_len": 4096, "meta": {},
            "is_baseline": is_baseline, "status": status, "status_message": None,
            "checked_at": "2026-11-12T10:00:00+00:00", "created_at": "2026-11-12T10:00:00+00:00", "has_key": False}

    def run_readonly(self, sql):
        raise NotImplementedError


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload(self, path, data, content_type) -> None:
        self.uploads[path] = data

    def signed_url(self, path, expires_in=300) -> str:
        return f"https://storage.test/{path}?token=t"


def strip_strings(value: Any) -> Any:
    if isinstance(value, str):
        return strip_vietnamese_marks(value)
    if isinstance(value, list):
        return [strip_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_strings(v) for k, v in value.items()}
    return value


def node1_answer(request: dict, mutate: Callable[[dict], None] | None = None) -> str:
    """Example payload, unaccented as a model reading unaccented OCR text writes it, bound to the request's
    own SOURCE_METADATA and OCR_TEXT."""
    user = request["messages"][1]["content"]
    metadata = json.loads(user.split("SOURCE_METADATA:\n", 1)[1].split("\n", 1)[0])
    raw_text = user.split("OCR_TEXT:\n", 1)[1].rstrip("\n")
    payload = strip_strings(copy.deepcopy(EXAMPLE))
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


class FakeOcrReader:
    """Stands in for LibraryOcrClient: ``read(data)`` returns a LibraryOcrResult or raises."""

    def __init__(self, text: str = OCR_TEXT, *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[bytes] = []

    def read(self, data: bytes) -> LibraryOcrResult:
        self.calls.append(data)
        if self.error is not None:
            raise self.error
        return LibraryOcrResult(text=self.text, boxes=self.text.count("\n") + 1, latency_ms=900.0, engine="A",
                                versions={"vietocr": "0.3.13"})


def lane_callers(**overrides: FakeCaller) -> Callable[[str], FakeCaller]:
    def make(lane: str) -> FakeCaller:
        if lane in overrides:
            return overrides[lane]
        return FakeCaller(lane, content=node1_answer, tpot_ms=20.0 if lane == "vanilla" else 14.0)
    return make


# The four lanes of the earlier demo are now ordinary pasted endpoints; vanilla is the chosen baseline.
LANE_IDS = {"vanilla": "ep_0000000a", "static": "ep_0000000b", "adaptive": "ep_0000000c", "eagle": "ep_0000000d"}


class FakeRegistry(EndpointRegistry):
    """Real registry logic over the FakeStore, with scripted callers instead of HTTP."""

    def __init__(self, store: FakeStore, callers: Callable[[str], Any]) -> None:
        super().__init__(store, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(599))))
        self._callers = callers

    def caller(self, endpoint_id: str) -> Any:
        row = self._require(endpoint_id)
        return self._callers(row["name"])


def lane_endpoints(baseline: str | None = "vanilla") -> list[dict]:
    return [endpoint(eid, lane, model_id=f"model-{lane}", is_baseline=lane == baseline) for lane, eid in LANE_IDS.items()]


def build(*, store: FakeStore | None = None, ocr: FakeOcrReader | None = None,
          lanes: Callable[[str], FakeCaller] | None = None, endpoints: list[dict] | None = None,
          callers: dict[str, Any] | None = None, storage: FakeStorage | None = None, questions: Any = None,
          reports: Any = None, bench: Any = None, clock: Callable[[], float] | None = None,
          settings: Settings | None = None) -> tuple[Any, FakeStore, FakeStorage]:
    """``endpoints`` default to the four lane endpoints (vanilla = baseline); ``lanes`` maps an endpoint name to its
    caller, or ``callers`` maps an endpoint id to its caller."""
    store = store or FakeStore()
    storage = storage or FakeStorage()
    settings = settings or Settings(demo_key=DEMO_KEY, frontend_origins=("https://demo.vercel.app",),
                                    writer_url=SECRET_URL)
    rows = lane_endpoints() if endpoints is None else endpoints
    for row in rows:
        store.insert_endpoint(row)
    if callers is not None:
        by_name = {row["name"]: callers[row["endpoint_id"]] for row in rows if row["endpoint_id"] in callers}
        caller_of = by_name.__getitem__
    else:
        caller_of = lanes or lane_callers()
    registry = FakeRegistry(store, caller_of)
    ocr_reader = ocr or FakeOcrReader()
    pipeline = InvoicePipeline(store, ocr_reader=lambda: ocr_reader, caller_for=registry.caller, storage=storage)
    deps = Deps(store=store, endpoints=registry, pipeline=pipeline, storage=storage, questions=questions,
                reports=reports, bench=bench, clock=clock or (lambda: 0.0))
    return create_app(settings, deps), store, storage


HEADERS = {"X-Demo-Key": DEMO_KEY}
