"""Upload job: image -> library OCR (marks stripped) -> Node 1 (lane map + vanilla comparison) -> demo database."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.api.compare import compare_fields, speedup
from src.api.grounding import grounding_report
from src.api.store import Store
from src.pipeline.extractor import Node1Extractor, Node1Success
from src.pipeline.item_consistency import invoice_sum_consistent, line_consistent
from src.pipeline.library_ocr import CHOSEN_ENGINE
from src.pipeline.node1_request import build_chat_request, normalize_text
from src.pipeline.postgres.writer import RunInfo
from src.pipeline.text_normalize import strip_vietnamese_marks
from src.pipeline.w1_workload import W1_V2_PROMPT_VERSION, compose_node1_prompt_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_PROMPT = REPO_ROOT / "src/pipeline/prompts/node1_extractor_system.md"
FULL_RECEIPT_RULES = REPO_ROOT / "src/pipeline/prompts/node1_full_receipt_rules.md"
NODE1_MAX_TOKENS = 3072
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png"}

log = logging.getLogger("src.api.jobs")


@lru_cache(maxsize=1)
def node1_system_prompt() -> str:
    """Prompt ``node1_extractor.v2``: identical to experiments/configs/w1_v2/system_prompt.md."""
    return compose_node1_prompt_v2(V1_PROMPT.read_text(encoding="utf-8"),
                                   FULL_RECEIPT_RULES.read_text(encoding="utf-8"))


def node1_request_row(job_id: str, storage_path: str, ocr_text: str, model: str) -> dict[str, Any]:
    sample = {"sample_id": job_id, "img_id": job_id, "image_path": storage_path, "raw_text": ocr_text}
    row = build_chat_request(sample, node1_system_prompt(), model, NODE1_MAX_TOKENS, 0.0, 42)
    row["prompt_version"] = W1_V2_PROMPT_VERSION
    return row


class JobFailed(Exception):
    def __init__(self, step: str, message: str) -> None:
        super().__init__(f"{step}: {message}")
        self.step = step


class InvoicePipeline:
    """Node 1 on the endpoints chosen for one upload. The first endpoint is stored (the baseline when the
    first one fails); every endpoint is compared with the baseline for speed and fields."""

    def __init__(self, store: Store, *, ocr_reader: Callable[[], Any], caller_for: Callable[[str], Any],
                 storage: Any = None, redact: Callable[[str], str] = lambda s: s) -> None:
        self.store = store
        self.storage = storage
        self.ocr_reader = ocr_reader
        self.caller_for = caller_for
        self._redact = redact

    def run(self, job_id: str, data: bytes, ext: str, endpoints: list[dict], baseline_id: str | None = None) -> None:
        result: dict[str, Any] = {
            "endpoints": [{k: e.get(k) for k in ("endpoint_id", "name", "model_id")} for e in endpoints],
            "primary": endpoints[0]["endpoint_id"], "baseline": baseline_id, "timings_ms": {}}
        try:
            invoice_id = self._run(job_id, data, ext, endpoints, baseline_id, result)
        except JobFailed as exc:
            log.info("job failed", extra={"job_id": job_id, "step": exc.step})
            self.store.update_job(job_id, "failed", result=result, error=self._redact(str(exc))[:500])
        except Exception as exc:  # never leave a job running
            log.exception("job crashed", extra={"job_id": job_id})
            self.store.update_job(job_id, "failed", result=result,
                                  error=self._redact(f"internal: {type(exc).__name__}: {exc}")[:500])
        else:
            self.store.update_job(job_id, "done", result=result, invoice_id=invoice_id)

    def read_text(self, data: bytes, result: dict) -> str:
        """OCR + mark stripping: the same text path as the W1_v3 package."""
        started = time.perf_counter()
        try:
            ocr_result = self.ocr_reader().read(data)
        except Exception as exc:
            raise JobFailed("ocr", self._redact(str(exc))) from exc
        result["timings_ms"]["ocr"] = (time.perf_counter() - started) * 1000
        text = normalize_text(strip_vietnamese_marks(ocr_result.text))
        result["ocr"] = {"text": text, "kind": "library", "engine": ocr_result.engine or CHOSEN_ENGINE,
                         "marks_stripped": True, "boxes": ocr_result.boxes, "latency_ms": ocr_result.latency_ms,
                         "versions": ocr_result.versions}
        if not text:
            raise JobFailed("ocr", "no text detected")
        return text

    def _run(self, job_id: str, data: bytes, ext: str, endpoints: list[dict], baseline_id: str | None,
             result: dict) -> str:
        storage_path = f"uploads/{job_id}.{ext}"
        if self.storage is not None:
            self.storage.upload(storage_path, data, CONTENT_TYPES[ext])
        result["storage_path"] = storage_path

        self.store.update_job(job_id, "ocr", result=result)
        text = self.read_text(data, result)

        self.store.update_job(job_id, "node1", result=result)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            outcomes = dict(zip((e["endpoint_id"] for e in endpoints),
                                pool.map(lambda e: self.run_endpoint(job_id, storage_path, text, e), endpoints)))
        result["timings_ms"]["node1"] = (time.perf_counter() - started) * 1000
        result["runs"] = {eid: outcome["report"] for eid, outcome in outcomes.items()}
        self.compare(outcomes, result["runs"], baseline_id)

        primary = endpoints[0]["endpoint_id"]
        stored = primary if outcomes[primary]["success"] else None
        if stored is None and baseline_id in outcomes and outcomes[baseline_id]["success"]:
            stored = baseline_id
        if stored is None:
            errors = "; ".join(f"{o['report']['name']}: {o['report']['error']}" for o in outcomes.values())
            raise JobFailed("node1", errors)
        success: Node1Success = outcomes[stored]["success"]
        payload = success.canonical_payload
        result["stored_endpoint"] = stored
        result["stored_from_fallback"] = stored != primary
        result["invoice"] = payload
        result["grounding"] = grounding_report(payload, text)
        result["item_check"] = {"lines": [line_consistent(item) for item in payload.get("items") or []],
                                "invoice_sum": invoice_sum_consistent(payload)}

        self.store.update_job(job_id, "saving", result=result)
        report = outcomes[stored]["report"]
        run = RunInfo(run_id=job_id, lane=None, model_name=report["model"], prompt_version=W1_V2_PROMPT_VERSION,
                      notes=f"web upload, endpoint {report['name']}")
        invoice_id = self.store.save_invoice(payload, run, storage_path=storage_path, latency_ms=success.latency_ms)
        result["invoice_id"] = invoice_id
        return invoice_id

    def run_endpoint(self, job_id: str, storage_path: str, ocr_text: str, endpoint: dict) -> dict[str, Any]:
        caller = self.caller_for(endpoint["endpoint_id"])
        row = node1_request_row(job_id, storage_path, ocr_text, caller.model)
        outcome = Node1Extractor(caller, caller.model).run_request(row)
        metrics = caller.last_metrics.as_dict() if caller.last_metrics else {}
        ok = isinstance(outcome, Node1Success)
        restored = [n.split(" ", 1)[0].removeprefix("source.") for n in (outcome.normalizations if ok else ())
                    if n.startswith("source") and "restored from the request" in n]
        report = {**metrics, "endpoint_id": endpoint["endpoint_id"], "name": caller.lane, "model": caller.model,
                  "valid": ok, "source_restored": restored,
                  "error": None if ok else self._redact(f"{outcome.error_code.value} at {outcome.stage.value}: "
                                                         f"{outcome.message}")[:400]}
        return {"success": outcome if ok else None, "report": report}

    @staticmethod
    def compare(outcomes: dict[str, dict], reports: dict[str, dict], baseline_id: str | None) -> None:
        base = outcomes.get(baseline_id) if baseline_id else None
        for eid, outcome in outcomes.items():
            report = reports[eid]
            report["is_baseline"] = eid == baseline_id
            report["speedup_tpot_vs_baseline"] = report["speedup_latency_vs_baseline"] = None
            report["equal_fields"] = report["total_fields"] = None
            report["differing_fields"] = []
            if base is None or eid == baseline_id:
                continue
            ref = base["report"]
            report["speedup_tpot_vs_baseline"] = speedup(ref.get("tpot_ms"), report.get("tpot_ms"))
            report["speedup_latency_vs_baseline"] = speedup(ref.get("latency_ms"), report.get("latency_ms"))
            if outcome["success"] and base["success"]:
                report.update(compare_fields(outcome["success"].canonical_payload, base["success"].canonical_payload))
