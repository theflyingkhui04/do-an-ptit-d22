"""Upload job: image -> OCR -> Node 1 (lane map + vanilla comparison) -> demo database."""

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
from src.pipeline import ocr
from src.pipeline.extractor import Node1Extractor, Node1Success
from src.pipeline.item_consistency import invoice_sum_consistent, line_consistent
from src.pipeline.node1_request import build_chat_request
from src.pipeline.postgres.writer import RunInfo
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
    def __init__(self, store: Store, *, ocr_caller: Callable[[], Any], lane_caller: Callable[[str], Any],
                 lane_map: dict[str, str], storage: Any = None, gpu_label: str = "A10G",
                 on_service: Callable[[str, bool, str | None], None] | None = None,
                 redact: Callable[[str], str] = lambda s: s) -> None:
        self.store = store
        self.storage = storage
        self.ocr_caller = ocr_caller
        self.lane_caller = lane_caller
        self.lane_map = lane_map
        self.gpu_label = gpu_label
        self._on_service = on_service or (lambda name, ok, message: None)
        self._redact = redact

    def run(self, job_id: str, data: bytes, ext: str, compare_lanes: tuple[str, ...] = ("vanilla",)) -> None:
        result: dict[str, Any] = {"gpu": self.gpu_label, "lane_map": dict(self.lane_map), "timings_ms": {}}
        try:
            invoice_id = self._run(job_id, data, ext, compare_lanes, result)
        except JobFailed as exc:
            log.info("job failed", extra={"job_id": job_id, "step": exc.step})
            self.store.update_job(job_id, "failed", result=result, error=self._redact(str(exc))[:500])
        except Exception as exc:  # never leave a job running
            log.exception("job crashed", extra={"job_id": job_id})
            self.store.update_job(job_id, "failed", result=result,
                                  error=self._redact(f"internal: {type(exc).__name__}: {exc}")[:500])
        else:
            self.store.update_job(job_id, "done", result=result, invoice_id=invoice_id)

    def _run(self, job_id: str, data: bytes, ext: str, compare_lanes: tuple[str, ...], result: dict) -> str:
        storage_path = f"uploads/{job_id}.{ext}"
        if self.storage is not None:
            self.storage.upload(storage_path, data, CONTENT_TYPES[ext])
        result["storage_path"] = storage_path

        self.store.update_job(job_id, "ocr", result=result)
        caller = self.ocr_caller()
        started = time.perf_counter()
        try:
            ocr_result = ocr.run_ocr(caller, data)
        except Exception as exc:
            if not isinstance(exc, ocr.OcrError):  # transport failure: the OCR service itself is down
                self._on_service("ocr", False, str(exc))
            raise JobFailed("ocr", self._redact(str(exc))) from exc
        self._on_service("ocr", True, None)
        result["timings_ms"]["ocr"] = (time.perf_counter() - started) * 1000
        result["ocr"] = {"text": ocr_result.text, "finish_reason": ocr_result.finish_reason,
                         "prompt_tokens": ocr_result.prompt_tokens, "completion_tokens": ocr_result.completion_tokens,
                         "scale": ocr_result.image.scale, "original_size": list(ocr_result.image.original_size),
                         "model": ocr.OCR_MODEL, "revision": ocr.OCR_REVISION,
                         "prompt_version": ocr.OCR_PROMPT_VERSION}
        if ocr_result.finish_reason != "stop":
            raise JobFailed("ocr", f"OCR finish_reason={ocr_result.finish_reason} (text may be cut)")

        self.store.update_job(job_id, "node1", result=result)
        node1_lane = self.lane_map["node1"]
        lanes = list(dict.fromkeys([node1_lane, *compare_lanes]))
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            outcomes = dict(zip(lanes, pool.map(lambda lane: self._run_lane(job_id, storage_path,
                                                                            ocr_result.text, lane), lanes)))
        result["timings_ms"]["node1"] = (time.perf_counter() - started) * 1000
        result["lanes"] = {lane: outcome["report"] for lane, outcome in outcomes.items()}
        self._compare(outcomes, result)

        stored_lane = node1_lane if outcomes[node1_lane]["success"] else None
        if stored_lane is None and "vanilla" in outcomes and outcomes["vanilla"]["success"]:
            stored_lane = "vanilla"
        if stored_lane is None:
            errors = "; ".join(f"{lane}: {o['report']['error']}" for lane, o in outcomes.items())
            raise JobFailed("node1", errors)
        success: Node1Success = outcomes[stored_lane]["success"]
        payload = success.canonical_payload
        result["stored_lane"] = stored_lane
        result["stored_from_fallback"] = stored_lane != node1_lane
        result["invoice"] = payload
        result["grounding"] = grounding_report(payload, ocr_result.text)
        result["item_check"] = {"lines": [line_consistent(item) for item in payload.get("items") or []],
                                "invoice_sum": invoice_sum_consistent(payload)}

        self.store.update_job(job_id, "saving", result=result)
        run = RunInfo(run_id=job_id, lane=stored_lane, model_name=outcomes[stored_lane]["report"]["model"],
                      prompt_version=W1_V2_PROMPT_VERSION, notes="web demo upload")
        invoice_id = self.store.save_invoice(payload, run, storage_path=storage_path,
                                             latency_ms=success.latency_ms)
        result["invoice_id"] = invoice_id
        return invoice_id

    def _run_lane(self, job_id: str, storage_path: str, ocr_text: str, lane: str) -> dict[str, Any]:
        caller = self.lane_caller(lane)
        row = node1_request_row(job_id, storage_path, ocr_text, caller.model)
        outcome = Node1Extractor(caller, caller.model).run_request(row)
        metrics = caller.last_metrics.as_dict() if caller.last_metrics else {}
        ok = isinstance(outcome, Node1Success)
        transport_failed = not ok and outcome.error_code.value == "TRANSPORT_ERROR"
        self._on_service(lane, not transport_failed, None if not transport_failed else outcome.message)
        report = {"lane": lane, "model": caller.model, "valid": ok, **metrics,
                  "error": None if ok else self._redact(f"{outcome.error_code.value} at {outcome.stage.value}: "
                                                         f"{outcome.message}")[:400]}
        return {"success": outcome if ok else None, "report": report}

    @staticmethod
    def _compare(outcomes: dict[str, dict], result: dict) -> None:
        vanilla = outcomes.get("vanilla")
        for lane, outcome in outcomes.items():
            report = result["lanes"][lane]
            report["speedup_tpot_vs_vanilla"] = None
            report["speedup_latency_vs_vanilla"] = None
            report["equal_fields"] = report["total_fields"] = None
            report["differing_fields"] = []
            if vanilla is None:
                continue
            ref = vanilla["report"]
            report["speedup_tpot_vs_vanilla"] = speedup(ref.get("tpot_ms"), report.get("tpot_ms"))
            report["speedup_latency_vs_vanilla"] = speedup(ref.get("latency_ms"), report.get("latency_ms"))
            if outcome["success"] and vanilla["success"]:
                comparison = compare_fields(outcome["success"].canonical_payload,
                                            vanilla["success"].canonical_payload)
                report.update(comparison)
