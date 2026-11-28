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
from src.api.job_log import JobLog
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
STEP_LABELS = {"ocr": "OCR", "node1": "Node 1", "saving": "lưu"}


# Numbers in the job log are written the Vietnamese way (1.234 ký tự, 1,2 s).
def _num(value: float | int) -> str:
    return f"{round(value):,}".replace(",", ".")


def _sec(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _dur(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    if not minutes:
        return f"{rest} s"
    return f"{minutes} phút" if not rest else f"{minutes} phút {rest} s"


def _waiting(who: str, event: dict[str, Any]) -> str:
    text = (f"{who} chưa sẵn sàng ({event.get('reason', 'HTTP 503')}), đã thử {event.get('tries', 1)} lần "
            f"trong {_dur(event.get('waited_s', 0))}")
    if event.get("retry_in_s"):
        text += f"; thử lại sau {_dur(event['retry_in_s'])}"
    if event.get("ready_timeout_s"):
        text += f", chờ tối đa {_dur(event['ready_timeout_s'])}"
    return text


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
        self.message = message


class InvoicePipeline:
    """Node 1 on the endpoints chosen for one upload. The first endpoint is stored (the baseline when the
    first one fails); every endpoint is compared with the baseline for speed and fields."""

    def __init__(self, store: Store, *, ocr_reader: Callable[[], Any], caller_for: Callable[[str], Any],
                 storage: Any = None, redact: Callable[[str], str] = lambda s: s,
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self._clock = clock
        self.storage = storage
        self.ocr_reader = ocr_reader
        self.caller_for = caller_for
        self._redact = redact

    def run(self, job_id: str, data: bytes, ext: str, endpoints: list[dict], baseline_id: str | None = None) -> None:
        result: dict[str, Any] = {
            "endpoints": [{k: e.get(k) for k in ("endpoint_id", "name", "model_id")} for e in endpoints],
            "primary": endpoints[0]["endpoint_id"], "baseline": baseline_id, "timings_ms": {}}
        started = time.perf_counter()
        events = JobLog(self.store, job_id, result, clock=self._clock)
        events.add("system", "info", f"Nhận ảnh {ext.upper()} ({_num(len(data) / 1024)} KB); "
                                     f"chạy Node 1 trên {len(endpoints)} endpoint")
        try:
            invoice_id = self._run(job_id, data, ext, endpoints, baseline_id, result, events)
        except JobFailed as exc:
            log.info("job failed", extra={"job_id": job_id, "step": exc.step})
            events.add("system", "error", f"Dừng ở bước {STEP_LABELS.get(exc.step, exc.step)}: "
                                          f"{self._redact(exc.message)[:400]}")
            self.store.update_job(job_id, "failed", result=result, error=self._redact(str(exc))[:500])
        except Exception as exc:  # never leave a job running
            log.exception("job crashed", extra={"job_id": job_id})
            error = self._redact(f"internal: {type(exc).__name__}: {exc}")[:500]
            events.add("system", "error", f"Lỗi nội bộ của backend: {error}")
            self.store.update_job(job_id, "failed", result=result, error=error)
        else:
            events.add("system", "ok", f"Hoàn tất sau {_sec(time.perf_counter() - started)} s")
            self.store.update_job(job_id, "done", result=result, invoice_id=invoice_id)

    def read_text(self, data: bytes, result: dict, events: JobLog | None = None) -> str:
        """OCR + mark stripping: the same text path as the W1_v3 package."""
        started = time.perf_counter()
        if events is not None:
            events.add("ocr", "info", "Gửi ảnh tới máy OCR (PaddleOCR tách dòng, VietOCR đọc chữ)")
        try:
            reader = self.ocr_reader()
            if events is not None:
                reader.progress = lambda event: events.add(
                    "ocr", "warn", _waiting("Máy OCR", event), key="ocr-wait",
                    data={"tries": event.get("tries"), "waited_s": event.get("waited_s")})
            ocr_result = reader.read(data)
        except Exception as exc:
            raise JobFailed("ocr", self._redact(str(exc))) from exc
        result["timings_ms"]["ocr"] = (time.perf_counter() - started) * 1000
        text = normalize_text(strip_vietnamese_marks(ocr_result.text))
        result["ocr"] = {"text": text, "kind": "library", "engine": ocr_result.engine or CHOSEN_ENGINE,
                         "marks_stripped": True, "boxes": ocr_result.boxes, "latency_ms": ocr_result.latency_ms,
                         "versions": ocr_result.versions}
        if not text:
            raise JobFailed("ocr", "no text detected")
        if events is not None:
            events.add("ocr", "ok", f"Đọc xong {len(text.splitlines())} dòng chữ trong "
                                    f"{_sec(time.perf_counter() - started)} s; đã bỏ dấu tiếng Việt")
        return text

    def _run(self, job_id: str, data: bytes, ext: str, endpoints: list[dict], baseline_id: str | None,
             result: dict, events: JobLog) -> str:
        storage_path = f"uploads/{job_id}.{ext}"
        if self.storage is not None:
            self.storage.upload(storage_path, data, CONTENT_TYPES[ext])
        result["storage_path"] = storage_path

        events.set_status("ocr")
        text = self.read_text(data, result, events)

        events.set_status("node1")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            outcomes = dict(zip((e["endpoint_id"] for e in endpoints),
                                pool.map(lambda e: self.run_endpoint(job_id, storage_path, text, e, events),
                                         endpoints)))
        result["timings_ms"]["node1"] = (time.perf_counter() - started) * 1000
        result["runs"] = {eid: outcome["report"] for eid, outcome in outcomes.items()}
        self.compare(outcomes, result["runs"], baseline_id)
        self._log_comparison(events, result["runs"], baseline_id)

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

        report = outcomes[stored]["report"]
        fallback = " (endpoint chọn đầu bị lỗi, lấy kết quả của mốc)" if stored != primary else ""
        events.add("system", "info", f"Lưu hoá đơn vào cơ sở dữ liệu từ {report['name']}{fallback}")
        events.set_status("saving")
        run = RunInfo(run_id=job_id, lane=None, model_name=report["model"], prompt_version=W1_V2_PROMPT_VERSION,
                      notes=f"web upload, endpoint {report['name']}")
        invoice_id = self.store.save_invoice(payload, run, storage_path=storage_path, latency_ms=success.latency_ms)
        result["invoice_id"] = invoice_id
        events.add("system", "ok", f"Đã lưu hoá đơn, mã {invoice_id}")
        return invoice_id

    def run_endpoint(self, job_id: str, storage_path: str, ocr_text: str, endpoint: dict,
                     events: JobLog | None = None) -> dict[str, Any]:
        src = endpoint["endpoint_id"]
        caller = self.caller_for(src)
        if events is not None:
            events.add(src, "info", f"Gửi yêu cầu Node 1 (prompt {W1_V2_PROMPT_VERSION}, "
                                    f"trả về tối đa {_num(NODE1_MAX_TOKENS)} token)")
            caller.progress = lambda event: self._caller_event(events, src, event)
        row = node1_request_row(job_id, storage_path, ocr_text, caller.model)
        outcome = Node1Extractor(caller, caller.model).run_request(row)
        metrics = caller.last_metrics.as_dict() if caller.last_metrics else {}
        ok = isinstance(outcome, Node1Success)
        restored = [n.split(" ", 1)[0].removeprefix("source.") for n in (outcome.normalizations if ok else ())
                    if n.startswith("source") and "restored from the request" in n]
        report = {**metrics, "endpoint_id": endpoint["endpoint_id"], "name": caller.lane, "model": caller.model,
                  "valid": ok, "source_restored": restored,
                  "error": None if ok else self._redact(f"{outcome.error_code.value} at {outcome.stage.value}: "
                                                         f"{outcome.message}")[:400],
                  # Each endpoint's own extraction, shown column by column in the console (design D13).
                  "payload": outcome.canonical_payload if ok else None}
        if events is not None:
            if ok:
                tokens, tpot = metrics.get("completion_tokens"), metrics.get("tpot_ms")
                events.add(src, "ok", f"Xong: {_num(tokens) if tokens else '?'} token, "
                                      f"TPOT {_num(tpot) if tpot else '?'} ms, JSON hợp lệ",
                           data={"tokens": tokens, "tpot_ms": tpot, "ttft_ms": metrics.get("ttft_ms")})
                if restored:
                    events.add(src, "warn", f"Model chép sai nguồn ({', '.join(restored)}); "
                                            "hệ thống đã điền lại giá trị đúng")
            else:
                events.add(src, "error", f"Lỗi: {report['error']}")
        return {"success": outcome if ok else None, "report": report}

    @staticmethod
    def _caller_event(events: JobLog, src: str, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "waiting":
            events.add(src, "warn", _waiting("Endpoint", event), key=f"{src}-wait",
                       data={"tries": event.get("tries"), "waited_s": event.get("waited_s")})
        elif kind == "first_token":
            events.add(src, "ok", f"Token đầu tiên sau {_sec(event['ttft_s'])} s, endpoint đang trả lời",
                       data={"ttft_s": event["ttft_s"]})
        elif kind == "receiving":
            events.add(src, "info", f"Đang nhận {_num(event['chars'])} ký tự", key=f"{src}-recv",
                       data={"chars": event["chars"]})

    @staticmethod
    def _log_comparison(events: JobLog, runs: dict[str, dict], baseline_id: str | None) -> None:
        base = runs.get(baseline_id) if baseline_id else None
        if base is None:
            return
        for eid, report in runs.items():
            if eid == baseline_id or report.get("total_fields") is None:
                continue
            speed = report.get("speedup_tpot_vs_baseline")
            events.add("system", "warn" if report["differing_fields"] else "ok",
                       f"{report['name']} so với mốc {base['name']}: {report['equal_fields']}/{report['total_fields']} "
                       f"trường giống" + (f", nhanh hơn {_sec(speed, 2)}× theo TPOT" if speed else ""))

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
