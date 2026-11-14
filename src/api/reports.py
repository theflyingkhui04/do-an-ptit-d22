"""Node 3 in the web console.

``run_request`` is Node 3 on its own (the Node 3 page): the person's request plus the tables of
``report_data.build_snapshot`` (fixed queries, no Node 2). The model is called twice on the chosen endpoint:
once to choose 2-4 charts from those tables (JSON, checked here; the web draws the numbers from the rows),
once to write the report with the frozen W3 v2 system prompt. The report text gets a number check.

``run`` is the chain mode (Node 1 -> 2 -> 3, comparable with M3-06): Q1-Q6 through Node 2, then the report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from src.api.questions import _NUMBER, QuestionService, _table_text, ungrounded_numbers
from src.api.report_data import build_snapshot, scope_label
from src.api.store import Store
from src.pipeline.node2_questions import QUESTIONS
from src.pipeline.report_writer import (
    NODE3_USER_TEMPLATE_PATH,
    W3_SYSTEM_PROMPT_PATH,
    Node3Writer,
)
from src.pipeline.text_to_sql import (
    REPO_ROOT,
    SEED,
    TEMPERATURE,
    Node2Result,
    response_content,
    usage_tokens,
)

NODE2_STATUS = {"ok": "ok", "no_data": "empty", "error": "sql_error"}
DASHBOARD_PROMPT_PATH = REPO_ROOT / "src/pipeline/prompts/node3_dashboard_demo.md"
REQUEST_TEMPLATE_PATH = REPO_ROOT / "src/pipeline/prompts/node3_request_user.md"
DASHBOARD_MAX_TOKENS = 320
CHART_TYPES = ("kpi", "bar", "column", "table")
MAX_CHARTS = 4
DEFAULT_CHARTS = (("kpis", "kpi"), ("by_date", "column"), ("by_store", "bar"), ("top_items", "table"))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def numeric_columns(table: dict) -> list[str]:
    """Columns whose non-empty values are all numbers (at least one)."""
    names = []
    for index, name in enumerate(table["columns"]):
        values = [row[index] for row in table["rows"] if row[index] is not None]
        if values and all(_is_number(v) for v in values):
            names.append(name)
    return names


def check_chart(choice: Any, tables: dict[str, dict]) -> dict | None:
    """One chart the web can draw from the rows, or None. ``value`` defaults to the last numeric column."""
    if not isinstance(choice, dict):
        return None
    table, kind = tables.get(str(choice.get("table"))), choice.get("type")
    if table is None or not table["rows"] or kind not in CHART_TYPES:
        return None
    numbers = numeric_columns(table)
    value = choice.get("value") if choice.get("value") in numbers else None
    if kind == "kpi":
        if len(table["rows"]) != 1 or not numbers:
            return None
        value = None
    elif kind in ("bar", "column"):
        if not numbers or len(numbers) == len(table["columns"]):  # needs a label column too
            return None
        value = value or numbers[-1]
    else:
        value = None
    title = choice.get("title")
    title = title.strip()[:80] if isinstance(title, str) and title.strip() else table["title"]
    return {"table": table["key"], "type": kind, "value": value, "title": title}


def parse_dashboard(raw: str, tables: dict[str, dict]) -> tuple[list[dict], str | None]:
    """Charts from the model's JSON (bad entries dropped), or ``([], reason)``."""
    start, end = raw.find("{"), raw.rfind("}")
    plan = None
    if 0 <= start < end:
        try:
            plan = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            plan = None
    if not isinstance(plan, dict) or not isinstance(plan.get("charts"), list):
        return [], "Model không trả JSON hợp lệ cho dashboard."
    charts, seen = [], set()
    for choice in plan["charts"]:
        chart = check_chart(choice, tables)
        if chart and (key := (chart["table"], chart["type"], chart["value"])) not in seen:
            seen.add(key)
            charts.append(chart)
    if not charts:
        return [], "Model không chọn được biểu đồ nào vẽ được từ các bảng."
    return charts[:MAX_CHARTS], None


def default_charts(tables: dict[str, dict]) -> list[dict]:
    return [c for key, kind in DEFAULT_CHARTS if (c := check_chart({"table": key, "type": kind}, tables))]


def vn_number(value: Any) -> str:
    """117000 -> 117.000, 0.5 -> 0,5: the way Vietnamese reports write numbers. Live run 13/11: given 63750 the
    model wrote 637,500 while adding its own separators; numbers already written this way are copied as they are."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    if isinstance(value, float) and not value.is_integer():
        whole, _, frac = f"{value:,.2f}".rstrip("0").partition(".")
        return whole.replace(",", ".") + ("," + frac if frac else "")
    return f"{int(value):,}".replace(",", ".")


def tables_text(tables: list[dict]) -> str:
    return "\n\n".join(
        f"### {t['title']} (bảng {t['key']}, {len(t['rows'])} dòng)\n"
        f"{_table_text(t['columns'], [[vn_number(v) for v in row] for row in t['rows']])}" for t in tables)


def dashboard_message(request: str, scope: str, tables: list[dict]) -> str:
    lines = [f"Yêu cầu: {request}", f"Phạm vi: {scope}", "", "Các bảng có sẵn:"]
    for t in tables:
        lines.append(f"- {t['key']}: {t['title']}; cột {', '.join(t['columns'])}; {len(t['rows'])} dòng; "
                     f"dòng đầu: {t['rows'][:3]}")
    return "\n".join(lines)


class ReportService:
    def __init__(self, store: Store, questions: QuestionService, *, clock: Callable[[], float] = time.perf_counter,
                 max_tokens: int = 768, redact: Callable[[str], str] = lambda s: s) -> None:
        self.store = store
        self.questions = questions
        self._clock = clock
        self.max_tokens = max_tokens
        self._redact = redact

    def _chat(self, caller: Any, messages: list[dict], max_tokens: int, temperature: float) -> tuple[dict, dict]:
        response = caller.chat_completions({"messages": messages, "temperature": temperature, "seed": SEED,
                                            "max_tokens": max_tokens})
        metrics = caller.last_metrics.as_dict() if getattr(caller, "last_metrics", None) else {}
        return response, metrics

    def run_request(self, request: str, limit: int | None, caller: Any) -> dict[str, Any]:
        """Node 3 on its own: fixed tables of the scope, charts chosen by the model, then the report."""
        started = self._clock()
        snapshot = build_snapshot(self.store.run_readonly, limit)
        data_ms = (self._clock() - started) * 1000
        tables = snapshot["tables"]
        by_key = {t["key"]: t for t in tables}
        scope = scope_label(snapshot["limit"])
        count = snapshot["invoice_count"]
        out: dict[str, Any] = {
            "request": request, "scope": {"limit": snapshot["limit"], "invoice_count": count, "label": scope},
            "tables": tables,
            "dashboard": {"status": "skipped", "charts": [], "error": None, "metrics": {}},
            "report": {"status": "skipped_no_data", "text": "", "error": None, "latency_ms": None,
                       "completion_tokens": None, "finish_reason": None, "metrics": {}},
            "number_check": {"numbers": 0, "ungrounded": []},
            "timings_ms": {"data": data_ms, "dashboard": 0.0, "report": 0.0}}
        if count == 0:
            return out

        started = self._clock()
        messages = [{"role": "system", "content": DASHBOARD_PROMPT_PATH.read_text(encoding="utf-8").strip()},
                    {"role": "user", "content": dashboard_message(request, scope, tables)}]
        try:
            response, metrics = self._chat(caller, messages, DASHBOARD_MAX_TOKENS, 0.0)
            charts, error = parse_dashboard(response_content(response)[0] or "", by_key)
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            charts, error, metrics = [], self._redact(f"Gọi endpoint lỗi: {exc}")[:400], {}
        out["dashboard"] = {"status": "ok" if charts else "fallback", "charts": charts or default_charts(by_key),
                            "error": error, "metrics": metrics}
        out["timings_ms"]["dashboard"] = (self._clock() - started) * 1000

        template = REQUEST_TEMPLATE_PATH.read_text(encoding="utf-8")
        user = (template.replace("{tables}", tables_text(tables)).replace("{scope}", scope)
                .replace("{invoice_count}", str(count)).replace("{request}", request).rstrip("\n"))
        messages = [{"role": "system", "content": W3_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")},
                    {"role": "user", "content": user}]
        started = self._clock()
        report = out["report"]
        try:
            response, metrics = self._chat(caller, messages, self.max_tokens, TEMPERATURE)
            text, finish = response_content(response)
            report.update(status="ok" if (text or "").strip() else "empty_response", text=text or "",
                          finish_reason=finish, completion_tokens=usage_tokens(response)[1], metrics=metrics)
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            report.update(status="transport_error", error=self._redact(f"Gọi endpoint lỗi: {exc}")[:400])
        report["latency_ms"] = out["timings_ms"]["report"] = (self._clock() - started) * 1000

        rows = [row for t in tables for row in t["rows"]]
        context = " ".join([request, scope, str(count), template, *(t["title"] for t in tables)])
        out["number_check"] = {"numbers": len(_NUMBER.findall(report["text"])),
                               "ungrounded": ungrounded_numbers(report["text"], rows, context)}
        return out

    def invoice_count(self) -> int:
        _, rows = self.store.run_readonly("SELECT COUNT(*) AS n FROM v_invoice_facts")
        return int(rows[0][0]) if rows else 0

    def run(self, caller: Any) -> dict[str, Any]:
        started = self._clock()
        results = []
        for question in QUESTIONS:
            result = self.questions.query(question.question, caller)
            results.append({"question_id": question.id, **result})
        node2_ms = (self._clock() - started) * 1000

        node2 = [Node2Result(question_id=r["question_id"], question=r["question"], sql=r["sql"],
                             status=NODE2_STATUS[r["status"]], error=r["error"], columns=r["columns"],
                             rows=r["rows"], row_count=r["row_count"]) for r in results]
        count = self.invoice_count()
        started = self._clock()
        node3 = Node3Writer(caller, caller.model, W3_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
                            max_tokens=self.max_tokens).write(node2, count)
        node3_ms = (self._clock() - started) * 1000

        text = node3.report_text or ""
        rows = [row for r in results for row in r["rows"]]
        # Numbers the report may quote besides the rows: the questions, the invoice count, the section numbers
        # and Q-ids of the frozen user template.
        context = " ".join([*(q.question for q in QUESTIONS), *(q.id for q in QUESTIONS), str(count),
                            NODE3_USER_TEMPLATE_PATH.read_text(encoding="utf-8")])
        metrics = caller.last_metrics.as_dict() if node3.status == "ok" and getattr(caller, "last_metrics", None) else {}
        return {"questions": results, "invoice_count": count,
                "report": {**node3.to_dict(), "text": text, "metrics": metrics},
                "number_check": {"numbers": len(_NUMBER.findall(text)),
                                 "ungrounded": ungrounded_numbers(text, rows, context)},
                "timings_ms": {"node2": node2_ms, "node3": node3_ms}}
