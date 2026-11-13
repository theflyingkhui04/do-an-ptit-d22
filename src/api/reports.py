"""Node 3 in the web console: Q1–Q6 through Node 2 on one endpoint, then the grounded Vietnamese report.

The report prompt is the frozen W3 v2 prompt (``src/pipeline/report_writer.py``). Charts on the web are drawn
from the Q1–Q6 rows returned here, never from the report text; the report text gets a number check.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.api.questions import _NUMBER, QuestionService, ungrounded_numbers
from src.api.store import Store
from src.pipeline.node2_questions import QUESTIONS
from src.pipeline.report_writer import (
    NODE3_USER_TEMPLATE_PATH,
    W3_SYSTEM_PROMPT_PATH,
    Node3Writer,
)
from src.pipeline.text_to_sql import Node2Result

NODE2_STATUS = {"ok": "ok", "no_data": "empty", "error": "sql_error"}


class ReportService:
    def __init__(self, store: Store, questions: QuestionService, *, clock: Callable[[], float] = time.perf_counter,
                 max_tokens: int = 768) -> None:
        self.store = store
        self.questions = questions
        self._clock = clock
        self.max_tokens = max_tokens

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
