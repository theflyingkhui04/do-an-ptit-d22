"""Node 3 VAT report writer: Node 2 results -> one grounded Vietnamese report.

The system prompt is the frozen W3 v2 prompt. The user message carries each
question, its SQL, status and at most ``rows_per_question`` result rows, and
tells the model to use only those numbers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.pipeline.text_to_sql import (
    REPO_ROOT,
    SEED,
    TEMPERATURE,
    TRANSPORT_ERRORS,
    Node2Result,
    response_content,
    transport_error_text,
    usage_tokens,
)

W3_SYSTEM_PROMPT_PATH = REPO_ROOT / "experiments/configs/w3_v2/system_prompt.md"
NODE3_USER_TEMPLATE_PATH = REPO_ROOT / "src/pipeline/prompts/node3_report_user.md"

DATA_STATUSES = ("ok", "empty")


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.replace("|", "\\|")


def markdown_table(columns: list[str], rows: list[list], limit: int = 20) -> str:
    lines = [
        "| " + " | ".join(_cell(c) for c in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    if len(rows) > limit:
        lines.append(f"… (còn {len(rows) - limit} dòng)")
    return "\n".join(lines)


def _question_block(r: Node2Result, rows_per_question: int) -> str:
    sql = f"```sql\n{r.sql}\n```" if r.sql else "SQL: không có"
    parts = [f"### {r.question_id}. {r.question}", sql, f"Trạng thái: {r.status}"]
    if r.status == "ok":
        parts.append(markdown_table(r.columns, r.rows, rows_per_question))
    elif r.status == "empty":
        parts.append("KHÔNG CÓ DỮ LIỆU")
    else:
        parts.append(f"LỖI TRUY VẤN: {r.status}")
    return "\n".join(parts)


def build_user_message(results: list[Node2Result], invoice_count: int, rows_per_question: int = 20) -> str:
    template = NODE3_USER_TEMPLATE_PATH.read_text(encoding="utf-8")
    blocks = "\n\n".join(_question_block(r, rows_per_question) for r in results)
    return template.replace("{invoice_count}", str(invoice_count)).replace("{question_blocks}", blocks).rstrip("\n")


@dataclass
class Node3Result:
    status: str
    report_text: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    input_question_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Node3Writer:
    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        *,
        max_tokens: int = 768,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._clock = clock

    def write(self, results: list[Node2Result], invoice_count: int) -> Node3Result:
        out = Node3Result(status="ok", input_question_ids=[r.question_id for r in results])
        if not any(r.status in DATA_STATUSES for r in results):
            out.status = "skipped_no_data"
            return out
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_message(results, invoice_count)},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": self.max_tokens,
            "seed": SEED,
        }
        start = self._clock()
        try:
            response = self.client.chat_completions(payload)
        except TRANSPORT_ERRORS as exc:
            out.status = "transport_error"
            out.error = transport_error_text(exc)
            return out
        out.latency_ms = round((self._clock() - start) * 1000.0, 3)
        out.report_text, out.finish_reason = response_content(response)
        out.prompt_tokens, out.completion_tokens = usage_tokens(response)
        if not out.report_text.strip():
            out.status = "empty_response"
        return out


def render_report_md(result: Node3Result, node2_results: list[Node2Result], meta: dict) -> str:
    lines = [
        "# Báo cáo tổng hợp VAT (Node 3)",
        "",
        f"- run_id: `{meta.get('run_id')}` · lane: `{meta.get('lane')}` · model: `{meta.get('model')}`",
        f"- Node 3 status: `{result.status}` · finish_reason: `{result.finish_reason}`",
        "",
        result.report_text or "_Không có báo cáo._",
        "",
        "## Phụ lục: kết quả Q1–Q6",
    ]
    for r in node2_results:
        lines += ["", _question_block(r, rows_per_question=20)]
    return "\n".join(lines) + "\n"
