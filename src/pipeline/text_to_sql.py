"""Node 2 Text-to-SQL runner: question -> SQL -> read-only guard -> SQLite rows.

One LLM request per question, no retry on failure (keeps request counts equal
across serving lanes). The system prompt is the frozen W2 v2 prompt, so Node 2
returns SQL only; ``views_used`` is derived from the SQL and ``explanation`` is
always ``None``.
"""

from __future__ import annotations

import http.client
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.pipeline.node2_questions import Node2Question
from src.pipeline.openai_compatible import OpenAICompatibleError
from src.pipeline.sql_guard import SqlGuardError, check_read_only, clean_sql

REPO_ROOT = Path(__file__).resolve().parents[2]
W2_SYSTEM_PROMPT_PATH = REPO_ROOT / "experiments/configs/w2_v2/system_prompt.md"

TEMPERATURE = 0.0
SEED = 42

# OpenAICompatibleClient wraps only HTTPError/URLError; read timeouts and dropped
# connections surface as raw OSError / HTTPException and must not crash the run.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    OpenAICompatibleError,
    OSError,
    http.client.HTTPException,
)


def transport_error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def open_read_only(db_path: Path) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def response_content(response: dict[str, Any]) -> tuple[str, str | None]:
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return content, choice.get("finish_reason")


def usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


@dataclass
class Node2Result:
    question_id: str
    question: str
    raw_response: str | None = None
    sql: str | None = None
    views_used: list[str] = field(default_factory=list)
    explanation: None = None
    status: str = "ok"
    error: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    row_count_truncated: bool = False
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Node2Runner:
    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        db_path: Path,
        *,
        max_tokens: int = 256,
        row_limit: int = 200,
        query_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.db_path = Path(db_path)
        self.max_tokens = max_tokens
        self.row_limit = row_limit
        self.query_timeout_s = query_timeout_s
        self._clock = clock

    def _payload(self, question: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": self.max_tokens,
            "seed": SEED,
        }

    def _execute(self, sql: str, result: Node2Result) -> None:
        conn = open_read_only(self.db_path)
        deadline = self._clock() + self.query_timeout_s
        conn.set_progress_handler(lambda: 1 if self._clock() > deadline else 0, 10_000)
        try:
            cursor = conn.execute(sql)
            fetched = cursor.fetchmany(self.row_limit + 1)
            result.columns = [col[0] for col in cursor.description or []]
        except sqlite3.OperationalError as exc:
            result.status = "timeout" if "interrupted" in str(exc).lower() else "sql_error"
            result.error = str(exc)
            return
        except sqlite3.Error as exc:
            result.status = "sql_error"
            result.error = str(exc)
            return
        finally:
            conn.close()
        result.row_count_truncated = len(fetched) > self.row_limit
        result.rows = [list(row) for row in fetched[: self.row_limit]]
        result.row_count = len(result.rows)
        # An aggregate over zero matching rows returns one all-NULL row; treat
        # it as "empty" so Node 3 reports missing data instead of "NULL".
        has_value = any(v is not None for row in result.rows for v in row)
        result.status = "ok" if has_value else "empty"

    def run_question(self, q: Node2Question) -> Node2Result:
        result = Node2Result(question_id=q.id, question=q.question)
        start = self._clock()
        try:
            response = self.client.chat_completions(self._payload(q.question))
        except TRANSPORT_ERRORS as exc:
            result.status = "transport_error"
            result.error = transport_error_text(exc)
            return result
        result.latency_ms = round((self._clock() - start) * 1000.0, 3)
        content, result.finish_reason = response_content(response)
        result.prompt_tokens, result.completion_tokens = usage_tokens(response)
        result.raw_response = content
        result.sql = clean_sql(content)
        try:
            result.views_used = list(check_read_only(result.sql))
        except SqlGuardError as exc:
            result.status = "guard_rejected"
            result.error = str(exc)
            return result
        self._execute(result.sql, result)
        return result

    def run_all(self, questions: Iterable[Node2Question]) -> list[Node2Result]:
        return [self.run_question(q) for q in questions]
