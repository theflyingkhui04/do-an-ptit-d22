"""Free-form Vietnamese questions: grounded Node 2 SQL, checked SQL, short answer without invented numbers."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from src.api.store import Store
from src.pipeline.sql_guard import clean_sql

REPO_ROOT = Path(__file__).resolve().parents[2]
SQLITE_SCHEMA = REPO_ROOT / "src/pipeline/sqlite/schema.sql"
NODE2_PROMPT = REPO_ROOT / "src/pipeline/prompts/node2_text_to_sql_demo.md"
ANSWER_PROMPT = REPO_ROOT / "src/pipeline/prompts/node3_answer_demo.md"
NODE2_PROMPT_VERSION = "node2_text_to_sql_demo.v1"
ANSWER_PROMPT_VERSION = "node3_answer_demo.v1"
VIEWS = ("v_invoice_facts", "v_item_facts", "v_tax_facts")
ROW_LIMIT = 200
ANSWER_TABLE_ROWS = 50
NO_DATA_ANSWER = "Không có dữ liệu khớp câu hỏi."
NODE2_MAX_TOKENS = 256
ANSWER_MAX_TOKENS = 256

ALLOWED_FUNCTIONS = frozenset({
    "ABS", "ARRAY_AGG", "AVG", "CASE", "CAST", "CEIL", "COALESCE", "CONCAT", "COUNT", "COUNT_IF",
    "CURRENT_DATE", "DATE_TRUNC", "EXTRACT", "FLOOR", "GREATEST", "GROUP_CONCAT", "IF", "LEAST", "LEFT",
    "LENGTH", "LOGICAL_AND", "LOGICAL_OR", "LOWER", "MAX", "MEDIAN", "MIN", "NULLIF", "PERCENTILE_CONT",
    "REPLACE", "RIGHT", "ROUND", "SPLIT_PART", "STDDEV", "STR_POSITION", "STR_TO_DATE", "SUBSTRING", "SUM",
    "TIMESTAMP_TRUNC", "TIME_TO_STR", "TRIM", "TRY_CAST", "TS_OR_DS_TO_DATE", "UPPER", "VARIANCE",
})
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
                   exp.Merge, exp.Copy, exp.Set, exp.TruncateTable, exp.Into, exp.Lock)


class SqlCheckError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@lru_cache(maxsize=1)
def view_columns() -> dict[str, tuple[str, ...]]:
    """Columns of the three views, read from the SQLite v2 schema (the Postgres mirror is tested equal)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQLITE_SCHEMA.read_text(encoding="utf-8"))
    return {v: tuple(row[1] for row in conn.execute(f"PRAGMA table_info({v})")) for v in VIEWS}


def build_node2_prompt(categorical: dict[str, list[Any]]) -> str:
    views = "\n".join(f"- {view}({', '.join(columns)})" for view, columns in view_columns().items())
    lines = []
    for column, values in categorical.items():
        shown = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values) or "(no value yet)"
        lines.append(f"- {column}: {shown}")
    template = NODE2_PROMPT.read_text(encoding="utf-8")
    return template.replace("{{VIEW_COLUMNS}}", views).replace("{{CATEGORICAL_VALUES}}", "\n".join(lines)).strip()


def _function_name(node: exp.Func) -> str:
    return node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name()


def check_demo_sql(raw: str) -> str:
    """Return executable SQL (one SELECT over the views, ``LIMIT`` ≤ 200) or raise ``SqlCheckError``."""
    sql = clean_sql(raw or "")
    if not sql:
        raise SqlCheckError("EMPTY", "no SQL statement")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SqlCheckError("PARSE_ERROR", str(exc).splitlines()[0][:200]) from exc
    if len(statements) != 1:
        raise SqlCheckError("MULTIPLE_STATEMENTS", "only one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SqlCheckError("NOT_SELECT", f"statement is {type(tree).__name__.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES) or (isinstance(node, exp.Select) and node.args.get("locks")):
            raise SqlCheckError("FORBIDDEN_CLAUSE", type(node).__name__.upper())

    for func in tree.find_all(exp.Func):
        if _function_name(func) not in ALLOWED_FUNCTIONS:
            raise SqlCheckError("FORBIDDEN_FUNCTION", _function_name(func).lower())

    columns = view_columns()
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    referenced: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if table.db or table.catalog or (name not in columns and name not in cte_names):
            raise SqlCheckError("FORBIDDEN_RELATION", ".".join(p for p in (table.db, table.name) if p))
        if name in columns:
            referenced.add(name)

    known = {c for view in referenced for c in columns[view]}
    known |= {alias.alias.lower() for alias in tree.find_all(exp.Alias) if alias.alias}
    known |= {c.lower() for cte in tree.find_all(exp.CTE) for c in cte.this.named_selects}
    for table_alias in tree.find_all(exp.TableAlias):
        known |= {c.name.lower() for c in table_alias.columns}
    for column in tree.find_all(exp.Column):
        if column.name and column.name.lower() not in known:
            raise SqlCheckError("UNKNOWN_COLUMN", column.sql(dialect="postgres"))

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        value = limit.expression if isinstance(limit, exp.Limit) else limit
        current = int(value.name) if isinstance(value, exp.Literal) and value.is_int else None
    if current is None or current > ROW_LIMIT:
        tree = tree.limit(ROW_LIMIT)
    return tree.sql(dialect="postgres")


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _candidates(token: str) -> set[float]:
    values = {float(token.replace(".", "").replace(",", ""))}
    for decimal, thousands in ((",", "."), (".", ",")):
        if decimal in token:
            head, _, tail = token.rpartition(decimal)
            try:
                values.add(float(head.replace(thousands, "").replace(decimal, "") + "." + tail))
            except ValueError:
                pass
    return values


def _allowed_numbers(rows: list[list[Any]], question: str) -> set[float]:
    allowed = {float(len(rows))}
    texts = [question]
    for row in rows:
        for cell in row:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float)):
                allowed.add(float(cell))
            elif cell is not None:
                texts.append(str(cell))
    for text in texts:
        for token in _NUMBER.findall(text):
            allowed |= _candidates(token)
            allowed |= {float(part) for part in re.split(r"[.,]", token) if part}
    return allowed


def numbers_are_grounded(answer: str, rows: list[list[Any]], question: str) -> bool:
    """Every number in ``answer`` equals (within 0.5, i.e. rounding to an integer) a value of the
    result rows, the row count, or a number of the question."""
    allowed = _allowed_numbers(rows, question)
    for token in _NUMBER.findall(answer):
        if not any(abs(c - a) <= 0.5 for c in _candidates(token) for a in allowed):
            return False
    return True


def _table_text(columns: list[str], rows: list[list[Any]]) -> str:
    lines = [" | ".join(columns)]
    for row in rows[:ANSWER_TABLE_ROWS]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


class QuestionService:
    def __init__(self, store: Store, *, lane_caller: Callable[[str], Any], lane_map: dict[str, str],
                 on_service: Callable[[str, bool, str | None], None] | None = None,
                 redact: Callable[[str], str] = lambda s: s) -> None:
        self.store = store
        self.lane_caller = lane_caller
        self.lane_map = lane_map
        self._on_service = on_service or (lambda name, ok, message: None)
        self._redact = redact

    def _call(self, step: str, messages: list[dict], max_tokens: int, calls: list[dict]) -> str:
        lane = self.lane_map["node2" if step == "node2" else "answer"]
        caller = self.lane_caller(lane)
        started = time.perf_counter()
        try:
            response = caller.chat_completions({"messages": messages, "temperature": 0.0, "seed": 42,
                                                "max_tokens": max_tokens})
        except Exception as exc:
            self._on_service(lane, False, str(exc))
            raise
        self._on_service(lane, True, None)
        metrics = caller.last_metrics.as_dict() if getattr(caller, "last_metrics", None) else {}
        calls.append({"step": step, "lane": lane, "wall_ms": (time.perf_counter() - started) * 1000, **metrics})
        return (response["choices"][0]["message"]["content"] or "").strip()

    def _execute(self, raw_sql: str) -> tuple[str, list[str], list[list[Any]]]:
        sql = check_demo_sql(raw_sql)
        columns, rows = self.store.run_readonly(sql)
        return sql, columns, rows

    def answer(self, question: str) -> dict[str, Any]:
        result: dict[str, Any] = {"question": question, "status": "error", "sql": None, "columns": [], "rows": [],
                                  "row_count": 0, "answer": None, "warning": None, "error": None, "repairs": 0,
                                  "first_error": None, "calls": [],
                                  "prompt_versions": {"node2": NODE2_PROMPT_VERSION, "answer": ANSWER_PROMPT_VERSION}}
        calls = result["calls"]
        messages = [{"role": "system", "content": build_node2_prompt(self.store.categorical_values())},
                    {"role": "user", "content": question}]
        try:
            raw_sql = self._call("node2", messages, NODE2_MAX_TOKENS, calls)
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            result["error"] = self._redact(f"node2 call failed: {exc}")[:400]
            return result
        try:
            sql, columns, rows = self._execute(raw_sql)
        except Exception as first:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            result["first_error"] = self._redact(str(first))[:400]
            result["repairs"] = 1
            repair = messages + [{"role": "assistant", "content": raw_sql},
                                 {"role": "user", "content": f"Câu SQL trên bị lỗi: {result['first_error']}. "
                                                             "Hãy viết lại đúng một câu SELECT, chỉ trả về SQL."}]
            try:
                raw_sql = self._call("node2", repair, NODE2_MAX_TOKENS, calls)
                sql, columns, rows = self._execute(raw_sql)
            except Exception as second:  # noqa: BLE001  (service boundary: reported, never raised to the user)
                result["sql"] = raw_sql
                result["error"] = self._redact(str(second))[:400]
                return result
        result.update(sql=sql, columns=columns, rows=rows, row_count=len(rows))
        if not rows:
            result.update(status="no_data", answer=NO_DATA_ANSWER)
            return result
        answer_messages = [
            {"role": "system", "content": ANSWER_PROMPT.read_text(encoding="utf-8").strip()},
            {"role": "user", "content": f"Câu hỏi: {question}\nKết quả ({len(rows)} dòng):\n"
                                        f"{_table_text(columns, rows)}"}]
        try:
            text = self._call("answer", answer_messages, ANSWER_MAX_TOKENS, calls)
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            result.update(status="ok", warning="answer_failed", error=self._redact(str(exc))[:400])
            return result
        result["status"] = "ok"
        if text and numbers_are_grounded(text, rows, question):
            result["answer"] = text
        else:
            result["warning"] = "fabricated_number" if text else "empty_answer"
        return result
