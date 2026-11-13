"""Free-form Vietnamese questions: grounded Node 2 SQL, checked SQL, short answer without invented numbers."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
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
NODE2_PROMPT_VERSION = "node2_text_to_sql_demo.v3"
ANSWER_PROMPT_VERSION = "node3_answer_demo.v1"
VIEWS = ("v_invoice_facts", "v_item_facts", "v_tax_facts")  # mirrored from SQLite v2
# Postgres demo only (src/pipeline/postgres/schema.sql): how and when an invoice entered the demo.
DEMO_VIEW_COLUMNS = {"v_upload_facts": ("invoice_id", "source", "uploaded_at", "uploaded_date")}
VIETNAM = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh, no daylight saving
ROW_LIMIT = 200
ANSWER_TABLE_ROWS = 50
NO_DATA_ANSWER = "Không có dữ liệu khớp câu hỏi."
NODE2_MAX_TOKENS = 256
ANSWER_MAX_TOKENS = 256

ALLOWED_FUNCTIONS = frozenset({
    # sqlglot models these operators as function nodes too
    "AND", "OR", "XOR", "EXISTS", "ARRAY",
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
    """Columns of the three mirrored views (read from the SQLite v2 schema) plus the demo-only view."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQLITE_SCHEMA.read_text(encoding="utf-8"))
    mirrored = {v: tuple(row[1] for row in conn.execute(f"PRAGMA table_info({v})")) for v in VIEWS}
    return {**mirrored, **DEMO_VIEW_COLUMNS}


def vietnam_today(now: datetime | None = None) -> str:
    """Today's calendar date in Vietnam as ``YYYY-MM-DD``."""
    return (now or datetime.now(timezone.utc)).astimezone(VIETNAM).strftime("%Y-%m-%d")


def build_node2_prompt(categorical: dict[str, list[Any]], today: str | None = None) -> str:
    views = "\n".join(f"- {view}({', '.join(columns)})" for view, columns in view_columns().items())
    lines = []
    for column, values in categorical.items():
        shown = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values) or "(no value yet)"
        lines.append(f"- {column}: {shown}")
    template = NODE2_PROMPT.read_text(encoding="utf-8")
    filled = template.replace("{{VIEW_COLUMNS}}", views).replace("{{CATEGORICAL_VALUES}}", "\n".join(lines))
    return filled.replace("{{TODAY}}", today or vietnam_today()).strip()


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

    _check_fanout(tree, columns)

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        value = limit.expression if isinstance(limit, exp.Limit) else limit
        current = int(value.name) if isinstance(value, exp.Literal) and value.is_int else None
    if current is None or current > ROW_LIMIT:
        tree = tree.limit(ROW_LIMIT)
    return tree.sql(dialect="postgres")


FANOUT_VIEWS = ("v_item_facts", "v_tax_facts")


def _check_fanout(tree: exp.Expression, columns: dict[str, tuple[str, ...]]) -> None:
    """[M5-01 canonical flow] Reject SUM/AVG/COUNT of an invoice-level column in a SELECT that joins
    v_invoice_facts with item or tax rows: each invoice would be counted once per item/tax row."""
    invoice_only = set(columns["v_invoice_facts"]).difference(*(columns[v] for v in FANOUT_VIEWS))
    for select in tree.find_all(exp.Select):
        tables = {t.name.lower() for t in select.find_all(exp.Table) if t.find_ancestor(exp.Select) is select}
        joined = tables.intersection(FANOUT_VIEWS)
        if "v_invoice_facts" not in tables or not joined:
            continue
        for agg in select.find_all(exp.Sum, exp.Avg, exp.Count):
            if agg.find_ancestor(exp.Select) is not select:
                continue
            if isinstance(agg, exp.Count) and isinstance(agg.this, exp.Distinct):
                continue
            hit = sorted({c.name.lower() for c in agg.find_all(exp.Column)} & invoice_only)
            if hit:
                raise SqlCheckError(
                    "JOIN_FANOUT",
                    f"{agg.sql(dialect='postgres')} adds invoice-level column {', '.join(hit)} once per row of "
                    f"{', '.join(sorted(joined))}; aggregate v_invoice_facts without that join (use EXISTS to filter)")


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


def ungrounded_numbers(text: str, rows: list[list[Any]], question: str) -> list[str]:
    """Numbers of ``text`` that match no value of ``rows``, the row count or ``question`` (same rule as the answer check)."""
    allowed = _allowed_numbers(rows, question)
    return [token for token in _NUMBER.findall(text)
            if not any(abs(c - a) <= 0.5 for c in _candidates(token) for a in allowed)]


class QuestionService:
    """Node 2 (question -> checked SQL -> rows) and the short answer, on the endpoint chosen for each call."""

    def __init__(self, store: Store, *, redact: Callable[[str], str] = lambda s: s,
                 today: Callable[[], str] = vietnam_today) -> None:
        self.store = store
        self._today = today
        self._redact = redact

    def _call(self, caller: Any, step: str, messages: list[dict], max_tokens: int, calls: list[dict]) -> str:
        started = time.perf_counter()
        response = caller.chat_completions({"messages": messages, "temperature": 0.0, "seed": 42,
                                            "max_tokens": max_tokens})
        metrics = caller.last_metrics.as_dict() if getattr(caller, "last_metrics", None) else {}
        calls.append({"step": step, "endpoint": caller.lane, "wall_ms": (time.perf_counter() - started) * 1000,
                      **metrics})
        return (response["choices"][0]["message"]["content"] or "").strip()

    def _execute(self, raw_sql: str) -> tuple[str, list[str], list[list[Any]]]:
        sql = check_demo_sql(raw_sql)
        columns, rows = self.store.run_readonly(sql)
        return sql, columns, rows

    def query(self, question: str, caller: Any) -> dict[str, Any]:
        """Node 2 only: one checked read-only SQL (one repair allowed) and its rows."""
        result: dict[str, Any] = {"question": question, "status": "error", "sql": None, "columns": [], "rows": [],
                                  "row_count": 0, "answer": None, "withheld_answer": None, "warning": None,
                                  "error": None, "repairs": 0, "first_error": None, "calls": [],
                                  "prompt_versions": {"node2": NODE2_PROMPT_VERSION, "answer": ANSWER_PROMPT_VERSION}}
        calls = result["calls"]
        system = build_node2_prompt(self.store.categorical_values(), self._today())
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": question}]
        try:
            raw_sql = self._call(caller, "node2", messages, NODE2_MAX_TOKENS, calls)
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
                raw_sql = self._call(caller, "node2", repair, NODE2_MAX_TOKENS, calls)
                sql, columns, rows = self._execute(raw_sql)
            except Exception as second:  # noqa: BLE001  (service boundary: reported, never raised to the user)
                result["sql"] = raw_sql
                result["error"] = self._redact(str(second))[:400]
                return result
        result.update(sql=sql, columns=columns, rows=rows, row_count=len(rows))
        if not rows or all(cell is None for row in rows for cell in row):  # nothing to answer from
            result["status"] = "no_data"
        else:
            result["status"] = "ok"
        return result

    def answer(self, question: str, caller: Any, *, answer_caller: Any = None) -> dict[str, Any]:
        """Node 2, then a short answer that may only use numbers of the result rows."""
        result = self.query(question, caller)
        if result["status"] == "no_data":
            result["answer"] = NO_DATA_ANSWER
        if result["status"] != "ok":
            return result
        columns, rows = result["columns"], result["rows"]
        answer_messages = [
            {"role": "system", "content": ANSWER_PROMPT.read_text(encoding="utf-8").strip()},
            {"role": "user", "content": f"Câu hỏi: {question}\nKết quả ({len(rows)} dòng):\n"
                                        f"{_table_text(columns, rows)}"}]
        try:
            text = self._call(answer_caller or caller, "answer", answer_messages, ANSWER_MAX_TOKENS, result["calls"])
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            result.update(warning="answer_failed", error=self._redact(str(exc))[:400])
            return result
        if text and numbers_are_grounded(text, rows, question):
            result["answer"] = text
        else:
            result["warning"] = "fabricated_number" if text else "empty_answer"
            result["withheld_answer"] = text or None  # audit only; the web never shows it
        return result
