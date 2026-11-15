"""Demo database access (Postgres schema ``demo``) behind a small protocol the API depends on."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from src.pipeline.postgres import writer

UNFINISHED = ("queued", "ocr", "node1", "saving")
ENDPOINT_COLUMNS = ("endpoint_id", "name", "base_url", "platform", "model_id", "max_model_len", "meta",
                    "is_baseline", "status", "status_message", "checked_at", "created_at")
ENDPOINT_FIELDS = frozenset(ENDPOINT_COLUMNS) | {"api_key"}
BENCH_COLUMNS = ("run_id", "kind", "status", "config", "progress", "result", "error", "created_at", "updated_at")
CATEGORICAL = {"v_invoice_facts": ("extraction_status", "document_type", "currency", "payment_method"),
               "v_tax_facts": ("tax_name",), "v_upload_facts": ("source",)}
TEXT_VALUE_COLUMNS = {"v_invoice_facts": ("seller_name", "buyer_name")}
MAX_ROWS = 200


class Store(Protocol):
    def ping(self) -> bool: ...
    def create_job(self, job_id: str) -> None: ...
    def update_job(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None,
                   invoice_id: str | None = None) -> None: ...
    def get_job(self, job_id: str) -> dict | None: ...
    def mark_unfinished_interrupted(self) -> int: ...
    def save_invoice(self, payload: dict, run: writer.RunInfo, *, storage_path: str | None,
                     latency_ms: float | None) -> str: ...
    def list_invoices(self, limit: int = MAX_ROWS) -> list[dict]: ...
    def get_invoice(self, invoice_id: str) -> dict | None: ...
    def categorical_values(self) -> dict[str, list[str]]: ...
    def text_values(self) -> dict[str, list[str]]: ...
    def run_readonly(self, sql: str) -> tuple[list[str], list[list[Any]]]: ...
    def list_endpoints(self) -> list[dict]: ...
    def get_endpoint(self, endpoint_id: str, *, with_key: bool = False) -> dict | None: ...
    def find_endpoint_by_url(self, base_url: str) -> dict | None: ...
    def insert_endpoint(self, row: dict) -> None: ...
    def update_endpoint(self, endpoint_id: str, fields: dict) -> None: ...
    def set_baseline(self, endpoint_id: str | None) -> None: ...
    def delete_endpoint(self, endpoint_id: str) -> bool: ...
    def create_bench(self, run_id: str, kind: str, config: dict) -> None: ...
    def update_bench(self, run_id: str, status: str, *, progress: dict | None = None, result: dict | None = None,
                     error: str | None = None) -> None: ...
    def get_bench(self, run_id: str) -> dict | None: ...
    def list_bench(self, limit: int = 20) -> list[dict]: ...
    def mark_bench_interrupted(self) -> int: ...


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


class PostgresStore:
    """One short connection per call; every call is one transaction with ``SET LOCAL`` settings,
    so it also works through a transaction-mode connection pooler."""

    def __init__(self, writer_url: str, reader_url: str | None = None) -> None:
        self._writer_url = writer_url
        self._reader_url = reader_url or writer_url

    @contextmanager
    def _tx(self, *, read_only: bool = False) -> Iterator[Any]:
        import psycopg

        url = self._reader_url if read_only else self._writer_url
        with psycopg.connect(url, autocommit=True, prepare_threshold=None, connect_timeout=10) as conn,                 conn.transaction():
            conn.execute("SET LOCAL search_path = demo")
            if read_only:
                conn.execute("SET LOCAL transaction_read_only = on")
                conn.execute("SET LOCAL statement_timeout = '5s'")
            yield conn

    def ping(self) -> bool:
        with self._tx() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1

    def create_job(self, job_id: str) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO demo_jobs (job_id, status) VALUES (%s, 'queued')", (job_id,))

    def update_job(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None,
                   invoice_id: str | None = None) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE demo_jobs SET status = %s, result = COALESCE(%s::jsonb, result), error = %s, "
                "invoice_id = COALESCE(%s, invoice_id), updated_at = now() WHERE job_id = %s",
                (status, None if result is None else json.dumps(result, ensure_ascii=False, default=str),
                 error, invoice_id, job_id))

    def get_job(self, job_id: str) -> dict | None:
        with self._tx() as conn:
            row = conn.execute("SELECT job_id, status, error, result, invoice_id, created_at, updated_at "
                               "FROM demo_jobs WHERE job_id = %s", (job_id,)).fetchone()
        if row is None:
            return None
        keys = ("job_id", "status", "error", "result", "invoice_id", "created_at", "updated_at")
        return {k: _plain(v) for k, v in zip(keys, row)}

    def mark_unfinished_interrupted(self) -> int:
        with self._tx() as conn:
            cur = conn.execute("UPDATE demo_jobs SET status = 'interrupted', error = 'backend restarted', "
                               "updated_at = now() WHERE status = ANY(%s)", (list(UNFINISHED),))
            return cur.rowcount

    def save_invoice(self, payload: dict, run: writer.RunInfo, *, storage_path: str | None,
                     latency_ms: float | None) -> str:
        with self._tx() as conn:
            invoice_id, _ = writer.insert_extraction(conn, payload, run, source="upload",
                                                     storage_path=storage_path, latency_ms=latency_ms)
        return invoice_id

    def list_invoices(self, limit: int = MAX_ROWS) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT i.invoice_id, i.seller_name, i.issued_date, i.total_amount_value, i.total_amount_currency, "
                "f.item_count, i.source, i.created_at, r.lane FROM invoices AS i "
                "JOIN v_invoice_facts AS f ON f.invoice_id = i.invoice_id "
                "JOIN extraction_runs AS r ON r.run_id = i.run_id "
                "ORDER BY (i.source = 'upload') DESC, i.created_at DESC, i.invoice_id LIMIT %s",
                (min(limit, MAX_ROWS),)).fetchall()
        keys = ("invoice_id", "seller_name", "issued_date", "total_amount_value", "total_amount_currency",
                "item_count", "source", "created_at", "lane")
        return [{k: _plain(v) for k, v in zip(keys, row)} for row in rows]

    def get_invoice(self, invoice_id: str) -> dict | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT i.invoice_id, i.raw_json, i.source, i.storage_path, i.created_at, i.latency_ms, r.lane, "
                "r.model_name, r.prompt_version FROM invoices AS i JOIN extraction_runs AS r ON r.run_id = i.run_id "
                "WHERE i.invoice_id = %s", (invoice_id,)).fetchone()
        if row is None:
            return None
        keys = ("invoice_id", "payload", "source", "storage_path", "created_at", "latency_ms", "lane",
                "model_name", "prompt_version")
        return {k: _plain(v) for k, v in zip(keys, row)}

    def text_values(self) -> dict[str, list[str]]:
        """Real names Node 2 text filters are matched against (``questions.relax_text_filters``)."""
        values: dict[str, list[str]] = {}
        with self._tx(read_only=True) as conn:
            for view, columns in TEXT_VALUE_COLUMNS.items():
                for column in columns:
                    rows = conn.execute(f"SELECT DISTINCT {column} FROM {view} WHERE {column} IS NOT NULL "
                                        "ORDER BY 1 LIMIT 500").fetchall()
                    values[column] = [r[0] for r in rows]
        return values

    def categorical_values(self) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        with self._tx() as conn:
            for view, columns in CATEGORICAL.items():
                for column in columns:
                    rows = conn.execute(f"SELECT DISTINCT {column} FROM {view} WHERE {column} IS NOT NULL "
                                        "ORDER BY 1 LIMIT 50").fetchall()
                    values[column] = [r[0] for r in rows]
        return values

    def run_readonly(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        with self._tx(read_only=True) as conn:
            cur = conn.execute(sql)
            columns = [d.name for d in cur.description or []]
            rows = [[_plain(v) for v in row] for row in cur.fetchmany(MAX_ROWS)]
        return columns, rows

    # ---------------------------------------------------------------- endpoints
    def _endpoint_row(self, row: tuple, *, with_key: bool) -> dict:
        data = {k: _plain(v) for k, v in zip(ENDPOINT_COLUMNS, row)}
        data["has_key"] = bool(row[len(ENDPOINT_COLUMNS)])
        if with_key:
            data["api_key"] = row[len(ENDPOINT_COLUMNS)]
        return data

    def _select_endpoints(self, where: str = "", params: tuple = ()) -> list[tuple]:
        with self._tx() as conn:
            return conn.execute(f"SELECT {', '.join(ENDPOINT_COLUMNS)}, api_key FROM endpoints {where} "
                                "ORDER BY created_at, endpoint_id", params).fetchall()

    def list_endpoints(self) -> list[dict]:
        return [self._endpoint_row(r, with_key=False) for r in self._select_endpoints()]

    def get_endpoint(self, endpoint_id: str, *, with_key: bool = False) -> dict | None:
        rows = self._select_endpoints("WHERE endpoint_id = %s", (endpoint_id,))
        return self._endpoint_row(rows[0], with_key=with_key) if rows else None

    def find_endpoint_by_url(self, base_url: str) -> dict | None:
        rows = self._select_endpoints("WHERE base_url = %s", (base_url,))
        return self._endpoint_row(rows[0], with_key=False) if rows else None

    @staticmethod
    def _db_value(key: str, value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False) if key == "meta" else value

    def insert_endpoint(self, row: dict) -> None:
        keys = [k for k in row if k in ENDPOINT_FIELDS]
        with self._tx() as conn:
            conn.execute(f"INSERT INTO endpoints ({', '.join(keys)}) VALUES ({', '.join(['%s'] * len(keys))})",
                         tuple(self._db_value(k, row[k]) for k in keys))

    def update_endpoint(self, endpoint_id: str, fields: dict) -> None:
        keys = [k for k in fields if k in ENDPOINT_FIELDS and k != "endpoint_id"]
        if not keys:
            return
        assignments = ", ".join(f"{k} = %s" for k in keys)
        with self._tx() as conn:
            conn.execute(f"UPDATE endpoints SET {assignments} WHERE endpoint_id = %s",
                         (*[self._db_value(k, fields[k]) for k in keys], endpoint_id))

    def set_baseline(self, endpoint_id: str | None) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE endpoints SET is_baseline = false WHERE is_baseline")
            if endpoint_id is not None:
                conn.execute("UPDATE endpoints SET is_baseline = true WHERE endpoint_id = %s", (endpoint_id,))

    def delete_endpoint(self, endpoint_id: str) -> bool:
        with self._tx() as conn:
            return conn.execute("DELETE FROM endpoints WHERE endpoint_id = %s", (endpoint_id,)).rowcount > 0

    # ---------------------------------------------------------------- measurement runs
    def create_bench(self, run_id: str, kind: str, config: dict) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO bench_runs (run_id, kind, status, config) VALUES (%s, %s, 'queued', %s)",
                         (run_id, kind, json.dumps(config, ensure_ascii=False, default=str)))

    def update_bench(self, run_id: str, status: str, *, progress: dict | None = None, result: dict | None = None,
                     error: str | None = None) -> None:
        def dump(value: dict | None) -> str | None:
            return None if value is None else json.dumps(value, ensure_ascii=False, default=str)

        with self._tx() as conn:
            conn.execute("UPDATE bench_runs SET status = %s, progress = COALESCE(%s::jsonb, progress), "
                         "result = COALESCE(%s::jsonb, result), error = %s, updated_at = now() WHERE run_id = %s",
                         (status, dump(progress), dump(result), error, run_id))

    def get_bench(self, run_id: str) -> dict | None:
        with self._tx() as conn:
            row = conn.execute(f"SELECT {', '.join(BENCH_COLUMNS)} FROM bench_runs WHERE run_id = %s",
                               (run_id,)).fetchone()
        return None if row is None else {k: _plain(v) for k, v in zip(BENCH_COLUMNS, row)}

    def list_bench(self, limit: int = 20) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(f"SELECT {', '.join(BENCH_COLUMNS)} FROM bench_runs ORDER BY created_at DESC "
                                "LIMIT %s", (limit,)).fetchall()
        return [{k: _plain(v) for k, v in zip(BENCH_COLUMNS, row)} for row in rows]

    def mark_bench_interrupted(self) -> int:
        with self._tx() as conn:
            return conn.execute("UPDATE bench_runs SET status = 'interrupted', error = 'backend restarted', "
                                "updated_at = now() WHERE status IN ('queued', 'running')").rowcount
