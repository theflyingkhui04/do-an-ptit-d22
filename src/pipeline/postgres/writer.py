"""[M5-01 web demo] Postgres writer for the demo database (schema ``demo``).

Same rows as the SQLite loader (``src/pipeline/invoice_rows.py``); the payload
is validated against ``invoice_extraction.v1`` before anything is written, and
every invoice is written in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.invoice_rows import (
    EVIDENCE_COLUMNS,
    INVOICE_COLUMNS,
    ITEM_COLUMNS,
    JSON_COLUMNS,
    TAX_COLUMNS,
    build_invoice_rows,
)
from src.pipeline.validation import DEFAULT_JSON_SCHEMA_PATH, validate_payload

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
ROLES_PATH = Path(__file__).with_name("roles.sql")
DEMO_INVOICE_COLUMNS: tuple[str, ...] = ("source", "storage_path")
SOURCES = ("devset", "upload")
VIEWS = ("v_invoice_facts", "v_item_facts", "v_tax_facts")


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    lane: str | None
    model_name: str | None
    prompt_version: str
    notes: str


def insert_sql(table: str, columns: tuple[str, ...]) -> str:
    """Parameterized INSERT; JSON text columns are cast to jsonb."""
    values = ", ".join("%s::jsonb" if column in JSON_COLUMNS else "%s" for column in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values})"


def apply_schema(conn: Any, *, roles: bool = False) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    if roles:
        conn.execute(ROLES_PATH.read_text(encoding="utf-8"))


def insert_extraction(
    conn: Any,
    payload: dict,
    run: RunInfo,
    *,
    source: str,
    storage_path: str | None = None,
    latency_ms: float | None = None,
    if_exists: str = "error",
    json_schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
) -> tuple[str, bool]:
    """Write one validated payload; return ``(invoice_id, written)``.

    ``if_exists="skip"`` leaves an existing invoice untouched (idempotent seeding);
    ``"error"`` refuses to overwrite it. Invalid payloads raise before any write.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if if_exists not in ("error", "skip"):
        raise ValueError("if_exists must be 'error' or 'skip'")
    validate_payload(payload, json_schema_path)
    rows = build_invoice_rows(payload, run.run_id, latency_ms=latency_ms)
    invoice_id = rows.invoice["invoice_id"]
    with conn.transaction():
        conn.execute(
            "INSERT INTO extraction_runs (run_id, lane, model_name, prompt_version, schema_version, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
            (run.run_id, run.lane, run.model_name, run.prompt_version, payload["schema_version"], run.notes))
        stored_version = conn.execute(
            "SELECT schema_version FROM extraction_runs WHERE run_id = %s", (run.run_id,)).fetchone()[0]
        if stored_version != payload["schema_version"]:
            raise ValueError(f"run_id={run.run_id!r} uses schema_version={stored_version!r}")
        if conn.execute("SELECT 1 FROM invoices WHERE invoice_id = %s", (invoice_id,)).fetchone():
            if if_exists == "skip":
                return invoice_id, False
            raise ValueError(f"invoice {invoice_id!r} is already stored")
        invoice = {**rows.invoice, "source": source, "storage_path": storage_path}
        columns = INVOICE_COLUMNS + DEMO_INVOICE_COLUMNS
        conn.execute(insert_sql("invoices", columns), tuple(invoice[c] for c in columns))
        for table, table_columns, table_rows in (("invoice_items", ITEM_COLUMNS, rows.items),
                                                 ("invoice_taxes", TAX_COLUMNS, rows.taxes),
                                                 ("field_evidence", EVIDENCE_COLUMNS, rows.evidence)):
            for row in table_rows:
                conn.execute(insert_sql(table, table_columns), tuple(row[c] for c in table_columns))
    return invoice_id, True
