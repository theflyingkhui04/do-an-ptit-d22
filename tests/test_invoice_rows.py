"""[M5-01 web demo] Validated Node 1 payload -> table rows, shared by the SQLite and Postgres writers."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import init_invoice_sqlite as loader

from src.pipeline import invoice_rows as rows_mod

EXAMPLE_JSON = REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
SCHEMA_SQL = REPO_ROOT / "src/pipeline/sqlite/schema.sql"


def money(raw: str, value: float) -> dict:
    return {"raw": raw, "value": value, "currency": "VND"}


def payload_with_items_and_tax() -> dict:
    payload = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    payload["items"] = [
        {"line_number": 1, "description": "Sữa tươi", "sku": "8934673",
         "quantity": {"raw": "2", "value": 2, "unit": "hộp"},
         "unit_price": money("10.000", 10000), "line_total": money("20.000", 20000), "vat_rate": None},
        {"line_number": 2, "description": "Bánh mì", "sku": None,
         "quantity": {"raw": None, "value": None, "unit": None},
         "unit_price": money("22.000", 22000), "line_total": money("22.000", 22000), "vat_rate": 10},
    ]
    payload["taxes"] = [{"tax_name": "VAT", "tax_rate": 10, "taxable_amount": money("20.000", 20000),
                         "tax_amount": money("2.000", 2000)}]
    return payload


def table_columns(table: str) -> list[str]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_rows_for_two_items_and_one_tax_line():
    payload = payload_with_items_and_tax()
    rows = rows_mod.build_invoice_rows(payload, "r1", latency_ms=12.5)
    invoice_id = f"r1:{payload['source']['sample_id']}"

    assert rows.invoice["invoice_id"] == invoice_id
    assert rows.invoice["latency_ms"] == 12.5
    assert json.loads(rows.invoice["raw_json"]) == payload
    assert rows.invoice["total_amount_value"] == payload["totals"]["total_amount"]["value"]
    assert [r["line_number"] for r in rows.items] == [1, 2]
    assert rows.items[1]["quantity_value"] is None and rows.items[1]["vat_rate"] == 10
    assert rows.taxes == [{"invoice_id": invoice_id, "tax_name": "VAT", "tax_rate": 10,
                           "taxable_amount_raw": "20.000", "taxable_amount_value": 20000,
                           "taxable_amount_currency": "VND", "tax_amount_raw": "2.000",
                           "tax_amount_value": 2000, "tax_amount_currency": "VND"}]
    assert len(rows.evidence) == len(payload["extraction"]["field_evidence"])


def test_column_lists_exist_in_the_sqlite_schema():
    for table, columns in (("invoices", rows_mod.INVOICE_COLUMNS), ("invoice_items", rows_mod.ITEM_COLUMNS),
                           ("invoice_taxes", rows_mod.TAX_COLUMNS), ("field_evidence", rows_mod.EVIDENCE_COLUMNS)):
        assert set(columns) <= set(table_columns(table)), table


def test_missing_source_ids_are_refused():
    payload = payload_with_items_and_tax()
    payload["source"]["sample_id"] = None
    payload["source"]["img_id"] = None
    with pytest.raises(ValueError, match="sample_id"):
        rows_mod.build_invoice_rows(payload, "r1")


def test_sqlite_loader_reuses_the_shared_rows():
    assert loader.make_invoice_id is rows_mod.make_invoice_id
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    loader.init_schema(conn, SCHEMA_SQL)
    payload = payload_with_items_and_tax()
    invoice_id = loader.insert_extraction(conn, payload, "r1", replace=False, latency_ms=12.5)
    expected = rows_mod.build_invoice_rows(payload, "r1", latency_ms=12.5)
    stored = dict(zip(rows_mod.INVOICE_COLUMNS, conn.execute(
        f"SELECT {', '.join(rows_mod.INVOICE_COLUMNS)} FROM invoices WHERE invoice_id = ?", (invoice_id,)).fetchone()))
    assert stored == expected.invoice
