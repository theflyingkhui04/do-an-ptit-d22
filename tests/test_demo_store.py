"""[M5-01 web demo] Postgres demo store: schema mirror of SQLite v2, writer, read-only reader.

Pure tests always run. Database tests need a disposable Postgres:
``DEMO_TEST_DATABASE_URL`` (owner connection) and ``DEMO_TEST_ALLOW_RESET=1``,
because they drop and recreate schema ``demo``. Never point them at Supabase.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import init_invoice_sqlite as loader

from src.pipeline.invoice_rows import INVOICE_COLUMNS
from src.pipeline.postgres import writer
from tests.test_invoice_rows import payload_with_items_and_tax

SQLITE_SCHEMA = REPO_ROOT / "src/pipeline/sqlite/schema.sql"
SQLITE_SCHEMA_SHA256 = "0f502a5b110af84a2100870e3c2d4f5eff5b02a43377d64c405814bf735b43a9"  # LF bytes, user_version 2
TABLES = ("extraction_runs", "invoices", "invoice_items", "invoice_taxes", "field_evidence")
DB_URL = os.environ.get("DEMO_TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not (DB_URL and os.environ.get("DEMO_TEST_ALLOW_RESET") == "1"),
                              reason="needs a disposable Postgres (DEMO_TEST_DATABASE_URL + DEMO_TEST_ALLOW_RESET=1)")


def sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    loader.init_schema(conn, SQLITE_SCHEMA)
    return conn


def postgres_statements() -> list[exp.Expression]:
    text = writer.SCHEMA_PATH.read_text(encoding="utf-8")
    text = re.sub(r"--[^\n]*", "", text)
    return [sqlglot.parse_one(s, read="postgres") for s in text.split(";") if s.strip()]


def postgres_view_columns() -> dict[str, list[str]]:
    views = {}
    for statement in postgres_statements():
        if isinstance(statement, exp.Create) and statement.args.get("kind") == "VIEW":
            views[statement.this.name] = statement.expression.named_selects
    return views


def postgres_table_columns() -> dict[str, list[str]]:
    tables = {}
    for statement in postgres_statements():
        if isinstance(statement, exp.Create) and statement.args.get("kind") == "TABLE":
            schema = statement.this
            tables[schema.this.name] = [c.name for c in schema.expressions if isinstance(c, exp.ColumnDef)]
    return tables


# ---------------------------------------------------------------- pure checks
def test_view_columns_match_sqlite_v2():
    conn = sqlite_conn()
    pg = postgres_view_columns()
    assert set(pg) == set(writer.VIEWS) | set(writer.DEMO_VIEWS)
    assert pg["v_upload_facts"] == ["invoice_id", "source", "uploaded_at", "uploaded_date"]
    for view in writer.VIEWS:
        assert pg[view] == [row[1] for row in conn.execute(f"PRAGMA table_info({view})")], view


def test_tables_keep_every_sqlite_column_and_add_demo_columns():
    conn = sqlite_conn()
    pg = postgres_table_columns()
    for table in TABLES:
        sqlite_columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        assert pg[table][: len(sqlite_columns)] == sqlite_columns, table
    assert pg["invoices"][-2:] == list(writer.DEMO_INVOICE_COLUMNS)
    assert pg["demo_jobs"][:2] == ["job_id", "status"]


def test_benchmark_sqlite_schema_is_untouched():
    data = SQLITE_SCHEMA.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == SQLITE_SCHEMA_SHA256


def test_reader_role_gets_only_the_three_views():
    roles = writer.ROLES_PATH.read_text(encoding="utf-8")
    grants = [line for line in roles.splitlines() if line.startswith("GRANT") and "demo_reader" in line]
    assert grants == ["GRANT USAGE ON SCHEMA demo TO demo_reader;",
                      "GRANT SELECT ON demo.v_invoice_facts, demo.v_item_facts, demo.v_tax_facts, demo.v_upload_facts TO demo_reader;"]
    for setting in ("default_transaction_read_only = on", "statement_timeout = '5s'", "search_path = demo"):
        assert f"ALTER ROLE demo_reader SET {setting};" in roles
    assert "PASSWORD '" not in roles.replace("PASSWORD '...'", "")


def test_insert_sql_casts_json_columns():
    sql = writer.insert_sql("invoices", INVOICE_COLUMNS)
    assert sql.startswith("INSERT INTO invoices (invoice_id, run_id,")
    assert sql.count("%s::jsonb") == 3
    assert sql.count("%s") == len(INVOICE_COLUMNS)


def test_bad_source_or_payload_is_refused_before_any_write():
    class Untouched:
        def __getattr__(self, name):
            raise AssertionError("database must not be touched")

    run = writer.RunInfo("r1", "adaptive", "m", "node1_extractor.v2", "test")
    with pytest.raises(ValueError, match="source"):
        writer.insert_extraction(Untouched(), payload_with_items_and_tax(), run, source="web")
    payload = payload_with_items_and_tax()
    payload["totals"]["total_amount"]["value"] = -1
    with pytest.raises(ValueError, match="invoice_extraction.v1"):
        writer.insert_extraction(Untouched(), payload, run, source="upload")


# ---------------------------------------------------------------- database
@pytest.fixture
def pg():
    import psycopg

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS demo CASCADE")
        writer.apply_schema(conn, roles=True)
        conn.execute("SET search_path = demo")
        yield conn


def reader_connection():
    import psycopg
    from psycopg.conninfo import make_conninfo

    return psycopg.connect(make_conninfo(DB_URL, user="demo_reader"), autocommit=True)


RUN = writer.RunInfo("r1", "adaptive", "qwen2.5-7b-adaptive", "node1_extractor.v2", "test")


@needs_db
def test_same_view_rows_in_sqlite_and_postgres(pg):
    payload = payload_with_items_and_tax()
    lite = sqlite_conn()
    loader.insert_extraction(lite, payload, "r1", replace=False, latency_ms=12.5)
    invoice_id, written = writer.insert_extraction(pg, payload, RUN, source="upload", latency_ms=12.5)
    assert written and invoice_id == f"r1:{payload['source']['sample_id']}"
    for view in writer.VIEWS:
        order = "invoice_id" + (", line_number" if view == "v_item_facts" else "")
        assert [tuple(r) for r in pg.execute(f"SELECT * FROM {view} ORDER BY {order}").fetchall()] == \
            lite.execute(f"SELECT * FROM {view} ORDER BY {order}").fetchall(), view
    assert pg.execute("SELECT count(*) FROM v_item_facts").fetchone()[0] == 2
    assert pg.execute("SELECT count(*) FROM v_tax_facts").fetchone()[0] == 1
    assert pg.execute("SELECT raw_json FROM invoices").fetchone()[0] == json.loads(json.dumps(payload))


@needs_db
def test_invalid_payload_writes_nothing(pg):
    payload = payload_with_items_and_tax()
    payload["document_type"] = "receipt"
    with pytest.raises(ValueError):
        writer.insert_extraction(pg, payload, RUN, source="upload")
    assert pg.execute("SELECT count(*) FROM invoices").fetchone()[0] == 0


@needs_db
def test_existing_invoice_is_skipped_or_refused(pg):
    payload = payload_with_items_and_tax()
    assert writer.insert_extraction(pg, payload, RUN, source="devset")[1] is True
    assert writer.insert_extraction(pg, payload, RUN, source="devset", if_exists="skip")[1] is False
    with pytest.raises(ValueError, match="already stored"):
        writer.insert_extraction(pg, payload, RUN, source="devset")
    assert pg.execute("SELECT count(*) FROM invoice_items").fetchone()[0] == 2


@needs_db
def test_reader_cannot_write_or_read_base_tables(pg):
    import psycopg

    writer.insert_extraction(pg, payload_with_items_and_tax(), RUN, source="upload")
    with reader_connection() as reader:
        assert reader.execute("SELECT count(*) FROM v_invoice_facts").fetchone()[0] == 1
        for sql in ("DELETE FROM invoices", "SELECT raw_json FROM invoices",
                    "INSERT INTO extraction_runs (run_id) VALUES ('x')"):
            with pytest.raises(psycopg.Error):
                reader.execute(sql)
    assert pg.execute("SELECT count(*) FROM invoices").fetchone()[0] == 1


@needs_db
def test_reader_queries_time_out_after_five_seconds(pg):
    import psycopg

    with reader_connection() as reader, pytest.raises(psycopg.errors.QueryCanceled):
        reader.execute("SELECT pg_sleep(6)")


# ---------------------------------------------------------------- seed
class FakeStorage:
    def __init__(self) -> None:
        self.uploads: dict[str, tuple[int, str]] = {}

    def upload(self, path: str, data: bytes, content_type: str) -> None:
        self.uploads[path] = (len(data), content_type)


def test_seed_rows_point_to_devset_images():
    import seed_demo_db as seed

    from src.pipeline.text_normalize import has_vietnamese_marks

    assert seed.DEFAULT_VALIDATED == Path("experiments/runs/m5_01/m5_01_v3_adaptive/validated_outputs.jsonl")
    assert "W1_v3" in seed.SEED_NOTES and "m5_01_v3_adaptive" in seed.SEED_NOTES
    rows = seed.load_validated(REPO_ROOT / seed.DEFAULT_VALIDATED)
    assert len(rows) == 25 and {r["lane"] for r in rows} == {"adaptive"}
    for row in rows:
        assert (REPO_ROOT / row["payload"]["source"]["image_path"]).is_file()
        assert not has_vietnamese_marks(json.dumps(row["payload"], ensure_ascii=False))
    assert len(seed.load_validated(REPO_ROOT / seed.W1_V2_VALIDATED)) == 29  # old seed kept for the old question set
    assert seed.storage_path_for("data/processed/m0_04_mcocr_devset/images/a.jpg") == "devset/a.jpg"


@needs_db
def test_seed_is_idempotent_and_uploads_each_image(pg):
    import seed_demo_db as seed

    rows = seed.load_validated(REPO_ROOT / seed.DEFAULT_VALIDATED)
    storage = FakeStorage()
    first = seed.seed(pg, rows, storage=storage)
    assert first == {"rows": 25, "written": 25, "skipped": 0, "images_uploaded": 25}
    assert all(content_type == "image/jpeg" and size > 0 for size, content_type in storage.uploads.values())
    second = seed.seed(pg, rows)
    assert second["written"] == 0 and second["skipped"] == 25
    assert pg.execute("SELECT count(*) FROM invoices WHERE source = 'devset'").fetchone()[0] == 25
    assert pg.execute("SELECT count(*) FROM v_invoice_facts").fetchone()[0] == 25
    notes = pg.execute("SELECT DISTINCT r.notes FROM invoices i JOIN extraction_runs r USING (run_id)").fetchall()
    assert notes == [(seed.SEED_NOTES,)]
    assert pg.execute("SELECT count(*) FROM v_item_facts").fetchone()[0] > 0


# ---------------------------------------------------------------- API store
@needs_db
def test_postgres_store_jobs_invoices_and_readonly_queries(pg):
    import psycopg
    import seed_demo_db as seed
    from psycopg.conninfo import make_conninfo

    from src.api.store import PostgresStore

    seed.seed(pg, seed.load_validated(REPO_ROOT / seed.DEFAULT_VALIDATED))
    store = PostgresStore(DB_URL, make_conninfo(DB_URL, user="demo_reader"))
    assert store.ping() is True

    store.create_job("up_aaaaaaaaaaaa")
    store.update_job("up_aaaaaaaaaaaa", "node1", result={"ocr": {"text": "x"}})
    store.create_job("up_bbbbbbbbbbbb")
    payload = payload_with_items_and_tax()
    payload["source"].update(sample_id="up_bbbbbbbbbbbb", img_id="up_bbbbbbbbbbbb",
                             image_path="uploads/up_bbbbbbbbbbbb.jpg")
    run = writer.RunInfo("up_bbbbbbbbbbbb", "adaptive", "qwen2.5-7b-adaptive", "node1_extractor.v2", "web demo upload")
    invoice_id = store.save_invoice(payload, run, storage_path="uploads/up_bbbbbbbbbbbb.jpg", latency_ms=9.5)
    store.update_job("up_bbbbbbbbbbbb", "done", result={"invoice_id": invoice_id}, invoice_id=invoice_id)
    assert store.mark_unfinished_interrupted() == 1
    assert store.get_job("up_aaaaaaaaaaaa")["status"] == "interrupted"
    assert store.get_job("up_aaaaaaaaaaaa")["result"] == {"ocr": {"text": "x"}}
    assert store.get_job("up_bbbbbbbbbbbb")["invoice_id"] == invoice_id
    assert store.get_job("up_missing0000") is None

    listed = store.list_invoices()
    assert len(listed) == 26 and listed[0]["invoice_id"] == invoice_id and listed[0]["source"] == "upload"
    assert listed[0]["item_count"] == 2 and listed[0]["lane"] == "adaptive"
    detail = store.get_invoice(invoice_id)
    assert detail["payload"]["source"]["sample_id"] == "up_bbbbbbbbbbbb"
    assert detail["storage_path"] == "uploads/up_bbbbbbbbbbbb.jpg" and detail["prompt_version"] == "node1_extractor.v2"

    values = store.categorical_values()
    assert values["extraction_status"] == ["partial"] and "VND" in values["currency"]

    columns, rows = store.run_readonly("SELECT count(*) AS n FROM v_invoice_facts")
    assert columns == ["n"] and rows == [[26]]
    for bad in ("DELETE FROM invoices", "SELECT raw_json FROM invoices"):
        with pytest.raises(psycopg.Error):
            store.run_readonly(bad)
    assert store.run_readonly("SELECT count(*) FROM v_item_facts")[1][0][0] > 0


@needs_db
def test_upload_view_reports_source_and_vietnam_date(pg):
    payload = payload_with_items_and_tax()
    writer.insert_extraction(pg, payload, RUN, source="upload")
    row = pg.execute("SELECT source, uploaded_date, "
                     "to_char(uploaded_at AT TIME ZONE 'Asia/Ho_Chi_Minh', 'YYYY-MM-DD') FROM v_upload_facts").fetchone()
    assert row[0] == "upload" and row[1] == row[2]
    with reader_connection() as reader:
        assert reader.execute("SELECT COUNT(*) FROM v_upload_facts WHERE source = 'upload'").fetchone()[0] == 1


# ---------------------------------------------------------------- [M5-01 web console] endpoints and bench runs
@needs_db
def test_postgres_store_endpoints_keep_the_key_private(pg):
    from src.api.store import PostgresStore

    store = PostgresStore(DB_URL)
    row = {"endpoint_id": "ep_00000001", "name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
           "base_url": "https://canduckhoi--flexsds-serve.modal.run/v1", "platform": "modal",
           "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "max_model_len": 4096,
           "meta": {"gpu": "NVIDIA L4", "adaptive": True}, "api_key": "ak-secret.as-secret", "is_baseline": False,
           "status": "ok", "status_message": None, "checked_at": "2026-11-12T10:00:00+00:00",
           "created_at": "2026-11-12T10:00:00+00:00"}
    store.insert_endpoint(row)
    store.insert_endpoint({**row, "endpoint_id": "ep_00000002", "name": "vanilla", "api_key": None,
                           "base_url": "https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1"})
    listed = store.list_endpoints()
    assert [e["endpoint_id"] for e in listed] == ["ep_00000001", "ep_00000002"]
    assert all("api_key" not in e for e in listed) and [e["has_key"] for e in listed] == [True, False]
    assert listed[0]["meta"] == {"gpu": "NVIDIA L4", "adaptive": True}
    assert store.get_endpoint("ep_00000001", with_key=True)["api_key"] == "ak-secret.as-secret"
    assert store.find_endpoint_by_url("https://canduckhoi--flexsds-serve.modal.run/v1")["endpoint_id"] == "ep_00000001"
    store.update_endpoint("ep_00000001", {"status": "down", "status_message": "không kết nối được"})
    store.set_baseline("ep_00000002")
    store.set_baseline("ep_00000001")
    rows = {e["endpoint_id"]: e for e in store.list_endpoints()}
    assert rows["ep_00000001"]["status"] == "down" and rows["ep_00000001"]["is_baseline"] is True
    assert rows["ep_00000002"]["is_baseline"] is False
    assert store.delete_endpoint("ep_00000002") is True and store.delete_endpoint("ep_00000002") is False
    with reader_connection() as reader, pytest.raises(psycopg_error()):
        reader.execute("SELECT api_key FROM demo.endpoints")


def psycopg_error():
    import psycopg

    return psycopg.Error


@needs_db
def test_postgres_store_bench_runs(pg):
    from src.api.store import PostgresStore

    store = PostgresStore(DB_URL)
    store.create_bench("bn_000000000001", "speed", {"endpoints": ["ep_1", "ep_2"], "loads": [1, 8]})
    store.update_bench("bn_000000000001", "running", progress={"done": 3, "total": 32})
    run = store.get_bench("bn_000000000001")
    assert run["status"] == "running" and run["progress"] == {"done": 3, "total": 32} and run["kind"] == "speed"
    assert store.mark_bench_interrupted() == 1
    assert store.get_bench("bn_000000000001")["status"] == "interrupted"
    store.update_bench("bn_000000000001", "done", result={"cells": []})
    assert [r["run_id"] for r in store.list_bench()] == ["bn_000000000001"]
    assert store.get_bench("bn_missing") is None


@needs_db
def test_store_names_for_text_filters_come_from_the_reader(pg):
    from psycopg.conninfo import make_conninfo

    from src.api.store import PostgresStore

    writer.insert_extraction(pg, payload_with_items_and_tax(), RUN, source="devset")
    store = PostgresStore(DB_URL, make_conninfo(DB_URL, user="demo_reader"))
    values = store.text_values()
    assert set(values) == {"seller_name", "buyer_name"}
    assert values["seller_name"] == [payload_with_items_and_tax()["seller"]["name"]]
