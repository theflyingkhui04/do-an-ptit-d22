"""SQLite schema v2: eagle lane, latency columns, refusal of older databases."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from init_invoice_sqlite import connect, init_schema, insert_extraction  # noqa: E402

SCHEMA_SQL = REPO_ROOT / "src/pipeline/sqlite/schema.sql"
EXAMPLE_JSON = REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
V1_DB = REPO_ROOT / "data/processed/m0_04_mcocr_devset/invoices_v1.sqlite"
VIEWS = ("v_invoice_facts", "v_item_facts", "v_tax_facts")


def view_columns(conn: sqlite3.Connection, view: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({view})")]


class SchemaV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self._tmp.name) / "v2.sqlite")
        init_schema(self.conn, SCHEMA_SQL)
        self.payload = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_user_version_is_2(self) -> None:
        self.assertEqual(self.conn.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_init_schema_is_repeatable_on_v2(self) -> None:
        init_schema(self.conn, SCHEMA_SQL)
        self.assertEqual(self.conn.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_all_four_lanes_accepted(self) -> None:
        for lane in ("vanilla", "static", "adaptive", "eagle"):
            self.conn.execute("INSERT INTO extraction_runs (run_id, lane) VALUES (?, ?)", (f"r-{lane}", lane))

    def test_unknown_lane_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO extraction_runs (run_id, lane) VALUES ('r', 'turbo')")

    def test_latency_stored_with_invoice(self) -> None:
        invoice_id = insert_extraction(self.conn, self.payload, "r1", replace=False, latency_ms=7955.334)
        stored = self.conn.execute("SELECT latency_ms FROM invoices WHERE invoice_id = ?", (invoice_id,)).fetchone()[0]
        self.assertEqual(stored, 7955.334)

    def test_latency_defaults_to_null(self) -> None:
        invoice_id = insert_extraction(self.conn, self.payload, "r1", replace=False)
        stored = self.conn.execute("SELECT latency_ms FROM invoices WHERE invoice_id = ?", (invoice_id,)).fetchone()[0]
        self.assertIsNone(stored)

    def test_negative_latency_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            insert_extraction(self.conn, self.payload, "r1", replace=False, latency_ms=-1.0)

    def test_negative_run_median_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO extraction_runs (run_id, latency_ms_median) VALUES ('r', -1)")


class OlderDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_v1_database_refused_and_left_unchanged(self) -> None:
        copy = self.root / "old.sqlite"
        shutil.copy(V1_DB, copy)
        before = hashlib.sha256(copy.read_bytes()).hexdigest()
        conn = sqlite3.connect(copy)
        try:
            with self.assertRaisesRegex(RuntimeError, "user_version 1.*new database path"):
                init_schema(conn, SCHEMA_SQL)
        finally:
            conn.close()
        self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), before)

    def test_unversioned_database_refused(self) -> None:
        conn = sqlite3.connect(self.root / "unversioned.sqlite")
        try:
            conn.execute("CREATE TABLE invoices (invoice_id TEXT)")
            with self.assertRaisesRegex(RuntimeError, "new database path"):
                init_schema(conn, SCHEMA_SQL)
        finally:
            conn.close()

    def test_views_keep_v1_columns(self) -> None:
        old = sqlite3.connect(V1_DB)
        new = sqlite3.connect(":memory:")
        try:
            self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 1)
            init_schema(new, SCHEMA_SQL)
            for view in VIEWS:
                self.assertEqual(view_columns(new, view), view_columns(old, view), view)
        finally:
            old.close()
            new.close()


if __name__ == "__main__":
    unittest.main()
