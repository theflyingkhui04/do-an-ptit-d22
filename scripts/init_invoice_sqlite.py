"""Create the MC-OCR pipeline SQLite database and optionally load an example."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.invoice_rows import (  # noqa: E402,F401  (make_invoice_id/money_value re-exported)
    EVIDENCE_COLUMNS,
    INVOICE_COLUMNS,
    ITEM_COLUMNS,
    TAX_COLUMNS,
    build_invoice_rows,
    make_invoice_id,
    money_value,
)
from src.pipeline.validation import (  # noqa: E402
    DEFAULT_JSON_SCHEMA_PATH,
    validate_payload,
)


DEFAULT_SCHEMA_PATH = Path("src/pipeline/sqlite/schema.sql")
DEFAULT_EXAMPLE_PATH = Path("src/pipeline/examples/node1_extractor_output_m0_04_001.json")
DEFAULT_RUN_ID = "m0_04_schema_smoke"
SCHEMA_USER_VERSION = 2


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    has_pipeline_tables = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('extraction_runs', 'invoices')
        )
        """
    ).fetchone()[0]
    if has_pipeline_tables and user_version == 0:
        raise RuntimeError(
            "This database uses the pre-final unversioned schema. "
            f"Create a new database path for canonical SQLite schema v{SCHEMA_USER_VERSION}."
        )
    if user_version not in (0, SCHEMA_USER_VERSION):
        raise RuntimeError(
            f"This database has user_version {user_version}; no automatic migration exists. "
            f"Use a new database path for canonical SQLite schema v{SCHEMA_USER_VERSION}."
        )
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def _insert(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], row: dict) -> None:
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                 tuple(row[column] for column in columns))


def insert_extraction(
    conn: sqlite3.Connection,
    payload: dict,
    run_id: str,
    replace: bool,
    json_schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
    latency_ms: float | None = None,
) -> str:
    validate_payload(payload, json_schema_path)
    rows = build_invoice_rows(payload, run_id, latency_ms=latency_ms)
    invoice_id = rows.invoice["invoice_id"]

    conn.execute(
        """
        INSERT OR IGNORE INTO extraction_runs (
            run_id, lane, model_name, prompt_version, schema_version, notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            None,
            None,
            "node1_extractor.v1",
            payload.get("schema_version", "invoice_extraction.v1"),
            "Local SQLite schema smoke run",
        ),
    )

    run_schema_version = conn.execute(
        "SELECT schema_version FROM extraction_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    if run_schema_version != payload["schema_version"]:
        raise ValueError(
            f"run_id={run_id!r} uses schema_version={run_schema_version!r}, "
            f"not {payload['schema_version']!r}"
        )

    if replace:
        conn.execute("DELETE FROM invoices WHERE invoice_id = ?", (invoice_id,))

    _insert(conn, "invoices", INVOICE_COLUMNS, rows.invoice)
    for item in rows.items:
        _insert(conn, "invoice_items", ITEM_COLUMNS, item)
    for tax in rows.taxes:
        _insert(conn, "invoice_taxes", TAX_COLUMNS, tax)
    for evidence in rows.evidence:
        _insert(conn, "field_evidence", EVIDENCE_COLUMNS, evidence)

    conn.commit()
    return invoice_id


def print_summary(conn: sqlite3.Connection) -> None:
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    print(f"sqlite_user_version: {user_version}")
    for table_name in (
        "extraction_runs",
        "invoices",
        "invoice_items",
        "invoice_taxes",
        "field_evidence",
    ):
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        print(f"{table_name}: {count}")

    rows = conn.execute(
        """
        SELECT sample_id, seller_name, issued_date, total_amount_value, item_count
        FROM v_invoice_facts
        ORDER BY sample_id
        LIMIT 5
        """
    ).fetchall()
    for row in rows:
        print("invoice_fact:", row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--json-schema",
        type=Path,
        default=DEFAULT_JSON_SCHEMA_PATH,
    )
    parser.add_argument("--load-example", action="store_true")
    parser.add_argument("--example", type=Path, default=DEFAULT_EXAMPLE_PATH)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = connect(args.db_path)
    try:
        init_schema(conn, args.schema)
        if args.load_example:
            payload = json.loads(args.example.read_text(encoding="utf-8"))
            invoice_id = insert_extraction(
                conn,
                payload,
                args.run_id,
                args.replace,
                json_schema_path=args.json_schema,
            )
            print(f"loaded invoice_id={invoice_id}")
        print_summary(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
