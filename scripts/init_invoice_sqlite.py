"""Create the MC-OCR pipeline SQLite database and optionally load an example."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, time
from pathlib import Path

from jsonschema import Draft202012Validator


DEFAULT_SCHEMA_PATH = Path("src/pipeline/sqlite/schema.sql")
DEFAULT_JSON_SCHEMA_PATH = Path("src/pipeline/schemas/invoice_extraction.schema.json")
DEFAULT_EXAMPLE_PATH = Path("src/pipeline/examples/node1_extractor_output_m0_04_001.json")
DEFAULT_RUN_ID = "m0_04_schema_smoke"


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
            "Create a new database path for canonical SQLite schema v1."
        )
    if user_version not in (0, 1):
        raise RuntimeError(f"Unsupported SQLite schema version: {user_version}")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def semantic_validation_errors(payload: dict) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return errors
    invoice_value = payload.get("invoice", {})
    invoice = invoice_value if isinstance(invoice_value, dict) else {}

    parsers = (
        ("invoice.issued_date", invoice.get("issued_date"), date.fromisoformat),
        ("invoice.issued_time", invoice.get("issued_time"), time.fromisoformat),
        ("invoice.issued_at", invoice.get("issued_at"), datetime.fromisoformat),
    )
    parsed = {}
    for field_path, value, parser in parsers:
        if value is None:
            continue
        try:
            parsed[field_path] = parser(value)
        except ValueError:
            errors.append(f"{field_path}: invalid calendar value {value!r}")

    issued_at = parsed.get("invoice.issued_at")
    issued_date = parsed.get("invoice.issued_date")
    issued_time = parsed.get("invoice.issued_time")
    if issued_at and issued_date and issued_at.date() != issued_date:
        errors.append("invoice.issued_at: date component conflicts with invoice.issued_date")
    if issued_at and issued_time and issued_at.time() != issued_time:
        errors.append("invoice.issued_at: time component conflicts with invoice.issued_time")

    items_value = payload.get("items", [])
    items = items_value if isinstance(items_value, list) else []
    line_numbers = [
        item.get("line_number")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("line_number"), int)
    ]
    if len(line_numbers) != len(set(line_numbers)):
        errors.append("items: line_number values must be unique")
    return errors


def validate_payload(payload: dict, json_schema_path: Path) -> None:
    schema = json.loads(json_schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    details = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
            for error in errors
    ]
    details.extend(semantic_validation_errors(payload))
    if details:
        raise ValueError(f"Payload does not match invoice_extraction.v1: {'; '.join(details)}")


def money_value(data: dict, name: str, key: str) -> object:
    return data.get(name, {}).get(key)


def make_invoice_id(run_id: str, payload: dict) -> str:
    source = payload.get("source", {})
    sample_id = source.get("sample_id")
    img_id = source.get("img_id")
    stable_id = sample_id or img_id
    if not stable_id:
        raise ValueError("source.sample_id or source.img_id is required to build invoice_id")
    return f"{run_id}:{stable_id}"


def insert_extraction(
    conn: sqlite3.Connection,
    payload: dict,
    run_id: str,
    replace: bool,
    json_schema_path: Path = DEFAULT_JSON_SCHEMA_PATH,
) -> str:
    validate_payload(payload, json_schema_path)
    invoice_id = make_invoice_id(run_id, payload)
    source = payload.get("source", {})
    seller = payload.get("seller", {})
    buyer = payload.get("buyer", {})
    invoice = payload.get("invoice", {})
    totals = payload.get("totals", {})
    extraction = payload.get("extraction", {})

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

    conn.execute(
        """
        INSERT INTO invoices (
            invoice_id, run_id, schema_version, document_type,
            sample_id, img_id, image_path, raw_text, raw_json,
            seller_name, seller_address, seller_tax_code, seller_phone,
            buyer_name, buyer_address, buyer_tax_code, buyer_phone,
            invoice_number, invoice_series, issued_at, issued_date, issued_time, currency, payment_method,
            subtotal_raw, subtotal_value, subtotal_currency,
            discount_raw, discount_value, discount_currency,
            vat_amount_raw, vat_amount_value, vat_amount_currency,
            total_amount_raw, total_amount_value, total_amount_currency,
            extraction_status, extraction_confidence, missing_fields_json, warnings_json
        )
        VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        (
            invoice_id,
            run_id,
            payload["schema_version"],
            payload["document_type"],
            source.get("sample_id"),
            source.get("img_id"),
            source.get("image_path"),
            source.get("raw_text"),
            json.dumps(payload, ensure_ascii=False),
            seller.get("name"),
            seller.get("address"),
            seller.get("tax_code"),
            seller.get("phone"),
            buyer.get("name"),
            buyer.get("address"),
            buyer.get("tax_code"),
            buyer.get("phone"),
            invoice.get("invoice_number"),
            invoice.get("invoice_series"),
            invoice.get("issued_at"),
            invoice.get("issued_date"),
            invoice.get("issued_time"),
            invoice.get("currency"),
            invoice.get("payment_method"),
            money_value(totals, "subtotal", "raw"),
            money_value(totals, "subtotal", "value"),
            money_value(totals, "subtotal", "currency"),
            money_value(totals, "discount", "raw"),
            money_value(totals, "discount", "value"),
            money_value(totals, "discount", "currency"),
            money_value(totals, "vat_amount", "raw"),
            money_value(totals, "vat_amount", "value"),
            money_value(totals, "vat_amount", "currency"),
            money_value(totals, "total_amount", "raw"),
            money_value(totals, "total_amount", "value"),
            money_value(totals, "total_amount", "currency"),
            extraction.get("status", "failed"),
            extraction.get("confidence"),
            json.dumps(extraction.get("missing_fields", []), ensure_ascii=False),
            json.dumps(extraction.get("warnings", []), ensure_ascii=False),
        ),
    )

    for item in payload.get("items", []):
        quantity = item.get("quantity", {})
        unit_price = item.get("unit_price", {})
        line_total = item.get("line_total", {})
        conn.execute(
            """
            INSERT INTO invoice_items (
                invoice_id, line_number, description, sku,
                quantity_raw, quantity_value, quantity_unit,
                unit_price_raw, unit_price_value, unit_price_currency,
                line_total_raw, line_total_value, line_total_currency,
                vat_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice_id,
                item["line_number"],
                item.get("description"),
                item.get("sku"),
                quantity.get("raw"),
                quantity.get("value"),
                quantity.get("unit"),
                unit_price.get("raw"),
                unit_price.get("value"),
                unit_price.get("currency"),
                line_total.get("raw"),
                line_total.get("value"),
                line_total.get("currency"),
                item.get("vat_rate"),
            ),
        )

    for tax in payload.get("taxes", []):
        taxable_amount = tax.get("taxable_amount", {})
        tax_amount = tax.get("tax_amount", {})
        conn.execute(
            """
            INSERT INTO invoice_taxes (
                invoice_id, tax_name, tax_rate,
                taxable_amount_raw, taxable_amount_value, taxable_amount_currency,
                tax_amount_raw, tax_amount_value, tax_amount_currency
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice_id,
                tax.get("tax_name"),
                tax.get("tax_rate"),
                taxable_amount.get("raw"),
                taxable_amount.get("value"),
                taxable_amount.get("currency"),
                tax_amount.get("raw"),
                tax_amount.get("value"),
                tax_amount.get("currency"),
            ),
        )

    for evidence in extraction.get("field_evidence", []):
        conn.execute(
            """
            INSERT INTO field_evidence (invoice_id, field_path, raw_text, source_label)
            VALUES (?, ?, ?, ?)
            """,
            (
                invoice_id,
                evidence["field_path"],
                evidence["raw_text"],
                evidence.get("source_label"),
            ),
        )

    conn.commit()
    return invoice_id


def print_summary(conn: sqlite3.Connection) -> None:
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    print(f"sqlite_user_version: {user_version}")
    for table_name in ("extraction_runs", "invoices", "invoice_items", "invoice_taxes", "field_evidence"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
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
    parser.add_argument("--json-schema", type=Path, default=DEFAULT_JSON_SCHEMA_PATH)
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
