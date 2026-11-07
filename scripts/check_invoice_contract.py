"""Check the frozen Node 1 JSON and Node 2 SQLite contracts end to end."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from init_invoice_sqlite import SCHEMA_USER_VERSION, connect, init_schema, insert_extraction  # noqa: E402
from src.pipeline.invoice_models import validate_invoice_payload  # noqa: E402
from src.pipeline.validation import validate_payload  # noqa: E402


JSON_SCHEMA_PATH = Path("src/pipeline/schemas/invoice_extraction.schema.json")
SQL_SCHEMA_PATH = Path("src/pipeline/sqlite/schema.sql")
EXAMPLE_PATH = Path("src/pipeline/examples/node1_extractor_output_m0_04_001.json")
RUN_ID = "m0_04_contract_check"


def assert_json_contract(payload: dict) -> None:
    validate_payload(payload, JSON_SCHEMA_PATH)
    schema = json.loads(JSON_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    invalid = copy.deepcopy(payload)
    invalid["source"]["sample_id"] = None
    assert list(validator.iter_errors(invalid)), "source.sample_id must be non-null"

    invalid = copy.deepcopy(payload)
    invalid["invoice"]["currency"] = "VNĐ"
    assert list(validator.iter_errors(invalid)), "currency must be an ISO 4217 code"

    invalid = copy.deepcopy(payload)
    invalid["invoice"]["issued_date"] = "2020-02-30"
    try:
        validate_payload(invalid, JSON_SCHEMA_PATH)
    except ValueError:
        pass
    else:
        raise AssertionError("semantic validation accepted an impossible calendar date")

    invalid = build_relational_payload(payload)
    invalid["items"].append(copy.deepcopy(invalid["items"][0]))
    try:
        validate_payload(invalid, JSON_SCHEMA_PATH)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "semantic validation accepted duplicate item line_number values"
        )


def assert_pydantic_contract(payload: dict) -> None:
    validate_invoice_payload(payload)

    invalid = copy.deepcopy(payload)
    invalid["invoice"]["currency"] = "VNĐ"
    try:
        validate_invoice_payload(invalid)
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "Pydantic contract accepted a non-ISO currency code"
        )

    invalid = copy.deepcopy(payload)
    invalid["invoice"]["issued_date"] = "2020-02-30"
    try:
        validate_invoice_payload(invalid)
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "Pydantic contract accepted an impossible calendar date"
        )

    invalid = build_relational_payload(payload)
    invalid["items"].append(copy.deepcopy(invalid["items"][0]))
    try:
        validate_invoice_payload(invalid)
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "Pydantic contract accepted duplicate item line_number values"
        )

    invalid = copy.deepcopy(payload)
    invalid["extraction"]["field_evidence"][0]["raw_text"] = (
        "not present in OCR text"
    )
    try:
        validate_invoice_payload(invalid)
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "Pydantic contract accepted non-verbatim field evidence"
        )


def build_relational_payload(payload: dict) -> dict:
    relational = copy.deepcopy(payload)
    relational["items"] = [
        {
            "line_number": 1,
            "description": "San pham kiem thu",
            "sku": "TEST-001",
            "quantity": {"raw": "2", "value": 2, "unit": "cai"},
            "unit_price": {
                "raw": "10,000",
                "value": 10000,
                "currency": "VND",
            },
            "line_total": {
                "raw": "20,000",
                "value": 20000,
                "currency": "VND",
            },
            "vat_rate": 10,
        }
    ]
    relational["taxes"] = [
        {
            "tax_name": "VAT",
            "tax_rate": 10,
            "taxable_amount": {
                "raw": "20,000",
                "value": 20000,
                "currency": "VND",
            },
            "tax_amount": {
                "raw": "2,000",
                "value": 2000,
                "currency": "VND",
            },
        }
    ]
    return relational


def assert_sqlite_contract(payload: dict) -> None:
    conn = connect(Path(":memory:"))
    try:
        init_schema(conn, SQL_SCHEMA_PATH)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        conn.execute("INSERT INTO extraction_runs (run_id, lane) VALUES ('lane-check', 'eagle')")
        conn.execute("DELETE FROM extraction_runs WHERE run_id = 'lane-check'")

        relational = build_relational_payload(payload)
        validate_payload(relational, JSON_SCHEMA_PATH)
        invoice_id = insert_extraction(
            conn,
            relational,
            RUN_ID,
            replace=False,
            latency_ms=12.5,
        )
        init_schema(conn, SQL_SCHEMA_PATH)
        assert conn.execute(
            "SELECT latency_ms FROM invoices WHERE invoice_id = ?", (invoice_id,)
        ).fetchone()[0] == 12.5

        invoice = conn.execute(
            """
            SELECT sample_id, seller_name, total_amount_value, item_count, tax_line_count
            FROM v_invoice_facts
            WHERE invoice_id = ?
            """,
            (invoice_id,),
        ).fetchone()
        assert invoice == (
            "m0_04_001",
            "MINIMART ANAN",
            42000.0,
            1,
            1,
        )

        item = conn.execute(
            """
            SELECT sku, quantity_value, line_total_value
            FROM v_item_facts
            WHERE invoice_id = ?
            """,
            (invoice_id,),
        ).fetchone()
        assert item == ("TEST-001", 2.0, 20000.0)

        tax = conn.execute(
            """
            SELECT tax_name, tax_rate, tax_amount_value
            FROM v_tax_facts
            WHERE invoice_id = ?
            """,
            (invoice_id,),
        ).fetchone()
        assert tax == ("VAT", 10.0, 2000.0)

        try:
            conn.execute(
                """
                UPDATE invoices
                SET missing_fields_json = ?
                WHERE invoice_id = ?
                """,
                ('"not-an-array"', invoice_id),
            )
        except sqlite3.IntegrityError:
            conn.rollback()
        else:
            raise AssertionError(
                "missing_fields_json accepted a non-array JSON value"
            )

        conn.execute(
            "DELETE FROM extraction_runs WHERE run_id = ?",
            (RUN_ID,),
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] == 0
    finally:
        conn.close()


def main() -> None:
    payload = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert_json_contract(payload)
    assert_pydantic_contract(payload)
    assert_sqlite_contract(payload)
    summary = {
        "task": "M4-03",
        "step": "invoice-contract-check",
        "status": "PASS",
        "json_schema": JSON_SCHEMA_PATH.as_posix(),
        "pydantic_model": "src.pipeline.invoice_models.InvoiceExtraction",
        "sqlite_user_version": SCHEMA_USER_VERSION,
        "example": EXAMPLE_PATH.as_posix(),
    }
    report_path = Path(
        "experiments/results/m4_03_invoice_contract_summary.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
