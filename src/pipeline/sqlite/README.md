# SQLite Schema v1 For MC-OCR Pipeline

Canonical DDL:

```text
src/pipeline/sqlite/schema.sql
```

This database is the handoff between:

- Node 1 Extractor: MC-OCR text to stable JSON.
- Node 2 Text-to-SQL: Vietnamese business question to SQL over receipt data.
- Node 3 SQL-to-BI: optional free-form report over query results.

Design notes:

- The schema mirrors `src/pipeline/schemas/invoice_extraction.schema.json`.
- Receipt/VAT fields are nullable because MC-OCR samples often lack tax code,
  invoice number, item rows, or VAT details.
- `raw_json`, `raw_text`, and `field_evidence` preserve auditability without
  making invoice logic part of the serving engine.
- `v_invoice_facts`, `v_item_facts`, and `v_tax_facts` are the preferred
  read-only surfaces for Text-to-SQL prompts. They keep Node 2 queries simple
  and predictable.
- Every invoice belongs to an `extraction_runs` row. Deleting a run cascades
  to the records produced by that run, preserving provenance boundaries.
- The DDL sets SQLite `PRAGMA user_version = 1`. Pre-final unversioned local
  databases are intentionally not migrated; create a new database path.

Initialize an empty database:

```powershell
python scripts\init_invoice_sqlite.py data\processed\m0_04_mcocr_devset\invoices_v1.sqlite
```

Smoke-test with the Node 1 example:

```powershell
python scripts\init_invoice_sqlite.py data\processed\m0_04_mcocr_devset\invoices_v1.sqlite --load-example --replace
```

Run the complete JSON-to-relational contract check:

```powershell
python scripts\check_invoice_contract.py
```
