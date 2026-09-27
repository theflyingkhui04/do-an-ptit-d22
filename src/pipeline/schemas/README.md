# Pipeline Schemas

This directory is the source of truth for structured outputs produced by
`src/pipeline/`.

## Node 1 Extractor v1 (Frozen)

Canonical schema:

```text
src/pipeline/schemas/invoice_extraction.schema.json
```

Role in the thesis flow:

- Node 1 is the low-entropy JSON workload in the MC-OCR pipeline.
- The receipt domain is only a measurement workload, following QD-08 in
  `docs/03-quyet-dinh-thiet-ke.md`; no receipt-specific logic belongs in the
  serving engine.
- The schema keeps keys stable and values nullable so the prompt can force a
  deterministic shape even when MC-OCR text lacks VAT, tax code, or item rows.
- `source.sample_id` and `source.raw_text` are pipeline-owned required values;
  the model copies them instead of inferring them.
- Money fields keep both `raw` text and normalized numeric `value` so later
  SQLite insertion can be deterministic without losing auditability.
- Currency accepts three-letter ISO 4217 codes. Date/time values remain local
  because MC-OCR does not provide reliable timezone evidence.

The cross-layer decisions shared with SQLite are frozen in
`src/pipeline/README.md`.

Validation:

```powershell
python scripts\validate_invoice_schema.py src\pipeline\examples\node1_extractor_output_m0_04_001.json
```

Example input/output:

```text
src/pipeline/examples/node1_extractor_input_m0_04_001.txt
src/pipeline/examples/node1_extractor_output_m0_04_001.json
```
