# MC-OCR Pipeline Contract v1

Status: **frozen for M0-04**. Any incompatible change must create a new JSON
`schema_version` and a new SQLite `user_version`.

## Scope

The MC-OCR pipeline is an end-to-end measurement workload for the thesis, not
the thesis contribution itself. Invoice-specific parsing, validation, and SQL
remain under `src/pipeline/`; the serving engine only exposes the frozen
OpenAI-compatible API described in `docs/05-thiet-ke-he-thong.md`.

Node 1 is a text-to-JSON extractor, not a visual OCR model. In the M0-04
dev-set, `source.raw_text` comes from MC-OCR `anno_texts`; images are retained
for audit. Replacing this text later with output from an upstream OCR engine
does not change the v1 contract.

Both Node 1 and Node 2 use an LLM through the same OpenAI-compatible endpoint.
Node 1 changes receipt text into constrained JSON; Node 2 changes a Vietnamese
question into read-only SQL. Ollama is used only for local functional checks.
Final performance measurements must use the vLLM serving lanes because Ollama
does not expose the thesis proposer or its telemetry.

```text
MC-OCR text
  -> Node 1 Extractor
  -> invoice_extraction.v1 JSON
  -> SQLite schema user_version=1
  -> Node 2 Text-to-SQL over stable views
```

## Node 1 Contract

Canonical files:

- Prompt: `src/pipeline/prompts/node1_extractor_system.md`
- JSON Schema: `src/pipeline/schemas/invoice_extraction.schema.json`
- Example input: `src/pipeline/examples/node1_extractor_input_m0_04_001.txt`
- Example output: `src/pipeline/examples/node1_extractor_output_m0_04_001.json`

The example input contains the pipeline-owned source metadata and OCR text for
one receipt. The example output shows the corresponding `invoice_extraction.v1`
object, including nullable fields, normalized date/time values, money objects,
and extraction evidence. Together they are the concrete input/output example
for the M0-04 acceptance checklist.

Frozen rules:

1. The pipeline owns `source.sample_id`, `source.img_id`,
   `source.image_path`, and `source.raw_text`. Node 1 copies these values; it
   does not infer or repair them. A response runner must compare them with the
   request envelope and reject a mismatched source before database insertion.
2. Missing scalar business fields are `null`; missing collections are `[]`.
   Node 1 must not invent item, invoice, buyer, or VAT data.
3. Dates use `YYYY-MM-DD`, times use `HH:MM[:SS]`, and a combined local
   timestamp uses `YYYY-MM-DDTHH:MM:SS`. No timezone is inferred from MC-OCR.
4. Currency uses a three-letter ISO 4217 code. A money object keeps both OCR
   `raw` text and normalized numeric `value`; SQLite uses `REAL` because this
   is an analytics benchmark, not an accounting ledger.
5. `extraction.confidence` is an optional audit hint. It is not used as a
   ground-truth quality metric or as an input to the adaptive proposer.
6. MC-OCR annotation labels stay outside the model prompt. They may be used
   later for audit or evaluation only.
7. `status=ok` requires an empty `missing_fields`; `status=partial` requires at
   least one missing field. Calendar values must be real and item
   `line_number` values must be unique before SQLite insertion.

## SQLite Contract

Canonical DDL: `src/pipeline/sqlite/schema.sql`.

- `extraction_runs` stores experimental provenance (`lane`, model, prompt, and
  schema version).
- `invoices` stores one Node 1 result per run and retains `raw_json`.
- `invoice_items`, `invoice_taxes`, and `field_evidence` store repeating data.
- `run_id` is mandatory. Deleting a run cascades to all data produced by it.
- `invoice_id` is deterministic: `<run_id>:<sample_id>`.
- JSON is validated against `invoice_extraction.v1` before insertion.

Node 2 should query these read-only surfaces:

| View | Grain | Intended questions |
|---|---|---|
| `v_invoice_facts` | One row per invoice | seller/buyer, dates, totals, payment, counts |
| `v_item_facts` | One row per item | quantities, unit prices, item totals, item VAT |
| `v_tax_facts` | One row per tax line | tax rates, taxable amounts, VAT amounts |

Direct table access is reserved for pipeline loading and audit. Node 2 must
generate read-only `SELECT` statements and must not query `raw_json` unless an
explicit audit task requires it.

## Business question catalog for Node 2

The following Vietnamese questions are the initial business contract for the
text-to-SQL workload. The generated SQL must be a read-only `SELECT` over the
listed view; the examples identify the intended grain and aggregation, not a
requirement to copy one fixed SQL string.

| ID | Câu hỏi nghiệp vụ | View ưu tiên | Phép tính/kết quả mong đợi |
|---|---|---|---|
| Q1 | Tổng chi tiêu trong tháng 08/2020 là bao nhiêu? | `v_invoice_facts` | `SUM(total_amount_value)` lọc theo khoảng `issued_date`. |
| Q2 | Mặt hàng nào được mua nhiều nhất? | `v_item_facts` | `SUM(quantity_value)` theo `description`, sắp xếp giảm dần. |
| Q3 | Cửa hàng nào có tổng giá trị hóa đơn cao nhất? | `v_invoice_facts` | `SUM(total_amount_value)` theo `seller_name`. |
| Q4 | Có bao nhiêu hóa đơn có phát sinh VAT? | `v_invoice_facts` hoặc `v_tax_facts` | Đếm hóa đơn có `vat_amount_value` hoặc dòng thuế tương ứng. |
| Q5 | Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu? | `v_invoice_facts` | `SUM(total_amount_value)` theo `issued_date`, sắp xếp tăng dần. |
| Q6 | Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu? | `v_tax_facts` | `SUM(tax_amount_value)` theo `tax_name`, `tax_rate`. |

Node 2 must reject non-`SELECT` statements and should return the generated SQL,
the selected view, and a short explanation so the result can be audited. These
questions cover invoice totals, item quantities, seller aggregation, VAT
presence, date grouping, and tax analysis without exposing `raw_json` to the
normal business-query path.

## Versioning Boundary

Compatible v1 changes may clarify descriptions, add examples, or add indexes.
Renaming/removing fields, changing nullability, changing normalized units, or
changing table/view semantics requires a new major contract version. The
pre-final unversioned local SQLite file must be rebuilt at a new path; no
automatic migration is claimed for it.

## Acceptance Checks

```powershell
python scripts\validate_invoice_schema.py src\pipeline\examples\node1_extractor_output_m0_04_001.json
python scripts\check_invoice_contract.py
```

## Local Ollama Step 7

```powershell
ollama pull qwen2.5:7b-instruct
python scripts\build_node1_requests.py --model qwen2.5:7b-instruct
python scripts\run_node1_extractor.py --replace-run
```

The runner uses `http://localhost:11434/v1` by default, checks that the model
exists, calls `/v1/chat/completions` in JSON mode, verifies source binding,
validates `invoice_extraction.v1`, and loads valid outputs into SQLite. Raw
responses and failures remain under the gitignored `data/processed/` tree; a
compact reproducibility summary is written to `experiments/results/`.

Before validation, the runner may apply these deterministic canonicalization
rules:

- compose `invoice.issued_at` from existing normalized `issued_date` and
  `issued_time`;
- replace a null required totals container with an empty money object whose
  `raw`, `value`, and `currency` fields are all null;
- default an omitted evidence `source_label` to null; and
- discard an evidence row whose `raw_text` is not a verbatim substring of the
  pipeline-owned source text.

Every applied rule is retained in the output's normalization audit list, and
the original assistant response remains unchanged in `raw_responses.jsonl`.
These rules only repair the canonical envelope or remove unsupported evidence.
Seller, buyer, item, money, and tax values are never inferred or overwritten by
post-processing without OCR evidence.
