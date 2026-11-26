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
  -> Node 2 Text-to-SQL over stable views (Q1-Q6)
  -> Node 3 VAT report (one grounded Vietnamese report)
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

SQLite schema **v2** (`PRAGMA user_version = 2`, M4-03). Compared with v1 it only
adds: lane `eagle`, `invoices.latency_ms` and
`extraction_runs.latency_ms_median`. The three views and the JSON contract are
unchanged. `init_schema` refuses a v1 database (no automatic migration): use a
new database path. The committed `invoices_v1.sqlite` files stay as M1–M3
evidence.

- `extraction_runs` stores experimental provenance (`lane`, model, prompt,
  schema version, and the median Node 1 request latency of the run).
- `invoices` stores one Node 1 result per run, retains `raw_json`, and records
  `latency_ms` of the request that produced it (client side, includes network).
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

## Node 2 / Node 3 runner and E2E command (M3-06)

Code:

- `src/pipeline/node2_questions.py` — Q1–Q6 above, verbatim.
- `src/pipeline/sql_guard.py` — read-only guard: one `SELECT`/`WITH`
  statement, only the three views (or CTE names), no base tables,
  `sqlite_master` or `raw_json`, no DML/DDL/`PRAGMA`/`ATTACH`/`load_extension`.
  Literals and comments are masked before checking.
- `src/pipeline/text_to_sql.py` — `Node2Runner`: one LLM request per question,
  then guard, then execution on a `mode=ro` + `PRAGMA query_only=ON`
  connection (5 s timeout, at most 200 rows).
- `src/pipeline/report_writer.py` — `Node3Writer`: one request that receives
  each question, its SQL, its status and at most 20 rows, and must use only
  those numbers.
- `scripts/run_pipeline_e2e.py` — Node 1 (existing CLI, subprocess) → SQLite of
  this run → Node 2 → Node 3.

Prompts and sampling: Node 2 uses the frozen W2 v2 system prompt
(`experiments/configs/w2_v2/system_prompt.md`), Node 3 the frozen W3 v2 system
prompt (`experiments/configs/w3_v2/system_prompt.md`); `temperature=0.0`,
`seed=42`, `max_tokens` 256 (Node 2) and 768 (Node 3). The Node 3 user message
template is `src/pipeline/prompts/node3_report_user.md`.

**Contract deviation (recorded):** the catalog above says Node 2 *should*
return the SQL, the selected view and a short explanation. Because Node 2
reuses the W2 v2 prompt (SQL text only, so pipeline output matches the
benchmark workload), `views_used` is derived from the SQL and `explanation` is
always `null`.

Node 2 status per question: `ok` (rows), `empty` (0 rows, or only NULL values —
an aggregate over zero matching rows returns one all-NULL row), `transport_error`,
`guard_rejected`, `sql_error`, `timeout`. A failed question is recorded and the
pipeline continues; the LLM is never called again to repair SQL, so request
counts stay equal across serving lanes. Raw network errors (read timeout,
dropped connection) are recorded as `transport_error` too. Node 3 status:
`ok`, `transport_error`, `empty_response` (blank content), or
`skipped_no_data` when no question returned `ok`/`empty`. Before Node 1 runs,
Node 1 outputs left by an earlier run with the same `run_id`/lane are deleted;
if Node 1 crashes without a summary, Node 2/3 are skipped.

Run on Modal (lane token is read from `.env`):

```powershell
python scripts\run_pipeline_e2e.py --lane vanilla --run-id <id>
```

Local functional check with Ollama (not a performance measurement):

```powershell
python scripts\run_pipeline_e2e.py --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct --run-id local-smoke --limit 5
```

Re-run Node 2/3 on an existing database without calling Node 1 again:
`--skip-node1 --db <path>`.
Record the serving GPU in the summary with `--hardware <label>` (e.g.
`modal-A10`; Modal GPU is chosen at deploy time with `PTIT_GPU`).

Outputs in `experiments/runs/m3_06/<run_id>-<lane>/`: `node1/`,
`node1_summary.json`, `invoices.sqlite` (not committed), `node2_results.jsonl`,
`node3_result.json`, `report.md`, `e2e_summary.json`. Exit code 0 requires
≥ 1 invoice, ≥ 1 Node 2 question `ok|empty` and Node 3 `ok`; otherwise 2.
Latency and token counts are client-side only; this command does not emit the
M1 request telemetry contract.

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
python scripts\run_node1_extractor.py --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct --run-id local-ollama --replace-run
```

The runner checks that the model exists, calls `/v1/chat/completions` in JSON
mode, verifies source binding, validates `invoice_extraction.v1`, and loads
valid outputs into SQLite.

## Running Node 1 on a serving lane (M4-03)

```powershell
python scripts\run_node1_extractor.py --lane eagle --hardware modal-A10
python scripts\run_node1_extractor.py --lane static --backend local
```

`--lane` accepts `vanilla`, `static`, `adaptive`, `eagle`. Without `--base-url`
or `--model`, the endpoint and served model follow the lane: Modal by default,
or ports 8001–8004 with `--backend local`. `--base-url` and `--model` (then
`PIPELINE_LLM_BASE_URL`, `PIPELINE_LLM_MODEL`) take precedence. The API key comes
from `--api-key`, `PIPELINE_LLM_API_KEY`, or the Modal proxy token.

Defaults: `run_id = m4_03_<lane>`, output in `experiments/runs/m4_03/<run_id>/`
(summary in `run_summary.json`), database `data/processed/invoices.db` (not
committed). Rerunning an existing `run_id` needs `--replace-run`. A wrong
database or an existing run stops before any server call.

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
