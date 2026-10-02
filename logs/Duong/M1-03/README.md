# M1-03 Worklog & Handoff - Duong to Dai

## 2026-09-26 - H1 Data/Contract/W1 Ready

### Scope completed by Duong

- Promoted the tracked M0-04 30-sample MC-OCR devset into the canonical W1 package.
- Froze model-facing W1 artifacts under `experiments/configs/w1/`.
- Added audit-only provenance so evaluator labels stay out of model-facing samples and requests.
- Added the Pydantic mirror for `invoice_extraction.v1` in `src/pipeline/invoice_models.py`.
- Added reusable W1 helper views in `src/pipeline/w1_workload.py`.
- Added build/check scripts:
  - `scripts/build_w1_workload.py`
  - `scripts/check_w1_dataset.py`
- Extended `scripts/check_invoice_contract.py` so the contract check now covers:
  - JSON Schema validation
  - Pydantic validation
  - semantic validation
  - SQLite schema smoke
- Added regression tests:
  - `tests/test_invoice_models.py`
  - `tests/test_w1_workload.py`

### Frozen W1 identity

- Workload ID: `w1_node1_invoice_extraction`
- Dataset version: `w1.m1_03.v1`
- Schema version: `invoice_extraction.v1`
- Prompt version: `node1_extractor.v1`
- Sample count: `30`
- Request count: `30`
- Package SHA-256: `e7cba647035f8834920d523ce8f6a70ef68896069ee71d3a862fac2fc42eff66`
- Prompt SHA-256: `3452c77da3a6485c8e92e371f2861acab9ec6424bbe52477a579345a7e90f8d9`
- Upstream M0-04 commit: `1c8c5c610168cd3f4c68e72d6130949228284d70`

### Canonical artifacts

- W1 samples: `experiments/configs/w1/samples.jsonl`
- W1 provenance: `experiments/configs/w1/provenance.jsonl`
- W1 manifest: `experiments/configs/w1/manifest.csv`
- W1 request package: `experiments/configs/w1/node1_chat_requests.jsonl`
- W1 input text package: `experiments/configs/w1/node1_inputs.jsonl`
- W1 prompt copy: `experiments/configs/w1/system_prompt.md`
- W1 package manifest: `experiments/configs/w1/package_manifest.json`

Tracked evidence:

- Dataset summary: `experiments/results/m1_03_w1_dataset_summary.json`
- Request summary: `experiments/results/m1_03_node1_requests_summary.json`
- W1 checker summary: `experiments/results/m1_03_w1_check_summary.json`
- Contract checker summary: `experiments/results/m1_03_invoice_contract_summary.json`

### Checks run

```powershell
python scripts\build_w1_workload.py
python scripts\check_w1_dataset.py
python scripts\check_invoice_contract.py
python -m unittest tests.test_invoice_models tests.test_w1_workload tests.test_run_node1_extractor
python -m compileall src\pipeline scripts tests
```

Result: all checks passed.

### Important caveats

- W1 is frozen from the M0-04 30-sample devset, not regenerated from Kaggle.
- Model-facing artifacts intentionally exclude `labels` and `key_text_pairs`.
- Annotation labels are retained only in `provenance.jsonl` for audit.
- M0-04 gives complete seller/address/timestamp/total evidence across the 30 samples.
- Buyer/items/VAT evidence is sparse or absent in the upstream OCR labels, so those fields remain nullable and must not be over-claimed.
- `origin/main` still does not contain M0-04 because GitHub branch protection requires review. This branch currently carries the M0-04 dependency explicitly.
## 2026-10-01 - PTIT thesis template + Chapter 1 draft finalized

### Scope completed by Duong

- Added the PTIT-style report template for the thesis manuscript:
  - `docs/00-template-quyen-bao-cao-ptit.md`
- Completed the draft for Chương 1 (Mở đầu) in:
  - `docs/01-gioi-thieu.md`
- Kept the chapter aligned to the project decisions and terminology in:
  - `docs/03-quyet-dinh-thiet-ke.md`
  - `docs/04-quy-trinh-va-ke-hoach.md`
  - `docs/06-ke-hoach-thuc-nghiem.md`
  - `docs/07-thuat-ngu.md`

### PTIT report template included

The standard template covers:

- cover and front matter structure
- chapter layout and numbering rules
- font/layout conventions
- figure/table naming conventions
- writing flow for introduction, problem statement, objectives, scope, contributions, and thesis structure
- final review checklist before thesis submission

### Chương 1 draft status

The draft now contains the core sections required for a PTIT-style introduction:

- problem background and motivation
- formal problem statement
- research questions (RQ1–RQ3)
- goals and scope
- outside-of-scope boundaries
- planned contribution summary
- thesis structure overview

This is now consistent with the work already tracked in the project docs and is ready to be expanded into the formal thesis manuscript in the next drafting cycle.
## Handoff to Dai

### Dai can start from here

Use the frozen W1 request package directly:

```text
experiments/configs/w1/node1_chat_requests.jsonl
```

Do not reconstruct the prompt inside the benchmark harness. If the harness needs an adapter, load and replay the existing request rows while preserving:

- `sample_id`
- `expected_source`
- `messages`
- `temperature=0.0`
- `seed=42`
- `response_format={"type":"json_object"}`
- `schema_version=invoice_extraction.v1`
- `prompt_version=node1_extractor.v1`

### Suggested next tasks for Dai

- Implement/refactor `src/pipeline/extractor.py` as the reusable Node 1 runtime core.
- Keep `scripts/run_node1_extractor.py` compatible with:

```powershell
python scripts\run_node1_extractor.py `
  --requests experiments\configs\w1\node1_chat_requests.jsonl `
  --output-dir experiments\runs\m1_03_node1_vllm_vanilla `
  --report experiments\results\m1_03_node1_vllm_summary.json `
  --run-id m1_03_node1_vllm_vanilla
```

- Validate every accepted payload with both:
  - `src/pipeline/schemas/invoice_extraction.schema.json`
  - `src.pipeline.invoice_models.InvoiceExtraction`
- Preserve source binding exactly: output `source` must match `expected_source`.
- Record failures in a stable taxonomy: HTTP/model error, JSON parse error, source mismatch, schema validation error, Pydantic validation error, SQLite insert error.
- For 30 W1 samples, the >=95% gate means at least `29/30` schema-valid outputs.

### Stop conditions for Dai

- If `node1_chat_requests.jsonl` needs semantic changes, stop and send back to Duong because the package hash must change and W1 checks must be rerun.
- If vLLM/M1-01 does not support `response_format`, seed, or non-streaming chat semantics consistently, stop before full acceptance and document the runtime contract gap.
- If schema-valid rate is below 95%, classify failures first; do not tune prompt/sample cherry-picks without bumping W1 identity and rerunning the full workload.
