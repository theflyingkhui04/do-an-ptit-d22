# Worklog Kỹ Thuật: [M1-03] Node 1 VAT JSON Extractor & W1 Schema-Valid Acceptance

- **Mã công việc:** `[M1-03]` — Node 1 (VAT JSON Extractor) Pipeline MC-OCR
- **Người thực hiện:** Đại (`daijestic`) — Dev / MLOps
- **Ngày nghiệm thu kỹ thuật:** 27/09/2026
- **Môi trường nghiệm thu chính:** Kaggle Notebook — NVIDIA Tesla T4
- **Branch:** `feat/m1-03-mc-ocr-node1-extractor`
- **Evidence commit:** `731ad13` — `test(m1-03): record W1 vanilla acceptance evidence`
- **Runner commit:** `f8ff030` — `refactor(m1-03): use reusable node1 runner`
- **Runtime validation commit:** `4b27ffd` — `feat(m1-03): add reusable node1 runtime validation`
- **Pure-core commit:** `183d6aa` — `refactor(pipeline): extract reusable node1 core`
- **W1 reproducibility fix:** `57badbe` — `fix(w1): make frozen package hashing cross-platform`
- **Trạng thái kỹ thuật:** `DONE WITH EVIDENCE` cho 2 task `[Đại]` thuộc M1-03
- **Trạng thái quy trình:** Branch đã push lên `origin`, chưa merge vào `main`; chờ PR review / PM merge
- **Trello:** READ-ONLY — không tự mark done / sửa card

---

## 1. Hai task được giao

### Task A — Module Node 1

> `[Đại] Lập trình module Node 1 (src/pipeline/extractor.py) gọi API vLLM qua OpenAI HTTP client`

Mục tiêu kỹ thuật:

- tách reusable extraction core khỏi script orchestration;
- gọi OpenAI-compatible API của vLLM;
- fail-fast trước measured loop nếu endpoint/model không hợp lệ;
- strict-parse assistant output thành đúng một JSON object;
- chỉ áp dụng các deterministic normalization đã được khóa;
- không để model sửa trusted `source.*`;
- chạy JSON Schema + semantic validation + Pydantic mirror trước khi coi sample là valid;
- preserve raw response để audit;
- trả structured success/failure envelope thay vì exception hỗn hợp.

### Task B — Test suite / acceptance >= 95%

> `[Đại] Xây dựng test suite đánh giá độ chính xác trích xuất, đạt tỉ lệ parsing JSON hợp lệ >= 95%`

Acceptance được hiện thực theo frozen W1 như:

```text
schema_valid_rate = fully_validated_count / request_count
```

Một sample chỉ được tính `fully_validated` nếu đi qua toàn bộ:

```text
HTTP success
→ response envelope hợp lệ
→ strict JSON object
→ allowed deterministic normalization
→ exact source binding
→ JSON Schema validation
→ semantic validation
→ Pydantic mirror validation
```

Không đồng nhất `json.loads()` thành công với fully validated sample.

---

## 2. Dependency direction và validation architecture

Ban đầu reusable validation logic nằm một phần trong:

```text
scripts/init_invoice_sqlite.py
```

trong khi typed mirror nằm ở:

```text
src/pipeline/invoice_models.py
```

Thiết kế được chốt:

```text
scripts/init_invoice_sqlite.py ─┐
                               ├──> src/pipeline/validation.py
src/pipeline/extractor.py ──────┘
```

Nguyên tắc:

```text
scripts → src
```

không phải:

```text
src → scripts
```

`src/pipeline/validation.py` trở thành reusable validation stack dùng chung cho extractor và SQLite tooling, tránh hai implementation bị drift.

Validation stack:

```text
JSON Schema
→ semantic_validation_errors
→ Pydantic mirror
```

JSON Schema v1 tiếp tục là frozen public contract; Pydantic là typed mirror và không tự bật global strict mode ngoài contract hiện hành.

---

## 3. `src/pipeline/extractor.py`

Module reusable bao gồm:

```text
FailureCode
ExtractorStage
ExtractorError
Node1Success
Node1Failure

parse_assistant_payload(...)
normalize_canonical_payload(...)
validate_source_binding(...)

Node1Extractor.preflight()
Node1Extractor.run_request()
```

### 3.1. Strict JSON parsing

Assistant content phải decode thành đúng một JSON object.

Các dạng bị reject:

- Markdown fenced JSON;
- prose + JSON;
- JSON array;
- empty/non-string content;
- malformed response envelope;
- non-standard numeric constants:
  - `NaN`
  - `Infinity`
  - `-Infinity`

Raw `assistant_content` được giữ nguyên để audit.

### 3.2. Source binding

`expected_source` đến từ trusted W1 request envelope.

Invariant:

```text
payload["source"] == expected_source
```

Không sửa hộ model bằng:

```python
payload["source"] = expected_source
```

Nếu missing key, unexpected key hoặc value mismatch:

```text
SOURCE_BINDING_ERROR
```

### 3.3. Canonicalization được phép

Chỉ các repair class đã khóa:

1. `issued_date + issued_time → issued_at`
2. null total money container → explicit empty Money object
3. missing `field_evidence[].source_label` → `null`
4. remove field-evidence row nếu `raw_text` không phải substring của trusted source text

Không được thay đổi/infer:

- seller / buyer;
- item;
- amount;
- VAT/tax;
- source provenance;
- identifier;
- semantic fact khác.

### 3.4. Failure taxonomy

Per-sample/runtime codes:

```text
TRANSPORT_ERROR
RESPONSE_ENVELOPE_ERROR
JSON_PARSE_ERROR
SOURCE_BINDING_ERROR
SCHEMA_VALIDATION_ERROR
SEMANTIC_VALIDATION_ERROR
PERSISTENCE_ERROR
```

Run-level / preflight:

```text
ENDPOINT_UNREACHABLE
MODEL_NOT_FOUND
BACKEND_CONTRACT_UNSUPPORTED
DATASET_INVALID
PROMPT_PACKAGE_INVALID
```

Endpoint/model unavailable được fail-fast trước sample loop, không biến thành 30 sample failures giả.

---

## 4. Thin runner

File:

```text
scripts/run_node1_extractor.py
```

được refactor thành orchestration layer quanh:

```text
Node1Extractor
```

Runner không duplicate:

- strict JSON parser;
- normalization;
- source binding;
- validation stack.

Runner chịu trách nhiệm:

- load frozen W1 request package;
- package-wide precondition checks;
- preflight endpoint/model;
- iterate requests;
- collect raw responses / validated outputs / failures;
- optional SQLite persistence;
- aggregate correctness summary;
- apply acceptance gate;
- ghi evidence artifacts.

---

## 5. Frozen W1 workload

Canonical workload:

```text
workload_id     = w1_node1_invoice_extraction
dataset_version = w1.m1_03.v1
schema_version  = invoice_extraction.v1
prompt_version  = node1_extractor.v1
sample_count    = 30
```

Frozen package identity:

```text
package_sha256 =
e7cba647035f8834920d523ce8f6a70ef68896069ee71d3a862fac2fc42eff66
```

Canonical request artifact:

```text
experiments/configs/w1/node1_chat_requests.jsonl
```

W1 package check xác minh:

- minimum sample count;
- unique sample IDs;
- provenance completeness;
- no annotation-label leakage vào model-facing request;
- request ↔ source binding;
- package hash.

---

## 6. Cross-platform W1 hash fix

Trên Windows với `core.autocrlf=true`, raw byte hash ban đầu bị thay đổi do LF/CRLF checkout representation.

Vấn đề:

```text
same semantic text
≠ same raw byte hash
```

nếu Git checkout đổi EOL.

Fix ở:

```text
src/pipeline/w1_workload.py
scripts/build_w1_workload.py
scripts/check_w1_dataset.py
tests/test_w1_workload.py
```

W1 v1 lưu explicit canonical line-ending policy theo artifact để:

- giữ nguyên frozen package identity đã phát hành;
- không phụ thuộc checkout OS;
- builder và checker dùng cùng một hash implementation.

Sau fix, `check_w1_dataset.py` PASS trên Windows và frozen package SHA không đổi.

---

## 7. Offline verification trước target runtime

Trước Kaggle acceptance run:

```bash
python -m py_compile scripts/run_node1_extractor.py
python -m unittest discover -s tests -v
python scripts/check_invoice_contract.py
python scripts/check_w1_dataset.py
git diff --check
```

Kết quả:

```text
57 tests: PASS
invoice-contract-check: PASS
check-w1-workload: PASS
git diff --check: PASS
```

Test suite bao phủ:

- strict JSON;
- non-finite JSON rejection;
- response envelope failures;
- exact source binding;
- deterministic normalization;
- business-value immutability;
- endpoint/model preflight;
- transport failure;
- schema failure;
- semantic failure;
- invalid dataset envelope;
- JSON Schema ↔ Pydantic parity matrix;
- W1 cross-platform hashing.

---

## 8. Target vLLM runtime

Acceptance chạy trên Vanilla lane với:

```text
backend  = openai-compatible
base_url = http://127.0.0.1:8001/v1
model    = qwen2.5-7b-vanilla
lane     = vanilla
```

Server model:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

Repo launch script:

```text
scripts/launch_lane1_vanilla.sh
```

Các server settings giữ explicit:

```text
port                   = 8001
served-model-name      = qwen2.5-7b-vanilla
quantization           = awq
dtype                  = half
gpu-memory-utilization = 0.85
max-model-len          = 4096
max-num-seqs           = 32
```

### Runtime compatibility fix

vLLM runtime không nhận flag cũ:

```text
--disable-log-requests
```

CLI trả:

```text
unrecognized arguments: --disable-log-requests
```

Script được sửa thành:

```text
--no-enable-log-requests
```

Mục đích scientific không đổi: request logging vẫn bị tắt; đây là compatibility fix cho CLI hiện hành, không phải thay đổi workload/protocol.

---

## 9. W1 Vanilla acceptance run

Run ID:

```text
m1_03_w1_vanilla
```

Artifacts:

```text
experiments/runs/m1_03_w1_vanilla/
├── raw_responses.jsonl
├── validated_outputs.jsonl
├── failures.jsonl
└── summary.json

experiments/results/
└── m1_03_node1_vllm_summary.json
```

Kết quả:

```text
request_count             = 30
fully_validated_count     = 30
sample_failure_count      = 0
persistence_failure_count = 0
failure_counts            = {}
```

Rates:

```text
transport_success_rate = 1.0
strict_json_rate       = 1.0
source_binding_rate    = 1.0
schema_valid_rate      = 1.0
```

Acceptance:

```text
threshold = 0.95
actual    = 1.00
result    = PASS
```

Tức là:

```text
30 / 30 = 100%
```

Với 30 sample, threshold 95% yêu cầu tối thiểu 29/30; run thực tế đạt 30/30.

---

## 10. Normalization và latency quan sát được

Run ghi:

```text
normalization_count = 40
```

Latency HTTP/model request:

```text
mean   = 19048.419 ms
median = 19033.616 ms
min    = 16983.839 ms
max    = 23289.709 ms
```

Token usage:

```text
prompt_tokens     = 45843
completion_tokens = 19186
total_tokens      = 65029
```

### Claim discipline

Các latency trên chỉ là **OBSERVATION của correctness acceptance run**.

Không dùng run M1-03 này để kết luận:

- throughput khoa học;
- TTFT/TPOT benchmark;
- speculative decoding speedup;
- concurrency scaling;
- p50/p95 ổn định.

Các kết luận performance phải đi qua canonical benchmark harness/protocol riêng.

---

## 11. `schema_valid_rate` không phải business-field accuracy

Kết quả:

```text
30/30 fully validated
schema_valid_rate = 100%
```

chứng minh mọi output trong run này:

- transport thành công;
- strict JSON;
- trusted source binding;
- JSON Schema pass;
- semantic pass;
- Pydantic mirror pass.

Nó **không tự chứng minh**:

```text
100% seller accuracy
100% amount accuracy
100% date accuracy
100% field-level extraction accuracy
```

Do đó claim hợp lệ là:

> Node 1 achieved a 100% fully validated/schema-valid rate (30/30) on frozen W1 under the target Vanilla vLLM acceptance run.

Không viết:

> Node 1 achieved 100% extraction accuracy.

Nếu cần field-level correctness so với ground truth, phải có metric/evaluation protocol riêng.

---

## 12. Raw evidence integrity

Committed runtime evidence:

```text
raw_responses.jsonl     = 30 records
validated_outputs.jsonl = 30 records
failures.jsonl          = 0 records
```

Raw response được giữ riêng với canonical validated payload để:

- debug model behavior;
- audit normalization;
- audit source-binding;
- không mất evidence do canonicalization.

W1 package hash được ghi trong summary để nối run với đúng frozen workload.

---

## 13. Đánh giá hai task `[Đại]`

### Task A — Node 1 module

- [x] reusable `src/pipeline/extractor.py`
- [x] OpenAI-compatible HTTP client integration
- [x] endpoint/model preflight
- [x] strict JSON parsing
- [x] safe deterministic normalization
- [x] exact source binding
- [x] JSON Schema validation
- [x] semantic validation
- [x] Pydantic mirror validation
- [x] structured success/failure taxonomy
- [x] thin runner dùng reusable core
- [x] actual Vanilla vLLM target run
- [x] raw evidence committed

```text
TASK A: DONE WITH EVIDENCE
```

### Task B — Test suite / >=95%

- [x] unit + regression test suite
- [x] shared validation parity tests
- [x] frozen W1 30 requests
- [x] no sample removed khỏi denominator
- [x] actual target-vLLM acceptance run
- [x] schema_valid_rate = 30/30 = 100%
- [x] required threshold = 95%
- [x] failures preserved instead of hidden/cherry-picked
- [x] summary + raw artifacts committed

```text
TASK B: DONE WITH EVIDENCE
```

---

## 14. Commit/evidence chain

Các commit chính:

```text
57badbe  fix(w1): make frozen package hashing cross-platform
183d6aa  refactor(pipeline): extract reusable node1 core
4b27ffd  feat(m1-03): add reusable node1 runtime validation
f8ff030  refactor(m1-03): use reusable node1 runner
731ad13  test(m1-03): record W1 vanilla acceptance evidence
```

Evidence commit cuối:

```text
731ad13
```

Branch:

```text
feat/m1-03-mc-ocr-node1-extractor
```

Tại thời điểm worklog được lập, branch đã push lên `origin` và chứa đầy đủ runtime evidence nhưng chưa merge vào `main`.

---

## 15. File bàn giao

### Core implementation

```text
src/pipeline/extractor.py
src/pipeline/validation.py
src/pipeline/invoice_models.py
src/pipeline/openai_compatible.py
```

### Runner/tooling

```text
scripts/run_node1_extractor.py
scripts/check_invoice_contract.py
scripts/check_w1_dataset.py
scripts/init_invoice_sqlite.py
scripts/launch_lane1_vanilla.sh
```

### Tests

```text
tests/test_extractor.py
tests/test_validation.py
tests/test_invoice_models.py
tests/test_run_node1_extractor.py
tests/test_w1_workload.py
```

### W1 package

```text
experiments/configs/w1/
```

### Acceptance summary

```text
experiments/results/m1_03_node1_vllm_summary.json
```

### Raw acceptance evidence

```text
experiments/runs/m1_03_w1_vanilla/
```

---

## 16. Rủi ro / giới hạn còn lại

### 16.1. Branch chưa merge

Technical acceptance của hai task đã đủ evidence trên task branch, nhưng code chưa trở thành `main` cho tới khi PR được review/merge.

### 16.2. 100% schema-valid không phải 100% semantic ground-truth accuracy

Cần giữ claim đúng metric như §11.

### 16.3. Acceptance run không phải benchmark performance

Latency trong summary chỉ phục vụ audit run này. Không dùng để claim speedup/throughput của đề tài.

### 16.4. Raw run artifacts đang là exception có chủ đích

`experiments/runs/` vốn được `.gitignore`; raw M1-03 evidence được force-add có chủ đích để nghiệm thu/reproducibility. Không suy từ đây rằng mọi future benchmark raw log đều phải commit.

### 16.5. CLI behavior phụ thuộc vLLM version

Flag request logging đã có một compatibility drift thực tế. Các script serving tiếp theo cần tiếp tục kiểm tra CLI đúng pinned vLLM version trước khi chạy experiment.

---

## 17. Kết luận

Hai task `[Đại]` của M1-03 đã hoàn thành về mặt kỹ thuật với evidence reproducible:

```text
Node 1 implementation        : PASS
Reusable validation stack    : PASS
Unit/regression tests         : PASS
Frozen W1 integrity           : PASS
Target vLLM run              : PASS
Strict JSON rate             : 100%
Source binding rate          : 100%
Schema-valid rate            : 100%
Acceptance threshold         : 95%
Sample failures              : 0/30
```

Kết luận kỹ thuật:

```text
M1-03 / ĐẠI — TWO ASSIGNED TASKS: DONE WITH EVIDENCE
```

Trạng thái quy trình:

```text
WAITING FOR PR REVIEW / PM MERGE
```

Không tự sửa Trello; PM là người cập nhật trạng thái card/task.
