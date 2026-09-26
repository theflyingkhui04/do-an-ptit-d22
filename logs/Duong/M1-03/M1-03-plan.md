# Code Plan M1-03 — Node 1 VAT JSON Extractor & Workload W1

> **Trạng thái:** Draft implementation plan — sẵn sàng review trước khi apply  
> **Nguồn:** `logs/Duong/M1-03/openspec.md` bản đã chốt  
> **Branch dự kiến:** `feat/m1-03-mc-ocr-node1-extractor`  
> **Merge target:** `main`  
> **File đích trong repo:** `logs/Duong/M1-03/plan.md`  
> **Nguyên tắc hiện tại:** Chưa tạo branch, chưa commit, chưa sửa Git. File này chỉ là kế hoạch thực thi.

---

## 0. Mục tiêu của plan

Plan này chuyển OpenSpec M1-03 thành các task có thể thực thi tuần tự, mỗi task đều nêu:

- file/module sẽ tạo hoặc sửa;
- hành vi cần triển khai;
- dependency upstream/downstream;
- test/check để xác nhận hoàn thành;
- artifact đầu ra;
- điểm dừng nếu phát hiện contract thay đổi.

M1-03 hoàn thành khi Node 1 có thể dùng frozen Workload W1 `>=30` mẫu, chạy qua target vLLM/OpenAI-compatible, đạt `schema_valid_rate >= 0.95`, có đủ provenance/audit và bàn giao được request package cho M1-02 mà không làm lệch task semantics.

---

# 1. Snapshot hiện tại trước implementation

## 1.1 M0-04 — hard upstream

Nguồn:

```text
branch: feature/m0-04-vat-invoice-schema-mc-ocr
PR: #2
```

Tại thời điểm lập plan, PR #2 vẫn là upstream bắt buộc của M1-03.

M0-04 đã có trên GitHub:

```text
data/processed/m0_04_mcocr_devset/
├── images/
├── invoices.sqlite
├── invoices_v1.sqlite
├── manifest.csv
├── samples.jsonl
└── node1/
    ├── inputs/
    ├── node1_inputs.jsonl
    ├── node1_chat_requests.jsonl
    ├── summary.json
    └── ollama_qwen2_5_7b/
        ├── raw_responses.jsonl
        ├── validated_outputs.jsonl
        ├── failures.jsonl
        └── summary.json
```

Tracked evidence:

```text
experiments/results/m0_04_mcocr_devset_manifest.csv
experiments/results/m0_04_mcocr_devset_summary.json
experiments/results/m0_04_node1_requests_summary.json
experiments/results/m0_04_node1_ollama_summary.json
```

Contract/code baseline:

```text
src/pipeline/prompts/node1_extractor_system.md
src/pipeline/schemas/invoice_extraction.schema.json
src/pipeline/openai_compatible.py
src/pipeline/sqlite/schema.sql

scripts/prepare_mcocr_devset.py
scripts/build_node1_requests.py
scripts/run_node1_extractor.py
scripts/init_invoice_sqlite.py
scripts/check_invoice_contract.py

tests/test_run_node1_extractor.py
```

### Baseline quan trọng

M0-04 hiện có:

```text
sample_count: 30
schema_version: invoice_extraction.v1
prompt_version: node1_extractor.v1
temperature: 0
seed: 42
max_tokens: 1024
response_format: json_object
Ollama structural success: 30/30
```

Legacy prompt hash M0-04:

```text
b983d8078a64d65235c38a4245232a396f31bc9d77453615e209d3d7305da173
```

**Lưu ý:** hash legacy được tính sau `.strip()`. W1 mới dùng canonical hashing theo OpenSpec nên không được giả định hash mới phải bằng hash legacy. Nếu khác chỉ do thuật toán canonicalization, worklog phải ghi rõ.

## 1.2 Coverage thực tế của 30 mẫu M0-04

30 mẫu đã chọn hiện đều có đúng các nhóm annotation sau:

```text
SELLER:      30/30
ADDRESS:     30/30
TIMESTAMPS:  30/30
TOTAL_COST:  30/30
```

Trong tracked selection hiện tại không có annotation thể hiện trực tiếp:

```text
BUYER
ITEMS / PRODUCT LINES
VAT / TAX
INVOICE NUMBER
```

Hệ quả:

- 30 mẫu hiện tại đủ **số lượng** để làm W1 minimum.
- Chúng phù hợp workload structured JSON entropy thấp.
- Nhưng chúng không phải một tập “VAT-rich”.
- Coverage checker M1-03 phải phản ánh đúng limitation này.
- Không được bịa coverage từ model output.
- Nếu trước freeze nhóm quyết định cần buyer/items/VAT coverage khác 0, mới chạy nhánh mở rộng dataset có kiểm soát.

## 1.3 M1-01 — runtime upstream cho final acceptance

Branch hiện có:

```text
feat/m1-01-vllm-dual-lane-server
```

Artifact đã thấy:

```text
requirements-server.txt
scripts/launch_lane1_vanilla.sh
scripts/launch_lane2_static.sh
scripts/switch_lane.sh
scripts/test_lane1.py
scripts/test_lane2.py
```

Runtime contract hiện tại:

```text
Vanilla:
  port: 8001
  served model: qwen2.5-7b-vanilla

Static:
  port: 8002
  served model: qwen2.5-7b-static-draft

health:
  /health

chat:
  /v1/chat/completions
```

M1-03 dùng **Vanilla lane** làm canonical correctness lane cho final `schema_valid_rate`.

Static lane chỉ cần optional compatibility smoke; performance sweep thuộc M1-02.

## 1.4 M1-02 — downstream consumer / integration peer

Branch hiện có:

```text
feat/m1-02-concurrency-sweep-baseline
```

M1-02 đang sửa:

```text
experiments/benchmark_harness.py
experiments/requirements-m1.txt
```

Harness hiện tại vẫn tự tạo request dạng:

```python
messages = [{"role": "user", "content": args.prompt}]
```

Trong khi W1 Node 1 cần:

```text
system message
+ user message chứa SOURCE_METADATA + OCR_TEXT
+ temperature
+ seed
+ max token policy
+ response_format
+ sample_id / expected_source ngoài model payload
```

Do đó M1-03 phải cung cấp adapter/loader chứng minh request equivalence. Không sửa concurrency scheduler trong branch M1-03.

---

# 2. Kiến trúc file mục tiêu

Đây là layout đề xuất sau implementation.

```text
logs/Duong/M1-03/
├── openspec.md
├── plan.md
└── README.md

src/pipeline/
├── openai_compatible.py              # REUSE / small changes if required
├── invoice_models.py                 # NEW — Pydantic mirror
├── validation.py                     # NEW — reusable JSON Schema + semantic validation
├── extractor.py                      # NEW — reusable Node 1 core
├── w1_workload.py                    # NEW — frozen package, hashing, loader/adapter
├── prompts/
│   └── node1_extractor_system.md     # REUSE, only modify with explicit prompt bump
├── schemas/
│   └── invoice_extraction.schema.json # AUTHORITATIVE, preserve v1
└── sqlite/
    └── schema.sql                    # preserve user_version=1

scripts/
├── prepare_mcocr_devset.py           # REUSE; only run if extension is necessary
├── build_node1_requests.py           # MODIFY / generalize for W1 package
├── check_w1_dataset.py               # NEW
├── run_node1_extractor.py            # MODIFY into CLI orchestration
├── init_invoice_sqlite.py            # MODIFY to import shared validation
└── check_invoice_contract.py         # MODIFY to include Pydantic parity

experiments/configs/w1/
├── manifest.csv                      # NEW promoted/frozen manifest
├── samples.jsonl                     # NEW promoted/frozen samples
├── provenance.jsonl                  # NEW
├── node1_chat_requests.jsonl         # NEW frozen requests
└── workload.json                     # NEW identities/settings

experiments/results/
├── m1_03_w1_dataset_summary.json     # NEW
├── m1_03_node1_requests_summary.json # NEW
└── m1_03_node1_vllm_summary.json     # NEW after final run

tests/
├── test_invoice_models.py            # NEW
├── test_validation.py                # NEW
├── test_extractor.py                 # NEW
├── test_w1_workload.py               # NEW
└── test_run_node1_extractor.py       # KEEP / update imports and legacy regressions
```

Không tạo module Node 2, không sửa proposer, không sở hữu benchmark concurrency logic.

---

# 3. Quyết định implementation đã khóa

## 3.1 Contract

```text
Authoritative contract:
src/pipeline/schemas/invoice_extraction.schema.json
```

Pydantic chỉ là typed mirror.

Nếu Pydantic không biểu diễn được v1 một cách tương thích:

```text
STOP
-> không sửa schema âm thầm
-> cập nhật OpenSpec trước
```

## 3.2 Dataset

Default:

```text
reuse 30 tracked M0-04 samples
preserve sample_id
preserve exact raw_text
```

Không regenerate MC-OCR nếu chưa có lý do.

## 3.3 Frozen W1 root

```text
experiments/configs/w1/
```

## 3.4 Final correctness backend

```text
M1-01 Vanilla
base URL: http://127.0.0.1:8001/v1
served model: qwen2.5-7b-vanilla
```

Giá trị trên phải được re-check/pin lại ngay trước final run.

## 3.5 Failure codes

Primary error codes:

```text
TRANSPORT_ERROR
RESPONSE_ENVELOPE_ERROR
JSON_PARSE_ERROR
SOURCE_BINDING_ERROR
SCHEMA_VALIDATION_ERROR
SEMANTIC_VALIDATION_ERROR
PERSISTENCE_ERROR
```

Run/preflight errors:

```text
ENDPOINT_UNREACHABLE
MODEL_NOT_FOUND
BACKEND_CONTRACT_UNSUPPORTED
DATASET_INVALID
PROMPT_PACKAGE_INVALID
```

## 3.6 Hashes

W1 phải có:

```text
prompt_sha256
dataset_sha256
w1_package_sha256
```

`w1_package_sha256` không phụ thuộc lane-specific served model name hoặc local absolute path.

---

# 4. Implementation Tasks

## 4.1 Pha 0 — Pin upstream và tạo điểm xuất phát sạch

- [ ] **0.1 Kiểm tra trạng thái PR #2 M0-04 ngay trước apply.** Ghi `upstream_m0_04_commit` vào `logs/Duong/M1-03/README.md`; verify bằng `git rev-parse` trên ref thực tế và lưu SHA trong worklog.
- [ ] **0.2 Chọn base branch đúng dependency.** Nếu PR #2 đã merge thì branch M1-03 từ `origin/main`; nếu chưa merge thì branch từ `feature/m0-04-vat-invoice-schema-mc-ocr`; verify `git merge-base` cho thấy M1-03 chứa M0-04.
- [ ] **0.3 Tạo branch `feat/m1-03-mc-ocr-node1-extractor` khi bắt đầu apply.** Verify `git branch --show-current` trả đúng tên. **Không thực hiện ở bước plan hiện tại.**
- [ ] **0.4 Chạy baseline tests trước khi sửa code:** `python -m unittest discover -s tests -v`, `python scripts/validate_invoice_schema.py src/pipeline/examples/node1_extractor_output_m0_04_001.json`, `python scripts/check_invoice_contract.py`; lưu kết quả pass/fail vào worklog.
- [ ] **0.5 Chạy `python -m compileall src scripts tests` trên baseline.** Verify exit code 0 để phân biệt lỗi có sẵn và lỗi do M1-03.
- [ ] **0.6 Snapshot upstream M1-01/M1-02.** Ghi branch/ref hiện tại vào worklog; chưa cần pin final commit M1-01 cho tới lúc chạy vLLM acceptance.

**Output pha 0:** branch/base sạch + baseline test evidence.

---

## 4.2 Pha 1 — Promote M0-04 devset thành W1 candidate

### Files

```text
src/pipeline/w1_workload.py        NEW
scripts/build_node1_requests.py    MODIFY
experiments/configs/w1/*           GENERATED/TRACKED CONFIG
```

### Core API đề xuất trong `src/pipeline/w1_workload.py`

```python
normalize_lf(text: str) -> str
sha256_hex(data: bytes) -> str
canonical_prompt_bytes(text: str) -> bytes
canonical_dataset_bytes(samples: list[dict]) -> bytes
compute_dataset_sha256(samples: list[dict]) -> str
compute_w1_package_sha256(metadata: dict) -> str

load_jsonl(path: Path) -> list[dict]
write_jsonl(path: Path, rows: Iterable[dict]) -> None

build_source_metadata(sample: dict) -> dict
build_node1_input_text(sample: dict) -> str
build_chat_request(...) -> dict

load_frozen_w1_requests(path: Path) -> list[dict]
```

### Tasks

- [ ] **1.1 Tạo `src/pipeline/w1_workload.py` và chuyển các helper thuần dữ liệu từ `scripts/build_node1_requests.py` vào module reusable.** Verify existing M0-04 request-building behavior vẫn pass bằng regression test.
- [ ] **1.2 Giữ nguyên `sample_id=m0_04_001...m0_04_030` khi promote.** Verify set `sample_id` trong W1 candidate bằng đúng set của M0-04 `samples.jsonl`.
- [ ] **1.3 Giữ nguyên exact normalized OCR text theo behavior M0-04.** Verify từng `expected_source.raw_text` của W1 candidate khớp request baseline hoặc deviation được ghi rõ.
- [ ] **1.4 Generalize `scripts/build_node1_requests.py` để hỗ trợ task/workload M1-03 mà không phá CLI M0-04.** Thêm explicit options cho W1 output, manifest/provenance/workload metadata; verify legacy invocation vẫn build được M0-04 output trong temp directory.
- [ ] **1.5 Tạo `experiments/configs/w1/samples.jsonl` từ tracked M0-04 samples, không đọc lại raw Kaggle.** Verify 30 rows, unique non-empty IDs.
- [ ] **1.6 Tạo `experiments/configs/w1/manifest.csv` từ tracked M0-04 manifest.** Verify sample set giữa manifest và samples là identical.
- [ ] **1.7 Tạo `experiments/configs/w1/node1_chat_requests.jsonl`.** Verify mỗi row có `sample_id`, `schema_version`, `prompt_version`, `expected_source`, full system+user messages, `temperature=0`, `seed=42`, token limit và `response_format`.
- [ ] **1.8 Không đưa `labels`, `key_text_pairs` hoặc gold annotation vào `request.messages`.** Verify automated leakage test tìm các evaluator-only keys/tokens trong envelope nhưng không thấy chúng trong model-visible request serialization.
- [ ] **1.9 Tính `prompt_sha256` theo canonical LF-only algorithm của OpenSpec.** Verify unit test cho CRLF/LF cho cùng hash, còn thay một ký tự model-visible làm hash đổi.
- [ ] **1.10 Tính `dataset_sha256` trên sorted canonical sample records.** Verify reorder input rows không đổi hash; thay `raw_text` một sample làm hash đổi.
- [ ] **1.11 Tính `w1_package_sha256` từ dataset/prompt/schema/prompt-version/generation-settings.** Verify đổi served model name hoặc local path không đổi package hash; đổi temperature/seed/prompt/data làm hash đổi.
- [ ] **1.12 Tạo `experiments/configs/w1/workload.json`.** Verify file chứa sample count, hashes, schema/prompt version, generation settings và relative artifact paths.
- [ ] **1.13 Tạo `experiments/results/m1_03_node1_requests_summary.json`.** Verify summary tham chiếu đúng frozen package và cùng `w1_package_sha256`.

### Verification command dự kiến

```bash
python scripts/build_node1_requests.py \
  --samples-jsonl data/processed/m0_04_mcocr_devset/samples.jsonl \
  --output-dir experiments/configs/w1 \
  --report-dir experiments/results \
  --workload-id w1 \
  --task-id M1-03
```

Tên option có thể tinh chỉnh khi code, nhưng output contract không đổi.

**Output pha 1:** frozen W1 candidate package từ M0-04.

---

## 4.3 Pha 2 — Provenance, coverage và freeze gate

### Files

```text
scripts/check_w1_dataset.py
experiments/configs/w1/provenance.jsonl
experiments/results/m1_03_w1_dataset_summary.json
```

### Tasks

- [ ] **2.1 Tạo `provenance.jsonl` cho 30 baseline samples.** Mỗi row giữ `sample_id`, `source_type=mc_ocr`, source row/image ID, source dataset và permission/license note; verify join 1:1 với W1 samples.
- [ ] **2.2 Không đưa absolute path hoặc local username vào provenance/fingerprint.** Verify file chỉ dùng repo-relative/stable identifiers.
- [ ] **2.3 Ghi rõ MC-OCR là nguồn research-use và provenance phải tham chiếu điều kiện sử dụng chính thức.** Verify worklog không mô tả dataset như public-domain/unrestricted nếu chưa có căn cứ.
- [ ] **2.4 Tạo `scripts/check_w1_dataset.py`.** Checker phải validate count >=30, unique IDs, manifest/sample/provenance join, non-empty raw text, no label leakage, prompt/package hashes và cross-file consistency.
- [ ] **2.5 Sinh `m1_03_w1_dataset_summary.json` tự động.** Verify có `sample_count`, `dataset_sha256`, source counts và coverage fields.
- [ ] **2.6 Coverage summary phải báo đúng limitation baseline hiện tại:** seller/address/timestamp/total có coverage; buyer/items/VAT không được tự gán coverage từ model prediction. Verify summary không dùng LLM output làm gold coverage.
- [ ] **2.7 Chạy freeze gate.** Nếu buyer/items/tax coverage = 0, worklog phải đánh dấu limitation và decision; không tự động fail nếu OpenSpec không đặt quota.
- [ ] **2.8 [CONDITIONAL] Nếu reviewer yêu cầu coverage khác 0 cho buyer/items/VAT, mới chạy extension path.** Download/use raw MC-OCR theo quyền sử dụng, chọn bổ sung theo R1, giữ 30 baseline cũ và thêm sample mới; verify dataset hash/package hash đổi.
- [ ] **2.9 [CONDITIONAL] Nếu raw MC-OCR không có field phù hợp, không tự tạo “hóa đơn thật” không rõ nguồn.** Synthetic/real extension chỉ được thêm khi provenance hợp lệ; verify `provenance.jsonl`.
- [ ] **2.10 Đóng băng W1 candidate thành final package sau coverage review.** Verify từ thời điểm freeze, mọi thay đổi sample/prompt/settings buộc package hash đổi và final run cũ không được gộp.

### Verification command dự kiến

```bash
python scripts/check_w1_dataset.py \
  --config-dir experiments/configs/w1 \
  --summary experiments/results/m1_03_w1_dataset_summary.json
```

Expected:

```text
sample_count >= 30
manifest/sample/provenance sets identical
label leakage = 0
hash verification = PASS
```

**Output pha 2:** frozen W1 + provenance + coverage evidence.

---

## 4.4 Pha 3 — Pydantic mirror và reusable validation

### Files

```text
requirements.txt
src/pipeline/invoice_models.py
src/pipeline/validation.py
scripts/init_invoice_sqlite.py
scripts/check_invoice_contract.py
tests/test_invoice_models.py
tests/test_validation.py
```

### `invoice_models.py` model tree đề xuất

```text
MoneyModel
QuantityModel
PartyModel
InvoiceMetadataModel
TaxLineModel
LineItemModel
FieldEvidenceModel
ExtractionMetadataModel
SourceModel
InvoiceExtractionModel
```

Pydantic config:

```text
extra = forbid
```

và field constraints phải mirror JSON Schema v1.

### `validation.py` API đề xuất

```python
load_json_schema(path: Path) -> dict
json_schema_errors(payload: dict, schema_path: Path) -> list[str]
semantic_validation_errors(payload: dict) -> list[str]
pydantic_validation_errors(payload: dict) -> list[str]
validate_invoice_payload(payload: dict, schema_path: Path) -> None
```

### Tasks

- [ ] **3.1 Thêm explicit dependency `pydantic` vào `requirements.txt`.** Dùng Pydantic v2 API; verify import trong target Python environment.
- [ ] **3.2 Tạo đầy đủ nested Pydantic models mirror `invoice_extraction.v1`.** Verify canonical M0-04 example pass `InvoiceExtractionModel.model_validate`.
- [ ] **3.3 Enforce `extra='forbid'` ở mọi object public.** Verify fixture có unexpected field fail cả Pydantic và JSON Schema.
- [ ] **3.4 Mirror nullability/ranges/enums:** currency, nonnegative money/quantity, tax 0..100, confidence 0..1, line number >=1; verify invalid fixture matrix.
- [ ] **3.5 Chuyển `semantic_validation_errors` khỏi `scripts/init_invoice_sqlite.py` sang `src/pipeline/validation.py`.** Verify SQLite loader và Node 1 runner cùng dùng một validator.
- [ ] **3.6 Giữ semantic checks hiện có:** real calendar date/time, `issued_at` consistency, evidence substring, unique line numbers. Verify legacy `check_invoice_contract.py` vẫn pass.
- [ ] **3.7 Thêm Pydantic parity vào `scripts/check_invoice_contract.py`.** Verify valid example pass cả hai, invalid examples fail cả hai.
- [ ] **3.8 Tạo `tests/test_invoice_models.py`.** Bao phủ valid fixture, missing required key, additional property, enum/range/nullability.
- [ ] **3.9 Tạo `tests/test_validation.py`.** Bao phủ impossible date, date/time conflict, duplicate line number, unsupported evidence.
- [ ] **3.10 Không generate/rewrite JSON Schema từ Pydantic trong M1-03.** Verify Git diff của schema v1 không đổi trừ khi có spec update riêng.

### Verification

```bash
python scripts/check_invoice_contract.py
python scripts/validate_invoice_schema.py src/pipeline/examples/node1_extractor_output_m0_04_001.json
python -m unittest tests.test_invoice_models tests.test_validation -v
```

**Output pha 3:** typed contract + một validation stack reusable.

---

## 4.5 Pha 4 — Refactor reusable Node 1 core

### Files

```text
src/pipeline/extractor.py
scripts/run_node1_extractor.py
tests/test_extractor.py
tests/test_run_node1_extractor.py
```

### API đề xuất trong `extractor.py`

```python
class FailureCode(str, Enum): ...
@dataclass / Pydantic model Node1Failure: ...
@dataclass / Pydantic model Node1Success: ...

parse_assistant_payload(response: dict) -> tuple[dict, str]
validate_source_binding(payload: dict, expected_source: dict) -> None
normalize_canonical_payload(payload: dict, expected_source: dict) -> list[str]

class Node1Extractor:
    preflight() -> None
    run_request(request_row: dict) -> Node1Success | Node1Failure
```

Có thể dùng dataclass cho result envelope; không cần biến internal result thành public invoice schema.

### Tasks

- [ ] **4.1 Tạo `src/pipeline/extractor.py` bằng cách di chuyển logic reusable khỏi CLI hiện tại, không copy-paste song song.** Verify `scripts/run_node1_extractor.py` không còn định nghĩa duplicated parse/source/canonicalization functions.
- [ ] **4.2 Di chuyển `parse_assistant_payload`.** Verify markdown fence/prose/non-object/empty content đều ra `JSON_PARSE_ERROR` hoặc `RESPONSE_ENVELOPE_ERROR` đúng stage.
- [ ] **4.3 Di chuyển `validate_source_binding`.** Verify mismatch sample_id/img_id/image_path/raw_text được phân loại `SOURCE_BINDING_ERROR`.
- [ ] **4.4 Di chuyển `normalize_canonical_payload`.** Giữ đúng 4 repair class hiện tại; verify không thêm business inference.
- [ ] **4.5 Tạo stable `FailureCode` enum.** Verify summary có thể group theo code mà không parse message text.
- [ ] **4.6 Tạo structured success/failure result có `sample_id`, latency, raw response, canonical payload, normalizations, error code/stage/message.** Verify raw assistant response không bị mutate.
- [ ] **4.7 `Node1Extractor.preflight()` kiểm tra endpoint/model trước loop.** Verify unreachable endpoint và model missing là run-level errors, không tạo 30 sample failures giả.
- [ ] **4.8 `Node1Extractor.run_request()` gọi OpenAI-compatible client, parse → canonicalize → source bind → validate.** Verify `--no-db` path không phụ thuộc SQLite.
- [ ] **4.9 Persistence phải nằm ngoài validation core hoặc callback rõ ràng.** Verify invalid payload không thể insert SQLite.
- [ ] **4.10 Update `tests/test_run_node1_extractor.py` để import canonicalizer từ `src.pipeline.extractor`.** Giữ nguyên 3 regression M0-04 hiện có.
- [ ] **4.11 Tạo `tests/test_extractor.py` cho response envelope, parse, source binding, error taxonomy và raw immutability.**
- [ ] **4.12 Test canonicalization không sửa seller/buyer/item/money/tax values.** Tạo sentinel payload và verify business values bit-for-bit trước/sau ngoài allowed structural fields.

### Verification

```bash
python -m unittest tests.test_extractor tests.test_run_node1_extractor -v
```

**Output pha 4:** reusable Node 1 core đúng Trello target `src/pipeline/extractor.py`.

---

## 4.6 Pha 5 — Refactor runner và artifact outputs

### Files

```text
scripts/run_node1_extractor.py
experiments/results/m1_03_node1_vllm_summary.json
```

Runtime raw output nên đi vào:

```text
experiments/runs/m1_03_node1_<backend-or-lane>/
```

nếu `.gitignore`/repo policy cho phép.

### Tasks

- [ ] **5.1 Biến `scripts/run_node1_extractor.py` thành CLI thin orchestration quanh `Node1Extractor`.** Verify script vẫn nhận `--requests`, `--base-url`, `--model`, `--output-dir`, `--report`, `--limit`, `--no-db`.
- [ ] **5.2 Loại hardcode M0-04 paths/run notes khỏi reusable M1 execution.** Legacy defaults MAY giữ tương thích, nhưng M1 command phải explicit.
- [ ] **5.3 Runner đọc `workload.json` hoặc request metadata để ghi `dataset_sha256`, `prompt_sha256`, `w1_package_sha256`.** Verify final summary có đủ identities.
- [ ] **5.4 Raw responses lưu nguyên response envelope.** Verify raw artifact không chứa canonicalized payload thay thế.
- [ ] **5.5 Validated outputs lưu canonical payload + normalization audit.** Verify mỗi row join được bằng `sample_id`.
- [ ] **5.6 Failures lưu stable `error_code`, `stage`, `message`, `run_id`, model/lane và latency khi có.**
- [ ] **5.7 Summary tính riêng `transport_success_rate`, `strict_json_rate`, `source_binding_rate`, `schema_valid_rate`.**
- [ ] **5.8 `schema_valid_rate = fully_validated / attempted`; failed sample không được loại denominator.** Verify synthetic test với 29 pass/1 fail cho rate 0.9666..., 28/2 fail cho 0.9333....
- [ ] **5.9 Summary group failure count theo `error_code`.**
- [ ] **5.10 DB persistence path dùng shared validation trước insert.** Verify `--no-db` full 30 run được; DB smoke riêng cũng pass.
- [ ] **5.11 Không ghi API key/secret vào report.** Verify artifact scan không chứa configured secret.

**Output pha 5:** M1-grade runner + reproducible artifacts.

---

## 4.7 Pha 6 — W1 adapter/handoff cho M1-02

### Files

```text
src/pipeline/w1_workload.py
tests/test_w1_workload.py
logs/Duong/M1-03/README.md
```

Không sửa scheduler trong `experiments/benchmark_harness.py` trên branch M1-03.

### Adapter API đề xuất

```python
load_frozen_w1_requests(path: Path) -> list[dict]

to_benchmark_payload(
    request_row: dict,
    served_model: str,
    stream: bool = True,
) -> dict

semantic_request_view(payload: dict) -> dict
```

`semantic_request_view` dùng để so request equivalence, loại các transport-only fields:

```text
model / served model override
stream
stream_options
request IDs
lane metadata
```

nhưng phải giữ:

```text
messages
temperature
seed
max-token semantic value
response_format
n
```

### Tasks

- [ ] **6.1 Implement frozen W1 loader.** Verify load đúng 30+ requests và giữ `sample_id`.
- [ ] **6.2 Implement `to_benchmark_payload`.** Chỉ cho phép override served model name và transport streaming fields; verify system+user messages không đổi.
- [ ] **6.3 Xử lý `max_tokens` vs M1-02 `max_completion_tokens` bằng mapping semantic 1:1.** Verify numeric limit không đổi và chỉ một field thích hợp được gửi cho target backend.
- [ ] **6.4 Giữ `response_format={"type":"json_object"}` nếu target vLLM hỗ trợ.** Nếu không hỗ trợ, STOP và quay lại OpenSpec; không silent drop.
- [ ] **6.5 Giữ `temperature=0` và seed policy.** Verify Vanilla/Static payload semantic view bằng nhau cho cùng sample.
- [ ] **6.6 Tạo `tests/test_w1_workload.py` cho full-request equivalence.**
- [ ] **6.7 Tạo integration smoke fixture cho một sample W1 qua adapter.** Verify returned benchmark payload có 2 messages, không phải user-only.
- [ ] **6.8 Document handoff cho Đại/M1-02:** path package, loader import, fields được phép override và fields cấm đổi.
- [ ] **6.9 Không merge/sửa M1-02 concurrency logic trong M1-03.** Verify Git diff không đụng `experiments/benchmark_harness.py` trừ khi team quyết định integration commit riêng và update spec.

**Output pha 6:** M1-02 có adapter contract thực tế để consume W1.

---

## 4.8 Pha 7 — Full regression suite trước khi gọi model

### Tasks

- [ ] **7.1 Chạy toàn bộ unit tests:** `python -m unittest discover -s tests -v`; verify 0 failures/errors.
- [ ] **7.2 Chạy schema fixture validation:** `python scripts/validate_invoice_schema.py ...`; verify PASS.
- [ ] **7.3 Chạy contract checker:** `python scripts/check_invoice_contract.py`; verify JSON Schema ↔ Pydantic ↔ semantic ↔ SQLite pass.
- [ ] **7.4 Chạy W1 dataset checker:** verify count/hash/provenance/leakage/cross-file consistency.
- [ ] **7.5 Chạy `python -m compileall src scripts tests`; verify exit 0.
- [ ] **7.6 Rebuild W1 package lần 2 trong temp directory và so hashes.** Verify same `dataset_sha256`, `prompt_sha256`, `w1_package_sha256`.
- [ ] **7.7 So M0-04 → W1 promoted sample identity.** Verify không mất hoặc tự đổi 30 baseline IDs.
- [ ] **7.8 Kiểm tra Git diff không sửa `invoice_extraction.schema.json` ngoài ý muốn.**

**Output pha 7:** code/data contract pass hoàn toàn mà chưa phụ thuộc server.

---

## 4.9 Pha 8 — Local functional smoke

Ollama chỉ là smoke, không phải acceptance cuối.

### Tasks

- [ ] **8.1 Nếu Ollama/OpenAI-compatible local có sẵn, preflight model.** Nếu không có local backend thì skip có ghi lý do; không block final nếu vLLM sẵn.
- [ ] **8.2 Chạy 3 sample đại diện bằng frozen W1 request package.** Chọn ít nhất một sample có date+time và một sample format khác seller/layout.
- [ ] **8.3 Verify raw/validated/failure artifacts đúng schema nội bộ.**
- [ ] **8.4 Verify source binding pass và raw response bất biến.**
- [ ] **8.5 Chạy DB smoke trên 1–3 valid payload.** Verify SQLite user_version=1 và view query hoạt động.
- [ ] **8.6 Không đưa latency Ollama vào M1 benchmark conclusion.**

Command dự kiến:

```bash
python scripts/run_node1_extractor.py \
  --requests experiments/configs/w1/node1_chat_requests.jsonl \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:7b-instruct \
  --limit 3 \
  --no-db \
  --output-dir experiments/runs/m1_03_local_smoke
```

**Output pha 8:** functional smoke không phụ thuộc M1-01.

---

## 4.10 Pha 9 — Pin M1-01 runtime và preflight vLLM contract

### Tasks

- [ ] **9.1 Fetch/pin exact M1-01 commit dùng cho final run.** Ghi `upstream_m1_01_commit_used_for_vllm_acceptance` vào worklog và final summary.
- [ ] **9.2 Re-check runtime contract:** health endpoint, base URL, served model name, vLLM version; không dùng giá trị trong plan nếu branch đã đổi.
- [ ] **9.3 Start/check Vanilla lane theo M1-01 instructions.** Verify `/health` = HTTP 200.
- [ ] **9.4 Preflight `/v1/models`.** Verify `qwen2.5-7b-vanilla` hoặc served name thực tế tồn tại.
- [ ] **9.5 Gửi 1 frozen W1 request non-streaming với `response_format`, temperature 0, seed 42.** Verify HTTP 200 + strict JSON.
- [ ] **9.6 Gửi cùng sample qua adapter streaming mode dùng cho M1-02.** Verify model-visible semantics tương đương và output stream hoàn chỉnh.
- [ ] **9.7 Nếu `response_format`, seed hoặc streaming semantics không được target vLLM hỗ trợ đồng nhất:** STOP, không silent fallback; cập nhật OpenSpec/plan/package identity trước full run.

**Output pha 9:** backend contract đã xác nhận trước khi tiêu tốn full run.

---

## 4.11 Pha 10 — Final vLLM correctness acceptance

Canonical run dùng Vanilla lane.

### Command dự kiến

```bash
python scripts/run_node1_extractor.py \
  --requests experiments/configs/w1/node1_chat_requests.jsonl \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen2.5-7b-vanilla \
  --run-id m1_03_node1_vllm_vanilla \
  --no-db \
  --output-dir experiments/runs/m1_03_node1_vllm_vanilla \
  --report experiments/results/m1_03_node1_vllm_summary.json
```

### Tasks

- [ ] **10.1 Chạy toàn bộ frozen W1, không dùng `--limit`.** Verify attempted count bằng frozen sample count.
- [ ] **10.2 Verify final summary ghi exact commit/runtime/model + three hashes.**
- [ ] **10.3 Verify `schema_valid_rate >= 0.95`.**
- [ ] **10.4 Nếu N=30, verify `valid_count >=29`; 28/30 phải fail.**
- [ ] **10.5 Verify failed rows vẫn nằm trong denominator và có stable `error_code`.**
- [ ] **10.6 Verify raw/validated/failure rows join được bằng `sample_id`.**
- [ ] **10.7 Verify raw response không bị overwrite sau canonicalization.**
- [ ] **10.8 Verify final run không dùng Ollama artifact hoặc M0-04 output làm substitute.**
- [ ] **10.9 Chạy DB integration smoke bằng một subset valid outputs hoặc explicit full persistence run nếu cần downstream Node 2.** Verify SQLite insert/view regression.
- [ ] **10.10 OPTIONAL: chạy 1–3 sample trên Static lane để xác nhận request compatibility.** Không dùng result này thay M1-02 performance sweep.
- [ ] **10.11 Nếu schema-valid <95%, phân tích failure code trước khi sửa prompt.** Mọi prompt change phải bump hash/package identity và rerun toàn bộ W1; không cherry-pick failed samples để rerun rồi gộp.

**Output pha 10:** `m1_03_node1_vllm_summary.json` đủ nghiệm thu M1-03.

---

## 4.12 Pha 11 — M1-02 handoff integration check

### Tasks

- [ ] **11.1 Pin exact M1-02 commit dùng để kiểm adapter.** Ghi `upstream_m1_02_commit_used_for_adapter_integration`.
- [ ] **11.2 Re-read current `experiments/benchmark_harness.py` vì branch có thể thay đổi.**
- [ ] **11.3 Verify M1-02 có thể load ít nhất 1 frozen W1 request qua adapter mà không reconstruct prompt.**
- [ ] **11.4 Verify semantic request view của harness payload == frozen W1 semantic view.**
- [ ] **11.5 Verify `sample_id` đi xuyên qua benchmark record/log để join với Node 1 validation result.**
- [ ] **11.6 Nếu M1-02 vẫn user-only và chưa consume adapter, ghi rõ integration gap trong worklog/handoff; không sửa scheduler ở branch M1-03.**
- [ ] **11.7 Bàn giao command/snippet import cho Đại.**

**Output pha 11:** W1 integration-ready cho M1-02.

---

## 4.13 Pha 12 — Worklog, report và PR readiness

### `logs/Duong/M1-03/README.md` phải có

```text
- scope
- upstream SHAs
- branch/base strategy
- W1 sample count
- dataset/prompt/package hashes
- coverage summary
- exact commands đã chạy
- test results
- final vLLM result
- failure breakdown nếu có
- M1-02 handoff status
- deviations so với OpenSpec/plan
```

### Tasks

- [ ] **12.1 Tạo/cập nhật `logs/Duong/M1-03/README.md` với evidence, không chỉ narrative.**
- [ ] **12.2 Link đến `experiments/results/m1_03_w1_dataset_summary.json`, `m1_03_node1_requests_summary.json`, `m1_03_node1_vllm_summary.json`.**
- [ ] **12.3 Cập nhật phần báo cáo/Chương 1 liên quan đúng Trello, chỉ claim số liệu đã đo.**
- [ ] **12.4 Không viết “accuracy >=95%” nếu chỉ đo schema validity; dùng đúng `schema_valid_rate`.**
- [ ] **12.5 Chạy full suite lần cuối:** unittest + schema + contract + W1 checker + compileall.
- [ ] **12.6 Chạy `git diff --check origin/main...HEAD`.**
- [ ] **12.7 Audit diff:** không secret, không unrelated Node2/proposer/concurrency changes, không generated heavy file ngoài policy.
- [ ] **12.8 Nếu M0-04 PR #2 merge sau khi branch M1-03 đã tạo, rebase/sync lên `origin/main`, resolve duplicate history và rerun toàn bộ gate.**
- [ ] **12.9 Chuẩn bị PR title:** `[M1-03] Hoàn thiện Node 1 VAT JSON Extractor và Workload W1`.
- [ ] **12.10 PR body liệt kê W1 hash, final valid/attempted, command checks, upstream commits và handoff M1-02.**

**Output pha 12:** branch sẵn sàng review/PR.

---

# 5. Test Matrix theo file

## `tests/test_invoice_models.py`

Phải có tối thiểu:

```text
valid canonical fixture
missing top-level key
missing nested required key
additional property
invalid document_type
currency invalid
negative money
negative quantity
VAT rate <0 / >100
confidence <0 / >1
line_number <1
status ok + missing_fields non-empty
status partial + missing_fields empty
```

## `tests/test_validation.py`

```text
valid date
impossible calendar date
valid HH:MM
valid HH:MM:SS
invalid time
issued_at/date conflict
issued_at/time conflict
duplicate item line number
evidence substring pass
evidence not substring fail
Pydantic vs JSON Schema parity
```

## `tests/test_extractor.py`

```text
response envelope missing choices
message content empty
strict JSON success
markdown wrapped JSON failure
JSON array instead of object failure
source exact pass
sample_id mismatch
img_id mismatch
image_path mismatch
raw_text mismatch
canonicalize issued_at
canonicalize null money container
canonicalize source_label
remove unsupported evidence
business sentinel values unchanged
raw assistant content unchanged
failure code mapping
```

## `tests/test_w1_workload.py`

```text
preserve 30 baseline IDs
manifest/sample join
provenance/sample join
no evaluator label leak
prompt CRLF/LF same hash
prompt content change changes hash
sample reorder same dataset hash
raw_text change changes dataset hash
local path change does not change package hash
temperature/seed change changes package hash
model name override does not change semantic task hash
adapter preserves messages
adapter preserves response_format
adapter maps max token field 1:1
stream true only changes transport fields
Vanilla/Static semantic request equality
```

## Integration checks

```text
M0-04 canonical example
SQLite insert
local 3-sample smoke
M1-01 vanilla preflight
full 30+ vLLM run
M1-02 adapter replay
```

---

# 6. Acceptance Mapping — OpenSpec → Plan

| OpenSpec | Plan task |
|---|---|
| A1/A2/A42 — W1 >=30 + reuse M0-04 | 1.2–1.7, 2.4–2.10 |
| A3/A36 — provenance | 2.1–2.4 |
| A4 — no label leak | 1.8, 2.4 |
| A5/A15/A38 — hashes | 1.9–1.13 |
| A6/A41 — coverage | 2.5–2.10 |
| A7/A37 — JSON Schema authoritative | 3.2–3.10 |
| A8 — Pydantic parity | 3.7–3.9 |
| A9 — strict JSON | 4.2, extractor tests |
| A10 — source binding | 4.3 |
| A11 — semantic validation | 3.5–3.9 |
| A12 — safe canonicalization | 4.4, 4.12 |
| A13 — raw audit | 4.6, 5.4–5.5 |
| A14/A40 — failure taxonomy | 4.5–4.8, 5.6/5.9 |
| A16/A17 — determinism/fairness | 1.7, 6.2–6.6 |
| A18/A44/A45 — vLLM final/Vanilla | 9.x, 10.x |
| A19/A20/A32 — >=95% denominator | 5.7–5.9, 10.3–10.5 |
| A21 — quality wording | 12.4 |
| A22/A31/A39 — M1-02 handoff | 6.x, 11.x |
| A23 — SQLite independence | 4.8–4.9, 5.10 |
| A24/A33 — tests/checks | 3.x–8.x, 12.5 |
| A25 — worklog/report | 12.1–12.4 |
| A27/A28/A43 — branch/base/pins | 0.x, 9.1, 11.1 |
| A29 — reusable extractor scope | 4.x |
| A30 — canonical W1 root | 1.x |
| A34 — PR hygiene | 12.6–12.10 |
| A35 — controlled extension | 2.7–2.9 |
| A46 — no M0-03 ownership | entire plan scope |
| A47 — M1-04 auditability | 12.x |

---

# 7. Commit Plan đề xuất

Không cần ép đúng số commit nếu implementation thực tế cần tách khác, nhưng mỗi commit nên atomic và test-pass.

## Commit 1 — W1 package foundation

```text
feat(w1): promote m0-04 devset into frozen workload package
```

Bao gồm:

```text
src/pipeline/w1_workload.py
scripts/build_node1_requests.py
scripts/check_w1_dataset.py
experiments/configs/w1/
dataset/request summaries
tests/test_w1_workload.py
```

Check trước commit:

```text
W1 checker PASS
hash reproducibility PASS
unit tests W1 PASS
```

## Commit 2 — Typed invoice contract

```text
feat(pipeline): add pydantic mirror for invoice_extraction.v1
```

Bao gồm:

```text
invoice_models.py
validation.py
requirements.txt
contract checker
validation tests
```

Check:

```text
JSON Schema/Pydantic parity PASS
SQLite contract PASS
```

## Commit 3 — Reusable Node 1

```text
refactor(pipeline): extract reusable node1 extractor core
```

Bao gồm:

```text
extractor.py
run_node1_extractor.py
test_extractor.py
legacy regression update
```

Check:

```text
extractor unit tests PASS
full unittest PASS
```

## Commit 4 — Benchmark handoff

```text
feat(w1): add benchmark request adapter and stable failure artifacts
```

Bao gồm:

```text
W1 adapter functions
request-equivalence tests
runner summary/failure taxonomy
handoff docs
```

Check:

```text
adapter semantic equivalence PASS
failure metrics tests PASS
```

## Commit 5 — Acceptance evidence

```text
test(m1-03): record target vllm node1 acceptance
```

Bao gồm compact tracked summaries only:

```text
m1_03_node1_vllm_summary.json
dataset/request summaries if updated
```

Raw heavy runs theo repo policy.

## Commit 6 — Documentation

```text
docs(m1-03): finalize worklog and benchmark handoff
```

Bao gồm:

```text
logs/Duong/M1-03/
report update
```

---

# 8. Blockers và cách xử lý

## B1 — M0-04 PR chưa merge

Không block code nếu branch từ M0-04 đúng cách.

Trước final PR:

```text
sync/rebase main
rerun all checks
```

## B2 — M1-01 Vanilla chưa chạy

Block **final vLLM acceptance**, không block:

```text
dataset
Pydantic
extractor
tests
local smoke
adapter
```

## B3 — vLLM không support response_format/seed/streaming contract

Không fallback âm thầm.

```text
STOP
update OpenSpec
bump package identity
rerun
```

## B4 — Baseline dataset thiếu buyer/items/VAT

Đã biết đây là limitation của 30 mẫu hiện tại.

Default:

```text
report honestly
do not invent labels
do not call it VAT-rich
```

Nếu reviewer yêu cầu coverage:

```text
run controlled extension path
preserve original 30
add provenance
new dataset hash
```

## B5 — Pydantic disagrees with JSON Schema

JSON Schema v1 thắng.

```text
fix Pydantic mirror
```

Nếu thực sự cần thay public contract:

```text
STOP
spec/version review
```

## B6 — M1-02 harness thay đổi giữa lúc làm

Pin commit khi integration test.

M1-03 contract không đổi theo mọi intermediate M1-02 change.

---

# 9. Definition of Ready để bắt đầu code

Có thể bắt đầu apply khi:

- [ ] OpenSpec M1-03 hiện tại được chấp nhận.
- [ ] Plan này được chấp nhận.
- [ ] Xác nhận strategy base theo trạng thái PR #2.
- [ ] Baseline M0-04 tests pass hoặc known failure đã được ghi.
- [ ] Không có yêu cầu mới làm thay đổi `invoice_extraction.v1`.

Không cần đợi M1-01/M1-02 hoàn thành để bắt đầu core implementation.

---

# 10. Definition of Done của plan

Plan được coi là thực hiện xong khi:

```text
[ ] branch đúng
[ ] M0-04 artifacts reused/pinned
[ ] W1 >=30 frozen
[ ] provenance + coverage + hashes có
[ ] JSON Schema v1 authoritative
[ ] Pydantic mirror pass
[ ] reusable extractor có
[ ] failure taxonomy có
[ ] W1 adapter cho M1-02 có
[ ] unit/regression suite pass
[ ] local smoke pass/skip có lý do
[ ] M1-01 Vanilla contract preflight pass
[ ] full W1 vLLM run hoàn thành
[ ] schema_valid_rate >=95%
[ ] nếu N=30: >=29/30 valid
[ ] worklog/report cập nhật
[ ] git diff hygiene pass
[ ] PR ready
```

---

# 11. Thứ tự thực thi ngắn gọn

```text
0. Pin/base
   ↓
1. Promote M0-04 → W1 candidate
   ↓
2. Provenance + coverage + freeze
   ↓
3. Pydantic + shared validation
   ↓
4. Reusable extractor
   ↓
5. Runner + stable artifacts
   ↓
6. M1-02 adapter
   ↓
7. Full local regression
   ↓
8. Local smoke
   ↓
9. M1-01 vLLM preflight
   ↓
10. Full vLLM acceptance
   ↓
11. M1-02 handoff check
   ↓
12. Worklog + PR readiness
```

Không đảo full vLLM run lên trước khi dataset/prompt/package được freeze, vì như vậy kết quả sẽ không có workload identity ổn định.
