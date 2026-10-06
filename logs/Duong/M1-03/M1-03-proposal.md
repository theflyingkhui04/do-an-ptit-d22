# Proposal M1-03 — Node 1 VAT JSON Extractor & Workload W1

> **Trạng thái:** Proposal để review trước khi apply  
> **Task:** M1-03 — Xây dựng Node 1 (VAT JSON Extractor) Pipeline MC-OCR & Chuẩn bị Dataset Workload W1  
> **Branch dự kiến:** `feat/m1-03-mc-ocr-node1-extractor`  
> **Merge target:** `main`  
> **OpenSpec:** `logs/Duong/M1-03/openspec.md`  
> **Code plan:** `logs/Duong/M1-03/plan.md`  
> **Proposal đích:** `logs/Duong/M1-03/proposal.md`  
> **Lead:** Dương Nguyễn Thế — 70%  
> **Partner:** Bùi Đức Đại — 30%  
> **Ngưỡng dataset đã chốt theo OpenSpec:** W1 >= 30 mẫu  
> **Hiện tại:** Chưa tạo/sửa branch hoặc code từ proposal này.

---

# 1. Why — Vì sao cần M1-03

M0-04 đã tạo được một vertical slice khả dụng cho Node 1:

```text
MC-OCR text
  -> prompt Node 1
  -> OpenAI-compatible request
  -> JSON invoice_extraction.v1
  -> validation
  -> SQLite
```

M0-04 đồng thời đã đưa 30 mẫu dev-set và các request/output functional lên GitHub. Tuy nhiên, M0-04 mới là baseline kỹ thuật/PoC. M1-03 cần biến baseline này thành một workload W1 có contract rõ, có thể tái lập, có owner rõ ràng và đủ ổn định để M1-02 sử dụng trong benchmark.

Vấn đề M1-03 giải quyết không phải là “viết lại pipeline hóa đơn”, mà là:

1. chuẩn hóa và freeze W1 từ artifact M0-04;
2. khóa contract `invoice_extraction.v1`;
3. có Pydantic mirror và validation dùng chung;
4. có reusable `src/pipeline/extractor.py`;
5. chạy Node 1 qua target vLLM/OpenAI-compatible;
6. đạt `schema_valid_rate >= 95%`;
7. bàn giao W1 đúng semantics cho M1-02;
8. để mỗi thành viên biết **phần nào mình sở hữu và dừng ở đâu**.

---

# 2. Upstream / Downstream

## 2.1 Upstream bắt buộc: M0-04

Nguồn:

```text
feature/m0-04-vat-invoice-schema-mc-ocr
PR #2
```

M1-03 kế thừa trực tiếp:

```text
data/processed/m0_04_mcocr_devset/samples.jsonl
data/processed/m0_04_mcocr_devset/manifest.csv
data/processed/m0_04_mcocr_devset/images/
data/processed/m0_04_mcocr_devset/node1/node1_chat_requests.jsonl

src/pipeline/prompts/node1_extractor_system.md
src/pipeline/schemas/invoice_extraction.schema.json
src/pipeline/openai_compatible.py

scripts/build_node1_requests.py
scripts/run_node1_extractor.py
scripts/check_invoice_contract.py
tests/test_run_node1_extractor.py
```

M1-03 **không làm lại devset từ đầu** nếu 30 mẫu baseline đã đủ điều kiện freeze.

## 2.2 Runtime upstream: M1-01

M1-01 cung cấp target runtime vLLM.

M1-03 cần M1-01 để chạy final acceptance:

```text
Vanilla endpoint
/v1/chat/completions
served model name
T4/vLLM runtime
health/model preflight
```

M1-03 không sở hữu server launch/config speculative decoding.

## 2.3 Downstream/integration peer: M1-02

M1-02 sở hữu:

```text
experiments/benchmark_harness.py
concurrency sweep
warm-up
repeat
TTFT / TPOT / ITL
Vanilla vs Static performance matrix
experiments/results/summary.csv
```

M1-03 cung cấp:

```text
frozen W1 package
full system+user request semantics
dataset/prompt/schema identity
sample_id
thin adapter/loader
validation result
```

M1-03 không sở hữu concurrency scheduler.

## 2.4 Downstream gate: M1-04

M1-04 sẽ dùng output M1-03 + M1-02 để nghiệm thu M1.

---

# 3. What Changes — M1-03 sẽ thay đổi gì

M1-03 đề xuất các thay đổi sau:

1. Promote 30 mẫu M0-04 thành W1 candidate/final nếu đạt coverage/provenance/freeze gate.
2. Chuẩn hóa W1 dưới `experiments/configs/w1/`.
3. Tạo provenance và fingerprint cho dataset/prompt/request package.
4. Giữ JSON Schema `invoice_extraction.v1` làm source-of-truth.
5. Bổ sung Pydantic models làm typed mirror.
6. Tách validation dùng chung khỏi script SQLite.
7. Tạo reusable `src/pipeline/extractor.py`.
8. Refactor `scripts/run_node1_extractor.py` thành CLI/orchestration layer.
9. Chuẩn hóa failure taxonomy và per-sample artifacts.
10. Cung cấp thin adapter để M1-02 replay full W1 request semantics.
11. Chạy final correctness trên M1-01 Vanilla vLLM.
12. Ghi worklog, summary và report evidence.

---

# 4. Ownership — Ai làm gì

Đây là ranh giới chính thức cho M1-03.

## 4.1 Dương — Owner Data / Contract / W1 / Documentation

Dương là **Lead M1-03**, chịu trách nhiệm khóa đầu vào và contract mà Đại sẽ code theo.

### Dương sở hữu

#### A. Upstream intake

- xác nhận M0-04 commit/ref được dùng;
- xác nhận 30 mẫu baseline tồn tại và nhất quán;
- không regenerate dataset nếu chưa có lý do;
- giữ nguyên sample identity khi promote.

#### B. Dataset W1

- `experiments/configs/w1/samples.jsonl`;
- `manifest.csv`;
- `provenance.jsonl`;
- coverage summary;
- dataset fingerprint;
- dataset freeze decision;
- quyết định có cần mở rộng ngoài 30 mẫu hay không.

#### C. Prompt/package contract

- system prompt W1;
- prompt version/hash;
- request builder semantics;
- expected source metadata;
- W1 package fingerprint;
- no-label-leakage gate.

#### D. Schema / typed contract

Dương sở hữu public contract:

```text
src/pipeline/schemas/invoice_extraction.schema.json
```

và chịu trách nhiệm thiết kế/kiểm chứng Pydantic mirror:

```text
src/pipeline/invoice_models.py
```

Nếu có tranh chấp Pydantic ↔ JSON Schema:

```text
JSON Schema v1 là authoritative.
```

Dương không được tự ý breaking-change schema mà không update OpenSpec/version.

#### E. Dataset/contract verification

Dương chịu trách nhiệm các check:

```text
sample count >=30
unique sample IDs
provenance completeness
coverage report
no evaluator-label leakage
dataset hash
prompt hash
package hash
JSON Schema/Pydantic parity
semantic validation contract
```

#### F. PM / Documentation

- `logs/Duong/M1-03/openspec.md`;
- `logs/Duong/M1-03/proposal.md`;
- `logs/Duong/M1-03/plan.md`;
- `logs/Duong/M1-03/README.md`;
- cập nhật phần báo cáo/chương theo Trello;
- ghi upstream SHAs, commands và acceptance evidence.

### Dương không sở hữu

Dương **không cần** tự làm:

- concurrency scheduler M1-02;
- benchmark sweep 1→32;
- vLLM server launcher M1-01;
- speculative-decoding engine;
- Node 2;
- full implementation của `extractor.py` nếu đã bàn giao đúng contract cho Đại.

---

## 4.2 Đại — Owner Node 1 Implementation / Test / M1-02 Bridge

Đại là Partner M1-03 và là owner kỹ thuật của phần Node 1 runtime theo checklist Trello.

### Đại sở hữu

#### A. Reusable Node 1 extractor

File chính:

```text
src/pipeline/extractor.py
```

Đại chịu trách nhiệm:

- OpenAI-compatible invocation;
- endpoint/model preflight;
- strict response envelope handling;
- strict JSON parse;
- canonicalization đã được OpenSpec cho phép;
- exact source binding;
- gọi shared validation;
- trả structured success/failure.

#### B. Runner refactor

Đại chịu trách nhiệm refactor:

```text
scripts/run_node1_extractor.py
```

để:

- không chứa duplicated business logic;
- sử dụng `Node1Extractor`;
- ghi raw/validated/failure artifacts;
- tính đúng denominator;
- hỗ trợ `--no-db`;
- dùng model/base URL explicit cho M1.

#### C. Failure taxonomy

Đại implement stable failure codes:

```text
TRANSPORT_ERROR
RESPONSE_ENVELOPE_ERROR
JSON_PARSE_ERROR
SOURCE_BINDING_ERROR
SCHEMA_VALIDATION_ERROR
SEMANTIC_VALIDATION_ERROR
PERSISTENCE_ERROR
```

và preflight/run-level codes theo OpenSpec.

#### D. Test suite

Đại chịu trách nhiệm chính cho:

```text
tests/test_extractor.py
tests/test_run_node1_extractor.py
```

và phối hợp với Dương ở:

```text
tests/test_invoice_models.py
tests/test_validation.py
tests/test_w1_workload.py
```

Đại phải chứng minh:

```text
strict JSON
source binding
canonicalization safety
failure taxonomy
raw response immutability
schema-valid denominator
>=95% final structural validity
```

#### E. M1-02 bridge

Đại chịu trách nhiệm tích hợp phía M1-02 sau khi Dương bàn giao W1.

M1-03 cần ít nhất thin adapter/loader chứng minh:

```text
full system+user messages
generation settings
response_format
sample_id
```

được giữ đúng khi đi vào benchmark harness.

Concurrency/warm-up/repeat vẫn là M1-02.

### Đại không sở hữu

Đại **không được** tự ý:

- đổi sample membership;
- đổi OCR text;
- đổi prompt để “cứu” failed sample;
- đổi public JSON Schema;
- thêm business-field inference trong canonicalizer;
- xóa failed sample khỏi denominator;
- sửa W1 semantics khác nhau giữa Vanilla và Static;
- làm lại M0-04 dataset selection mà không có decision của Dương/OpenSpec.

---

# 5. Shared Responsibilities — Phần bắt buộc phối hợp

Có ba điểm không thể hoàn thành bởi một người riêng lẻ.

## S1 — Contract handshake

**Dương cung cấp:**

```text
frozen W1 candidate
invoice_extraction.v1
Pydantic contract
prompt
expected_source semantics
hashes
```

**Đại xác nhận:**

```text
extractor có thể consume contract
không cần schema workaround
không cần prompt hack
```

Nếu Đại cần breaking change để code được:

```text
STOP
-> trả lại Dương
-> update OpenSpec trước
```

## S2 — Final vLLM correctness

**Đại lead execution:**

- endpoint preflight;
- extractor run;
- failure classification;
- test/runtime debugging.

**Dương lead acceptance review:**

- đúng frozen W1;
- đúng prompt/package hash;
- đúng dataset;
- đúng denominator;
- không drift contract;
- worklog/result evidence.

Cả hai mới coi final correctness pass khi:

```text
schema_valid_rate >= 0.95
```

Nếu N=30:

```text
minimum valid = 29/30
```

## S3 — Handoff cho M1-02

**Dương bàn giao semantics/package.**

**Đại bàn giao integration bridge vào harness.**

Không ai được tự một phía tuyên bố handoff hoàn tất nếu:

- W1 package chưa freeze; hoặc
- harness chưa replay được full Node 1 semantics.

---

# 6. Handoff Gates — Điểm bàn giao và điểm dừng

Đây là phần quan trọng nhất để tránh hai người làm chồng nhau.

## H0 — M0-04 → Dương

### Input

```text
30-sample devset
manifest
prompt v1
schema v1
request builder
functional Ollama outputs
```

### Dương được bắt đầu khi

- M0-04 artifacts đọc được;
- schema/prompt baseline rõ;
- base branch strategy đã xác nhận.

---

## H1 — Dương → Đại: W1/Contract Ready

Đây là điểm **Dương dừng phần chuẩn bị kỹ thuật đầu vào** và Đại có thể code extractor dựa trên contract ổn định.

### Dương phải bàn giao

```text
experiments/configs/w1/
  samples.jsonl
  manifest.csv
  provenance.jsonl
  node1_chat_requests.jsonl
  workload.json

src/pipeline/schemas/invoice_extraction.schema.json
src/pipeline/invoice_models.py hoặc contract Pydantic đã thống nhất
src/pipeline/prompts/node1_extractor_system.md

experiments/results/m1_03_w1_dataset_summary.json
experiments/results/m1_03_node1_requests_summary.json
```

### H1 PASS khi

- sample count >=30;
- IDs unique;
- sample/manifest/provenance sets khớp;
- label leakage = 0;
- schema/Pydantic parity pass;
- prompt hash có;
- dataset hash có;
- W1 package hash có;
- request rows chứa full system+user semantics.

### Sau H1, Dương dừng ở đâu?

Dương **không tiếp tục thay prompt/dataset/schema tùy ý** trong khi Đại implement.

Nếu cần thay:

```text
unfreeze
-> bump identity
-> thông báo Đại
-> rerun relevant tests
```

---

## H2 — Đại → Dương: Extractor Ready

Đây là điểm Đại hoàn thành core Node 1 trước khi chạy final target runtime.

### Đại phải bàn giao

```text
src/pipeline/extractor.py
refactored scripts/run_node1_extractor.py
stable failure taxonomy
unit tests
local/in-memory validation evidence
```

### H2 PASS khi

- strict JSON tests pass;
- source-binding tests pass;
- canonicalization safety pass;
- validation path không cần SQLite;
- raw response immutable;
- failure codes ổn định;
- full unit/regression suite pass.

### Sau H2, Đại dừng ở đâu?

Đại không tự sửa W1 để đạt test.

Nếu lỗi do contract:

```text
STOP
-> trả Dương review
```

Nếu lỗi do extractor:

```text
Đại tiếp tục sửa trong scope Node 1.
```

---

## H3 — Dương + Đại → M1-01 runtime: Acceptance Ready

Trước full vLLM run:

### Dương xác nhận

```text
W1 vẫn frozen
hashes không đổi
sample count đúng
```

### Đại xác nhận

```text
extractor/tests pass
endpoint contract expected rõ
```

### H3 chỉ PASS khi M1-01 Vanilla có

```text
health PASS
served model tồn tại
/v1/chat/completions hoạt động
response_format/seed semantics chấp nhận được
```

Nếu M1-01 chưa sẵn sàng:

```text
M1-03 technical core có thể ready
nhưng final acceptance = BLOCKED
```

---

## H4 — Đại → Dương: Final vLLM Result

Đại chạy full frozen W1 trên target Vanilla lane.

### Output

```text
raw responses
validated outputs
failures
m1_03_node1_vllm_summary.json
```

### H4 PASS khi

```text
attempted == frozen sample count
schema_valid_rate >=0.95
N=30 -> >=29 valid
failed sample vẫn ở denominator
failure codes đầy đủ
hashes/runtime provenance đầy đủ
```

### Nếu fail <95%

Đại trước tiên phân loại lỗi.

Không được:

```text
drop failed samples
rerun riêng failed rồi gộp
silent fallback
business-rule postprocessing
```

Nếu cần prompt/schema/data change:

```text
STOP
-> trả Dương
-> unfreeze/version
-> rebuild package
-> rerun toàn bộ
```

---

## H5 — M1-03 → M1-02: W1 Integration Ready

### Dương bàn giao

```text
W1 frozen package
workload identity
prompt/schema identities
dataset provenance
```

### Đại bàn giao

```text
loader/adapter
semantic equivalence test
sample_id propagation
```

### H5 PASS khi

M1-02 có thể replay một W1 sample với:

```text
system message unchanged
user message unchanged
temperature unchanged
seed policy unchanged
response_format unchanged
token limit semantically unchanged
sample_id traceable
```

Từ đây:

```text
M1-02 sở hữu concurrency/performance benchmark.
M1-03 không tiếp tục sang scheduler/sweep.
```

Đây là **điểm dừng kỹ thuật chính của Đại trong M1-03**.

---

## H6 — Dương: Documentation / M1-03 Closeout

Sau H4/H5, Dương hoàn tất:

```text
logs/Duong/M1-03/README.md
result links
upstream SHAs
coverage
hashes
commands
final valid/attempted
M1-02 handoff status
report/chapter update
```

Đây là **điểm dừng chính của Dương trong M1-03**.

Sau H6:

```text
M1-03 -> PR review / M1-04
```

Dương không tiếp tục làm M1-02 benchmark matrix trong M1-03.

---

# 7. Responsibility Matrix

| Work item | Dương | Đại | Stop/Handoff |
|---|---|---|---|
| Pin M0-04 artifacts | **Owner** | Review | H0 |
| 30+ W1 samples | **Owner** | Consumer | H1 |
| Provenance | **Owner** | — | H1 |
| Coverage report | **Owner** | Review | H1 |
| Prompt W1 | **Owner** | Consumer | H1 |
| JSON Schema v1 | **Owner** | Consumer | H1 |
| Pydantic mirror | **Owner** | Review/test | H1 |
| W1 hashes/freeze | **Owner** | Verify | H1 |
| `extractor.py` | Review | **Owner** | H2 |
| Response parsing | — | **Owner** | H2 |
| Source binding implementation | Contract | **Owner** | H2 |
| Canonicalization implementation | Contract | **Owner** | H2 |
| Failure taxonomy | Review | **Owner** | H2 |
| Extractor unit tests | Review | **Owner** | H2 |
| Schema/Pydantic tests | **Owner** | Partner | H1/H2 |
| Final vLLM run | Acceptance review | **Execution owner** | H4 |
| `schema_valid_rate` | Verify denominator | **Compute/run** | H4 |
| W1 loader | **Contract owner** | **Implementation partner** | H5 |
| Harness adapter | Review semantics | **Owner** | H5 |
| Concurrency sweep | — | **M1-02 only** | ngoài M1-03 |
| Worklog M1-03 | **Owner** | cung cấp evidence | H6 |
| Chapter/report update | **Owner** | cung cấp số liệu | H6 |
| PR technical review | **Lead** | Partner | closeout |

---

# 8. Current Trello Status vs Proposal

Tại thời điểm proposal được tạo, checklist Trello đang thể hiện:

```text
[Dương] Dataset        COMPLETE
[Dương] Pydantic       COMPLETE

[Đại] extractor.py     INCOMPLETE
[Đại] test suite       INCOMPLETE

[Dương] W1 package     INCOMPLETE
[Dương] report         INCOMPLETE
```

Proposal hiểu trạng thái này như sau:

- Hai mục Dương đã complete về mặt task/checklist, nhưng khi apply M1-03 vẫn phải **verify/reuse/persist** output tương ứng trong branch/package cuối.
- Không coi việc checklist complete là lý do để regenerate hoặc redesign.
- Phần tiếp theo có thể chạy song song:
  - Dương khóa W1 package + provenance + hashes;
  - Đại implement extractor/test dựa trên schema/prompt đã chốt.
- H1 phải được xác nhận trước khi Đại dựa vào W1 như một frozen benchmark package.

**Lưu ý:** checklist Trello cũ vẫn có wording `50+`, nhưng proposal/OpenSpec hiện tại dùng quyết định dự án đã chốt là **30+ mẫu**. Không tự tăng ngưỡng trở lại trong implementation.

---

# 9. In Scope

M1-03 bao gồm:

```text
W1 data promotion/freeze
provenance
coverage
prompt/package identity
JSON Schema/Pydantic contract
shared validation
Node 1 extractor
Node 1 runner
failure artifacts
schema-valid metric
M1-02 handoff adapter
target vLLM correctness acceptance
M1-03 worklog/report
```

---

# 10. Out of Scope

M1-03 không bao gồm:

```text
M1-01 server implementation
speculative decoding algorithm
draft model tuning
M1-02 concurrency scheduler
warm-up/repeat orchestration
full Vanilla vs Static performance sweep
Node 2 Text-to-SQL
Node 3 BI generation
semantic field F1 nếu chưa có gold metric riêng
```

---

# 11. Stop Conditions

## Dương phải dừng và review lại nếu

- cần thay `invoice_extraction.v1` breaking semantics;
- cần đổi sample membership sau freeze;
- cần đổi prompt sau freeze;
- provenance một sample không hợp lệ;
- muốn claim VAT/items/buyer coverage mà baseline không có evidence;
- Pydantic không mirror được JSON Schema v1;
- Đại yêu cầu schema workaround để extractor chạy.

## Đại phải dừng và trả lại Dương nếu

- extractor cần đổi schema;
- extractor cần đổi prompt/data để pass;
- target backend không hỗ trợ `response_format`/seed contract;
- phải infer/sửa business facts trong canonicalizer mới đạt 95%;
- W1 request package không đủ thông tin;
- sample/source semantics mâu thuẫn.

## Cả hai phải dừng final acceptance nếu

- M1-01 runtime không ổn định;
- frozen W1 hash thay đổi giữa run;
- model/prompt/settings khác giữa samples;
- denominator bị thay đổi;
- có silent fallback;
- output final không truy vết được upstream commit/hash.

---

# 12. Success Criteria

M1-03 được xem là hoàn thành khi:

```text
W1 >=30 sample
dataset frozen
provenance đầy đủ
coverage được báo cáo trung thực
JSON Schema v1 stable
Pydantic parity pass
extractor.py reusable
validation independent of SQLite
raw/validated/failure artifacts audit được
stable error codes
final target vLLM run hoàn thành
schema_valid_rate >=95%
N=30 -> >=29/30 valid
M1-02 adapter/handoff pass
worklog/report hoàn tất
```

---

# 13. Proposed Execution Order

Có thể chạy một phần song song nhưng dependency phải giữ:

```text
DƯƠNG
M0-04 intake
  ↓
W1 candidate + provenance + hashes
  ↓
schema/Pydantic contract
  ↓
H1 CONTRACT READY
                    \
                     \
                      -> ĐẠI
                         extractor.py
                         ↓
                         tests
                         ↓
                         H2 EXTRACTOR READY
                              \
                               \
DƯƠNG verify freeze -------------+----> M1-01 Vanilla
                                        ↓
                                   final vLLM run
                                        ↓
                                   H4 >=95%
                                        ↓
                       +----------------+----------------+
                       |                                 |
                    DƯƠNG                            ĐẠI
                 closeout docs                M1-02 adapter/handoff
                       |                                 |
                       +------------- H5/H6 -------------+
                                        ↓
                                     M1-04
```

---

# 14. Proposal Decision

Proposal đề xuất M1-03 theo mô hình ownership sau:

```text
Dương = Data + Contract + W1 + Acceptance Review + Documentation
Đại  = Extractor + Runtime Validation + Test Suite + M1-02 Bridge
```

Ranh giới bàn giao cốt lõi:

```text
H1: Dương -> Đại
    W1 + schema + prompt + hashes ready

H2: Đại -> Dương
    extractor + tests ready

H4: Đại -> Dương
    final vLLM result ready

H5: M1-03 -> M1-02
    W1 adapter/handoff ready

H6: Dương
    worklog/report closeout
```

Sau H5, Đại chuyển trọng tâm sang M1-02.  
Sau H6, Dương kết thúc scope M1-03 và chuyển sang review/PR/M1-04.
