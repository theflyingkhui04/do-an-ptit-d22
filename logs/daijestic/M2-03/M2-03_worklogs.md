# Worklog Kỹ Thuật: [M2-03] Lossless Verification — Token-ID Gate, Diagnostics & Shared Verifier Integration

## 0. Thông Tin Công Việc

- **Task:** `[M2-03] Xây dựng Kịch bản Kiểm chứng Lossless Verification (100% Token Match @ temp=0)`
- **Đề tài:** *Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs*
- **Người thực hiện:** Bùi Thiên Đại (daijestic) — Lead 70%, kiểm chứng, harness, thí nghiệm và evidence.
- **Partner:** Cấn Đức Khôi — 30%, shared lossless verifier và API liên quan.
- **Sprint:** Sprint 3, 05/10–14/10/2026; card Trello đến hạn 12/10/2026.
- **Ngày ghi nhận:** 13/10/2026.
- **Trello:** https://trello.com/c/ZdXW2cJ3
- **Repository:** https://github.com/theflyingkhui04/do-an-ptit-d22
- **Task branch:** `feat/m2-03-lossless-verification`.
- **Branch HEAD đã push:** `31e4fb06b3b1a3e6fbbf69fb0084f5aba3d42f54`.
- **Canonical experiment:** `m2-03-20261013-r03`.
- **Runtime Git SHA ghi trong canonical summary:** `6c5adda05a562eebfc7182ead3e7d6377737c380` — **không phải** SHA của commit refactor mới.
- **Trạng thái:** **IN PROGRESS / LOSSLESS GATE FAILED**; implementation checkpoint đã test, commit và push; chưa chứng minh lossless, chưa xác nhận hoàn thành DoD.

Đây là nhật ký của Đại, tiếp nối cách ghi evidence trong `logs/daijestic/M1-02/README.md` và `logs/daijestic/M2-01/M2-01_worklogs.md`. Worklog phân biệt kết quả thực nghiệm, chẩn đoán, test offline và công việc chưa làm. Không coi một smoke test hay replay thành scientific benchmark.

---

## 1. Bối Cảnh, Yêu Cầu Và Acceptance Criteria

### 1.1. Mục tiêu

Kiểm chứng đầu ra **Vanilla** và **Adaptive Speculative Decoding** ở `temperature=0.0` trên ít nhất 50 prompt đa dạng, gồm W1 (invoice extraction), W2 (Text-to-SQL) và W3 (free-form business analysis). Đối chiếu **generated output token IDs theo đúng thứ tự**, không chỉ text. Ghi lại vị trí token đầu tiên sai khác, finish reason, tính đầy đủ của output, môi trường và provenance.

Theo Trello và `docs/06-ke-hoach-thuc-nghiem.md` §4, kết quả kỳ vọng cho gate là **100% khớp, 0 token discrepancy**. Nếu có sai khác, phải báo FAIL; không làm tròn hay đặt ngưỡng chấp nhận tùy ý. Target logits verification được yêu cầu trong mô tả task nhưng chưa có protocol/measurement đầy đủ để kết luận.

### 1.2. Phạm vi trách nhiệm

- **Đại:** harness thu thập/đối chiếu dữ liệu, full W1/W2/W3 run, diagnostics, checksum, offline regression, báo cáo failure evidence.
- **Khôi:** module `experiments/lossless_verifier.py`, bao gồm `verify_token_sequence`; test `tests/test_lossless_verifier.py` và các helper aggregate/export.
- **Giới hạn:** Việc gọi shared verifier trong harness đã được thực hiện. **Chưa** khẳng định harness đã được chuyển hoàn toàn sang API aggregate/export của Khôi, cũng không tự thay canonical summary schema.

### 1.3. Nguyên tắc kết luận

`temperature=0` và cùng seed **không tự chứng minh** target logits, KV-cache internal state hoặc mọi lần chạy GPU phải cho chuỗi tuyệt đối giống nhau. Token-ID equality là một phép kiểm tra *empirical* nghiêm ngặt cho các execution được đo; nó không tương đương chứng minh định lý bảo toàn phân phối với sampling bất kỳ.

---

## 2. Implementation, Interface Và Kiểm Soát Dữ Liệu

### 2.1. Các artifacts quan trọng

| Artifact | Chức năng |
|---|---|
| `tests/verify_lossless.py` | Harness thu thập, strict validation và canonical comparison |
| `tests/test_verify_lossless.py` | Offline unit/contract tests cho harness |
| `tests/test_verify_lossless_evidence_guard.py` | Bảo vệ historical raw/summary khỏi overwrite |
| `experiments/lossless_verifier.py` | Shared token comparator do Khôi triển khai |
| `tests/test_lossless_verifier.py` | Unit tests cho shared verifier |
| `experiments/results/m2_lossless_verification_summary.json` | Canonical lossless result và gate |
| `experiments/runs/m2_03/m2-03-20261013-r03/` | Vanilla/Adaptive raw JSONL, reference metadata và summary |
| `experiments/runs/m2_03/SHA256SUMS` | Checksum manifest cho evidence files |
| `experiments/runs/m2_03/diagnostics/cache_ablation_r1_r3_summary.json` | Tổng hợp ablation diagnostic, không thay canonical gate |

### 2.2. Comparator contract

`compare_pair(reference, adaptive)` xác minh identity fields: `workload`, `sample_id`, `dataset_version`, `prompt_version`, `request_sha256`, lane identity, request success, prompt token IDs, output token IDs và `finish_reason`.

Output token IDs phải là một `list` không rỗng gồm số nguyên không âm; `bool` không được chấp nhận như token ID. Sai khác thứ tự token, độ dài chuỗi, hoặc missing token data đều là failure. Text equality chỉ là diagnostic, **không** quyết định PASS.

Canonical output giữ các trường: `workload`, `sample_id`, `passed`, `failures`, `first_mismatch`, `vanilla_token_count`, `adaptive_token_count`, `finish_reason_vanilla`, `finish_reason_adaptive`, `truncated`, `content_equal_diagnostic_only`.

`first_mismatch.position_zero_based` dùng index bắt đầu từ **0**; nếu một lane hết token trước, ID phía thiếu là `null`.

### 2.3. Fail-closed và provenance

- Không biến `missing token_ids`/malformed response thành PASS.
- Không coi `finish_reason=length` là full completion; phải gắn `truncated`.
- Không overwrite canonical raw JSONL/summary ở bước diagnostic hay replay.
- Lưu input hash, run identity, implementation/runtime/launcher hashes; tách Git SHA tại thời điểm **chạy GPU** khỏi commit chỉ **refactor harness** về sau.

---

## 3. Canonical Experiment r03 — Kết Quả Thực Tế

### 3.1. Cấu hình và nguồn

```text
run_id          = m2-03-20261013-r03
comparison      = vanilla vs adaptive
temperature     = 0.0
expected_pairs  = 54
W1 / W2 / W3    = 30 / 12 / 12
GPU             = Tesla T4
vLLM            = 0.30.0
PyTorch         = 2.13.0 (runtime +cu130 theo Colab evidence)
Torch CUDA      = 13.0
driver          = 580.82.07
runtime SHA     = 6c5adda05a562eebfc7182ead3e7d6377737c380
```

Đây là runtime canonical **Colab**, không phải Kaggle offline restore. Mỗi cặp có danh tính workload/prompt và output token IDs trong raw evidence. Đối chiếu giới hạn ở 54 cặp đã thực sự ghi nhận; không suy rộng sang toàn bộ stochastic sampling hoặc mọi hardware.

### 3.2. Kết quả theo workload

| Workload | Valid pairs | Exact matches | Mismatches | Truncated pairs |
|---|---:|---:|---:|---:|
| W1 | 30 | 25 | 5 | 0 |
| W2 | 12 | 12 | 0 | 7 |
| W3 | 12 | 7 | 5 | 0 |
| **Tổng** | **54** | **44** | **10** | **7** |

```text
exact_sequence_match_rate = 44 / 54 = 0.8148148148 (~81.48%)
token_id_gate_passed      = false
full_completion_verified = false
task_done                = false
logits_status            = PM_DECISION_REQUIRED
```

**FACT:** Canonical **FAIL** tiêu chí zero-discrepancy. W2 có 12 cặp token IDs khớp nhưng 7 cặp dừng vì length; không được dùng 12/12 để tuyên bố toàn bộ W2 đã hoàn tất nội dung.

**OBSERVATION:** 10 mismatch phân bổ W1=5, W3=5; không có mismatch output token-ID trong 12 cặp W2 được ghi nhận. Thời điểm sinh token đầu tiên lệch phụ thuộc request.

### 3.3. Một canonical failure cụ thể

Với `w3_business_v2_004`, canonical r03 ghi vị trí lệch đầu tiên **index 279**. Giá trị tại index này: Vanilla `79126`, Adaptive `128273`. Đây là **canonical comparison**, không phải diagnostic two-probe cold-start được thực hiện sau đó.

Không sửa/hide mismatch, không đổi gate thành text equality, không loại W3 khỏi mẫu để đạt 100%.

---

## 4. Debugging, Controlled Diagnostics Và Điều Chưa Biết

### 4.1. Câu hỏi điều tra

Vì sao với cùng prompt, seed và greedy decoding, output có thể khác? Cần phân biệt:

1. **Token prefix equality:** Hai chuỗi token đã sinh có cùng prefix.
2. **KV-cache state equivalence:** Các tensor cache thực sự tương đương theo từng layer/head/position.
3. **Target logits equivalence:** Target model tạo cùng logits tại vị trí cần chọn token.

(1) **không tự động chứng minh** (2) hoặc (3). Prefix giống nhau không quan sát cache layout, scheduling, các phép tính GPU hoặc logits. Không quy nguyên nhân mismatch chỉ từ token trace.

### 4.2. Cold-start cache ON/OFF ablation — chỉ diagnostic

Tập trung **một prompt** `w3_business_v2_004`, fixed `k=4`, hai requests liên tiếp sau cold-start, lặp R1–R3. Tham chiếu `experiments/runs/m2_03/diagnostics/cache_ablation_r1_r3_summary.json`.

| Replication | Prefix cache ON — attempt 1/2 | Prefix cache OFF — attempt 1/2 |
|---|---|---|
| R1 | MATCH / FAIL | MATCH / MATCH |
| R2 | MATCH / FAIL | MATCH / MATCH |
| R3 | MATCH / FAIL | MATCH / MATCH |

Tổng cộng **6 diagnostic cells validated**, mỗi cell có 2 attempts. Khi cache ON, attempt thứ hai lệch Vanilla tại **zero-based index 200**; cache OFF cho hai attempts cùng khớp Vanilla. Các trace được đánh dấu `trace_reconstruction=true` trong summary. Đây **không phải** 54-prompt canonical rerun.

**OBSERVATION:** Outcome có tương quan lặp lại với trạng thái prefix caching trong thiết lập diagnostic cụ thể.

**INFERENCE, chưa phải root cause:** Cache policy hoặc history có thể góp phần làm thay đổi execution state. Chưa chứng minh KV-cache bị corrupt hoặc vLLM prefix caching implementation sai. Cần so sánh trạng thái cache/logits hoặc tracing sâu hơn theo protocol được chốt.

Với attempt cache-ON gặp mismatch, acceptance delta tại speculative step 13 được quan sát **sau** divergence; không được coi nó là nguyên nhân của token sai đầu tiên.

### 4.3. Evidence của lỗi vận hành, không xóa khỏi nhật ký

R2/R3 từng có failed-prestart do thiếu lane 3 sau restart. Các attempt lỗi được archive dưới `*.failed_prestart_missing_lane3/`, kèm manifest/log/run state, thay vì xóa hoặc ghi đè. Sau phục hồi và kiểm tra điều kiện start, các diagnostic cells hợp lệ mới được tổng hợp. Đây là **operational failures**, không được đếm như valid measurement cells.

Các Colab runtime snapshots, logs và local diff backups được bảo toàn trong diagnostics. Không đánh đồng snapshot scripts/logs với source tree frozen của canonical r03.

---

## 5. Khôi Phục Evidence Sang Kaggle Và Offline Validation

### 5.1. Vì sao sử dụng Kaggle

Sau khi Colab runtime bị reset/gián đoạn, checkpoint trên GitHub giúp giữ canonical raw evidence. Đại checkout task branch tại Kaggle và kiểm tra integrity trước khi sửa code.

```text
Branch checkpoint = 03a32e4
SHA256SUMS        = 114/114 PASS
Runtime snapshots = 4 Git blob checks PASS
Git status        = CLEAN trước chỉnh sửa
```

Kaggle runtime khi restore:

```text
GPU       = 2 x Tesla T4 (15,360 MiB mỗi GPU)
Python    = 3.13.15
PyTorch   = 2.11.0+cu128
CUDA      = 12.8 (Torch runtime)
driver    = 580.178.04
vLLM      = NOT INSTALLED
```

**CONFLICT / LIMITATION:** Kaggle **khác stack** Colab r03 (`vLLM 0.30.0`, `PyTorch 2.13.0+cu130`, driver 580.82.07). Do đó Kaggle chỉ phục vụ **offline tests, replay và Git review**, không được ghi là canonical GPU replication. Không cài dependency chưa pin để tạo ra benchmark không tương đương.

### 5.2. Offline test baseline

```bash
python -m pytest -q \
  tests/test_verify_lossless.py \
  tests/test_lossless_verifier.py \
  tests/test_verify_lossless_evidence_guard.py
```

Kết quả ban đầu: **32 passed**.

---

## 6. TDD — Tích Hợp Shared Verifier Mà Không Làm Drift Canonical Contract

### 6.1. Vấn đề thiết kế

Trước refactor, `tests/verify_lossless.py::compare_pair()` tự duyệt token sequence bằng vòng lặp và tìm mismatch đầu tiên. Trong khi đó Khôi đã triển khai `experiments.lossless_verifier.verify_token_sequence()`. Duplicated comparison logic có nguy cơ drift nếu hai nơi sửa độc lập.

**Quyết định trong phạm vi subtask:** Giữ strict input/identity/finish-reason validation của harness, nhưng **delegate riêng phần output token sequence comparison** cho shared verifier. Không sửa canonical schema hoặc threshold.

### 6.2. RED — Test mới phải fail trước khi implementation đổi

Thêm vào `tests/test_verify_lossless.py`:

```text
test_compare_pair_delegates_to_shared_verifier
```

Test patch `shared_verifier.verify_token_sequence` với spy, kiểm tra gọi đúng một lần, kiểm tra token args và ánh xạ `first_mismatch` của canonical JSON.

Trước thay đổi implementation:

```text
1 failed, 22 deselected  (targeted test)
1 failed, 32 passed      (full 3-file suite)
Failure: expected shared verifier to be called once; actual 0 calls
```

Đây là **RED hợp lệ** vì test phát hiện chính xác hành vi cần thay đổi, không phải syntax error/environment error.

### 6.3. GREEN — Implementation

`tests/verify_lossless.py` import:

```python
from experiments import lossless_verifier as shared_verifier
```

Trong `compare_pair()`:

```python
verification = shared_verifier.verify_token_sequence(
    vanilla_tokens=ref_ids,
    adaptive_tokens=ada_ids,
    prompt_id=adaptive.get("sample_id", ""),
    workload=adaptive.get("workload", "W1"),
)
```

Khi `is_match=false`, ánh xạ `first_discrepancy.index`, `expected_token_id`, `actual_token_id` về canonical `first_mismatch`; gắn `generated_token_ids_mismatch`. Nếu verifier báo mismatch mà thiếu discrepancy thì raise `VerificationError` (fail-closed). Chỉ gọi shared verifier sau khi token IDs qua validation; không bỏ các checks còn lại.

### 6.4. Regression result

```text
33 passed in 2.94s
```

`git diff --check`: PASS. Final code review: hai files, không phát hiện BLOCKER/MAJOR/MINOR; comment NIT về RED được sửa trước commit.

### 6.5. Behavioral equivalence — Replay 54 canonical pairs

Một offline replay đọc **các raw JSONL đã checksum**, index theo `(workload, sample_id)`, gọi `compare_pair()` phiên bản mới và so sánh toàn bộ dict kết quả với từng object trong **historical canonical summary**.

```text
REPLAY: PASS (54/54 comparison objects unchanged)
CANONICAL_GATE: FAIL preserved (44/54 exact)
TRUNCATED_PAIRS: 7
HISTORICAL_SUMMARY: UNCHANGED
```

**FACT:** 54 comparison objects khớp hoàn toàn trong replay này; summary SHA256 được kiểm tra không đổi. Điều này là **regression equivalence trên dataset hiện có**, không chứng minh mọi input/failure mode đều bảo toàn và không chứng minh lossless.

---

## 7. Git Timeline, Review Và Pull Request Scope

### 7.1. Checkpoint history trên branch M2-03

```text
56ab2ec  test(m2-03): preserve lossless harness and failure evidence
bf1fe66  chore(m2-03): checkpoint raw diagnostics and cache replication evidence
03a32e4  chore(m2-03): preserve Colab pre-reset runtime snapshots
31e4fb0  test(m2-03): delegate token comparison to shared verifier
```

Commit mới `31e4fb0` đã tạo và push thành công lên `origin/feat/m2-03-lossless-verification`; commit chứa **đúng 2 file**:

```text
tests/verify_lossless.py
tests/test_verify_lossless.py
2 files changed, 58 insertions(+), 11 deletions(-)
```

Sau commit: `git status --short` không có output. GitHub remote SHA được xác minh khớp `31e4fb06b3b1a3e6fbbf69fb0084f5aba3d42f54`.

### 7.2. Compare với `develop`

Sau khi tạo `origin/develop` remote-tracking ref (Kaggle `--single-branch` ban đầu thiếu ref):

```text
origin/develop..HEAD   = 4 commits
PR diff               = 158 files, 18,208 insertions
working tree          = CLEAN
```

Tại thời điểm kiểm tra, branch M2-03 ahead `develop` 4 commits, behind 1 commit. Commit phía `develop` bổ sung `experiments/notebooks/kaggle_benchmark_eagle3.ipynb`, không trực tiếp chạm hai files shared-comparator vừa review. Các con số ahead/behind có thể thay đổi sau khi team push tiếp; phải fetch và review lại trước merge.

**PR hiện chưa có** tại thời điểm kiểm tra; đây là **planned merge**, không ghi thành merged evidence. 158 files bao gồm harness, raw JSONL, summary, diagnostics, server logs, `.py.bak`, failed-prestart archives và nhiều scripts; cần review phạm vi, không đánh đồng với commit 2-file của integration.

### 7.3. Secret audit và giới hạn

Regex scan trên **added diff lines** kiểm tra GitHub token patterns, OpenAI keys, AWS access keys và Authorization headers:

```text
NO PATTERN MATCHES DETECTED
```

Đây **không phải comprehensive secret scan**, không bao phủ mọi secret shape, lịch sử cũ hoặc dữ liệu nhạy cảm trong logs. GitHub PAT từng xuất hiện trong quá trình thao tác trước đây; phải xác minh token cũ đã revoke/rotate, không bao giờ đưa credential vào source, PR hay worklog.

### 7.4. Quyền merge — cập nhật theo xác nhận của Đại

**Đại được tự quyết định mở PR, review và merge vào `develop` theo quyền đã được team giao; không cần xin PM chỉ để merge `develop`.** Chỉ merge vào `main` mới cần PM duyệt. Quyền này không tự động cho phép thay đổi scientific protocol/acceptance criteria hoặc đánh dấu Trello DONE.

Trước merge `develop`: review toàn bộ 158 files, bảo đảm không có secret, chạy CI/relevant tests trên head mới nhất và mô tả rõ **WIP / negative result**. Nếu repository có branch protection/check bắt buộc thì tuân thủ. Không force-push hoặc merge một PR đang xung đột chưa hiểu nguyên nhân.

---

## 8. Findings, Severity Và Giới Hạn Claim

| Finding | Severity | Evidence | Ý nghĩa |
|---|---|---|---|
| Canonical lossless gate FAIL 44/54 | **BLOCKER — acceptance** | Canonical summary r03 | Không tuyên bố task đạt zero-discrepancy |
| 7 cặp W2 truncated | **MAJOR — completeness** | `finish_reason=length` | Exact prefix không xác nhận full completion |
| Target logits chưa đo/so sánh | **MAJOR — scope/protocol** | `logits_status=PM_DECISION_REQUIRED` | Chưa chứng minh logits equivalence |
| Cache ON/OFF outcome khác biệt | **HYPOTHESIS / diagnostic** | R1–R3 six validated cells | Không được quy ngay root cause là cache bug |
| Kaggle khác runtime canonical | **NON-BLOCKING cho offline test**, BLOCKER nếu dùng như canonical replication | Runtime versions | Không so trực tiếp runtime khác nhau |
| 158-file PR gồm artifacts lớn | **REVIEW RISK** | Git diff | Cần kiểm soát security, purpose, provenance |

**Defense guardrail:** Cùng token prefix không chứng minh cùng KV-cache; cùng KV-cache được giả định cũng không thay thế phép đo target logits; cùng logits gần đúng chưa đồng nghĩa cùng token trong mọi chế độ tie-breaking/precision. Kết luận phải giới hạn ở điều thực sự đã đo.

---

## 9. Reproducibility Checklist & Evidence Index

Các thuộc tính canonical có trong summary/raw/provenance gồm: workload và frozen input package hashes, request identity, seed/temperature theo request, output/prompt token IDs, finish reason, runtime SHA, hardware, vLLM/Torch/CUDA/driver versions, launcher/implementation hashes, timestamps và raw SHA256. Không được thay thế các giá trị runtime SHA bằng SHA của refactor về sau.

Evidence chính:

- `experiments/results/m2_lossless_verification_summary.json`
- `experiments/runs/m2_03/m2-03-20261013-r03/vanilla.jsonl`
- `experiments/runs/m2_03/m2-03-20261013-r03/adaptive.jsonl`
- `experiments/runs/m2_03/m2-03-20261013-r03/reference_meta.json`
- `experiments/runs/m2_03/SHA256SUMS`
- `experiments/runs/m2_03/diagnostics/cache_ablation_r1_r3_summary.json`
- `tests/verify_lossless.py`, `tests/test_verify_lossless.py`
- `experiments/lossless_verifier.py`, `tests/test_lossless_verifier.py`
- Commit `31e4fb0`, branch `feat/m2-03-lossless-verification`

Không lưu raw secrets trong worklog. Chỉ dùng paths và commit identifiers để người khác tái hiện quá trình kiểm chứng.

---

## 10. Definition Of Done Và Bàn Giao

| Acceptance / subtask | Trạng thái | Evidence hoặc gap |
|---|---|---|
| Harness so khớp token IDs trên 50+ prompts W1/W2/W3 | **DONE WITH EVIDENCE cho implementation/execution** | 54 valid pairs; tests và raw |
| Shared verifier integration | **DONE WITH EVIDENCE cho integration** | TDD RED→GREEN 33 PASS, replay 54/54, commit/push `31e4fb0` |
| Zero discrepancy trên canonical prompts | **NOT MET** | 44/54 match; 10 mismatches |
| Full-completion claim | **NOT VERIFIED** | 7 truncated pairs |
| Compare target logits | **PENDING PROTOCOL** | Chưa có raw logits evidence |
| Canonical raw + summary lưu Git | **DONE WITH EVIDENCE** | 4 checkpoint commits; checksum |
| Chương 4: phương pháp luận lossless | **NOT VERIFIED / cần cập nhật bản thảo** | Chưa có evidence Chương 4 được review |
| PR/merge vào `develop` | **NOT YET DONE tại ngày ghi nhận** | Có quyền tự quyết; cần PR review/CI |
| Merge vào `main` | **NOT AUTHORIZED WITHOUT PM APPROVAL** | Không thực hiện |
| Trello DoD | **IN PROGRESS** | PM là người quản lý card; không tự chỉnh Trello |

### Các quyết định còn chờ PM (không phải xin quyền merge `develop`)

1. Chốt **sequential vs simultaneous lane protocol**, nhất là tính tương đương cache history.
2. Chốt **target logits measurement và comparison tolerance/format** theo pinned runtime; không tự đổi acceptance gate.
3. Chốt hướng xử lý **44/54 negative result** và **7 truncated pairs**: remedial investigation, boundaries và evidence bổ sung; không xóa result hoặc cherry-pick samples.

### Hành động tiếp theo (đề xuất)

Tạo và review **worklog này** trong `logs/daijestic/M2-03/M2-03_worklogs.md`, kiểm tra staged diff. Sau đó Đại có thể tự quyết định mở PR vào `develop` với mô tả rõ WIP/negative results, review security/artifact scope và merge khi branch checks đáp ứng. Không sửa Trello, không chuyển M2-04 và không tuyên bố M2-03 DONE khi canonical acceptance chưa đạt.

---

## 11. Retrospective Và Câu Hỏi Bảo Vệ

**Điều hiệu quả:** Checkpoint raw evidence kèm checksums trước runtime reset; giữ cả unsuccessful diagnostic attempts; TDD test phải RED vì đúng nguyên nhân; replay toàn bộ historical comparison objects để xác minh refactor không drift result.

**Điều cần cải thiện:** Tách canonical versus diagnostic outputs và compute/runtime environments rõ hơn; tránh PR tích tụ hàng trăm files thiếu ranh giới review; xác định sớm logits protocol cùng conditions cho cache-history equivalence.

**Câu hỏi bảo vệ 1:** “44/54 match có nghĩa thuật toán đã lossless chưa?” — **Không**. Acceptance gate quy định 100%, nên result hiện tại FAIL. Chỉ được nói phần lớn cặp được đo khớp, với 10 cặp không khớp; chưa xác định root cause.

**Câu hỏi bảo vệ 2:** “W2 đạt 12/12 sao vẫn chưa chứng minh full completion?” — Vì 7 cặp kết thúc bằng `length`; chuỗi có thể khớp trong phạm vi generation limit nhưng chưa chứng minh completion đầy đủ.

**Câu hỏi bảo vệ 3:** “Cache OFF chạy giống nhau, vậy lỗi chắc chắn do prefix cache?” — **Không**. R1–R3 hỗ trợ cache-associated effect trên một prompt, không đo KV tensors/target logits và chưa cô lập mọi confounder.

**Câu hỏi bảo vệ 4:** “Vì sao 33 tests PASS mà task vẫn FAIL?” — Unit tests chứng minh comparator/harness xử lý dữ liệu đúng theo contract. Canonical GPU run kiểm chứng hành vi model runtime và có 10 mismatches. Hai cấp evidence khác nhau, không triệt tiêu nhau.

**Câu hỏi bảo vệ 5:** “Vì sao vẫn merge code vào `develop`?” — Có thể tích hợp implementation, test và negative-result evidence phục vụ nghiên cứu, nhưng merge **không** được dùng để tuyên bố 100% token match hay hoàn thành DoD. `main` vẫn theo quy trình PM approval.
