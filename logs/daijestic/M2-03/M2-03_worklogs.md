# Worklog Kỹ Thuật & Handoff: [M2-03] Lossless Verification trên Adaptive Speculative Decoding

## 0. Thông Tin Công Việc

- **Task:** `[M2-03] Xây dựng Kịch bản Kiểm chứng Lossless Verification (100% Token Match @ temp=0)`
- **Người thực hiện chính:** Bùi Thiên Đại (daijestic) — Lead phần kiểm chứng lossless / forensic debugging
- **PM / Partner:** Cấn Đức Khôi
- **Sprint gốc:** Sprint 3 · M2 Adaptive v0 & Lossless
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Branch điều tra ban đầu:** `feat/m2-03-lossless-verification`
- **Branch closure trung gian:** `fix/m2-03-lossless-closure`
- **Branch clean / source-of-truth hiện tại:** `fix/m2-03-lossless-clean`
- **Remote tip hiện tại:** `9dd82330cc8e538a24c4a5c45a5287f5c45133e5`
- **GPU chính đã dùng:** NVIDIA Tesla T4, Turing, compute capability 7.5 (sm75)
- **Runtime chính:** vLLM 0.30.0, CUDA/PyTorch theo frozen execution evidence của từng run
- **Target model:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Target revision:** `b25037543e9394b818fdfca67ab2a00ecc7dd641`
- **Draft model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Draft revision:** `7ae557604adf67be50417f59c2c2f167def9a775`
- **Frozen Protocol-B plan SHA256:** `6d9af3ced74d37dc73928b2fee4ede9e3e95048d8d6445cabb8ecedc38e17389`
- **Acceptance gate khoa học:** Vanilla vs Adaptive phải khớp **54/54 token sequences** ở `temperature=0` theo frozen prompt set W1/W2/W3.
- **Kết quả canonical mới nhất trên T4:** **45/54 exact**, 9 deterministic mismatches còn lại.
- **Trạng thái Trello/PM hiện tại:** `Blocked · Đã có hướng giải quyết: DEFERRED (Hoãn sang Sprint cuối)`.
- **Trạng thái kỹ thuật:** **M2-03 chưa DONE theo acceptance 54/54. Giai đoạn điều tra T4 đã được PM dừng/re-scope; evidence đã được preserve để quay lại ở sprint cuối trên GPU mới hơn.**
- **Ngày handoff tạm đóng:** 23/10/2026.

Tài liệu này được viết để Dương/Khôi/người khác chưa theo dõi các cuộc debug trước vẫn có thể hiểu: task là gì, tại sao mất nhiều vòng điều tra, source-of-truth nào được dùng, code/evidence ở đâu, vấn đề nào đã loại được, vấn đề nào còn UNKNOWN, và khi mở lại task phải bắt đầu từ đâu.

---

## 1. Source Of Truth Và Quy Tắc Đọc Handoff

### 1.1. Thứ tự ưu tiên nguồn

Khi có xung đột, đọc theo thứ tự:

1. **PM decision mới nhất** — quyết định scope / dừng / defer.
2. **Trello card M2-03** — trạng thái quản lý chính thức.
3. **Frozen experiment protocol** — đặc biệt `docs/06-ke-hoach-thuc-nghiem.md` và Protocol-B artifacts.
4. **Execution SHA + raw evidence thực chạy** — mạnh hơn code local không chạy.
5. **Runbooks / authenticated transforms / tests**.
6. **Worklog và handoff** — giải thích bối cảnh, không thay thế raw evidence.
7. **External/upstream docs** — chỉ là supporting context, không được dùng thay project-local evidence.

### 1.2. Nguồn quản lý chính thức hiện tại

Trello card M2-03 hiện ghi:

```text
Status: Blocked · Đã có hướng giải quyết: DEFERRED
Sprint gốc: Sprint 3 · M2 Adaptive v0 & Lossless
PM decision: dừng thực nghiệm T4 trong sprint hiện tại,
preserve evidence, chuyển mục tiêu 54/54 sang sprint cuối trên GPU mới hơn.
```

**Lưu ý quan trọng:** PM decision là source-of-truth cho **scope và tiến độ**, nhưng các câu diễn giải kỹ thuật trong Trello như “Turing là root cause” không tự động trở thành FACT khoa học. Raw evidence hiện chỉ cho phép kết luận hẹp hơn, được ghi ở §16.

### 1.3. Frozen protocol

`docs/06-ke-hoach-thuc-nghiem.md` quy định lossless verification:

```text
>= 50 prompts, W1/W2/W3
temperature = 0
seed cố định
Vanilla và Adaptive dùng cùng prompt
nếu có mismatch thì đó là acceptance failure, không phải sai số chấp nhận được
```

M2-03 về sau dùng frozen set **54 prompts** và Protocol-B paired verification để loại bỏ nhầm lẫn do run-to-run noise.

---

## 2. Mục Tiêu, Acceptance Và Điều Không Được Đánh Đồng

### 2.1. Mục tiêu

Kiểm chứng Adaptive Speculative Decoding giữ nguyên output của Vanilla ở `temperature=0`.

Acceptance thực nghiệm:

```text
Vanilla token IDs == Adaptive token IDs
cho toàn bộ 54 prompts

PASS = 54/54
FAIL = chỉ cần 1 prompt lệch
```

### 2.2. Những thứ KHÔNG đồng nghĩa task DONE

```text
unit tests PASS            != 54/54 empirical PASS
server HTTP 200            != lossless PASS
speculation k>0 hoạt động  != lossless PASS
targeted case PASS         != canonical PASS
Git push thành công        != bug fixed
observer chạy xong         != evidence hợp lệ
```

### 2.3. Trạng thái cuối hiện tại

```text
Canonical rowwise-v2:     45/54  -> FAIL
Vanilla repeatability:    54/54  -> PASS
Adaptive repeatability:   54/54  -> PASS
Remaining mismatches:     9 deterministic
General root cause:       UNKNOWN
Controlled P177 cause:    layer0 attention path causal cho 1 case
Broad M005 A/B:           INVALID_EVIDENCE / observer non-neutral
Stage1 neutral observer:  local implementation only, GPU NOT RUN
PM status:                DEFERRED / re-scoped
M2-03 DONE:               NO
```

---

## 3. Bản Đồ Branch Và Vì Sao Phải Tách Branch

```text
feat/m2-03-lossless-verification
│
│  lossless verifier
│  cache ON/OFF
│  KV forensics
│  CUDA graph / PIECEWISE controls
│  index368 target-forward shape
│  attention dispatch / forced-3D / q3
│
│  tip: a7b52e7
│
├──────────────► fix/m2-03-lossless-closure
│                │
│                │ canonical/index81 reproduction checks
│                │ same-process Vanilla control
│                │
│                └─ tip: 56604cd
│
└──────────────► fix/m2-03-lossless-clean
                 │
                 │ authenticated Protocol-B baseline 44/54
                 │ boundary279 -> layer1 KV
                 │ draft tokenizer revision fix
                 │ early timeline -> P177
                 │ k=0 control
                 │ layer0 attention substitution causal test
                 │ production rowwise-v2
                 │ targeted PASS
                 │ canonical 45/54
                 │ m0_04_005 A/B regression diagnostic
                 │ lifecycle bug fix
                 │ broad observer non-neutral evidence
                 │
                 └─ remote tip: 9dd8233
```

Lý do mở `fix/m2-03-lossless-clean`: branch đầu đã tích lũy rất nhiều instrumentation và experiments. Để kết luận khoa học không bị lẫn planned code, stale observer, historical transforms và execution code, nhóm tạo clean branch, authenticate source, chạy paired verification rồi chỉ deep-read phần liên quan.

---

## 4. Phase 1 — Phát Hiện Mismatch Và Loại Cache Là Root Cause Duy Nhất

### 4.1. Lossless verifier đầu tiên

Các mốc:

```text
27fc09e  feat: add lossless verifier module and tests
56ab2ec  test(m2-03): preserve lossless harness and failure evidence
```

Mục đích ban đầu là chuyển từ “Adaptive server chạy được” sang “Adaptive có lossless thật không”. Verifier so token IDs thay vì chỉ so text hậu xử lý.

**OBSERVATION:** xuất hiện mismatch giữa Vanilla và Adaptive.

**Kết luận:** task chuyển sang forensic debugging; unit tests/controller logic không đủ làm acceptance evidence.

### 4.2. Giả thuyết prefix cache / KV cache

Các mốc chính:

```text
c9186ee  cache lookup probe
35b6db6  preserve Cache ON lookup diagnostic
1066bcd  preserve Cache OFF lookup control evidence
990d0b1  checkpoint isolated V1 cache-off verification
bdb26dd  preserve full cache-off 47-of-54 evidence
```

Hypothesis:

```text
Adaptive dùng prefix/KV cache khác Vanilla
-> internal state khác
-> logits khác
-> token khác
```

Minimal discriminator là Cache OFF.

Kết quả:

```text
Cache OFF = 47/54
```

**FACT:** mismatch vẫn tồn tại khi cache-off.

**Kết luận:** prefix cache/cache lookup **không phải root cause duy nhất**.

**Quan trọng:** `47/54` là historical Cache-OFF regime, không phải production baseline về sau.

---

## 5. Phase 1 — KV / Execution-State Forensics

Sau cache-off, câu hỏi chuyển từ token-level sang model-state:

```text
Token prefix equality
    không tự chứng minh
KV-cache state equivalence
    không tự chứng minh
Target logits equivalence
```

### 5.1. KV probes

Các mốc:

```text
78713f2  add KV recompute diagnostic control
bbc979e  preserve index60 GPU diagnostic evidence
f2d75e7  bind KV boundary capture to scheduler dispatch
a0274c9  preserve r08 valid KV diagnostic
```

Trong quá trình này, nhiều lỗi không phải production bug mà là **diagnostic binding bug**:

- stale pre-ready token bị observer liên kết nhầm với API output;
- scheduler row / producer-consumer lineage không được bind chặt;
- refill/discard của KV làm raw storage không thể đọc ngây thơ;
- request-level telemetry không đủ xác định logits row nào tương ứng generated token nào.

Các fix diagnostic liên quan:

```text
4e09ed0  accept proven KV refill discard
6560c2f  complete pinned draft model inventory
c493723  exclude stale pre-ready token from API linkage
```

Bài học: **cùng prompt/cùng seed/temp=0 không đảm bảo cùng internal state; observer phải chứng minh lineage trước khi đọc tensor.**

---

## 6. Phase 1 — CUDA Graph / PIECEWISE / Forward-Shape Controls

Các mốc:

```text
5787388  add Vanilla PIECEWISE diagnostic control
8b208bf  preserve PIECEWISE control GPU evidence
c0364c1  isolate Adaptive CUDA graph execution
feffdf5  preserve Adaptive no-cudagraph diagnostic
```

Hypothesis: graph regime hoặc capture policy làm Vanilla/Adaptive khác nhau.

Kết quả: thay CUDA graph regime không tự động khôi phục equality.

**Kết luận:** CUDA graph **alone** không giải thích toàn bộ mismatch.

Tuy nhiên investigation cho thấy **target-forward shape** khác nhau mới là hướng đáng chú ý.

---

## 7. Phase 1 — Index368 Và Attention Dispatch

Các mốc:

```text
3c50ec0  capture index-368 execution state
693cf3a  preserve r13 index-368 diagnostic
127067c  isolate index-368 target forward shape
bca75b1  capture index-368 attention dispatch
a360ce2  preserve r15 attention dispatch diagnostic
d24f32d  preserve r14-r15 attention dispatch control
```

Conceptual finding:

```text
Vanilla decode:
M = 1
q_len = 1

Speculative verify:
M = k + 1
q_len > 1
```

Cùng một logical generated position có thể được tính trong execution shape khác. Với floating-point GPU arithmetic, thay tile/reduction/dispatch shape có thể làm bit pattern khác dù toán học lý tưởng giống nhau.

### 7.1. Forced 3D / q3 experiments

```text
07de3a1  force guarded 3d attention at index368
dc8a33d  preserve r16 forced-3d diagnostic
031bc43  admit guarded q3 split-kv verification
6f2d6ff  observe production q3 attention routing
bb2a09b  preserve production q3 regression
```

Thay attention route làm behavior thay đổi, nhưng không đóng được full gate và có regression mới.

**Kết luận:** attention/execution form là nghi phạm mạnh, nhưng “đổi attention làm token đổi” chưa đủ là causal proof.

---

## 8. Phase 2 — `fix/m2-03-lossless-closure`: Reproduction Và Same-Process Controls

Các mốc:

```text
8263a8c  preserve canonical production gate fail
60045f3  observe canonical index81 boundary
1e68c13  observe canonical V2 index81 boundary
0c2816d  ignore v2 startup warmup before request
a7b52e7  preserve index81 non-reproduction
56604cd  preserve vanilla same-process control
```

Mục tiêu phase này là tránh kết luận sai rằng mismatch do attention trong khi thực tế có thể do process/session/startup asymmetry.

**OBSERVATION:** một số diagnostic boundary không reproduce ổn định khi thay process/instrumentation.

**Same-process Vanilla control:** loại được giải thích đơn giản “Vanilla khác vì process khác”.

**Kết luận:** environment/process startup alone không giải thích full mismatch; execution form / instrumentation sensitivity vẫn còn.

---

## 9. Phase 3 — Clean Baseline Với Protocol-B

Implementation:

```text
a63baa1  feat(m2-03): add protocol B paired verification
```

Evidence preservation:

```text
4372f8b  evidence(m2-03): preserve protocol B paired verification
```

Raw directory:

```text
experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00/
```

Directory này chứa:

- `pair1-vanilla/run.json`, responses, server log, environment, reservation, capture claim;
- `pair1-adaptive/...`;
- `pair2-vanilla/...`;
- `pair2-adaptive/...`;
- `adaptive_telemetry.jsonl`;
- `experiment.json`;
- `frozen_plan.json`;
- `execution_sha.txt`;
- `execution_sources.tar.gz`;
- `packages.txt`, `gpu_uuid.txt`, `nvidia_before/after.txt`;
- `SHA256SUMS`;
- `summary.json`.

### 9.1. Protocol-B result

```text
Vanilla repeatability: 54/54
Adaptive repeatability: 54/54
pair1 Vanilla vs Adaptive: 44/54
pair2 Vanilla vs Adaptive: 44/54
```

**FACT:** mỗi mode tự repeat hoàn toàn.

**FACT:** paired mismatch lặp lại deterministic.

**Kết luận:** không còn hợp lý để giải thích mismatch bằng random Vanilla nondeterminism đơn thuần.

### 9.2. Baseline terminology phải giữ đúng

```text
47/54 = historical Cache-OFF regime
44/54 = authenticated pre-rowwise Protocol-B baseline
45/54 = rowwise-v2 canonical
```

Khi đánh giá production fix phải dùng `44/54 -> 45/54`, không dùng `47/54 -> 45/54`.

---

## 10. Boundary279 -> Layer1 KV -> Draft Tokenizer Revision

Các mốc:

```text
04c5bae  add boundary279 capture
da104fe  propagate observer import path
f4fc645  preserve boundary279 diagnostic
31d31f2  preserve executed V2 runner sources
c8d5c6d  capture V2 boundary scores
3b32710  preserve interpretable boundary279 run
8d8d8cc  add historical KV boundary probe
c777af9  exclude draft layers from KV probe
10bce95  preserve boundary279 state-different run
978f628  localize layer1 KV divergence
```

### 10.1. Diagnostic bug: target vs draft layer

KV probe ban đầu có thể lẫn draft layers với target layers. Sau khi loại draft layers khỏi probe, target-model state difference vẫn tồn tại.

### 10.2. Layer1 localization

Investigation đi từ:

```text
final token mismatch
-> logits/state
-> layer1 K/V mismatch
```

Nhưng layer1 mismatch chưa nói layer1 là origin; nó có thể nhận input đã khác từ layer0.

### 10.3. Draft tokenizer revision bug

Các commit:

```text
1b7397b  pin draft tokenizer revision
7ddddc7  use draft model revision for tokenizer
```

Bug cụ thể: draft tokenizer loader không truyền đúng:

```python
revision=spec.draft_model_config.revision
```

Đây là **runtime correctness/provenance bug thật**, đặc biệt dễ lộ sau session reset hoặc cache khác nhau.

**Nhưng:** fix này không biến canonical thành 54/54; không được gọi nó là general lossless root cause.

---

## 11. Early Timeline -> Position177 -> Controlled Causal Intervention

### 11.1. Early trace

```text
93b9548  trace early adaptive verification
```

Mục tiêu: đi ngược từ layer1 để tìm boundary sớm hơn.

### 11.2. P177 postwrite KV

```text
563f0e9  capture position177 postwrite kv
c306aef  support strided kv cache views
9c0f15e  preserve position177 postwrite divergence
```

Observer ban đầu phải sửa vì live KV cache view là strided; đọc storage sai cách sẽ tạo evidence sai.

Sau fix, P177 target K/V divergence được preserve.

### 11.3. k=0 control

```text
529e7f5  add position177 k0 control
692f21c  resolve control gpu selector
124db36  preserve p177 k0 control
```

Mục đích: phân biệt target model intrinsic behavior với speculative multi-row execution.

### 11.4. Layer0 attention substitution

```text
8c04f23  add p177 layer0 attention substitution
8c08c05  preserve p177 attention causal diagnostic
```

Preserved artifact:

```text
experiments/runs/m2_03/diagnostics/
p177-attention-8c04f23-20261019T075533Z.tar.gz
```

Controlled intervention giữ:

```text
k = 3
M = 4
target positions = [177, 178, 179, 180]
```

Không tắt speculation, không chạy thêm target model forward, không đổi scheduler/sampler. Chỉ thay **layer0 P177 attention output** bằng output tương đương q_len=1 segmented attention.

Kết quả:

```text
layer1 P177 K/V sau substitution == Vanilla
```

### 11.5. Claim boundary

**FACT / PROVEN FOR CONTROLLED CASE:** layer0 attention path là causal cho controlled P177 boundary.

**KHÔNG ĐƯỢC KẾT LUẬN:** layer0 attention path là root cause duy nhất của toàn bộ 54 prompts hay toàn bộ 9 mismatch còn lại.

---

## 12. Production Candidate — T4 Row-wise Segmented Attention

Production commit:

```text
346b1ad  fix(m2-03): add T4 rowwise segmented attention
```

Files chính:

```text
patches/vllm-0.30.0-t4-rowwise-attention.patch
src/server/t4_rowwise_attention.py
tests/test_t4_rowwise_attention.py
tests/fixtures/m203_v030_triton_attention_helpers.py
docs/runbooks/m2-03-rowwise-attention.md
```

Profile:

```text
m203.t4-rowwise-segmented3d.v2
```

Ý tưởng: vẫn giữ một target forward M-row, nhưng mỗi query row dùng effective causal prefix `Lr` tương ứng với single-token semantics, tránh dùng final verify length chung cho mọi row.

Frozen Qwen target shape được admission:

```text
28 query heads
4 KV heads
head size 128
GQA = 7 query heads / KV head
FP16 Q/K/V/output
PIECEWISE graph regime
sm75/T4
```

Không đổi planner, drafts, rejection sampler, k schedule, KV update order hay sampling policy.

---

## 13. Startup Failure Sau Rowwise-v1 Và Composed v2

Sau khi cài rowwise-v1 trên Kaggle, Adaptive startup fail trước targeted capture.

Root cause không phải attention numerical error mà là installer mới **quên compose approved draft tokenizer revision fix**.

Commit sửa:

```text
959b9d7  fix(m2-03): compose draft tokenizer revision into T4 fix
```

Rowwise-v2 = attention transform + `draft-tokenizer-revision-v1`.

Runbook ghi rõ original/patched hashes và restore behavior fail-closed.

**Bài học:** production transform mới phải compose tất cả runtime fixes đã được approve; không được authenticate một subset rồi làm mất fix trước đó.

---

## 14. Targeted GPU PASS Và Full Canonical FAIL

### 14.1. Targeted verification

Evidence commit:

```text
dd10e41  evidence(m2-03): preserve rowwise v2 targeted pass
```

Artifact:

```text
experiments/runs/m2_03/diagnostics/
rowwise-v2-targeted-959b9d7-20261019T231319Z.tar.gz
```

Artifact SHA256:

```text
083515b70ea2af5c5ea9216c207175420e0699ad4f13da3ce32f2a74caf0c1a1
```

Selected controlled case:

```text
w3_business_v2_004
```

Kết quả targeted:

```text
284/284 token IDs exact
real speculation vẫn active
k > 0
M-row verification retained
```

**FACT:** production candidate sửa được case đã causal-localize.

**Nhưng:** targeted PASS không được coi là task DONE.

### 14.2. Full canonical Protocol-B sau rowwise-v2

Evidence commit:

```text
5ae38fd  evidence(m2-03): preserve rowwise v2 canonical 45 of 54
```

Artifact:

```text
experiments/runs/m2_03/diagnostics/
rowwise-v2-canonical-45of54-dd10e41-20261020T134722Z.tar.gz
```

Artifact SHA256:

```text
9579200d1258160dab69028193b3d448997080b5744aa3756e9fbc6b3919a2ca
```

Four canonical runs đều complete 54 requests:

```text
pair1-vanilla   exit 0, complete=True, records=54
pair1-adaptive  exit 0, complete=True, records=54
pair2-vanilla   exit 0, complete=True, records=54
pair2-adaptive  exit 0, complete=True, records=54
```

Analyzer rerun result:

```text
Vanilla repeatability: 54/54
Adaptive repeatability: 54/54
pair1 V/A: 45/54
pair2 V/A: 45/54
interpretation_status: NOT_YET_APPROVED
task_done: false
```

### 14.3. 9 deterministic mismatches sau rowwise-v2

Các mismatch canonical mới nhất:

```text
W1 m0_04_005       first mismatch index162
W1 m0_04_026       index610
W1 m0_04_029       index514
W2 w2_sql_v2_003   index87
W3 w3_business_v2_001 index385
W3 w3_business_v2_002 index48
W3 w3_business_v2_009 index562
W3 w3_business_v2_010 index151
W3 w3_business_v2_012 index166
```

Cả pair1 và pair2 cho cùng mismatch set/positions.

### 14.4. Net effect của rowwise-v2

So với authenticated pre-rowwise Protocol-B `44/54`:

```text
6 mismatch cũ được sửa
4 mismatch cũ còn lại
5 regression mới xuất hiện
--------------------------------
9 mismatch còn lại
44/54 -> 45/54
```

**Kết luận:** fix có causal value nhưng generalization hiện chưa đủ đúng để đạt 54/54.

---

## 15. M005 Regression Diagnostic — Từ Causal Question Đến Observer Effect

### 15.1. Vì sao chọn `m0_04_005`

Trước rowwise, sample này match. Sau rowwise-v2:

```text
Vanilla token162  = 341
Adaptive token162 = 5212
first divergence = generated index162
```

Nó là candidate tốt để hỏi liệu generalized rowwise policy có tạo regression mới.

### 15.2. A/B diagnostic implementation

Commit:

```text
c3add4c  diagnostic(m2-03): add m005 rowwise A/B probe
```

Files:

```text
experiments/m005_ab.py
experiments/m005_ab_sources.py
experiments/m005_ab_observer.py
experiments/protocol_b_capture.py
tests/test_m005_ab.py
docs/runbooks/m2-03-m005-ab.md
```

Condition A:

```text
rowwise-v2 + draft tokenizer fix + observer
```

Condition B:

```text
native pinned attention + same draft tokenizer fix + same observer
```

Frozen: speculation/controller/sampling/cache/model revisions/PIECEWISE/request order.

Expected discriminator nếu evidence comparable:

```text
A token162 = 5212
B token162 = 341
```

### 15.3. A/B attempt #1 — lifecycle bug

Lần đầu A/B không tạo scientific result. Failure:

```text
prepare() tạo sẵn A/
capture() gọi reserve_run() với reserved=False
reserve_run() mkdir(A, exist_ok=False)
-> FileExistsError
```

Fix commit:

```text
47e2d5d  fix(m2-03): repair m005 A/B reservation lifecycle
```

Fix dùng existing `reserve_run(...)` và `reserved=True`; generic Protocol-B behavior không bị nới lỏng.

### 15.4. A/B attempt #2 — operational PASS, scientific INVALID

Cả hai arm hoàn thành install/verify/launch/capture/seal/stop/restore.

Analyzer:

```text
classification = INVALID_EVIDENCE
error = Observer failed/unbound
task_done = false
```

Raw observer sequence A và B:

```text
bound
cache_lookup
observer_error
```

Error tại `scores`:

```text
EvidenceError: Missing attention/unsupported grammar
```

### 15.5. Discovery quan trọng hơn observer error — A không reproduce treatment baseline

Condition A lẽ ra phải reproduce rowwise canonical `5212`, nhưng thực tế:

```text
A token162 = 341
B token162 = 341
```

Để kiểm conditioning, canonical archive được extract và so trực tiếp.

Kết quả:

```text
m0_04_001 A == canonical
m0_04_002 A == canonical
m0_04_003 A == canonical
m0_04_004 A == canonical
```

Selected request:

```text
same request_sha256 = True
m0_04_005 prefix [0:162] identical
first A-vs-canonical divergence = exactly index162
A token162 = 341
canonical Adaptive token162 = 5212
```

**FACT:** five-request conditioning là đủ tới boundary.

**FACT:** broad observer đã làm treatment A không còn reproduce uninstrumented rowwise-v2.

**Kết luận:** broad observer **numerically non-neutral**, nên A/B causal attribution bị invalid.

### 15.6. Preserved raw evidence

Commit:

```text
9dd8233  evidence(m2-03): preserve m005 observer non-neutral A/B
```

Artifact:

```text
experiments/runs/m2_03/diagnostics/
m005-ab-observer-nonneutral-47e2d5d-20261020T235304Z.tar.gz
```

Artifact SHA256:

```text
2a47023c9945a009ec0da4a4ee783fb3c3b7c1b5e13a5da86f618bb0a535be04
```

Đây là remote tip hiện tại của `fix/m2-03-lossless-clean`.

---

## 16. Claim Boundary — Được Phép Nói Gì Và Không Được Nói Gì

### 16.1. FACT / observation đã có evidence

1. Protocol-B pre-rowwise:

```text
Vanilla repeatability 54/54
Adaptive repeatability 54/54
Vanilla vs Adaptive 44/54
```

2. Cache OFF historical run vẫn chỉ 47/54; cache alone không giải thích toàn bộ divergence.
3. Có controlled P177 case mà layer0 attention substitution phục hồi layer1 P177 K/V equality với Vanilla.
4. Production rowwise-v2 sửa hoàn toàn targeted `w3_business_v2_004`.
5. Canonical rowwise-v2 chỉ đạt 45/54, deterministic ở cả hai pairs.
6. Rowwise-v2 vừa sửa mismatch cũ vừa tạo regression mới.
7. Broad M005 observer không neutral: instrumented A đổi token162 từ canonical 5212 thành 341.
8. T4 là compute capability 7.5; current vLLM batch-invariance support yêu cầu NVIDIA compute capability >= 8.0.
9. PM đã quyết định dừng điều tra T4 trong sprint hiện tại và defer absolute 54/54 sang sprint cuối trên GPU mới hơn.

### 16.2. INFERENCE hợp lý nhưng chưa phải general proof

- M=1 decode và M=k+1 verify có execution/arithmetic organization khác nhau; numerical non-invariance là explanation phù hợp với nhiều evidence.
- T4/sm75 + current vLLM/Triton path là môi trường rủi ro cho deterministic equivalence hơn GPU mới trong phạm vi batch-invariance support.
- Chuyển sang L4/A10/A100-class GPU là experiment có information gain cao.

### 16.3. KHÔNG ĐƯỢC PHÉP KẾT LUẬN

Không được viết trong báo cáo như FACT rằng:

```text
“Turing hardware bị lỗi decode token”
“T4 shared-memory overflow là root cause M2-03”
“Triton compiler bug là root cause 9 mismatch”
“rowwise attention là root cause của toàn bộ mismatch”
“A100/L4/A10 chắc chắn đạt 54/54”
“M2-03 đã đạt acceptance 54/54”
```

Project-local evidence chưa chứng minh các câu trên.

### 16.4. Cách diễn đạt khoa học an toàn

Có thể nói:

> Trên Tesla T4/sm75 với frozen vLLM/speculative stack, nhóm quan sát deterministic numerical divergence giữa Vanilla single-token decode và Adaptive multi-row speculative verification. Controlled intervention chứng minh layer0 attention path là causal tại một P177 boundary, nhưng production generalization chưa đạt 54/54 và còn tạo regression mới. Current evidence chưa chứng minh Turing hardware tự thân là root cause chung.

---

## 17. External Technical Context — Chỉ Supporting, Không Thay Raw Evidence

Các tài liệu upstream/official hiện hỗ trợ quyết định đổi GPU:

- NVIDIA CUDA GPU table: T4 = compute capability 7.5; A100/A30 = 8.0; A10 = 8.6; L4 = 8.9.
- vLLM Batch Invariance docs: NVIDIA support yêu cầu compute capability >= 8.0 và feature vẫn đang beta.
- Official FlashAttention-2 CUDA README: Ampere/Ada/Hopper supported; Turing dùng separate Turing implementation; BF16 yêu cầu Ampere/Ada/Hopper.

Những tài liệu này **không chứng minh** M2-03 sẽ tự động đạt 54/54 khi đổi hardware. Chúng chỉ cho thấy PM decision chuyển sang GPU mới hơn có cơ sở kỹ thuật và có information gain.

---

## 18. Stage1 Neutrality Probe — Local WIP Sau Evidence `9dd8233`

Sau khi broad observer bị chứng minh non-neutral, Codex được giao implement một Stage1 observer nhẹ hơn.

Codex report cho biết đã tạo local files:

```text
experiments/m005_neutrality.py
experiments/m005_neutrality_observer.py
experiments/m005_neutrality_sources.py
```

và sửa local:

```text
experiments/m005_ab.py
tests/test_m005_ab.py
```

Mục tiêu Stage1:

```text
A only
không B
metadata-only route/lineage
không Q/K/V bytes
không logits/scores bytes
không .cpu()
không .item()
không argmax/reduction trước selected commit
```

Pass criterion dự kiến:

```text
first four conditioned outputs exact canonical
same m0_04_005 request identity
prefix [0:162] exact canonical
token162 = 5212
k > 0 proven
actual M/q_len proven
rowwise route proven
no observer error
```

Codex báo local CPU/static verification:

```text
155 tests PASS
96 Protocol-B tests PASS
173 relevant tests PASS + 33 subtests
py_compile PASS
git diff --check PASS
```

**Nhưng:** Stage1 GPU neutrality **NOT RUN**.

### 18.1. Current local working tree tại lúc handoff

User-provided `git status` ngày 23/10/2026:

```text
On branch fix/m2-03-lossless-clean
Your branch is up to date with 'origin/fix/m2-03-lossless-clean'.

modified:
  experiments/m005_ab.py
  tests/test_m005_ab.py

untracked:
  AGENTS.md
  docs/handoffs/
  experiments/m005_neutrality.py
  experiments/m005_neutrality_observer.py
  experiments/m005_neutrality_sources.py
  m203-early-timeline-probe.diff
  m203-kv-probe-full.diff
  m203-layer1-localization.diff
  m203-m005-ab-diagnostic.diff
  m203-m005-reservation-lifecycle-fix.diff
  m203-p177-attention-substitution.diff
  m203-p177-gpu-resolution-fix.diff
  m203-p177-k0-control.diff
  m203-postwrite-position177.diff
  m203-postwrite-strided-storage-fix.diff
  m203-postwrite-view-interval-fix.diff
  m203-rowwise-v2-tokenizer-composed.diff
  m203-t4-rowwise-production.diff
  m203-tokenizer-revision-correction.diff
  m203-tokenizer-revision-fix.diff
  m203-v1-target-layer-fix.diff
```

### 18.2. Handoff rule cho local WIP

- `9dd8233` là **remote source-of-truth** hiện tại.
- Stage1 neutrality implementation là **local WIP**, chưa review actual diff cuối, chưa commit, chưa push, chưa GPU run.
- Không được coi Stage1 implementation là accepted source chỉ vì Codex tests PASS.
- Trước khi quay lại task: review diff, xác minh protected evidence không bị chạm, quyết định có preserve local WIP bằng commit/patch riêng hay reset về remote.
- Không commit các `.diff`, `AGENTS.md`, `docs/handoffs/` một cách cơ học; phân loại file nào là intended artifact trước.

---

## 19. PM Decision Ngày 21/10/2026 Và Trạng Thái Quản Lý

Trello M2-03 hiện ghi chính thức:

```text
Status: Blocked · Đã có hướng giải quyết: DEFERRED
Hoãn sang Sprint cuối
```

PM direction:

1. Dừng đào sâu 9 prompt còn lại trên T4 trong sprint hiện tại.
2. Giữ/preserve evidence và hướng điều tra.
3. Chuyển hạ tầng sang Modal.com / GPU mới hơn như L4 hoặc A10-class hardware.
4. Quay lại deterministic qualification + paired lossless verification ở sprint cuối.
5. Gỡ blocker cho các task sau/M3 thay vì tiếp tục tiêu tốn quota T4.

### 19.1. Cách hiểu đúng

**Management status:** PM RE-SCOPED / DEFERRED — được phép đổi WIP khỏi M2-03.

**Scientific status:** acceptance 54/54 vẫn FAIL.

Hai trạng thái này không mâu thuẫn:

```text
Sprint phase trên T4 có thể được PM đóng/defer
trong khi
scientific acceptance của M2-03 vẫn chưa PASS.
```

---

## 20. Evidence Inventory — File Nào Làm Gì

### 20.1. Frozen methodology

```text
docs/06-ke-hoach-thuc-nghiem.md
```

Đọc để xác nhận lossless criterion và measurement discipline trước khi mở lại task.

### 20.2. Protocol-B authenticated baseline

```text
experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00/
```

Dùng để chứng minh:

```text
V/V = 54/54
A/A = 54/54
V/A = 44/54
```

### 20.3. P177 causal evidence

```text
experiments/runs/m2_03/diagnostics/
p177-attention-8c04f23-20261019T075533Z.tar.gz
```

Preserved by commit `8c08c05`.

Dùng cho claim: layer0 attention path causal **cho controlled P177 case**.

### 20.4. Production rowwise-v2 implementation

```text
patches/vllm-0.30.0-t4-rowwise-attention.patch
src/server/t4_rowwise_attention.py
tests/test_t4_rowwise_attention.py
docs/runbooks/m2-03-rowwise-attention.md
```

Production code commit lineage:

```text
346b1ad  rowwise v1
959b9d7  compose tokenizer revision -> rowwise v2
```

### 20.5. Targeted rowwise-v2 PASS

```text
experiments/runs/m2_03/diagnostics/
rowwise-v2-targeted-959b9d7-20261019T231319Z.tar.gz
```

Preserved at `dd10e41`.

### 20.6. Full canonical rowwise-v2 FAIL 45/54

```text
experiments/runs/m2_03/diagnostics/
rowwise-v2-canonical-45of54-dd10e41-20261020T134722Z.tar.gz
```

Preserved at `5ae38fd`.

Đây là canonical evidence hiện tại cho production candidate trên T4.

### 20.7. M005 broad A/B observer evidence

```text
experiments/runs/m2_03/diagnostics/
m005-ab-observer-nonneutral-47e2d5d-20261020T235304Z.tar.gz
```

Preserved at `9dd8233`.

Dùng để chứng minh observer non-neutral / INVALID_EVIDENCE, không dùng để kết luận A/B causality.

### 20.8. Runbooks

```text
docs/runbooks/m2-03-rowwise-attention.md
docs/runbooks/m2-03-m005-ab.md
```

Runbooks giải thích authentication, install/verify/restore, frozen conditions, classification logic và test commands.

### 20.9. Problem log

Trello trỏ tới:

```text
logs/problems/2026-10-16-m2-03-lossless-verification-repeatability.md
```

Đọc cùng worklog này khi cần narrative của mid-investigation.

---

## 21. Timeline Commit Tóm Tắt Theo Chuỗi Nhân-Quả

| Commit | Vai trò | Điều học được / quyết định |
|---|---|---|
| `27fc09e` | lossless verifier | Bắt đầu so token equivalence thực sự |
| `56ab2ec` | preserve first failure | M2-03 empirical gate FAIL |
| `c9186ee` / `35b6db6` | cache lookup diagnostics | Nghi prefix cache |
| `1066bcd` / `bdb26dd` | Cache-OFF controls | Cache alone bị loại; historical 47/54 |
| `78713f2` -> `a0274c9` | KV probes | Cần model-state evidence và lineage |
| `5787388` / `feffdf5` | PIECEWISE/CUDA controls | CUDA graph alone không đủ giải thích |
| `693cf3a` / `127067c` | index368 forward shape | M=1 vs M>1 trở thành hướng chính |
| `bca75b1` / `d24f32d` | attention dispatch | Attention path phụ thuộc execution form |
| `031bc43` / `bb2a09b` | q3 production experiment | Có regression; chưa thể generalize |
| `a7b52e7` | index81 non-reproduction | Boundary diagnostic có thể nhạy instrumentation |
| `56604cd` | same-process Vanilla | Process startup alone không giải thích full issue |
| `a63baa1` / `4372f8b` | Protocol-B | Authenticated baseline 44/54; deterministic |
| `3b32710` / `10bce95` | boundary279 / KV | Internal target state khác |
| `978f628` | layer1 localization | Mismatch đi tới layer1 K/V |
| `1b7397b` / `7ddddc7` | tokenizer revision | Runtime provenance bug thật được sửa |
| `563f0e9` / `9c0f15e` | P177 postwrite | Tìm được early target-state divergence |
| `124db36` | P177 k=0 control | Phân biệt speculation path |
| `8c04f23` / `8c08c05` | attention substitution | Causal proof cho controlled P177 |
| `346b1ad` | rowwise production v1 | Production candidate từ causal finding |
| `959b9d7` | rowwise v2 | Compose tokenizer revision đúng |
| `dd10e41` | targeted PASS evidence | Case causal-localized được sửa hoàn toàn |
| `5ae38fd` | canonical 45/54 | Global acceptance vẫn FAIL |
| `c3add4c` | M005 A/B | Test generalized rowwise regression |
| `47e2d5d` | A/B lifecycle fix | Diagnostic infrastructure bug được sửa |
| `9dd8233` | non-neutral A/B evidence | Broad observer tự làm trajectory đổi; evidence invalid |

---

## 22. Những Hypothesis Đã Bị Loại Hoặc Thu Hẹp

### 22.1. Prefix cache là nguyên nhân duy nhất

**Rejected.** Cache-OFF vẫn mismatch.

### 22.2. Vanilla tự nondeterministic ngẫu nhiên

**Rejected cho Protocol-B frozen condition.** Vanilla repeatability 54/54.

### 22.3. Adaptive tự nondeterministic ngẫu nhiên

**Rejected cho Protocol-B frozen condition.** Adaptive repeatability 54/54.

### 22.4. CUDA graph alone là nguyên nhân

**Không đủ.** no-cudagraph/PIECEWISE controls không khôi phục full equality.

### 22.5. Process/session startup alone

**Không đủ.** same-process controls không giải thích full issue.

### 22.6. Draft tokenizer revision là toàn bộ root cause

**Rejected as general root cause.** Đây là bug runtime/provenance thật nhưng sau fix vẫn còn lossless mismatch.

### 22.7. Rowwise attention fix giải quyết toàn bộ M2-03

**Rejected.** Targeted PASS nhưng canonical chỉ 45/54 và tạo regression mới.

### 22.8. M005 broad A/B chứng minh rowwise không causal

**Không hợp lệ.** Condition A không reproduce canonical treatment do observer non-neutral; analyzer đúng khi trả INVALID_EVIDENCE.

---

## 23. Vấn Đề Chưa Giải Quyết

### 23.1. General root cause của 9 deterministic mismatches

**Status:** UNKNOWN.

Có evidence execution-form/attention arithmetic rất quan trọng, nhưng chưa đủ để nói một operator duy nhất giải thích mọi mismatch.

### 23.2. Observer neutrality

Broad observer đã fail neutrality. Stage1 metadata-only probe được implement local nhưng chưa GPU verify.

### 23.3. Hardware dependence

Chưa có paired experiment cùng frozen stack kiểu:

```text
T4/sm75 result
vs
L4/A10/A100-class result
```

Do đó chưa được phép nói Turing hardware tự thân là root cause.

### 23.4. 54/54 final acceptance

**NOT ACHIEVED.** Deferred theo PM.

---

## 24. Trạng Thái Task Khi Tạm Đóng

### 24.1. PM / Trello

```text
Blocked
DEFERRED to final sprint
T4 investigation stopped for current sprint
```

### 24.2. Engineering

```text
production rowwise-v2 exists
controlled targeted case PASS
full canonical 45/54 FAIL
9 deterministic mismatches remain
```

### 24.3. Scientific

```text
controlled P177 attention causality: PROVEN for one case
general root cause: UNKNOWN
Turing hardware root cause: NOT PROVEN
```

### 24.4. Local repository

```text
remote source-of-truth = 9dd8233
local Stage1 WIP exists but uncommitted/unpushed
```

### 24.5. Final status string

```text
M2-03 = PM RE-SCOPED / DEFERRED WITH EVIDENCE
NOT DONE WITH 54/54 EVIDENCE
```

---

## 25. Hướng Tiếp Theo Khi Mở Lại Ở Sprint Cuối

Không tiếp tục từ đầu. Reuse evidence/tooling hiện có.

### FIRST ACTION khi reopen

1. Inventory repo và xác nhận branch/SHA source-of-truth.
2. Re-read PM/Trello decision và `docs/06-ke-hoach-thuc-nghiem.md`.
3. Quyết định xử lý local Stage1 WIP: review/preserve hoặc reset, không để trạng thái mơ hồ.
4. Bring up newer GPU platform theo PM decision.
5. Record exact GPU, compute capability, driver, CUDA, PyTorch, vLLM, model revisions, Git SHA.
6. Chạy **deterministic qualification trước**, không full benchmark ngay.
7. Chạy paired Vanilla repeatability + Adaptive repeatability + V/A comparison theo frozen 54 prompts.
8. Nếu GPU mới đạt 54/54: preserve raw evidence, không suy ngược rằng T4 “hardware bug” nếu chưa có controlled cross-hardware attribution.
9. Nếu vẫn fail: dùng Stage1 neutrality-first probe hoặc minimum discriminator; không quay lại broad tensor observer trước khi neutrality PASS.
10. Chỉ sau empirical 54/54 mới claim lossless acceptance DONE.

### Không được làm khi reopen

- Không rerun hàng loạt diagnostics cũ chỉ vì có GPU mới.
- Không đổi workload/seed/cache/protocol để ép số đẹp.
- Không loại mismatch/outlier tùy tiện.
- Không dùng Batch Invariance / FA2 như “proof” nếu project frozen protocol chưa approve thay đổi đó.
- Nếu thay backend/runtime protocol, ghi rõ đó là experiment khác và cần PM/protocol approval nếu ảnh hưởng frozen acceptance.

---

## 26. Handoff Cho Dương / Khôi / Reviewer

Nếu chỉ có 5 phút để đọc, cần nắm 8 ý sau:

1. **Task hỏi 54/54 lossless, không hỏi server có chạy hay không.**
2. **Protocol-B chứng minh mismatch deterministic:** V/V 54/54, A/A 54/54, V/A 44/54.
3. **Cache, process và CUDA graph đơn lẻ đã bị loại/thu hẹp.**
4. **P177 controlled intervention chứng minh layer0 attention path causal cho một boundary.**
5. **Production rowwise-v2 sửa targeted case nhưng full gate chỉ 45/54.**
6. **M005 A/B broad observer không đáng tin vì observer tự đổi token 5212 -> 341 đúng tại boundary.**
7. **Remote evidence đã preserve tới `9dd8233`; local Stage1 neutrality code chưa commit/push.**
8. **PM đã defer T4 investigation; khi quay lại phải chạy qualification trên GPU mới, nhưng không được claim Turing là general root cause trước controlled evidence.**

---

## 27. Checklist Tạm Đóng

- [x] Lossless verifier tồn tại.
- [x] Protocol-B paired verification chạy và raw evidence được commit.
- [x] Vanilla repeatability 54/54 được chứng minh.
- [x] Adaptive repeatability 54/54 được chứng minh.
- [x] Pre-rowwise V/A 44/54 được preserve.
- [x] Cache-OFF negative result được preserve.
- [x] KV / boundary / layer localization được thực hiện.
- [x] Draft tokenizer revision bug được sửa.
- [x] P177 causal attention intervention được preserve.
- [x] Production rowwise-v2 được implement/authenticate.
- [x] Targeted `w3_business_v2_004` PASS.
- [x] Full rowwise-v2 canonical 45/54 được preserve.
- [x] 9 deterministic mismatch list được xác định.
- [x] M005 A/B lifecycle bug được sửa.
- [x] Broad A/B non-neutral observer evidence được preserve.
- [x] PM/Trello đã re-scope/defer task.
- [ ] Stage1 neutrality local diff được review/decide preserve hay discard.
- [ ] Stage1 GPU neutrality PASS.
- [ ] New-GPU deterministic qualification.
- [ ] New-GPU paired 54-prompt lossless verification.
- [ ] 54/54 empirical gate PASS.
- [ ] Final PM acceptance / final-sprint closure.

**Current closure:** T4 phase closed/deferred by PM; scientific task acceptance remains open.

---

## 28. Retrospective Và Bài Học Kỹ Thuật

1. `temperature=0` không tự chứng minh bitwise-equivalent forward arithmetic.
2. Token prefix equality không tự chứng minh KV equivalence.
3. KV equivalence không được suy ra từ raw storage nếu layout/lineage chưa authenticate.
4. Diagnostic tool có thể là nguồn bug; phải phân biệt production failure và observer failure.
5. Negative result như Cache-OFF 47/54 có giá trị vì loại được hypothesis.
6. Causal intervention mạnh hơn correlation: P177 substitution là bước quan trọng nhất về root-cause evidence.
7. Targeted PASS không thay canonical gate.
8. Production fix có thể sửa một causal case nhưng tạo regression nơi khác.
9. Instrumentation readback/synchronization có thể làm numerical trajectory thay đổi; observer neutrality phải là acceptance gate của diagnostic.
10. Execution SHA, source SHA, evidence checkpoint phải tách rõ.
11. Session reset/Kaggle volatility làm mất raw directories nếu chưa preserve; evidence quan trọng phải seal/commit.
12. Không được gọi một hardware architecture là root cause chỉ vì đổi sang hardware mới là hướng hợp lý.
13. PM có thể re-scope tiến độ mà không làm thay đổi truth value của scientific gate.
14. “DONE sprint phase” và “DONE WITH 54/54 EVIDENCE” là hai khái niệm khác nhau.

---

## 29. Kết Luận Bàn Giao

M2-03 bắt đầu từ một failure đơn giản ở token-level nhưng đã đi qua nhiều lớp kiểm chứng: cache controls, KV/state forensics, CUDA-graph controls, target-forward shape, attention dispatch, clean paired protocol, layer localization, causal P177 intervention, production rowwise candidate, canonical regression analysis và cuối cùng là observer-neutrality failure.

Giá trị chính của giai đoạn T4 không phải “đã đạt 54/54”, mà là đã biến một mismatch mơ hồ thành tập evidence có cấu trúc:

```text
random-noise explanation             -> bị loại trong frozen Protocol-B
cache-only explanation               -> bị loại
process-startup-only explanation     -> không đủ
attention/execution-form mechanism   -> có controlled causal evidence ở P177
production generalization            -> targeted PASS, canonical FAIL
broad M005 causal probe              -> invalid vì observer non-neutral
```

**Trạng thái cuối:** canonical production rowwise-v2 trên T4 đạt **45/54**, còn **9 deterministic mismatches**, general root cause **UNKNOWN**, full 54/54 **chưa đạt**. PM đã quyết định dừng thêm T4 investigation trong sprint hiện tại và defer sang sprint cuối với GPU mới hơn. Toàn bộ raw evidence quan trọng đã được preserve đến remote commit `9dd8233`; Stage1 neutrality probe mới vẫn là local WIP và không được xem là accepted implementation cho đến khi review/commit/GPU verify.

Khi mở lại, hành động đúng không phải “debug lại từ đầu”, mà là **qualification trên GPU mới + paired verification theo frozen protocol**, rồi chỉ mở sâu diagnostics nếu empirical gate vẫn fail.

**Không tự sửa Trello, không tự đổi frozen protocol, không claim Turing hardware root cause, và không đánh dấu M2-03 54/54 DONE khi chưa có raw empirical evidence.**
