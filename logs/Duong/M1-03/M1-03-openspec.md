# OpenSpec M1-03 — Node 1 VAT JSON Extractor & Dataset Workload W1

> **Trạng thái:** Đặc tả đã rà soát, sẵn sàng để viết code plan  
> **Mã:** `M1-03`  
> **File đích dự kiến:** `logs/Duong/M1-03/openspec.md`  
> **Phạm vi:** Node 1 VAT JSON Extractor + Dataset Workload W1 + validation + benchmark handoff  
> **Ngày rà soát:** 26/09/2026  
> **Nguyên tắc:** Chưa tạo/sửa branch, commit, PR hoặc bất kỳ file nào trong Git repository.

---

## 0. Kết luận rà soát bản OpenSpec trước

Bản trước đúng hướng về nghiệp vụ và đã bám khá sát M0-04, nhưng còn một số điểm chưa đủ chặt để gọi là một OpenSpec “spec-driven” hoàn chỉnh.

### 0.1 Các thiếu sót đã phát hiện và đã sửa trong bản này

1. **Bản trước áp nhầm cấu trúc OpenSpec nhiều file.**  
   Workflow của nhóm dùng **một file OpenSpec duy nhất** đặt trong worklog M1-03. Bản này đã bỏ toàn bộ giả định về `openspec/changes/`, `.openspec.yaml`, `proposal.md`, `design.md`, `specs/` và `tasks.md` riêng lẻ.

2. **Đã tách rõ đặc tả và code plan.**  
   File này chỉ khóa yêu cầu, contract, acceptance criteria và design constraints. **Code plan chưa được viết** và sẽ là bước tiếp theo sau khi đặc tả được duyệt.

4. **Ngưỡng dataset được chốt theo mô tả chính của M1-03.**  
   Tài liệu này thống nhất dùng **30+ mẫu** làm tiêu chí bắt buộc cho W1. Các tham chiếu 30+ trước đây đã được loại bỏ để tránh mâu thuẫn trong plan và nghiệm thu.

5. **Thiếu ràng buộc khoa học của Workload W1.**  
   Tài liệu thực nghiệm của repo định nghĩa W1 là **JSON Extractor, entropy thấp**, và yêu cầu cùng prompt, `temperature=0`, seed cố định giữa các lane. Bản này đưa các điều kiện đó thành contract.

6. **Thiếu cơ chế freeze dataset/prompt trước benchmark.**  
   Nếu vừa xem kết quả vừa thay prompt/dataset, so sánh Vanilla/Static không còn sạch. Bản này yêu cầu freeze W1 và prompt trước lượt benchmark M1 chính thức.

7. **Thiếu định danh phiên bản dữ liệu/prompt đủ mạnh.**  
   Bản trước có prompt hash nhưng chưa yêu cầu dataset identity rõ ràng. Bản này yêu cầu hash/fingerprint cho manifest/request package để phát hiện drift.

8. **Chưa tách đủ các lớp “thành công”.**  
   “HTTP 200”, “parse được JSON”, “đúng source binding”, “đúng schema”, “đúng nghiệp vụ” là các khái niệm khác nhau. Bản này định nghĩa taxonomy và mẫu số metric rõ ràng.

9. **Chưa khóa Pydantic ↔ JSON Schema là một contract duy nhất.**  
   Trello yêu cầu Pydantic, còn M0-04 đã có JSON Schema v1. Bản này yêu cầu hai lớp phải tương thích và có regression test chống drift.

10. **Thiếu ràng buộc chống data leakage.**  
    M0-04 đã giữ MC-OCR labels ngoài prompt. Bản này nâng thành requirement bắt buộc cho W1 cuối.

11. **Thiếu tách dev-set và final benchmark set.**  
    M0-04 30 mẫu phù hợp làm dev/regression set. Bản này yêu cầu W1 cuối 30+ mẫu được freeze riêng để tránh “tune trên chính tập đo”.

12. **Thiếu yêu cầu provenance/quyền sử dụng dữ liệu nếu bổ sung hóa đơn thực tế.**  
    Nếu ngoài MC-OCR có dữ liệu thật, W1 phải loại thông tin nhạy cảm không cần thiết và ghi nguồn/quyền sử dụng. Không được đưa hóa đơn cá nhân tùy tiện vào benchmark.

13. **Thiếu mối liên hệ với mục tiêu đề tài.**  
    Pipeline hóa đơn là **dụng cụ đo**, không phải đóng góp domain. Bản này khóa rõ Node 1 không được làm thay đổi engine/proposer và không nhúng logic VAT vào engine.

14. **Thiếu yêu cầu fairness giữa các lane.**  
    Cùng một `sample_id` phải tạo ra cùng semantic request ở Vanilla và Static; không được dùng prompt/response-mode khác để “giúp” một lane.

15. **Thiếu điều kiện fail-fast khi backend không hỗ trợ request contract.**  
    Không được âm thầm bỏ `response_format`, đổi temperature hay đổi seed khác nhau theo lane.

16. **Thiếu contract cho raw artifact và failure artifact.**  
    Bản này quy định raw response phải bất biến, canonical output tách riêng, lỗi có mã loại và sample ID.

17. **Thiếu “Open Questions” mang tính blocking trước khi viết tasks.**  
    OpenSpec khuyến nghị không tạo `tasks.md` nếu câu hỏi thiết kế có thể thay đổi thứ sẽ được xây. Bản này đưa các câu hỏi blocking xuống cuối Design.

---

# 1. Nguồn sự thật đã đối chiếu

## 1.1 Trello M1-03

M1-03 yêu cầu về bản chất:

- Chuẩn bị dữ liệu hóa đơn VAT/receipt tiếng Việt cho Workload W1.
- Thiết kế/kiểm chứng schema trích xuất gồm bên bán, bên mua, hàng hóa, VAT, tổng tiền.
- Xây Node 1 gọi server vLLM qua OpenAI-compatible Chat Completions.
- Tỉ lệ output đúng schema đạt ít nhất 95%.
- Đóng gói prompt/dataset W1 cho Benchmark Harness.
- Có test, experiment artifact, worklog và cập nhật báo cáo.

**Ngưỡng dữ liệu M1-03 được chốt:**
- W1 final SHALL có **ít nhất 30 mẫu hợp lệ**.
- Mọi acceptance criterion, test plan và benchmark handoff SHALL dùng cùng ngưỡng này.

## 1.2 Baseline từ branch M0-04

Branch tham chiếu:

```text
feature/m0-04-vat-invoice-schema-mc-ocr
```

So với `main`, branch này đang ahead 6 commits và đã có:

- MC-OCR dev-set preparation.
- 30 mẫu dev-set chất lượng cao.
- Prompt Node 1.
- `invoice_extraction.v1` JSON Schema.
- Request builder theo OpenAI-compatible chat format.
- OpenAI-compatible HTTP client.
- Runner gọi Ollama local qua `/v1/chat/completions`.
- Source-binding validation.
- Semantic validation cho date/time/evidence/line number.
- SQLite v1 + analytical views.
- Deterministic canonicalization.
- Regression tests.
- Raw/validated/failure artifacts.
- Kết quả functional PoC: 30/30 sample parse/validate thành công trên Ollama.

M0-04 là **baseline kỹ thuật và contract**, không phải bằng chứng nghiệm thu M1.

Trạng thái Git tại thời điểm rà soát:
- PR M0-04: `#2 — [M0-04] Hoàn thiện phân tích hóa đơn VAT và pipeline MC-OCR`.
- PR đang `open`, chưa merge vào `main`.
- Branch head M0-04 tại thời điểm kiểm tra: `1c8c5c610168cd3f4c68e72d6130949228284d70`.
- Vì vậy M1-03 có dependency trực tiếp vào PR #2 như mô tả tại §2.1.


## 1.3 Ràng buộc từ tài liệu thiết kế/thực nghiệm

Repo định nghĩa:

- W1 = **JSON Extractor, entropy thấp**.
- W2 = Text-to-SQL, entropy trung bình.
- W3 = phân tích tự do, entropy cao.
- Pipeline hóa đơn là **dụng cụ đo**, không phải domain contribution.
- So sánh lane phải cố định prompt/generation:
  - `temperature = 0`
  - seed cố định
  - cùng tập prompt
- Benchmark M1 dùng vLLM/OpenAI-compatible target path.
- Ollama chỉ phù hợp functional/local smoke test.
- Benchmark harness chịu trách nhiệm concurrency, warm-up, repeat, telemetry; Node 1 không được nhúng logic sweep vào pipeline.

---

# 2. Hợp đồng thực thi M1-03

Phần này khóa rõ **sẽ làm ở branch nào, kế thừa từ đâu, phạm vi file nào dự kiến thay đổi, bàn giao gì cho M1-02 và kiểm tra gì trước khi coi M1-03 là hoàn thành**.

## 2.1 Branch và quan hệ phụ thuộc Git

### Branch triển khai

Branch M1-03 được chốt theo Trello:

```text
feat/m1-03-mc-ocr-node1-extractor
```

Merge target cuối:

```text
main
```

PR title đề xuất:

```text
[M1-03] Hoàn thiện Node 1 VAT JSON Extractor và Workload W1
```

### Dependency bắt buộc từ M0-04

M1-03 kế thừa trực tiếp contract/code của:

```text
feature/m0-04-vat-invoice-schema-mc-ocr
```

Tại thời điểm đặc tả này, PR M0-04 `#2` vẫn **open** và chưa merge vào `main`.

Do đó:

- **Ưu tiên:** merge M0-04 PR #2 trước, sau đó tạo `feat/m1-03-mc-ocr-node1-extractor` từ `main` mới nhất.
- **Nếu bắt buộc triển khai M1-03 trước khi PR #2 merge:** tạo branch M1-03 từ head của `feature/m0-04-vat-invoice-schema-mc-ocr`, không tạo từ `main` hiện tại.
- Sau khi M0-04 merge, branch M1-03 SHALL rebase/update lên `main` và chạy lại toàn bộ acceptance checks.
- Diff PR M1-03 cuối cùng SHALL chỉ thể hiện delta của M1-03; không được lặp lại 107 file thay đổi của M0-04 như thể đó là code mới M1-03.

### Điều kiện trước khi tạo branch

Branch chỉ được tạo khi:

1. OpenSpec này được duyệt.
2. Code plan M1-03 được duyệt.
3. Base strategy ở trên đã xác định theo trạng thái PR #2 lúc bắt đầu code.

**Bước hiện tại không tạo branch và không thay đổi Git.**

## 2.2 Mục tiêu implementation cụ thể

M1-03 SHALL hoàn thành các nhóm việc sau.

### Nhóm A — Chốt W1 Dataset 30+ mẫu

- Dùng 30 mẫu M0-04 làm baseline; MAY dùng chính 30 mẫu đó làm W1 final nếu đáp ứng toàn bộ coverage/provenance/freeze checks.
- Nếu cần mở rộng, bổ sung sample nhưng minimum vẫn là 30.
- Freeze sample membership, exact OCR text, sample ID và provenance.
- Tạo dataset fingerprint.
- Kiểm tra không leak MC-OCR annotation/gold labels vào model-visible input.
- Sinh coverage summary cho seller/buyer/date-time/total/items/VAT-tax.
- Tạo canonical W1 artifacts để M1-02 sử dụng.

### Nhóm B — Chốt typed invoice contract

- Giữ `invoice_extraction.v1` làm public baseline.
- Bổ sung Pydantic models theo yêu cầu Trello.
- Chọn một source-of-truth strategy giữa Pydantic và JSON Schema trong code plan.
- Có regression check để hai validator không drift.
- Giữ semantic checks hiện có: calendar, `issued_at`, source evidence, unique item line number.
- Không thêm breaking field change nếu chưa review/bump version.

### Nhóm C — Tách reusable Node 1 Extractor

- Tạo/refactor reusable module theo target Trello:

```text
src/pipeline/extractor.py
```

- Module này SHALL không phụ thuộc vào CLI path hoặc SQLite để validate được một response.
- Reuse OpenAI-compatible client hiện có khi phù hợp.
- Tách rõ:
  - request execution;
  - response parse;
  - canonicalization;
  - source binding;
  - schema/Pydantic/semantic validation;
  - structured success/failure result.
- `scripts/run_node1_extractor.py` SHOULD trở thành orchestration/CLI layer thay vì nơi chứa toàn bộ business logic.

### Nhóm D — Chuẩn hóa W1 request package

Canonical handoff area được chốt:

```text
experiments/configs/w1/
```

Package SHOULD chứa tối thiểu thông tin tương đương:

```text
experiments/configs/w1/
├── manifest.csv
├── samples.jsonl
├── node1_chat_requests.jsonl
└── workload.json
```

`workload.json` hoặc artifact tương đương SHALL ghi:

- workload ID/version;
- sample count;
- dataset fingerprint;
- schema version;
- prompt version/hash;
- model-facing generation settings;
- request package paths.

Tên file có thể được tinh chỉnh trong code plan, nhưng **canonical root `experiments/configs/w1/` được coi là contract của M1-03** để tránh tiếp tục phụ thuộc vào path `m0_04_*`.

### Nhóm E — Handoff cho Benchmark Harness M1-02

M1-03 không sở hữu concurrency scheduler, nhưng SHALL cung cấp W1 contract đủ để M1-02 replay đúng request.

Điểm tích hợp bắt buộc:

- Harness phải nhận được **full Chat Completions semantics**, không chỉ một chuỗi prompt.
- Mỗi request W1 cần giữ:
  - system message;
  - user message;
  - model/generation config;
  - response-format policy;
  - `expected_source`;
  - sample ID;
  - schema/prompt/dataset identity.
- Nếu M1-02 harness dùng streaming để đo TTFT/TPOT, streaming MAY thay đổi transport mode nhưng MUST NOT thay đổi task semantics hoặc model-visible messages.
- Harness hiện tại đang tự tạo một user-only request; do đó M1-02 cần adapter/loader tương thích frozen W1 package trước khi dùng kết quả đó như benchmark Node 1 chính thức.
- Việc sửa scheduler/concurrency logic vẫn thuộc M1-02, không mở rộng scope M1-03.

## 2.3 Bản đồ file dự kiến

Đây là **scope map**, chưa phải code plan chi tiết.

| Path | Hành động dự kiến | Vai trò |
|---|---|---|
| `logs/Duong/M1-03/openspec.md` | CREATE | OpenSpec một-file của M1-03 |
| `logs/Duong/M1-03/README.md` | CREATE | Worklog/nghiệm thu M1-03 |
| `src/pipeline/extractor.py` | CREATE | Reusable Node 1 extractor |
| `src/pipeline/openai_compatible.py` | REUSE / MODIFY nếu cần | Transport OpenAI-compatible |
| `src/pipeline/prompts/node1_extractor_system.md` | REUSE / MODIFY có version/hash | Frozen Node 1 prompt |
| `src/pipeline/schemas/invoice_extraction.schema.json` | REUSE, chỉ MODIFY nếu contract review yêu cầu | JSON Schema v1 |
| `src/pipeline/` Pydantic model file | CREATE | Typed application contract |
| `scripts/prepare_mcocr_devset.py` | REUSE / GENERALIZE | Dataset selection/preprocessing baseline |
| `scripts/build_node1_requests.py` | MODIFY / GENERALIZE | Build canonical W1 package |
| `scripts/run_node1_extractor.py` | MODIFY | CLI orchestration dùng reusable extractor |
| `scripts/check_invoice_contract.py` | MODIFY | Contract regression gồm Pydantic + JSON Schema |
| `experiments/configs/w1/` | CREATE | Frozen W1 handoff package + `provenance.jsonl` |
| `experiments/results/` | ADD compact M1-03 summaries | Tracked reproducibility evidence, gồm coverage JSON |
| `tests/` | ADD/MODIFY | Unit/regression/integration tests |
| `requirements.txt` | MODIFY | Thêm `pydantic` nếu chưa có |
| `experiments/benchmark_harness.py` | KHÔNG sở hữu trong M1-03 | Chỉ ghi interface requirement; implementation thuộc M1-02 |

Generated/raw heavy data SHOULD tiếp tục nằm ở khu vực gitignored phù hợp; không commit ảnh/SQLite/raw logs lớn nếu không cần.

## 2.4 Các check bắt buộc

Checks được chia thành **baseline checks đã tồn tại**, **M1-03 checks phải bổ sung**, và **final acceptance checks**.

### Gate 0 — Git/base sanity trước khi code

Trước khi triển khai:

- xác nhận branch hiện tại đúng `feat/m1-03-mc-ocr-node1-extractor`;
- xác nhận branch base chứa toàn bộ M0-04 baseline;
- xác nhận không vô tình code M1-03 từ `main` cũ khi PR #2 chưa merge;
- sau khi sync/rebase M0-04, working tree sạch trước khi bắt đầu change.

### Gate 1 — Baseline regression checks

Các checks M0-04 sau SHALL tiếp tục pass sau refactor:

```bash
python -m unittest discover -s tests -v
python scripts/validate_invoice_schema.py src/pipeline/examples/node1_extractor_output_m0_04_001.json
python scripts/check_invoice_contract.py
```

Ngoài ra SHOULD chạy:

```bash
python -m compileall src scripts tests
```

### Gate 2 — Dataset W1 checks

M1-03 SHALL có automated check tương đương các điều kiện:

- sample count `>= 30`;
- `sample_id` unique/non-empty;
- source record/provenance đầy đủ;
- exact OCR text không rỗng;
- no annotation/gold label leakage vào model-visible request;
- coverage summary được tạo;
- dataset fingerprint tồn tại;
- rebuild không đổi input → fingerprint không đổi;
- thay sample/raw text → fingerprint đổi;
- frozen manifest và request package có cùng sample set.

Tên CLI checker cụ thể được chốt ở code plan, nhưng các assertions trên là bắt buộc.

### Gate 3 — Contract/validation checks

Automated tests SHALL kiểm tra tối thiểu:

- Pydantic ↔ JSON Schema contract equivalence;
- impossible date/time fail;
- conflicting `issued_at` fail;
- duplicate item `line_number` fail;
- source-binding mismatch fail;
- unsupported evidence fail/remove theo canonicalization contract;
- additional property fail;
- missing required nested key fail;
- `status=ok` với non-empty `missing_fields` fail;
- `status=partial` với empty `missing_fields` fail;
- canonicalization không tạo business value;
- validation chạy được khi không bật SQLite.

### Gate 4 — Local functional smoke

Ollama hoặc compatible local backend MAY được dùng để smoke test:

- endpoint reachable;
- model preflight pass;
- 1–3 representative sample chạy end-to-end;
- raw response, validated output và failure artifact được ghi đúng.

Local smoke **không phải final M1 acceptance**.

### Gate 5 — Final vLLM acceptance

Frozen W1 SHALL được chạy qua target OpenAI-compatible vLLM path với cùng prompt/config đã freeze.

Final run SHALL xác nhận:

- attempted samples >= 30;
- `schema_valid_rate >= 0.95`;
- denominator giữ cả failed samples;
- raw/validated/failure artifacts truy ngược được theo `sample_id`;
- prompt hash + dataset fingerprint + schema version được ghi;
- no silent request fallback;
- same W1 semantics có thể replay bởi M1-02.

Nếu W1 có **đúng 30 mẫu**, gate 95% tương đương:

```text
ceil(0.95 × 30) = 29
```

Nghĩa là tối thiểu **29/30 sample** phải fully valid.  
`28/30 = 93.33%` SHALL fail M1-03.

Nếu W1 có `N > 30`, minimum valid sample count SHALL là:

```text
ceil(0.95 × N)
```

### Gate 6 — PR readiness

Trước khi mở/merge PR M1-03:

```bash
git diff --check origin/main...HEAD
```

và kiểm tra:

- full unit/regression suite pass;
- W1 final acceptance pass;
- không có secret/API key;
- không commit generated images/SQLite/raw logs ngoài policy;
- `logs/Duong/M1-03/README.md` có evidence/commands/kết quả;
- compact reproducibility summary nằm đúng khu vực `experiments/results/`;
- diff không chứa unrelated M0-04 duplication sau khi base đã được sync.

## 2.5 Definition of Done kỹ thuật

M1-03 chỉ được đánh dấu hoàn thành khi đồng thời đạt:

1. Branch đúng và dependency M0-04 được xử lý sạch.
2. Frozen W1 có ít nhất 30 sample.
3. Dataset/prompt/schema có identity/version rõ.
4. Reusable `src/pipeline/extractor.py` tồn tại và được test.
5. Pydantic + JSON Schema không drift.
6. Source binding + semantic validation là hard gate.
7. W1 package tồn tại dưới canonical handoff area.
8. M1-02 có interface contract đủ để replay đúng full Node 1 requests.
9. Target vLLM final run đạt `schema_valid_rate >= 95%`.
10. Nếu N=30 thì ít nhất 29/30 sample fully valid.
11. Raw/validated/failure artifacts và summary có thể audit.
12. Baseline regression checks vẫn pass.
13. Worklog/nghiệm thu M1-03 được cập nhật.
14. Không có change ngoài scope.

## 2.6 Stop conditions / trường hợp phải dừng và review lại spec

Không tiếp tục code theo plan cũ nếu xảy ra một trong các trường hợp:

- M0-04 thay đổi public contract sau khi M1-03 đã branch.
- Cần breaking change `invoice_extraction.v1`.
- Target vLLM không thể giữ cùng request semantics giữa các lane.
- W1 final không thể đạt >=30 sample hợp lệ với provenance chấp nhận được.
- Pydantic và JSON Schema không thể đồng bộ mà phải thay đổi public contract.
- Benchmark harness chỉ chạy được user-only prompt và chưa có adapter cho full Node 1 request.
- Muốn dùng post-processing để “sửa” business field nhằm đạt 95%.
- Muốn thay prompt/dataset sau khi xem final benchmark nhưng vẫn giữ cùng workload identity.

Trong các trường hợp này, SHALL cập nhật OpenSpec/code plan trước khi tiếp tục.

## 2.7 Ownership / phân công thực thi

Theo card/checklist M1-03:

### Dương — Lead M1-03

Chịu trách nhiệm chính:

- W1 dataset 30+ mẫu;
- provenance/coverage/freeze;
- invoice contract và Pydantic schema;
- prompt/dataset package;
- worklog `logs/Duong/M1-03/`;
- cập nhật nội dung báo cáo liên quan.

### Đại — Partner M1-03 / owner M1-02 integration

Chịu trách nhiệm chính/phối hợp:

- reusable `src/pipeline/extractor.py`;
- OpenAI-compatible/vLLM invocation;
- test suite extractor;
- schema-valid-rate execution;
- adapter/handoff để M1-02 Benchmark Harness replay đúng W1 requests.

### Boundary với M1-02

- M1-03 **định nghĩa và bàn giao W1 contract**.
- M1-02 **sở hữu concurrency sweep, streaming metrics, warm-up/repeat và benchmark scheduling**.
- Nếu cần sửa `experiments/benchmark_harness.py` để consume full W1 request, change đó SHOULD được track trong M1-02 hoặc được ghi rõ là integration commit phối hợp, tránh scope ownership mơ hồ.

## 2.8 Traceability — Trello → Implementation Evidence → Check

| Cam kết M1-03 | Evidence dự kiến | Check nghiệm thu |
|---|---|---|
| Dataset W1 30+ mẫu | `experiments/configs/w1/manifest.csv`, `samples.jsonl` | sample count >=30, unique IDs, provenance, coverage |
| Không leak annotation | frozen request package | automated request-content audit |
| Pydantic invoice schema | Pydantic model trong `src/pipeline/` | Pydantic ↔ JSON Schema equivalence tests |
| Giữ `invoice_extraction.v1` | `src/pipeline/schemas/invoice_extraction.schema.json` | `check_invoice_contract.py` + schema validation |
| Node 1 reusable | `src/pipeline/extractor.py` | unit tests + endpoint integration tests |
| Gọi vLLM/OpenAI-compatible | extractor/runner + final run artifact | preflight + target vLLM run |
| Output strict JSON | raw/validated/failure artifacts | strict parse tests/rate |
| Source binding | validated output/failure records | source mismatch tests |
| Schema-valid >=95% | `experiments/results/m1_03_node1_vllm_summary.json` hoặc equivalent | `valid / attempted >= 0.95`; nếu N=30 cần >=29 |
| Frozen prompt | prompt file + version/hash in workload metadata | hash recorded and stable |
| Frozen dataset | workload metadata + dataset fingerprint | rebuild-stability test |
| Bàn giao M1-02 | `experiments/configs/w1/` | harness/adapter can replay full request semantics |
| Raw audit | per-sample raw responses | `sample_id` joinable, raw unchanged |
| Worklog kỹ thuật | `logs/Duong/M1-03/README.md` | ghi commands, kết quả, deviations |
| Cập nhật báo cáo | artifact/report path theo worklog | review manual trước PR |
| Git hygiene | PR `feat/m1-03-mc-ocr-node1-extractor -> main` | full gates + `git diff --check` |

### Tên evidence khuyến nghị

Để dễ truy vết, compact tracked summaries SHOULD dùng tên ổn định:

```text
experiments/results/m1_03_w1_dataset_summary.json
experiments/results/m1_03_node1_requests_summary.json
experiments/results/m1_03_node1_vllm_summary.json
```

Tên có thể đổi trong code plan nếu có lý do, nhưng worklog và PR SHALL chỉ ra mapping cuối cùng.



## 2.9 Dependency & Artifact Intake — đầu vào phải lấy từ branch/task khác

M1-03 **không phải một task độc lập từ số 0**. Nó nằm giữa các output của M0-04/M1-01 và input của M1-02/M1-04. Code plan SHALL bắt đầu bằng việc xác nhận và pin các upstream artifacts thay vì tái tạo chúng tùy ý.

### Dependency graph

```text
M0-04 VAT schema + MC-OCR devset
        |
        |  hard data/code upstream
        v
M1-03 Node 1 + Frozen W1
        |
        +--------------------+
        |                    |
        | runtime upstream   | workload/package downstream
        | from M1-01         v
        |                 M1-02 Harness + Sweep
        |                    |
        +--------------------+
                             v
                         M1-04 Gate
```

Chi tiết hơn:

```text
M0-04  ---> M1-03 <--- M1-01 runtime
                |
                v
             M1-02
                |
                v
             M1-04
```

M0-03 là dependency **gián tiếp** thông qua M1-02; M1-03 không cần cherry-pick/reimplement M0-03 riêng.

### D1 — M0-04 là hard upstream bắt buộc

Nguồn:

```text
branch: feature/m0-04-vat-invoice-schema-mc-ocr
PR: #2
```

M0-04 đã commit lên GitHub **cả devset và các artifact Node 1**, không chỉ summary.

Các artifact M1-03 SHALL ưu tiên reuse:

| Artifact M0-04 | Trạng thái sử dụng trong M1-03 |
|---|---|
| `data/processed/m0_04_mcocr_devset/samples.jsonl` | **MUST REUSE** làm nguồn 30 sample baseline nếu không có lý do loại sample |
| `data/processed/m0_04_mcocr_devset/manifest.csv` | **MUST REUSE** để đối chiếu sample/provenance baseline |
| `data/processed/m0_04_mcocr_devset/images/` | **REUSE FOR AUDIT**, không dùng làm model input Node 1 |
| `data/processed/m0_04_mcocr_devset/node1/node1_inputs.jsonl` | **REFERENCE/REGRESSION** cho input formatting |
| `data/processed/m0_04_mcocr_devset/node1/node1_chat_requests.jsonl` | **MUST COMPARE AGAINST** khi build W1 frozen package |
| `data/processed/m0_04_mcocr_devset/node1/inputs/` | **REFERENCE/REGRESSION** |
| `data/processed/m0_04_mcocr_devset/node1/ollama_qwen2_5_7b/raw_responses.jsonl` | **REFERENCE ONLY**, không phải final M1 output |
| `data/processed/m0_04_mcocr_devset/node1/ollama_qwen2_5_7b/validated_outputs.jsonl` | **REGRESSION ORACLE/AUDIT**, không phải gold semantic truth |
| `data/processed/m0_04_mcocr_devset/invoices_v1.sqlite` | **REFERENCE/SMOKE**, không phải source-of-truth W1 |
| `experiments/results/m0_04_mcocr_devset_manifest.csv` | **TRACKED EVIDENCE** của baseline selection |
| `experiments/results/m0_04_mcocr_devset_summary.json` | **TRACKED EVIDENCE**: baseline 30 sample |
| `experiments/results/m0_04_node1_requests_summary.json` | **TRACKED EVIDENCE** cho prompt/request v1 |
| `experiments/results/m0_04_node1_ollama_summary.json` | **TRACKED FUNCTIONAL BASELINE**, 30/30 local |
| `src/pipeline/prompts/node1_extractor_system.md` | **MUST REUSE/REVIEW**, không viết prompt mới từ đầu |
| `src/pipeline/schemas/invoice_extraction.schema.json` | **MUST REUSE**, authoritative public contract |
| `src/pipeline/openai_compatible.py` | **REUSE** transport layer nếu phù hợp |
| `scripts/prepare_mcocr_devset.py` | **REUSE ONLY IF** cần regenerate/expand dataset |
| `scripts/build_node1_requests.py` | **REUSE/GENERALIZE** |
| `scripts/run_node1_extractor.py` | **REFACTOR**, không bỏ behavior đã được test |
| `scripts/check_invoice_contract.py` | **EXTEND**, không thay bằng checker khác mà mất regression |
| `tests/test_run_node1_extractor.py` | **KEEP + EXTEND** |

#### Hệ quả

Nếu M1-03 chốt dùng đúng 30 baseline samples đã commit:

- **không cần tải lại Kaggle MC-OCR** chỉ để hoàn thành W1;
- `data/raw/mc_ocr_2021` không phải runtime prerequisite;
- W1 final được **promote/normalize từ tracked M0-04 artifacts** sang canonical package `experiments/configs/w1/`;
- phải giữ nguyên sample identity và exact OCR text trừ khi spec ghi rõ sample bị loại/thay vì lỗi dữ liệu.

Chỉ cần raw Kaggle/source dataset khi:

- muốn chọn thêm sample;
- muốn thay sample lỗi;
- muốn regenerate để kiểm chứng selection;
- hoặc baseline committed artifacts bị phát hiện không nhất quán.

### D2 — M1-01 là runtime upstream cho final vLLM acceptance

Nguồn/task:

```text
Trello: [M1-01] Server vLLM 2 làn & Telemetry
branch: feat/m1-01-vllm-dual-lane-server
```

M1-01 hiện cung cấp/đang xây:

```text
scripts/launch_lane1_vanilla.sh
scripts/launch_lane2_static.sh
scripts/switch_lane.sh
scripts/test_lane1.py
scripts/test_lane2.py
requirements-server.txt
```

Runtime contract hiện thấy từ branch:

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

M1-03 **không copy server implementation của M1-01** vào branch của mình.

M1-03 cần output/runtime từ M1-01 để chạy:

- OpenAI-compatible endpoint;
- exact served model name;
- endpoint health/model preflight;
- target T4/vLLM runtime đủ ổn định để chạy frozen W1.

#### Dependency gate

M1-03 có thể hoàn thành trước các phần:

- dataset freeze;
- schema/Pydantic;
- reusable extractor;
- local tests;
- Ollama/compatible smoke;
- W1 package;
- M1-02 adapter test ở mức serialization/unit.

Nhưng **M1-03 final vLLM acceptance SHALL BLOCK** nếu M1-01 chưa cung cấp ít nhất một target vLLM lane chạy được.

Canonical correctness gate của M1-03 SHOULD chạy trên **Vanilla lane** trước:

```text
qwen2.5-7b-vanilla
```

Lý do: kiểm tra Node 1 correctness không nên phụ thuộc speculative behavior.

Static lane replay/performance comparison thuộc M1-02 baseline sweep; M1-03 MAY smoke-test static lane để xác nhận interface compatibility nhưng không cần lấy performance sweep làm điều kiện để hoàn thành extractor.

### D3 — M1-02 là downstream consumer và integration peer, không phải hard upstream cho core M1-03

Nguồn/task:

```text
Trello: [M1-02] Benchmark Harness & Sweep
branch: feat/m1-02-concurrency-sweep-baseline
```

M1-02 hiện sửa:

```text
experiments/benchmark_harness.py
experiments/requirements-m1.txt
```

Branch M1-02 hiện vẫn xây request theo dạng user-only prompt trong `benchmark_harness.py`. Vì vậy nó **chưa tự động tương đương Node 1 frozen request contract**.

Quan hệ trách nhiệm:

**M1-03 cung cấp:**
- `experiments/configs/w1/`;
- W1 manifest/request package;
- prompt/schema/dataset identities;
- thin loader/adapter;
- request-equivalence integration test.

**M1-02 cung cấp:**
- concurrency scheduler;
- warm-up/repeat;
- lane execution;
- streaming TTFT/TPOT/ITL;
- raw benchmark runs;
- `experiments/results/summary.csv`.

M1-03 **không cần chờ M1-02 hoàn thành full sweep để merge core extractor**, nhưng handoff adapter SHALL được chứng minh trước khi M1-03 được coi là integration-ready.

### D4 — M0-03 chỉ là transitive dependency

M1-02 kế thừa environment/harness từ M0-03. M1-03 không trực tiếp tiêu thụ M0-03 output ngoài interface mà M1-02 đã chuẩn hóa.

Do đó code plan M1-03 SHALL NOT:
- cherry-pick M0-03 riêng;
- sửa benchmark timing logic từ M0-03;
- nhận ownership environment sweep.

### D5 — M1-04 là downstream gate, không phải input

M1-04 sẽ tiêu thụ:

- final W1 package;
- M1-03 schema-valid summary;
- M1-02 Vanilla/Static benchmark results;
- worklogs/reproducibility evidence.

M1-03 SHALL bàn giao artifact đủ rõ để M1-04 review mà không cần chạy lại data preparation bằng tay.

### Upstream artifact pinning

Vì các branch M1-01/M1-02 vẫn đang phát triển, OpenSpec **không khóa cứng commit SHA của chúng tại thời điểm viết**.

Khi bắt đầu code/apply, worklog M1-03 SHALL ghi:

```text
upstream_m0_04_commit
upstream_m1_01_commit_used_for_vllm_acceptance
upstream_m1_02_commit_used_for_adapter_integration
```

`workload.json`/final summary SHOULD ghi các commit thực sự đã dùng nếu chúng ảnh hưởng kết quả.

M0-04 branch head đã được kiểm tra trong lần rà soát này, nhưng code plan SHALL pin lại SHA tại thời điểm bắt đầu implementation để tránh dùng một branch đã thay đổi mà không nhận biết.

### Dependency readiness matrix

| Dependency | Có cần để bắt đầu M1-03? | Có cần để final M1-03? | Output M1-03 lấy |
|---|---:|---:|---|
| M0-04 | **Có** | **Có** | schema, prompt, devset 30, request builder/runner/tests |
| M1-01 | Không cho dataset/schema; có cho runtime | **Có** | target vLLM endpoint/model contract |
| M1-02 | Không | Không cho core correctness; **Có cho handoff integration readiness** | harness interface/adapter target |
| M0-03 | Không trực tiếp | Không trực tiếp | thông qua M1-02 |
| M1-04 | Không | Không | đây là downstream consumer |

### Gate mới — kiểm tra upstream intake trước code plan/apply

Trước implementation, SHALL xác nhận:

1. M0-04 `samples.jsonl` có đủ 30 sample baseline.
2. `manifest.csv` và `samples.jsonl` có cùng sample set.
3. M0-04 request package vẫn khớp prompt/schema version v1.
4. Không rebuild dataset từ Kaggle nếu không có lý do được ghi trong plan.
5. Commit/ref M0-04 dùng làm base được ghi lại.
6. M1-01 runtime contract/served model name được pin trước final vLLM run.
7. M1-02 adapter target được kiểm tra lại vì harness branch có thể tiếp tục thay đổi.


---

# 3. Đề xuất thay đổi

## Why

M0-04 đã chứng minh được vertical slice Node 1: MC-OCR text → LLM → `invoice_extraction.v1` → validation → SQLite. Tuy nhiên M1 cần biến PoC đó thành một **Workload W1 ổn định, có thể tái lập và công bằng giữa các serving lane**, đủ lớn để dùng làm baseline benchmark và đủ chặt để không nhầm “JSON hợp lệ” với “trích xuất đúng”.

Nếu dataset, prompt, source binding, schema hoặc generation parameters thay đổi giữa các lane/run, kết quả Vanilla vs Static sẽ không còn đo cùng một workload.

## What Changes

- Mở rộng từ dev-set 30 mẫu của M0-04 thành **W1 final >= 30 mẫu**.
- Tách rõ **dev/regression set** và **frozen benchmark set**.
- Đóng băng prompt, schema, generation settings và request semantics trước benchmark chính thức.
- Giữ `invoice_extraction.v1` làm baseline contract, trừ khi review xác nhận cần breaking change.
- Bổ sung Pydantic model/semantic validation tương thích với JSON Schema.
- Tách reusable Node 1 extraction behavior khỏi CLI/file orchestration.
- Định nghĩa source binding và chống hallucination thành acceptance contract.
- Định nghĩa rõ các lớp success/failure và metric `schema_valid_rate`.
- Đóng gói W1 để M1-02 sử dụng trực tiếp, không tự build prompt lại.
- Lưu raw output, canonical output, failures và provenance đầy đủ.
- Final validation chạy qua target vLLM/OpenAI-compatible path; Ollama chỉ là smoke test.
- Không thay đổi engine speculative decoding, proposer, hoặc concurrency harness trong M1-03.

## Capabilities

### New Capabilities

- `w1-dataset`: xây dựng, freeze, định danh và kiểm tra chất lượng dataset Workload W1.
- `node1-invoice-extraction`: hành vi text-to-JSON của Node 1 qua OpenAI-compatible API.
- `invoice-extraction-validation`: parse, source binding, schema/semantic validation, canonicalization an toàn.
- `w1-benchmark-artifacts`: request package, result artifacts, provenance và metric bàn giao cho M1-02.

### Modified Capabilities

Không có capability OpenSpec hiện hữu để modify vì repository hiện chưa có `openspec/specs/`.

## Impact

Các vùng dự kiến bị ảnh hưởng khi bước implementation bắt đầu:

- `src/pipeline/`
- `scripts/` hoặc tooling tương đương
- `tests/`
- `data/processed/` hoặc khu vực build artifact
- `experiments/configs/`
- `experiments/runs/`
- `experiments/results/`
- `logs/Duong/M1-03/`
- dependency cho Pydantic nếu chưa có

Không thuộc phạm vi change:

- core vLLM proposer/controller
- static/adaptive decoding implementation
- benchmark concurrency scheduler của M1-02
- Node 2 Text-to-SQL
- Node 3 BI/report generation

---

# 4. Đặc tả W1 Dataset

## Purpose

Capability này định nghĩa cách tạo, kiểm tra, đóng băng và định danh Workload W1 để tất cả serving lane nhận chính xác cùng một tập tác vụ Node 1, không bị data drift hoặc prompt leakage làm sai lệch phép đo.

## ADDED Requirements

### Requirement: W1 final có tối thiểu 30 mẫu

Hệ thống **SHALL** tạo một frozen W1 benchmark set chứa ít nhất 30 `sample_id` duy nhất và hợp lệ.

#### Scenario: Đủ số lượng
- **WHEN** W1 final manifest được tạo
- **THEN** số `sample_id` duy nhất SHALL >= 30
- **AND** mỗi sample SHALL ánh xạ tới đúng một source record.

#### Scenario: Không đủ số lượng
- **WHEN** số sample đạt tiêu chí < 30
- **THEN** bước freeze SHALL thất bại
- **AND** artifact đó SHALL NOT được gắn nhãn W1 final.

### Requirement: M0-04 dev-set là upstream baseline và có thể được promote thành W1 final

30 mẫu M0-04 **MAY** được tái sử dụng làm W1 final nếu đáp ứng đầy đủ tiêu chí coverage, provenance, freeze và benchmark fairness; nếu chưa đáp ứng thì dùng làm dev/regression baseline và build lại W1 final.

#### Scenario: Tái sử dụng M0-04
- **WHEN** 30 sample M0-04 đáp ứng coverage, provenance, freeze và request-contract requirements
- **THEN** chúng MAY được dùng làm W1 final
- **AND** nếu dùng cho prompt/schema tuning trước đó, việc freeze/versioning SHALL bảo đảm final benchmark không bị prompt/data drift.

### Requirement: W1 baseline phải ưu tiên lấy từ tracked M0-04 artifacts

Nếu 30 sample M0-04 đáp ứng validation/freeze criteria, M1-03 **SHALL** dùng chúng làm nguồn W1 thay vì regenerate ngẫu nhiên từ raw MC-OCR.

#### Scenario: Baseline hợp lệ
- **WHEN** M0-04 `samples.jsonl` và manifest có 30 sample nhất quán
- **THEN** M1-03 SHALL promote/normalize các sample đó vào W1 package
- **AND** giữ nguyên `sample_id` và exact OCR text.

#### Scenario: Cần thay sample
- **WHEN** một baseline sample bị loại vì provenance/data defect
- **THEN** lý do SHALL được ghi rõ
- **AND** replacement SHALL tuân theo selection/provenance rules
- **AND** dataset fingerprint SHALL thay đổi.

### Requirement: Sample ID ổn định

Mỗi sample **SHALL** có `sample_id` không rỗng, duy nhất và độc lập với model, lane, run ID và vị trí dòng.

#### Scenario: Build lại request
- **WHEN** cùng frozen dataset được build lại
- **THEN** cùng document SHALL giữ nguyên `sample_id`.

### Requirement: Dataset có provenance đầy đủ

Mỗi sample **SHALL** có đủ metadata để truy ngược nguồn và tái tạo model input.

#### Scenario: Audit một sample
- **WHEN** reviewer tra cứu `sample_id`
- **THEN** manifest SHALL chỉ ra source dataset, source row hoặc source identifier tương đương, image ID/path nếu có, và exact Node 1 input text.

### Requirement: Không rò rỉ annotation/gold label vào prompt

MC-OCR `anno_labels`, gold fields và evaluator-only metadata **MUST NOT** xuất hiện trong model-visible prompt trừ khi chính chuỗi đó tồn tại tự nhiên trong receipt text.

#### Scenario: Sample có nhãn SELLER
- **WHEN** source row chứa annotation label `SELLER`
- **THEN** label này MAY được giữ trong evaluation metadata
- **BUT** SHALL NOT được chèn vào system/user content.

### Requirement: Tiền xử lý text phải xác định và tái lập được

Mọi normalization trước khi build request **SHALL** là deterministic và không thay đổi nghĩa nghiệp vụ.

#### Scenario: Sửa mojibake
- **WHEN** text khớp mẫu encoding corruption đã biết
- **THEN** repair SHALL cho cùng kết quả ở mọi lần build.

#### Scenario: Dọn whitespace
- **WHEN** loại blank line hoặc trim whitespace
- **THEN** thứ tự các dòng nội dung SHALL được bảo toàn.

### Requirement: Dataset final phải được freeze trước benchmark

W1 final **SHALL** được đóng băng trước khi chạy ma trận M1 chính thức.

#### Scenario: Sau khi freeze
- **WHEN** một sample, OCR text, selection rule hoặc manifest thay đổi
- **THEN** dataset identity SHALL thay đổi
- **AND** kết quả benchmark cũ SHALL NOT được gộp như cùng một workload version.

### Requirement: Dataset phải có identity/fingerprint

W1 final **SHALL** có một fingerprint xác định từ canonical manifest hoặc canonical request package.

#### Scenario: Rebuild không đổi
- **WHEN** build lại từ cùng canonical data/config
- **THEN** fingerprint SHALL giống nhau.

#### Scenario: Input thay đổi
- **WHEN** raw text hoặc sample membership thay đổi
- **THEN** fingerprint SHALL khác.

### Requirement: W1 semantic input độc lập serving lane

Cùng một `sample_id` **SHALL** có cùng task semantics ở mọi lane được so sánh.

#### Scenario: Vanilla và Static
- **WHEN** sample được gửi vào `vanilla` và `static`
- **THEN** system prompt version/hash, user content, schema version và generation settings SHALL giống nhau
- **EXCEPT** metadata không làm thay đổi nhiệm vụ như lane/run identifier.

### Requirement: W1 phải đại diện cho JSON extraction entropy thấp

Dataset/package **SHALL** duy trì bản chất W1 là structured JSON extraction, không trộn free-form analysis vào cùng workload.

#### Scenario: Request W1
- **WHEN** harness đọc một request W1
- **THEN** expected output SHALL là structured invoice JSON theo frozen contract
- **AND** request SHALL NOT yêu cầu giải thích văn xuôi dài ngoài JSON.

### Requirement: Dữ liệu bổ sung phải có quyền sử dụng và kiểm soát thông tin nhạy cảm

Nếu W1 bổ sung hóa đơn ngoài MC-OCR, dữ liệu **SHALL** là public/licensed/synthetic hoặc đã được xử lý phù hợp để không đưa thông tin nhạy cảm không cần thiết vào benchmark.

#### Scenario: Hóa đơn thực tế chứa PII
- **WHEN** sample có thông tin cá nhân không cần cho benchmark
- **THEN** sample SHALL được loại hoặc làm sạch trước khi freeze
- **AND** provenance SHALL ghi nguồn/loại dữ liệu.

### Requirement: Báo cáo coverage dataset

W1 final manifest/summary **SHALL** báo cáo coverage tối thiểu cho các nhóm field quan trọng.

#### Scenario: Freeze W1
- **WHEN** final summary được tạo
- **THEN** SHALL có số lượng/tỷ lệ sample có seller, buyer, date/time, total, items và tax/VAT evidence
- **SO THAT** reviewer có thể phát hiện dataset 30+ nhưng quá đơn điệu.

---

# 5. Đặc tả Node 1 Invoice Extraction

## Purpose

Capability này định nghĩa hành vi quan sát được của Node 1: nhận text hóa đơn/receipt tiếng Việt và trả về đúng một object `invoice_extraction.v1` có thể audit, không bịa dữ liệu nghiệp vụ và hoạt động qua endpoint OpenAI-compatible.

## ADDED Requirements

### Requirement: Node 1 dùng OpenAI-compatible Chat Completions

Node 1 **SHALL** gọi model qua OpenAI-compatible `/v1/chat/completions` hoặc endpoint tương đương đúng contract đang được server vLLM cung cấp.

#### Scenario: Backend hợp lệ
- **WHEN** endpoint reachable và model tồn tại
- **THEN** Node 1 SHALL gửi frozen request contract
- **AND** ghi model/endpoint/run provenance.

#### Scenario: Model không tồn tại
- **WHEN** configured model không có trên endpoint
- **THEN** run SHALL fail-fast trước benchmark loop
- **AND** không tính lỗi này như extraction failure của từng sample.

### Requirement: Final vLLM acceptance phụ thuộc runtime contract từ M1-01

M1-03 final correctness run **SHALL** sử dụng target vLLM runtime được M1-01 cung cấp, ưu tiên Vanilla lane làm canonical correctness lane.

#### Scenario: M1-01 runtime chưa sẵn sàng
- **WHEN** không có target vLLM endpoint/model chạy được
- **THEN** M1-03 MAY hoàn thành local implementation/tests
- **BUT** SHALL NOT đánh dấu final vLLM acceptance là pass.

#### Scenario: Vanilla runtime sẵn sàng
- **WHEN** M1-01 Vanilla endpoint health/preflight pass
- **THEN** frozen W1 SHALL được replay trên endpoint đó
- **AND** kết quả dùng cho `schema_valid_rate` final của M1-03.

### Requirement: Không có silent fallback làm đổi workload

Node 1 **MUST NOT** âm thầm bỏ hoặc đổi generation setting vì backend khác nhau.

#### Scenario: `response_format` không được hỗ trợ
- **WHEN** target backend không hỗ trợ request contract đã freeze
- **THEN** run SHALL báo configuration incompatibility
- **AND** không được tự chạy một request khác semantics chỉ trên lane đó.

### Requirement: Response thành công phải là đúng một JSON object

Assistant content hợp lệ **SHALL** parse trực tiếp thành một JSON object.

#### Scenario: JSON thuần
- **WHEN** response content là một JSON object hợp lệ
- **THEN** strict JSON parse SHALL pass.

#### Scenario: Markdown fence hoặc prose bao quanh
- **WHEN** content không parse trực tiếp thành một object
- **THEN** sample SHALL được phân loại `json_parse_failure`.

### Requirement: Top-level contract là invoice_extraction.v1

Output **SHALL** chứa đúng các top-level keys:

`schema_version`, `document_type`, `source`, `seller`, `buyer`, `invoice`, `totals`, `taxes`, `items`, `extraction`.

#### Scenario: Field nghiệp vụ bị thiếu
- **WHEN** model không có bằng chứng cho scalar field
- **THEN** key SHALL vẫn tồn tại với `null`.

#### Scenario: Collection không có dữ liệu
- **WHEN** không có items/taxes/evidence
- **THEN** collection SHALL là `[]`.

### Requirement: Source metadata thuộc quyền sở hữu pipeline

`source.sample_id`, `source.img_id`, `source.image_path`, `source.raw_text` **SHALL** khớp request envelope tuyệt đối.

#### Scenario: Copy chính xác
- **WHEN** model echo source đúng tuyệt đối
- **THEN** source-binding SHALL pass.

#### Scenario: Model “sửa” source text
- **WHEN** model normalize, rút gọn hoặc sửa source field
- **THEN** source-binding SHALL fail
- **AND** payload SHALL NOT được tính valid.

### Requirement: Không được bịa business facts

Node 1 **MUST NOT** tạo seller, buyer, item, tax, invoice number, payment hoặc amount nếu OCR text không hỗ trợ.

#### Scenario: Không có buyer
- **WHEN** source không có buyer evidence
- **THEN** buyer fields SHALL là `null`.

#### Scenario: Không có item lines
- **WHEN** item table không thể xác định từ source
- **THEN** `items` SHALL là `[]`
- **OR** chỉ chứa row có evidence đủ rõ
- **AND** không được bịa item để cân tổng tiền.

### Requirement: Phân loại document type dựa trên evidence

`document_type` **SHALL** thuộc `retail_receipt`, `vat_invoice`, `unknown`.

#### Scenario: VAT evidence rõ
- **WHEN** source có evidence rõ đây là VAT/tax invoice
- **THEN** `vat_invoice` MAY được dùng.

#### Scenario: Chỉ là retail receipt
- **WHEN** không có VAT invoice evidence
- **THEN** Node 1 SHALL NOT gắn `vat_invoice` chỉ vì workload tên VAT Extractor.

### Requirement: Chuẩn hóa ngày giờ không suy diễn timezone

Date/time **SHALL** theo frozen format và không suy ra timezone nếu source không có.

#### Scenario: Date DD/MM/YYYY
- **WHEN** source có ngày hợp lệ
- **THEN** `issued_date` SHALL là `YYYY-MM-DD`.

#### Scenario: Có cả date và time
- **WHEN** cả hai được xác định
- **THEN** `issued_at` SHALL là `YYYY-MM-DDTHH:MM:SS`.

#### Scenario: Không có timezone
- **WHEN** source không chứa timezone
- **THEN** output SHALL NOT thêm offset/timezone.

### Requirement: Money giữ raw và normalized value

Mỗi money object **SHALL** giữ cả representation OCR và numeric value.

#### Scenario: VND có dấu phân cách
- **WHEN** raw amount là dạng như `42.000`
- **THEN** `raw` SHALL giữ evidence phù hợp
- **AND** `value` SHALL là numeric normalized equivalent
- **AND** currency SHALL là `VND` nếu không có currency khác.

### Requirement: Không suy subtotal từ total

`totals.subtotal` **MUST NOT** được copy từ final total khi source chỉ có tổng tiền cuối.

#### Scenario: Chỉ có “Tổng tiền”
- **WHEN** source chỉ có final total
- **THEN** amount SHALL map vào `totals.total_amount`
- **AND** subtotal raw/value SHALL là `null`.

### Requirement: Item line number phải duy nhất

Mỗi item **SHALL** có positive integer `line_number`, và các line number SHALL duy nhất trong invoice.

#### Scenario: Duplicate line number
- **WHEN** hai item cùng `line_number`
- **THEN** payload SHALL fail semantic validation.

### Requirement: Tax chỉ được tạo từ evidence

`taxes` **SHALL** chỉ chứa dòng thuế có căn cứ từ source.

#### Scenario: Không có VAT evidence
- **WHEN** source không cho phép xác định tax line
- **THEN** `taxes` SHALL là `[]`
- **AND** post-processing SHALL NOT suy tax từ phép trừ subtotal/total.

### Requirement: Extraction status nhất quán với missing_fields

`extraction.status` **SHALL** là `ok`, `partial` hoặc `failed`.

#### Scenario: status ok
- **WHEN** status = `ok`
- **THEN** `missing_fields` SHALL rỗng.

#### Scenario: status partial
- **WHEN** status = `partial`
- **THEN** `missing_fields` SHALL có ít nhất một field quan trọng.

#### Scenario: status failed
- **WHEN** không có useful business extraction
- **THEN** status MAY là `failed`.

### Requirement: Evidence phải là substring nguyên văn

Mỗi `field_evidence.raw_text` **SHALL** là verbatim substring của `source.raw_text`.

#### Scenario: Evidence hợp lệ
- **WHEN** evidence cho `seller.name` tồn tại nguyên văn trong source
- **THEN** evidence MAY được giữ.

#### Scenario: Evidence tự tạo
- **WHEN** evidence không xuất hiện nguyên văn
- **THEN** evidence SHALL NOT xuất hiện trong canonical validated output.

### Requirement: Confidence chỉ là audit hint

`extraction.confidence` **MUST NOT** được dùng như ground-truth quality metric hoặc input cho proposer.

#### Scenario: Không có cơ sở confidence
- **WHEN** confidence không thể biện minh
- **THEN** field SHALL là `null`
- **AND** schema validity SHALL được đánh giá độc lập.

### Requirement: Generation settings của W1 phải cố định

Final W1 extraction **SHALL** dùng deterministic/low-variance generation settings giống nhau giữa các lane.

#### Scenario: Baseline comparison
- **WHEN** cùng sample chạy Vanilla và Static
- **THEN** `temperature` SHALL là 0
- **AND** seed SHALL giống nhau khi backend hỗ trợ seed
- **AND** max output token policy SHALL giống nhau
- **AND** response-format policy SHALL giống nhau.

---

# 6. Đặc tả Validation

## Purpose

Capability này định nghĩa ranh giới validation giữa raw LLM response và một Node 1 output đủ điều kiện benchmark, đồng thời ngăn canonicalization trở thành một “extractor thứ hai” âm thầm sửa hoặc bịa dữ liệu.

## ADDED Requirements

### Requirement: Validation pipeline có thứ tự rõ ràng

Một sample **SHALL** đi qua các lớp kiểm tra theo thứ tự logic:

1. transport/response envelope
2. strict JSON parsing
3. permitted deterministic canonicalization
4. exact source binding
5. schema validation
6. semantic validation
7. optional persistence

#### Scenario: JSON sai
- **WHEN** strict JSON parse fail
- **THEN** sample SHALL không được ghi nhận là schema-valid.

#### Scenario: Source mismatch
- **WHEN** schema shape đúng nhưng source binding sai
- **THEN** sample SHALL invalid.

### Requirement: Pydantic và JSON Schema mô tả cùng một public contract

Pydantic model và `invoice_extraction.v1` JSON Schema **SHALL** đồng nhất về required fields, nullability, enum, ranges và nested shape.

#### Scenario: Canonical valid fixture
- **WHEN** fixture được Pydantic chấp nhận
- **THEN** JSON Schema SHALL cũng chấp nhận.

#### Scenario: Drift
- **WHEN** cùng payload được một validator accept nhưng validator còn lại reject vì public contract khác nhau
- **THEN** test SHALL fail
- **AND** change SHALL không đạt acceptance.

### Requirement: Có một source of truth được xác định rõ

Implementation plan **MUST** chọn một cơ chế chống drift:
- Pydantic model là authoritative và generate JSON Schema; **hoặc**
- JSON Schema là authoritative và Pydantic được regression-test chống contract đó.

#### Scenario: Code plan
- **WHEN** tasks được viết
- **THEN** plan SHALL chỉ rõ source of truth
- **AND** không được để hai schema tiến hóa độc lập không kiểm soát.

### Requirement: Calendar validation là semantic, không chỉ regex

Validation **SHALL** reject impossible calendar values.

#### Scenario: Ngày không tồn tại
- **WHEN** `issued_date = 2026-02-31`
- **THEN** validation SHALL fail.

#### Scenario: issued_at mâu thuẫn
- **WHEN** `issued_at` date/time component khác `issued_date`/`issued_time`
- **THEN** validation SHALL fail.

### Requirement: Canonicalization chỉ sửa cấu trúc xác định được

Canonicalization **MAY** thực hiện deterministic repair không tạo business fact mới.

#### Scenario: Compose issued_at
- **WHEN** date và time hợp lệ tồn tại nhưng `issued_at = null`
- **THEN** canonicalizer MAY compose `issued_at`
- **AND** SHALL log normalization.

#### Scenario: Null totals container
- **WHEN** required money container bị trả về `null`
- **THEN** canonicalizer MAY thay bằng `{raw:null,value:null,currency:null}`
- **AND** SHALL log normalization.

#### Scenario: Missing source_label
- **WHEN** evidence object thiếu `source_label`
- **THEN** canonicalizer MAY đặt `null`
- **AND** SHALL log normalization.

#### Scenario: Evidence không thuộc source
- **WHEN** evidence text không là substring
- **THEN** canonicalizer MAY loại evidence row
- **AND** SHALL log việc loại.

### Requirement: Canonicalization không được tạo business value

Canonicalization **MUST NOT** tự điền seller, buyer, item, tax, invoice number, quantity, price, subtotal, VAT, total hoặc payment method.

#### Scenario: Có thể suy VAT bằng toán
- **WHEN** subtotal và total cho phép tính ra VAT
- **THEN** canonicalizer SHALL NOT tự thêm VAT nếu source không có evidence theo contract.

### Requirement: Raw assistant response phải bất biến

Raw assistant content **SHALL** được lưu nguyên trạng tách khỏi canonical payload.

#### Scenario: Payload bị normalize
- **WHEN** canonicalization áp dụng bất kỳ thay đổi nào
- **THEN** raw response SHALL không bị ghi đè
- **AND** canonical output SHALL có danh sách normalization đã áp dụng.

### Requirement: Failure taxonomy phải tách nguyên nhân

Mỗi failure **SHALL** thuộc một lớp đủ rõ để audit, tối thiểu:

- `transport_failure`
- `response_envelope_failure`
- `json_parse_failure`
- `source_binding_failure`
- `schema_validation_failure`
- `semantic_validation_failure`
- `persistence_failure`

#### Scenario: Aggregate failure report
- **WHEN** run có failure
- **THEN** summary SHALL có count theo category
- **AND** per-sample artifact SHALL có `sample_id`, category và message.

### Requirement: Persistence không quyết định Node 1 validity

Node 1 validation **SHALL** chạy được độc lập với SQLite.

#### Scenario: `--no-db` hoặc equivalent mode
- **WHEN** database persistence bị tắt
- **THEN** strict JSON/source/schema/semantic validation SHALL vẫn hoạt động đầy đủ.

### Requirement: SQLite v1 chỉ nhận validated payload

Nếu persistence được bật, chỉ payload đã pass validation **SHALL** được ghi vào SQLite v1.

#### Scenario: Invalid payload
- **WHEN** sample fail source/schema/semantic validation
- **THEN** invoice data SHALL NOT được insert như valid extraction.

---

# 7. Đặc tả Benchmark Artifacts

## Purpose

Capability này định nghĩa request package, provenance, raw output, validated output, failure artifact và metric cần bàn giao để M1-02 có thể chạy cùng W1 trên nhiều serving lane mà không tự tái tạo hoặc làm lệch workload.

## ADDED Requirements

### Requirement: W1 có frozen request package

Mỗi sample **SHALL** có request envelope chứa tối thiểu:

- `sample_id`
- `schema_version`
- `prompt_version`
- `prompt_hash`
- `dataset_id` hoặc dataset fingerprint
- `expected_source`
- Chat Completions request payload

#### Scenario: Harness load
- **WHEN** M1-02 load W1
- **THEN** harness SHALL có thể submit request trực tiếp
- **AND** không cần tự tạo lại business prompt.

### Requirement: Prompt phải có version và content hash

System prompt **SHALL** có logical version và cryptographic content hash.

#### Scenario: Chỉ sửa một câu prompt
- **WHEN** prompt content thay đổi
- **THEN** hash SHALL thay đổi
- **AND** benchmark output mới SHALL không bị gộp với prompt cũ như cùng config.

### Requirement: Dataset/request package phải có identity

Frozen request package **SHALL** có stable fingerprint hoặc equivalent identifier.

#### Scenario: Một raw OCR text thay đổi
- **WHEN** request package rebuild
- **THEN** package identity SHALL khác.

### Requirement: Run provenance phải đầy đủ

Mỗi run **SHALL** ghi tối thiểu:

- run ID
- lane
- model name/revision nếu biết
- endpoint/backend
- schema version
- prompt version/hash
- dataset/package identity
- generation settings
- commit SHA của code khi benchmark chính thức chạy
- hardware/runtime context do benchmark harness cung cấp
- output artifact locations

#### Scenario: So hai run
- **WHEN** reviewer so sánh hai summary
- **THEN** reviewer SHALL xác định được chúng có cùng W1/prompt/model contract hay không.

### Requirement: Mỗi sample có raw, validated hoặc failure artifact

Pipeline **SHALL** giữ raw response và kết quả xử lý theo `sample_id`.

#### Scenario: Sample hợp lệ
- **WHEN** sample pass toàn bộ validation
- **THEN** raw response và validated canonical payload SHALL có thể truy ngược cùng `sample_id`.

#### Scenario: Sample lỗi
- **WHEN** sample fail
- **THEN** raw response nếu có và failure record SHALL được giữ.

### Requirement: schema_valid_rate có mẫu số cố định

Metric acceptance SHALL được tính:

```text
schema_valid_rate = valid_samples / attempted_samples
```

Trong đó `valid_samples` chỉ gồm sample đã pass:
- strict JSON parse
- permitted canonicalization
- exact source binding
- JSON/Pydantic public contract
- semantic validation

#### Scenario: Final M1-03 acceptance
- **WHEN** frozen W1 final chạy trên target vLLM path
- **THEN** `schema_valid_rate` SHALL >= 0.95.

#### Scenario: Sample fail bị giữ trong mẫu số
- **WHEN** một sample fail JSON/schema/source
- **THEN** sample SHALL vẫn nằm trong `attempted_samples`
- **AND** không được xóa khỏi denominator để nâng tỷ lệ.

### Requirement: Schema validity không đồng nghĩa semantic accuracy

Report **SHALL** tách structural validity khỏi correctness của business fields.

#### Scenario: Hallucinated field nhưng đúng schema
- **WHEN** payload pass schema nhưng có value không được source hỗ trợ
- **THEN** payload SHALL NOT được mô tả là “trích xuất đúng nghiệp vụ” chỉ vì schema-valid.

### Requirement: Benchmark chính thức dùng target vLLM path

Ollama **MAY** dùng local smoke test; acceptance M1 **SHALL** chạy trên OpenAI-compatible vLLM target path.

#### Scenario: Ollama 30/30
- **WHEN** local Ollama run đạt 100%
- **THEN** kết quả đó SHALL chỉ được ghi là functional evidence
- **AND** SHALL NOT thay thế final vLLM acceptance run.

### Requirement: W1 handoff không chứa concurrency scheduler

W1 artifact **SHALL** mô tả task semantics; concurrency/warm-up/repeat nằm ở M1-02 harness.

#### Scenario: Harness chạy concurrency 1→32
- **WHEN** M1-02 thay concurrency
- **THEN** W1 semantic request SHALL không đổi.

### Requirement: Benchmark harness phải replay đầy đủ Node 1 request contract

M1-02 harness **SHALL** consume full frozen W1 request semantics hoặc một adapter chứng minh tương đương; user-only prompt không đủ để đại diện cho Node 1 contract.

#### Scenario: Harness load W1
- **WHEN** harness chạy sample W1
- **THEN** model-visible system/user messages SHALL tương đương canonical request package
- **AND** generation/response-format policy SHALL không bị thay đổi theo lane
- **AND** sample/result SHALL giữ `sample_id` để join với extraction validation artifacts.

#### Scenario: Streaming benchmark
- **WHEN** harness bật `stream=true` để đo TTFT/TPOT
- **THEN** streaming transport MAY khác functional runner
- **BUT** task semantics, prompt content và generation parameters SHALL giữ tương đương.

### Requirement: Same-prompt fairness giữa lane

So sánh Vanilla/Static/Adaptive **SHALL** dùng cùng frozen prompt package.

#### Scenario: Lane comparison
- **WHEN** hai lane chạy cùng sample
- **THEN** request task content SHALL giống nhau
- **AND** mọi khác biệt output/performance SHALL không do prompt drift.

---

# 8. Thiết kế

# Design

## Context

M0-04 đã có một vertical slice khá hoàn chỉnh:

```text
MC-OCR source
  -> dataset preparation
  -> Node 1 input/request builder
  -> OpenAI-compatible model call
  -> strict JSON parse
  -> deterministic canonicalization
  -> exact source binding
  -> JSON Schema + semantic validation
  -> optional SQLite persistence
  -> raw/validated/failure artifacts
```

M1-03 không nên viết lại từ đầu. Mục tiêu là **harden + refactor + freeze** để biến vertical slice thành W1 benchmark-grade.

W1 có vai trò đặc biệt trong đồ án: đây là workload structured JSON có entropy thấp, được kỳ vọng có acceptance rate/speculative benefit cao hơn W2/W3. Vì vậy, mọi thay đổi “giúp model” như guided output, prompt leakage hoặc post-processing mạnh đều có thể làm thay đổi chính workload đang đo.

## Goals

- W1 final >= 30 sample, frozen và reproducible.
- Node 1 reusable ngoài CLI.
- Public contract ổn định.
- Pydantic + JSON Schema không drift.
- Source binding và anti-hallucination là hard gate.
- vLLM path chạy được end-to-end.
- W1 handoff trực tiếp cho M1-02.
- Metric và artifact đủ để audit từng sample.
- Không làm thay đổi engine/proposer contribution.

## Non-Goals

- OCR vision model.
- Fine-tuning/training model cho hóa đơn.
- Huấn luyện draft head.
- Thay đổi vLLM speculative-decoding core.
- Viết benchmark sweep scheduler.
- Viết Node 2/Node 3.
- Tối ưu prompt dựa trên kết quả final benchmark rồi vẫn gọi đó là cùng workload.
- Dùng confidence tự khai báo của LLM làm correctness metric.

## Decision 1: Giữ `invoice_extraction.v1` làm baseline contract

M0-04 đã freeze schema v1 và SQLite user_version=1. M1 SHOULD ưu tiên giữ nguyên field names/nullability/normalized units.

Breaking change chỉ được phép khi:
- có requirement M1 không thể biểu diễn đúng trong v1; và
- review chấp thuận bump version.

Không bump version chỉ để “model dễ trả lời hơn”.

## Decision 2: Tách dev/regression set khỏi W1 final

Đề xuất:

- M0-04 30 sample → baseline/dev-regression set có thể tái sử dụng.
- M1-03 SHALL bảo đảm W1 final có ít nhất 30 sample; có thể giữ đúng 30 sample nếu chúng đáp ứng coverage và freeze criteria, hoặc mở rộng thêm.
- Prompt/schema được debug trên dev set.
- Khi W1 final được freeze, không chỉnh prompt/schema dựa trên lỗi của final set rồi vẫn dùng cùng version/hash.

Nếu bắt buộc sửa prompt sau freeze:
- bump prompt version/hash;
- rerun toàn bộ final set;
- không trộn summary cũ/mới.

## Decision 3: Pydantic + JSON Schema là một public contract

Trello yêu cầu Pydantic, M0-04 đã có JSON Schema Draft 2020-12.

Hai lựa chọn implementation hợp lệ:

**Phương án A — ưu tiên:** Pydantic model là source of truth và generate schema tương thích contract v1.  
**Phương án B:** JSON Schema v1 là source of truth; Pydantic model phải có equivalence tests.

Code plan phải chọn một phương án. Không để hai file schema được sửa độc lập mà không có drift test.

## Decision 4: Reusable extractor không sở hữu CLI/file layout

Reusable Node 1 layer nên chịu trách nhiệm:

- build/submit request hoặc nhận frozen request
- parse response
- canonicalization
- source binding
- validation
- trả structured result/failure

CLI/tooling chịu trách nhiệm:

- đọc JSONL
- loop sample
- path
- report files
- SQLite flags
- logging

Như vậy M1-02 có thể tích hợp mà không import script CLI.

## Decision 5: Labels/gold metadata không được model nhìn thấy

MC-OCR labels chỉ dùng:
- dataset selection
- audit
- evaluation

Không được chèn label vào prompt. Đây là hard requirement để giữ workload trung thực.

## Decision 6: Freeze request semantics trước lane comparison

Một W1 request package phải đủ để harness chỉ “replay” request.

Harness không được:
- tự build prompt khác
- tự thêm field hint
- thay response format theo lane
- thay temperature/seed theo lane

## Decision 7: Canonicalization cực hẹp

Giữ các repair hiện có của M0-04 vì chúng có tính cấu trúc và audit được:

- compose `issued_at`
- replace null required money container bằng empty money object
- default missing `source_label = null`
- bỏ unsupported evidence row

Không mở rộng canonicalizer thành rule-based invoice parser.

## Decision 8: Validity và quality là hai trục khác nhau

M1-03 gate bắt buộc: `schema_valid_rate >= 95%`.

Ngoài ra nên có semantic audit/field metrics nếu có gold data, nhưng metric này phải:
- định nghĩa riêng;
- không được gọi nhầm là schema-validity;
- không làm thay đổi denominator của acceptance gate.

## Decision 9: OpenAI-compatible transport phải provider-independent

Tầng pipeline chỉ phụ thuộc contract:
- list/check model nếu cần
- chat completions
- JSON response envelope

Ollama/vLLM là backend cụ thể. Final M1 dùng vLLM.

## Decision 10: Dataset identity và prompt identity là bắt buộc

Mọi kết quả benchmark cuối phải truy được tới:
- dataset fingerprint
- prompt hash
- schema version
- model
- lane
- commit
- generation settings

Nếu thiếu một trong các định danh quan trọng này, kết quả không nên được coi là fully reproducible.

## Decision 11: W1 giữ bản chất low-entropy structured generation

Không thêm free-form explanation vào expected output. Không yêu cầu Chain-of-Thought. Không thay đổi output thành prose rồi parse hậu kỳ.

W1 chính là constrained structured generation để tạo điểm entropy thấp trong bản đồ W1/W2/W3.

## Decision 12: SQLite là downstream persistence, không phải validator duy nhất

Validation phải chạy độc lập với database. SQLite chỉ nhận validated payload và phục vụ pipeline analytical workload/downstream.

## Data flow đề xuất

```text
Raw MC-OCR / approved source data
        |
        v
Dataset preparation
        |
        +--> provenance / coverage / manifest
        |
        v
Frozen W1 sample records
        |
        v
Frozen request builder
        |
        +--> dataset fingerprint
        +--> prompt version/hash
        +--> expected_source
        |
        v
M1-02 harness / direct runner
        |
        v
OpenAI-compatible vLLM endpoint
        |
        v
Raw assistant response  ------------------------+
        |                                       |
        v                                       |
Strict JSON parse                               |
        |                                       |
        v                                       |
Permitted canonicalization                      |
        |                                       |
        +--> normalization audit                 |
        |                                       |
        v                                       |
Exact source binding                            |
        |                                       |
        v                                       |
Pydantic + JSON Schema + semantic validation    |
        |                                       |
     pass/fail                                   |
        |                                       |
        +--> validated output / failure --------+
        |
        +--> optional SQLite v1 persistence
```

## Artifact contract đề xuất

### Tracked/reproducible config

Nên thuộc nhóm source-controlled:

- frozen W1 manifest hoặc compact config/manifest đủ nhỏ
- workload config
- prompt
- schema/model definitions
- compact aggregate summary
- scripts/tests

### Raw/heavy artifacts

Có thể gitignore nếu lớn:

- copied images
- raw response JSONL quy mô lớn
- temporary SQLite database
- full generated intermediate files

Nhưng nếu raw artifact không commit:
- summary phải ghi exact path/runtime ID;
- workflow phải có cách tái tạo;
- final thesis result source-of-truth phải theo quy ước `experiments/` chung của project.

## Metric definitions

### `attempted_samples`

Số sample mà final W1 run đã cố xử lý sau khi run preflight thành công.

### `transport_success_rate`

```text
responses_received / attempted_samples
```

### `strict_json_rate`

```text
strict_json_objects / attempted_samples
```

### `source_binding_rate`

```text
source_bound_payloads / attempted_samples
```

### `schema_valid_rate`

```text
fully_validated_payloads / attempted_samples
```

Acceptance M1-03:

```text
schema_valid_rate >= 0.95
```

### Semantic quality

Nếu có gold labels:
- field exact match/F1/normalized match cần định nghĩa riêng.
- không thay thế schema validity.
- không tự suy rằng 95% schema-valid = 95% extraction accuracy.

## Failure model

Mỗi failure record tối thiểu:

```json
{
  "sample_id": "w1_...",
  "category": "json_parse_failure",
  "stage": "strict_json",
  "message": "...",
  "latency_ms": 0,
  "run_id": "...",
  "lane": "...",
  "model": "..."
}
```

Không bắt buộc field names y hệt ví dụ này ở implementation, nhưng thông tin tương đương phải tồn tại.

## Test strategy tối thiểu

Code plan sau này phải bao phủ ít nhất:

1. Retail receipt hợp lệ.
2. VAT invoice có buyer/tax/items.
3. Missing buyer → null.
4. Missing items → `[]`.
5. Total có nhưng subtotal không có.
6. Chuẩn hóa date/time.
7. Compose `issued_at`.
8. Impossible date fail.
9. `issued_at` mâu thuẫn với date/time fail.
10. Negative money fail.
11. Negative quantity fail.
12. VAT rate >100 fail.
13. Duplicate item line number fail.
14. Source sample ID mismatch fail.
15. Source raw text mismatch fail.
16. Evidence không phải substring.
17. Markdown-wrapped JSON fail.
18. Missing nested required key fail.
19. Additional property fail.
20. `ok` + non-empty missing fields fail.
21. `partial` + empty missing fields fail.
22. Pydantic/JSON Schema equivalence fixture.
23. Backend unreachable.
24. Model unavailable.
25. Response envelope thiếu `choices[0].message.content`.
26. Canonicalization audit.
27. Raw response bất biến.
28. Persistence off vẫn validate.
29. Invalid payload không insert SQLite.
30. Cùng frozen sample sinh cùng request semantics giữa lane.
31. Dataset fingerprint ổn định khi input không đổi.
32. Dataset fingerprint thay đổi khi sample/raw text đổi.
33. Prompt hash thay đổi khi prompt content đổi.
34. Failed samples vẫn nằm trong denominator.
35. Final vLLM W1 run đạt `schema_valid_rate >= 0.95`.

## Risks and mitigations

### Risk: Dataset quá dễ

Nếu chọn 30 receipt gần như cùng template, W1 vẫn đủ số lượng nhưng kém đại diện.

**Mitigation:** coverage summary và diversity review trước freeze.

### Risk: Data leakage từ MC-OCR labels

Có thể làm task dễ giả tạo.

**Mitigation:** audit request package để bảo đảm labels chỉ ở evaluator metadata.

### Risk: Prompt overfit final W1

Chỉnh prompt sau khi xem từng lỗi final set làm metric optimistic.

**Mitigation:** dev set riêng; freeze prompt trước final acceptance run.

### Risk: Pydantic/JSON Schema drift

Hai validator dần không cùng contract.

**Mitigation:** một source-of-truth + equivalence regression.

### Risk: Canonicalization che lỗi model

Rule hậu kỳ càng nhiều càng biến Node 1 thành hybrid parser.

**Mitigation:** chỉ structural repair; mọi rule log được; cấm business inference.

### Risk: Backend-specific request behavior

Ollama/vLLM có thể khác support cho seed/response format.

**Mitigation:** preflight capability/config check; fail-fast thay vì silent fallback.

### Risk: Kết quả “95%” bị hiểu thành field accuracy

Đây là rủi ro báo cáo khoa học.

**Mitigation:** luôn ghi rõ `schema_valid_rate`, không viết “accuracy 95%” nếu chưa có semantic metric.

### Risk: Dùng invoice domain như contribution

Đi ngược QĐ-08.

**Mitigation:** giữ business logic trong `src/pipeline`; engine không biết workload là hóa đơn.

---

## Các quyết định đã chốt trước code plan

Các điểm dưới đây không còn là câu hỏi mở.

### Quyết định R1 — Tiêu chí chọn thêm sample ngoài baseline 30 mẫu

**Không bổ sung sample chỉ để tăng số lượng.** W1 minimum vẫn là 30 mẫu; 30 mẫu M0-04 có thể là W1 final nếu đạt toàn bộ gate.

Chỉ mở rộng dataset khi một trong các điều kiện sau xảy ra:

- coverage hiện tại không đủ đại diện cho contract cần benchmark;
- có sample hỏng/provenance không hợp lệ phải loại;
- cần tăng diversity để tránh 30 mẫu gần như cùng template/seller/layout.

Nếu bổ sung sample, sample mới SHALL ưu tiên theo thứ tự:

1. Có OCR text tiếng Việt đủ đọc và image/source provenance rõ.
2. Có `seller`, thời gian và `total_amount` hoặc evidence tương đương.
3. Tăng diversity về seller/layout/độ dài OCR.
4. Ưu tiên sample có item lines, VAT/tax, invoice number hoặc buyer fields vì đây là các nhánh schema baseline 30 mẫu có thể bao phủ yếu.
5. Không thêm duplicate/near-duplicate chỉ để tăng count.
6. Không hạ chất lượng OCR quá mức chỉ để đủ số.

Đối với nguồn MC-OCR, default selection baseline tiếp tục kế thừa:

```text
anno_image_quality >= 0.85
required labels: SELLER, ADDRESS, TIMESTAMPS
preferred label: TOTAL_COST
```

Nếu selection rule thay đổi để lấy diversity khác, SHALL bump dataset identity và ghi rõ lý do trong worklog.

**Không đặt hard quota giả tạo** cho buyer/VAT/items nếu source dataset không có đủ evidence. Thay vào đó, coverage SHALL được đo và báo cáo; nếu một nhóm field quan trọng có coverage bằng 0, reviewer phải quyết định bổ sung data hay ghi rõ limitation trước khi freeze.

### Quyết định R2 — Provenance cho dữ liệu ngoài MC-OCR

M1-03 SHALL có file machine-readable riêng:

```text
experiments/configs/w1/provenance.jsonl
```

Mỗi dòng keyed by `sample_id`, tối thiểu:

```json
{
  "sample_id": "w1_001",
  "source_type": "mc_ocr|synthetic|real",
  "source_dataset": "MC-OCR 2021",
  "source_reference": "dataset row/image id hoặc reference đã rút gọn",
  "license_or_permission": "dataset license / synthetic / explicit permission / internal approved",
  "pii_status": "none|redacted|contains_required_business_data_only",
  "collected_at": null,
  "notes": null
}
```

Quy tắc:

- MC-OCR: ghi dataset/source row/image ID và license/source reference ở mức phù hợp.
- Synthetic: ghi rõ `source_type=synthetic` và generator/process.
- Hóa đơn thật: chỉ được dùng nếu có quyền sử dụng phù hợp; dữ liệu cá nhân không cần thiết SHALL được redact trước khi freeze.
- Không lưu secret, account identifier hoặc thông tin cá nhân không cần cho workload.
- `manifest.csv` giữ metadata benchmark; `provenance.jsonl` giữ nguồn/quyền sử dụng. Hai file join bằng `sample_id`.
- Missing/unknown permission đối với external real invoice SHALL làm sample không đủ điều kiện freeze.

### Quyết định R3 — Source-of-truth của invoice contract

**JSON Schema `invoice_extraction.v1` là source-of-truth chính thức cho M1-03.**

Lý do:

- contract này đã frozen từ M0-04;
- SQLite user_version=1 và fixtures đang phụ thuộc vào nó;
- dùng Pydantic làm authoritative ngay ở M1 sẽ tạo rủi ro generated schema lệch contract đang ổn định.

Pydantic SHALL là typed application mirror của JSON Schema.

Regression strategy:

- canonical valid fixtures phải pass cả JSON Schema và Pydantic;
- invalid fixture matrix phải bị reject tương đương ở hai phía;
- semantic validators như impossible calendar, duplicate line number và evidence substring vẫn được test riêng;
- nếu Pydantic muốn thay đổi public contract, SHALL sửa JSON Schema/version trước, không âm thầm diverge.

### Quyết định R4 — Cơ chế freeze và fingerprint

M1-03 dùng **SHA-256 trên canonical content**, không hash tùy tiện toàn bộ directory.

#### Prompt hash

`prompt_sha256` SHALL được tính trên model-visible system prompt sau normalization kỹ thuật duy nhất:

1. decode UTF-8;
2. normalize line endings `CRLF/CR -> LF`;
3. bảo toàn toàn bộ ký tự/content còn lại;
4. encode UTF-8;
5. SHA-256.

Không trim/collapse whitespace mang nội dung.

**Bất kỳ thay đổi nào trong model-visible prompt, kể cả một dòng comment/ghi chú nằm trong file và được gửi cho model, SHALL làm hash đổi.**

Nếu có comment chỉ phục vụ developer và không muốn nó gây drift, comment đó không nên nằm trong model-visible prompt file/package.

#### Dataset fingerprint

`dataset_sha256` SHALL được tính trên canonical sample records:

- sort tăng dần theo `sample_id`;
- fixed field set tối thiểu: `sample_id`, `img_id`, exact `raw_text`, source/provenance stable identifiers;
- serialize JSON UTF-8 với key order xác định, separators canonical, LF;
- không đưa absolute local path, timestamp build hoặc field môi trường tạm thời vào fingerprint.

#### Request-package fingerprint

`w1_package_sha256` SHALL phụ thuộc tối thiểu vào:

```text
dataset_sha256
prompt_sha256
schema_version
prompt_version
canonical generation settings
```

Nhờ đó:
- đổi data → package drift;
- đổi prompt → package drift;
- đổi generation semantics → package drift;
- đổi đường dẫn local nhưng không đổi semantic content → không drift.

### Quyết định R5 — Integration M1-02 phải có adapter chứng minh replay

M1-03 **không chỉ ghi contract**. M1-03 SHALL cung cấp một **thin W1 loader/adapter + integration smoke test** chứng minh frozen package có thể được chuyển thành request mà Benchmark Harness dùng.

Scope adapter:

- đọc một frozen W1 request;
- giữ nguyên system/user messages;
- giữ generation settings/response-format semantics;
- gắn `sample_id`/workload identity cho logging;
- MAY bật streaming transport để M1-02 đo TTFT/TPOT nếu model semantics không đổi.

Không thuộc adapter:

- concurrency scheduling;
- warm-up/repeat loops;
- summary matrix;
- lane orchestration.

Các phần đó vẫn thuộc M1-02.

Acceptance của handoff:

- serialize request trước/after adapter;
- so sánh model-visible messages + semantic generation config;
- integration smoke trên ít nhất 1 sample;
- result giữ `sample_id` để join với validation artifacts.

### Quyết định R6 — Failure artifact dùng mã lỗi chuỗi ổn định

Không dùng numeric-only `E001/E002` làm primary code vì khó đọc và dễ lệch mapping.

Mỗi per-sample failure SHALL có `error_code` thuộc enum ổn định tối thiểu:

```text
TRANSPORT_ERROR
RESPONSE_ENVELOPE_ERROR
JSON_PARSE_ERROR
SOURCE_BINDING_ERROR
SCHEMA_VALIDATION_ERROR
SEMANTIC_VALIDATION_ERROR
PERSISTENCE_ERROR
```

Preflight/run-level errors MAY dùng:

```text
ENDPOINT_UNREACHABLE
MODEL_NOT_FOUND
BACKEND_CONTRACT_UNSUPPORTED
DATASET_INVALID
PROMPT_PACKAGE_INVALID
```

Failure record tối thiểu:

```json
{
  "sample_id": "w1_001",
  "error_code": "SOURCE_BINDING_ERROR",
  "stage": "source_binding",
  "message": "source.raw_text mismatch",
  "run_id": "m1_03_...",
  "lane": "vanilla",
  "model": "...",
  "latency_ms": 1234.5
}
```

Summary SHALL group/count theo `error_code`.

### Quyết định R7 — Coverage summary là artifact tự động

Coverage không chỉ ghi tay trong worklog.

Artifact authoritative:

```text
experiments/results/m1_03_w1_dataset_summary.json
```

Tối thiểu chứa:

```json
{
  "sample_count": 30,
  "dataset_sha256": "...",
  "source_type_counts": {},
  "coverage": {
    "seller_name": {"count": 0, "rate": 0.0},
    "seller_tax_code": {"count": 0, "rate": 0.0},
    "buyer": {"count": 0, "rate": 0.0},
    "issued_date": {"count": 0, "rate": 0.0},
    "total_amount": {"count": 0, "rate": 0.0},
    "items": {"count": 0, "rate": 0.0},
    "taxes_or_vat": {"count": 0, "rate": 0.0}
  }
}
```

Giá trị thực được generator tính từ frozen dataset/audit metadata, không điền tay.

`logs/Duong/M1-03/README.md` SHALL:
- link tới summary JSON;
- ghi ngắn các điểm coverage đáng chú ý;
- không trở thành source-of-truth duy nhất.

CSV coverage MAY được sinh thêm nếu thuận tiện phân tích, nhưng JSON summary là artifact bắt buộc.


## Open Questions còn lại trước code plan/apply

Các điểm dưới đây vẫn cần được code plan hoặc preflight target environment xác nhận, nhưng không làm thay đổi các quyết định R1–R7 ở trên.

### OQ-1 — Target vLLM hỗ trợ `response_format`/seed/streaming thế nào?

M0-04 dùng `response_format={"type":"json_object"}` và seed cố định. Trước final W1 run cần xác nhận target vLLM lane hỗ trợ đồng nhất.

Nếu không hỗ trợ:
- không silent fallback;
- cập nhật OpenSpec/package semantics trước benchmark;
- rerun package identity/hash nếu generation semantics thay đổi.

### OQ-2 — Có cần metric semantic field quality ngoài schema-validity không?

Trello bắt buộc `schema_valid_rate >=95%`.

Nếu nhóm muốn claim field accuracy/F1:
- phải có gold/evaluation annotations;
- định nghĩa metric riêng;
- không đồng nhất với schema-validity;
- không dùng final W1 để tune rồi báo cùng metric mà không version lại.

### OQ-3 — Chính sách commit raw/heavy artifact cuối cùng

Đã chốt compact summaries/config phải tracked. Code plan cần xác nhận:
- raw responses/runs giữ gitignored hay artifact storage nào;
- copied images/SQLite có commit hay không;
- retention đủ để audit thesis result.


# 9. Acceptance Matrix

| ID | Tiêu chí | Điều kiện đạt |
|---|---|---|
| A1 | Kích thước W1 | >= 30 unique frozen samples |
| A2 | Dev/final discipline | M0-04 có thể là final W1 nếu đáp ứng đầy đủ freeze/coverage/provenance; nếu dùng để tune thì phải version/freeze rõ |
| A3 | Provenance | Mỗi sample truy ngược được source + exact input |
| A4 | Không leak label | MC-OCR/gold labels không nằm trong model-visible prompt |
| A5 | Dataset identity | Có fingerprint/version ổn định |
| A6 | Coverage | Có summary seller/buyer/date/total/items/tax |
| A7 | Contract | `invoice_extraction.v1` hoặc version bump được phê duyệt |
| A8 | Pydantic ↔ JSON Schema | Không drift public contract |
| A9 | Strict JSON | Success output là đúng một JSON object |
| A10 | Source binding | Pipeline-owned source fields khớp tuyệt đối |
| A11 | Semantic validation | Date/time/evidence/line-number rules được enforce |
| A12 | Safe canonicalization | Không tạo business facts |
| A13 | Raw audit | Raw response bất biến, canonical output tách riêng |
| A14 | Failure taxonomy | Phân loại được transport/JSON/source/schema/semantic/persistence |
| A15 | Prompt identity | Có prompt version + hash |
| A16 | Determinism | `temperature=0`, seed/policy cố định giữa lane |
| A17 | Lane fairness | Cùng sample = cùng semantic request |
| A18 | Final backend | Acceptance run qua target vLLM/OpenAI-compatible path |
| A19 | Schema-valid gate | `schema_valid_rate >= 0.95` |
| A20 | Denominator discipline | Failure vẫn nằm trong attempted samples |
| A21 | Quality wording | Không gọi schema validity là semantic accuracy |
| A22 | Harness handoff | M1-02 dùng frozen W1 không rebuild task semantics |
| A23 | SQLite independence | Validation chạy được khi không persistence |
| A24 | Tests | Các case chính + edge cases + failure modes có regression |
| A25 | Worklog/report | Có `logs/Duong/M1-03/` và tài liệu nghiệm thu cần thiết |
| A26 | No Git mutation ở bước spec | Không commit/branch/PR trước khi duyệt plan |

---

| A27 | Git branch | Dùng `feat/m1-03-mc-ocr-node1-extractor`, target `main` |
| A28 | Base dependency | Branch chứa đầy đủ M0-04 baseline; không branch từ `main` cũ khi PR #2 chưa merge |
| A29 | File scope | Có reusable `src/pipeline/extractor.py`; không kéo concurrency scheduler vào M1-03 |
| A30 | Canonical W1 root | Handoff package nằm dưới `experiments/configs/w1/` hoặc path được spec-update chính thức |
| A31 | Harness interface | M1-02 replay full Node 1 semantics, không chỉ user-only prompt |
| A32 | Exact 30-sample gate | Nếu N=30, ít nhất 29/30 fully valid |
| A33 | Regression commands | Unit + schema + invoice contract + compile checks pass |
| A34 | PR hygiene | `git diff --check origin/main...HEAD` pass và không có unrelated/generated-heavy changes |


| A35 | Dataset extension policy | Chỉ mở rộng khi coverage/provenance/diversity cần; không tăng count vô nghĩa |
| A36 | Provenance artifact | `experiments/configs/w1/provenance.jsonl` hợp lệ và join được bằng `sample_id` |
| A37 | Contract source-of-truth | JSON Schema v1 authoritative; Pydantic mirror pass equivalence regression |
| A38 | Freeze hashing | `prompt_sha256`, `dataset_sha256`, `w1_package_sha256` tính theo canonical rules |
| A39 | M1-02 adapter | Thin loader/adapter + integration smoke chứng minh full-request replay |
| A40 | Failure codes | Stable textual `error_code` enum và summary group-by code |
| A41 | Coverage artifact | `m1_03_w1_dataset_summary.json` sinh tự động và worklog tham chiếu |


| A42 | M0-04 artifact reuse | W1 lấy từ tracked `samples.jsonl`/manifest nếu baseline hợp lệ; không regenerate vô cớ |
| A43 | Upstream pinning | Worklog ghi commit M0-04 và commit M1-01/M1-02 thực sự dùng |
| A44 | Runtime dependency | Final M1-03 correctness run dùng target vLLM runtime từ M1-01 |
| A45 | Correctness lane | Vanilla lane là canonical final schema-validity lane; static performance thuộc M1-02 |
| A46 | Transitive dependency | Không kéo M0-03 benchmark ownership trực tiếp vào M1-03 |
| A47 | Downstream handoff | M1-04 có thể audit W1 + schema-valid result mà không rebuild thủ công |


# 10. Deliverables cho code plan sau này

Đây **không phải `tasks.md`**, chỉ là danh sách đầu ra bắt buộc để plan không bỏ sót:

- W1 final frozen dataset >= 30.
- Manifest + coverage summary + dataset fingerprint.
- Frozen Node 1 request package.
- Versioned system prompt + prompt hash.
- `invoice_extraction.v1` contract hoặc explicit reviewed version bump.
- Pydantic validation model.
- Drift/equivalence tests.
- Reusable Node 1 extractor layer.
- OpenAI-compatible/vLLM integration.
- Source-binding validator.
- Semantic validator.
- Deterministic canonicalization + audit log.
- Unit/regression/integration tests.
- Raw response artifacts.
- Validated output artifacts.
- Failure artifacts.
- Aggregate M1-03 report.
- `schema_valid_rate` calculation.
- M1-02 handoff/config.
- SQLite persistence integration regression.
- `logs/Duong/M1-03/` worklog.
- Report/chapter update theo Trello.

---

- Branch `feat/m1-03-mc-ocr-node1-extractor` với base strategy đúng dependency M0-04.
- Canonical W1 package dưới `experiments/configs/w1/`.
- Harness interface contract cho full system+user Node 1 request.
- Evidence file/worklog ghi các command checks và kết quả final vLLM.
- Upstream dependency record ghi exact M0-04/M1-01/M1-02 commits đã tiêu thụ.
- Mapping M0-04 artifacts → W1 final artifacts để chứng minh không rebuild/relabel tùy ý.


# 11. Ràng buộc cho code plan

- Không hạ ngưỡng 30+ về 30+ nếu chưa thống nhất lại Trello.
- Không đưa MC-OCR label vào prompt.
- Không đổi prompt giữa Vanilla/Static.
- Không dùng `temperature > 0` cho final W1 baseline nếu methodology chưa đổi.
- Không loại failed sample khỏi denominator.
- Không dùng canonicalization để bịa business field.
- Không coi Ollama PoC là final vLLM benchmark.
- Không gộp schema-validity với field accuracy.
- Không nhúng concurrency sweep vào Node 1.
- Không đưa invoice-specific logic vào speculative-decoding engine.
- Không sửa `invoice_extraction.v1` breaking semantics mà không bump/review version.
- Không bắt đầu code/apply trước khi Open Questions có ảnh hưởng kiến trúc được chốt.
- Không động Git cho tới khi bạn yêu cầu.
- Khi bắt đầu Git, SHALL dùng branch `feat/m1-03-mc-ocr-node1-extractor`.
- Không branch M1-03 từ `main` cũ nếu M0-04 PR #2 chưa merge.
- Không coi benchmark harness user-only request hiện tại là tương đương Node 1 nếu chưa có adapter/full-request support.
- Không nghiệm thu với 28/30; với N=30 cần tối thiểu 29 valid sample.
- Không tải/chọn lại MC-OCR từ đầu nếu tracked 30-sample M0-04 baseline đã đáp ứng W1.
- Không coi M1-02 là upstream blocker cho core extractor; M1-02 là consumer/integration peer.
- Không đánh dấu final M1-03 pass khi chưa chạy frozen W1 trên target vLLM runtime từ M1-01.



---

# 12. Trạng thái đặc tả

File này là **OpenSpec duy nhất của M1-03** và dự kiến được đặt tại:

```text
logs/Duong/M1-03/openspec.md
```

Trạng thái hiện tại:

```text
[x] Phạm vi và nguồn sự thật
[x] Requirements / Scenarios
[x] Design decisions
[x] Acceptance Matrix
[x] Các quyết định R1–R7 đã chốt
[ ] Code plan — chưa viết
[ ] Branch `feat/m1-03-mc-ocr-node1-extractor` — chưa tạo
[ ] Implementation — chưa bắt đầu
```

Không cần tách file này thành cấu trúc OpenSpec khác.


---

# 13. Tài liệu tham chiếu

## Trong repository

- `docs/03-quyet-dinh-thiet-ke.md`
- `docs/05-thiet-ke-he-thong.md`
- `docs/06-ke-hoach-thuc-nghiem.md`
- `src/pipeline/README.md` trên branch M0-04
- `src/pipeline/prompts/node1_extractor_system.md`
- `src/pipeline/schemas/invoice_extraction.schema.json`
- `src/pipeline/sqlite/schema.sql`
- `src/pipeline/openai_compatible.py`
- `scripts/prepare_mcocr_devset.py`
- `scripts/build_node1_requests.py`
- `scripts/run_node1_extractor.py`
- `scripts/init_invoice_sqlite.py`
- `scripts/check_invoice_contract.py`
- `tests/test_run_node1_extractor.py`
- `experiments/results/m0_04_mcocr_devset_summary.json`
- `experiments/results/m0_04_node1_ollama_summary.json`
- Trello card `[M1-03] Xây dựng Node 1 (VAT JSON Extractor) Pipeline MC-OCR & Chuẩn bị Dataset Workload W1`


---

## 14. Kết luận

Bản OpenSpec này coi M0-04 là **contract baseline**, nhưng bổ sung những thứ cần thiết để M1-03 thực sự trở thành một workload benchmark đáng tin cậy:

- dataset final >= 30 và được freeze;
- tách dev set/final set;
- dataset + prompt fingerprint;
- Pydantic ↔ JSON Schema consistency;
- strict source binding;
- anti-hallucination;
- deterministic canonicalization có giới hạn;
- failure taxonomy;
- cùng request semantics giữa các lane;
- final acceptance trên vLLM;
- `schema_valid_rate >= 95%` với denominator không bị làm đẹp;
- không nhầm structural validity với semantic extraction quality.

Bước kế tiếp sau khi duyệt tài liệu này là **viết code plan**, chưa phải sửa code.
