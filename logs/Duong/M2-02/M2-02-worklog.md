# [M2-02] Worklog Dương — Memory Pressure Rule & tích hợp Làn 3

> **Task:** [M2-02] Hiện thực hóa Bộ điều khiển Cây nháp Thích ứng (Adaptive Controller v0)<br>
> **Vai trò:** Dương — Partner 40% (Lead: Khôi 60%)<br>
> **Nhánh:** `feat/m2-02-memory-pressure-rule` (tách từ `origin/develop` @ `eb4db8c`)<br>
> **Ngày cập nhật:** 11/10/2026<br>
> **Trạng thái:** Đã code + test xong Memory Pressure Rule, **chưa commit**. Thiết kế đang **chờ duyệt** (xem mục 1).

---

## 0. Phạm vi của Dương theo Trello

Checklist thẻ M2-02 (Trello `xPn54Jvn`):

| Checklist | Hạn | Trạng thái |
|---|---|---|
| [Dương] Cài đặt Memory Pressure Rule giám sát KV-cache và VRAM Peak tự ngắt k = 0 | 09/10 12:00 | Code + test xong trên nhánh, chưa commit/PR |
| [Dương] Đóng gói script `launch_lane3_adaptive.sh` mở port 8003 và telemetry hook | 09/10 17:00 | **Blocked** — chờ chốt cách tích hợp controller vào vLLM (Câu 1) |

---

## 1. Ghi nhận quy trình

Theo quy trình dự án, thiết kế phải được duyệt trước khi code. Ở lần làm này, thiết kế
tóm tắt đã được đề xuất nhưng **chỉ việc tạo nhánh được duyệt**; code được viết
trước khi thiết kế được duyệt chính thức. Biện pháp khắc phục:

- Toàn bộ thiết kế được ghi lại ở mục 3 để duyệt hồi tố.
- Chưa commit/push — nếu thiết kế bị yêu cầu chỉnh, sửa trực tiếp trên nhánh.
- Các ngưỡng (90 / 80 / 3 bước / 1500 MB) lấy từ câu trả lời của Khôi, là
  **đề xuất cần Khôi xác nhận lại** khi review PR.

---

## 2. Bốn câu hỏi chốt với Khôi và kết quả xác minh

Khôi đã trả lời cả 4 câu. Câu 2 và 4 dùng nguyên văn. Câu 1 và 3 được đối chiếu
lại với mã nguồn **vLLM v0.30.0** (phiên bản thực tế trong log M1:
`experiments/runs/m1_02/m1-02-f307e33-20261004T102725Z/*.server.log`, engine V1
`EngineCore`) và phát hiện điểm sai.

### Câu 1 — Tích hợp controller vào vLLM ở đâu?

- **Khôi đề xuất:** hook vào `vllm/spec_decode/proposer_worker.py` /
  `multi_step_worker.py`, launcher gọi `python -m src.server.launch_adaptive_server`.
- **Xác minh:** các file trên thuộc engine **V0**, không còn trong v0.30.0 (chỉ có
  `vllm/v1/spec_decode/`). `src/server/` chưa tồn tại trong repo. Cờ
  `--speculative-model` / `--num-speculative-tokens` là cú pháp cũ; Làn 2 dùng
  `--speculative-config` JSON.
- **Cơ chế có sẵn trong v0.30.0 tìm được:**
  1. `custom_class` proposer (`vllm/v1/spec_decode/custom_class_proposer.py`):
     **không nạp draft model** (`draft_model_config = target`), `propose()` không
     nhận `request_id` → phải tự viết lại phần chạy 0.5B. Không khuyến nghị.
  2. Dynamic SD theo batch size: `num_speculative_tokens_per_batch_size` trong
     `--speculative-config`; scheduler tra `dynamic_sd_lookup[batch_size]`
     (`vllm/v1/core/sched/scheduler.py:1392`) và truyền K cho draft model
     (`vllm/v1/worker/gpu_model_runner.py:5278`). Tự chuyển CUDA graph sang
     PIECEWISE.
  3. Scheduler tùy biến: `SchedulerConfig.scheduler_cls`
     (`vllm/config/scheduler.py:159`). Scheduler có đủ `request_id`, số token
     draft/accepted mỗi request (`scheduler.py:1983`) và `get_kv_cache_usage()`.
- **Đề xuất (chờ Khôi duyệt):** Làn 3 = draft model 0.5B như Làn 2
  (`num_speculative_tokens=4`) + `AdaptiveScheduler(Scheduler)` của nhóm:
  feedback acceptance trong `update_from_output`, cắt `spec_token_ids` theo k_i
  từng request trong `update_draft_token_ids`, đặt K cả batch = max(k_i) qua
  `dynamic_sd_lookup`, đọc KV-cache trực tiếp. Phương án dự phòng không cần code:
  chỉ bật dynamic SD theo batch size.
- **Chưa kiểm chứng (cần spike Colab):** tên cờ CLI `--scheduler-cls`; K=0 trong
  lookup lúc runtime; cắt `spec_token_ids` có tương thích async scheduling;
  free VRAM thực tế lúc idle.

### Câu 2 — Tín hiệu chuẩn cho Memory Pressure Rule

- **Chốt (Khôi):** điều kiện OR — KV-cache usage > 90% **hoặc** free VRAM < 1500 MB.
- **Rủi ro ghi nhận:** vLLM cấp phát trước VRAM (`gpu_memory_utilization 0.85`;
  log M1: model 6.23 GiB, KV cache 3.6 GiB). Sprint 1 ghi VRAM peak ~13.8 GB →
  free VRAM có thể **luôn** < 1500 MB kể cả lúc rảnh, khiến Làn 3 luôn k=0. Vì vậy
  điều kiện VRAM có cờ tắt `enable_vram_guard`. Cần đo free VRAM thực tế trên T4.

### Câu 3 — Metric KV-cache

- **Khôi đề xuất:** `vllm:gpu_cache_usage_factor`.
- **Xác minh:** tên này **không tồn tại** trong v0.30.0. Tên đúng là
  **`vllm:kv_cache_usage_perc`** (`vllm/v1/metrics/loggers.py:567`), giá trị
  0.0–1.0 dù tên có "perc". Tên cũ `vllm:gpu_cache_usage_perc` cũng không còn.
- **Trong process:** `Scheduler.get_kv_cache_usage()` trả fraction 0–1
  (`scheduler.py:2466`) → truyền `100 * usage` vào controller. `/metrics` chỉ cần
  cho harness/telemetry.

### Câu 4 — Hành vi khi có áp lực bộ nhớ

- **Chốt (Khôi):** ép k=0 ngay; giữ tối thiểu 3–5 bước (cooldown); chỉ phục hồi
  khi KV-cache < 80% (hysteresis); sau đó trả quyền cho Budget Rule.
- **Cụ thể hóa khi code:** chọn `memory_cooldown_steps=3` (cấu hình được). Bỏ
  phương án "giữ đến hết request" để thống nhất một quy tắc — **cần Khôi xác nhận**.

### Bằng chứng phụ: KV-cache usage đỉnh ở M1

Lấy từ dòng `GPU KV cache usage: x%` trong server log M1 (KV cache 55,536 tokens):

| Làn / workload | KV usage max (đến c=32) |
|---|---|
| static W1 | 43.1% |
| static W3 | 21.8% |
| vanilla W1 | 22.0% |
| vanilla W3 | 9.9% |
| static / vanilla W2 | ≤ 3.5% |

→ Ngưỡng 90% **chưa từng bị chạm** ở workload M1. Memory Pressure Rule là lưới an
toàn; muốn chứng minh nó kích hoạt trên GPU cần test riêng (giảm
`--gpu-memory-utilization` hoặc prompt dài).

Phát hiện thêm: log ghi **vLLM 0.30.0**, thẻ Trello Sprint 1 ghi 0.29.0 → cần
thống nhất.

---

## 3. Thiết kế Memory Pressure Rule (chờ duyệt)

Thiết kế đã được chuyển thành bộ OpenSpec theo quy ước của nhóm (giống M1-03):

- Proposal: [M2-02-proposal.md](M2-02-proposal.md)
- Specs (R1–R7, có scenario) + Design (máy trạng thái, telemetry, câu hỏi mở): [M2-02-openspec.md](M2-02-openspec.md)
- Code plan (T0–T5, DoD): [M2-02-plan.md](M2-02-plan.md)

Tóm tắt: KV > 90% hoặc free VRAM < 1500 MB → k=0; giữ tối thiểu 3 bước; chỉ phục
hồi khi KV < 80% và VRAM đủ; mọi ngưỡng cấu hình được, VRAM guard tắt được.

---

## 4. Hiện thực

| File | Thay đổi |
|---|---|
| `src/proposer/controller.py` | 4 tham số mới + validate; máy trạng thái trong `apply_memory_pressure_rule`; `reset`/`get_stats`; tham số `kv_cache_usage_pct` ở `BudgetController.select_budget` |
| `src/proposer/adapter.py` | `propose()` nhận và truyền `kv_cache_usage_pct` |
| `tests/test_memory_pressure_rule.py` | 13 test mới |

---

## 5. Kiểm thử

Môi trường: conda `aic2026-py311` (Python 3.11.15, pytest 9.1.1).

```powershell
conda activate aic2026-py311
python -m pytest tests -q
```

| Lần chạy | Kết quả |
|---|---|
| Trước khi code (TDD đỏ) | 12 failed, 1 passed (`test_kv_cache_at_exact_high_threshold_does_not_trigger` pass sẵn vì kiểm tra trường hợp không có áp lực) |
| Sau khi code — toàn bộ `tests/` | 143 passed, 1 failed (lỗi CRLF, xem dưới) |
| Sau khi sửa CRLF cục bộ | **144 passed** |

Test từng fail: `test_benchmark_workload_adapter.py::test_real_frozen_packages_if_present`
— sai SHA-256 do `core.autocrlf=true` đổi LF→CRLF khi checkout 31 file trong
`experiments/configs/`. Nội dung trong repo đúng (hash blob = manifest), **không liên
quan** thay đổi M2-02.

Cách sửa (chỉ trên máy Dương, không commit): thêm `experiments/configs/** -text` vào
`.git/info/attributes`, ghi lại các file lệch bằng đúng blob trong git
(`git cat-file blob :<file> > <file>`), rồi `git update-index --really-refresh`.
Sau đó `git status` sạch và SHA-256 W1 = `0de2387c…` khớp manifest. Thành viên khác
dùng Windows có thể gặp cùng lỗi; nếu muốn sửa cho cả nhóm cần thêm `.gitattributes`
qua PR riêng.

Các test mới bao phủ: kích hoạt KV-cache dù α cao; đúng 90% không kích hoạt;
cooldown 3 bước; chống ping-pong; hysteresis vùng 80–90; kích hoạt lại giữa
cooldown thì đếm lại; VRAM guard dùng chung cooldown; phục hồi VRAM; tắt VRAM
guard; `reset`; `get_stats`; validate tham số; adapter bỏ qua draft khi áp lực.

---

## 6. Phân loại điểm chưa rõ

**Đã xác minh**
- Tên metric `vllm:kv_cache_usage_perc` (0–1) và `Scheduler.get_kv_cache_usage()` trong vLLM v0.30.0.
- vLLM v0.30.0 có `scheduler_cls`, dynamic SD theo batch size, `custom_class` không nạp draft model.
- KV usage đỉnh ở M1 ≤ 43.1%.
- Unit test Memory Pressure Rule pass.

**Giả định cần xác nhận**
- Ngưỡng 90 / 80 / cooldown 3 / 1500 MB (Khôi đề xuất, chưa có số đo T4 cho Làn 3).
- 4 câu hỏi mở B8 đã được Dương chốt 11/10 (xem [M2-02-openspec.md](M2-02-openspec.md) mục B8); B8-1, B8-4 bám câu trả lời của Khôi; B8-2, B8-3 là chi tiết hiện thực Dương quyết. Chỉ phần bổ sung "launcher tắt VRAM guard theo số đo T4" cần Khôi đồng ý.

**Quyết định còn mở**
- Cách tích hợp Làn 3 vào vLLM (Câu 1) — Khôi (Lead) quyết.
- Giữ hay bỏ VRAM guard sau khi đo free VRAM thực tế.

**Rủi ro / Blocker**
- Launcher Làn 3 blocked bởi Câu 1 → kéo theo sweep 54 ô (M2-04) và Cổng M2 (14/10).
- `free_vram < 1500` có thể luôn đúng trên T4 → Làn 3 suy biến thành Vanilla nếu không tắt guard.
- `controller.py` là file Khôi làm Lead — cần Khôi review PR.

---

## 7. Việc tiếp theo

1. Duyệt thiết kế mục 3 → commit `feat(m2-02): add KV-cache memory pressure rule with cooldown and hysteresis`.
2. Gửi Khôi kết quả xác minh Câu 1 & 3, đề xuất spike Colab cho `AdaptiveScheduler`.
3. Sau khi chốt Câu 1: viết `launch_lane3_adaptive.sh` (port 8003) + smoke test 10 request.
4. Push + mở PR vào `develop` (xin phép từng bước).

---

## 8. Cập nhật 12/10 — gộp code Khôi và hiện thực Làn 3

- **Gộp `develop` @ `c9ee178` (Khôi):** Khôi đã sửa lỗi kẹt k=0 bằng *periodic probing*
  (`probe_interval=5`, `probe_k=2`) và tách `mean_accept_len` (τ thật) khỏi `mean_k`.
  Xung đột ở `controller.py` (tham số `__init__`, khởi tạo, `reset`, `get_stats`) — giữ
  cả hai bên. Memory Pressure Rule vẫn được xét trước Budget Rule nên không probe khi
  bộ nhớ đang căng. 147 test pass sau khi gộp.
- **Khôi giao Dương quyết toàn bộ M2-02** → thiết kế Làn 3 ghi tại openspec Part C.
- **Đọc mã nguồn vLLM v0.30.0** xác nhận các điểm móc (openspec C1). Phát hiện quan
  trọng: Làn 1/2 ở M1 chạy *async scheduling* (tự bật với `draft_model`), nên Làn 3 kế
  thừa `AsyncScheduler` để so sánh công bằng.
- **Hiện thực:** `src/server/budget_planner.py` (logic thuần, 18 test),
  `src/server/adaptive_scheduler.py` (glue, 10 test với vLLM giả),
  `scripts/launch_lane3_adaptive.sh`, `scripts/switch_lane.sh lane3`,
  `scripts/smoke_lane3_adaptive.py`. Toàn bộ `tests/`: **175 passed**.
- **Chưa kiểm chứng (cần T4):** vLLM thật nạp được `--scheduler-cls`; K=0 trong
  `dynamic_sd_lookup`; cắt placeholder theo request dưới async; free VRAM idle. Các
  điểm này được smoke test bao phủ, có cờ dự phòng không cần sửa code (README).

---

## 9. Kết quả smoke test Làn 3 trên Colab T4 (12/10, nghiệm thu checklist 4)

- **Môi trường:** Colab T4 (sm_75), vLLM 0.30.0, nhánh `feat/m2-02-memory-pressure-rule` @ `1a7fa02`.
- **Bằng chứng (commit `eb97aac`):**
  `experiments/results/m2_02_lane3_smoke_summary.json`,
  `experiments/runs/m2_02/{smoke_vanilla_outputs,smoke_static_outputs}.json`,
  `experiments/runs/m2_02/lane3_telemetry.jsonl`,
  `experiments/runs/m2_02/{vanilla,static,adaptive}.server.log`.
- **Cách chạy:** 10 request (5 W1 + 5 W3 v2, frozen), temperature 0, `max_tokens=256`,
  gửi tuần tự từng request (c=1), lần lượt trên Làn 1 → Làn 2 → Làn 3.

### 9.1. Nghiệm thu C4 — **ĐẠT (`passed: true`)**

| Tiêu chí C4 | Kết quả |
|---|---|
| Server Làn 3 khởi động với `--scheduler-cls` | ✅ log: `Using custom scheduler class src.server.adaptive_scheduler.AdaptiveAsyncScheduler` |
| 10 request hoàn tất, không crash | ✅ 10/10 |
| Telemetry 1 dòng/request | ✅ 10/10 |
| ≥ 2 giá trị k khác nhau | ✅ `[0, 2, 3, 4]` |
| Output temp=0 khớp Vanilla | ✅ 10/10 (Làn 2 lệch Vanilla 2 mẫu W3 — sai số fp16 của speculative decoding, đáng ghi cho M2-03) |
| Free VRAM sau khi nạp model | 3142 MB → **giữ** VRAM guard (quyết định B8-4: không cần tắt) |

### 9.2. Hành vi controller (từ telemetry)

| Workload | k chiếm ưu thế | mean k | τ (mean_accept_len) | α_ema cuối |
|---|---|---|---|---|
| W1 (5 mẫu) | k=4 (48/55 bước), k=3 lúc khởi động | 3.87 | 3.79 | 1.00 |
| W3 (5 mẫu) | k=0 (~60%), k=2 do probing (~33%) | 0.66–1.17 | 1.15–1.58 | 0.51–0.94 |

Controller phản ứng đúng thiết kế: W1 (entropy thấp) nháp sâu, W3 (entropy cao) ngắt
về k=0 và định kỳ thử lại k=2. Memory Pressure Rule không kích hoạt (KV usage thấp ở c=1).

### 9.3. ⚠️ Rủi ro cho Cổng M2 — Làn 3 chậm hơn trên W3

Độ trễ trung bình (c=1, 256 token, bỏ request khởi động đầu tiên):

| | Vanilla | Static | Adaptive |
|---|---|---|---|
| W1 | 7.71 s | 5.95 s (**1.29×**) | 5.79 s (**1.33×**) |
| W3 | 6.73 s | 7.82 s (0.86×) | 8.59 s (**0.78×**) |

Tiêu chí Cổng M2 "W3 Adaptive ≥ 1.00×" **chưa đạt** trong smoke test. Đây không phải
benchmark (c=1, 5 mẫu) nên chỉ là tín hiệu, cần M2-04 đo chính thức. Giả thuyết có căn cứ:

1. **K=0 không miễn phí:** vLLM vẫn chạy 1 lượt forward của draft model mỗi bước để giữ
   KV cache của draft đồng bộ (`vllm/v1/spec_decode/llm_base_proposer.py:620`), nên k=0
   vẫn chậm hơn Vanilla thật.
2. **CUDA graph bị hạ cấp:** log Làn 3 xác nhận `Overriding cudagraph_mode from
   FULL_AND_PIECEWISE to PIECEWISE` (do bật dynamic SD); Làn 1/2 dùng FULL.
3. **Probing dày:** `probe_interval=5` làm ~33% bước W3 chạy k=2 với α thấp.

Đề xuất cho M2-04 / Khôi (chưa làm): đo riêng chi phí bước k=0 so với Vanilla; thử
`probe_interval` lớn hơn (ví dụ 20); cân nhắc bảng `num_speculative_tokens_per_batch_size`
nhiều tầng để vLLM chụp CUDA graph cho nhiều K.

### 9.4. Vấn đề nhỏ phát hiện khi chạy

- **`switch_lane.sh` (sửa ở `993a9a2`):** (a) báo "vLLM đã dừng" giả vì `pgrep -f vllm`
  chạy trước khi launcher gọi `vllm serve` → nay theo dõi PID launcher; (b) không tắt
  `VLLM::EngineCore` (chữ hoa) → Làn 2 lỗi khởi động vì VRAM còn bị giữ → nay tắt và
  chờ GPU nhả bộ nhớ; (c) timeout 300 s quá ngắn khi tải model lần đầu (179 s) → mặc định 900 s.
- **Log `AdaptiveAsyncScheduler ready` không xuất hiện:** logger của module `src.*`
  không nằm trong cấu hình logging của vLLM. Không ảnh hưởng chức năng; nên sửa để dễ debug.
- **Telemetry chưa có `memory_pressure_active`:** `AdaptiveProposerAdapter.get_telemetry`
  chỉ chọn một số trường. Nên bổ sung để M2-04 thấy khi nào Memory Pressure kích hoạt.

### 9.5. Trạng thái checklist M2-02 phần Dương

| Checklist | Trạng thái |
|---|---|
| 3. Memory Pressure Rule | ✅ Code + 13 test; đã chạy trong Làn 3 không lỗi |
| 4. Launcher Làn 3 (port 8003) + telemetry hook | ✅ Smoke test T4 đạt C4 |

Còn lại: merge `develop` mới (`061e9c2`) vào nhánh, mở PR vào `develop`, Khôi review,
rồi mới tick checklist trên Trello (thao tác Trello cần xin phép).
