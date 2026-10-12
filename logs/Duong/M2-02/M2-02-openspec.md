# OpenSpec M2-02 — Memory Pressure Rule

> **Trạng thái:** Đặc tả chờ duyệt<br>
> **Change:** `m2-02-memory-pressure-rule`<br>
> **Proposal:** [M2-02-proposal.md](M2-02-proposal.md) · **Plan:** [M2-02-plan.md](M2-02-plan.md)<br>
> **Ngày:** 11/10/2026

---

# Part A — Specs

## Capability: `adaptive-memory-pressure`

### Requirement R1: KV-cache trigger
The controller SHALL return k=0 when `kv_cache_usage_pct` is strictly greater than
`kv_cache_high_pct`, regardless of the acceptance rate.

- **Scenario R1.1** — GIVEN `initial_alpha=0.95`, `kv_cache_high_pct=90`
  WHEN `select_budget(kv_cache_usage_pct=90.5)`
  THEN `k == 0` AND reason chứa "KV-cache".
- **Scenario R1.2 (biên)** — GIVEN `initial_alpha=0.80`
  WHEN `select_budget(kv_cache_usage_pct=90.0)`
  THEN `k == 3` (không kích hoạt tại đúng ngưỡng).

### Requirement R2: VRAM guard trigger
WHEN `enable_vram_guard` is true, the controller SHALL return k=0 when
`free_vram_mb < min_free_vram_mb`. WHEN `enable_vram_guard` is false, free VRAM
MUST NOT trigger memory pressure.

- **Scenario R2.1** — GIVEN `min_free_vram_mb=1500`, KV=10
  WHEN free VRAM = 1000 THEN `k == 0` AND reason chứa "VRAM".
- **Scenario R2.2** — GIVEN `enable_vram_guard=False`
  WHEN free VRAM = 500, KV=10 THEN `k == 3`.

### Requirement R3: Cooldown
After a trigger, the controller SHALL keep k=0 for at least `memory_cooldown_steps`
consecutive `select_budget` calls, counting the trigger step.

- **Scenario R3.1** — GIVEN cooldown=3
  WHEN KV theo bước = `95, 50, 50, 50`
  THEN k = `0, 0, 0, 3`.
- **Scenario R3.2 (VRAM dùng chung cooldown)** — GIVEN cooldown=3, KV=10
  WHEN free VRAM = `1000, 16000, 16000, 16000`
  THEN k = `0, 0, 0, 3`.

### Requirement R4: Hysteresis recovery
After the cooldown, the controller SHALL release memory pressure only when
`kv_cache_usage_pct < kv_cache_recover_pct` AND (VRAM guard disabled OR
`free_vram_mb >= min_free_vram_mb`). Otherwise k MUST stay 0.

- **Scenario R4.1** — GIVEN cooldown=3, high=90, recover=80
  WHEN KV = `95, 85, 85, 85, 80, 79.9`
  THEN k = `0, 0, 0, 0, 0, 3`.
- **Scenario R4.2** — GIVEN cooldown=1, KV=10
  WHEN free VRAM = `1000, 1200, 1600`
  THEN k = `0, 0, 3`.
- **Scenario R4.3 (chống ping-pong)** — WHEN KV = `90.5, 89.9, 90.5, 89.9, 90.5, 89.9`
  THEN k = 0 ở cả 6 bước.

### Requirement R5: Re-trigger restarts cooldown
A new trigger during PRESSURE SHALL restart the cooldown count.

- **Scenario R5.1** — GIVEN cooldown=3
  WHEN KV = `95, 50, 95, 50, 50, 50` THEN k = `0, 0, 0, 0, 0, 3`.

### Requirement R6: Configuration validation
The controller MUST raise `ValueError` when `kv_cache_recover_pct > kv_cache_high_pct`
or `memory_cooldown_steps < 1`.

- **Scenario R6.1** — high=80, recover=90 → `ValueError`.
- **Scenario R6.2** — cooldown=0 → `ValueError`.

### Requirement R7: State lifecycle & interface
- `reset()` SHALL clear memory pressure state.
- `get_stats()` SHALL expose `memory_pressure_active: bool`.
- `AdaptiveProposerAdapter.propose()` SHALL accept `kv_cache_usage_pct` (default 0.0),
  forward it to the controller, and MUST NOT call `draft_generator` when k=0.
- Existing callers that omit `kv_cache_usage_pct` MUST keep their current behavior
  as long as no memory trigger occurs. After a trigger (including the VRAM guard),
  R3–R5 apply: k stays 0 for the cooldown instead of recovering on the next step.
  This is an intentional behavior change for `AdaptiveDraftController`.

- **Scenario R7.1** — trigger KV=95 → `reset()` → KV=10 → `k == 3`, `memory_pressure_active == False`.
- **Scenario R7.2** — adapter, `initial_alpha=0.95`, `kv_cache_usage_pct=92`
  → `budget_k == 0`, `skipped_draft == True`, `draft_generator` không được gọi.

### Input contract
| Tham số | Kiểu | Đơn vị / miền | Nguồn dự kiến lúc runtime |
|---|---|---|---|
| `kv_cache_usage_pct` | float | phần trăm 0–100 | `100 * Scheduler.get_kv_cache_usage()` (vLLM v0.30.0, `vllm/v1/core/sched/scheduler.py:2466`) hoặc `100 * vllm:kv_cache_usage_perc` |
| `free_vram_mb` | float | MB | NVML / `torch.cuda.mem_get_info` (tầng tích hợp quyết định) |

Giá trị ngoài miền (âm, > 100) không được chuẩn hóa ở tầng controller — tầng tích hợp
chịu trách nhiệm cung cấp đúng đơn vị (quyết định B8-3, giới hạn L2).

---

# Part B — Design

## B1. Thành phần

| Thành phần | Hiện có / Dự kiến | File |
|---|---|---|
| `BudgetController` (Khôi) | Hiện có; thêm tham số `kv_cache_usage_pct` không dùng | `src/proposer/controller.py` |
| `AdaptiveDraftController` (Khôi) | Hiện có; Dương mở rộng `apply_memory_pressure_rule` | `src/proposer/controller.py` |
| `AdaptiveProposerAdapter` (Khôi) | Hiện có; thêm tham số truyền xuống | `src/proposer/adapter.py` |
| Tầng tích hợp vLLM (đề xuất `AdaptiveScheduler`) | **Dự kiến**, ngoài scope | — |

## B2. Luồng mỗi bước decode

```text
select_budget(batch_size, free_vram_mb, kv_cache_usage_pct)
  ├─ 1. apply_memory_pressure_rule(...)  → (0, reason) | None
  │       nếu trả (0, reason) → BudgetDecision(k=0), ghi history, return
  └─ 2. apply_budget_rule(effective_alpha) (Khôi) → k ∈ {0, 2, 3, 4}
```

## B3. Máy trạng thái

```text
             KV > high  OR  (guard AND free_vram < min)
NORMAL ─────────────────────────────────────────────────► PRESSURE (k=0, steps=1)
  ▲                                                          │
  │   steps >= cooldown AND KV < recover AND VRAM ok         │ mỗi bước tiếp theo:
  └──────────────────────────────────────────────────────────┤  - vượt ngưỡng lại   → steps=1
                                                             │  - steps < cooldown  → k=0, steps+1
                                                             │  - KV ∈ [recover, high] → k=0
```

Trạng thái: `_memory_pressure_active: bool`, `_memory_pressure_steps: int` — nằm trong
**từng controller** (mỗi request một controller qua adapter).

## B4. Telemetry
- `BudgetDecision.reason` phân biệt: `KV-cache pressure`, `VRAM pressure`,
  `Memory pressure cooldown: step i/N`, `Memory pressure hold: KV-cache …`.
- `get_stats()["memory_pressure_active"]`; `history` ghi từng quyết định (có sẵn).

## B5. Error handling
- Validate tham số ở constructor (R6).
- Không ném lỗi lúc runtime; thiếu tín hiệu KV (`0.0` mặc định) = không áp lực.

## B6. Testing strategy
- Unit test thuần Python, không cần GPU: `tests/test_memory_pressure_rule.py` (R1–R7).
- Regression: chạy lại toàn bộ `tests/`.
- Kiểm chứng GPU (ngoài scope): sau khi có tầng tích hợp, chạy Làn 3 với
  `gpu_memory_utilization` thấp để buộc KV > 90%.

## B7. Quyết định thiết kế & đề xuất QĐ mới cho `docs/03`
- **Đề xuất QĐ:** "Memory Pressure Rule dùng KV-cache usage làm tín hiệu chính, VRAM
  làm phòng hộ có thể tắt; hysteresis 90/80 + cooldown 3 bước." — *chờ Khôi duyệt,
  không tự ghi vào `docs/03`.*

## B8. Quyết định đã chốt (Dương duyệt 11/10/2026)

Quan hệ với 4 câu trả lời của Khôi (11/10, xem worklog mục 2):
- **B8-1** làm rõ câu 4 của Khôi ("3–5 bước *hoặc* đến hết request") — chọn cooldown.
- **B8-2, B8-3** là chi tiết hiện thực Khôi chưa đề cập — Dương quyết, thông báo trong PR.
- **B8-4** giữ đúng điều kiện OR Khôi chốt ở câu 2; **chỉ phần "launcher có thể tắt
  VRAM guard theo số đo T4" là bổ sung, cần Khôi đồng ý.**

Tiêu chí chốt: Memory Pressure Rule là lưới an toàn (KV usage đỉnh M1 = 43.1%), nên
ưu tiên đơn giản, có test, giải thích được trong Chương 3 và **không làm Làn 3 suy
biến thành Vanilla**.

| # | Câu hỏi | Quyết định | Lý do |
|---|---|---|---|
| 1 | Giữ k=0 đến hết request? | **Bỏ.** Chỉ dùng cooldown + hysteresis | W3 sinh output dài; chạm ngưỡng một lần sẽ mất lợi ích nháp cả phần còn lại. Cooldown + hysteresis đã đủ chống ping-pong |
| 2 | Trạng thái theo request hay toàn engine? | **Theo từng request cho M2**; chuyển sang guard chung engine trong change tích hợp vLLM (`AdaptiveScheduler`) | Mọi controller nhận cùng giá trị KV mỗi bước nên kích hoạt đồng thời. Chưa có tầng tích hợp để đặt guard chung |
| 3 | Clamp `kv_cache_usage_pct` về [0, 100]? | **Không clamp, không raise** | vLLM trả 0–1 nên ngoài miền chỉ do lỗi tích hợp. Lỗi nguy hiểm là nhầm đơn vị (0.92 thay vì 92) — clamp không bắt được. Raise trong vòng decode có thể làm crash server |
| 4 | Giữ VRAM guard? | **Giữ trong code, mặc định `enable_vram_guard=True`.** Cấu hình Làn 3 quyết định theo số đo free VRAM idle trên T4: nếu < 1500 MB thì tắt và ghi lý do vào worklog | vLLM cấp phát trước ~85% VRAM; token nháp nằm trong pool KV đã cấp → free VRAM gần như không đổi theo k, có thể luôn < 1500 MB. Đổi mặc định sang False sẽ làm fail 2 test của Khôi (`test_budget_controller.py:261`, `test_budget_transition_scenarios.py:192`) |

### Câu hỏi còn mở (chờ Khôi — từ spec review 11/10)
- **Q-A (mâu thuẫn docs):** VRAM là *trần ngân sách giảm dần* như `docs/05` §2.2 hay
  *lưới an toàn nhị phân* như code hiện tại? F4 (`docs/06`) đo VRAM đỉnh hay KV usage?
- **Q-B (α khi k=0):** khi Memory Pressure ép k=0, tầng tích hợp có gọi
  `update(num_accepted=0, num_drafted=0)` không? Với code hiện tại, `num_drafted=0` vẫn
  đẩy α cũ vào cửa sổ và tăng `step_count`, làm lệch `effective_alpha` sau khi hết áp
  lực. Đề xuất: tầng tích hợp MUST NOT gọi `update` khi `num_drafted=0`, hoặc controller
  bỏ qua mẫu này.

### Known limitations
- **L1 (từ quyết định 2):** request mới vào khi KV ∈ [recover, high] bắt đầu ở NORMAL nên
  vẫn sinh nháp, trong khi các request cũ đang giữ k=0. Ảnh hưởng không đáng kể với
  workload hiện tại; được khắc phục khi chuyển guard lên mức engine.
- **L2 (từ quyết định 3):** controller không tự phát hiện nhầm đơn vị. Change tích hợp
  vLLM MUST có test kiểm tra giá trị truyền vào là `100 * get_kv_cache_usage()`.

---

# Part C — Làn 3: tích hợp controller vào vLLM (checklist 4)

> **Trạng thái:** Thiết kế đã được Dương duyệt 12/10/2026 (Khôi giao Dương quyết toàn bộ M2-02).
> Căn cứ: đọc mã nguồn vLLM **v0.30.0** (tag `ced6857`).

## C1. Căn cứ từ mã nguồn vLLM v0.30.0
| Điểm | Vị trí | Hệ quả thiết kế |
|---|---|---|
| CLI `--scheduler-cls` | `vllm/engine/arg_utils.py:1628` | Nạp lớp scheduler của nhóm, không fork vLLM |
| Async scheduling tự bật với `draft_model` | `vllm/config/vllm.py` (~1438–1487) | Làn 1/2 ở M1 chạy async → Làn 3 MUST kế thừa `AsyncScheduler` để so sánh công bằng |
| Placeholder nháp `[-1]*K` gán trong `AsyncScheduler._update_after_schedule` | `vllm/v1/core/sched/async_scheduler.py` | Cắt placeholder về `k_i` để giới hạn nháp từng request |
| K cả batch = `dynamic_sd_lookup[batch_size]` | `scheduler.py:1392` | Ghi `max(k_i)` vào bảng tra mỗi bước; cần bật `num_speculative_tokens_per_batch_size` |
| `make_spec_decoding_stats(..., num_draft_tokens, num_accepted_tokens, ..., request_id)` gọi 1 lần/request/bước có nháp | `scheduler.py:2798` | Điểm phản hồi α; không gọi khi k=0 (giải quyết Q-B) |
| `_free_request(request)` | `scheduler.py:2561` | Điểm kết thúc request → ghi telemetry |
| `get_kv_cache_usage()` trả 0–1 | `scheduler.py:2466` | Tín hiệu Memory Pressure, nhân 100 (L2) |

## C2. Thành phần
| File | Vai trò | Kiểm thử |
|---|---|---|
| `src/server/budget_planner.py` | `AdaptiveBudgetPlanner` — logic thuần, không import vLLM: `plan_step`, `record_verification`, `finish_request` (ghi JSONL), `from_env` | Unit test local |
| `src/server/adaptive_scheduler.py` | `AdaptiveAsyncScheduler(AsyncScheduler)` — glue mỏng ghi đè `schedule`, `_update_after_schedule`, `make_spec_decoding_stats`, `_free_request` | Smoke test Colab |
| `scripts/launch_lane3_adaptive.sh` | Port 8003, cấu hình như Làn 2 + `num_speculative_tokens=4`, `num_speculative_tokens_per_batch_size=[[1,32,4]]`, `--scheduler-cls` | Smoke test Colab |
| `scripts/smoke_lane3_adaptive.py` | 10 request temp=0 tới 8003 (+8001 Vanilla), kiểm tra không crash, ≥2 giá trị k, output khớp Vanilla, in free VRAM idle | Chạy trên Colab |

## C3. Requirements
- **R8** `plan_step` SHALL gọi `select_budget` đúng 1 lần cho mỗi request decode, truyền `kv_cache_usage_pct = 100 * kv_usage`, và trả `batch_k = max(k_i)` (0 nếu không có request decode), mọi `k_i <= max_k`.
- **R9** `record_verification` SHALL bỏ qua khi `num_drafted <= 0` và chuyển `(accepted, drafted)` cho adapter khi > 0.
- **R10** `finish_request` SHALL ghi đúng 1 dòng JSON vào file telemetry (nếu được cấu hình) gồm `request_id`, `lane="adaptive"`, `tree_budgets`, `mean_accept_len`, `mean_k`, và giải phóng phiên; request không tồn tại MUST NOT gây lỗi.
- **R11** `from_env` SHALL đọc `ADAPTIVE_CONTROLLER_KWARGS` (JSON), `ADAPTIVE_TELEMETRY_PATH`, `ADAPTIVE_ENABLE_VRAM_GUARD`, `ADAPTIVE_MAX_K`, `ADAPTIVE_MIN_BATCH_K`, `ADAPTIVE_PER_REQUEST`; JSON sai MUST raise `ValueError` lúc khởi động (không phải lúc decode).
- **R12** Cờ dự phòng: `ADAPTIVE_PER_REQUEST=0` → mọi request dùng `batch_k`; `ADAPTIVE_MIN_BATCH_K=m` → `batch_k >= m` khi có request decode (k_i vẫn có thể 0).

## C4. Nghiệm thu checklist 4 (smoke test Colab)
1. Server Làn 3 khởi động với `--scheduler-cls` không lỗi.
2. 10 request hoàn tất, server không crash.
3. File telemetry có ≥ 2 giá trị k khác nhau trong `tree_budgets`.
4. Output temp=0 của 10 request khớp Vanilla (8001).
5. Ghi `experiments/results/m2_02_lane3_smoke_summary.json`.
