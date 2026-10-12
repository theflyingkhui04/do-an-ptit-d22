# Proposal M2-02 — Memory Pressure Rule cho Adaptive Draft Controller

> **Trạng thái:** Đề xuất chờ duyệt (Dương → Khôi)<br>
> **Task:** [M2-02] Hiện thực hóa Bộ điều khiển Cây nháp Thích ứng (Adaptive Controller v0) — phần Partner của Dương<br>
> **Change:** `m2-02-memory-pressure-rule`<br>
> **Branch:** `feat/m2-02-memory-pressure-rule` (từ `origin/develop` @ `eb4db8c`) → PR vào `develop`<br>
> **Lead M2-02:** Cấn Đức Khôi (60%) · **Partner:** Dương Nguyễn Thế (40%)<br>
> **OpenSpec (requirements + design):** [M2-02-openspec.md](M2-02-openspec.md)<br>
> **Code plan:** [M2-02-plan.md](M2-02-plan.md)<br>
> **Ngày:** 11/10/2026

> ⚠️ **Ghi chú quy trình:** code của change này đã được viết **trước** khi proposal/design
> được duyệt (xem [M2-02-worklog.md](M2-02-worklog.md) mục 1). Proposal này được lập để
> duyệt hồi tố; code chưa commit và sẽ được chỉnh theo kết quả duyệt.

---

## 1. Problem Statement

`AdaptiveDraftController` chọn k ∈ {0, 2, 3, 4} theo acceptance rate α. Khi bộ nhớ
GPU sắp cạn, việc tiếp tục sinh nháp làm tăng tiêu thụ KV-cache và có thể gây OOM
hoặc đẩy request ra khỏi batch (preemption). Controller hiện chỉ có một kiểm tra
free VRAM đơn giản, không theo dõi KV-cache và không có cơ chế chống dao động khi
tín hiệu bộ nhớ lơ lửng quanh ngưỡng.

## 2. Motivation

- Cổng M2 yêu cầu "controller co giãn cây thành công mà không gây crash server vLLM".
- Checklist Trello M2-02: *"[Dương] Cài đặt Memory Pressure Rule giám sát KV-cache và
  VRAM Peak tự ngắt k = 0"* (hạn 09/10, đã trễ).
- Nếu k bật/tắt liên tục quanh ngưỡng (ping-pong), hiệu năng giảm và hành vi khó
  phân tích trong Chương 5.

## 3. Current State (có bằng chứng — số dòng tính tại `develop` @ `eb4db8c`)

| Thành phần | Trạng thái trên `develop` @ `eb4db8c` |
|---|---|
| `BudgetController.select_budget` | Kiểm tra `free_vram_mb < min_free_vram_mb` → k=0 — `src/proposer/controller.py` |
| `AdaptiveDraftController.apply_memory_pressure_rule(free_vram_mb, kv_cache_usage_pct)` | Hook Khôi tách sẵn cho Dương; nhận `kv_cache_usage_pct` nhưng **không dùng**; không có trạng thái — `src/proposer/controller.py:278` |
| `AdaptiveProposerAdapter.propose` | Không truyền tín hiệu KV-cache — `src/proposer/adapter.py` |
| Test | 37 test controller/adapter pass (`tests/test_budget_*.py`, `tests/test_proposer_adapter.py`); không test nào về KV-cache |
| Tích hợp vLLM | **Chưa có** — không có code nào nối controller vào vLLM |
| KV usage thực tế M1 | Đỉnh 43.1% (static W1, c≤32) — server log `experiments/runs/m1_02/m1-02-f307e33-20261004T102725Z/` |

## 4. Objectives

1. Kích hoạt k=0 khi KV-cache usage vượt ngưỡng cao **hoặc** free VRAM dưới ngưỡng.
2. Giữ k=0 tối thiểu một số bước (cooldown) và chỉ phục hồi khi KV-cache xuống dưới
   ngưỡng phục hồi (hysteresis) — loại bỏ ping-pong.
3. Toàn bộ ngưỡng cấu hình được; điều kiện VRAM tắt được.
4. Không làm hỏng test hiện có của Khôi. Hành vi chỉ thay đổi **sau khi có trigger**:
   trước đây VRAM thấp ép k=0 đúng 1 bước; nay cooldown + hysteresis áp dụng cho cả
   VRAM (thay đổi có chủ đích, cần Khôi review).

## 5. Scope / Out of Scope

**In scope**
- Logic Memory Pressure Rule trong `AdaptiveDraftController`.
- Truyền `kv_cache_usage_pct` qua `AdaptiveProposerAdapter.propose`.
- Unit test.

**Out of scope** (change riêng / người khác)
- Tích hợp controller vào vLLM, đọc `Scheduler.get_kv_cache_usage()` lúc runtime — phụ thuộc quyết định Câu 1 (Khôi).
- Launcher `scripts/launch_lane3_adaptive.sh` (port 8003).
- Smoke test GPU, đo free VRAM thực tế trên T4.
- Cập nhật Chương 3 quyển (checklist của Khôi).

## 6. Proposed Solution

Máy trạng thái 2 trạng thái `NORMAL` / `PRESSURE` trong từng controller, chạy trước
Budget Rule mỗi bước decode. Chi tiết: [M2-02-openspec.md](M2-02-openspec.md) mục Design.

Tham số đề xuất (**Đề xuất – cần xác nhận**, lấy từ trả lời của Khôi 11/10):

| Tham số | Giá trị |
|---|---|
| `kv_cache_high_pct` | 90.0 |
| `kv_cache_recover_pct` | 80.0 |
| `memory_cooldown_steps` | 3 |
| `min_free_vram_mb` | 1500.0 (có sẵn) |
| `enable_vram_guard` | True |

## 7. Alternatives & Trade-offs

| Phương án | Ưu | Nhược | Chọn? |
|---|---|---|---|
| A. Ngưỡng đơn, không trạng thái (như hiện tại) | Đơn giản | Ping-pong quanh ngưỡng | ✗ |
| B. **Hysteresis + cooldown theo bước** | Chống ping-pong, dễ test, xác định | Thêm trạng thái trong controller | ✓ |
| C. Giữ k=0 đến hết request | Đơn giản nhất | Request dài mất toàn bộ lợi ích SD sau 1 lần chạm ngưỡng | ✗ Bỏ (chốt 11/10, B8-1) |
| D. Trạng thái áp lực toàn engine (dùng chung mọi request) | Đúng bản chất KV-cache dùng chung | Cần tầng tích hợp vLLM (chưa có) | Hoãn sang change tích hợp vLLM (chốt 11/10, B8-2) |

## 8. Dependencies

- **Upstream:** M2-02 phần Khôi (`AdaptiveDraftController`, Budget Rule) — đã merge `develop` (PR #9, #10).
- **Downstream:** tầng tích hợp vLLM Làn 3 (chưa có) sẽ gọi `select_budget(..., kv_cache_usage_pct=100 * scheduler.get_kv_cache_usage())`.
- **Tham chiếu kỹ thuật:** vLLM v0.30.0 — metric `vllm:kv_cache_usage_perc` (0–1), `Scheduler.get_kv_cache_usage()`.
- **Bàn giao cho spike/change tích hợp vLLM (Câu 1):**
  1. Đo free VRAM lúc idle trên T4 với cấu hình Làn 3 → quyết định `enable_vram_guard` cho launcher (B8-4).
  2. Test bắt lỗi đơn vị: giá trị truyền vào controller MUST là `100 * get_kv_cache_usage()` (L2).
  3. Chuyển trạng thái áp lực lên guard chung engine (B8-2, L1).

## 9. Risks & Mitigations

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| free VRAM trên T4 luôn < 1500 MB do vLLM cấp phát trước → Làn 3 luôn k=0 | Cao | Cờ `enable_vram_guard`; đo free VRAM idle trên T4 trước sweep; < 1500 MB thì launcher tắt guard (B8-4) |
| Ngưỡng 90% không bao giờ chạm ở workload hiện tại → không có bằng chứng GPU rule hoạt động | Trung bình | Unit test đủ kịch bản; test GPU riêng với `gpu_memory_utilization` thấp (ngoài scope) |
| Sửa file `controller.py` do Khôi làm Lead → xung đột | Thấp | Chỉ sửa trong hook đã giao + 1 tham số tùy chọn; Khôi review PR |
| Code viết trước khi duyệt | — | Chưa commit; chỉnh theo kết quả duyệt |
| **Mâu thuẫn tài liệu ↔ code (chưa tự chọn bên):** `docs/05` §2.2 dùng free VRAM làm *trần ngân sách giảm dần* (`N_max = calculate_max_safe_budget(free_vram)`); code dùng rule *nhị phân* (k=0 hoặc nhường Budget Rule) với KV-cache là tín hiệu chính. `docs/06` F4 hứa "Adaptive hạ cây khi VRAM căng" — nhưng vLLM cấp phát trước VRAM nên VRAM đỉnh có thể như nhau ở mọi làn | Trung bình | Gửi Khôi quyết; đề xuất QĐ mới (openspec B7): KV-cache là tín hiệu chính, F4 đo KV usage thay VRAM đỉnh |

## 10. Deliverables

- `src/proposer/controller.py`, `src/proposer/adapter.py` (sửa)
- `tests/test_memory_pressure_rule.py` (mới)
- Log: `logs/Duong/M2-02/` (proposal, openspec, plan, worklog, README)

## 11. Acceptance Criteria

- [ ] Toàn bộ requirement R1–R7 trong [M2-02-openspec.md](M2-02-openspec.md) có test pass.
- [ ] 37 test controller/adapter cũ vẫn pass.
- [ ] Khôi (Lead) duyệt thiết kế và ngưỡng.
- [ ] PR vào `develop` sẵn sàng review.
