# Code Plan M2-02 — Memory Pressure Rule + Làn 3 (checklist 3 & 4 của Dương)

> **Trạng thái:** Đã duyệt (Khôi giao Dương quyết toàn bộ, 12/10). T1–T3 được viết
> trước khi duyệt và đã đối chiếu lại với spec (T4).<br>
> **Spec:** [M2-02-openspec.md](M2-02-openspec.md) · **Proposal:** [M2-02-proposal.md](M2-02-proposal.md)<br>
> **Branch:** `feat/m2-02-memory-pressure-rule` (đã fast-forward lên `c9ee178`) · **Ngày:** 12/10/2026

## Tổng quan task

| ID | Việc | Owner | Phụ thuộc | Trạng thái |
|---|---|---|---|---|
| T0 | Duyệt proposal + spec | Dương | — | ✅ 12/10 |
| T1 | Test R1–R7 (TDD đỏ) | Dương | T0 | ✅ |
| T2 | Máy trạng thái Memory Pressure trong controller | Dương | T1 | ✅ |
| T3 | Truyền `kv_cache_usage_pct` qua adapter | Dương | T2 | ✅ |
| T4 | Đối chiếu code ↔ spec; gộp probing của Khôi (`c9ee178`) | Dương | T0 | ✅ 147 → 175 test pass |
| T6 | `AdaptiveBudgetPlanner` (R8–R12) — TDD | Dương | T4 | ✅ 18 test |
| T7 | `AdaptiveAsyncScheduler` glue + test với vLLM giả | Dương | T6 | ✅ 10 test |
| T8 | `launch_lane3_adaptive.sh`, `switch_lane.sh lane3`, `smoke_lane3_adaptive.py` | Dương | T7 | ✅ viết xong, `bash -n` OK |
| T9 | **Smoke test trên T4** (nghiệm thu C4) | Dương | T8 | ⏳ cần GPU — hướng dẫn trong README |
| T5 | Commit + push + PR vào `develop` | Dương; review: Khôi | T9 | ⏳ sau T9 (xin phép từng bước) |

Điểm đồng bộ: **T9** (kết quả smoke quyết định cờ VRAM / dự phòng) và **T5** (Khôi
review PR vì `controller.py`, `adapter.py` do Khôi làm Lead).

---

## T1 — Test R1–R7
- **File:** `tests/test_memory_pressure_rule.py` (mới)
- **Bước:** mỗi scenario trong spec → 1 test; chạy để xác nhận fail trước khi code.
- **Lệnh:** `python -m pytest tests/test_memory_pressure_rule.py -q` (env conda `aic2026-py311`, Python 3.11)
- **Nghiệm thu:** test fail vì thiếu tính năng (không fail do lỗi cú pháp/import).
- **Kết quả đã có:** 12 failed, 1 passed (R1.2 pass sẵn — đúng kỳ vọng vì là kịch bản không áp lực).

## T2 — Máy trạng thái
- **File:** `src/proposer/controller.py`
- **Bước:**
  1. `AdaptiveDraftController.__init__`: thêm `kv_cache_high_pct`, `kv_cache_recover_pct`, `memory_cooldown_steps`, `enable_vram_guard` + validate (R6); khởi tạo `_memory_pressure_active`, `_memory_pressure_steps`.
  2. `apply_memory_pressure_rule`: hiện thực B3.
  3. `reset()` xóa trạng thái; `get_stats()` thêm `memory_pressure_active` (R7).
  4. `AdaptiveDraftController.select_budget` đã có sẵn tham số `kv_cache_usage_pct` — giữ nguyên.
- **Input/Output:** như Input contract trong spec.
- **Nghiệm thu:** test R1–R6 pass.

## T3 — Adapter
- **File:** `src/proposer/adapter.py`; `BudgetController.select_budget` nhận thêm `kv_cache_usage_pct` (không dùng).
- **Nghiệm thu:** R7.2 pass; 37 test cũ pass.

## T4 — Đối chiếu sau duyệt
- So code hiện có với spec đã duyệt; nếu ngưỡng/hành vi thay đổi (ví dụ cooldown 5, giữ đến hết request) → sửa test trước, rồi sửa code.
- **Nghiệm thu:** mọi scenario của spec *đã duyệt* có test tương ứng và pass.

## T5 — Verify, commit, PR
- **Lệnh:** `python -m pytest tests -q` → kỳ vọng toàn bộ pass (đã đạt 144 passed ngày 11/10 sau khi sửa CRLF cục bộ).
- `git status`, `git diff`, stage có chủ đích 3 file code/test + `logs/Duong/M2-02/`.
- **Commit đề xuất:**
  - `feat(m2-02): add KV-cache memory pressure rule with cooldown and hysteresis`
  - `docs(m2-02): add Duong worklog, proposal, spec and plan for memory pressure rule`
- PR vào `develop`, mô tả theo mẫu PR của nhóm; reviewer: Khôi.
- **Mỗi bước commit / push / PR xin phép riêng.**

---

## Definition of Done (3 điều kiện `rule_trello.md`) — áp dụng cho phần việc này

| Điều kiện | Phần Memory Pressure Rule | Ghi chú |
|---|---|---|
| 1. Code chạy được — PR vào `develop` sẵn sàng | T5 | |
| 2. Có số liệu thực nghiệm | Chỉ có kết quả unit test + số liệu KV usage M1 | Bằng chứng GPU phụ thuộc tầng tích hợp vLLM — ngoài scope, ghi rõ trong PR |
| 3. Có tài liệu | Worklog + spec trong `logs/Duong/M2-02/`; đoạn Chương 3 do Khôi phụ trách | Dương có thể đóng góp 1 đoạn mô tả Memory Pressure Rule cho Khôi |

Checklist Trello *"[Dương] Cài đặt Memory Pressure Rule…"* chỉ đề nghị tick sau khi
PR được Khôi duyệt (tick là thao tác ghi Trello — cần xin phép).
