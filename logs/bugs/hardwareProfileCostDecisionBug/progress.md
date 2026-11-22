# Nhật ký tiến độ triển khai: Kết nối ma trận chi phí phần cứng vào bộ chọn ngân sách Hardware-Adaptive (R6)

- **Nhánh thực thi**: `fix/r6-hardware-cost-driven-budget` (tách từ `develop@58f21ee`, giữ nguyên nhánh audit `audit/flexsds-v023-hardwareadaptive-20261120`)
- **Ngày hoàn thành pha CPU/unit**: 2026-11-22
- **Đặc tả tham chiếu**: `logs/bugs/hardwareProfileCostDecisionBug/2026-11-22-hardware-adaptive-cost-driven-budget-spec.md` (`SPEC-HWCOST-20261122`)
- **Kế hoạch tham chiếu**: `logs/bugs/hardwareProfileCostDecisionBug/plan.md`
- **Trạng thái**: Hoàn thành toàn bộ 4 tác vụ trên CPU (sẵn sàng cho pha kiểm chứng GPU & benchmark ON/OFF công bằng)

---

## 1. Danh sách commit trên nhánh `fix/r6-hardware-cost-driven-budget`

| Thứ tự | Commit SHA | Nội dung |
|---:|---|---|
| 1 | `e85bd7a` | `fix(modal): port adaptive packaging and native interpreter fixes from v0.2.3 audit` (gộp chọn lọc `modal_app.py`, `modal_auth.py`, `pyproject.toml`, `test_modal_native_interpreter.py`, `test_modal_packaging_contract.py` từ `1eb7e59`, giữ nguyên `version = "0.3.4"`) |
| 2 | `05c3f4b` | `docs(spec): add R6 hardware cost-driven budget controller spec and plan` |
| 3 | `6e56027` | `feat(cost_model): add non-monotonicity diagnostic helper for measured profiles` (Tác vụ 1 — FR-04) |
| 4 | `392bfd0` | `feat(controller): drive AdaptiveDraftController K selection from EmpiricalCostProfile` (Tác vụ 2 — FR-01, FR-02, FR-03 core, AC-01 → AC-04, Review Focus #1–#5) |
| 5 | `01fdd32` | `feat(planner): emit hardware cost decision telemetry and warn on non-monotonic profiles` (Tác vụ 3 — FR-03 telemetry, FR-04 warning, AC-05) |

---

## 2. Kết quả kiểm thử và xác minh toàn vẹn dữ liệu

### 2.1. Kiểm tra toàn vẹn profile A10G (NFR-01)
- `git diff -- src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`: **rỗng** (không sửa đổi dữ liệu đo gốc).
- `validate_measured_profile(...)` trên `profile_sm_86_remote_a10g_fcda5486b12d.json`: **`A10G_PROFILE_VALID_OK`** (khớp `raw_sha256` và `evidence.groups`).

### 2.2. Kết quả pytest trên nhánh mới
- **Bộ test mới (`tests/test_hardware_cost_driven_controller.py`):** **11 passed / 0 failed** (phủ toàn bộ AC-01 → AC-05 và 5 ca biên Review Focus: không đơn điệu ở `alpha = 0`, chi phí `NaN`/không dương, kẹp `probe_k` vào `candidate_budgets`, tie-break ưu tiên `k` nhỏ hơn, và batch trung gian `B = 3`).
- **Bộ test mở rộng (controller, planner, memory pressure, adapter, tree adaptive, cost model, lineage, local runner, m4 readiness, modal native interpreter):**
  - Lệnh: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py tests/test_budget_controller.py tests/test_budget_planner.py tests/test_memory_pressure_rule.py tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_modal_native_interpreter.py tests/test_local_runner.py tests/test_m4_readiness.py tests/test_cost_model*.py tests/test_lineage*.py tests/test_adaptive*.py -q`
  - Kết quả: **166 passed, 88 skipped, 0 failed**.
- **Ghi chú giới hạn môi trường CPU cục bộ (không phải lỗi hồi quy):**
  - `tests/test_vram_memory_lifecycle.py`: không thu thập được trong Python hệ thống do chưa cài `torch`.
  - `tests/test_modal_packaging_contract.py`: cần gói `modal` SDK cài trong cùng Python environment để import `modal`.
  - `tests/test_serving_profile_guard.py`: 2 case gọi mạng tới HuggingFace (`test_eagle_adaptive_still_requires_empirical_profile`) fail do sandbox không có kết nối mạng (cùng giới hạn môi trường đã ghi nhận ở mục 17 báo cáo audit v0.2.3); các case còn lại trong file đều pass.

---

## 3. Việc tiếp theo để đóng chứng cứ GPU cho báo cáo đồ án

1. **Bổ sung chế độ Static EAGLE matched-config** (cùng `VLLM_USE_V2_MODEL_RUNNER=0`, `--enforce-eager`, `--compilation-config {"mode": 0}` với Adaptive ON) để so sánh thuật toán công bằng không bị nhiễu bởi CUDA Graphs / `torch.compile`.
2. **Chạy 1 đợt kiểm chứng GPU trên Modal A10G (từ nhánh này):**
   - Đóng **R6 trên GPU thật**: xác nhận log `serving_steps.jsonl` có `decision_policy == "cost_model"`, `hardware_profile.device == "NVIDIA A10"`, `cost_k_ms` khớp profile A10G, và `K` thực sự co giãn theo `batch_size` + `alpha_eff`.
   - Đóng các khoảng trống GPU còn lại của **R3/R4/R5/R7**: `K=4`, fallback + thoát fallback (recovery), `effective batch > 1` với mixed-K, và chu trình memory/KV guard → cooldown → recovery.
3. **Chạy kiểm chứng lossless sạch + benchmark batch sweep (`B = 1, 4, 8, 16, 32`):**
   - Chạy `Vanilla (x2)` vs `Static EAGLE (matched)` vs `Adaptive EAGLE (R6 đã sửa)` không gắn forensic observer gây nhiễu để chốt đồng thời kết luận correctness và hiệu năng cuối cùng cho báo cáo.
