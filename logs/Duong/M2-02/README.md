# M2-02 - Hồ sơ phần việc của Dương (Partner 40%)

## Phạm vi

[M2-02] Hiện thực hóa Bộ điều khiển Cây nháp Thích ứng (Adaptive Controller v0).
Phần của Dương: checklist 3 (Memory Pressure Rule) và checklist 4 (launcher Làn 3 +
telemetry hook). Khôi giao Dương quyết toàn bộ thiết kế phần này (12/10).

| Tài liệu | Nội dung | Trạng thái |
|---|---|---|
| [M2-02-proposal.md](M2-02-proposal.md) | Vấn đề, hiện trạng, phương án, rủi ro, tiêu chí nghiệm thu | Đã duyệt |
| [M2-02-openspec.md](M2-02-openspec.md) | Part A/B: Memory Pressure Rule (R1–R7). Part C: Làn 3 (R8–R12, C4 nghiệm thu) | Đã duyệt |
| [M2-02-plan.md](M2-02-plan.md) | Code plan T0–T9, DoD | Cập nhật 12/10 |
| [M2-02-worklog.md](M2-02-worklog.md) | Nhật ký: 4 câu hỏi với Khôi, xác minh vLLM v0.30.0, test | Cập nhật 12/10 |

Worklog phần của Khôi (Lead): [logs/Khoi/M2-02/](../../Khoi/M2-02/).

## Trạng thái

| Checklist | Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|---|
| 3 | Memory Pressure Rule (KV-cache + VRAM, cooldown, hysteresis) | Code + test xong, đã gộp với probing của Khôi (`c9ee178`); chưa commit/PR | [controller.py](../../../src/proposer/controller.py), [test_memory_pressure_rule.py](../../../tests/test_memory_pressure_rule.py) |
| 4 | Tầng tích hợp vLLM (`AdaptiveAsyncScheduler` + `AdaptiveBudgetPlanner`) | Code + unit/glue test xong (vLLM giả); **chờ smoke test Colab** | [budget_planner.py](../../../src/server/budget_planner.py), [adaptive_scheduler.py](../../../src/server/adaptive_scheduler.py), [test_budget_planner.py](../../../tests/test_budget_planner.py), [test_adaptive_scheduler_glue.py](../../../tests/test_adaptive_scheduler_glue.py) |
| 4 | Launcher `launch_lane3_adaptive.sh` (port 8003) + `switch_lane.sh lane3` | Viết xong, qua `bash -n`; **chờ chạy trên T4** | [launch_lane3_adaptive.sh](../../../scripts/launch_lane3_adaptive.sh), [switch_lane.sh](../../../scripts/switch_lane.sh) |
| 4 | Smoke test 10 request | Script xong; **chờ chạy trên T4** | [smoke_lane3_adaptive.py](../../../scripts/smoke_lane3_adaptive.py) |

## Kiểm thử đã chạy (local, không GPU)

```powershell
conda activate aic2026-py311
python -m pytest tests -q
```

Kết quả 12/10/2026: **175 passed**.

## Chạy smoke test trên Colab / Kaggle T4 (bước còn lại của checklist 4)

```bash
# 0. Môi trường: vLLM 0.30.0 như M1; repo ở nhánh feat/m2-02-memory-pressure-rule
pip show vllm | grep Version            # phải là 0.30.0

# 1. Tham chiếu Vanilla (Làn 1)
bash scripts/switch_lane.sh lane1
python scripts/smoke_lane3_adaptive.py reference --port 8001 \
    --model qwen2.5-7b-vanilla --out experiments/runs/m2_02/smoke_vanilla_outputs.json

# 2. Làn 3 Adaptive + so sánh
bash scripts/switch_lane.sh lane3        # log: logs/vllm_lane3.log
python scripts/smoke_lane3_adaptive.py adaptive --port 8003 \
    --model qwen2.5-7b-adaptive \
    --reference experiments/runs/m2_02/smoke_vanilla_outputs.json \
    --telemetry experiments/runs/m2_02/lane3_telemetry.jsonl \
    --summary experiments/results/m2_02_lane3_smoke_summary.json
bash scripts/switch_lane.sh stop
```

Đạt khi `passed: true` (4 điều kiện mục C4 của openspec). Nếu lỗi:

| Triệu chứng | Xử lý (không sửa code) |
|---|---|
| `gpu_free_mb_after_server_load` < 1500 → mọi k = 0 | `export ADAPTIVE_ENABLE_VRAM_GUARD=0` rồi chạy lại bước 2 |
| Lỗi runner khi K = 0 | `export ADAPTIVE_MIN_BATCH_K=2` |
| Lỗi khi cắt nháp theo request (async) | `export ADAPTIVE_PER_REQUEST=0` |
| Không nạp được scheduler (`ModuleNotFoundError: src`) | Chạy từ gốc repo; launcher đã đặt `PYTHONPATH` |

Ghi lại lựa chọn cờ và log `logs/vllm_lane3.log` (đoạn khởi động có dòng
`AdaptiveAsyncScheduler ready`) vào worklog.
