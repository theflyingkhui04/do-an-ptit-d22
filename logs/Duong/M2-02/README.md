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

Kết quả 12/10/2026: **177 passed**.

## Chạy smoke test trên Colab / Kaggle T4 (bước còn lại của checklist 4)

```bash
# 0. Môi trường: repo ở nhánh feat/m2-02-memory-pressure-rule; vLLM PHẢI là 0.30.0 như M1
#    (requirements-server.txt không ghim version, scheduler dựa vào API nội bộ của 0.30.0)
pip show vllm | grep Version
pip install "vllm==0.30.0"               # chỉ khi dòng trên khác 0.30.0

# 0b. Preflight (~5 s, trước khi nạp model): import scheduler với vLLM thật
python -c "import src.server.adaptive_scheduler as m; print('OK', m.AdaptiveAsyncScheduler.__mro__[1])"
#     Kỳ vọng: OK <class 'vllm.v1.core.sched.async_scheduler.AsyncScheduler'>

# 1. Tham chiếu Vanilla (Làn 1)
bash scripts/switch_lane.sh lane1
python scripts/smoke_lane3_adaptive.py reference --port 8001 \
    --model qwen2.5-7b-vanilla --out experiments/runs/m2_02/smoke_vanilla_outputs.json

# 1b. Tham chiếu Static (Làn 2) — để phân biệt sai số số học với lỗi code
bash scripts/switch_lane.sh lane2
python scripts/smoke_lane3_adaptive.py reference --port 8002 \
    --model qwen2.5-7b-static-draft --out experiments/runs/m2_02/smoke_static_outputs.json

# 2. Làn 3 Adaptive + so sánh (lần đầu compile có thể lâu, script chờ tối đa 600 s)
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
| `outputs_match_vanilla: false` | Chạy đoạn so sánh bên dưới. Nếu mọi mẫu lệch của Làn 3 cũng lệch ở Làn 2 → sai số fp16 của speculative decoding, **không phải lỗi controller**; ghi vào worklog, lossless đầy đủ thuộc M2-03 |
| `switch_lane.sh` báo hết thời gian chờ nhưng log vẫn đang compile | Server vẫn chạy nền: chờ `curl -s 127.0.0.1:8003/health` trả 200 rồi chạy tiếp bước 2 |
| `gpu_free_mb_after_server_load` < 1500 → mọi k = 0 | `export ADAPTIVE_ENABLE_VRAM_GUARD=0` rồi chạy lại bước 2 |
| Lỗi runner khi K = 0 | `export ADAPTIVE_MIN_BATCH_K=2` |
| Lỗi khi cắt nháp theo request (async) | `export ADAPTIVE_PER_REQUEST=0` |
| Không nạp được scheduler (`ModuleNotFoundError: src`) | Chạy từ gốc repo; launcher đã đặt `PYTHONPATH` |

So sánh mẫu lệch Vanilla của Làn 2 và Làn 3:

```bash
python - <<'EOF'
import json
load = lambda p: {r["sample_id"]: r.get("content") for r in json.load(open(p, encoding="utf-8"))}
van = load("experiments/runs/m2_02/smoke_vanilla_outputs.json")
sta = load("experiments/runs/m2_02/smoke_static_outputs.json")
ada = {r["sample_id"]: r.get("content") for r in json.load(open(
    "experiments/results/m2_02_lane3_smoke_summary.json", encoding="utf-8"))["requests"]}
lane2 = {s for s in van if sta.get(s) != van[s]}
lane3 = {s for s in van if ada.get(s) != van[s]}
print("lech Lan 2:", sorted(lane2)); print("lech Lan 3:", sorted(lane3))
print("Lan 3 lech rieng (can dieu tra):", sorted(lane3 - lane2))
EOF
```

### Đã kiểm chứng bằng đọc mã nguồn vLLM v0.30.0 (12/10, chưa chạy GPU)

| Điểm rủi ro | Kết luận | Căn cứ |
|---|---|---|
| `--scheduler-cls` nạp lớp của nhóm | ✅ Hỗ trợ; vLLM khuyên kế thừa `AsyncScheduler` | `config/scheduler.py:222-233`, `engine/arg_utils.py:1628` |
| Constructor được gọi bằng keyword | ✅ `__init__(*args, **kwargs)` nhận đủ | `v1/engine/core.py:162-170` |
| K = 0 trong `dynamic_sd_lookup` | ✅ Hỗ trợ chính thức ("Dynamic SD decided K=0") | `v1/spec_decode/llm_base_proposer.py:620` |
| Cắt nháp theo từng request dưới async | ✅ Runner lấy `draft_len` riêng từng request | `v1/worker/gpu_model_runner.py:1838-1858` |
| Đọc free VRAM trong scheduler | ✅ 1 GPU → executor `uni`, cùng tiến trình worker; thêm chốt `is_initialized()` | `config/parallel.py:1011` |
| **CUDA graph của Làn 3** | ⚠️ Bật dynamic SD → vLLM ép `cudagraph_mode` từ `FULL_AND_PIECEWISE` sang `PIECEWISE`. Làn 1/2 vẫn FULL → Làn 3 có thể chậm hơn do cơ chế này, **không phải do thuật toán**. Không ảnh hưởng smoke test, nhưng **phải ghi vào M2-04** khi so sánh | `config/vllm.py:972-990` |

Ghi lại lựa chọn cờ và log `logs/vllm_lane3.log` (đoạn khởi động có dòng
`AdaptiveAsyncScheduler ready`) vào worklog.
