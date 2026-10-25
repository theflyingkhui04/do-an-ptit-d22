# [M3-02] Worklog Dương — Đo C_T và C_D trên GPU L4

> **Task:** `[M3-02] Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4`
> **Phần của Dương (Partner 30%):** đo thời gian forward Target 7B AWQ (C_T) và Draft 0.5B (C_D) theo context length.
> **Phần của Khôi (Lead 70%, không làm ở đây):** C_V(k), ma trận C(N, B), `experiments/results/hardware_profile_l4.json`, `src/proposer/cost_model.py`.
> **Ngày đo:** 25/10/2026 · Nhánh `feat/m3-02-forward-cost-profiler`

## 1. Kết quả (đã xác minh)

Đơn vị: ms cho một bước decode (median của 5 lần đo, sau 3 lần warm-up). L = độ dài prompt (token), B = số request chạy cùng lúc. **60/60 ô đo được, 0 ô bị cờ variance (> 15%).**

**C_T — Qwen2.5-7B-Instruct-AWQ (bf16, AWQ Marlin)**

| L \ B | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 256 | 19.02 | 18.96 | 19.30 | 20.03 | 20.29 | 23.17 |
| 512 | 18.91 | 19.22 | 19.63 | 19.92 | 21.39 | 25.21 |
| 1024 | 19.04 | 19.69 | 19.90 | 20.80 | 22.95 | 29.89 |
| 2048 | 19.21 | 20.00 | 20.61 | 22.44 | 27.28 | 40.25 |
| 3072 | 19.35 | 20.38 | 21.89 | 24.92 | 32.11 | 52.63 |

**C_D — Qwen2.5-0.5B-Instruct (bf16)**

| L \ B | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 256 | 4.89 | 4.96 | 4.93 | 5.18 | 5.51 | 6.00 |
| 512 | 4.93 | 5.36 | 5.19 | 5.51 | 5.53 | 6.73 |
| 1024 | 4.95 | 5.13 | 5.43 | 5.33 | 6.36 | 6.73 |
| 2048 | 5.07 | 5.35 | 5.31 | 6.29 | 6.29 | 8.91 |
| 3072 | 5.21 | 5.27 | 5.73 | 6.08 | 7.96 | 10.74 |

Evidence: `experiments/runs/m3_02/20261025-m302-l4/{forward_costs.jsonl,env.json,summary.json}`; tổng hợp `experiments/results/m3_02_forward_costs_l4.json`. Dry run 1 ô/model: `experiments/runs/m3_02/20261025-m302-dry/`.

## 2. Nhận xét

- **Ở B nhỏ, chi phí gần như không phụ thuộc L:** C_T ≈ 19 ms, C_D ≈ 5 ms từ L = 256 đến 3072 — decode bị chặn bởi việc đọc trọng số (memory-bound). Từ B ≥ 16 và L ≥ 2048 chi phí attention trên KV cache mới nổi lên (C_T 52.6 ms ở L = 3072, B = 32).
- **C_D / C_T ≈ 0.26 ở B = 1** (0.257–0.269). Một lượt forward của model nháp không rẻ: nháp k token tuần tự tốn khoảng k × 5 ms trên nền 19 ms của target.
- **Liên hệ phát hiện A1 (audit 25/10):** vLLM vẫn chạy forward model nháp khi K = 0 để giữ KV cache của drafter. Trên L4, phần này khoảng 5 ms/bước (~26% một bước target) — cỡ độ lớn khớp với Làn 3 chậm hơn Vanilla ở W3. Đây là số đo trên L4; M2-04 đo trên T4 nên chỉ so được về xu hướng.

## 3. Phương pháp

- vLLM 0.30.0 offline `LLM` trong hàm GPU L4 trên Modal (`experiments/modal_runner.py::profile_forward`), cùng cấu hình làn: bf16, `awq_marlin` (target), `max_model_len` 4096, `gpu_memory_utilization` 0.85, `max_num_seqs` 32, `generation_config vllm`, revision đã pin, `HF_HUB_OFFLINE=1`.
- `step_ms = (T(1+N) − T(1)) / N`, N = 64; T(m) là wall time một lần `generate` B prompt độ dài đúng L với `max_tokens=m`, `ignore_eos`, temperature 0. Prefill triệt tiêu trong hiệu.
- Tắt prefix caching; mỗi request có token id khác nhau, xác định theo seed (`experiments/modal_profiler.py::prompt_token_ids`).
- Dung lượng KV (target 226 944 token, draft 1 511 168 token) đủ cho mọi ô nên không ô nào bị bỏ.
- Mỗi model một container (chạy song song).

## 4. Giới hạn

- Một ô là trung bình trên context [L, L + 64] chứ không đúng L.
- Với L·B lớn, vLLM chia prefill thành chunk và có thể chạy chồng với decode của request khác ở vài bước đầu; hiệu T(1+N) − T(1) có thể thấp hơn chi phí decode thuần một chút ở các ô lớn. Chưa đo tách.
- Draft đo như một model độc lập, chưa gồm chi phí ghép với target trong spec decoding (verify C_V(k) là phần của Khôi).

## 5. Chạy lại

```powershell
modal run -m experiments.modal_runner::profile_costs --run-id <id>                       # cả lưới, 2 model
modal run -m experiments.modal_runner::profile_costs --run-id <id> --contexts 256 --batches 1 --warmup 1 --repeats 2   # thử nhanh
```

File `experiments/results/m3_02_forward_costs_l4.json` chỉ được ghi khi chạy đủ lưới mặc định với ≥ 5 lần đo.

## 6. Chi phí & kiểm thử

- Modal: khoảng **0.85 USD** cho dry run + lưới đầy đủ (billing 25/10). Tổng từ 24/10: 4.00 / 30 USD.
- Test: `tests/test_modal_profiler.py` 24 test; toàn bộ suite **948 passed**.

## 7. Bàn giao cho Khôi

- Đầu vào cho C(N, B): `cells[]` trong `experiments/results/m3_02_forward_costs_l4.json` (`model`, `context_len`, `batch_size`, `step_ms_median`, `step_ms_p90`, …). Định dạng là đề xuất của Dương — Khôi cần thì đổi.
- Gợi ý (cần Khôi xác nhận): với `draft_model`, chi phí một bước spec ≈ C_T,verify(k) + (k + 1) × C_D, và **k = 0 vẫn tốn ≈ C_T + C_D** chứ không phải C_T.
