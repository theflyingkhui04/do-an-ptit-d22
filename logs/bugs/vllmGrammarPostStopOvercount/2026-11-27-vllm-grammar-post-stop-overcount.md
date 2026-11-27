# Lỗi đếm: nháp sau EOS bị tính accepted khi có ràng buộc JSON (vLLM 0.30)

- **Mã tài liệu:** BUG-VLLM-GRAMMAR-POSTSTOP-20261127
- **Ngày:** 2026-11-27 · **Người soạn:** Dương · **Người duyệt luật evidence:** Khôi (trong PR)
- **Trạng thái:**
  - Nguyên nhân gốc đã xác minh: mã nguồn vLLM đã cài, telemetry của campaign, và chạy lại bằng xgrammar 0.2.9.
  - Đã xử lý ở bộ tổng hợp bằng luật C1–C6; nhánh `fix/m5-01-grammar-post-stop-counters`.
  - Chưa vá vLLM.
- **Môi trường:** vLLM 0.30.0 + backport PR #55708 (`vllm-0.30.0-plus-backport-55708`), xgrammar 0.2.9, NVIDIA A10, async scheduling, EAGLE-3.
- **Phát hiện ở:** campaign `flexsds-fast-a10-2611b`, ô `A10__adaptive__W1_V3__c32` bị đánh INVALID, `flexsds benchmark aggregate` từ chối cả campaign.

## 1. Triệu chứng

| Request (harness → server) | Mẫu | Histogram | Accepted | Drafted | Số bước spec |
|---|---|---|---:|---:|---:|
| `req-05a67192…` → `chatcmpl-9208c03fdee1d64f` | `m0_04_006`, r004 | `[1, 2, 1, 0, 0]` | 4 | 3 | 4 |
| `req-bc8827c1…` → `chatcmpl-8e38a0b97c790850` | `m0_04_008`, r005 | `[0, 1, 1, 0, 0]` | 3 | 2 | 2 |

Đây là 2 trong 160 request của ô, tức 2 trong 960 request W1_V3 thuộc làn speculative của campaign. W2 và W3 không có trường hợp nào.

## 2. Cơ chế

Mã nguồn đối chiếu là bản cài thật, lưu tại `experiments/runs/m4_02_04/20261110-async-lineage-design/installed/vllm/` (`installed_source_manifest.json`: `vllm 0.30.0`).

1. W1_V3 gửi `response_format: {"type": "json_object"}`, nên vLLM áp ngữ pháp JSON của xgrammar. W2 và W3 không có ràng buộc này.
2. Sau dấu `}` đóng JSON, ngữ pháp chỉ cho phép `<|im_end|>`. Giả sử EAGLE đề xuất `[<|im_end|>, x]`:
   - **Đánh dấu invalid.** `Scheduler.update_draft_token_ids_in_output` (`v1/core/sched/scheduler.py`) gọi `StructuredOutputManager.validate_tokens`. Hàm `XgrammarGrammar.validate_tokens` (`v1/structured_output/backend_xgrammar.py`) nhận `<|im_end|>`, thấy `matcher.is_terminated()` và `break`. Kết quả là `x` bị tính invalid và được đệm thành `-1`.
   - **Bộ đếm per-request trừ token invalid.** Trong `Scheduler.update_from_output`, `adj_draft_tokens = num_draft_tokens − num_invalid_spec_tokens`, cho drafted = 1.
   - **Mask được mở.** `_fill_bitmasks` (`v1/structured_output/__init__.py`) ghi `_full_mask` khi `grammar.is_terminated()`, nên vị trí của `x` không bị ràng buộc.
   - **Worker verify nháp thật.** Ở chế độ async, worker dùng nháp thật trên GPU chứ không dùng bản đã đệm `-1` (`gpu_model_runner._prepare_input_ids` scatter `self._draft_token_ids`). Mô hình đích trùng `x`, nên `num_accepted = len(generated) − 1 = 2`.
   - **Phần sau EOS bị cắt.** Bước kiểm tra stop cắt mọi token sau `<|im_end|>`, nên câu trả lời không chứa `x`.
3. Kết quả: ở lượt verify cuối, accepted vượt drafted tối đa K − 1.

## 3. Bằng chứng

- **Telemetry** (`experiments/runs/flexsds_bench/flexsds-fast-a10-2611b/servers/A10/W1_V3__adaptive/telemetry/serving_steps.jsonl`):
  - Hai request chạy K = 0 trong 1066 và 1115 bước. Chỉ 6 và 4 bước cuối có K = 2.
  - Bước bất thường là lượt verify cuối. Nó ghi `verification_feedback_skipped` (`stale-or-mismatched-lineage`) khoảng 1,3–1,5 ms trước `request_cleanup` trong `serving_lineage.jsonl`.
- **Chạy lại bằng xgrammar 0.2.9** ([xg_term.py](./xg_term.py), tokenizer `Qwen/Qwen2.5-7B-Instruct`, builtin JSON grammar):
  - văn bản trả về có 1072 token, cộng `<|im_end|>` thành 1073, khớp `output_tokens`;
  - sau `}` chỉ `<|im_end|>` hợp lệ;
  - `validate([<|im_end|>, x])` công nhận 1 token với mọi `x`;
  - sau EOS, `validate` trả `[]`.
- **Đầu ra:** hai câu trả lời là JSON hợp lệ, giống vanilla từng ký tự, cùng số token (1073, 1119).
- **Vì sao chỉ lộ ở ô này:** chênh lệch nhỏ nhất `drafted − accepted` trên 160 request của từng ô W1_V3:

  | Ô | Chênh lệch nhỏ nhất |
  |---|---:|
  | static c1 / c8 / c32 | 761 / 724 / 741 |
  | adaptive c1 / c8 | 755 / 716 |
  | adaptive c32 | −1 |

  Ở adaptive c32, request gần như chạy toàn K = 0, nên các bước trước gần như không có nháp bị từ chối để che phần dư.

## 4. Lỗi ẩn và chặn trên sai lệch α

Phần đếm dư (≤ K − 1 token mỗi request) cũng có ở những request vẫn thoả `accepted ≤ drafted`. Không tách được phần này từ bộ đếm tổng. Chặn trên sai lệch α được tính bằng `Σ_request (num_spec_tokens − 1) / Σ drafted`:

| Ô W1_V3 (campaign 2611b) | α | Chặn trên sai lệch |
|---|---:|---:|
| static c1 / c8 / c32 | 0,4139 / 0,4135 / 0,4141 | ≤ 0,11 điểm % |
| adaptive c1 / c8 | 0,3957 / 0,4143 | ≤ 0,17 điểm % |
| adaptive c32 | 0,4452 | ≤ 0,76 điểm % |

TPOT, throughput và độ trễ không bị ảnh hưởng, vì `output_tokens` là số sau khi cắt ở stop token.

## 5. Xử lý (đã làm)

Thiết kế: `openspec/changes/flexsds-bench-grammar-post-stop-counters/design.md` (cục bộ).

- **Nhận diện.** `experiments/aggregate_results.py` có `classify_spec_counter_record` và `partition_spec_counter_records`. Một bản ghi chỉ được coi là lỗi này khi thoả cả sáu điều kiện:
  - C1: bộ đếm là số nguyên không âm;
  - C2: accepted > drafted;
  - C3: có `response_format` `json_object` hoặc `json_schema`;
  - C4: phần dư ≤ `num_spec_tokens − 1`;
  - C5: `output_tokens < max_completion_tokens`;
  - C6: histogram tự nhất quán.

  Thiếu một điều kiện thì ô vẫn INVALID.
- **Ngưỡng an toàn.** Nếu một ô có hơn 5% bản ghi bị loại thì ô INVALID. Đây là đề xuất, cần Khôi xác nhận.
- **Phạm vi áp dụng.** Chỉ đường campaign `flexsds benchmark` dùng luật này (`tolerate_known_counter_artifacts=True`). M3/M5 giữ luật chặt.
- **Ghi lại.** Bản ghi bị loại chỉ bị bỏ khỏi α và τ. Danh sách nằm ở khoá `speculative_counter_exclusions` của summary và mục "Ghi chú bộ đếm speculative" trong report.

## 6. Còn mở

- **447 dòng `stale-or-mismatched-lineage`** ở server W1_V3 adaptive, mỗi request đúng một lần, trên 489 request. W2 và W3 có 0 dòng. Nhiều khả năng xảy ra ở lượt verify cuối, nhưng chưa xác minh. Chúng không ảnh hưởng α của campaign vì α lấy từ bộ đếm vLLM; chúng chỉ làm R6 bỏ qua một lần feedback ngay trước khi request kết thúc.
- **Vá vLLM tận gốc** (không tính token sau stop là accepted) để dành nếu cần đo lại. Khi đó phải đổi cổng SHA của `experiments/vllm_runtime_patch.py`.
