# Kiểm tra lossless của R6 (23/11/2026)

Bằng chứng cho bảng lossless ở Chương 4 của quyển. Ba làn được so khớp từng token ID với `vanilla` ở nhiệt độ 0.

| So sánh | Số prompt khớp | W1 | W2 | W3 |
|---|---|---|---|---|
| `vanilla` lần 1 và lần 2 | 54/54 | 30/30 | 12/12 | 12/12 |
| `vanilla` và EAGLE K=3 (`static_matched_k3`) | 30/54 | 21/30 | 9/12 | 0/12 |
| `vanilla` và EAGLE + R6 (`adaptive_r6`) | 34/54 | 21/30 | 12/12 | 1/12 |

Mỗi làn có 7 câu trả lời dừng do chạm giới hạn `max_tokens = 1024` (`finish_reason` khác `stop`).

## Cấu hình
- Code `229b3e4`. Đây là R6 trước bản sửa ước lượng α (PR #53) và chọn K theo batch (PR #55).
- Runtime và mô hình:
  - vLLM 0.30.0;
  - target `Qwen/Qwen2.5-7B-Instruct-AWQ`;
  - draft `duckhoi/qwen2.5-7b-eagle3.1-speculator` (`eagle3`).
- Mọi làn chạy với cùng thiết lập thực thi (matched: eager, async scheduling) và tắt prefix caching.
- Mỗi lần gửi một request, `temperature = 0`, `seed = 42`.
- 54 request lấy từ các gói đóng băng:
  - W1: 30 request, `experiments/configs/w1/node1_chat_requests.jsonl`;
  - W2: 12 request, `experiments/configs/w2_v2/chat_requests.jsonl`;
  - W3: 12 request, `experiments/configs/w3_v2/chat_requests.jsonl`.

  Mã băm của các gói nằm trong `summary.json`.
- Phần cứng: mỗi làn chạy trong một container Modal riêng, cả ba đều là NVIDIA A10, driver 610.57.04. Không ghi được UUID của GPU.
- Đo ngoài repo bằng script `campaign.py` (giai đoạn `lossless`), sau đó chép vào đây. Các file không bị sửa; mã băm từng file nằm trong `summary.json`.

## File
- `<làn>/captures_r*.jsonl`: mỗi dòng gồm `text`, `token_ids`, `finish_reason` và `usage` của một request. Riêng `vanilla` có hai lần chạy.
- `<làn>/server_lossless.log`: log vLLM của lần đo.
- `summary.json`: số khớp theo workload, vị trí token lệch đầu tiên của từng prompt, cấu hình và mã băm.

Số khớp được tính lại bằng `experiments.lossless_verifier.verify_token_sequence` trên `token_ids`.

## Trạng thái
Đây là phép đo sau ngày đóng băng thuật toán (15/11). PM cần xác nhận trước khi quyển dùng làm số chính thức (M-06, M-10). Phần phân tích nguyên nhân các điểm lệch (hai token gần hoà điểm trong bf16) nằm ở Chương 4 của quyển.
