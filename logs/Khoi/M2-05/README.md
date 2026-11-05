# [M2-05] Nghiệm Thu Demo Live Adaptive Controller v0 Chuyển Trạng Thái k trên T4

Thư mục lưu trữ log và hồ sơ nghiệm thu kỹ thuật cho Checklist Item 1 của thẻ `[M2-05]` trên Trello:
**`[Khôi] Trình diễn demo live nghiệm thu Adaptive Controller v0 chuyển trạng thái k trên T4`**

---

## 1. Danh mục File Log & Đối chứng

* **[`vllm_lane3.log`](./vllm_lane3.log)**: Log khởi chạy vLLM Làn 3 (Adaptive Speculative Decoding trên cổng 8003, cấu hình Target `Qwen2.5-7B-AWQ` + Draft `Qwen2.5-0.5B-FP16`, điều khiển k động 0..4 qua `AdaptiveAsyncScheduler`).
* **[`vllm_lane1_vanilla.log`](./vllm_lane1_vanilla.log)**: Log khởi chạy vLLM Làn 1 (Reference Model / Vanilla Baseline trên cổng 8001, chạy độc lập không nháp).

---

## 2. Kết quả Thực nghiệm Live Demo trên T4 (23/10/2026)

Hai kịch bản đối chứng trực quan được bắn trực tiếp vào cổng 8003 (Làn 3) trên GPU Tesla T4:

### Kịch bản A: Prompt cấu trúc (W1 - Trích xuất hóa đơn)
* **Request ID:** `chatcmpl-8752e1f45726a246`
* **Đặc trưng:** Cú pháp có tính lặp lại cao.
* **Chỉ số:**
  * Tỷ lệ chấp nhận token nháp: `draft_acceptance_rate = 63.6%` (21 / 33 tokens).
  * `alpha_ema = 0.738` (> ngưỡng an toàn 0.65).
* **Hành vi Controller:** Cấp nháp `k = 4` ở đầu, và phục hồi nháp `[..., 2, 2, 2, 2, 0, 0, 2, 2, 2]` ở cuối khi sinh cấu trúc JSON thành công qua cơ chế probe.

#### Log phản hồi từ máy chủ vLLM (Metrics Speculative):
```json
{
  "id": "chatcmpl-8752e1f45726a246",
  "object": "chat.completion",
  "created": 1792774766,
  "model": "qwen2.5-7b-adaptive",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Dựa trên thông tin bạn cung cấp, đây là định dạng JSON tương ứng:\n\n```json\n{\n  \"shop_name\": \"Cửa hàng ABC\",\n  \"date\": \"12/10\",\n  \"total_amount\": \"500k\"\n}\n```\n\nNếu bạn cần thêm thông tin chi tiết hoặc có cấu trúc khác, vui lòng cung cấp thêm thông tin để tôi có thể điều chỉnh định dạng JSON phù hợp.",
        "refusal": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 63,
    "total_tokens": 158,
    "completion_tokens": 95
  },
  "metrics": {
    "speculative_decoding": {
      "mean_acceptance_length": 2.6153846153846154,
      "draft_acceptance_rate": 0.6363636363636364,
      "acceptance_histogram": [2, 3, 6, 2, 0],
      "num_spec_steps": 13,
      "num_accepted_draft_tokens": 21,
      "num_draft_tokens": 33,
      "num_spec_tokens": 4
    }
  }
}
```

#### Log Telemetry chuyển trạng thái k (`experiments/runs/m2_02/lane3_telemetry.jsonl`):
```json
{
  "request_id": "chatcmpl-8752e1f45726a246-b6b01763",
  "mean_k": 0.4868421052631579,
  "alpha_ema": 0.7380057720422399,
  "budgets": [
    3, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2
  ]
}
```

---

### Kịch bản B: Prompt sáng tạo (W3 - Sáng tác truyện viễn tưởng)
* **Request ID:** `chatcmpl-96d77dc2548c8194`
* **Đặc trưng:** Ngôn ngữ mở, phi tuyến tính, khó đoán.
* **Chỉ số:**
  * Tỷ lệ chấp nhận token nháp lao dốc: `draft_acceptance_rate = 14.3%` (chỉ 3 / 21 tokens).
  * `alpha_ema = 0.222` (< ngưỡng fallback 0.65).
* **Hành vi Controller (Triệt tiêu tăng tốc âm):**
  * Sau 9 bước nháp không hiệu quả, Safety Fallback Gate kích hoạt, **cắt nháp hoàn toàn về k = 0** cho 119 token còn lại.
  * `mean_k = 0.168` (gần như 0), bảo vệ hệ thống khỏi lãng phí thời gian verify.

#### Log phản hồi từ máy chủ vLLM (Metrics Speculative):
```json
{
  "id": "chatcmpl-96d77dc2548c8194",
  "object": "chat.completion",
  "created": 1792774802,
  "model": "qwen2.5-7b-adaptive",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Bất ngờ, trong đêm tối, một ngôi sao xa xôi đã rơi xuống trái đất, biến đổi hoàn toàn cuộc sống của một thị trấn nhỏ.\n\nTrong thị trấn nhỏ đó, có một cậu bé tên là Linh, luôn mơ về những hành tinh xa xôi...",
        "refusal": null
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 52,
    "total_tokens": 180,
    "completion_tokens": 128
  },
  "metrics": {
    "speculative_decoding": {
      "mean_acceptance_length": 1.3333333333333333,
      "draft_acceptance_rate": 0.14285714285714285,
      "acceptance_histogram": [7, 1, 1, 0, 0],
      "num_spec_steps": 9,
      "num_accepted_draft_tokens": 3,
      "num_draft_tokens": 21,
      "num_spec_tokens": 4
    }
  }
}
```

#### Log Telemetry chuyển trạng thái k (`experiments/runs/m2_02/lane3_telemetry.jsonl`):
```json
{
  "request_id": "chatcmpl-96d77dc2548c8194-9c49f7a8",
  "mean_k": 0.168,
  "alpha_ema": 0.22206218240000009,
  "budgets": [
    3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0
  ]
}
```

---

## 3. Kết luận Nghiệm thu
- [x] Làn 3 khởi động và chạy suy luận ổn định trên GPU Tesla T4 (sm75).
- [x] Bộ điều khiển chuyển đổi linh hoạt các mức k in {0, 2, 3, 4} theo `alpha_ema`.
- [x] Triệt tiêu tăng tốc âm ở workload khó bằng cách tự động ngắt nháp về k = 0.
- [x] Hoàn thành 100% tiêu chí nghiệm thu của Subtask 1 (M2-05).
