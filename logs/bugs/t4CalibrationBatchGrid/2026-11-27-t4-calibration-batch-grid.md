# Hồ sơ chi phí T4 chỉ đo được tới B=8

- **Mã tài liệu:** BUG-T4-CALIB-BATCHGRID-20261127
- **Ngày:** 2026-11-27 · **Người soạn:** Dương
- **Trạng thái:**
  - Đã xử lý bằng cách chấp nhận hồ sơ ngắn (PR #59): batch lớn hơn 8 dùng chi phí của B=8.
  - Đã thêm cờ `profile --gpu-memory-utilization` (PR #60).
  - Đo B=16/32 trên T4 **chưa làm được**. Dương quyết định 27/11: giữ hồ sơ B≤8 và không thử thêm lúc này.
- **Hồ sơ đang dùng:** `~/.flexsds/profiles/profile_sm_75_remote_t4_fcda5486b12d_full.json` (26/11/2026, lưới batch `[1, 2, 4, 8]`, K = 0–4, chế độ full). Bản sao lưu: `~/.flexsds/profiles-backup/profile_sm_75_remote_t4_fcda5486b12d_full.b1-8.2611.json`.

## 1. Triệu chứng
- `flexsds serve --gpu T4 --dry-run` ở chế độ adaptive in traceback `ValueError: Profile batch grid does not cover max_num_seqs=32; recalibrate` và thoát mã 1.
- Campaign `flexsds-fast-t4-2711` (27/11) sẽ có 9 ô adaptive fail ở bước khởi động server. Campaign này đã được dừng để sửa.

## 2. Nguyên nhân

**Phục vụ thật trên T4 chạy được B=32**: đợt M1 đã đo c=32 trên T4, và vLLM khi phục vụ bật chunked prefill. Chỉ **bước hiệu chỉnh** bị giới hạn.

Bước hiệu chỉnh (`src/flexsds/core/calibration.py`, `calibration_engine_kwargs`) cần bước decode được đo có **đúng hình dạng**: cả B chuỗi ở ngữ cảnh 1024, mỗi chuỗi K+1 token. Muốn vậy, cả lô phải được prefill trong **một** bước:
- `enable_chunked_prefill=False`;
- `max_num_batched_tokens = max(4096, max(batches) × 1024)`, tức 32.768 token ở B=32.

Khi đó vLLM phải dành bộ nhớ activation cho một lượt prefill 32k token. Trên T4 (14,58 GiB dùng được), phần còn lại cho KV cache vì thế rất nhỏ.

Không thể bật chunked prefill: chuỗi nào prefill xong trước sẽ decode trước và vượt ngữ cảnh 1024, nên bước đúng hình dạng không bao giờ xuất hiện và lần đo báo lỗi `No exact measured … sample`.

## 3. Các lần đo

| Ngày | `gpu_memory_utilization` | Kết quả |
|---|---|---|
| 26/11 | 0.85 (mặc định) | KV cache khoảng 8,4k token, không đủ cho B=16 × 1024. Hồ sơ được tạo với `--batches 1,2,4,8` |
| 27/11 | 0.95 (`--gpu-memory-utilization 0.95`) | Đo xong B = 1, 2, 4, 8 (K=0). **Hết bộ nhớ (OOM) ở B=16** |

Log của lần 27/11 nằm trên máy Dương, không có trong repo: `~/.flexsds/logs/modal-calibration-eha_32uj.log`, Modal T4, workspace `nguyenduongwork2004`, chi phí khoảng 0,35 USD.
- Lúc khởi động: `Model loading took 5.94 GiB`; trọng số cộng phần ngoài torch 6,94 GiB, activation đỉnh 4,98 GiB, `Available KV cache memory: 1.93 GiB`, `GPU KV cache size: 34,800 tokens`. Như vậy B=32 × (1024 + 12) ≈ 33,2k token vừa đủ về dung lượng.
- Tới B=16: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.31 GiB. GPU 0 has a total capacity of 14.58 GiB of which 117.31 MiB is free … 10.98 GiB is allocated by PyTorch, and 3.33 GiB is reserved by PyTorch but unallocated.`
- Suy luận: bộ nhớ bị phân mảnh sau các lượt đo B nhỏ hơn. PyTorch gợi ý `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; chưa thử.

## 4. Xử lý hiện tại
- PR #59: lưới batch phải là dãy liên tục bắt đầu từ 1 trong `1, 2, 4, 8, 16, 32`. Lưới ngắn hơn `max_num_seqs` vẫn được chấp nhận, khi đó batch lớn hơn dùng chi phí B=8 (cách planner `EmpiricalCostProfile` vốn làm).
  - Khi khởi động, server in `Profile measured up to B=8; batches above use the B=8 costs`.
  - `benchmark plan` in `(measured B<=8; larger batches use B=8 costs)`.
- PR #60: thêm cờ `flexsds profile --gpu-memory-utilization 0.50..0.98` (mặc định 0.85); giá trị chỉ được ghi vào evidence khi có đặt khác mặc định.

## 5. Giới hạn cần ghi trong báo cáo
- Adaptive trên T4 ở c > 8 dùng **chi phí B=8 ngoại suy**. Chi phí một bước thực tế tăng theo B, nên planner đánh giá thấp chi phí và có thể chọn K lớn hơn mức tối ưu. Số adaptive T4 ở c=32 phải đọc kèm giới hạn này.
- Các GPU khác (A10, L40S, A100) có hồ sơ đủ 6 batch.

## 6. Hướng làm tiếp (chưa làm)
- Thử `expandable_segments:True` trong tiến trình hiệu chỉnh (chế độ fast, khoảng 0,3–0,5 USD).
- Hoặc đo mỗi batch bằng một engine riêng, đặt `max_num_batched_tokens = B × 1024` theo từng batch, để B nhỏ không làm phân mảnh bộ nhớ trước khi đo B lớn.
