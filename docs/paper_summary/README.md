# Tóm tắt 6 paper nền

Sáu bài này là toàn bộ nền lý thuyết của đồ án, xếp theo **mạch phát triển** chứ không theo năm. Đọc tuần tự sẽ thấy rõ mỗi bài giải quyết hạn chế của bài trước, và **khoảng trống mà đồ án nhắm vào**.

| # | Paper | Năm / Nơi | Trục đóng góp | Trạng thái |
|---|---|---|---|---|
| [1](01-speculative-decoding-leviathan2023.md) | Speculative Decoding | ICML 2023 | Nền móng: nháp + xác thực, chứng minh **lossless** | ✅ peer-reviewed |
| [2](02-eagle-li2024.md) | EAGLE | ICML 2024 | Nháp ở **tầng đặc trưng** thay vì tầng token | ✅ peer-reviewed |
| [3](03-eagle2-li2024.md) | EAGLE-2 | EMNLP 2024 | **Cây nháp động** theo độ tự tin — *anchor của đồ án* | ✅ peer-reviewed |
| [4](04-talon-liu2026.md) | Talon | arXiv 01/2026 | Cây theo **ngân sách token**, bỏ fixed depth/width | ⚠️ preprint |
| [5](05-pearl-liu2025.md) | PEARL | ICLR 2025 | Trục **lịch chạy**: chồng lấn nháp ‖ xác thực | ✅ peer-reviewed |
| [6](06-dcut-liu2026.md) | D-cut | arXiv 07/2026 | **Cost-model phần cứng** + cắt tỉa theo batch | ⚠️ preprint |

## Mạch vấn đề → giải pháp

```
Paper 1  Cần model nháp riêng, chuỗi tuyến tính, K cố định
   │
   ├─(bỏ model nháp riêng)──────────────>  Paper 2  EAGLE: draft head ở mức feature
   │                                          │  nhưng cây vẫn TĨNH
   │                                          ▼
   │                                       Paper 3  EAGLE-2: cây động theo confidence
   │                                          │  nhưng depth/width vẫn cứng
   │                                          ▼
   │                                       Paper 4  Talon: chỉ cố định NGÂN SÁCH N
   │                                             (vẫn chỉ đúng ở batch = 1)
   │
   ├─(trục lịch chạy)──────────────────>  Paper 5  PEARL: song song hoá 2 pha
   │                                             (đòi nhiều GPU)
   │
   └─(trục phần cứng / batched)────────>  Paper 6  D-cut: SD có thể CHẬM HƠN AR
                                                 khi concurrency cao → cắt theo cost-model
```

## Ba điều rút ra cho đồ án

1. **Lossless là miễn phí nếu không đụng vào tiêu chí chấp nhận.** Cả 6 paper đều chỉ thay đổi *cách chọn ứng viên nháp* hoặc *lịch chạy*; không bài nào sửa Acceptance Criterion. Đồ án phải giữ nguyên kỷ luật này.
2. **Chỗ trống thật nằm ở phần cứng, không ở thuật toán chọn cây.** Confidence-driven tree đã bị EAGLE-2 + Talon chiếm. D-cut chứng minh rằng "cấu hình tối ưu phụ thuộc GPU" là bài toán thật và **chưa ai giải cho GPU phổ thông ở batch nhỏ** — đúng ô mà đồ án đứng (xem [QĐ-04](../03-quyet-dinh-thiet-ke.md)).
3. **Đo ở batch = 1 là đo trong điều kiện dễ nhất.** D-cut cho thấy speedup có thể tụt xuống 0.67× khi concurrency tăng. Báo cáo của nhóm phải là **thang concurrency**, không phải một con số duy nhất.

## Lưu ý khi trích dẫn

Paper 4 và 6 là **preprint 2026 chưa qua bình duyệt**. Dùng làm tham chiếu state-of-the-art và làm luận cứ cho hướng đi, **không** dùng làm nền móng kỹ thuật. Anchor chính thức vẫn là **EAGLE-2 (EMNLP 2024)**.

Mọi con số speedup trong 6 bài đều là **tác giả tự báo, best-case** trên phần cứng của họ — phải tự đo lại trên GPU của nhóm.
