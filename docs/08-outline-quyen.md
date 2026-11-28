# Dàn ý quyển báo cáo

> Đây là **đích đến của điều kiện Done #3** ([04](04-quy-trinh-va-ke-hoach.md)): *"có ít nhất một đoạn văn viết vào bản nháp quyển"*. Không có file này thì điều kiện đó không thực thi được.
> Cách dùng: làm xong việc gì → mở đúng mục tương ứng dưới đây → viết ngay vài đoạn. **Tuần 10 là để gọt, không phải để viết từ đầu.**

## Nguyên tắc

Phần lớn nội dung **đã nằm trong `docs/`**. Viết quyển chủ yếu là **chuyển thể và nối mạch**, không phải sáng tác lại. Cột "Nguồn" chỉ rõ lấy từ đâu.

## Dàn ý

| Chương | Mục | Nguồn | Người viết chính & Hỗ trợ | Viết được từ |
|:--:|---|---|:--:|---|
| **1** | **Mở đầu** | | Dương (Lead) + Đại | **Tuần 1** |
| | 1.1 Bối cảnh & động lực | [01 §1.2](01-gioi-thieu.md) | | |
| | 1.2 Phát biểu bài toán | [01 §1.3](01-gioi-thieu.md) | | |
| | 1.3 Câu hỏi nghiên cứu (RQ1–3) | [01 §1.3](01-gioi-thieu.md) | | |
| | 1.4 Mục tiêu, phạm vi, **ngoài phạm vi** | [01 §1.4](01-gioi-thieu.md) | | |
| | 1.5 Đóng góp | [01 §3](01-gioi-thieu.md) | | |
| | 1.6 Bố cục quyển | — | | |
| **2** | **Cơ sở lý thuyết & tổng quan** | | Khôi (Lead) + Dương | **Tuần 1–3** |
| | 2.1 Nút thắt suy luận LLM (bandwidth vs compute bound) | [paper_summary/01](paper_summary/01-speculative-decoding-leviathan2023.md) | | |
| | 2.2 Giải mã suy đoán & chứng minh lossless | [paper_summary/01](paper_summary/01-speculative-decoding-leviathan2023.md) | | |
| | 2.3 Nháp ở mức đặc trưng: EAGLE | [paper_summary/02](paper_summary/02-eagle-li2024.md) | | |
| | 2.4 Cây nháp động: EAGLE-2, Talon | [03](paper_summary/03-eagle2-li2024.md), [04](paper_summary/04-talon-liu2026.md) | | |
| | 2.5 Các trục khác: PEARL (lịch chạy), D-cut (phần cứng) | [05](paper_summary/05-pearl-liu2025.md), [06](paper_summary/06-dcut-liu2026.md) | | |
| | 2.6 Các hướng đã cân nhắc và loại | [02-tong-quan](02-tong-quan-tai-lieu.md) | | |
| | 2.7 **Khoảng trống nghiên cứu** | [paper_summary/README](paper_summary/) §"Ba điều rút ra" | | |
| **3** | **Thiết kế hệ thống** | | Khôi (Lead) + Đại | **sau M2 (14/10)** |
| | 3.1 Kiến trúc tổng thể | [05 §1](05-thiet-ke-he-thong.md) | | |
| | 3.2 Đo hồ sơ phần cứng | [05 §2.1](05-thiet-ke-he-thong.md) | | |
| | 3.3 Bộ điều khiển ngân sách cây | [05 §2.2](05-thiet-ke-he-thong.md) | | |
| | 3.4 Đảm bảo lossless | [QĐ-11](03-quyet-dinh-thiet-ke.md) | | |
| | 3.5 Tích hợp vLLM & endpoint | [05 §3–4](05-thiet-ke-he-thong.md) | | |
| | 3.6 **Các quyết định thiết kế và lý do** | [03](03-quyet-dinh-thiet-ke.md) — chuyển thể QĐ-01→11 | | |
| | 3.7 Sản phẩm minh hoạ: web console FlexSDS Benchmark (M5-01) | [chuong-3-7](chuong-3-7-web-console.md), QĐ-18–19 | Dương | bản nháp 23/11 |
| **4** | **Thực nghiệm** | | Đại (Lead) + Khôi | **sau M1 (04/10)** |
| | 4.1 Thiết lập: phần cứng, mô hình, phần mềm | [05](05-thiet-ke-he-thong.md), [QĐ-09](03-quyet-dinh-thiet-ke.md) | | |
| | 4.2 Workload & tập dữ liệu | [01 §1.6](01-gioi-thieu.md) | | |
| | 4.3 Chỉ số & giao thức đo | [06 §3](06-ke-hoach-thuc-nghiem.md), [07](07-thuat-ngu.md) | | |
| | 4.4 Xác minh lossless | [06 §4](06-ke-hoach-thuc-nghiem.md) | | |
| **5** | **Kết quả & bàn luận** | | Đại (Lead) + Khôi + Dương | **sau M4 (15/11)** |
| | 5.1 Chuẩn đối chứng (Baseline): vanilla vs static | F1 | | |
| | 5.2 RQ1 — adaptive vs static | F1, F2 | | |
| | 5.3 RQ2 — **tăng tốc âm & an toàn bộ nhớ** | F2, F4 | | |
| | 5.4 Bằng chứng thích ứng (ngân sách cây theo thời gian) | F3 | | |
| | 5.5 RQ3 — **bản đồ use-case** | F5, F6 | | |
| | 5.6 Mối đe doạ tới tính hợp lệ | [06 §7](06-ke-hoach-thuc-nghiem.md) | | |
| **6** | **Kết luận & hướng phát triển** | | Cả nhóm | **Tuần 10** |
| | 6.1 Tóm tắt đóng góp | | | |
| | 6.2 Hạn chế | — mọi thứ bị cắt ở [04 §5](04-quy-trinh-va-ke-hoach.md) | | |
| | 6.3 Hướng phát triển | | | |
| | **Tài liệu tham khảo** | [references.md](references.md) | Dương | liên tục |
| | **Phụ lục** — bảng kết quả đầy đủ, hướng dẫn tái lập | `experiments/results/summary.csv` | Đại | sau M4 |

## Lịch viết

```
T1        Chương 1 nháp xong  ·  Chương 2 bắt đầu
T3        Chương 2 nháp xong  ·  Chương 4.1–4.3 (thiết lập — viết được ngay sau M1)
T5        Chương 3 nháp (sau M2 - 21/10)
T7–T8     Chương 3 hoàn chỉnh (TreeAdaptive)  ·  Chương 4 hoàn chỉnh (đối chứng 4 làn)
T9        ▲ M4 đóng băng code & số liệu (15/11 Freeze)
T10       Chương 5 + 6  ·  Ráp toàn bộ quyển >= 50 trang  ·  Thống nhất thuật ngữ theo [07]
T11       ▲ Nộp quyển chính thức (06/12/2026)  ·  Slide bảo vệ & tổng duyệt
```

**Cột mốc kiểm tra:** tới **23/10** (sau M3) mà bản nháp dưới **20 trang** là đang trễ, bất kể code chạy tốt cỡ nào ([04 §8](04-quy-trinh-va-ke-hoach.md)).

## Chuẩn bị bảo vệ — câu hỏi phản biện dự kiến

Chuẩn bị sẵn câu trả lời + slide dự phòng cho từng câu:

| Câu hỏi | Trả lời ở đâu |
|---|---|
| *"Tăng tốc 3× thì chất lượng có giảm không?"* | Bảng xác minh lossless — [06 §4](06-ke-hoach-thuc-nghiem.md). Chấm dứt trong 10 giây |
| *"Sao không dùng EAGLE-3 mà dùng EAGLE-2?"* | [M3-04] Đề tài đã mở rộng Làn 4 và tự huấn luyện đầu nháp EAGLE-3.1 cho Qwen2.5-7B (15.000 steps trên A100), chứng minh hiệu quả vượt trội |
| *"Cây động theo confidence là EAGLE-2 làm rồi mà?"* | [QĐ-04](03-quyet-dinh-thiet-ke.md) — twist là **chi phí đo thật từ phần cứng**, không chỉ dựa vào confidence |
| *"D-cut 2026 đã làm cost-model rồi, khác gì?"* | Khác **chế độ**: D-cut nhắm batched concurrency cao trên GPU datacenter; đồ án nhắm batch nhỏ trên GPU phổ thông — [01 §1.5](01-gioi-thieu.md) |
| *"Sao lại là hoá đơn? Có phải ép domain không?"* | [QĐ-08](03-quyet-dinh-thiet-ke.md) — pipeline là **dụng cụ đo**, bỏ đi thì đóng góp vẫn còn nguyên |
| *"Sao không tự train draft head?"* | [M3-04] Đã tự huấn luyện đầu nháp EAGLE-3.1 7B trên A100 để đưa vào Làn 4 làm bài đo đối chứng thành phần (Ablation Study) |
| *"Kết quả chỉ trên một GPU, một model?"* | [06 §7](06-ke-hoach-thuc-nghiem.md) — thừa nhận thẳng, nêu ở Hạn chế |
| *"Nếu adaptive không nhanh hơn static thì sao?"* | RQ2 vẫn đứng: loại bỏ tăng tốc âm và OOM là đóng góp độ tin cậy — [01 §1.3](01-gioi-thieu.md) |
