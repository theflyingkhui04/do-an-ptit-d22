# Bản đồ tài liệu

Danh sách **đầy đủ** các tài liệu đồ án cần có, vai trò của từng file, và ai chịu trách nhiệm. Mở file này trước khi định viết bất cứ tài liệu mới nào — để không tạo ra file trùng vai trò.

## Nguyên tắc

1. **Mỗi sự thật nằm đúng một chỗ.** Các file khác thì *liên kết* tới, không chép lại. Đội không gặp mặt nhau mà để trôi hai bản chép khác nhau thì sẽ dẫn tới hai người làm theo hai bản khác nhau.
2. **Quyết định thì ghi vào [03](03-quyet-dinh-thiet-ke.md), không ghi vào file mô tả.** File mô tả nói *cái gì*; nhật ký quyết định nói *vì sao*. Bảo vệ trước hội đồng sống bằng vế "vì sao".
3. **Tài liệu là điều kiện Done, không phải việc làm cuối kỳ.** Xem [Definition of Done](04-quy-trinh-va-ke-hoach.md#definition-of-done).

---

## Tài liệu đồ án

| File | Vai trò | Ai giữ | Trạng thái |
|---|---|---|---|
| [00-template-quyen-bao-cao-ptit.md](00-template-quyen-bao-cao-ptit.md) | Mẫu template quyển báo cáo theo chuẩn PTIT + checklist trình bày | Dương | ✅ |
| [01-gioi-thieu.md](01-gioi-thieu.md) | Tên đề tài, bối cảnh, phát biểu bài toán, **câu hỏi nghiên cứu**, mục tiêu/phạm vi, đầu ra, đóng góp | Cả nhóm | ✅ đã cập nhật 11/09 |
| [02-tong-quan-tai-lieu.md](02-tong-quan-tai-lieu.md) | Khảo sát 3 hướng (KV-cache / nén prompt / spec decoding) + lý do chọn hướng C | Dương | ✅ |
| [03-quyet-dinh-thiet-ke.md](03-quyet-dinh-thiet-ke.md) | **Nhật ký quyết định (ADR)** — mọi QĐ lớn kèm lý do + các câu hỏi còn mở | Khôi | ✅ |
| [04-quy-trinh-va-ke-hoach.md](04-quy-trinh-va-ke-hoach.md) | Quy trình làm việc, nhịp, phân vai, MoSCoW, lộ trình mốc, sổ rủi ro | Khôi (PM) | ✅ |
| [05-thiet-ke-he-thong.md](05-thiet-ke-he-thong.md) | Kiến trúc, **2 giao diện đóng băng**, thuật toán bộ điều khiển, sơ đồ luồng | Khôi | 🟡 khung sườn — chốt trong Tuần 1 |
| [06-ke-hoach-thuc-nghiem.md](06-ke-hoach-thuc-nghiem.md) | **Giao thức đo**, ma trận kết quả, cách xác minh lossless, kỷ luật thống kê | Đại | 🟡 khung sườn — chốt trước M1 |
| [07-thuat-ngu.md](07-thuat-ngu.md) | Bảng thuật ngữ & ký hiệu (alpha, tau, TPOT, ITL, MAT…) — **dùng thống nhất trong cả quyển** | Dương | ✅ |
| [08-outline-quyen.md](08-outline-quyen.md) | Dàn ý quyển báo cáo, ánh xạ chương ↔ tài liệu ↔ người viết | Dương | ✅ |
| [references.md](references.md) | Danh mục trích dẫn kèm arXiv ID và trạng thái bình duyệt | Dương | ✅ |
| [paper_summary/](paper_summary/) | Tóm tắt sâu 6 paper nền + mạch vấn đề→giải pháp | Khôi | ✅ |
| [slide/](slide/) | Slide bảo vệ | Dương | 🟡 sau M4 |

## Tài liệu kỹ thuật (ngoài `docs/`)

| File | Vai trò | Trạng thái |
|---|---|---|
| [../README.md](../README.md) | Mặt tiền repo — tóm tắt 60 giây cho người lạ (gồm cả hội đồng) | ✅ |
| [../src/README.md](../src/README.md) | Bản đồ module mã nguồn + cách chạy | 🟡 cập nhật khi có code |
| [../experiments/README.md](../experiments/README.md) | Cách chạy benchmark + nơi chứa kết quả | 🟡 cập nhật khi có harness |
| `../experiments/results/` | **Số liệu thô + bảng tổng hợp** (nguồn sự thật duy nhất cho mọi biểu đồ) | ⬜ tạo ở M1 |
| [../requirements.txt](../requirements.txt) | Phụ thuộc — **phải pin phiên bản trước M1** | 🟡 chưa pin |
| [../.env.example](../.env.example) | Mẫu biến môi trường (`.env` thật bị gitignore) | ✅ |
| [../logs/M0-01/README.md](../logs/M0-01/README.md) | Worklog kỹ thuật & đo đạc thực nghiệm PoC T4 (M0-01) | ✅ |

## Chưa có — và cố tình chưa có

| Thứ | Vì sao chưa |
|---|---|
| `CONTRIBUTING.md` | Luật hợp tác đã nằm trong [04 §3](04-quy-trinh-va-ke-hoach.md); tách ra chỉ tạo bản chép thứ hai |
| Tài liệu API | Endpoint theo chuẩn OpenAI — trỏ tới đặc tả của OpenAI, không viết lại |
| Sổ rủi ro riêng | Đã nằm trong [04 §7](04-quy-trinh-va-ke-hoach.md) |

---

## Đọc theo vai

- **Hội đồng / GVHD:** [../README.md](../README.md) → [01](01-gioi-thieu.md) → [03](03-quyet-dinh-thiet-ke.md)
- **Thành viên mới bắt tay vào code:** [01](01-gioi-thieu.md) → [05](05-thiet-ke-he-thong.md) → [07](07-thuat-ngu.md) → [04 §3–4](04-quy-trinh-va-ke-hoach.md)
- **Người chạy benchmark:** [06](06-ke-hoach-thuc-nghiem.md) → [07](07-thuat-ngu.md)
- **Người muốn hiểu nền lý thuyết:** [02](02-tong-quan-tai-lieu.md) → [paper_summary/](paper_summary/)
