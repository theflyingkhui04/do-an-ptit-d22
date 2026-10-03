# Mẫu template quyển báo cáo chuẩn PTIT

> File này là bản mẫu dùng để thống nhất hình thức, cấu trúc và phong cách viết cho quyển báo cáo và chuyên đề đồ án theo chuẩn PTIT.

## 1. Cấu trúc quyển báo cáo

1. Trang bìa
2. Trang phụ bìa / thông tin đề tài
3. Lời cam đoan
4. Lời cảm ơn
5. Tóm tắt nội dung (Abstract)
6. Mục lục
7. Danh mục hình vẽ, bảng, ký hiệu
8. Chương 1: Mở đầu
9. Chương 2: Cơ sở lý luận / Tổng quan
10. Chương 3: Thiết kế hệ thống / Phương pháp
11. Chương 4: Thực nghiệm
12. Chương 5: Kết quả và bàn luận
13. Chương 6: Kết luận và hướng phát triển
14. Tài liệu tham khảo
15. Phụ lục (nếu cần)

## 2. Hình thức trình bày

### 2.1. Font chữ và căn lề

- Font: Times New Roman hoặc Arial thống nhất toàn tài liệu.
- Cỡ chữ: 12–13 cho nội dung, 14–16 cho tiêu đề chương.
- Căn lề: trái 3.0 cm, phải 2.5 cm, trên 3.0 cm, dưới 3.0 cm.
- Dãn dòng: 1.5 lines.
- Chữ in nghiêng dùng cho thuật ngữ nước ngoài, tên khoa học, tên tập tin/hệ thống nếu cần thiết.

### 2.2. Tiêu đề chương và mục

- Chương: số La Mã hoặc số Ả Rập theo quy định của trường; ví dụ: Chương 1.
- Mục: "1.1", "1.2", "2.1" ...
- Tiêu đề mục phải ngắn gọn, mô tả rõ nội dung, không viết câu dài quá mức.

### 2.3. Bảng biểu, hình vẽ

- Mỗi hình và bảng phải có nhãn và tiêu đề mô tả.
- Trích dẫn hình/bảng trong văn bản phải sử dụng quy ước: "Hình 1.1" hoặc "Bảng 4.1".
- Nên đặt tên file và đường dẫn thật cho mỗi biểu đồ/số liệu khi đánh dấu nguồn dữ liệu.

### 2.4. Cách trích dẫn

- Sử dụng trích dẫn chuẩn theo kiểu APA/IEEE tùy quy định của bộ môn.
- Mỗi khẳng định khoa học nên đi kèm nguồn bản gốc hoặc nguồn đã được kiểm chứng.
- Tránh trích dẫn lại không cần thiết; ưu tiên trích dẫn các nguồn gốc đáng tin cậy.

## 3. Lưu ý về phong cách viết

- Viết theo kiểu khoa học, rõ ràng, mạch lạc, không viết kiểu nhật ký linh tinh.
- Nội dung cần có: bối cảnh, bài toán, mục tiêu, phương pháp, kết quả và đóng góp.
- Mỗi chương nên có một "điểm nhấn" rõ ràng: chương này giải quyết vấn đề gì, bằng cách nào, và đưa đến kết luận gì.
- Tránh các câu mơ hồ như "đề tài rất quan trọng" nếu không có căn cứ và đánh giá cụ thể.

## 4. Mẫu khung nội dung của Chương 1

### 1.1. Bối cảnh và động lực

- Nêu bối cảnh kỹ thuật / ứng dụng.
- Chỉ ra sự gia tăng của bài toán trong thực tế.
- Nêu lý do cần giải quyết vấn đề và mức độ ảnh hưởng của nó.

### 1.2. Phát biểu bài toán

- Mô tả vấn đề kỹ thuật chính cần giải.
- Chỉ ra các hạn chế của phương pháp hiện có hoặc rào cản thực tế.
- Đặt vấn đề trong ngữ cảnh nghiên cứu đang triển khai.

### 1.3. Câu hỏi nghiên cứu

- RQ1: vấn đề nào cần kiểm tra?
- RQ2: phương pháp đề xuất có giải quyết được vấn đề không?
- RQ3: phương án đề xuất phù hợp với loại workload hay tình huống nào?

### 1.4. Mục tiêu và phạm vi

- Mục tiêu tổng quát.
- Mục tiêu cụ thể (G1, G2, G3 ...).
- Giới hạn phạm vi: mô hình, nền tảng, dữ liệu, phần cứng, điều kiện thực nghiệm.
- Những phần nằm ngoài phạm vi để tránh hiểu nhầm.

### 1.5. Đóng góp của đề tài

- Nêu các đóng góp mới về phương pháp, hệ thống, đánh giá và tính ứng dụng.
- Mỗi đóng góp nên được mô tả rõ về giá trị học thuật và giá trị thực tiễn.

### 1.6. Bố cục của luận án / báo cáo

- Tóm tắt ngắn gọn nội dung từng chương.
- Giúp người đọc hiểu được mạch logic của toàn bộ quyển báo cáo.

## 5. Gợi ý viết chuẩn trong đồ án này

Với đề tài về "Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs", nên ưu tiên sử dụng các dạng câu sau trong Chương 1:

- "Trong các hệ thống tự host LLM, độ trễ sinh trở thành nút thắt ..."
- "Speculative decoding là một kỹ thuật giữ nguyên đầu ra nhưng giảm độ trễ ..."
- "Tuy nhiên, trên GPU phổ thông, cấu hình nháp tĩnh có thể gây ra tăng tốc âm hoặc OOM ..."
- "Vì vậy, đề tài tập trung vào việc xây dựng một bộ điều khiển thích ứng ..."
- "Mục tiêu của đề tài là xác định khi nào việc bật hoặc tắt speculative decoding là hợp lý ..."

## 6. Checklist chốt trước khi nộp bản thảo

- [ ] Đã có bìa và trang thông tin đề tài
- [ ] Hình vẽ, bảng và biểu đồ có nhãn rõ
- [ ] Tài liệu tham khảo đầy đủ và thống nhất
- [ ] Chương 1 có bối cảnh, bài toán, RQ, mục tiêu, phạm vi, đóng góp
- [ ] Đã kiểm tra nhất quán thuật ngữ trong cả quyển
- [ ] Mỗi khẳng định có nguồn hoặc lời giải thích rõ
- [ ] Không có đoạn văn dài, lủng củng, thiếu mạch logic

## 7. Gợi ý dùng như tài liệu nền cho Chương 1

Mẫu này không thay thế nội dung nghiên cứu, mà là phương án khung sườn để chuyển nội dung của [docs/01-gioi-thieu.md](01-gioi-thieu.md) thành văn bản quyển báo cáo chuẩn PTIT. Các đoạn văn cụ thể nên được viết dựa trên các quyết định và dữ liệu thực tế đã ghi trong [docs/03-quyet-dinh-thiet-ke.md](03-quyet-dinh-thiet-ke.md), [docs/04-quy-trinh-va-ke-hoach.md](04-quy-trinh-va-ke-hoach.md), [docs/06-ke-hoach-thuc-nghiem.md](06-ke-hoach-thuc-nghiem.md), và [docs/07-thuat-ngu.md](07-thuat-ngu.md).
