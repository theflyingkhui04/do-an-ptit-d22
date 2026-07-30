# experiments/ — Đánh giá

Nơi chứa phần đánh giá end-to-end. Dự kiến:

- **Tập workflow tương phản** — agent nhiều-lượt, sinh output dài, batch ngắn (Dify/LangGraph).
- **Pipeline đo** — thu số liệu qua **Langfuse** + instrument riêng (acceptance-rate, tok/s cấp server); chạy nhiều lần, lấy phân phối.
- **Phân tích & dashboard** — đối chiếu 3 làn (AR thường / EAGLE-2 gốc / bản thích ứng), dựng **bản đồ use-case**.

Quy tắc đo: cùng phần cứng, warm-up, lặp lại nhiều lần, báo cáo variance (wall-clock trên GPU chia sẻ rất nhiễu).

*Chưa hiện thực — đang ở giai đoạn thiết kế/đề cương.*
