# Nhật ký quyết định thiết kế

Ghi lại các quyết định lớn và **lý do** — để bảo vệ trước hội đồng và để nhóm không lặp lại tranh luận cũ.

## QĐ-01 — Bỏ "ép domain", chọn chiến lược bám paper + biến tấu
- **Quyết định:** không gắn đề tài vào một domain ứng dụng (nông nghiệp, công nghiệp…); thay vào đó **tái lập một paper mạnh + thêm một biến tấu nhỏ, đo được**.
- **Lý do:** ép kiến trúc vào domain thiếu "thước đo" dẫn tới đắp logic chắp vá. Bám paper de-risk tính khả thi (đã có người chứng minh), luôn có baseline để so, và novelty nhỏ-gọn là đủ cho đồ án. "Tech thuần" không phải là không có ngữ cảnh — mà là đổi domain-bẩn lấy **benchmark có thước đo**.

## QĐ-02 — Khung phần cứng
- **Quyết định:** làm việc chủ lực trên **T4 16GB** (Colab/Kaggle), **mượn A100 một lần** cho bước huấn luyện (nếu cần). Máy local: laptop / GPU phổ thông.
- **Lý do:** phản ánh đúng người dùng mục tiêu (sinh viên, hạn chế tài nguyên) và giữ chi phí thấp. Loại bỏ các hướng đòi hỏi huấn luyện nặng dài hạn.

## QĐ-03 — Chọn hướng C (tối ưu tốc độ / speculative decoding)
- **Quyết định:** engine lõi = **speculative decoding**, anchor **EAGLE-2**. A (KV-cache) là dự phòng; B (nén prompt) loại.
- **Lý do:** demo trực quan mạnh nhất & lossless; khớp workflow agent nhiều-lượt; câu chuyện chi phí self-host sạch (B cần API trả phí phía sau mới "kể" được chuyện tiết kiệm đô). Chi tiết: [02-tong-quan-tai-lieu.md](02-tong-quan-tai-lieu.md).

## QĐ-04 — Biến tấu (twist) phải đi *quá* cây-động-theo-confidence
- **Quyết định:** twist = **nháp thích ứng phần cứng** (đưa đặc tính compute/băng thông GPU đích vào quyết định nháp), *không* phải "làm cây động theo confidence".
- **Lý do:** "cây động theo confidence" **đã là đóng góp lõi của EAGLE-2** → làm lại = trùng, không tính là mới. Các hướng twist thật sự mới, khả thi:
  1. **Chỉnh cây theo phần cứng** (hay nhất cho ngữ cảnh này): đưa tỉ lệ compute/băng thông của GPU đích vào cost-model chọn cây → tối đa hoá speedup *thực* trên GPU phổ thông. Training-free.
  2. **Ngân sách nháp thích ứng theo lịch sử acceptance** (đang "trúng" thì nháp dài hơn). Training-free.
  3. **Calibrate acceptance-predictor** (theo vị trí/nhiệt độ) để tỉa cây chuẩn hơn. Cần fit nhẹ trên dev-set.
- **Ghi nhớ:** twist **không cần đánh bại EAGLE-2 toàn diện** — chỉ cần thắng/ngang trong **một regime cụ thể** (GPU băng thông thấp, hoặc target 4-bit) là đủ.

## QĐ-05 — Bỏ ý tưởng "app quản lý model" tổng quát
- **Quyết định:** không xây app launcher/console đa dụng (chọn model, đọc phần cứng, quản lý đa người dùng…).
- **Lý do:** ~80% chức năng đó **đã có** (Ollama, LM Studio, Open WebUI, vLLM) → nguy cơ làm bản clone kém hơn, còn phần tối ưu (đóng góp thật) bị đẩy thành mock. Dồn công vào engine + đánh giá thay vì plumbing.

## QĐ-06 — Đóng gói & tích hợp qua endpoint tương thích OpenAI
- **Quyết định:** phơi engine dưới dạng endpoint `/v1/chat/completions`; đo end-to-end bằng cách cắm vào Dify/LangGraph thật.
- **Lý do:** OpenAI-compat là "hợp đồng mỏng" — mọi framework (Dify, LangChain, LangGraph) cắm được ngay. Không cần tự xây app; hệ sinh thái sẵn có làm lớp giao diện. Làn "vanilla" dùng Ollama/llama.cpp làm baseline để so.

## QĐ-07 — Phương pháp đánh giá
- **Quyết định:** đo end-to-end trên **vài workflow tương phản** (agent nhiều-lượt, sinh output dài, batch ngắn) bằng **Langfuse** + **instrument riêng** (acceptance-rate, tok/s cấp server); chạy nhiều lần lấy phân phối.
- **Lý do:** một luồng chỉ là một điểm dữ liệu; vài luồng mới vẽ được **bản đồ use-case**. Langfuse cho câu chuyện end-to-end nhưng **không** bắt acceptance-rate/tok-s → phải instrument thêm. Wall-clock trên Colab nhiễu → kỷ luật thí nghiệm (warm-up, lặp lại, báo cáo variance). "Dùng AI để vẽ biểu đồ" chỉ là công cụ, không phải điểm cộng.

## Bản đồ use-case (giả thuyết cần kiểm chứng)
- **Nên bật tối ưu tốc độ:** tương tác real-time; **agent nhiều lượt gọi tuần tự** (độ trễ cộng dồn); **sinh output dài**.
- **Không nên / vô ích:** batch throughput lớn (spec decoding có thể *giảm* throughput ở batch lớn); input dài / output ngắn (nghẽn ở prefill → hướng A/B); nghẽn bộ nhớ / context siêu dài (→ hướng A).
