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

## QĐ-08 — Dùng pipeline hoá đơn MC-OCR làm **dụng cụ đo**, không phải làm domain
- **Quyết định:** đánh giá end-to-end trên pipeline hoá đơn tiếng Việt (MC-OCR), 2–3 node.
- **Vì sao điều này KHÔNG mâu thuẫn với QĐ-01:** QĐ-01 loại bỏ việc *ép kiến trúc vào một domain để lấy chỗ đứng*. Ở đây domain **không tham gia vào đóng góp** — không có logic nghiệp vụ hoá đơn nào nằm trong engine, và engine không biết nó đang xử lý hoá đơn. Pipeline được chọn thuần tuý vì tính chất **đo lường**: nó cho **ba chế độ entropy tương phản trong cùng một quy trình** (JSON khuôn mẫu → SQL ràng buộc → văn xuôi tự do), tức là ba điểm dữ liệu cho bản đồ use-case, cộng thêm hiệu ứng **độ trễ cộng dồn** mà một prompt đơn lẻ không tạo ra được.
- **Kiểm tra chéo:** nếu có ai hỏi *"bỏ hoá đơn đi thì đóng góp còn nguyên không?"* — câu trả lời phải là **còn nguyên**. Nếu một lúc nào đó câu trả lời thành "không", tức là đã trượt về ép domain và phải quay lại QĐ này.
- **Rủi ro:** tiêu ~15% ngân sách vào phần không phải đóng góp. Giảm thiểu: node 3 nằm ở tầng COULD, cắt đầu tiên khi trễ.

## QĐ-09 — Mô hình đích: Qwen2.5-7B-Instruct **4-bit (AWQ/GPTQ)**
- **Quyết định:** target = Qwen2.5-7B-Instruct lượng tử hoá 4-bit, chạy trên T4 16GB. Loại phương án 7B FP16.
- **Lý do:** 7B FP16 tốn ~14GB trên tổng 16GB → gần như chắc chắn OOM khi concurrency > 4, và không còn chỗ cho KV cache của cây nháp. 4-bit giải phóng ~10GB cho KV cache và cây, cho phép quét được cả thang concurrency — vốn là một trong các trục kết quả chính.
- **Hệ quả thú vị (nên khai thác khi bảo vệ, không nên giấu):** lượng tử hoá làm **mô hình đích chạy nhanh hơn**, tức là `t_verify` giảm, trong khi `t_draft` gần như giữ nguyên. Tỉ lệ `t_verify / t_draft` — chính là đại lượng quyết định speculative decoding có lời hay không — **dịch chuyển theo hướng bất lợi**. Nói cách khác, **4-bit làm speculative decoding khó ăn hơn**. Đây không phải điểm yếu của đề tài mà là **đúng luận điểm của đề tài**: cấu hình tĩnh tối ưu cho FP16/A100 sẽ sai trên 4-bit/T4, và chỉ bộ điều khiển đo-thật mới bám theo được.
- **Rủi ro phần cứng cần kiểm chứng ở M0:** T4 là kiến trúc Turing (sm_75) → **không hỗ trợ bfloat16** (phải dùng fp16), **không chạy được FlashAttention-2** (cần sm_80+, vLLM sẽ lùi về backend khác), và một số nhân lượng tử hoá tối ưu (ví dụ Marlin) **yêu cầu sm_80+**. Phải xác nhận tổ hợp vLLM + AWQ/GPTQ + EAGLE chạy được trên sm_75 **trong Tuần 1**, trước khi xây bất cứ thứ gì lên trên.

## QĐ-10 — **Không** tự huấn luyện đầu nháp; dùng checkpoint công bố
- **Quyết định:** dùng checkpoint EAGLE-2 đã công bố cho mô hình đích. Loại bỏ hạng mục "huấn luyện draft head trên dữ liệu hoá đơn tiếng Việt" khỏi phạm vi (README trước đây liệt kê nó như một luận cứ novelty).
- **Lý do:** đây là hạng mục **rủi ro cao, thời lượng không ước lượng được, và không thuộc đóng góp**. Đóng góp nằm ở **bộ điều khiển cây nháp**; chất lượng draft head chỉ ảnh hưởng tới α — tức là dịch chuyển mọi làn *cùng một lượng*, không làm thay đổi kết luận so sánh giữa các làn. Với ngân sách ~455 giờ-người ([04 §0](04-quy-trinh-va-ke-hoach.md)), giữ nó lại là đánh đổi sai.
- **Hệ quả cần xử lý:** phải tồn tại checkpoint EAGLE-2 cho **đúng** mô hình đích đã chọn. Nếu không có → đổi mô hình đích cho khớp checkpoint, **không** đổi hướng sang tự train. Kiểm chứng ở M0.

## QĐ-11 — Xác minh lossless bằng thực nghiệm, dù đã đúng về lý thuyết
- **Quyết định:** dù tính lossless được bảo toàn theo thiết kế (không đụng vào tiêu chí chấp nhận), vẫn chạy một phép xác minh khớp-từng-ký-tự giữa làn vanilla và làn optimized, và đưa vào quyển.
- **Lý do:** đây là điểm phản biện dễ đoán nhất của hội đồng (*"tăng tốc 3× thì có mất chất lượng không?"*). Một bảng so khớp đầu ra chấm dứt tranh luận trong 10 giây. Chi phí gần như bằng không, giá trị phòng thủ rất cao. Giao thức: [06 §4](06-ke-hoach-thuc-nghiem.md).

---

## Câu hỏi còn mở (phải đóng, kèm hạn)

| # | Câu hỏi | Vì sao quan trọng | Hạn chốt | Ai |
|---|---|---|---|---|
| **M-01** | **Vì sao neo EAGLE-2 mà không phải EAGLE-3?** | EAGLE-3 (`arXiv:2503.01840`) là bản mạnh nhất của dòng này và là baseline mà [Talon](paper_summary/04-talon-liu2026.md) dùng. Hội đồng gần như chắc chắn sẽ hỏi *"sao không dùng bản mới nhất?"*. **Hiện chưa có câu trả lời.** Giả thuyết cần kiểm chứng: EAGLE-3 đòi training-time test nên nặng hơn, và checkpoint/tích hợp vLLM ít sẵn hơn — nếu đúng thì đó là lý do chính đáng; nếu sai thì phải cân nhắc đổi anchor. | **M0 · 20/09** | Khôi |
| **M-02** | Có checkpoint EAGLE-2 cho Qwen2.5-7B không, và có chạy được với bản 4-bit không? | Chặn QĐ-10. Nếu không có → phải đổi mô hình đích. | **M0 · 20/09** | Khôi |
| **M-03** | vLLM `custom_class` proposer có cho đổi cấu hình cây **theo từng bước** không? | Rủi ro R1, mức chí mạng. Nếu không → phải fork vLLM hoặc tự viết vòng lặp suy luận. | **M0 · 20/09** | Khôi |
| **M-04** | GVHD có chấp nhận **kết quả âm tính** (adaptive không thắng stock) không? | Quyết định có cần phương án xoay góc kể chuyện ở cổng 01/11 hay không. Hỏi sớm thì rẻ. | **Tuần 1** | Khôi |
| **M-05** | Ngày bảo vệ và hạn nộp quyển chính thức | Toàn bộ lịch đang chạy theo giả định xấu nhất | khi khoa thông báo | TV3 |

---

## Bản đồ use-case (giả thuyết cần kiểm chứng)
- **Nên bật tối ưu tốc độ:** tương tác real-time; **agent nhiều lượt gọi tuần tự** (độ trễ cộng dồn); **sinh output dài**.
- **Không nên / vô ích:** batch throughput lớn (spec decoding có thể *giảm* throughput ở batch lớn); input dài / output ngắn (nghẽn ở prefill → hướng A/B); nghẽn bộ nhớ / context siêu dài (→ hướng A).
