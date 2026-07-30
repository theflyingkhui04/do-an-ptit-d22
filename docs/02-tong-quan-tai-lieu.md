# Tổng quan tài liệu & lý do chọn hướng

Tài liệu này tóm tắt khảo sát ba hướng tối ưu suy luận LLM được cân nhắc cho đồ án, và giải thích vì sao chọn **tối ưu tốc độ (speculative decoding)**. Danh mục trích dẫn đầy đủ ở [references.md](references.md).

## Ba hướng được cân nhắc

| Hướng | Ý tưởng | Anchor tiêu biểu |
|---|---|---|
| **A. Nén KV-cache** | Bỏ bớt/gộp các mục KV ít quan trọng khi sinh → giảm VRAM, chứa context dài hơn | SnapKV, KeyDiff, Ada-KV |
| **B. Nén prompt/context** | Cắt token ít thông tin trong prompt → ít token đầu vào hơn | LLMLingua-2, LongLLMLingua |
| **C. Speculative decoding** | Nháp + xác thực → tăng tốc sinh **lossless** | EAGLE-2/-3, Medusa, Lookahead |

## Chấm điểm so sánh (theo ràng buộc đồ án)

Ràng buộc: 3 người, GPU phổ thông (T4 16GB + mượn A100 một lần), ưu tiên training-free, **phải demo được sự cải thiện một cách trực quan**.

| Tiêu chí | A. KV-cache | B. Nén prompt | C. Spec decoding |
|---|:--:|:--:|:--:|
| Mật độ paper / momentum | cao | cao | cao |
| Khả thi trên GPU phổ thông | ✅ tốt | ✅✅ tốt nhất | ⚠️ trung bình |
| Không cần huấn luyện | ✅ | ✅ | ⚠️ (đầu nháp cần train, có checkpoint sẵn) |
| Trực quan hoá được cải thiện | ✅ (VRAM, heatmap) | ⚠️ (prompt co, $ giảm) | ✅✅ (đua tok/s, lossless) |
| Câu chuyện chi phí khi **tự host** | ✅ (VRAM/context) | ⚠️ (mạnh nhất khi có API trả phí phía sau) | ✅ (tốc độ → throughput/UX) |

## Vì sao chọn C (tối ưu tốc độ)

1. **Trực quan mạnh nhất, lossless:** demo hai làn cùng sinh một đoạn văn — bên tối ưu nhanh 2–3×, chữ y hệt → không tranh cãi "có mất chất lượng không".
2. **Khớp bối cảnh sản phẩm:** người dùng mục tiêu chạy **workflow agent nhiều lượt** (Dify/LangGraph) — độ trễ cộng dồn, nên tăng tốc decode lợi kép.
3. **Câu chuyện chi phí self-host sạch:** khi tự host, "token free" là ảo tưởng — tốc độ trực tiếp quy ra GPU-giây/throughput/UX; C không cần một API trả phí phía sau để có ý nghĩa (khác điểm yếu của B).
4. **Đánh giá gọn về chất lượng:** vì lossless, chỉ cần *xác minh một lần* đầu ra khớp baseline; trục đánh giá chính là tốc độ (speedup, acceptance-rate).

## Chi tiết hướng C

- **Cơ chế:** ở chế độ sinh từng token (batch nhỏ), GPU nghẽn **băng thông bộ nhớ**, phần tính toán rảnh → xác thực nhiều token nháp trong *một* lượt gần như "miễn phí". Tính lossless đến từ cơ chế **xác thực bằng rejection-sampling** giữ nguyên phân phối của mô hình đích.
- **EAGLE / EAGLE-2:** thay mô hình nháp riêng bằng một **đầu nháp nhẹ dự đoán ở mức đặc trưng (feature)** của chính mô hình đích. **EAGLE-2** bổ sung **cây nháp động** điều chỉnh theo độ tự tin của bộ nháp.
- **⚠️ Lưu ý về "twist":** ý tưởng "làm cây động theo confidence" **chính là đóng góp lõi của EAGLE-2** → không phải điểm mới. Biến tấu của đồ án phải đi *quá* mức đó — xem [03-quyet-dinh-thiet-ke.md](03-quyet-dinh-thiet-ke.md).
- **Chuẩn đánh giá:** speedup wall-clock, **acceptance-rate / độ dài chấp nhận trung bình τ**; bộ tham chiếu **Spec-Bench** (bên thứ ba, chạy Vicuna-7B FP16 trên RTX 3090 24GB).
- **Khả thi phần cứng:** 7B FP16 (~14GB) **không vừa** 8GB và sát nút trên T4 16GB → dùng target **1–3B** hoặc **7B lượng tử 4-bit**; dùng checkpoint EAGLE-2 sẵn có (train đầu nháp một lần trên A100 nếu cần target riêng).

## Vì sao *không* chọn A / B (tóm tắt)

- **B (nén prompt):** rẻ và an toàn nhất, nhưng câu chuyện "tiết kiệm chi phí" **mạnh nhất khi có API trả phí phía sau**; thuần self-host thì lợi ích tụt về latency/throughput, kém giật gân. Trực quan cũng nhẹ hơn.
- **A (KV-cache):** khả thi & paper dày nhất, câu chuyện "chứa tài liệu dài trên máy rẻ" tốt; nhưng demo thiên về "khả năng/bộ nhớ", ít đã mắt hơn đua tốc độ. Là **phương án dự phòng mạnh** nếu C gặp rủi ro.

## Caveat quan trọng (trung thực khi bảo vệ)

- Hầu hết con số speedup/nén trong paper là **tác giả tự báo, best-case** (SnapKV 3.6×/8.2× đo ở 16K token trên A100-80GB; EAGLE-3 tới 6.5×, Medusa 2.2–3.6× trên Vicuna). Nguồn trung lập hiếm; Spec-Bench là ngoại lệ.
- **Sức chứa thực trên đúng GPU của nhóm chưa được paper nào đo** → việc đầu tiên khi hiện thực là **đo capacity thực tế** (model + KV cache + activations).
- **EAGLE là training-based**; "training-free" chỉ đúng khi dùng checkpoint công bố cho target phổ biến.

*Nguồn: khảo sát tài liệu có kiểm chứng (deep-research, 2026-07). Chi tiết trích dẫn: [references.md](references.md).*
