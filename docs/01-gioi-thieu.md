# 1. Giới thiệu

## 1.1. Tên đề tài

**(TV)** *Tăng tốc suy luận mô hình ngôn ngữ lớn tự-host cho tự động hoá quy trình: Giải mã suy đoán thích ứng phần cứng trên GPU phổ thông.*
**(EN)** *Accelerating Self-Hosted LLM Inference for Workflow Automation: Hardware-Aware Speculative Decoding on Consumer GPUs.*

*Phương án khác:* "Giải mã suy đoán thích ứng phần cứng cho workflow agent chi phí thấp" / "Tối ưu độ trễ sinh của LLM tự-host cho Dify/LangGraph".

## 1.2. Bối cảnh và động lực

Mô hình ngôn ngữ lớn (LLM) đang trở thành công cụ nền cho việc **tự động hoá các tác vụ lặp lại** — đặc biệt qua các nền tảng orchestration như **Dify** và **LangGraph**, nơi người dùng ghép nhiều lời gọi LLM thành một quy trình. Nhu cầu này tăng nhanh trong nhóm lập trình viên và sinh viên IT.

Tuy nhiên tồn tại một rào cản thực tế: **API thương mại tốn phí** với người dùng cá nhân ở quy mô sử dụng thường xuyên, trong khi nhiều tình huống lại yêu cầu **dữ liệu riêng tư, hoạt động offline, hoặc không phụ thuộc nhà cung cấp**. Hệ quả là lựa chọn khả dĩ duy nhất thường là **tự host mô hình mã nguồn mở trên phần cứng khiêm tốn** (laptop, GPU phổ thông).

Một câu hỏi tự nhiên đặt ra: *khi các mô hình đóng (GPT, Claude…) đã rất mạnh, tại sao còn phải tối ưu?* Câu trả lời gồm hai vế. Thứ nhất, **chính các hệ phục vụ LLM quy mô lớn cũng được xây trên đúng những kỹ thuật tối ưu suy luận này** (speculative decoding, quản lý KV-cache, lượng tử hoá) — nên đây là nghiên cứu hạ tầng lõi, không phải cạnh tranh với mô hình đóng. Thứ hai, **khi tự host, "chi phí mỗi token" không biến mất mà chuyển hoá** thành chi phí phần cứng: thời gian GPU, dung lượng VRAM và độ trễ. Trên GPU phổ thông, mô hình mã nguồn mở chạy **chậm tới mức khó dùng** cho các quy trình tương tác hoặc agent nhiều bước.

Trong các trục tối ưu, **độ trễ sinh (decode latency)** là nút thắt của trải nghiệm tương tác và đặc biệt của **các workflow agent gọi LLM nhiều lượt tuần tự** — nơi độ trễ **cộng dồn** qua từng bước. Đây chính là bối cảnh mà đề tài hướng tới.

## 1.3. Phát biểu bài toán

**Giải mã suy đoán (speculative decoding)** là hướng tăng tốc **không đổi đầu ra (lossless)**: một cơ chế nháp nhẹ đề xuất trước nhiều token, mô hình đích xác thực chúng trong một lượt truyền xuôi, giữ nguyên phân phối kết quả. **EAGLE-2** là một trong các phương pháp mạnh nhất, sử dụng *cây nháp động* điều chỉnh theo độ tự tin của bộ nháp.

Vấn đề: cấu hình của EAGLE-2 được thiết kế và kiểm chứng chủ yếu trên **GPU cao cấp băng thông rất cao** (A100, RTX 3090). Chi phí thực của mỗi bước giải mã là sự đánh đổi giữa lợi ích (xác thực song song) và chi phí (dựng/xác thực cây), và cán cân này **phụ thuộc trực tiếp vào tỉ lệ compute/băng thông của GPU**. Trên GPU phổ thông băng thông thấp — môi trường thực tế của nhóm người dùng mục tiêu — điểm tối ưu **dịch chuyển**, khiến tăng tốc thực tế **thấp hơn tiềm năng**. Song song đó, lợi ích của tối ưu tốc độ **phụ thuộc mạnh vào loại workload**, nhưng hiện chưa có đặc tả rõ ràng *"quy trình nào trên phần cứng phổ thông thực sự được hưởng lợi"*.

> **Câu hỏi nghiên cứu:**
>
> 1. Việc để quyết định nháp **thích ứng theo đặc tính phần cứng thực tế** có cải thiện tốc độ giải mã của EAGLE-2 trên GPU phổ thông không, trong khi vẫn giữ lossless?
> 2. Trên các quy trình tự động hoá thực tế (Dify/LangGraph), **loại workload nào thực sự hưởng lợi** từ tối ưu tốc độ, và hưởng lợi tới mức nào?

## 1.4. Mục tiêu và phạm vi

**Mục tiêu:** (a) tái lập EAGLE-2 làm baseline chạy được; (b) xây thành phần **nháp thích ứng phần cứng** dựa trên đo thực (profiling) độ trễ trên GPU đích; (c) đóng gói thành **endpoint tương thích OpenAI** cắm thẳng vào Dify/LangGraph; (d) **đánh giá end-to-end** trên tập quy trình thực tế và rút ra **bản đồ use-case**. (e) benchmark model đã tối ưu trên các tập dữ liệu đa dạng

**Phạm vi:** suy luận đơn GPU, phục vụ độ trễ thấp (batch nhỏ); mô hình đích cỡ 1–8B (FP16 hoặc lượng tử hoá 4-bit); sử dụng checkpoint EAGLE-2 công bố sẵn, chỉ huấn luyện đầu nháp một lần trên GPU thuê nếu cần mô hình đích riêng. **Mục tiêu không phải đánh bại EAGLE-2 toàn diện**, mà chứng minh cải thiện đo được **trong chế độ phần cứng phổ thông**, kèm đặc tả regime sử dụng.

## 1.5. Hướng tiếp cận (tóm tắt)

Đề tài (1) tái lập pipeline EAGLE-2; (2) **mô hình hoá chi phí** một bước giải mã theo đặc tính GPU đích và điều chỉnh chiến lược nháp cho khớp phần cứng; (3) phơi bày dưới dạng **endpoint OpenAI-compatible**; (4) **đo end-to-end** trên nhiều workflow tương phản bằng **Langfuse** kết hợp instrument riêng (acceptance-rate, tok/s); (5) đối chiếu ba làn — sinh tự hồi quy thường / EAGLE-2 gốc / bản thích ứng — để dựng **bản đồ use-case** cho biết khi nào nên bật tối ưu tốc độ.

---

# 2. Sản phẩm đầu ra (Output)

1. **Máy phục vụ suy luận tối ưu, tương thích OpenAI** (`/v1/chat/completions`) — cắm trực tiếp vào Dify/LangGraph mà không cần sửa ứng dụng; hỗ trợ hai làn *vanilla* (baseline) và *optimized*.
2. **Thành phần nháp thích ứng phần cứng** — module điều chỉnh chiến lược cây nháp theo hồ sơ phần cứng đo được, tích hợp trên nền serving.
3. **Bộ quy trình đánh giá (benchmark suite)** — tập workflow Dify/LangGraph tương phản (agent nhiều-lượt, sinh output dài, batch ngắn) cùng pipeline đo tự động (Langfuse + instrument acceptance-rate/tok-s).
4. **Bảng điều khiển so sánh trực quan** — hiển thị *cùng phần cứng, vanilla vs optimized*: tok/s, độ trễ, thời gian hoàn tất workflow, acceptance-rate.
5. **Bản đồ use-case & khung quyết định** — regime workload nào nên/không nên bật tối ưu tốc độ trên GPU phổ thông, có số liệu chứng minh.
6. **Mã nguồn mở + báo cáo** — có thể tái lập.

---

# 3. Đóng góp (Contributions)

1. **Về phương pháp:** một biến thể **giải mã suy đoán thích ứng phần cứng** — đưa đặc tính compute/băng thông của GPU đích vào quyết định nháp của EAGLE-2 — cho tốc độ cao hơn trên GPU phổ thông băng thông thấp, giữ nguyên tính lossless. (Phân biệt rõ với đóng góp *cây động theo confidence* vốn đã có sẵn của EAGLE-2.)
2. **Về đánh giá:** một **phương pháp và bộ kết quả đo end-to-end trên quy trình tự động hoá thực tế** (Dify/LangGraph), thay vì chỉ microbenchmark tách rời — phản ánh đúng lợi ích người dùng cuối cảm nhận.
3. **Về hiểu biết hệ thống:** một **đặc tả regime use-case** kèm **khung quyết định** trả lời "khi nào tối ưu tốc độ thực sự đáng bật trên phần cứng phổ thông" — bao gồm cả các trường hợp *không* nên dùng (batch throughput, input dài / output ngắn).
4. **Về ứng dụng:** một **hệ thống tự-host mã nguồn mở, cắm-là-chạy** giúp người dùng hạn chế tài nguyên chạy workflow LLM tương tác **nhanh hơn, riêng tư, không phụ thuộc API trả phí**.
