# Quy trình Làm việc & Kế hoạch Dự án

> **Cập nhật:** 14/09/2026.  
> **Bối cảnh thực tế:** Nhóm gồm 3 thành viên, làm việc từ xa (lệch giờ, trao đổi online là chính), thời gian dành cho đồ án ~10–15h/người/tuần. Hạn bảo vệ dự kiến: giữa tháng 12/2026.

---

## 0. Các mốc thời gian cốt lõi

Kế hoạch được lập theo kịch bản an toàn nhất (phòng trường hợp lịch thi/bảo vệ diễn ra sớm):

| Mốc thời gian | Thời điểm dự kiến | Ý nghĩa & Hành động |
|---|---|---|
| **Ngày bảo vệ** | Giữa tháng 12/2026 | Lịch chính thức theo thông báo của Học viện. |
| **Hạn nộp quyển** | **~30/11/2026** | Nộp trước bảo vệ 2 tuần để thầy cô phản biện đọc quyển. |
| **Đóng băng Code & Số liệu (Freeze)** | **15/11/2026** | **Mốc quan trọng nhất:** Sau ngày này, tuyệt đối không sửa code hay chạy lại benchmark; 2 tuần cuối dành trọn thời gian để chỉnh sửa câu chữ báo cáo và chuẩn bị slide. |
| **Quỹ thời gian cả nhóm** | ~35 giờ/tuần | Tổng cộng ~450 giờ làm việc cho cả 3 thành viên trong suốt kỳ đồ án. |

---

## 1. Mô hình làm việc: Vì sao chọn Lean / Kanban?

### 1.1. Không dùng Waterfall (Thác nước)
Mô hình Waterfall yêu cầu phải chốt cứng mọi thứ từ đầu rồi làm tuần tự từng giai đoạn. Với một đề tài nghiên cứu hệ thống, ta **chưa thể biết trước 100% kết quả** (ví dụ: *Liệu thuật toán thích ứng có thực sự chạy nhanh hơn EAGLE-2 trên GPU T4 không?*). Nếu làm theo Waterfall và đợi đến tuần thứ 10 mới chạy thử benchmark, khi gặp sự cố sẽ không còn đường lùi.

### 1.2. Không áp dụng Scrum một cách máy móc
Scrum chuẩn chỉ phù hợp với các đội dự án ngồi cùng nhau và làm sản phẩm theo yêu cầu khách hàng:
* **Họp Daily hằng ngày:** Nhóm 3 người làm việc lệch giờ, nếu ép họp mỗi ngày sẽ rất mệt mỏi và sớm bị bỏ cuộc sau 1–2 tuần.
* **Họp Planning, Review, Retro quá nhiều:** Tốn tới 4–5 tiếng họp mỗi tuần, chiếm gần 15% tổng thời gian làm việc của cả nhóm.
* **Ước lượng điểm số (Story Points/Velocity):** Công việc nghiên cứu và gỡ lỗi hệ thống (như cấu hình CUDA, vLLM trên T4) rất khó đoán trước thời gian chính xác, việc cam kết khối lượng cứng theo sprint chỉ gây áp lực ảo.

### 1.3. Giải pháp lựa chọn: Lean / Kanban định hướng giải quyết rủi ro sớm (Risk-First)
* **Quản lý trực quan bằng Kanban:** Công việc được đưa lên bảng (Trello/Notion). Ai rảnh lúc nào vào kéo việc làm lúc đó, trạng thái công việc luôn minh bạch.
* **Tập trung vào 1 việc (WIP = 1):** Mỗi người tại một thời điểm chỉ nhận đúng **1 task duy nhất**, làm xong mới nhận việc mới để tránh dở dang.
* **Giải quyết rủi ro trước (Risk-First):** Các việc khó nhất, dễ gây "tắc đường" nhất (như kiểm tra xem mô hình có chạy được trên GPU T4 không) phải làm ngay trong 1–2 tuần đầu tiên.
* **Cố định thời hạn, linh hoạt phạm vi (Fixed Date, Variable Scope):** Hạn nộp quyển của trường là cố định; nếu thiếu thời gian, ta sẽ chủ động cắt giảm các tính năng phụ (như làm web giao diện) để bảo vệ các mục tiêu cốt lõi.

---

## 2. Nhịp phối hợp & Giao tiếp nhóm

Làm việc online không có nghĩa là không nói chuyện, mà là **giao tiếp ngắn gọn, đúng trọng tâm và đúng lịch**:

### 2.1. Cuộc gọi 45 phút tối thứ Bảy (Cố định lúc 20h30)
Cuộc gọi diễn ra nhanh gọn qua Google Meet/Discord với đúng 3 nội dung:
1. **Demo sản phẩm chạy được (15 phút):** Từng người chia sẻ màn hình những gì mình đã làm chạy được trong tuần (không nói suông, không dùng slide).
2. **Tháo gỡ điểm nghẽn (20 phút):** Tập trung giải quyết các lỗi khó, các vấn đề kỹ thuật đang khiến ai đó bị kẹt lại.
3. **Chốt việc tuần mới (10 phút):** Mỗi người kéo task từ danh sách việc cần làm vào phần việc của mình.

### 2.2. Cập nhật nhanh qua tin nhắn (Thứ Hai & Thứ Năm)
Khoảng 21h tối thứ 2 và thứ 5, mỗi thành viên nhắn đúng 3 dòng vào nhóm chat chung:
```text
- Đã xong: [Mã task hoặc tính năng đã merge/chạy được]
- Đang làm: [Đúng 1 task đang tập trung]
- Điểm nghẽn: [Gặp lỗi gì cần giúp? Nếu không có thì ghi "Không"]
```
> ⚠️ **Quy tắc phối hợp:** Gặp lỗi hoặc bị tắc công việc quá **24 giờ** mà không nhắn lên nhóm để hỏi thì coi như vi phạm quy trình làm việc.

---

## 3. Phân công nhiệm vụ & Cơ chế Ghép cặp Luân phiên 70/30 (Pair-Rotation System)

Đồ án tốt nghiệp đòi hỏi **cả 3 thành viên đều phải hiểu sâu và nắm vững toàn bộ kiến trúc**, từ tầng thuật toán Core-Engine, bộ công cụ đo kiểm Benchmark đến Pipeline ứng dụng thực tế. Dự án không chia rẽ biệt lập ("nước sông không phạm nước giếng"), mà vận hành theo mô hình **Ghép cặp Luân phiên 70/30**:

### 3.1. Phân vai hạt nhân (Core Ownership - 70% effort)
Mỗi thành viên là "kiến trúc sư trưởng" (Lead) chịu trách nhiệm chính về mặt kỹ thuật, chất lượng code và tiến độ của một luồng trọng điểm:

| Kỹ sư đảm nhiệm | Vai trò | Trách nhiệm hạt nhân (70% effort) | Vùng mã nguồn | Giao diện chuẩn hóa |
|---|---|---|---|---|
| **Khôi** | **Dev + PM** *(AI Systems & Quản lý dự án)* | Nghiên cứu thuật toán điều khiển cây nháp, tích hợp vào vLLM, xây dựng API server. Duyệt và merge code vào nhánh `main`. Viết Chương 2 & 3. | `src/proposer/`, `src/server/` | Cung cấp endpoint chuẩn OpenAI (`/v1/chat/completions`) và dữ liệu đo đạc (telemetry JSON). |
| **Đại** | **Dev + IT** *(MLOps, Benchmark & Hạ tầng)* | Xây dựng harness benchmark tự động, quản trị môi trường máy chủ GPU T4, chạy sweep concurrency, thu thập log JSONL và vẽ biểu đồ F1→F6. Viết Chương 4 & 5. | `experiments/` | Gọi API server của Khôi và đọc dữ liệu telemetry để xuất ra file CSV/biểu đồ. |
| **Dương** | **Dev + BA** *(AI Application & Trưởng nhóm Tài liệu)* | Phân tích nghiệp vụ hóa đơn VAT, thiết kế schema dữ liệu, xây dựng pipeline 3 node (Trích xuất JSON → SQL → Báo cáo). Định dạng quyển báo cáo chuẩn PTIT và thiết kế slide. Viết Chương 1 & 6. | `src/pipeline/`, `docs/` | Gửi dữ liệu hóa đơn qua API server của Khôi dưới dạng các request thông thường. |

### 3.2. Cơ chế Ghép cặp và Luân phiên (Pair-Rotation Matrix)
* **Quy tắc 2 người/task:** Tại một thời điểm, mỗi task phát triển chỉ có đúng **2 người cùng làm**:
  * **1 Lead (70% effort):** Chủ trì kiến trúc, gánh ~70% checklist subtasks (4/6 mục).
  * **1 Partner (30% effort):** Hỗ trợ code, viết test/script, gánh ~30% checklist subtasks (2/6 mục) và đóng vai trò Peer Reviewer cho PR.
* **Cân bằng tải:** Mỗi thành viên trong tuần luôn có: `70% (Lead task của mình) + 30% (Partner task của bạn) = 100% effort/tuần`.
* **Ma trận luân phiên theo chu kỳ tuần (Tuần A / Tuần B):**

| Luồng công việc | Tuần A (Tuần 1: 22/09 – 28/09) | Tuần B (Tuần 2: 29/09 – 04/10) |
|---|---|---|
| **AI Systems / Engine** | **Khôi (Lead 70%)** + Dương (Partner 30%) | **Khôi (Lead 70%)** + Đại (Partner 30%) |
| **Benchmark / Harness** | **Đại (Lead 70%)** + Khôi (Partner 30%) | **Đại (Lead 70%)** + Dương (Partner 30%) |
| **Data / Pipeline VAT** | **Dương (Lead 70%)** + Đại (Partner 30%) | **Dương (Lead 70%)** + Khôi (Partner 30%) |

### 3.3. Hai giao diện đóng băng (Frozen Interfaces)
Hai giao diện này được chốt cứng làm "hợp đồng kỹ thuật", giúp người Partner (30%) tự tin tham gia đóng góp code vào luồng của Lead mà không lo xung đột hay làm vỡ cấu trúc:
1. **Giao diện A (Engine ↔ Đo lường):** Khôi cung cấp log JSON telemetry chứa các thông số: `acceptance_rate`, `tree_budgets`, `tpot_ms`, `vram_peak_mb`. Đại đọc cấu trúc JSON này để phân tích và sweep tự động.
2. **Giao diện B (Engine ↔ Pipeline ứng dụng):** Pipeline của Dương giao tiếp với server của Khôi thuần túy qua giao thức HTTP chuẩn OpenAI (`POST /v1/chat/completions`).

---

## 4. Quản lý công việc trên Bảng Kanban

Nhóm theo dõi tiến độ trên bảng Trello với luồng chuyển trạng thái thẻ:
```text
Backlog  →  To-Do (Tuần)  →  In Progress (Tối đa 1 việc/người)  →  Blocked  →  Done
```

### Tiêu chí Hoàn thành (Definition of Done)
Một công việc chỉ được kéo sang cột **Done** khi đáp ứng đủ 3 điều kiện:
1. ✅ **Mã nguồn chạy được:** Đã merge vào nhánh `main` và chạy thử không phát sinh lỗi.
2. ✅ **Có số liệu đo lường:** Đã lưu lại log hoặc kết quả chạy vào thư mục `experiments/` (đối với các task về hiệu năng).
3. ✅ **Có nội dung viết vào quyển:** Đã viết tối thiểu 1–2 đoạn văn giải thích kết quả hoặc thiết kế đó vào bản thảo quyển báo cáo.

> 📌 **Nguyên tắc:** *"Việc gì làm xong mà chưa viết vào bản thảo quyển báo cáo thì coi như chưa xong."* Làm đến đâu viết tài liệu đến đó để tránh áp lực viết dồn vào cuối kỳ.

---

## 5. Phân tầng phạm vi công việc (Nguyên tắc MoSCoW)

Chủ động phân loại ngay từ đầu để biết cái gì bắt buộc phải có, cái gì có thể cắt giảm nếu thiếu thời gian:

### MUST — Bắt buộc phải có để đủ điều kiện bảo vệ (Chiếm ~55% công sức)
* Mô hình đích (Qwen2.5-7B 4-bit) chạy ổn định trên GPU T4 kèm cơ chế EAGLE-2 mặc định của vLLM để làm chuẩn so sánh (baseline).
* Bộ công cụ đo tự động (Benchmark Harness) thu thập đầy đủ: Acceptance rate $\alpha$, TPOT, TTFT, VRAM trên thang tải concurrency ($1 \rightarrow 32$).
* Bộ điều khiển cây nháp thích ứng (Adaptive Proposer): Tự động điều chỉnh cây nháp theo $\alpha$ và bộ nhớ VRAM còn trống.
* Thử nghiệm xác minh không giảm chất lượng (Lossless Verification): So khớp 100% từng ký tự với cách sinh truyền thống ở `temperature = 0`.
* Bảng so sánh 3 làn trên cùng phần cứng: `Vanilla` · `Static EAGLE-2` · `Adaptive`.
* Quyển báo cáo đầy đủ 6 chương theo chuẩn PTIT và slide thuyết trình.

### SHOULD — Giúp đồ án đạt điểm giỏi và có đóng góp rõ ràng (~30% công sức)
* Module đo phần cứng lúc khởi động (Hardware Profiler) để nạp bảng chi phí thực tế vào thuật toán chọn cây.
* Bản đồ trường hợp sử dụng (Use-case Map): Chỉ rõ những tình huống nào nên bật tối ưu và tình huống nào nên tắt để tránh giảm tốc độ.
* Pipeline xử lý hóa đơn tiếng Việt (MC-OCR) tối thiểu 2 node (Trích xuất JSON và Text-to-SQL) để kiểm thử luồng thực tế.

### COULD — Chỉ làm nếu hoàn thành MUST và SHOULD trước ngày 23/10 (~15% công sức)
* **[ADDITIONAL WORK] Huấn luyện Draft Head chuẩn EAGLE-3 cho Qwen 3.5 9B (Domain-Adapted Head):** Mượn GPU A100 (Google Colab Pro / RunPod) huấn luyện 2–3 giờ trên tập dữ liệu MC-OCR / chuẩn EAGLE-3. Chạy đo đạc đối chứng (Ablation Study) để chứng minh khả năng tổng quát hóa của thuật toán trên thế hệ mô hình mới nhất Qwen 3.5 9B.
* Mở rộng thêm Node 3 (Tự động sinh báo cáo phân tích tài chính) kèm giao diện web demo trực quan (Streamlit/Gradio).
* Tích hợp thử nghiệm với nền tảng Dify hoặc LangGraph.
* Mượn thêm GPU cao cấp (như A100) chạy thử nghiệm đối chứng để chứng minh khả năng tự thích ứng trên nhiều loại phần cứng khác nhau.

### WON'T — Dứt khoát không làm (Tránh mất thời gian và lạc hướng)
* ❌ **Không tự huấn luyện (train/fine-tune) lại mô hình đích (Target Model 7B/9B):** Chỉ huấn luyện đầu nháp phụ trợ (Draft Head) ở tầng COULD trong Sprint 5 mở rộng, tuyệt đối không sa đà vào việc pretrain/finetune full weights mô hình đích gây quá tải tài nguyên.
* ❌ **Không tự viết ứng dụng quản lý mô hình (Launcher/Web UI phức tạp):** Tận dụng các công cụ có sẵn.

---

## 6. Lộ trình 6 Mốc phát triển (Milestones)

Mỗi mốc thời gian tập trung kiểm chứng và giải quyết dứt điểm một câu hỏi kỹ thuật quan trọng nhất (Đã cập nhật theo lịch trình ép tiến độ tối ưu):

| Mốc | Thời hạn | Mục tiêu kỹ thuật cốt lõi | Đầu ra tối thiểu cần đạt |
|---|---|---|---|
| **M0 · PoC Nền tảng** | **20/09** | *Đề tài có thực sự chạy được trên GPU T4 16GB không?* | [ĐÃ HOÀN THÀNH] vLLM và Qwen chạy được trên T4; chạy thử được EAGLE-2 mặc định; chốt xong 2 giao diện đóng băng. |
| **M1 · Baseline Benchmark** | **04/10** | *Hệ thống đo lường có đáng tin cậy không? Chuẩn so sánh ban đầu thế nào?* | [ĐÃ HOÀN THÀNH] Bộ đo tự động hoàn thành; có bảng số liệu đo 2 làn (vanilla & static) trên concurrency 1 $\rightarrow$ 32. |
| **M2 · Adaptive v0 & Lossless** | **14/10** *(Ép sớm)* | *Việc can thiệp điều khiển cây nháp lúc đang chạy có thực hiện được không?* | Bộ điều khiển co/giãn cây theo tỉ lệ đoán trúng $\alpha$; script test chứng minh lossless khớp 100% từng ký tự; sweep 54 ô đối chứng M2. |
| **M3 · Hardware-Aware v1** | **23/10** *(Ép sớm)* | *Đưa bảng chi phí đo từ phần cứng vào thuật toán có giúp cải thiện tốc độ không?* | Hoàn thành module profiler phần cứng lúc bật máy; chạy đối chứng hiệu năng; vượt qua Cổng kiểm soát 23/10; bản thảo quyển $\ge 20$ trang. |
| **M4 · Train 9B & Đóng băng** | **15/11** *(Mở rộng)* | *Dữ liệu thực nghiệm đã đủ độ dày dặn và tin cậy để bảo vệ chưa?* | **Train xong EAGLE-3 Draft Head cho Qwen 3.5 9B trên A100**; xuất biểu đồ Ablation Study; tích hợp E2E Pipeline; **ĐÓNG BĂNG TOÀN BỘ CODE & SỐ LIỆU (15/11)**; quyển $\ge 35$ trang. |
| **M5 · Quyển & Bảo vệ** | **30/11** | *Sản phẩm báo cáo và slide đã chỉn chu chưa?* | Hoàn thiện quyển báo cáo $\ge 50$ trang nộp trường; hoàn thành slide thuyết trình; tổng duyệt thử trả lời 20+ câu hỏi phản biện. |

### Kế hoạch theo từng tuần (Timesheet Mới)
```text
Tuần 1  (14–20/09):   ▲ M0: PoC Nền tảng · Chốt 2 giao diện · Viết nháp Chương 1
Tuần 2  (21–27/09):   Xây dựng harness benchmark tự động · Đo thử làn vanilla
Tuần 3  (28/09–04/10): ▲ M1: Baseline Benchmark · Đo làn EAGLE-2 tĩnh · Quét concurrency
Tuần 4  (05–14/10):   ▲ M2: ADAPTIVE v0 & LOSSLESS · Ép tiến độ (Profiler T4, Lossless 100%, Sweep 3 làn M2)
Tuần 5  (15–23/10):   ▲ M3: HARDWARE-AWARE v1 & CỔNG SINH TỬ 23/10 (Cost Model, Sweep M3, Quyển >= 20 trang)
Tuần 6  (24–31/10):   🚀 SPRINT 5 (Mở rộng P1): Train EAGLE-3 Draft Head cho Qwen 3.5 9B trên A100
Tuần 7  (01–07/11):   🚀 SPRINT 5 (Mở rộng P2): Benchmark Ablation Study Qwen 3.5 9B & Bản đồ Use-case
Tuần 8  (08–15/11):   ▲ M4: ĐÓNG BĂNG TOÀN BỘ CODE & SỐ LIỆU (15/11 FREEZE) · E2E Pipeline · Quyển >= 35 trang
Tuần 9  (16–22/11):   Ráp và hoàn thiện quyển báo cáo >= 50 trang chuẩn PTIT
Tuần 10 (23–29/11):   ▲ M5: Nộp quyển chính thức · Thiết kế slide bảo vệ chuyên nghiệp
Tuần 11 (30/11–06/12): Tập dượt thuyết trình · Chuẩn bị 20+ câu hỏi phản biện hội đồng
Tuần 12 (07–13/12):   Tuần đệm dự phòng / Chờ ngày bảo vệ chính thức
```

### Ba cổng đánh giá & Điều chỉnh phạm vi (Checkpoints)
1. **Ngày 04/10:** [ĐÃ QUA] Nếu chưa có kết quả đo ổn định của 2 làn cơ sở → Cắt bỏ toàn bộ các mục ở tầng COULD, thu nhỏ pipeline hóa đơn về 1 node duy nhất.
2. **Ngày 23/10 (Cổng kiểm soát sống còn - Sớm 9 ngày):** Nếu phương pháp Adaptive không giúp tăng tốc hơn so với cấu hình tĩnh → **Chủ động chuyển hướng góc kể chuyện của đồ án**: Đề tài trở thành *"Nghiên cứu thực nghiệm chuyên sâu về các ranh giới suy hao và giới hạn vật lý của giải mã suy đoán trên GPU phổ thông"*. Nếu vượt qua thành công, lập tức kích hoạt task huấn luyện Qwen 3.5 9B ở Sprint 5.
3. **Ngày 15/11:** Bất kể tính năng nào chưa xong đều phải dừng lại để đóng băng mã nguồn; các tính năng dở dang sẽ được đưa vào phần "Hạn chế & Hướng phát triển trong tương lai".

---

## 7. Bảng Quản trị Rủi ro & Phương án Dự phòng

| Mã | Rủi ro tiềm ẩn | Khả năng | Mức ảnh hưởng | Phương án xử lý |
|---|---|:--:|:--:|---|
| **R1** | vLLM không hỗ trợ đổi kích thước cây nháp theo từng bước sinh | Trung bình | Rất cao | Kiểm tra ngay trong Tuần 1 (M0). Nếu không hỗ trợ trực tiếp, nhóm sẽ fork nhẹ vLLM hoặc chuyển sang vòng lặp suy luận độc lập. |
| **R2** | GPU T4 16GB bị tràn bộ nhớ (OOM) khi chạy tải lớn | Trung bình | Cao | Dùng mô hình lượng tử hóa 4-bit (AWQ/GPTQ) để giải phóng VRAM cho KV cache. Hạ trần concurrency nếu cần và nêu rõ giới hạn phần cứng trong báo cáo. |
| **R3** | Môi trường máy chủ (Colab/Kaggle) bị nhiễu thời gian đo | Cao | Cao | Bỏ qua 3 request đầu (warm-up), lặp lại ít nhất 5 lần mỗi cấu hình và lấy giá trị trung vị (median). Đan xen các lần chạy giữa các làn để triệt tiêu độ trôi phần cứng. |
| **R4** | Thuật toán Adaptive không chạy nhanh hơn bản mặc định | Trung bình | Cao | Sử dụng phương án tại Cổng kiểm soát 23/10: Chuyển hướng thành đề tài nghiên cứu thực nghiệm đánh giá ranh giới hiệu quả. |
| **R5** | Một thành viên bận việc riêng hoặc thi cử trong 1–2 tuần | Cao | Trung bình | Nhờ cơ chế Ghép cặp Luân phiên 70/30 và 2 giao diện đóng băng, người Partner trong cặp của tuần đó có thể gánh tiếp phần việc mà không làm nghẽn tiến độ chung. |
| **R6** | Xung đột mã nguồn khi làm việc chung theo cặp | Trung bình | Trung bình | Luôn làm việc trên nhánh tính năng riêng (`feat/...`), tuân thủ chuẩn giao tiếp module; Partner review chéo PR trước khi Khôi merge vào nhánh `main`. |
| **R7** | Nhà trường thông báo lịch bảo vệ sớm hơn dự kiến | Trung bình | Cao | Toàn bộ kế hoạch đã được thiết kế để đóng băng vào 15/11, tạo sẵn 2–3 tuần đệm an toàn trước ngày bảo vệ. |

---

## 8. Thước đo Tiến độ Thực tế

Nhóm không dùng các chỉ số lý thuyết như điểm story points hay số lượng commit. Tiến độ được đánh giá bằng **3 con số cụ thể**:
1. **Số rủi ro kỹ thuật đã được gỡ bỏ:** Đã kiểm chứng xong bao nhiêu câu hỏi chặn (M-01, M-02, M-03...).
2. **Số ô dữ liệu đã đo được:** Tiến độ điền đầy ma trận kết quả thí nghiệm (60 ô kết hợp giữa 3 làn, 4 bài test và 5 mức tải).
3. **Số trang bản thảo quyển báo cáo:** Đến ngày **23/10**, nếu bản thảo quyển chưa đạt tối thiểu **20 trang** thì nhóm phải dừng việc phát triển tính năng mới để tập trung viết báo cáo.

---

## 9. Tóm tắt cốt lõi trong một câu

> **Quản lý theo bảng việc trực quan Kanban, ưu tiên xử lý rủi ro kỹ thuật sớm, họp ngắn gọn 45 phút tối thứ Bảy, làm đến đâu viết báo cáo đến đó, và dứt khoát đóng băng toàn bộ mã nguồn vào ngày 15/11.**
