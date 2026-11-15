# CHƯƠNG 1: MỞ ĐẦU

> Bản thảo M3-06 (05/11/2026). Chuyển thể từ [01-gioi-thieu.md](01-gioi-thieu.md) theo dàn ý [08 §Chương 1](08-outline-quyen.md); các quyết định thiết kế dẫn chiếu [03-quyet-dinh-thiet-ke.md](03-quyet-dinh-thiet-ke.md), thuật ngữ theo [07-thuat-ngu.md](07-thuat-ngu.md). Người viết chính: Dương; hỗ trợ: Đại.

**Tên đề tài.** *FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs*.

**Tên tiếng Việt:** *FlexSDS: Hệ thống phục vụ giải mã suy đoán EAGLE thích ứng theo phần cứng cho mô hình ngôn ngữ lớn*.

> Đồng bộ tên ngày 14/11/2026; nội dung chương là bản thảo nghiên cứu. Trạng thái triển khai xem [README](../README.md).

---

## 1.1. Bối cảnh và động lực

Trong vài năm gần đây, mô hình ngôn ngữ lớn (Large Language Model — LLM) đã chuyển từ một đối tượng nghiên cứu thành một thành phần hạ tầng phổ biến của phần mềm. Một hình thức sử dụng ngày càng phổ biến là **tự động hoá các tác vụ lặp lại** thông qua các nền tảng điều phối (orchestration) như Dify hay LangGraph: người dùng ghép nhiều lời gọi LLM thành một quy trình, trong đó đầu ra của bước trước trở thành đầu vào của bước sau. Một quy trình xử lý chứng từ điển hình có thể gồm bước trích xuất thông tin, bước truy vấn dữ liệu và bước viết báo cáo, mỗi bước là một lời gọi mô hình riêng.

Khi triển khai các quy trình như vậy, người dùng đứng trước hai lựa chọn. Lựa chọn thứ nhất là dùng API thương mại; cách này đơn giản nhưng **tốn phí theo lượng token** và trở nên đắt đỏ khi quy trình chạy thường xuyên, đồng thời buộc dữ liệu đi qua hạ tầng của bên thứ ba. Nhiều tình huống thực tế — dữ liệu nội bộ, chứng từ tài chính, môi trường không có Internet, hoặc yêu cầu không phụ thuộc nhà cung cấp — khiến lựa chọn này không khả thi. Lựa chọn thứ hai là **tự host (self-host) một mô hình mã nguồn mở trên phần cứng khiêm tốn**, thường là một GPU phổ thông với 16–24 GB bộ nhớ.

Một câu hỏi tự nhiên là: khi các mô hình đóng đã rất mạnh, vì sao vẫn cần tối ưu suy luận cho mô hình tự host? Câu trả lời có hai vế. Thứ nhất, các hệ thống phục vụ LLM quy mô lớn cũng được xây dựng trên chính những kỹ thuật tối ưu suy luận này; đây là bài toán hạ tầng lõi chứ không phải sự cạnh tranh về chất lượng với mô hình đóng. Thứ hai, khi tự host, "chi phí mỗi token" không biến mất mà **chuyển hoá** thành thời gian GPU, dung lượng VRAM và độ trễ mà người dùng phải chờ.

Trong các trục tối ưu, **độ trễ giải mã (decode latency)** là nút thắt quan trọng nhất đối với trải nghiệm tương tác. Quá trình sinh văn bản của LLM là tự hồi quy: mỗi token mới đòi hỏi một lượt truyền xuôi qua toàn bộ mô hình, và ở batch nhỏ, lượt truyền này bị giới hạn bởi băng thông bộ nhớ hơn là bởi năng lực tính toán. Vấn đề càng rõ với **các quy trình agent gọi LLM nhiều lượt tuần tự**: độ trễ của từng bước **cộng dồn** thành độ trễ của cả quy trình. Một quy trình ba bước, mỗi bước chờ vài giây, nhanh chóng trở thành trải nghiệm chờ đợi hàng chục giây. Giảm độ trễ của từng lời gọi vì thế có tác động trực tiếp và nhân lên ở cấp quy trình.

## 1.2. Phát biểu bài toán

**Giải mã suy đoán (speculative decoding)** là một hướng tăng tốc nổi bật vì nó **không làm thay đổi phân phối đầu ra (lossless)**. Ý tưởng cốt lõi là dùng một cơ chế nháp rẻ — một mô hình nhỏ hoặc một đầu nháp gắn vào mô hình đích — để đề xuất trước nhiều token, sau đó mô hình đích xác thực toàn bộ các token đề xuất trong **một** lượt truyền xuôi. Những token được chấp nhận theo tiêu chí chấp nhận của thuật toán được giữ lại; token đầu tiên bị từ chối được thay bằng token do mô hình đích sinh ra. Vì tiêu chí chấp nhận được thiết kế để giữ nguyên phân phối của mô hình đích, chất lượng đầu ra không đổi trong khi số lượt truyền xuôi đắt đỏ giảm đi. Họ phương pháp EAGLE, đặc biệt là **EAGLE-2**, mở rộng ý tưởng này bằng một *cây nháp động* mà hình dạng được điều chỉnh theo độ tự tin của bộ nháp (Chương 2).

Tuy nhiên, EAGLE-2 và các phương pháp cùng dòng chủ yếu được thiết kế và kiểm chứng trên **GPU cao cấp có băng thông rất lớn** (A100, H800, RTX 3090). Khi chuyển sang GPU phổ thông, nhóm nhận diện ba chế độ hỏng:

**(a) Tăng tốc âm (negative speedup).** Lợi ích của việc xác thực song song dựa trên giả định GPU đang nghẽn băng thông, nên phần năng lực tính toán thừa có thể dùng "miễn phí" để xác thực nhiều token. Giả định này yếu đi khi tỉ lệ tính toán/băng thông của GPU khác đi, khi mô hình đích được lượng tử hoá (mô hình đích chạy nhanh hơn nên biên lợi ích thu hẹp), hoặc khi nhiều request chạy đồng thời. Khi đó, xác thực một cây nháp lớn trở thành chi phí thuần và hệ thống chạy **chậm hơn cả giải mã thông thường**. Hiện tượng này đã được đo trực tiếp trong công trình D-cut, với tốc độ tụt xuống 0,67× trên Llama-3.1-8B ở mức đồng thời 32; ở mốc M1, nhóm cũng quan sát được tăng tốc âm trên workload văn bản tự do (Chương 4, §4.5.2).

**(b) Cạn bộ nhớ (OOM).** Cây nháp, KV cache của các nhánh nháp và KV cache của các request đồng thời cùng tranh một lượng VRAM rất hạn chế. Một cấu hình cây tĩnh không có cơ chế co lại khi bộ nhớ eo hẹp, nên hệ thống có thể sập thay vì chỉ chạy chậm đi.

**(c) Thiếu hướng dẫn bật/tắt.** Lợi ích của giải mã suy đoán phụ thuộc mạnh vào loại workload: nội dung có cấu trúc lặp (JSON, SQL) dễ đoán hơn nhiều so với văn xuôi tự do. Hiện chưa có một đặc tả rõ ràng về *quy trình nào trên phần cứng phổ thông thực sự được hưởng lợi*, nên người dùng phải tự thử nghiệm.

Điểm chung của cả ba chế độ hỏng là **tham số nháp được đặt cố định lúc khởi động, trong khi điều kiện thực tế thay đổi theo từng bước giải mã, theo từng workload và theo từng GPU**. Bài toán của đề tài vì thế được phát biểu như sau: *xây dựng một cơ chế quyết định ngân sách nháp lúc chạy, dựa trên chi phí đo thực trên đúng GPU đang phục vụ và trên telemetry thời gian thực, sao cho giữ được tính lossless, tận dụng được lợi ích khi có lợi, và tự lui về giải mã thường khi tăng tốc không còn có lời hoặc khi bộ nhớ eo hẹp.*

## 1.3. Câu hỏi nghiên cứu

Đề tài trả lời ba câu hỏi nghiên cứu:

- **RQ1.** Việc để quyết định nháp **thích ứng theo đặc tính phần cứng đo được lúc chạy** (độ trễ nháp và xác thực, VRAM còn trống, tỉ lệ chấp nhận α) có cải thiện tốc độ giải mã so với cấu hình nháp tĩnh trên GPU phổ thông không, trong khi vẫn giữ lossless?
- **RQ2.** Cơ chế thích ứng có **loại bỏ được chế độ tăng tốc âm và chế độ OOM** hay không — tức là có bảo đảm tốc độ *không tệ hơn* giải mã thông thường không?
- **RQ3.** Trên một quy trình tự động hoá thực tế, **loại workload nào thực sự hưởng lợi** từ tối ưu tốc độ, và hưởng lợi đến mức nào?

RQ2 có một vị trí đặc biệt: nó vẫn trả lời được ngay cả khi RQ1 cho kết quả âm tính. Nếu bộ điều khiển thích ứng không nhanh hơn cấu hình tĩnh tốt nhất nhưng loại bỏ được các chế độ hỏng, đó vẫn là một đóng góp về **độ tin cậy hệ thống** — điều mà người vận hành một hệ thống tự host quan tâm không kém tốc độ đỉnh. RQ3 gắn kết quả đo với quyết định thực tế của người dùng: khi nào nên bật, khi nào nên tắt.

## 1.4. Mục tiêu, phạm vi và ngoài phạm vi

### Mục tiêu

- **G1.** Thiết lập các làn đối chứng chạy được trên GPU phổ thông: làn giải mã thường (vanilla) và làn giải mã suy đoán cấu hình tĩnh (static), trên engine phục vụ vLLM.
- **G2.** Xây dựng **Adaptive Proposer** — bộ điều khiển chỉnh ngân sách nháp lúc chạy từ hồ sơ phần cứng đo thực và telemetry (α, VRAM, mức sử dụng KV cache), có sàn an toàn tắt hẳn suy đoán (QĐ-13).
- **G3.** Đóng gói thành **endpoint tương thích OpenAI** (`/v1/chat/completions`) để cắm thẳng vào các nền tảng điều phối mà không phải sửa ứng dụng (QĐ-06).
- **G4.** **Đánh giá end-to-end** trên pipeline phân tích hoá đơn tiếng Việt, so sánh các làn trên thang mức đồng thời (concurrency), và rút ra **bản đồ use-case**.

### Phạm vi

- **Suy luận đơn GPU**, trọng tâm là chế độ phục vụ độ trễ thấp ở batch nhỏ. Thang concurrency (1 → 32) được quét để xác định ranh giới nơi tối ưu hết tác dụng, không nhằm tối ưu thông lượng lớn.
- **Mô hình đích** Qwen2.5-7B-Instruct lượng tử hoá 4-bit (AWQ) (QĐ-09). Phần cứng mục tiêu ban đầu là NVIDIA T4 16 GB (QĐ-02, QĐ-09); từ Sprint 4, các phép đo mở rộng chạy trên NVIDIA L4 24 GB qua nền tảng serverless Modal (Chương 4, §4.1.4).
- **Mô hình nháp.** Không tự huấn luyện đầu nháp trên đường găng (QĐ-10). Theo chiến lược hai luồng (QĐ-12), luồng chính dùng mô hình nháp độc lập Qwen2.5-0.5B-Instruct; luồng mở rộng dùng một đầu nháp EAGLE-3.1 làm bài đối chứng nâng cao, chạy ngoài đường găng. Do vLLM 0.30.0 chưa cho đổi cấu trúc cây theo từng bước, cây nháp được "ép phẳng" thành chuỗi đơn nhánh và ngân sách nháp tương đương độ sâu k ∈ {0, 2, 3, 4} (QĐ-12).
- **Giữ nguyên tiêu chí chấp nhận** của giải mã suy đoán, nên tính lossless được bảo toàn theo thiết kế; đề tài vẫn chạy một phép xác minh thực nghiệm so khớp đầu ra giữa các làn (QĐ-11).

### Ngoài phạm vi

Để tránh hiểu nhầm khi đánh giá, nhóm nói rõ những điều đề tài **không** làm:

- **Không cải thiện chất lượng đầu ra.** Đề tài là lossless: về lý thuyết, đầu ra của làn tăng tốc có cùng phân phối với làn thường. Chất lượng nghiệp vụ của pipeline hoá đơn (ví dụ độ đúng của câu SQL) là giới hạn của mô hình đích, được ghi nhận nhưng không phải mục tiêu tối ưu.
- **Không huấn luyện hay tinh chỉnh mô hình đích.**
- **Không nhằm vượt EAGLE-2 trên mọi cấu hình** — chỉ cần thắng hoặc ngang trong chế độ GPU phổ thông, kèm đặc tả rõ chế độ nào (QĐ-04).
- **Không xây ứng dụng quản lý mô hình** đa dụng (QĐ-05).
- **Không gắn đề tài vào một domain ứng dụng.** Pipeline hoá đơn chỉ là dụng cụ đo (QĐ-01, QĐ-08).

## 1.5. Hướng tiếp cận và đóng góp

### Hướng tiếp cận

Hướng tiếp cận gồm ba tầng thông tin. **Lúc khởi động**, hệ thống đo hồ sơ phần cứng: thời gian nháp và thời gian xác thực theo độ sâu nháp, cùng dung lượng VRAM khả dụng. **Lúc phục vụ**, hệ thống thu telemetry thời gian thực: tỉ lệ chấp nhận α trượt theo thời gian, mức sử dụng KV cache và kích thước batch hiện tại. **Ở mỗi bước giải mã**, bộ điều khiển chọn ngân sách nháp tối đa hoá tỉ số giữa lợi ích kỳ vọng và **chi phí đo thật trên GPU đang chạy**, trong đó luôn có lựa chọn "tắt hẳn suy đoán" (k = 0). Điểm phân biệt đề tài với các cây động theo độ tự tin (đóng góp sẵn có của EAGLE-2 và Talon) là mẫu số của phép tối ưu: chi phí đo trên phần cứng thực, không phải hằng số lý thuyết. Khuôn mẫu "lợi ích/chi phí đo thật" lấy cảm hứng từ D-cut, nhưng D-cut nhắm chế độ batch lớn trên GPU trung tâm dữ liệu, còn đề tài nhắm batch nhỏ trên GPU phổ thông — một chế độ chưa được phủ.

### Workload đánh giá

Đánh giá chạy trên một quy trình thật thay vì các microbenchmark tách rời. Pipeline hoá đơn tiếng Việt trên tập MC-OCR gồm ba node: **Node 1** trích xuất văn bản hoá đơn thành JSON có cấu trúc rồi nạp vào SQLite; **Node 2** chuyển câu hỏi nghiệp vụ tiếng Việt thành câu SQL chỉ đọc; **Node 3** viết báo cáo tổng hợp từ kết quả truy vấn. Pipeline được chọn vì nó cho **ba chế độ entropy tương phản trong cùng một quy trình**: JSON lặp khuôn (entropy thấp — chế độ thuận lợi nhất cho suy đoán), SQL có cú pháp ràng buộc (entropy trung bình) và văn xuôi tự do (entropy cao — nơi tăng tốc âm dễ xuất hiện và bộ điều khiển phải biết tắt). Độ trễ đầu-cuối bằng tổng độ trễ ba node, minh hoạ trực tiếp hiệu ứng cộng dồn ở §1.1. Ba node tương ứng với workload W1, W2, W3; khi nối liền, chúng tạo thành W4 (Chương 4, §4.2).

### Đóng góp

1. **Về phương pháp — bộ điều khiển nháp thích ứng phần cứng.** Đưa chi phí đo thực trên GPU đích (không phải hằng số lý thuyết, không chỉ độ tự tin của bộ nháp) vào quyết định ngân sách nháp, giữ nguyên tính lossless.
2. **Về độ tin cậy hệ thống — loại bỏ hai chế độ hỏng.** Bộ điều khiển có sàn an toàn: khi tăng tốc không còn có lợi hoặc khi bộ nhớ eo hẹp, nó thu nhỏ ngân sách hoặc tắt hẳn suy đoán, nhằm bảo đảm hệ thống không chậm hơn giải mã thường và không OOM.
3. **Về đánh giá — phương pháp và bộ số liệu end-to-end theo thang concurrency** trên một quy trình thật, thay vì một con số tăng tốc đo ở batch bằng 1 (điều kiện dễ nhất).
4. **Về hiểu biết — đặc tả chế độ sử dụng (use-case regime)** kèm khung quyết định, bao gồm cả những trường hợp **không nên** bật tối ưu.

Mức độ đạt được của từng đóng góp được đánh giá bằng số liệu ở Chương 5; những đóng góp chưa được kiểm chứng đầy đủ sẽ được nêu rõ trong phần hạn chế ở Chương 6.

## 1.6. Bố cục quyển

Phần còn lại của quyển được tổ chức như sau:

- **Chương 2 — Cơ sở lý thuyết và tổng quan.** Trình bày nút thắt của suy luận LLM (giới hạn băng thông so với giới hạn tính toán), cơ chế giải mã suy đoán và chứng minh tính lossless, họ phương pháp EAGLE và cây nháp động, các trục tối ưu liên quan (PEARL, D-cut), các hướng đã cân nhắc và loại bỏ, và khoảng trống nghiên cứu mà đề tài nhắm tới.
- **Chương 3 — Thiết kế hệ thống.** Mô tả kiến trúc tổng thể, cách đo hồ sơ phần cứng, bộ điều khiển ngân sách nháp và các cơ chế an toàn, cách bảo đảm lossless, việc tích hợp vào vLLM và endpoint tương thích OpenAI, cùng các quyết định thiết kế quan trọng và lý do.
- **Chương 4 — Thực nghiệm.** Trình bày môi trường phần cứng và phần mềm, mô hình, bộ workload W1–W4 và tập dữ liệu, các chỉ số và giao thức đo, kết quả đối chứng cơ sở và phép xác minh lossless.
- **Chương 5 — Kết quả và bàn luận.** Trả lời RQ1–RQ3: so sánh làn thích ứng với làn tĩnh, phân tích tăng tốc âm và an toàn bộ nhớ, bằng chứng về sự thích ứng theo thời gian, và bản đồ use-case.
- **Chương 6 — Kết luận và hướng phát triển.** Tóm tắt đóng góp, nêu các hạn chế và đề xuất hướng phát triển.

Cuối quyển là danh mục tài liệu tham khảo và phụ lục, gồm bảng kết quả đầy đủ và hướng dẫn tái lập thực nghiệm.
