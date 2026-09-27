# CHƯƠNG 1. GIỚI THIỆU

> **Bản nháp phục vụ nghiệm thu M0-04**<br>
> Đề tài: *Giải mã suy đoán thích ứng phần cứng cho suy luận LLM tự-host trên GPU phổ thông*<br>
> Nhóm thực hiện: Đồ án tốt nghiệp D22 PTIT
> Ngày cập nhật: 20/09/2026

## 1.1. Đặt vấn đề

Trong vài năm gần đây, mô hình ngôn ngữ lớn (Large Language Model - LLM) đã trở thành một thành phần quan trọng trong nhiều hệ thống phần mềm có khả năng tự động hóa. LLM được sử dụng để trả lời câu hỏi, tóm tắt tài liệu, trích xuất thông tin, sinh mã nguồn, phân tích dữ liệu và điều khiển các tác vụ nhiều bước. Khi được kết hợp với các nền tảng điều phối như Dify hoặc LangGraph, một yêu cầu của người dùng thường không còn tương ứng với một lần gọi mô hình đơn lẻ mà trở thành một workflow gồm nhiều node LLM liên tiếp.

Sự phát triển này tạo ra nhu cầu triển khai mô hình theo hướng tự chủ hơn. Việc sử dụng API thương mại đem lại chất lượng và khả năng mở rộng tốt, nhưng đi kèm chi phí theo số token, sự phụ thuộc vào nhà cung cấp và các yêu cầu về truyền dữ liệu ra bên ngoài. Với những bài toán chứa dữ liệu nội bộ, dữ liệu tài chính hoặc dữ liệu hóa đơn, tổ chức có thể cần một hệ thống chạy trong mạng riêng, có thể kiểm soát mô hình, dữ liệu và chi phí vận hành. Khi đó, tự host LLM trên GPU phổ thông là một hướng tiếp cận thực tế.

Tuy nhiên, tự host một LLM có kích thước lớn trên phần cứng giới hạn đặt ra bài toán về độ trễ. Độ trễ không chỉ ảnh hưởng đến trải nghiệm của một lời gọi; trong một workflow nhiều bước, thời gian của từng node được cộng dồn. Một pipeline gồm ba node tuần tự có thể phải chờ kết quả của node trước trước khi bắt đầu node sau. Vì vậy, một cải thiện nhỏ ở thời gian sinh của mỗi node có thể tạo ra tác động đáng kể lên độ trễ đầu cuối. Ngược lại, một cơ chế tối ưu thiếu ổn định có thể làm tăng thời gian xử lý hoặc làm hệ thống hết bộ nhớ khi nhiều yêu cầu chạy đồng thời.

Đề tài này tập trung vào kỹ thuật **giải mã suy đoán** (speculative decoding), trong đó một mô hình nháp nhẹ đề xuất trước một số token và mô hình đích xác thực các token đó theo lô. Nếu các token được chấp nhận với tỷ lệ cao, mô hình đích có thể sinh được nhiều token trong một lượt xác thực thay vì phải thực hiện một lượt cho từng token. Về lý thuyết, kỹ thuật này có thể giảm độ trễ mà vẫn giữ nguyên phân phối đầu ra của mô hình đích. Các phương pháp như EAGLE-2 đã cho thấy hiệu quả của cây nháp động trên một số phần cứng và workload.

Vấn đề của nghiên cứu này là hiệu quả của giải mã suy đoán không cố định. Nó phụ thuộc vào mô hình, checkpoint nháp, phần cứng, dung lượng VRAM còn lại, mức độ đồng thời và đặc tính của workload. Một cấu hình cây nháp phù hợp ở concurrency thấp có thể trở thành chi phí thuần khi concurrency tăng. Với GPU phổ thông, khoảng cách giữa năng lực tính toán, băng thông bộ nhớ và dung lượng VRAM có thể làm cho cấu hình tối ưu thay đổi theo thời điểm chạy. Do đó, việc đặt cố định ngân sách cây nháp ngay lúc khởi động có nguy cơ tạo ra ba tình huống: tốc độ chậm hơn giải mã thường, hết VRAM, hoặc không có quy tắc rõ ràng để quyết định khi nào nên bật hay tắt tối ưu.

Từ bối cảnh đó, đề tài xây dựng một bộ điều khiển **Adaptive Proposer**. Bộ điều khiển sử dụng hồ sơ phần cứng đo được lúc khởi động và telemetry trong thời gian chạy, bao gồm thời gian xử lý, tỷ lệ chấp nhận token, lượng VRAM còn trống và mức tải hiện tại, để lựa chọn ngân sách cây nháp. Hệ thống không chỉ tìm cấu hình nhanh nhất mà còn phải có khả năng giảm ngân sách hoặc tắt suy đoán khi lợi ích kỳ vọng không còn đủ bù chi phí xác thực.

## 1.2. Bối cảnh ứng dụng và lý do chọn nghiệp vụ hóa đơn

Để đánh giá một hệ thống suy luận, chỉ sử dụng một prompt độc lập là chưa đủ. Một prompt đơn thường không thể phản ánh sự khác biệt về độ khó sinh token giữa các tác vụ, cũng không phản ánh độ trễ cộng dồn của workflow. Vì vậy, đề tài sử dụng pipeline phân tích hóa đơn tiếng Việt làm workload ứng dụng. Pipeline này không phải là sản phẩm thương mại chính của đề tài mà là một quy trình đủ thực tế để tạo ra các dạng sinh văn bản có mức entropy khác nhau.

Hóa đơn bán lẻ và hóa đơn giá trị gia tăng (VAT) chứa nhiều trường thông tin có cấu trúc: tên và địa chỉ bên bán, mã số thuế, thông tin người mua, số hóa đơn, ngày lập, danh sách hàng hóa, số lượng, đơn giá, thành tiền, thuế suất và tổng tiền. Bên cạnh các trường có cấu trúc, dữ liệu đầu vào từ MC-OCR thường chứa lỗi nhận dạng, dòng xuống dòng không ổn định, ký tự tiếng Việt bị sai mã hóa và thông tin bị thiếu. Đây là môi trường phù hợp để kiểm tra khả năng tuân thủ một schema cố định của LLM.

Pipeline được chia thành ba node. Node 1 nhận văn bản OCR và sinh JSON hóa đơn. Đây là tác vụ text-to-JSON có cấu trúc lặp lại, vì vậy được xem là workload entropy thấp. Node 2 nhận câu hỏi nghiệp vụ bằng tiếng Việt và sinh truy vấn SQL chỉ đọc trên các view ổn định của SQLite. Đây là tác vụ text-to-SQL có cú pháp ràng buộc nhưng có nhiều cách diễn đạt và lựa chọn cột, nên có entropy trung bình. Node 3, nếu được triển khai đầy đủ, nhận kết quả truy vấn và sinh báo cáo hoặc diễn giải cho người dùng. Đây là tác vụ văn bản tự do, có entropy cao hơn và là nơi phù hợp để quan sát trường hợp giải mã suy đoán không đem lại lợi ích.

Trong phạm vi M0-04, Node 1 đã được hiện thực ở mức chạy được với backend Ollama thông qua giao diện OpenAI-compatible. SQLite schema và các view phục vụ Node 2 cũng đã được thiết kế. Node 2 và Node 3 là các bước tiếp theo của pipeline, không được tuyên bố là đã hoàn thiện trong bản nghiệm thu này. Cách phân chia này giúp tách rõ phần đã kiểm chứng khỏi phần còn trong kế hoạch.

## 1.3. Phát biểu bài toán

Bài toán của đề tài có thể phát biểu như sau: **làm thế nào để lựa chọn ngân sách giải mã suy đoán phù hợp với phần cứng và trạng thái tải hiện tại, nhằm giảm độ trễ sinh của LLM trên GPU phổ thông mà vẫn giữ tính lossless và tránh các trạng thái tăng tốc âm hoặc hết VRAM?**

Trong giải mã thường, mô hình đích sinh từng token theo chuỗi. Với một chuỗi đầu ra dài, mỗi bước sinh yêu cầu mô hình thực hiện một lượt xử lý và cập nhật KV cache. Giải mã suy đoán bổ sung một mô hình nháp hoặc một nhánh nháp để đề xuất nhiều token. Mô hình đích sau đó xác thực các token đề xuất. Khi tỷ lệ chấp nhận cao, số token hữu ích trên mỗi lượt xác thực tăng và độ trễ trung bình có thể giảm. Khi tỷ lệ chấp nhận thấp hoặc chi phí xác thực cây lớn, phần công việc bổ sung có thể lớn hơn phần tiết kiệm.

Hiệu quả thực tế có thể thay đổi theo những yếu tố sau:

- **Đặc tính phần cứng:** thời gian chạy mô hình nháp, thời gian xác thực, băng thông bộ nhớ và lượng VRAM khả dụng.
- **Đặc tính mô hình:** mô hình đích, mô hình nháp, kích thước checkpoint và mức lượng tử hóa.
- **Đặc tính workload:** JSON khuôn mẫu, SQL có ràng buộc cú pháp hay văn bản tự do.
- **Mức concurrency:** số request xử lý đồng thời và lượng KV cache tương ứng.
- **Trạng thái runtime:** tỷ lệ chấp nhận gần đây, độ trễ quan sát được và dao động VRAM trong quá trình phục vụ.

Một cấu hình tĩnh không phản ứng được với toàn bộ các yếu tố trên. Nếu chọn cây quá nhỏ, hệ thống không khai thác được cơ hội tăng tốc; nếu chọn cây quá lớn, hệ thống có thể tăng chi phí xác thực hoặc chiếm VRAM quá mức. Bộ điều khiển thích ứng vì vậy cần đưa lựa chọn “không dùng suy đoán” vào không gian quyết định, thay vì luôn bắt buộc phải dùng một cây nháp.

Một yêu cầu quan trọng khác là tính đúng đắn. Adaptive Proposer chỉ được thay đổi cách thực hiện giải mã, không được thay đổi phân phối đầu ra của mô hình đích. Tiêu chí lossless được kế thừa từ cơ chế xác thực của speculative decoding. Trong thực nghiệm, nhóm sẽ kiểm tra thêm sự khớp đầu ra giữa làn vanilla và làn adaptive ở nhiệt độ bằng không, nhưng không xem đây là phép đo chất lượng ngôn ngữ độc lập.

## 1.4. Câu hỏi nghiên cứu

Từ bài toán trên, đề tài đặt ra ba câu hỏi nghiên cứu.

**RQ1.** Việc lựa chọn ngân sách cây nháp thích ứng theo đặc tính phần cứng đo được và telemetry trong thời gian chạy có cải thiện tốc độ giải mã so với EAGLE-2 cấu hình tĩnh trên GPU phổ thông hay không, trong khi vẫn giữ tính lossless?

Câu hỏi này tập trung vào hiệu năng. Việc trả lời không chỉ dựa trên một giá trị speedup ở concurrency bằng một mà cần quan sát nhiều workload và nhiều mức concurrency. Các chỉ số dự kiến gồm TTFT, TPOT, ITL, độ trễ đầu cuối, throughput, tỷ lệ chấp nhận và VRAM đỉnh.

**RQ2.** Cơ chế thích ứng có loại bỏ được các trạng thái tăng tốc âm và hết VRAM hay không?

RQ2 quan trọng vì một hệ thống thực tế không chỉ cần nhanh ở điều kiện thuận lợi. Nếu adaptive không luôn nhanh hơn cấu hình tĩnh nhưng có thể tự giảm ngân sách khi tốc độ không còn lợi hoặc khi VRAM căng thẳng, hệ thống vẫn đem lại giá trị về độ tin cậy. Mục tiêu vận hành là tránh tình huống tối ưu hóa làm hệ thống tệ hơn vanilla hoặc bị dừng do OOM.

**RQ3.** Loại workload nào trong một workflow tự động hóa thực tế thực sự hưởng lợi từ giải mã suy đoán, và hưởng lợi tới mức nào?

RQ3 liên kết kết quả đo với quyết định sử dụng. JSON có cấu trúc lặp lại có thể tạo ra tỷ lệ chấp nhận cao; SQL có thể đạt mức trung gian; văn bản tự do có thể có tỷ lệ chấp nhận thấp và cần được điều khiển thận trọng. Kết quả cuối cùng cần cho biết khi nào nên bật speculative decoding, khi nào nên dùng cấu hình nhỏ và khi nào nên tắt.

## 1.5. Mục tiêu của đề tài

Đề tài hướng tới bốn mục tiêu chính.

**Mục tiêu G1 - xây dựng baseline:** Tái lập môi trường phục vụ LLM với endpoint tương thích OpenAI, sử dụng Qwen2.5-7B-Instruct lượng tử hóa trên GPU mục tiêu. Baseline vanilla là cơ sở để so sánh công bằng với cấu hình tĩnh và adaptive.

**Mục tiêu G2 - xây dựng bộ điều khiển thích ứng:** Thiết kế bộ điều khiển sử dụng cost model được đo trên GPU thực tế, kết hợp với tỷ lệ chấp nhận, VRAM còn trống và mức tải để chọn ngân sách cây nháp. Bộ điều khiển cần có cơ chế an toàn, bao gồm khả năng thu nhỏ hoặc tắt cây.

**Mục tiêu G3 - giữ nguyên giao diện tích hợp:** Đóng gói các làn phục vụ thông qua giao diện `/v1/chat/completions`. Ứng dụng gọi LLM không cần biết bên dưới đang sử dụng vanilla, static hay adaptive. Ranh giới này giúp tách phần pipeline nghiệp vụ khỏi engine suy luận và cho phép thay đổi backend giữa vLLM và Ollama trong giai đoạn kiểm thử chức năng.

**Mục tiêu G4 - đánh giá trên workload end-to-end:** Xây dựng pipeline MC-OCR gồm trích xuất JSON, truy vấn SQL và diễn giải kết quả; sau đó đo trên các workload và mức concurrency được định nghĩa trước. Kết quả cần được lưu bằng log có thể tái lập và phân tích theo protocol thống nhất.

## 1.6. Phạm vi nghiên cứu

Đề tài tập trung vào suy luận trên một GPU, ưu tiên độ trễ của request và khả năng phục vụ ổn định ở batch nhỏ đến trung bình. Các mức concurrency dự kiến gồm 1, 2, 4, 8, 16 và 32; trần thực tế có thể được hạ nếu phần cứng không đủ VRAM. Nghiên cứu không nhằm tối ưu hệ thống phân tán hoặc thay thế toàn bộ hạ tầng serving của doanh nghiệp.

Mô hình đích được chọn là Qwen2.5-7B-Instruct lượng tử hóa 4-bit, với môi trường đo chính thức dự kiến sử dụng vLLM và checkpoint phù hợp với thiết kế EAGLE-2. Ollama với bản `qwen2.5:7b-instruct` được dùng cho functional PoC trên máy cá nhân. Hai backend cùng phục vụ một họ mô hình và cùng giao diện OpenAI-compatible, nhưng kết quả Ollama không được dùng thay cho số liệu benchmark vLLM vì Ollama không cung cấp đầy đủ các lane và telemetry của đề tài.

Về dữ liệu, M0-04 sử dụng một dev-set 30 mẫu được chọn từ MC-OCR. Input của Node 1 là văn bản OCR, không phải ảnh đưa trực tiếp vào LLM thị giác. Ảnh được giữ lại để audit và đối chiếu. Nhãn annotation không được đưa vào prompt nhằm tránh rò rỉ đáp án. Node 1 phải sao chép metadata nguồn do pipeline cung cấp, trả về JSON theo schema và bị từ chối nếu source binding không khớp.

Phạm vi không bao gồm huấn luyện mô hình đích hoặc huấn luyện mô hình nháp mới. Đề tài cũng không đánh giá chất lượng nghiệp vụ hóa đơn như một bài toán OCR độc lập, không thay thế hệ thống kế toán và không cam kết dữ liệu trích xuất được dùng trực tiếp cho quyết toán thuế. Schema được thiết kế cho workload phân tích và benchmark; các trường thiếu được biểu diễn bằng `null` hoặc mảng rỗng thay vì suy đoán.

Trong M0-04, Node 1 và lớp nạp SQLite đã có runner functional. Node 2 mới dừng ở hợp đồng dữ liệu và các view đọc ổn định; Node 3 được xem là tầng có thể cắt giảm nếu không cần cho mốc đo đầu tiên. Việc ghi rõ phạm vi giúp tránh nhầm lẫn giữa một thành phần đã chạy được và một thành phần mới chỉ được thiết kế.

## 1.7. Kiến trúc workload và luồng xử lý

Luồng xử lý dự kiến của pipeline như sau:

```text
MC-OCR anno_texts
        |
        v
Node 1 Extractor: OCR text -> invoice_extraction.v1 JSON
        |
        v
SQLite user_version=1
        |
        v
Node 2 Text-to-SQL: Vietnamese question -> read-only SELECT
        |
        v
Node 3 SQL-to-BI: query result -> analytical explanation
```

Node 1 là ranh giới đầu tiên giữa dữ liệu không cấu trúc và dữ liệu quan hệ. JSON Schema quy định các nhóm `seller`, `buyer`, `invoice`, `totals`, `taxes`, `items` và `extraction`. Các trường tiền tệ giữ cả chuỗi OCR thô và giá trị số chuẩn hóa. Điều này cho phép vừa truy vấn tổng hợp trong SQLite vừa quay lại kiểm tra bằng chứng nguồn khi cần.

SQLite sử dụng bảng `invoices` cho thông tin cấp hóa đơn, `invoice_items` cho các dòng hàng, `invoice_taxes` cho các dòng thuế và `field_evidence` cho bằng chứng trích xuất. Các view `v_invoice_facts`, `v_item_facts` và `v_tax_facts` là bề mặt đọc ưu tiên cho Node 2. Node 2 chỉ được sinh câu lệnh `SELECT`, không được sửa dữ liệu và không truy vấn `raw_json` trong các câu hỏi nghiệp vụ thông thường.

Một số câu hỏi nghiệp vụ mẫu gồm: “Tổng chi tiêu theo ngày hoặc theo tháng là bao nhiêu?”, “Cửa hàng nào có tổng tiền cao nhất?”, “Mặt hàng nào xuất hiện nhiều nhất?”, “Tổng tiền VAT theo khoảng thời gian là bao nhiêu?” và “Những hóa đơn nào có tổng tiền vượt một ngưỡng nhất định?”. Các câu hỏi này giúp xác định các cột và grain cần có trong view, đồng thời tạo workload text-to-SQL có thể tái lập.

## 1.8. Ý nghĩa của đề tài

Về mặt kỹ thuật, đề tài đưa quyết định ngân sách cây nháp từ một cấu hình cố định sang một quyết định có quan sát chi phí thực tế. Cách tiếp cận này phù hợp với thực tế GPU phổ thông, nơi tài nguyên bị giới hạn và trạng thái tải thay đổi nhanh. Đóng góp không nằm ở việc tạo ra một mô hình ngôn ngữ mới mà ở lớp điều khiển và phương pháp đánh giá khả năng vận hành.

Về mặt hệ thống, đề tài nhấn mạnh rằng tăng tốc cần đi kèm điều kiện an toàn. Một hệ thống nhanh hơn ở batch bằng một nhưng bị OOM ở concurrency cao chưa phải là một hệ thống tốt cho workflow thực tế. Việc ghi nhận VRAM, tỷ lệ chấp nhận và tình huống tắt suy đoán giúp biến câu hỏi “có nhanh hơn không?” thành câu hỏi đầy đủ hơn: “nhanh hơn ở workload nào, mức tải nào, trên phần cứng nào, và có giữ được tính ổn định hay không?”.

Về mặt ứng dụng, pipeline hóa đơn cung cấp một workload dễ giải thích trước hội đồng. Người đọc có thể thấy rõ sự khác biệt giữa JSON có schema, SQL có ràng buộc và văn bản tự do. Đồng thời, SQLite và các file JSONL tạo ra một đường kiểm toán từ input, output LLM đến dữ liệu phân tích. Đây là cơ sở để tái lập thí nghiệm và để phân biệt lỗi của model, lỗi của pipeline với lỗi của engine serving.

## 1.9. Cấu trúc các chương tiếp theo

Chương 2 trình bày tổng quan các hướng nghiên cứu liên quan, gồm tối ưu KV cache, nén prompt và speculative decoding, từ đó giải thích lý do lựa chọn hướng giải mã suy đoán thích ứng. Chương 3 mô tả các quyết định thiết kế, kiến trúc serving, cost model và cơ chế điều khiển ngân sách cây nháp.

Chương 4 trình bày quy trình phát triển, phân vai, mốc thực hiện và các tiêu chí nghiệm thu. Chương 5 mô tả thiết kế hệ thống và các giao diện đóng băng, bao gồm endpoint OpenAI-compatible và contract của pipeline MC-OCR. Chương 6 trình bày protocol thực nghiệm, ma trận workload, các mức concurrency, chỉ số đo và phương pháp kiểm chứng lossless.

Các kết quả chạy benchmark, phân tích speedup, VRAM, tỷ lệ chấp nhận và bản đồ use-case sẽ được cập nhật sau khi môi trường vLLM và ba serving lane hoàn tất. Riêng kết quả Ollama local trong M0-04 chỉ xác nhận rằng Node 1 có thể gọi LLM, kiểm schema, kiểm source binding và nạp dữ liệu vào SQLite; kết quả này không đại diện cho số liệu hiệu năng chính thức của đề tài.

## 1.10. Tóm tắt chương

Chương này đã trình bày bối cảnh tự host LLM trên GPU phổ thông, vấn đề độ trễ cộng dồn trong workflow nhiều node và giới hạn của cấu hình speculative decoding tĩnh. Từ đó, đề tài đặt ra bài toán xây dựng Adaptive Proposer có khả năng lựa chọn ngân sách cây nháp theo phần cứng và trạng thái runtime, đồng thời duy trì tính lossless và tránh các trạng thái tăng tốc âm hoặc hết VRAM.

Ba câu hỏi nghiên cứu tập trung lần lượt vào hiệu năng, độ tin cậy và phạm vi workload hưởng lợi. Pipeline hóa đơn tiếng Việt từ MC-OCR được chọn làm workload vì tạo ra ba mức entropy: Node 1 text-to-JSON, Node 2 text-to-SQL và Node 3 SQL-to-BI. M0-04 đã đặt nền tảng bằng JSON Schema, SQLite schema, dev-set 30 mẫu và runner Node 1 qua Ollama. Những thành phần này là cơ sở để tiếp tục hiện thực Node 2, hoàn thiện benchmark vLLM và đánh giá toàn diện trong các mốc tiếp theo.

## 1.11. Nguyên tắc chuyển sang bản thảo chính thức

Bản Markdown này là nguồn nội dung đã chốt để đưa vào quyển đồ án. Khi chuyển sang Word hoặc LaTeX, nhóm giữ nguyên thứ tự mục và nội dung học thuật của chương. Khổ giấy, lề, phông chữ, cỡ chữ, giãn dòng, cách đánh số trang, bảng và hình sẽ được áp dụng theo template chính thức của khoa; các thông số trình bày này chưa được xem là đã khóa trong M0-04.

Nội dung hiện có khoảng 3.800 từ và dự kiến tương đương 5–7 trang nội dung học thuật sau khi áp dụng template chính thức. Số trang thực tế chỉ được xác nhận trên bản đã dàn trang. Các trang bìa, nhận xét, mục lục, danh mục bảng/hình và tài liệu tham khảo nằm ngoài nội dung Chương 1.
