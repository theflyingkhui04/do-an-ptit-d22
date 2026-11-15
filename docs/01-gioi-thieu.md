# 1. Giới thiệu

> Cập nhật **11/09/2026**. Bản này đồng bộ với [README](../README.md), [nhật ký quyết định](03-quyet-dinh-thiet-ke.md) và [kế hoạch](04-quy-trinh-va-ke-hoach.md). Thuật ngữ & ký hiệu: [07-thuat-ngu.md](07-thuat-ngu.md).

## 1.1. Tên đề tài

**(TV)** *FlexSDS: Hệ thống phục vụ giải mã suy đoán EAGLE thích ứng theo phần cứng cho mô hình ngôn ngữ lớn.*
**(EN)** *FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs.*

> Tên đề tài được đồng bộ ngày 14/11/2026. Các câu hỏi nghiên cứu dưới đây mô tả mục tiêu; trạng thái tích hợp CLI/runtime hiện tại xem [README](../README.md) và [runbook CLI/Modal](runbooks/flexsds-cli-modal-calibration.md).

## 1.2. Bối cảnh và động lực

Mô hình ngôn ngữ lớn (LLM) đang trở thành công cụ nền cho việc **tự động hoá các tác vụ lặp lại** — đặc biệt qua các nền tảng orchestration như **Dify** và **LangGraph**, nơi người dùng ghép nhiều lời gọi LLM thành một quy trình.

Tồn tại một rào cản thực tế: **API thương mại tốn phí** ở quy mô sử dụng thường xuyên, trong khi nhiều tình huống yêu cầu **dữ liệu riêng tư, hoạt động offline, hoặc không phụ thuộc nhà cung cấp**. Lựa chọn khả dĩ thường là **tự host mô hình mã nguồn mở trên phần cứng khiêm tốn**.

Một câu hỏi tự nhiên: *khi các mô hình đóng đã rất mạnh, tại sao còn phải tối ưu?* Hai vế trả lời. Thứ nhất, **chính các hệ phục vụ LLM quy mô lớn cũng được xây trên đúng những kỹ thuật này** — đây là nghiên cứu hạ tầng lõi, không phải cạnh tranh với mô hình đóng. Thứ hai, **khi tự host, "chi phí mỗi token" không biến mất mà chuyển hoá** thành thời gian GPU, VRAM và độ trễ.

Trong các trục tối ưu, **độ trễ sinh (decode latency)** là nút thắt của trải nghiệm tương tác và đặc biệt của **workflow agent gọi LLM nhiều lượt tuần tự** — nơi độ trễ **cộng dồn** qua từng bước.

## 1.3. Phát biểu bài toán

**Giải mã suy đoán (speculative decoding)** là hướng tăng tốc **không đổi đầu ra (lossless)**: một cơ chế nháp nhẹ đề xuất trước nhiều token, mô hình đích xác thực chúng trong một lượt truyền xuôi, giữ nguyên phân phối kết quả. **EAGLE-2** dùng *cây nháp động* điều chỉnh theo độ tự tin của bộ nháp (chi tiết: [paper_summary/03](paper_summary/03-eagle2-li2024.md)).

Vấn đề nằm ở chỗ EAGLE-2 và các phương pháp cùng dòng được thiết kế và kiểm chứng chủ yếu trên **GPU cao cấp băng thông rất cao** (A100, RTX 3090, H800). Trên GPU phổ thông, ba chế độ hỏng xuất hiện:

**(a) Tăng tốc âm (negative acceleration).** Lợi ích của việc xác thực song song đến từ giả định GPU đang **nghẽn băng thông** nên phần tính toán rảnh. Giả định này yếu dần khi tỉ lệ compute/băng thông của GPU thay đổi, khi mô hình đích được lượng tử hoá (target chạy nhanh hơn → biên lợi thu hẹp), hoặc khi nhiều request chạy đồng thời. Khi đó việc xác thực một cây nháp lớn trở thành **chi phí thuần**, và hệ thống chạy **chậm hơn cả giải mã thường**. Hiện tượng này đã được đo trực tiếp trong [D-cut](paper_summary/06-dcut-liu2026.md): speedup tụt xuống **0.67×** trên Llama-3.1-8B ở concurrency 32.

**(b) Cạn VRAM (OOM).** Cây nháp, KV cache của các nhánh nháp và KV cache của các request đồng thời cùng tranh một lượng VRAM rất hạn chế. Cấu hình cây tĩnh không có cơ chế co lại khi bộ nhớ eo hẹp → hệ thống sập thay vì chạy chậm đi.

**(c) Không có hướng dẫn bật/tắt.** Lợi ích phụ thuộc mạnh vào loại workload, nhưng chưa có đặc tả rõ ràng *"quy trình nào trên phần cứng phổ thông thực sự được hưởng lợi"*. Người dùng phải tự mò.

Điểm chung của cả ba: **tham số nháp được đặt cố định lúc khởi động, trong khi điều kiện thực tế thay đổi theo từng bước giải mã và theo từng GPU.**

> **Câu hỏi nghiên cứu:**
>
> **RQ1.** Việc để quyết định nháp **thích ứng theo đặc tính phần cứng đo được lúc chạy** (độ trễ nháp/xác thực, VRAM còn trống, tỉ lệ chấp nhận) có cải thiện tốc độ giải mã so với EAGLE-2 cấu hình tĩnh trên GPU phổ thông không, trong khi vẫn giữ **lossless**?
>
> **RQ2.** Cơ chế thích ứng có **loại bỏ được chế độ tăng tốc âm và chế độ OOM** hay không — tức là có đảm bảo tốc độ *không bao giờ tệ hơn* giải mã thường không?
>
> **RQ3.** Trên một quy trình tự động hoá thực tế, **loại workload nào thực sự hưởng lợi** từ tối ưu tốc độ, và hưởng lợi tới mức nào?

RQ2 đáng chú ý ở chỗ nó **có thể trả lời được kể cả khi RQ1 cho kết quả âm tính**: nếu bộ điều khiển thích ứng không nhanh hơn EAGLE-2 tĩnh nhưng loại bỏ được các chế độ hỏng, đó vẫn là một đóng góp về độ tin cậy hệ thống (xem cổng kiểm soát 01/11 trong [04 §6](04-quy-trinh-va-ke-hoach.md)).

## 1.4. Mục tiêu và phạm vi

**Mục tiêu:**

- **G1.** Tái lập EAGLE-2 trên vLLM làm chuẩn đối chứng (baseline) chạy được trên GPU phổ thông.
- **G2.** Xây **Adaptive Proposer** — bộ điều khiển chỉnh ngân sách cây nháp lúc chạy từ hồ sơ phần cứng đo thực + telemetry (α, VRAM).
- **G3.** Đóng gói thành **endpoint tương thích OpenAI** (`/v1/chat/completions`), cắm thẳng vào Dify/LangGraph mà không sửa ứng dụng.
- **G4.** **Đánh giá end-to-end** trên pipeline phân tích hoá đơn tiếng Việt, so ba làn trên **thang concurrency**, và rút ra **bản đồ use-case**.

**Phạm vi:**

- Suy luận **đơn GPU**, trọng tâm là chế độ phục vụ độ trễ thấp; có quét thang concurrency (1 → 32) để **xác định ranh giới nơi tối ưu hết tác dụng**, chứ không nhằm tối ưu cho throughput lớn.
- Mô hình đích: **Qwen2.5-7B-Instruct lượng tử hoá 4-bit (AWQ/GPTQ)** trên T4 16GB ([QĐ-09](03-quyet-dinh-thiet-ke.md)).
- **Dùng checkpoint EAGLE-2 công bố sẵn — không tự huấn luyện đầu nháp** ([QĐ-10](03-quyet-dinh-thiet-ke.md)).
- Giữ nguyên **tiêu chí chấp nhận** của speculative decoding → tính lossless được bảo toàn theo thiết kế, không phải theo thực nghiệm.

**Ngoài phạm vi (nói rõ để tránh hiểu nhầm khi bảo vệ):**

- ❌ **Không** cải thiện chất lượng đầu ra. Đề tài lossless → đầu ra **theo định nghĩa là giống hệt** làn thường. Không có đánh giá chất lượng sinh; chỉ có **một phép xác minh** rằng đầu ra khớp từng ký tự ([06 §4](06-ke-hoach-thuc-nghiem.md)).
- ❌ **Không** huấn luyện hay tinh chỉnh mô hình đích.
- ❌ **Không** nhằm đánh bại EAGLE-2 trên mọi cấu hình — chỉ cần **thắng hoặc ngang trong chế độ GPU phổ thông**, kèm đặc tả regime ([QĐ-04](03-quyet-dinh-thiet-ke.md)).
- ❌ **Không** xây app quản lý mô hình ([QĐ-05](03-quyet-dinh-thiet-ke.md)).

## 1.5. Hướng tiếp cận (tóm tắt)

```
┌─ Lúc khởi động ────────────────────────────────────────────┐
│  ĐO HỒ SƠ PHẦN CỨNG                                        │
│  t_draft, t_verify theo cỡ cây · băng thông · VRAM khả dụng │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌─ Lúc phục vụ ──────────────────────────────────────────────┐
│  TELEMETRY THỜI GIAN THỰC                                  │
│  α trượt (tỉ lệ chấp nhận) · VRAM còn trống · batch hiện tại│
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌─ Mỗi bước giải mã ─────────────────────────────────────────┐
│  CHỌN NGÂN SÁCH CÂY                                        │
│  cây = argmax ( lợi ích kỳ vọng / chi phí đo thật )         │
│  → gồm cả lựa chọn "TẮT hẳn suy đoán"                       │
└────────────────────────────────────────────────────────────┘
```

Điểm mấu chốt: mẫu số là **chi phí đã đo trên đúng GPU đang chạy**, không phải hằng số lý thuyết. Đây là điều phân biệt đề tài với *cây động theo confidence* — vốn đã là đóng góp sẵn có của EAGLE-2 và Talon ([QĐ-04](03-quyet-dinh-thiet-ke.md)). Khuôn mẫu "tối đa hoá lợi ích/chi phí-đo-thật" lấy cảm hứng từ [D-cut](paper_summary/06-dcut-liu2026.md), nhưng D-cut nhắm chế độ **batched concurrency cao trên GPU datacenter**, còn đề tài này nhắm **batch nhỏ trên GPU phổ thông** — hai chế độ khác nhau, chưa được phủ.

## 1.6. Workload đánh giá — pipeline hoá đơn tiếng Việt

Đánh giá chạy trên một quy trình thật thay vì microbenchmark tách rời, dùng tập **MC-OCR**:

```
[Văn bản hoá đơn thô — MC-OCR]
        │
        ▼  NODE 1 · EXTRACTOR    → JSON có cấu trúc (cửa hàng, mặt hàng, giá, thuế) → SQLite
        ▼  NODE 2 · TEXT-TO-SQL  → câu hỏi tiếng Việt → SQL trên CSDL hoá đơn
        ▼  NODE 3 · SQL-TO-BI    → báo cáo phân tích      [tầng COULD — cắt nếu trễ]
```

Đây **không phải** việc gắn đề tài vào một domain ứng dụng (điều đã bị loại ở [QĐ-01](03-quyet-dinh-thiet-ke.md)). Pipeline này đóng vai **dụng cụ đo**, được chọn vì nó cho **ba chế độ entropy tương phản trong cùng một quy trình** — thứ mà một prompt đơn lẻ không cho được:

| Node             | Đặc tính sinh                            | α kỳ vọng | Vai trò trong thí nghiệm                                                                                       |
| ---------------- | ------------------------------------------- | :----------: | ----------------------------------------------------------------------------------------------------------------- |
| 1 · Extractor   | JSON lặp khuôn, entropy**thấp**    |     cao     | Chế độ*tốt nhất* của suy đoán — cây nên mọc sâu                                                    |
| 2 · Text-to-SQL | cú pháp ràng buộc,**trung bình** |     vừa     | Chế độ cân bằng                                                                                              |
| 3 · SQL-to-BI   | văn xuôi tự do, entropy**cao**     |    thấp    | Chế độ*xấu nhất* — nơi tăng tốc âm xuất hiện, và nơi bộ điều khiển phải biết **tắt** |

Ngoài ra, độ trễ end-to-end `= Lat(N1) + Lat(N2) + Lat(N3)` minh hoạ trực tiếp luận điểm **độ trễ cộng dồn** ở §1.2. Đây là lý do chọn pipeline nhiều node thay vì một lời gọi đơn.

---

# 2. Sản phẩm đầu ra

| # | Đầu ra                                                                                              | Tầng  |
| - | ----------------------------------------------------------------------------------------------------- | ------ |
| 1 | **Máy phục vụ tương thích OpenAI** với hai làn *vanilla* / *optimized*              | MUST   |
| 2 | **Adaptive Proposer** — bộ điều khiển cây nháp theo hồ sơ phần cứng, tích hợp vLLM | MUST   |
| 3 | **Bộ đo (benchmark harness)** — tự động, quét thang concurrency, ghi α/TPOT/TTFT/VRAM   | MUST   |
| 4 | **Xác minh lossless** — chứng minh đầu ra khớp từng ký tự với làn vanilla            | MUST   |
| 5 | **Bản đồ use-case & khung quyết định** — khi nào nên/không nên bật, có số liệu   | SHOULD |
| 6 | **Pipeline hoá đơn MC-OCR** dùng làm workload đo                                          | SHOULD |
| 7 | **Mã nguồn mở + quyển báo cáo** tái lập được                                         | MUST   |
| 8 | Dashboard trực quan, tích hợp Dify                                                                 | COULD  |

Phân tầng đầy đủ: [04 §5](04-quy-trinh-va-ke-hoach.md).

---

# 3. Đóng góp

1. **Về phương pháp — bộ điều khiển nháp thích ứng phần cứng.** Đưa **chi phí đo thực trên GPU đích** (không phải hằng số lý thuyết, không phải chỉ độ tự tin của bộ nháp) vào quyết định ngân sách cây nháp của EAGLE-2, giữ nguyên lossless. Phân biệt rõ với *cây động theo confidence* — đóng góp vốn đã có của EAGLE-2/Talon.
2. **Về độ tin cậy hệ thống — loại bỏ hai chế độ hỏng.** Bộ điều khiển có sàn an toàn: khi tăng tốc không còn lợi hoặc khi VRAM eo hẹp, nó co cây hoặc **tắt hẳn suy đoán**, đảm bảo hệ thống *không bao giờ chậm hơn* giải mã thường và *không OOM* — hai điều mà cấu hình tĩnh không hứa được.
3. **Về đánh giá — phương pháp và bộ số liệu end-to-end theo thang concurrency** trên quy trình thật, thay vì một con số speedup đo ở batch = 1 (điều kiện dễ nhất).
4. **Về hiểu biết — đặc tả regime use-case** kèm khung quyết định, bao gồm cả các trường hợp **không nên** bật tối ưu.
