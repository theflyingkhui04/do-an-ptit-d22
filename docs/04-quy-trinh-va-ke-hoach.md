# Quy trình & Kế hoạch dự án

> Lập ngày **11/09/2026**. Vai trò: quản lý dự án.
> Ràng buộc đầu vào: **3 thành viên, trình độ ngang nhau, ~10–15h/người/tuần, không gặp mặt được, ngày bảo vệ chưa chốt (dự kiến 12/2026)**.

---

## 0. Giả định đang dùng (phải xác nhận lại khi khoa ra thông báo)

| Giả định | Giá trị đang dùng | Ảnh hưởng nếu sai |
|---|---|---|
| Ngày bảo vệ | giữa 12/2026 | — |
| Hạn nộp quyển | trước bảo vệ 2–3 tuần → **~30/11** | dịch toàn bộ lịch |
| **Code & kết quả freeze** | **15/11/2026** (kịch bản xấu nhất) | mốc quan trọng nhất; nếu deadline thật muộn hơn thì phần dư là *buffer*, **không** phải chỗ để nhét thêm scope |
| Năng lực nhóm | ~35 giờ-người/tuần → **~455 giờ-người** tổng | cắt scope sâu hơn nếu thấp hơn |

**Nguyên tắc số 1:** deadline chưa chốt thì lập kế hoạch theo ngày sớm nhất có thể xảy ra. Biết deadline muộn hơn là tin vui; biết muộn thì là thảm hoạ.

---

## 1. Chọn quy trình: **không Scrum, không Waterfall**

### Kết luận
> **Kanban theo dòng chảy liên tục + các mốc timebox cố định ngày, thả scope**, đặt trên một **xương sống ưu tiên-rủi-ro (risk-first)**.
> Đây là quy trình dạng **Lean/Kanban**, không phải Scrum — dù nhiều người quen gọi gộp cả hai là "Agile".

### Vì sao **không Waterfall**

Waterfall đòi biết trước đầu ra trước khi bắt tay làm. Đồ án này **không thoả điều kiện đó**, vì câu hỏi trung tâm là một câu hỏi nghiên cứu chưa ai trả lời:

> *Liệu adaptive proposer có thật sự thắng EAGLE-2 tĩnh trên T4 không?*

Chưa đo thì chưa biết. Và **có thể câu trả lời là không** — đó vẫn là một kết quả hợp lệ để bảo vệ, nhưng chỉ khi phát hiện sớm. Waterfall sẽ đẩy phát hiện đó về tận tuần 10, lúc không còn đường xoay. Với dự án có rủi ro kỹ thuật là rủi ro lớn nhất, Waterfall **giấu rủi ro về cuối** — đúng chỗ nguy hiểm nhất.

### Vì sao **không Scrum**

Scrum hợp lý ở chỗ nó chia nhỏ theo sprint, nhưng bộ máy của nó đổ vỡ với đội hình này:

| Thành phần Scrum | Vì sao không hợp ở đây |
|---|---|
| Daily standup (đồng bộ, hằng ngày) | **Không gặp mặt được.** Ép họp hằng ngày → bỏ trong 2 tuần → mất luôn cả kỷ luật. |
| Sprint Planning + Review + Retro | ~4h ceremonies/sprint. Trên tổng 35h/tuần đó là **hơn 5% ngân sách** cho họp hành trong một dự án 3 người. |
| Vai trò PO / SM / Dev tách bạch | 3 người thì ai cũng là Dev. Vai trò rỗng. |
| Sprint commitment (cam kết khối lượng sprint) | Việc nghiên cứu **không ước lượng được**. "Làm cho draft head chạy trên T4" có thể mất 2h hoặc 2 tuần. Cam kết sprint chỉ tạo ra nợ và cảm giác thất bại. |
| Velocity | Vô nghĩa khi phần lớn task là one-off, không lặp lại. |

**Scrum được thiết kế cho đội 5–9 người, ngồi cùng chỗ, làm sản phẩm có yêu cầu thay đổi từ bên ngoài.** Đây là đội 3 người, phân tán, yêu cầu do chính nhóm đặt ra, và biến động đến từ *kết quả đo* chứ không từ khách hàng. Sai công cụ.

### Cái thực sự cần

| Nhu cầu thật | Cơ chế |
|---|---|
| Phát hiện rủi ro sớm | **Risk-first**: mỗi mốc giết một điều-chưa-biết-lớn-nhất, không phải hoàn thiện một tầng kiến trúc |
| Làm việc lệch giờ nhau | **Kanban**: kéo việc khi rảnh, không chờ nhịp sprint; trạng thái nằm trên bảng chứ không trong đầu ai |
| Không ai bị chặn | **Sở hữu module + giao diện mỏng**: 3 người ngang trình → chia theo *ranh giới code*, không theo kỹ năng |
| Deadline không co giãn | **Timebox cố định ngày, thả scope**: đến hạn thì giao thứ đang có, không dời hạn |
| Ít họp mà vẫn không lạc nhau | **1 buổi đồng bộ 45 phút/tuần + phần còn lại viết ra văn bản** |

---

## 2. Nhịp làm việc (cadence)

Đội async **không phải là đội zero-sync**. Cần đúng một điểm đồng bộ — nhiều hơn thì lãng phí, ít hơn thì trôi.

### 2.1. Một cuộc gọi 45 phút mỗi tuần — thứ Bảy

Chỉ 3 mục, theo đúng thứ tự, có người bấm giờ:
1. **Bảng demo (15')** — mỗi người *chiếu màn hình* thứ mình chạy được. Không slide, không "em đang làm dở". Chạy được hoặc chưa.
2. **Gỡ chặn (20')** — chỉ nêu vấn đề đang chặn. Ai không bị chặn thì ngồi im.
3. **Chốt việc tuần tới (10')** — mỗi người kéo việc từ cột Ready.

Ghi biên bản vào file, 10 dòng. Ai vắng thì đọc biên bản, **không tổ chức họp bù**.

### 2.2. Standup viết, thứ Hai và thứ Năm

Nhắn vào nhóm chat, đúng 3 dòng:
```
Xong:    <việc đã merge / số đã đo được>
Đang:    <đúng 1 việc>
Chặn:    <ai/cái gì đang chặn — hoặc "không">
```
Dòng "Chặn" là dòng quan trọng nhất. Quy ước cứng: **bị chặn quá 24h mà không nhắn là lỗi quy trình, không phải lỗi kỹ thuật.**

### 2.3. Điểm lại 2 tuần một lần (gộp vào cuộc gọi tuần chẵn, +15')

Ba câu: *Cái gì đang chậm hơn tưởng? Giả định nào vừa bị kết quả đo bác bỏ? Có cần cắt scope không?*

---

## 3. Phân vai & sở hữu module

3 người ngang trình → **nút thắt không phải kỹ năng, mà là va chạm code và chờ nhau**. Vì vậy chia theo ranh giới file, mỗi người là người duyệt PR cuối cùng cho vùng của mình.

| Luồng | Sở hữu | Vùng code | Giao diện với luồng khác |
|---|---|---|---|
| **A · Engine** | Khôi (kiêm tích hợp) | `src/proposer/`, tích hợp vLLM | Phơi ra một class proposer + một dict telemetry |
| **B · Đo lường** | TV2 | `experiments/`, harness benchmark, phân tích | Chỉ gọi endpoint HTTP + đọc dict telemetry |
| **C · Pipeline & Quyển** | TV3 | `src/pipeline/`, dữ liệu MC-OCR, `docs/` | Chỉ gọi endpoint `/v1/chat/completions` |

**Hai giao diện này phải được chốt và viết ra trong Tuần 1**, trước khi có ai viết dòng code thứ hai. Chốt xong thì ba luồng chạy song song **mà không cần chờ nhau** — đây chính là thứ cứu một đội không gặp mặt được.

**Khôi giữ vai tích hợp** (người duy nhất được merge vào `main`), vì đang nắm nhật ký thiết kế. Không luân phiên vai này — luân phiên vai tích hợp trong dự án 13 tuần chỉ tạo ra đứt gãy ngữ cảnh.

### Luật hợp tác
- **WIP tối đa 1 việc/người.** Muốn nhận việc mới thì phải đẩy việc cũ sang Done.
- **Branch sống tối đa 3 ngày.** Lâu hơn thì chẻ nhỏ task ra. (Đội async + branch dài = địa ngục merge.)
- **PR không quá ~300 dòng thay đổi.** Duyệt trong 24h, không chờ tới buổi gọi tuần.
- **Mọi quyết định lớn ghi vào [03-quyet-dinh-thiet-ke.md](03-quyet-dinh-thiet-ke.md)** dưới dạng QĐ-xx. Đội async quên mất *vì sao* nhanh hơn đội ngồi cùng phòng rất nhiều.

---

## 4. Bảng Kanban

```
Backlog  →  Ready  →  Doing (WIP≤1)  →  Review  →  Done
```
- **Ready** = đã đủ rõ để bắt tay ngay: biết đầu ra là gì, biết thế nào là xong.
- **Done** phải thoả **cả ba** điều kiện dưới.

### Definition of Done

Một task chỉ xong khi:
1. ✅ Code đã merge vào `main`, chạy được từ đầu bằng một lệnh.
2. ✅ **Có con số đo được ghi vào `experiments/`** (nếu là task có ảnh hưởng hiệu năng).
3. ✅ **Có ít nhất một đoạn văn viết vào bản nháp quyển.**

Điều 3 không phải thủ tục. Quy tắc vận hành của dự án này là:

> **Kết quả nào chưa được viết ra thì coi như chưa tồn tại.**

Đây là biện pháp phòng vệ trực tiếp cho cái bẫy kinh điển của đồ án: tuần 12 mới bắt đầu viết quyển, rồi phát hiện không nhớ nổi cấu hình nào đã sinh ra biểu đồ nào.

---

## 5. Phân tầng scope (MoSCoW) — cắt ngay bây giờ

Ngân sách ~455 giờ-người. Scope trong README hiện vượt xa con số đó. Cắt ngay từ đầu, chứ không cắt trong hoảng loạn ở tuần 11.

### MUST — không có thì không có đồ án (~55% ngân sách)
- vLLM chạy được target model trên T4, có EAGLE-2 stock làm chuẩn so sánh.
- Harness benchmark đo được: acceptance α, TPOT/ITL, TTFT, VRAM, theo **thang concurrency**.
- **Adaptive Proposer**: điều chỉnh cây nháp lúc chạy theo α + VRAM còn trống.
- **Xác minh lossless**: cùng seed, cùng prompt → đầu ra khớp từng ký tự với làn vanilla.
- Ba làn so sánh trên cùng phần cứng: vanilla · EAGLE-2 tĩnh · adaptive.
- Quyển + slide bảo vệ.

### SHOULD — thứ biến bài này từ "chạy được" thành "có đóng góp" (~30%)
- **Profiler phần cứng lúc khởi động** (đo `t_draft`, `t_verify`, băng thông) nạp vào cost-model → **đây mới là phần "hardware-aware"**, tức là novelty thật (QĐ-04).
- Bản đồ use-case: chỉ rõ regime nào **nên tắt** tối ưu (D-cut đã chứng minh regime đó tồn tại — xem [paper_summary/06](paper_summary/06-dcut-liu2026.md)).
- Pipeline hoá đơn MC-OCR, tối thiểu 2 node, để đo end-to-end.

### COULD — chỉ làm nếu MUST + SHOULD xong trước 01/11 (~15%)
- Node 3 (SQL-to-BI), dashboard Streamlit.
- Tích hợp Dify/LangGraph.
- Chạy đối chứng trên GPU thứ hai (A100) để chứng minh cost-model thích ứng được — *điểm cộng mạnh nếu kịp*.

### WON'T — cắt ngay hôm nay, ghi rõ lý do trong quyển
- ❌ **Tự huấn luyện EAGLE draft head trên dữ liệu hoá đơn tiếng Việt.**
  Dùng checkpoint EAGLE-2 đã công bố. Lý do: đây là hạng mục *rủi ro cao, tốn thời gian không đoán trước được, và **không** thuộc đóng góp của đồ án*. Đóng góp nằm ở **bộ điều khiển cây nháp**, không ở chất lượng draft head. README đã liệt kê đây là phương án dự phòng — quyết định của PM là **đẩy nó thành phương án chính ngay từ đầu**, không chờ.
- ❌ App quản lý model (đã loại ở QĐ-05).
- ❌ Bất kỳ hạng mục nào cần training kéo dài.

---

## 6. Lộ trình theo mốc — **cố định ngày, thả scope**

Mỗi mốc giết một điều-chưa-biết. Đến ngày thì giao thứ đang có; **không dời ngày**.

| Mốc | Hạn | Câu hỏi rủi ro được trả lời | Đầu ra tối thiểu |
|---|---|---|---|
| **M0 · Xương sống** | **20/09** (T1) | *Toàn bộ đề tài có khả thi trên T4 không?* | vLLM + Qwen chạy trên T4; EAGLE-2 stock chạy được; in ra **một** con số tok/s; đã chốt 2 giao diện |
| **M1 · Đường cơ sở** | **04/10** (T3) | *Đo có đáng tin không?* | Harness tự động; **kết quả 2 làn trên thang concurrency 1→32**, có variance qua nhiều lần chạy |
| **M2 · Adaptive v0** | **18/10** (T5) | *Điều khiển cây lúc chạy có làm được không?* | Proposer đổi cây theo α; **lossless đã chứng minh**; có/không nhanh hơn stock — đều được ghi nhận |
| **M3 · Hardware-aware v1** | **01/11** (T7) | *Đưa đặc tính GPU vào có ăn thua không?* | Profiler + cost-model; đo trên ≥2 cấu hình phần cứng nếu mượn được A100 |
| **M4 · Đóng băng** | **15/11** (T9) | *Câu chuyện đã đủ để bảo vệ chưa?* | **Toàn bộ số liệu freeze**; biểu đồ xuất xong; demo pipeline quay video |
| **M5 · Quyển & bảo vệ** | **30/11** (T11) | — | Quyển nộp; slide; đã tổng duyệt thử 1 lần |

### Lịch tuần

```
T1  14–20/09   M0 Xương sống  ·  chốt giao diện  ·  bắt đầu viết Chương 1–2
T2  21–27/09   Harness benchmark  ·  làn vanilla
T3  28/09–04/10  ▲ M1 Đường cơ sở  ·  làn EAGLE-2 stock  ·  sweep concurrency
T4  05–11/10   Proposer khung sườn  ·  telemetry α  ·  dữ liệu MC-OCR
T5  12–18/10   ▲ M2 Adaptive v0  ·  xác minh lossless
T6  19–25/10   Profiler phần cứng  ·  cost-model  ·  pipeline node 1–2
T7  26/10–01/11  ▲ M3 Hardware-aware v1  ·  (cửa sổ mượn A100)
T8  02–08/11   Quét kết quả đầy đủ  ·  bản đồ use-case  ·  chốt COULD
T9  09–15/11   ▲ M4 ĐÓNG BĂNG  ·  biểu đồ  ·  quay demo
T10 16–22/11   Viết quyển toàn thời gian (3 người)
T11 23–29/11   ▲ M5 Nộp  ·  slide
T12 30/11–06/12  Luyện bảo vệ  ·  dựng câu hỏi phản biện
T13 07–13/12   Đệm / Bảo vệ
```

**Luồng C (quyển) chạy song song từ Tuần 1**, không đợi tới T10. T10 là để *ráp và gọt*, không phải để *viết từ đầu*.

### Ba cổng kiểm soát (quyết định cắt scope, không thương lượng)

| Ngày | Nếu điều này chưa đúng | Thì làm |
|---|---|---|
| **04/10** | Chưa có đường cơ sở 2 làn chạy được | Cắt toàn bộ COULD; pipeline rút còn 1 node |
| **01/11** | Adaptive chưa nhanh hơn stock ở *bất kỳ* regime nào | **Xoay góc kể chuyện**: đề tài trở thành *nghiên cứu thực nghiệm về giới hạn của speculative decoding trên GPU phổ thông*. Vẫn đủ bảo vệ — D-cut cho thấy "SD chạy chậm hơn" là phát hiện có giá trị thật. Dừng tối ưu, dồn vào đo và phân tích. |
| **15/11** | Bất cứ thứ gì chưa xong | Vẫn đóng băng. Viết nó vào mục Hạn chế & Hướng phát triển. |

Cổng 01/11 là cổng quan trọng nhất, nên nói rõ trước với GVHD ngay từ đầu: **kết quả âm tính có được chấp nhận không?** Hỏi câu này ở tuần 1 thì rẻ; hỏi ở tuần 10 thì đã muộn.

---

## 7. Sổ rủi ro

| # | Rủi ro | Khả năng | Tác động | Dấu hiệu nhận biết | Đối sách |
|---|---|:--:|:--:|---|---|
| R1 | vLLM `custom_class` không cho đổi cây **theo từng bước** | Trung bình | **Chí mạng** | M0 trượt | Kiểm chứng **ngay Tuần 1**. Nếu bí → fork vLLM hoặc lùi về script suy luận tự viết (chậm hơn nhưng kiểm soát được) |
| R2 | T4 16GB không đủ cho target + draft + KV cache | Trung bình | Cao | OOM lúc M0 | Target 4-bit hoặc model 1–3B. **Đo capacity thật trước** ([02-tong-quan](02-tong-quan-tai-lieu.md) đã cảnh báo) |
| R3 | Wall-clock trên Colab/Kaggle nhiễu, số vô nghĩa | **Cao** | Cao | Lệch >10% giữa các lần chạy | Warm-up + ≥5 lần lặp + báo cáo trung vị & phương sai. Đóng gói thành quy tắc trong harness từ M1 |
| R4 | Adaptive không thắng stock EAGLE-2 | Trung bình | Cao | M3 | Cổng 01/11 → xoay thành nghiên cứu thực nghiệm |
| R5 | Một thành viên biến mất (thi cử, việc riêng) | **Cao** | Trung bình | Vắng 2 standup liên tiếp | Giao diện mỏng giúp cô lập thiệt hại; luồng C có thể bị hấp thụ bởi 2 người còn lại |
| R6 | Va chạm merge do 3 người ngang trình cùng đụng engine | Trung bình | Trung bình | PR conflict lặp lại | Sở hữu module + branch ≤3 ngày + PR ≤300 dòng |
| R7 | Khoa chốt deadline **sớm hơn** dự kiến | Trung bình | Cao | Thông báo của khoa | Đã plan theo kịch bản xấu nhất; M4 = 15/11 chính là lớp đệm |
| R8 | Không mượn được A100 | Trung bình | Thấp | T7 | Phần đối chứng đa-phần-cứng nằm ở tầng COULD, mất không ảnh hưởng MUST |

---

## 8. Chỉ số theo dõi (cập nhật mỗi tuần, 5 phút)

Đội nghiên cứu **không đo bằng story points**. Đo bằng ba thứ:

1. **Rủi ro đã bị giết / còn lại** — thước đo tiến độ thật của dự án nghiên cứu.
2. **Số ô đã có số liệu** trên bảng kết quả (3 làn × N tác vụ × 5 mức concurrency). Đây là bảng cuối cùng sẽ nằm trong quyển; điền dần được bao nhiêu ô là biết đang ở đâu.
3. **Số trang nháp quyển.** Nếu tới 01/11 mà dưới 20 trang thì đang trễ, bất kể code chạy tốt cỡ nào.

Không theo dõi: velocity, giờ công, số commit.

---

## 9. Tóm tắt một dòng

> **Kanban + mốc cố định ngày, xương sống ưu tiên-rủi-ro, một cuộc gọi 45 phút/tuần, phần còn lại viết ra văn bản; cắt scope trước chứ không cắt lúc hoảng; code đóng băng 15/11 bất chấp.**
