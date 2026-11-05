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
- **Rủi ro phần cứng cần kiểm chứng ở M0:** T4 là kiến trúc Turing (sm_75) → **không hỗ trợ bfloat16** (phải dùng fp16), **không chạy được FlashAttention-2** (cần sm_80+, vLLM sẽ lùi về backend khác), và một số nhân lượng tử hoá tối ưu (ví dụ Marlin) **yêu cầu sm_80+**. Phải xác nhận tổ hợp vLLM + AWQ/GPTQ + EAGLE chạy được trên sm_75 **trong Tuần 1**, trước khi xây bất cứ thứ gì lên trên. *(Đã kiểm chứng thành công tại [logs/M0-01/README.md](../logs/M0-01/README.md): fallback float16, Triton Attention và MarlinLinearKernel chạy ổn định trên T4 đạt ~22–38 tok/s).*

## QĐ-10 — **Không** tự huấn luyện đầu nháp; dùng checkpoint công bố
- **Quyết định:** dùng checkpoint EAGLE-2 đã công bố cho mô hình đích. Loại bỏ hạng mục "huấn luyện draft head trên dữ liệu hoá đơn tiếng Việt" khỏi phạm vi (README trước đây liệt kê nó như một luận cứ novelty).
- **Lý do:** đây là hạng mục **rủi ro cao, thời lượng không ước lượng được, và không thuộc đóng góp**. Đóng góp nằm ở **bộ điều khiển cây nháp**; chất lượng draft head chỉ ảnh hưởng tới α — tức là dịch chuyển mọi làn *cùng một lượng*, không làm thay đổi kết luận so sánh giữa các làn. Với ngân sách ~455 giờ-người ([04 §0](04-quy-trinh-va-ke-hoach.md)), giữ nó lại là đánh đổi sai.
- **Hệ quả cần xử lý:** phải tồn tại checkpoint EAGLE-2 cho **đúng** mô hình đích đã chọn. Nếu không có → đổi mô hình đích cho khớp checkpoint, **không** đổi hướng sang tự train. Kiểm chứng ở M0.

## QĐ-11 — Xác minh lossless bằng thực nghiệm, dù đã đúng về lý thuyết
- **Quyết định:** dù tính lossless được bảo toàn theo thiết kế (không đụng vào tiêu chí chấp nhận), vẫn chạy một phép xác minh khớp-từng-ký-tự giữa làn vanilla và làn optimized, và đưa vào quyển.
- **Lý do:** đây là điểm phản biện dễ đoán nhất của hội đồng (*"tăng tốc 3× thì có mất chất lượng không?"*). Một bảng so khớp đầu ra chấm dứt tranh luận trong 10 giây. Chi phí gần như bằng không, giá trị phòng thủ rất cao. Giao thức: [06 §4](06-ke-hoach-thuc-nghiem.md).

## QĐ-12 — Chiến lược hai luồng song song (Dual-Track Pivot) & Cụ thể hóa cấu trúc cây nháp ở M2
- **Quyết định:**
  1. **Luồng 1 — Main Production Track (Đường găng tốt nghiệp - Critical Path):**
     - Cấu hình: Target Model `Qwen2.5-7B-Instruct-AWQ` kết hợp Draft Model độc lập `Qwen2.5-0.5B-Instruct` trên GPU T4 16GB.
     - Phạm vi: Bám sát tiến độ Sprint 3 (M2 Adaptive Controller v0 & Lossless Verification). Toàn bộ 36 ô thực nghiệm M1 và 54 ô đối chứng 3 làn M2 sử dụng cấu hình này.
     - Mục tiêu: Bảo đảm 100% an toàn tiến độ bảo vệ tốt nghiệp, ổn định trên hạ tầng T4 mà không phụ thuộc vào rủi ro huấn luyện.
  2. **Luồng 2 — R&D / Outsource Effort Track (Tự huấn luyện mở rộng EAGLE-3.1 Head):**
     - Cấu hình: Tự huấn luyện một draft head EAGLE-3.1 chuyên biệt cho `Qwen2.5-7B-Instruct` trên GPU A100 mượn/thuê.
     - Vai trò: Đóng vai trò là bài đo đối chứng nâng cao (Ablation Study) trong Chương 5 nhằm so sánh chéo hiệu năng của bộ điều khiển `AdaptiveDraftController` khi gắn với Draft Model độc lập 0.5B so với khi gắn với EAGLE-3.1 Head; đồng thời phục vụ demo trực quan tại buổi bảo vệ.
     - Cơ chế cô lập: Chạy hoàn toàn song song bên ngoài đường găng, không chặn hay làm trễ tiến độ Luồng 1.
  3. **Cụ thể hóa cấu trúc cây nháp (Tree Degeneration):**
     - Do nhân vLLM 0.30.0 chưa hỗ trợ thay đổi cấu trúc ma trận cây nháp (tree topology) linh hoạt theo từng bước suy luận, nhóm quyết định "ép phẳng" cây nháp thành chuỗi tuần tự đơn nhánh (single-branch degenerate tree).
     - Khi đó, ngân sách cây N* tương đương với độ sâu nháp k* trong tập ứng viên {0, 2, 3, 4}.
- **Lý do & Giá trị:**
  - *Triệt tiêu rủi ro:* Huấn luyện mô hình luôn tiềm ẩn rủi ro hội tụ và tương thích kernel vLLM trên GPU Turing sm_75. Phân tách hai luồng giúp bảo toàn tuyệt đối hạn nộp đồ án.
  - *Tính độc lập kiến trúc:* Bộ điều khiển `AdaptiveDraftController` và adapter hoàn toàn training-free và độc lập với mô hình nháp. Khi checkpoint EAGLE-3.1 hoàn tất, việc tích hợp chỉ là thay đổi cấu hình launcher mà không cần sửa đổi thuật toán điều khiển.

## QĐ-13 — Bộ điều khiển Thích ứng Làn 3 & Cơ chế An toàn (Memory Pressure & Periodic Probing)
- **Quyết định:**
  1. **Tích hợp Serving Engine V1 (vLLM v0.30.0):**
     - Kế thừa `AsyncScheduler` qua lớp tùy biến `AdaptiveAsyncScheduler` (nạp qua cờ `--scheduler-cls`) và kích hoạt Dynamic Speculative Decoding (`num_speculative_tokens_per_batch_size`).
     - Tầng `AdaptiveBudgetPlanner` chuyển quyết định per-request (`k_i`) sang batch level (`K = max(k_i)`), đồng thời cắt placeholder nháp (`spec_token_ids`) cho từng request để tương thích với async scheduling.
  2. **Memory Pressure Rule (Ngắt nhị phân chống OOM):**
     - Giám sát 2 điều kiện OR: Tỉ lệ chiếm dụng KV-cache > 90% (trích xuất từ `vllm:kv_cache_usage_perc`) HOẶC VRAM trống < 1500 MB (kèm cờ `enable_vram_guard`).
     - Máy trạng thái Cooldown & Hysteresis: Ép `k = 0` ngay khi chạm ngưỡng, giữ tối thiểu 3 bước liên tiếp (`memory_cooldown_steps = 3`), và chỉ giải phóng phục hồi khi KV-cache giảm sâu dưới 80% (`kv_cache_recover_pct = 80.0`).
  3. **Thoát bẫy bế tắc (Deadlock) bằng Periodic Probing:**
     - Khi `alpha < 0.65`, hệ thống fallback về `k = 0`. Để tránh đóng băng `alpha` vĩnh viễn (do `k = 0` không sinh token nháp nên không có phản hồi verify), controller tự động phát động 1 bước thăm dò nháp nhỏ `k = 2` sau mỗi `probe_interval` bước fallback.
     - Giá trị `probe_interval` được điều chỉnh từ 5 lên 20: Dữ liệu smoke test T4 cho thấy `probe_interval = 5` làm ~33% bước trên W3 bị ép nháp vô ích, kéo độ trễ Làn 3 xuống 0.78x. Việc nâng lên 20 bước giúp giảm tần suất probing xuống dưới 5%, triệt tiêu overhead trên workload khó mà vẫn giữ nguyên khả năng tự phục hồi.
- **Lý do & Giá trị:**
  - Bảo đảm an toàn tuyệt đối trước nguy cơ OOM khi tải cao hoặc context dài.
  - Phục hồi động sau vùng ngữ cảnh khó mà không cần can thiệp thủ công từ bên ngoài.

## QĐ-14 *(đề xuất — chờ Khôi duyệt)* — Node 3 bản tối thiểu thuộc DoD của M3-06; web demo vẫn ở tầng COULD
- **Bối cảnh:** QĐ-08 và [04 §5](04-quy-trinh-va-ke-hoach.md) xếp Node 3 vào tầng COULD (cắt đầu tiên khi trễ), nhưng card Trello [M3-06] yêu cầu "Node 2 Text-to-SQL và Node 3 VAT Report chạy với backend Modal" và "E2E OCR → SQL → VAT Report".
- **Đề xuất:** đưa **Node 3 bản tối thiểu** vào M3-06 — một request sinh một báo cáo tiếng Việt từ kết quả Q1–Q6, không có giao diện web. Dashboard/web demo vẫn giữ nguyên ở tầng COULD. QĐ-08 không thay đổi.
- **Lý do:** chi phí thấp (dùng lại prompt W3 v2 và client sẵn có), đổi lại có workload W4 đủ ba chế độ entropy trong một quy trình — đúng mục đích đo độ trễ cộng dồn của QĐ-08.
- **Bằng chứng:** `src/pipeline/report_writer.py`, `scripts/run_pipeline_e2e.py`, `experiments/runs/m3_06/`.

## QĐ-15 *(đề xuất — chờ Khôi duyệt)* — Node 2/Node 3 dùng nguyên prompt đóng băng của W2 v2/W3 v2
- **Đề xuất:** system prompt của Node 2 và Node 3 là bản đã đóng băng `experiments/configs/w2_v2/system_prompt.md` và `w3_v2/system_prompt.md`; `temperature = 0`, `seed = 42`; Node 2 `max_tokens = 256` (W2 dùng 96, quá ngắn cho câu có `GROUP BY`), Node 3 `max_tokens = 768`. Không gọi lại mô hình để sửa SQL lỗi.
- **Lý do:** đầu ra pipeline khớp với workload benchmark và số request bằng nhau giữa các làn, nên so sánh các làn công bằng.
- **Hệ quả (đã ghi trong `src/pipeline/README.md`):** README catalog nói Node 2 *nên* trả SQL, view và lời giải thích; vì prompt W2 v2 chỉ trả SQL nên `explanation` luôn `null` và `views_used` được suy ra từ SQL.

## QĐ-16 *(đề xuất — chờ Khôi duyệt)* — GPU dự phòng trên Modal qua `PTIT_GPU`
- **Bối cảnh:** ngày 04/11/2026 Modal không cấp được L4 (ba lần thử), trong khi T4/A10G/L40S/A100/H100 cấp được ngay.
- **Đề xuất:** biến `PTIT_GPU` chọn GPU lúc deploy — **L4 mặc định và là phần cứng tham chiếu**; `A10G` hoặc `T4` (cần `PTIT_DTYPE_PROFILE=fp16-fallback`) chỉ dùng cho nghiệm thu chức năng khi thiếu L4. Mọi bằng chứng chạy trên GPU khác phải ghi rõ GPU (`--hardware`) và **không** được so độ trễ với số liệu L4.
- **Bằng chứng:** `experiments/modal_lanes.py` (`resolve_gpu`), `tests/test_modal_runner_config.py`, `experiments/M3_00_MODAL_RUNBOOK.md`.

---

## Câu hỏi còn mở (phải đóng, kèm hạn)

| # | Câu hỏi | Vì sao quan trọng | Hạn chốt | Ai | Trạng thái |
|---|---|---|---|---|---|
| **M-01** | **Vì sao neo EAGLE-2 mà không phải EAGLE-3?** | EAGLE-3 (`arXiv:2503.01840`) là bản mạnh nhất nhưng checkpoint ít sẵn và khó tích hợp trên sm_75. Đã chuyển hóa thành Luồng 2 (Dual-Track) tự train EAGLE-3.1 làm Ablation Study. | **Đã đóng (10/10)** | Khôi | ✅ Đã chốt theo QĐ-12 |
| **M-02** | Có checkpoint EAGLE-2 cho Qwen2.5-7B không, và có chạy được với bản 4-bit không? | Thử nghiệm Sprint 1 cho thấy checkpoint cộng đồng lỗi kiến trúc trên vLLM và CUDA 13 bỏ sm_75 trên SGLang. Đã thay thế chuẩn xác bằng Luồng 1 (Draft 0.5B) chạy 100% ổn định trên T4. | **Đã đóng (10/10)** | Khôi | ✅ Đã chốt theo QĐ-12 |
| **M-03** | vLLM `custom_class` proposer có cho đổi cấu hình cây **theo từng bước** không? | Đã giải quyết ở M2 bằng giải pháp ép phẳng cây tuần tự k in {0, 2, 3, 4} và điều phối qua `AdaptiveScheduler` / `AdaptiveProposerAdapter`. | **Đã đóng (11/10)** | Khôi, Dương | ✅ Đã chốt |
| **M-04** | GVHD có chấp nhận **kết quả âm tính** (adaptive không thắng stock) không? | Quyết định có cần phương án xoay góc kể chuyện ở cổng 01/11 hay không. Hỏi sớm thì rẻ. | **Tuần 1** | Khôi | Đang theo dõi |
| **M-05** | **Ngày bảo vệ và hạn nộp quyển chính thức** | Hạn nộp quyển chính thức: **06/12/2026**; lịch bảo vệ dự kiến: giữa tháng 12/2026 theo thông báo của Học viện. Lộ trình 3 sprint cuối và kỷ luật đóng băng 15/11 đã được đồng bộ. | **Đã đóng (04/11)** | Cả nhóm | ✅ Đã chốt theo thông báo Học viện |

---

## Bản đồ use-case (giả thuyết cần kiểm chứng)
- **Nên bật tối ưu tốc độ:** tương tác real-time; **agent nhiều lượt gọi tuần tự** (độ trễ cộng dồn); **sinh output dài**.
- **Không nên / vô ích:** batch throughput lớn (spec decoding có thể *giảm* throughput ở batch lớn); input dài / output ngắn (nghẽn ở prefill → hướng A/B); nghẽn bộ nhớ / context siêu dài (→ hướng A).
