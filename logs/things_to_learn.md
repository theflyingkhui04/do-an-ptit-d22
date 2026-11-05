# Danh mục kiến thức trọng tâm cần nghiên cứu: Cẩm nang toàn diện cho đồ án tốt nghiệp

Tài liệu này tổng hợp toàn bộ các mảng kiến thức lý thuyết, kỹ thuật hệ thống, cơ chế tối ưu phần cứng và từ khóa tra cứu phục vụ đồ án tốt nghiệp: **"Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference"** (PTIT D22).

Nội dung được cấu trúc thành 6 phần hoàn chỉnh:
1. **Bộ nhớ KV cache:** Bản chất toán học, kiến trúc chú ý GQA và cơ chế quản lý bộ nhớ.
2. **Giải mã suy đoán thích ứng (Adaptive Speculative Decoding):** Nguyên lý không tổn hao (lossless), tiến hóa EAGLE, bộ điều khiển thích ứng (Adaptive Proposer) và rào chắn OOM.
3. **Cơ chế tối ưu hóa tính toán chú ý: PagedAttention và FlashAttention:** Bản chất phân trang bộ nhớ và cơ chế tối ưu truy xuất IO-aware.
4. **Lượng tử hóa mô hình (Quantization & Quantized Models):** Các kỹ thuật PTQ (AWQ, GPTQ, RTN) và bài toán chạy mô hình 7B trên GPU 16GB.
5. **Các chỉ số đo lường hiệu năng mô hình ngôn ngữ lớn (Benchmarking Metrics with vLLM):** TTFT, TPOT, ITL, thông lượng, tỷ lệ chấp thuận và quy trình quét tải.
6. **Ma trận đối chiếu câu hỏi phản biện bảo vệ đồ án:** Chuẩn bị câu trả lời trước hội đồng chấm đồ án.

---

## Phần 1: Cơ chế và tối ưu hóa bộ nhớ KV cache trong mô hình ngôn ngữ lớn

### 1. Bản chất và kiến trúc KV cache trong mô hình Transformer
- **Tại sao cần KV cache:** Quá trình sinh tự hồi quy (autoregressive generation) của mô hình Transformer chia làm hai pha rõ rệt:
  + *Prefill phase:* Tính toán toàn bộ biểu diễn cho chuỗi prompt đầu vào một lần duy nhất; pha này bị giới hạn bởi năng lực tính toán (compute-bound).
  + *Decode phase:* Sinh từng token kế tiếp tuần tự; ở mỗi bước sinh, nếu không lưu trữ các vector Key và Value của các token trước đó, mô hình sẽ phải tính toán lại toàn bộ với độ phức tạp $O(N^2)$. KV cache giúp giảm độ phức tạp mỗi bước xuống $O(N)$ bằng cách tái sử dụng trạng thái chú ý đã tính.
- **Nghẽn băng thông bộ nhớ (Memory-bandwidth bound):** Trong pha decode, cường độ số học (arithmetic intensity - số phép tính FLOPS trên mỗi byte nạp từ VRAM) cực kỳ thấp. Tốc độ sinh token bị giới hạn bởi băng thông đọc trọng số và KV cache từ VRAM vào vi xử lý GPU.
- **Kiến trúc chú ý và kích thước KV cache:**
  + *MHA (Multi-Head Attention):* Số head của Key-Value bằng số head của Query ($n_{\text{kv}} = n_{\text{q}}$), tốn nhiều bộ nhớ nhất.
  + *MQA (Multi-Query Attention):* Tất cả Query heads dùng chung duy nhất 1 Key head và 1 Value head ($n_{\text{kv}} = 1$).
  + *GQA (Grouped-Query Attention):* Chia $n_{\text{q}}$ Query heads thành các nhóm dùng chung $n_{\text{kv}}$ Key-Value heads.
  + *Mô hình của đồ án (Qwen2.5-7B):* Sử dụng GQA với $n_{\text{q}} = 28$ và $n_{\text{kv}} = 4$. Nhờ vậy, dung lượng KV cache giảm xuống còn $4/28 \approx 1/7$ so với kiến trúc MHA truyền thống.
- **Công thức tính dung lượng bộ nhớ KV cache:**
  $$\text{Dung lượng mỗi token (bytes)} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}$$
  *(Trong đó số 2 đầu tiên đại diện cho hai tensor Key và Value; số 2 thứ hai đại diện cho kiểu dữ liệu FP16/BF16 chiếm 2 bytes).*

### 2. Quản lý bộ nhớ KV cache trong vLLM
- **Cơ chế PagedAttention:** Lấy cảm hứng từ cơ chế phân trang bộ nhớ ảo trong hệ điều hành:
  + Thay vì cấp phát một khối bộ nhớ liên tục cho độ dài ngữ cảnh tối đa (gây lãng phí và phân mảnh trầm trọng), vLLM chia nhỏ KV cache thành các khối cố định (blocks, thường là 16 hoặc 32 token).
  + Các block được lưu rải rác trên VRAM vật lý và được quản lý logic qua bảng trang (*block table*).
  + Cơ chế này giúp triệt tiêu hoàn toàn phân mảnh bộ nhớ ngoài (external fragmentation) và kiểm soát phân mảnh nội bộ dưới $4\%$.
- **Cơ chế cấp phát động và lập lịch liên tục (Continuous batching):**
  + Block được cấp phát động theo từng bước sinh token và được giải phóng ngay khi câu trả lời hoàn tất.
  + Cơ chế lập lịch ở mức vòng lặp (iteration-level scheduling) ghép nối linh hoạt các request mới vào lô giải mã.
- **Giám sát dung lượng và phòng chống quá tải:**
  + Tham số cấu hình `gpu_memory_utilization` ấn định tỷ lệ VRAM dành cho trọng số mô hình và vùng đệm KV cache.
  + Phương thức `scheduler.get_kv_cache_usage()` trả về tỷ lệ phần trăm khối KV cache đang bị chiếm dụng tại thời gian thực.
  + Cơ chế xử lý khi cạn kiệt block: Đình chỉ và hoán đổi sang RAM máy chủ (*swap to CPU*) hoặc hủy bỏ để tính toán lại (*preemption and recomputation*).

### 3. KV cache trong giải mã suy đoán (Speculative Decoding & EAGLE-2)
- **Cấu trúc KV cache kép (Dual KV cache):** Cả mô hình đích (target model) và mô hình nháp (draft model) đều phải duy trì KV cache riêng biệt.
  + Draft model (EAGLE head) hoạt động ở tầng biểu diễn đặc trưng (feature level), cần KV cache nhỏ gọn để tự hồi quy sinh cây nháp.
  + Target model duy trì KV cache lớn của toàn bộ ngữ cảnh chính thức.
- **Cơ chế chú ý dạng cây (Tree Attention Mask):**
  + Khi sinh cây nháp trong EAGLE-2, các nhánh token con chia sẻ chung tiền tố (prefix) KV cache của nút cha.
  + Khi mô hình đích xác thực toàn bộ cây nháp trong một bước duy nhất, mặt nạ chú ý dạng cây (*tree attention mask*) được áp dụng để đảm bảo mỗi nút chỉ nhìn thấy KV cache của tổ tiên hợp lệ trên nhánh của nó.
- **Xác thực song song nhiều token (Multi-token verification):**
  + Khác với decode thông thường ($q = 1$), bước xác thực cây nháp đưa vào lô truy vấn có độ dài $q = k + 1 > 1$ cùng lúc truy vấn vào bảng trang KV cache đã tích lũy.
- **Cơ chế hoàn tác và ghi nhận (Rollback và commit KV cache):**
  + Sau khi mô hình đích chấm điểm và lấy mẫu, chỉ những token được chấp nhận (*accepted tokens*) mới được ghi nhận chính thức vào block table của target model.
  + Toàn bộ các nhánh con bị từ chối (*rejected tokens*) phải bị cắt tỉa khỏi bảng phân bổ block, tránh gây rò rỉ hoặc rác bộ nhớ.

### 4. Áp lực bộ nhớ, an toàn VRAM và rào chắn chống tràn ram (OOM)
- **Tranh chấp bộ nhớ trên phần cứng giới hạn (Tesla T4 16GB):**
  + Trọng số mô hình Qwen2.5-7B lượng tử hóa 4-bit (AWQ/GPTQ) chiếm khoảng $4.5 - 5.0$ GB VRAM.
  + Vùng nhớ còn lại (~11 GB) phải chia sẻ giữa: vùng đệm kích hoạt tạm thời (activations), bảng trang KV cache của các luồng phục vụ, và cây nháp suy đoán.
  + Khi số luồng đồng thời tăng cao ($B = 32$) cùng ngữ cảnh dài ($L \ge 2048$), KV cache phình to chiếm trên $80 - 90\%$ bộ nhớ khả dụng.
- **Rủi ro OOM từ cây nháp tĩnh:**
  + Nếu cấu hình cây nháp tĩnh (luôn sinh $k = 4$ hoặc $k = 8$ token nháp), kích thước tensor kích hoạt và nhu cầu cấp phát block mới tăng đột biến tại các bước xác thực, dẫn đến sụp đổ server vì lỗi `CUDA Out of Memory`.
- **Luật bảo vệ bộ nhớ (Memory Pressure Rule) của đồ án:**
  + Thuật toán lập ngân sách thích ứng giám sát liên tục `kv_cache_usage` và `free_vram_mb`.
  + Ngưỡng kích hoạt: Nếu tỷ lệ dùng KV cache $> 90\%$ hoặc VRAM khả dụng $< 1500$ MB, hệ thống tự động cưỡng chế ngân sách nháp $k = 0$, đưa về giải mã tuần tự để bảo vệ server tuyệt đối không bị OOM.

### 5. Các kỹ thuật tối ưu và nén KV cache nâng cao (đối chiếu học thuật)
- **Cắt tỉa và loại bỏ token (KV cache eviction & pruning):**
  + *H2O (Heavy-Hitter Oracle):* Giữ lại một tỷ lệ nhỏ token nhận nhiều điểm chú ý nhất (heavy hitters) kết hợp với cửa sổ token cục bộ gần nhất.
  + *StreamingLLM (Attention Sinks):* Phát hiện hiện tượng các token đầu tiên nhận lượng điểm chú ý khổng lồ dù ngữ cảnh đã trôi xa; giữ cố định 4 token đầu tiên (*sink tokens*) và một cửa sổ trượt cục bộ giúp mô hình suy luận vô tận mà không bị suy giảm perplexity hay tràn RAM.
  + *SnapKV / Ada-KV:* Tự động định vị và giữ lại các cụm token quan trọng nhất của prompt trước khi bước vào decode.
- **Lượng tử hóa KV cache (KV cache quantization):** Lượng tử hóa tensor Key và Value từ FP16 xuống FP8 (E4M3/E5M2) hoặc INT4/INT8, giảm dung lượng bộ nhớ từ 2 đến 4 lần mà chất lượng câu trả lời gần như không đổi.

### 6. Bảng từ khóa tra cứu về KV cache
| Chủ đề | Từ khóa tìm kiếm học thuật (Keywords) |
| :--- | :--- |
| **Lý thuyết nền** | `Grouped-Query Attention (GQA)`, `LLM decode memory bandwidth bound`, `Arithmetic intensity of Transformer` |
| **Hạ tầng vLLM** | `PagedAttention paper (Woosuk Kwon et al., SOSP 2023)`, `Block table memory allocation vLLM`, `Continuous batching` |
| **KV cache trong suy đoán** | `Speculative decoding tree attention mask`, `Multi-token verification KV cache`, `EAGLE-2 draft tree KV cache` |
| **An toàn bộ nhớ** | `KV cache capacity planning GPU VRAM`, `Speculative decoding memory overhead concurrency`, `vLLM OOM guard` |
| **Nén & Eviction** | `Attention Sinks StreamingLLM`, `H2O Heavy Hitter Oracle KV cache`, `SnapKV`, `Quantized KV Cache FP8` |

---

## Phần 2: Giải mã suy đoán và cơ chế điều khiển thích ứng phần cứng (Adaptive Speculative Decoding)

### 1. Cơ sở lý thuyết của giải mã suy đoán truyền thống
- **Bài toán giải quyết:** Giải mã tự hồi quy của LLM phải tải hàng tỷ trọng số từ VRAM qua từng token đơn lẻ. Giải mã suy đoán giải quyết nút thắt này bằng cách biến một chuỗi các thao tác memory-bound tuần tự thành một thao tác xác thực song song tập trung.
- **Luồng hoạt động 3 bước tiêu chuẩn:**
  1. *Đoán nháp (Drafting):* Sử dụng một mô hình nhỏ, nhẹ, sinh nhanh chuỗi $k$ token nháp tiềm năng.
  2. *Kiểm tra song song (Parallel verification):* Đưa toàn bộ $k$ token nháp vào mô hình lớn (target model) trong đúng một lượt forward duy nhất.
  3. *Lấy mẫu loại bỏ (Rejection sampling):* Duyệt tuần tự từng token nháp theo phân phối xác suất. Chấp nhận các token trùng khớp phân phối và lấy mẫu token thay thế ngay tại vị trí bị từ chối đầu tiên.
- **Chứng minh bảo toàn phân phối (Lossless proof):**
  + Gọi phân phối của target model là $P(x)$ và draft model là $Q(x)$.
  + Xác suất chấp nhận token nháp $x$ là:
    $$\beta = \min\left(1, \frac{P(x)}{Q(x)}\right)$$
  + Nếu token bị từ chối, token mới được lấy mẫu lại từ phân phối hiệu chỉnh:
    $$P'(x) = \frac{\max(0, P(x) - Q(x))}{\sum_{x'} \max(0, P(x') - Q(x'))}$$
  + *Ý nghĩa cốt tử:* Thuật toán đảm bảo phân phối xác suất đầu ra sau suy đoán hoàn toàn trùng khớp $100\%$ với việc sinh trực tiếp từ mô hình đích ban đầu.

### 2. Thiết kế mô hình nháp và tiến hóa dòng họ EAGLE
- **Các tiêu chí thiết kế mô hình nháp:**
  + Tốc độ sinh của draft model phải đủ nhanh ($t_{\text{draft}} \ll t_{\text{target}}$).
  + Tỷ lệ chấp thuận trung bình $\alpha$ phải đủ cao để bù đắp chi phí forward nháp.
  + Điểm hòa vốn (Break-even condition):
    $$\alpha \cdot t_{\text{target}} > t_{\text{draft}} + t_{\text{verify\_overhead}}$$
- **Tiến hóa từ EAGLE-1 đến EAGLE-3:**
  + *EAGLE-1 (Li et al., ICML 2024):* Nháp ở tầng vector biểu diễn đặc trưng (feature-level) của tầng Transformer trên cùng, giải quyết độ bất định ngữ cảnh (feature uncertainty) và không bị phụ thuộc vào tầng embedding của từ vựng.
  + *EAGLE-2 (Li et al., EMNLP 2024):* Bổ sung cơ chế dựng cây nháp động dựa trên độ tự tin của ngữ cảnh (confidence-aware dynamic tree), mở rộng nhánh ở các vị trí có độ tự tin cao và cắt nhánh ở vị trí có độ bất định lớn.
  + *EAGLE-3 (Li et al., 2025):* Tối ưu hóa quá trình huấn luyện đầu nháp bằng kỹ thuật mô phỏng kiểm thử lúc train (training-time test), đạt tốc độ tăng tốc rất cao nhưng yêu cầu chi phí huấn luyện lại lớn.

### 3. Động lực đề tài: Hiện tượng sụt tốc và nút thắt khi tăng tải đồng thời
- **Hiện tượng sụt tốc (Negative acceleration):**
  + Khi chạy các bài toán có entropy cao (suy luận logic khó, sinh câu tự do), tỷ lệ chấp thuận $\alpha$ giảm sâu ($< 0.3$).
  + Hầu hết token nháp bị từ chối, nhưng hệ thống vẫn mất thời gian chạy mô hình nháp $\to$ tốc độ sinh chậm hơn cả chế độ bình thường (speedup $< 1.0\times$).
- **Nút thắt khi tăng concurrency ($B = 1 \to 32$):**
  + Tại $B = 1$, GPU rảnh rỗi tài nguyên tính toán $\to$ speculative decoding đạt hiệu quả tối đa ($2.0\times - 3.0\times$).
  + Khi $B$ tăng lên 16 hoặc 32, GPU đạt ngưỡng bão hòa tính toán (compute saturation). Chi phí forward xác thực cây nháp cho cả lô 32 request tăng vọt, trong khi lợi ích từ việc đọc trọng số bị san sẻ $\to$ biên độ tăng tốc thu hẹp hoặc trở thành gánh nặng.

### 4. Đóng góp cốt lõi: Bộ điều khiển thích ứng (Adaptive Proposer)
- **Mô hình chi phí phần cứng $C(k, B)$:**
  + Khảo sát độ trễ phần cứng trong pha khởi động (startup warm-up profiling) để lập bảng ma trận thời gian thực thi một bước theo ngân sách nháp $k$ và kích thước lô $B$.
- **Ước lượng kỳ vọng số token:**
  $$E[\tau] = 1 + \sum_{i=1}^k \alpha^i$$
- **Hàm mục tiêu tối ưu hóa thời gian thực:**
  + Tại mỗi bước lập lịch, bộ điều khiển tính toán và chọn ngân sách $k^*$ sao cho:
    $$k^* = \arg\max_{k \in \{0, 2, 3, 4\}} \frac{E[\tau]}{C(k, B)}$$
  + Tỷ lệ $\alpha$ được cập nhật tức thời theo trung bình động hàm mũ (Exponential Moving Average - EMA) với hệ số suy giảm $\beta = 0.8$.
- **Cơ chế thoái lui (Graceful fallback):** Khi $\alpha$ tụt dưới ngưỡng hiệu quả ($0.65$), hệ thống tự động chọn $k = 0$ (tắt suy đoán), đảm bảo hệ thống không bao giờ chạy chậm hơn chế độ giải mã truyền thống.

### 5. Kỹ thuật hệ thống và tối ưu kernel trên GPU Tesla T4
- **Đặc thù GPU Tesla T4 (Kiến trúc Turing, Compute capability 7.5):**
  + Bộ nhớ 16GB, không có phần cứng gia tốc thế hệ mới của Ampere/Hopper (không hỗ trợ FP8, Tensor Core FP16 giới hạn băng thông).
  + FlashAttention-2 không được tối ưu hoàn chỉnh cho mọi kích thước tensor tùy ý trên sm_75.
- **Triton Row-wise Segmented Attention:**
  + Module `src/server/t4_rowwise_attention.py` can thiệp trực tiếp vào backend chú ý của vLLM.
  + Khi xác thực cây nháp ($q > 1$), kernel Triton row-wise đảm bảo phân bổ tính toán chú ý chuẩn xác trên bộ nhớ phân đoạn của T4 mà không làm phát sinh chi phí phân bổ tensor tạm thời trên GPU.
- **Chi phí forward tối thiểu của draft model:**
  + Trong vLLM, khi hạ ngân sách $k = 0$, draft model vẫn cần chạy một lượt forward nhỏ để duy trì đồng bộ KV cache của chính nó ($C_D/C_T \approx 0.26$ trên T4). Việc mô hình hóa chính xác chi phí này giúp bộ điều khiển đưa ra quyết định cắt giảm nháp chuẩn xác.

### 6. Rào chắn an toàn bộ nhớ (Memory Guard) & phòng chống OOM
- **Tích hợp với bộ lập lịch vLLM:**
  + Lớp `AdaptiveBudgetPlanner` nhận trực tiếp tín hiệu `kv_usage` từ bộ lập lịch của vLLM và lượng VRAM còn trống từ driver NVIDIA.
- **Nguyên lý kích hoạt rào chắn:**
  + Kiểm tra an toàn trước mỗi bước giải mã:
    $$\text{Nếu } \text{VRAM}_{\text{free}} < 1500\text{ MB} \quad \text{hoặc} \quad \text{KV}_{\text{usage}} > 90\% \implies \text{Ép } k = 0$$
  + Đảm bảo hệ thống vận hành bền bỉ qua hơn 3,000 requests tải kịch trần ở mức concurrency 32 mà không gặp bất kỳ lỗi sập server nào.

### 7. Luồng thực nghiệm đánh giá theo đặc tính tải (Pipeline 3 chặng)
- **Luồng xử lý hóa đơn tiếng Việt (MC-OCR):**
  + *Node 1 (Extractor - Hóa đơn sang JSON):* Cấu trúc thông tin dạng khóa - giá trị cố định, độ bất định thấp $\to \alpha > 0.8 \to$ bộ điều khiển tự động đẩy $k = 4$ để bứt phá tốc độ tối đa ($~3.0\times$).
  + *Node 2 (Text-to-SQL):* Chuyển ngữ tự nhiên sang câu lệnh truy vấn cấu trúc, độ phức tạp trung bình $\to$ bộ điều khiển duy trì $k = 2$ hoặc $3$.
  + *Node 3 (SQL-to-BI Report):* Sinh báo cáo tổng hợp tự do, văn phong linh hoạt, entropy cao $\to \alpha < 0.6 \to$ bộ điều khiển tự động hạ ngân sách hoặc tắt suy đoán ($k = 0$) để loại bỏ hiện tượng sụt tốc.

### 8. Bảng từ khóa tra cứu về giải mã suy đoán thích ứng
| Chủ đề | Từ khóa tìm kiếm học thuật (Keywords) |
| :--- | :--- |
| **Cơ sở toán học** | `Speculative decoding Leviathan paper`, `Rejection sampling distribution matching lossless`, `Break-even latency speculative decoding` |
| **Kiến trúc EAGLE** | `EAGLE feature-level speculative sampling (Li et al., ICML 2024)`, `EAGLE-2 dynamic draft tree`, `EAGLE-3 training-time test` |
| **Thích ứng động** | `Hardware-aware speculative decoding`, `Adaptive draft tree budgeting`, `Cost model speculative decoding latency matrix` |
| **Kỹ thuật phần cứng & vLLM** | `vLLM custom proposer backend`, `Triton attention multi-token verification`, `Turing T4 rowwise attention kernel` |
| **Hệ thống phục vụ** | `Time per output token (TPOT)`, `Inter-token latency (ITL)`, `Continuous batching concurrency sweep` |

---

## Phần 3: Cơ chế tối ưu hóa tính toán chú ý: PagedAttention và FlashAttention

### 1. Cơ chế PagedAttention chuyên sâu
- **Vấn đề cốt tử của các hệ thống phục vụ LLM truyền thống:**
  + Trước khi có vLLM, các framework cấp phát trước một mảng bộ nhớ liên tục cho mỗi request dựa trên độ dài ngữ cảnh tối đa (ví dụ 2048 hoặc 4096 tokens).
  + Thực tế câu trả lời thường ngắn hơn nhiều, dẫn đến **phân mảnh nội bộ (internal fragmentation)** chiếm tới $60 - 80\%$ VRAM.
  + Ngoài ra, việc các request đến và đi liên tục làm bộ nhớ vật lý bị chia cắt thành các mảnh vụn không thể cấp phát cho request mới, gây ra **phân mảnh ngoài (external fragmentation)**.
- **Kiến trúc phân trang bộ nhớ của PagedAttention:**
  + Lấy cảm hứng từ bảng trang (page table) trong hệ điều hành máy tính: Bộ nhớ KV cache vật lý được chia thành các khối cố định gọi là **Physical Blocks** (mỗi block chứa 16 hoặc 32 tokens).
  + Mỗi chuỗi sinh được gán một danh sách **Logical Blocks**, được ánh xạ tới các physical blocks ngẫu nhiên trong VRAM thông qua **Block Table**.
  + Các block vật lý hoàn toàn không cần nằm liên tục nhau trên bộ nhớ.
- **Cơ chế chia sẻ bộ nhớ và sao chép khi ghi (Copy-on-Write):**
  + Khi thực hiện sinh song song nhiều nhánh từ cùng một prompt (Parallel Sampling) hoặc tìm kiếm theo chùm (Beam Search), các chuỗi con ban đầu trỏ chung vào cùng các block vật lý của prompt gốc.
  + Chỉ khi một nhánh sinh ra token mới làm thay đổi nội dung block, vLLM mới tạo bản sao block vật lý riêng cho nhánh đó (Copy-on-Write - COW). Cơ chế này tiết kiệm tới $55\%$ VRAM khi phục vụ các tác vụ phân nhánh.
- **Kernel tính toán PagedAttention trên GPU:**
  + Khi tính toán attention, kernel đọc trực tiếp con trỏ từ block table, nạp từng block không liên tục từ VRAM vào bộ nhớ chia sẻ cực nhanh trên chip (shared memory / SRAM) của GPU để thực hiện phép nhân ma trận và tính toán softmax.

### 2. FlashAttention và kỹ thuật chú ý tối ưu truy xuất bộ nhớ (IO-Aware Attention)
- **Nút thắt IO của phép tính Attention tiêu chuẩn:**
  + Phép tính Attention chuẩn:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  + Trong triển khai chuẩn, các ma trận trung gian $S = QK^T$ và $P = \text{softmax}(S)$ có kích thước $N \times N$. Với ngữ cảnh dài (ví dụ $N = 4096$), ma trận $S$ và $P$ chiếm dung lượng khổng lồ.
  + GPU phải liên tục ghi ma trận $S$ và $P$ ra bộ nhớ băng thông cao (HBM/VRAM), rồi lại đọc ngược vào bộ nhớ chia sẻ (SRAM) để nhân với $V$. Tốc độ bị nghẽn hoàn toàn do băng thông đọc/ghi bộ nhớ (Memory IO), dù năng lực tính toán của nhân Tensor Core vẫn chưa chạy hết công suất.
- **Nguyên lý cốt lõi của FlashAttention (Tri Dao et al.):**
  + **Tiling (Phân khối):** Chia nhỏ các ma trận $Q, K, V$ thành các khối nhỏ (blocks) sao cho vừa vặn với dung lượng SRAM cực nhanh (chỉ từ 100 KB đến vài trăm KB trên mỗi Streaming Multiprocessor - SM).
  + **Online Softmax:** Áp dụng thuật toán tính softmax từng phần (online softmax trick). Khi duyệt qua từng khối của $K$ và $V$, kernel cập nhật dần hệ số chuẩn hóa và kết quả đầu ra $O$ mà không bao giờ cần lưu toàn bộ ma trận $N \times N$ xuống HBM.
- **Tiến hóa của FlashAttention:**
  + *FlashAttention-1:* Đề xuất cơ chế Tiling và Online Softmax, tính toán lại attention trong pha backward (recomputation) thay vì lưu activation.
  + *FlashAttention-2:* Tối ưu hóa phân chia công việc song song trên chiều độ dài chuỗi (sequence length), giảm các phép toán phi ma trận (rescaling non-matmul FLOPs), đạt tới $73\%$ hiệu suất đỉnh của A100.
  + *FlashAttention-3:* Tận dụng phần cứng mới của kiến trúc Hopper (NVIDIA H100) như Tensor Memory Accelerator (TMA) và thực thi bất đồng bộ dạng ma trận (WGMMA), hỗ trợ native FP8.
- **FlashAttention trong pha Prefill và pha Decode:**
  + *Prefill phase:* Độ dài Query $N$ lớn $\implies$ ma trận $N \times N$ rất lớn $\implies$ FlashAttention mang lại hiệu quả tăng tốc vượt trội ($2\times - 4\times$) và giảm bộ nhớ từ $O(N^2)$ xuống $O(N)$.
  + *Decode phase:* Độ dài Query chỉ có 1 token ($q = 1$). Phép tính $QK^T$ thực chất là phép nhân ma trận - vector (GEMV) chứ không phải phép nhân ma trận - ma trận (GEMM). FlashAttention ít phát huy tác dụng ở decode tuần tự thông thường, mà chủ yếu cần đến PagedAttention để nạp nhanh các khối KV cache.
- **Giới hạn phần cứng trên GPU Tesla T4 (Kiến trúc Turing):**
  + T4 có năng lực tính toán (Compute Capability) 7.5, không hỗ trợ đầy đủ các tối ưu hóa phần cứng hiện đại của FlashAttention-2.
  + Khi áp dụng giải mã suy đoán (Speculative Decoding), bước xác thực yêu cầu Query có độ dài $q = k + 1 > 1$ truy vấn vào các block bộ nhớ PagedAttention. vLLM mặc định trên T4 gặp lỗi hoặc suy giảm hiệu năng trên các shape tensor này $\implies$ Đây là lý do đồ án phải áp dụng bản vá **Triton Row-wise Attention** (`src/server/t4_rowwise_attention.py`).

### 3. So sánh PagedAttention và FlashAttention trong phục vụ mô hình
| Tiêu chí | PagedAttention | FlashAttention |
| :--- | :--- | :--- |
| **Mục tiêu chính** | Giải quyết bài toán **quản lý bộ nhớ VRAM** (chống phân mảnh, hỗ trợ chia sẻ). | Giải quyết bài toán **tốc độ truy xuất IO** giữa SRAM và HBM của GPU. |
| **Cơ chế cốt lõi** | Phân trang logic thành các block cố định, dùng block table như hệ điều hành. | Chia khối (tiling) kết hợp thuật toán tính online softmax trực tiếp trong SRAM. |
| **Pha hưởng lợi lớn nhất** | **Decode phase** (quản lý KV cache tích lũy của hàng chục luồng đồng thời). | **Prefill phase** (xử lý prompt dài với ma trận chú ý $N \times N$). |
| **Mối quan hệ trong vLLM** | vLLM kết hợp cả hai: dùng PagedAttention để tổ chức bộ nhớ và dùng kernel FlashAttention/Triton để thực hiện tính toán trên từng trang. |

---

## Phần 4: Lượng tử hóa mô hình (Quantization và Quantized Models)

### 1. Nguyên lý và phân loại lượng tử hóa
- **Khái niệm cơ bản:** Lượng tử hóa là quá trình ánh xạ các trọng số và vector kích hoạt từ kiểu dữ liệu số thực dấu phẩy động độ chính xác cao (FP16 hoặc BF16 - 16 bits) sang kiểu dữ liệu có độ chính xác thấp hơn (INT8, INT4 - 8 hoặc 4 bits, hoặc định dạng FP8).
- **Phân loại theo giai đoạn áp dụng:**
  + *QAT (Quantization-Aware Training):* Đưa sai số lượng tử hóa vào quá trình huấn luyện mô hình ngay từ đầu. Chất lượng tốt nhất nhưng tốn chi phí tính toán khổng lồ, đòi hỏi dữ liệu huấn luyện gốc.
  + *PTQ (Post-Training Quantization):* Lượng tử hóa mô hình đã huấn luyện xong chỉ bằng một tập dữ liệu hiệu chuẩn nhỏ (calibration data, khoảng vài chục đến vài trăm mẫu), không cần đạo hàm huấn luyện lại. Đây là phương pháp được sử dụng trong đồ án.
- **Phân loại theo thành phần được lượng tử hóa:**
  + *Weight-only Quantization (W4A16, W8A16):* Chỉ lượng tử hóa các ma trận trọng số lưu trên VRAM. Khi thực hiện tính toán, trọng số được giải lượng tử (dequantize) nhanh trên SRAM thành FP16 để nhân với activations FP16. Rất hiệu quả cho decode phase (vốn bị nghẽn băng thông nạp trọng số).
  + *Weight-Activation Quantization (W8A8, W4A4):* Lượng tử hóa cả trọng số và tensor kích hoạt. Đòi hỏi GPU có nhân Tensor Core hỗ trợ phép nhân ma trận số nguyên (INT8/INT4 GEMM).
- **Các chế độ ánh xạ:**
  + *Đối xứng (Symmetric):* Ánh xạ đối xứng qua điểm 0 (chỉ cần scale factor $S$).
  + *Bất đối xứng (Asymmetric):* Cho phép dịch điểm 0 (cần cả scale factor $S$ và zero-point $Z$).
  + *Kích thước nhóm (Group size):* Thường chia ma trận thành các nhóm (ví dụ group-size = 128) để tính scale factor riêng, giảm thiểu sai số ngoại lai (outliers).

### 2. Các kỹ thuật lượng tử hóa phổ biến: RTN, GPTQ và AWQ
- **RTN (Round-to-Nearest):**
  + Phương pháp đơn giản nhất: Chia cho tỷ lệ scale rồi làm tròn trực tiếp đến số nguyên gần nhất.
  + Chạy cực nhanh nhưng gây mất mát thông tin trầm trọng; khi đưa mô hình xuống 4-bit, độ bất định perplexity tăng vọt, mô hình mất khả năng trả lời logic.
- **GPTQ (Generalized Post-Training Quantization):**
  + Thuật toán tối ưu hóa bậc hai dựa trên giải thuật Optimal Brain Surgeon.
  + Sử dụng ma trận đạo hàm bậc hai (Hessian matrix) để ước lượng mức độ ảnh hưởng của từng trọng số. Khi làm tròn một cột trọng số sang INT4, thuật toán sẽ bù trừ sai số tích lũy vào các trọng số chưa lượng tử hóa ở các cột tiếp theo.
  + Cho chất lượng mô hình 4-bit rất cao, thời gian lượng tử hóa mất khoảng 1 đến 2 giờ cho mô hình 7B.
- **AWQ (Activation-aware Weight Quantization):**
  + Quan sát đột phá của nhóm tác giả (Lin et al.): Trọng số không có vai trò quan trọng như nhau; chỉ có khoảng **$1\%$ trọng số là trọng yếu (salient weights)** quyết định toàn bộ năng lực mô hình.
  + Điểm mấu chốt: Sự quan trọng của trọng số không nằm ở độ lớn của chính nó, mà nằm ở **biên độ của tensor kích hoạt (activation distribution)** đi qua nó.
  + Giải pháp của AWQ: Nhân tỷ lệ bảo vệ (per-channel scaling) cho $1\%$ kênh trọng số quan trọng này trước khi thực hiện lượng tử hóa, giúp giảm thiểu tối đa sai số làm tròn mà không cần giải bài toán nghịch đảo Hessian phức tạp. Mô hình AWQ giữ được chất lượng tương đương FP16 trên hầu hết các bài kiểm thử ngôn ngữ.

### 3. Vai trò của lượng tử hóa trong bài toán thực tế của đồ án
- **Lý do bắt buộc phải dùng mô hình 4-bit (AWQ/GPTQ):**
  + Phần cứng mục tiêu của đồ án là GPU **Tesla T4 16GB**.
  + Mô hình gốc Qwen2.5-7B ở định dạng FP16 chiếm khoảng $14.5$ GB VRAM. Nếu nạp nguyên bản, GPU chỉ còn khoảng $1.5$ GB trống $\implies$ vLLM sẽ báo lỗi OOM ngay khi concurrency $B > 4$, và hoàn toàn không còn chỗ chứa cây nháp của speculative decoding.
  + Khi lượng tử hóa 4-bit (W4A16), kích thước mô hình co lại chỉ còn **$4.5 - 5.0$ GB VRAM**, giải phóng hơn **$10$ GB VRAM** cho bảng trang PagedAttention và các tensor kích hoạt. Điều này cho phép đồ án quét trọn vẹn thang concurrency từ $B = 1$ đến $B = 32$.
- **Tác động của Quantization lên giải mã suy đoán:**
  + Mô hình đích (Target) chạy ở dạng 4-bit AWQ, trong khi đầu nháp EAGLE-2 (Draft) chạy ở dạng FP16 hoặc FP8.
  + Cơ chế lấy mẫu loại bỏ (Rejection Sampling) vẫn đảm bảo tính chất **không tổn hao (lossless)**: Các token sinh ra sau suy đoán có phân phối xác suất hoàn toàn đồng nhất với mô hình đích 4-bit gốc.

---

## Phần 5: Các chỉ số đo lường hiệu năng mô hình ngôn ngữ lớn (Benchmarking Metrics with vLLM)

### 1. Nhóm chỉ số đo lường độ trễ (Latency metrics)
- **TTFT (Time To First Token - Thời gian đến token đầu tiên):**
  + Đo bằng mili-giây (ms). Là khoảng thời gian tính từ khi client gửi prompt đến khi nhận được token đầu tiên trả về.
  + Phản ánh tốc độ của **pha prefill**. Rất quan trọng với trải nghiệm người dùng tương tác trực tiếp (chatbot).
- **TPOT (Time Per Output Token - Thời gian trên mỗi token sinh ra):**
  + Đo bằng ms/token. Là thời gian trung bình để sinh ra một token mới trong **pha decode**:
    $$\text{TPOT} = \frac{\text{Thời gian sinh toàn bộ câu}}{\text{Số lượng token đầu ra}}$$
  + **Đây là mục tiêu tối ưu hóa số một của giải mã suy đoán.** Giải mã suy đoán sinh được nhiều token trên một bước xác thực, do đó kéo tụt chỉ số TPOT xuống thấp nhất có thể.
- **ITL (Inter-Token Latency - Độ trễ giữa các token liên tiếp):**
  + Đo thời gian ngắt quãng giữa hai token xuất hiện liên tiếp nhau trong luồng streaming.
  + Trong giải mã suy đoán, khi một cây nháp được chấp nhận $k$ token cùng lúc, các token này được trả về gần như đồng thời ($ITL \approx 0$), sau đó có một khoảng chờ xác thực tiếp theo. Độ lệch chuẩn của ITL phản ánh độ ổn định của luồng sinh.
- **E2E Latency (End-to-End Latency - Độ trễ toàn trình):**
  + Tổng thời gian hoàn thành từ đầu đến cuối một yêu cầu:
    $$\text{E2E Latency} = \text{TTFT} + (\text{Số token đầu ra} - 1) \times \text{TPOT}$$

### 2. Nhóm chỉ số đo lường thông lượng và hiệu quả tăng tốc (Throughput & Speedup)
- **Output Token Throughput (Thông lượng token đầu ra):**
  + Đo bằng tokens/s. Tổng số token sinh ra được trên toàn bộ các luồng trong một giây:
    $$\text{Throughput} = \frac{\text{Tổng số token sinh ra của tất cả request}}{\text{Tổng thời gian thực thi}}$$
  + Chỉ số này tăng dần theo concurrency và đạt điểm bão hòa khi GPU chạm ngưỡng tính toán.
- **Request Throughput (Thông lượng yêu cầu / RPS):**
  + Số request hoàn thành xử lý trên mỗi giây (Requests Per Second).
- **Hệ số gia tốc (Speedup Ratio):**
  + Đo lường mức độ tăng tốc của giải mã suy đoán so với giải mã truyền thống (Vanilla):
    $$\text{Speedup} = \frac{\text{TPOT}_{\text{vanilla}}}{\text{TPOT}_{\text{speculative}}} \quad \text{hoặc} \quad \frac{\text{Latency}_{\text{vanilla}}}{\text{Latency}_{\text{speculative}}}$$
  + Nếu $\text{Speedup} > 1.0\times$: Tăng tốc thành công.
  + Nếu $\text{Speedup} < 1.0\times$: Xảy ra hiện tượng **sụt tốc (negative acceleration)**.

### 3. Nhóm chỉ số đặc thù của giải mã suy đoán và tính an toàn hệ thống
- **Tỷ lệ chấp thuận $\alpha$ (Acceptance Rate):**
  + Tỷ lệ phần trăm token nháp được mô hình đích chấp nhận:
    $$\alpha = \frac{\text{Số token nháp được chấp nhận}}{\text{Tổng số token nháp do draft model sinh ra}}$$
  + Phụ thuộc vào chất lượng của đầu nháp EAGLE-2 và độ phức tạp (entropy) của văn bản đầu vào.
- **Số token chấp thuận trung bình trên mỗi bước ($\tau$):**
  + Mỗi bước xác thực, nếu chấp nhận $m$ token nháp thì hệ thống sinh được tổng cộng $m + 1$ token (bao gồm cả token bổ chính từ target model).
  + $\tau$ càng lớn thì hệ thống chạy càng nhanh.
- **Đỉnh tiêu thụ VRAM (VRAM Peak):**
  + Dung lượng bộ nhớ lớn nhất bị chiếm dụng trong suốt phiên kiểm thử tải nặng (đo bằng MB hoặc GB qua `nvidia-smi` hoặc thư viện NVML).
  + Khẳng định tính an toàn bộ nhớ: Hệ thống không bao giờ được chạm ngưỡng 16GB để tránh OOM.
- **Phân vị độ trễ (P50, P90, P99 Latency & Variance):**
  + Phản ánh tính nhất quán của hệ thống phục vụ. P99 Latency đo độ trễ của $1\%$ các request chạy chậm nhất.
  + Khi concurrency tăng lên 32, hiện tượng tranh chấp block PagedAttention làm tăng phương sai TPOT (đã được ghi nhận trong thực nghiệm đồ án).

### 4. Công cụ và quy trình đo lường thực nghiệm trong đồ án
- **Công cụ đo lường:**
  + `vllm benchmark_serving`: Kịch bản bắn tải đa luồng tiêu chuẩn của vLLM, giả lập người dùng gửi request theo phân phối Poisson hoặc tải cố định để đo throughput, TTFT, TPOT ở các mức concurrency $B \in [1, 32]$.
  + `Langfuse`: Nền tảng quan trắc (observability) gắn trực tiếp vào pipeline 3 chặng để ghi nhận chi tiết độ trễ, số token và chi phí thực thi của từng node (Node 1 Extractor, Node 2 Text-to-SQL, Node 3 BI Report).
- **Quy trình đo lường chuẩn hóa 3 làn so sánh (Three-lane Benchmark):**
  1. *Làn 1 (Vanilla Baseline):* Qwen2.5-7B AWQ giải mã tự hồi quy thông thường ($k = 0$).
  2. *Làn 2 (Stock EAGLE-2):* vLLM tích hợp EAGLE-2 với cây nháp tĩnh cấu hình cố định.
  3. *Làn 3 (Adaptive Controller):* vLLM tích hợp bộ điều khiển thích ứng phần cứng tự động chỉnh $k$ theo thời gian thực và kích hoạt rào chắn bảo vệ VRAM.

---

## Phần 6: Ma trận đối chiếu câu hỏi phản biện bảo vệ đồ án

| Câu hỏi tiềm năng của hội đồng phản biện | Mảng kiến thức cốt lõi cần dùng để trả lời |
| :--- | :--- |
| *1. "Tại sao giải mã suy đoán sinh ra kết quả nhanh hơn mà không làm giảm độ chính xác hoặc đổi nghĩa của mô hình gốc?"* | Trình bày toán học về **Rejection sampling của Leviathan** (Phần 2, Mục 1) và công thức bù trừ phân phối xác suất giữa $P(x)$ và $Q(x)$. |
| *2. "Tại sao không dùng sẵn cơ chế Speculative Decoding mặc định của vLLM mà nhóm phải chế tạo Adaptive Proposer?"* | Nêu rõ 2 nhược điểm chết người của cây nháp tĩnh: **sụt tốc (negative acceleration)** khi $\alpha$ thấp và **tràn ram (OOM)** khi concurrency cao (Phần 2, Mục 3 & 4). |
| *3. "Cơ chế PagedAttention giải quyết bài toán gì của hệ thống phục vụ LLM?"* | Trình bày cơ chế **phân trang logic thành các block cố định**, bảng trang block table loại bỏ hoàn toàn phân mảnh ngoài và kiểm soát phân mảnh nội bộ dưới $4\%$ (Phần 3, Mục 1). |
| *4. "Tại sao FlashAttention tăng tốc vượt trội ở prefill nhưng lại ít phát huy tác dụng ở bước decode thông thường?"* | Giải thích sự khác biệt giữa phép toán ma trận - ma trận (GEMM) ở prefill và phép toán ma trận - vector (GEMV) ở decode; decode chủ yếu bị memory-bandwidth bound nạp KV cache (Phần 3, Mục 2). |
| *5. "Tại sao nhóm chọn phương pháp lượng tử hóa AWQ 4-bit thay vì RTN hay FP16?"* | Giải thích vì sao FP16 gây OOM trên T4 16GB, RTN làm hỏng chất lượng, và **AWQ bảo vệ 1% trọng số quan trọng dựa trên phân phối kích hoạt** giúp giữ nguyên độ chính xác (Phần 4, Mục 2 & 3). |
| *6. "Nhóm đánh giá hiệu năng mô hình bằng các chỉ số nào và tại sao TPOT lại quan trọng hơn TTFT đối với đề tài này?"* | Nêu định nghĩa TTFT, TPOT, E2E Latency và phân tích rằng speculative decoding nhắm thẳng vào việc **giảm chu kỳ nạp trọng số ở pha decode**, do đó TPOT là thước đo trực tiếp nhất (Phần 5, Mục 1 & 2). |
| *7. "Bản vá Triton Row-wise Attention của nhóm can thiệp vào tầng nào và giải quyết hạn chế gì của GPU T4?"* | Nêu hạn chế kiến trúc Turing sm_75 khi xác thực cây nháp có độ dài truy vấn $q > 1$ trên các block PagedAttention không liên tục (Phần 2, Mục 5 và Phần 3, Mục 2). |
| *8. "Làm thế nào hệ thống đảm bảo không bao giờ bị OOM khi chạy 32 luồng đồng thời?"* | Trình bày **Memory Pressure Rule**: liên tục đọc `kv_cache_usage` và `free_vram_mb`, cưỡng chế hạ $k = 0$ khi KV cache $> 90\%$ hoặc VRAM $< 1500$ MB (Phần 1, Mục 4 và Phần 2, Mục 6). |
