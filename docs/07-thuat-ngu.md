# Thuật ngữ & ký hiệu

Bảng quy chiếu dùng **thống nhất** trong mã nguồn, tài liệu và quyển báo cáo. Ký hiệu nào không có ở đây thì đừng dùng — hãy thêm vào đây trước.

## Ký hiệu chính

| Ký hiệu | Tên | Định nghĩa | Đơn vị |
|:--:|---|---|---|
| `α` | **Tỉ lệ chấp nhận** (acceptance rate) | Tỉ lệ token nháp được mô hình đích chấp nhận. **Quan trọng:** đây là tỉ lệ *trên mỗi token*, không phải trên mỗi bước | 0–1 |
| `τ` | **Độ dài chấp nhận trung bình** (mean accepted length) | Số token sinh ra trung bình **mỗi lượt gọi mô hình đích**. Luôn ≥ 1 | token |
| `K` | Độ dài chuỗi nháp | Số token nháp trong một vòng (speculative decoding tuyến tính) | token |
| `N` | **Ngân sách cây** | Tổng số node trong cây nháp đưa đi xác thực | token |
| `D` | Độ sâu cây | Số tầng của cây nháp | tầng |
| `t_draft` | Độ trễ nháp | Thời gian một lượt chạy của bộ nháp | ms |
| `t_verify` | Độ trễ xác thực | Thời gian một lượt truyền xuôi của mô hình đích | ms |
| `c` | Tỉ lệ chi phí | `t_verify / t_draft` — **đại lượng quyết định suy đoán có lời hay không** | — |

### Ghi chú về `τ` và `MAT`

Các paper gọi cùng một đại lượng bằng hai tên: **EAGLE/EAGLE-2 gọi là `τ`** (average acceptance length), **[Talon](paper_summary/04-talon-liu2026.md) gọi là `MAT`** (Mean Accepted Tokens). **Trong đồ án này luôn dùng `τ`.** Khi trích số từ Talon thì đổi tên về `τ` và ghi chú.

Đừng nhầm `α` với `τ`: `α` là xác suất một token nháp sống sót; `τ` là *số token thu được mỗi lượt*. Với chuỗi tuyến tính, `τ = (1 − α^(K+1)) / (1 − α)`. Với cây thì không có công thức đóng — phải đo.

## Chỉ số phục vụ (serving)

| Ký hiệu | Tên đầy đủ | Nghĩa | Vai trò trong đồ án |
|:--:|---|---|---|
| **TTFT** | Time To First Token | Từ lúc gửi request tới token đầu tiên | Ràng buộc — **không được tệ đi** |
| **TPOT** | Time Per Output Token | Thời gian trung bình mỗi token sau token đầu | **Mục tiêu tối ưu chính** |
| **ITL** | Inter-Token Latency | Khoảng cách giữa hai token liên tiếp | Bản chất giống TPOT; ITL là chuỗi, TPOT là trung bình |
| **E2E** | End-to-end latency | Toàn bộ thời gian một request / một pipeline | Thước đo người dùng thật cảm nhận |
| **Throughput** | — | Token/giây ở cấp máy chủ, cộng toàn batch | Trục phụ — nơi suy đoán thường **thua** |

Quan hệ: `E2E ≈ TTFT + TPOT × (số token ra − 1)`.

## Khái niệm

| Thuật ngữ | Nghĩa |
|---|---|
| **Lossless** | Phân phối đầu ra **không đổi** so với giải mã thường. Hệ quả của tiêu chí chấp nhận, không phải của việc đo chất lượng |
| **Target model** (`M_p`) | Mô hình đích — mô hình chất lượng cao, chạy chậm, là chuẩn đúng |
| **Draft model / Draft head** (`M_q`) | Cơ chế nháp — nhẹ, nhanh, đề xuất trước |
| **Acceptance Criterion** | Thuật toán duyệt token nháp (modified rejection sampling) giữ nguyên phân phối. **Không được đụng vào** |
| **Bonus token** | Token miễn phí thu được khi cả chuỗi nháp được chấp nhận |
| **Draft tree** | Nhiều nhánh nháp song song thay cho một chuỗi tuyến tính |
| **Tree attention mask** | Mặt nạ chú ý cho phép xác thực cả cây trong một lượt, các nhánh không thấy nhau |
| **Memory-bandwidth bound** | GPU chờ nạp trọng số, phần tính toán rảnh — **điều kiện để suy đoán có lời** |
| **Compute bound** | GPU bận tính toán — suy đoán **mất lợi thế**, có thể phản tác dụng |
| **Negative acceleration** | Speculative decoding chạy **chậm hơn** giải mã thường (speedup < 1.0×) |
| **Miscalibration** | Độ tự tin của bộ nháp không phản ánh đúng xác suất được chấp nhận |
| **Prefill / Decode** | Hai pha: xử lý prompt (song song, compute-bound) / sinh từng token (tuần tự, bandwidth-bound). **Suy đoán chỉ tăng tốc pha decode** |

## Ba làn so sánh (dùng đúng tên này trong mọi bảng/biểu đồ)

| Tên làn | Nghĩa |
|---|---|
| `vanilla` | Giải mã tự hồi quy thường, không suy đoán — **mẫu số của mọi con số speedup** |
| `static` | EAGLE-2 với cấu hình cây cố định (mặc định của vLLM) |
| `adaptive` | Adaptive Proposer của đồ án |

## Quy ước viết số

- **Speedup** luôn so với `vanilla` **trên cùng phần cứng, cùng workload, cùng thang concurrency**. Speedup so với làn khác thì phải ghi rõ.
- Ghi **trung vị + khoảng biến thiên** qua nhiều lần chạy, không ghi một con số trần trụi (wall-clock trên GPU chia sẻ rất nhiễu — xem [06](06-ke-hoach-thuc-nghiem.md)).
- Con số trích từ paper phải ghi rõ **"tác giả tự báo"** kèm phần cứng của họ. Không đặt cạnh số tự đo mà không chú thích.

## Thuật ngữ Anh–Việt (dùng nhất quán trong quyển)

| English | Tiếng Việt dùng trong quyển |
|---|---|
| speculative decoding | giải mã suy đoán |
| draft / drafting | nháp / sinh nháp |
| verification | xác thực |
| acceptance rate | tỉ lệ chấp nhận |
| draft tree | cây nháp |
| lossless | không đổi đầu ra (giữ nguyên từ *lossless* khi cần ngắn gọn) |
| throughput | thông lượng |
| latency | độ trễ |
| quantization | lượng tử hoá |
| profiling | đo hồ sơ (phần cứng) |
