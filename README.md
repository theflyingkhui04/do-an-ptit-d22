<div align="center">

<img src="logo/flexsds_banner.gif" alt="PTIT - FlexSDS Banner" width="100%" />

# FlexSDS: Hệ thống phục vụ suy luận phỏng đoán thích ứng nhận biết phần cứng

**Bộ công cụ phục vụ suy luận phỏng đoán thông minh cho các mô hình ngôn ngữ lớn tự lưu trữ**  
*Hardware-Aware Adaptive Speculative Decoding Serving Toolkit for Self-Hosted LLMs*

[![PTIT](https://img.shields.io/badge/PTIT-Đồ%20án%20tốt%20nghiệp%20D22-ED1C24?style=for-the-badge&logoColor=white)](https://portal.ptit.edu.vn/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.30.0%20Core%20V1-blue?style=for-the-badge)](https://github.com/vllm-project/vllm)
[![Modal](https://img.shields.io/badge/Modal-GPU%20L4%20Serverless-00C7B7?style=for-the-badge)](https://modal.com)
[![Status](https://img.shields.io/badge/Trạng%20thái-Sprint%205%20%2F%20M4%20Freeze-brightgreen?style=for-the-badge)]()

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/icon_dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="logo/icon_white.webp">
  <img alt="FlexSDS Logo" src="logo/icon_dark.webp" width="90" height="90">
</picture>

</div>

---

## 1. Tổng quan đề tài

Trong các hệ thống xử lý tài liệu tự động và chuỗi tác vụ đa bước (multi-step LLM pipeline), **độ trễ sinh từ trong pha giải mã (Decode Latency)** là nút thắt cổ chai lớn nhất và bị tích lũy lũy tiến qua từng lời gọi mô hình. Kỹ thuật suy luận phỏng đoán truyền thống (Static Speculative Decoding) sử dụng cấu hình nháp cố định thường gặp phải hai rào cản nghiêm trọng trong môi trường triển khai thực tế:
1. **Hiện tượng tăng tốc âm (Negative Speedup):** Khi gặp các tác vụ suy luận tự do có entropy cao hoặc khi hệ thống chịu tải đồng thời lớn ($c \ge 16$), việc ép chạy mô hình nháp vô ích biến nó thành tải ký sinh, làm giảm tốc độ thực tế (chỉ đạt $0.67\times - 0.81\times$ so với suy luận tuần tự cơ sở).
2. **Nguy cơ tràn bộ nhớ (Out-Of-Memory - OOM):** Việc đồng thời nạp cả mô hình đích, mô hình nháp và phân bổ bộ đệm KV dễ dẫn đến tràn VRAM trên các dòng GPU phổ thông (NVIDIA Tesla T4 16GB, RTX 4090 24GB).

**FlexSDS** (*Flexible Speculative Decoding Serving*) là giải pháp mã nguồn mở hoàn chỉnh do nhóm sinh viên PTIT nghiên cứu và phát triển. Hệ thống tích hợp lớp điều khiển thích ứng **TreeAdaptive**, tự động nhận diện phần cứng, nạp ma trận chi phí thực nghiệm và giải ngưỡng hòa vốn động $\alpha^*(N, B)$ theo thời gian thực để co giãn cấu trúc cây nháp hoặc ngắt về suy luận tuần tự cơ sở, **bảo đảm tăng tốc tối đa nhưng không bao giờ bị chậm hơn chế độ mặc định (Lossless Guarantee)**.

---

## 2. Ứng dụng thực nghiệm: Luồng đa tác vụ trên hóa đơn bán hàng

Để kiểm chứng năng lực thích ứng trong môi trường thực tế, FlexSDS được triển khai phục vụ quy trình xử lý và phân tích hóa đơn bán hàng tiếng Việt (bộ dữ liệu MC-OCR) với 3 tác vụ đại diện:

```text
┌────────────────────────────────────────────────────────────────────────┐
│             HỆ THỐNG ĐA TÁC VỤ TRÊN HÓA ĐƠN BÁN HÀNG                   │
├────────────────────────────────────────────────────────────────────────┤
│                      [Ảnh / Thư mục hóa đơn tải lên]                   │
│                                     │                                  │
│                                     ▼                                  │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ TÁC VỤ 1: TRÍCH XUẤT NỘI DUNG HÓA ĐƠN THÀNH JSON (MC-OCR W1)       │ │
│ │ • Trích xuất 7 trường pháp lý: Người bán, MST, Ngày, Tiền, VAT...   │ │
│ │ • Cú pháp JSON cố định, alpha > 90% -> Tăng tốc tối đa (1.35x-1.4x) │ │
│ └──────────────────────────────────┬─────────────────────────────────┘ │
│                                    │                                   │
│                                    ▼                                   │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ TÁC VỤ 2: SINH MÃ SQL TRUY VẤN DỮ LIỆU HÓA ĐƠN (TEXT-TO-SQL W2)   │ │
│ │ • Chuyển câu hỏi quản trị kinh doanh tiếng Việt sang truy vấn SQL  │ │
│ │ • Cú pháp SQL bán cấu trúc, alpha ~ 80% -> Tăng tốc ổn định 1.3x    │ │
│ └──────────────────────────────────┬─────────────────────────────────┘ │
│                                    │                                   │
│                                    ▼                                   │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ TÁC VỤ 3: HỎI ĐÁP & PHÂN TÍCH TÀI CHÍNH THÔNG MINH (INVOICE CHAT) │ │
│ │ • Khung chat tương tác trực tiếp về các hóa đơn vừa xử lý          │ │
│ │ • Văn bản tự do, alpha < 60% -> Tự ngắt nháp (k=0), tránh trễ âm   │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Các đột phá kỹ thuật nổi bật

* **Lớp thích ứng nhận biết phần cứng (Hardware-Aware TreeAdaptive v0.2):** Tự động nhận diện kiến trúc GPU (`sm_89` NVIDIA L4, `sm_75` Tesla T4) và giải nghiệm phương trình hòa vốn theo thời gian thực:
  $$\alpha^*(N, B) = \arg_\alpha \left( \frac{\mathbb{E}[\tau(N, \alpha)]}{C(N, B)} = \frac{1}{C(0, B)} \right)$$
* **Triệt tiêu hoàn toàn hiện tượng tăng tốc âm:** Khi tỷ lệ chấp nhận thực tế $\alpha < \alpha^*$ hoặc khi bộ đệm KV vượt ngưỡng $90\%$, hệ thống lập tức ngắt suy đoán về chế độ an toàn ($N=0$), bảo đảm tốc độ luôn $\ge 1.00\times$.
* **Bộ gác cổng an toàn tương thích (Pre-flight Guard):** Quét dung lượng VRAM thực tế, kiểm tra độ lệch từ vựng (Heterogeneous Vocabulary) giữa mô hình đích và mô hình nháp trước khi khởi động tiến trình serving.
* **Đa nền tảng phục vụ:** Hỗ trợ khởi chạy server vLLM cục bộ (Local Dedicated Runner) hoặc triển khai tức thì lên nền tảng đám mây không máy chủ (Modal Cloud GPU L4) với chuẩn giao tiếp OpenAI API `/v1/chat/completions`.
* **Giao diện dòng lệnh tương tác (Interactive CLI Wizard):** Menu hỏi–đáp từng bước có nhận diện thương hiệu PTIT bằng ký tự khối Unicode và mã màu ANSI TrueColor 24-bit.

---

## 4. Bốn chế độ suy luận đối chứng (4 Serving Lanes)

Hệ thống cung cấp cơ chế đo kiểm thực nghiệm chuẩn hóa giữa 4 làn phục vụ trên cùng một hạ tầng phần cứng:

1. **Làn 1 (`vanilla`):** Giải mã tuần tự cơ sở của mô hình đích Qwen2.5-7B-Instruct (chuẩn đối chứng $1.00\times$).
2. **Làn 2 (`static`):** Suy luận phỏng đoán tĩnh với mô hình nháp độc lập Qwen2.5-0.5B-Instruct ($k=3$ cố định).
3. **Làn 3 (`adaptive`):** Suy luận phỏng đoán thích ứng động TreeAdaptive co giãn ngân sách nháp theo tải và $\alpha$.
4. **Làn 4 (`eagle`):** Đầu nháp suy đoán chuyên biệt EAGLE-3.1 tự huấn luyện cho miền hóa đơn tiếng Việt (`duckhoi/qwen2.5-7b-eagle3.1-speculator`).

---

## 5. Hướng dẫn cài đặt và sử dụng nhanh

### 5.1. Cài đặt package

```bash
# Clone kho mã nguồn từ GitHub
git clone https://github.com/theflyingkhui04/do-an-ptit-d22.git
cd do-an-ptit-d22

# Cài đặt môi trường ở chế độ phát triển (Editable Mode)
pip install -e .
```

### 5.2. Khởi chạy giao diện dòng lệnh tương tác (Interactive Wizard)

Chỉ cần gõ lệnh `flexsds` trên terminal để khởi động trợ lý hướng dẫn từng bước:

```bash
flexsds
```

Trợ lý sẽ tự động:
1. Quét thông số phần cứng GPU và hiển thị banner nhận diện thương hiệu PTIT.
2. Hướng dẫn chọn tác vụ, mô hình đích (Target), mô hình nháp (Draft).
3. Chạy Pre-flight Guard kiểm tra an toàn VRAM và xung đột bảng từ vựng.
4. Tự động nạp ma trận chi phí thực nghiệm và khởi chạy server phục vụ.

### 5.3. Sử dụng lệnh trực tiếp (Direct CLI Commands)

```bash
# Quét nhanh phần cứng và trạng thái VRAM
flexsds detect

# Khởi chạy endpoint serving trên nền tảng Modal Cloud
flexsds serve --backend modal --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft Qwen/Qwen2.5-0.5B-Instruct

# Khởi chạy server phục vụ cục bộ trên GPU máy chủ
flexsds serve --backend local --port 8000 --gpu-memory-utilization 0.90
```

---

## 6. Web demo và bảng đo lường metrics đối chứng

Hệ thống tích hợp giao diện web demo trực quan hỗ trợ:
* Tải lên một hoặc nhiều ảnh hóa đơn (kéo thả cả thư mục hoặc chụp ảnh trực tiếp).
* Lựa chọn 3 tác vụ xử lý độc lập: Trích xuất JSON, Sinh câu truy vấn SQL, hoặc Chat hỏi đáp tài chính.
* **Thiết lập đa endpoint song song:** Cho phép thêm URL của nhiều server (Vanilla vs. Static vs. FlexSDS Adaptive) để chạy đồng thời cùng một dữ liệu đầu vào.
* **Bảng so sánh metrics thời gian thực:**
  * **TPOT (Time per Output Token - ms/token):** Đo tốc độ sinh từ thực tế của từng endpoint.
  * **Throughput (tokens/s):** Tốc độ xả token ra màn hình.
  * **Tỉ lệ chấp nhận nháp $\alpha$ (%):** Phản ánh đặc trưng ngữ nghĩa của tác vụ.
  * **Trạng thái thích ứng:** Huy hiệu hiển thị quyết định co giãn cây của FlexSDS (`[Boost: k=4]`, `[Balanced: k=2]`, hoặc `[Fallback: k=0]`).
  * **Hệ số tăng tốc (Speedup vs. Vanilla):** Trực quan hóa hiện tượng tăng tốc âm của chế độ tĩnh và khả năng duy trì tốc độ an toàn của FlexSDS.

---

## 7. Cấu trúc kho mã nguồn

```text
do-an-ptit-d22/
├── logo/                              # Tài nguyên hình ảnh, logo PTIT và banner GIF
│   ├── flexsds_banner.gif             # Banner động độ phân giải cao hiển thị trên README
│   ├── flexsds_banner_static.webp     # Banner tĩnh
│   ├── icon_dark.webp                 # Logo biểu tượng chế độ nền tối (Dark mode)
│   └── icon_white.webp                # Logo biểu tượng chế độ nền sáng (Light mode)
├── pyproject.toml                     # Cấu hình package Python và entrypoint dòng lệnh flexsds
├── src/
│   ├── flexsds/                       # Gói mã nguồn công cụ FlexSDS
│   │   ├── cli/                       # Giao diện CLI, Interactive Wizard và banner ANSI
│   │   ├── core/                      # Bộ quét phần cứng, bộ gác cổng an toàn & TreeAdaptive
│   │   ├── profiles/                  # Danh mục ma trận chi phí thực nghiệm (GPU L4 & T4)
│   │   └── server/                    # Bộ điều phối runner (Modal serverless & Local vLLM)
│   ├── server/                        # Bộ lập lịch thích ứng vLLM AdaptiveAsyncScheduler
│   ├── proposer/                      # Bộ điều khiển nháp và mô hình chi phí thực nghiệm
│   └── pipeline/                      # Đường ống xử lý hóa đơn MC-OCR (JSON, SQL, Report)
├── experiments/                       # Bộ công cụ đo benchmark 4 làn và kịch bản thực nghiệm
├── docs/                              # Tài liệu kỹ thuật, quy định thiết kế và outline báo cáo
└── tests/                             # Bộ kiểm thử tự động toàn diện (unit & regression tests)
```

---

## 8. Đội ngũ thực hiện

Nhóm áp dụng mô hình **Ghép cặp Luân phiên 70/30 (Pair-Rotation System)**: Mỗi thành viên chủ trì (Lead) 70% mảng chuyên môn cốt lõi, đồng thời luân phiên đồng hành (Partner) 30% ở các mảng còn lại:

| Chuyên môn cốt lõi (Lead 70%) | Thành viên thực hiện | Vai trò trong dự án |
| :--- | :--- | :--- |
| **AI Systems & Core-Engine** | **Cấn Đức Khôi** ([@theflyingkhui04](https://github.com/theflyingkhui04)) | Lead Architecture & Core vLLM Scheduler |
| **MLOps & Infrastructure** | **Bùi Đức Đại** | Lead Benchmark Harness & GPU Telemetry |
| **AI Application & Data** | **Dương Nguyễn Thế** | Lead Pipeline MC-OCR & Invoices Analytics |
| **Giảng viên hướng dẫn** | *(Cập nhật theo quyết định của khoa)* | Cố vấn khoa học & Giám sát chất lượng |

---

## 9. Bản quyền

Dự án được phát triển phục vụ đồ án tốt nghiệp đại học tại **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**, niên khóa 2022–2026. Mã nguồn được phát hành theo giấy phép mã nguồn mở Apache 2.0.
