# Kịch bản sử dụng chuẩn cho kỹ năng tác tử thông minh FlexSDS (Happy path runbook)

> **Tài liệu:** Kịch bản sử dụng FlexSDS Assistant
> **Dự án:** Đồ án tốt nghiệp đại học — Ngành Công nghệ Thông tin, PTIT  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead), Bùi Đức Đại (Reviewer)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/flexsds-skill/happy-path.md`  
> **Ngày lập:** 12/11/2026  
> **Hệ sinh thái:** Chuẩn Open Agent Skill (`skills.sh`, Claude Code, Antigravity, OpenAI Codex, Cursor)  
> **Trạng thái:** Tài liệu đặc tả vận hành chính thức  

---

## 1. Mục đích tài liệu

Tài liệu này đặc tả chi tiết kịch bản sử dụng chuẩn (**Happy path**) khi một người dùng tương tác bằng ngôn ngữ tự nhiên với một tác tử lập trình AI (AI Coding Agent như Claude Code, Antigravity, Codex...) được tích hợp kỹ năng **FlexSDS Assistant** (`flexsds-assistant`).

Khác với CLI truyền thống nơi con người phải tự gõ từng lệnh và chọn từng menu, kỹ năng tác tử cho phép:
1. **Tự động nhận diện phần cứng (Zero-config audit):** Tác tử tự chạy công cụ quét GPU, đo VRAM và kiến trúc SM.
2. **Kiểm tra an toàn trước khi tải (Pre-download safety check):** Tác tử tự đọc metadata Hugging Face (<10 KB), ngăn ngừa nguy cơ tràn bộ nhớ (OOM) và lãng phí băng thông mạng.
3. **Ước lượng tốc độ tăng tốc phân tích (Speedup forecasting):** Tác tử áp dụng mô hình toán học chi phí phần cứng và tỷ lệ chấp nhận token của EAGLE để tính trước con số tăng tốc ($S$) so với vLLM nguyên bản (vanilla baseline) trước khi khởi chạy.
4. **Tự động điều phối và cấu hình (Auto-tuning & execution):** Tác tử tự chọn tham số phục vụ tối ưu (ngân sách $k^*$, định dạng lượng tử hóa AWQ, cổng mạng), hỗ trợ hướng dẫn lấy API key Modal BYOC hoặc liên kết Google Colab.
5. **Giám sát và kiểm chứng sức khỏe (Healthcheck & verification):** Tác tử tự gửi truy vấn thử nghiệm mẫu và dọn dẹp tiến trình khi kết thúc.

---

## 2. Luồng tương tác ba bên (Tripartite interaction model)

```
┌─────────────────┐       Ngôn ngữ tự nhiên       ┌────────────────────────┐
│   Người dùng    │ ────────────────────────────> │  AI Agent (Claude /    │
│  (Human User)   │ <──────────────────────────── │ Antigravity / Codex)   │
└─────────────────┘       Báo cáo & xác nhận      └────────────────────────┘
                                                              │
                                       Gọi công cụ ngầm       │ Trả kết quả JSON
                                     (CLI --json / Scripts)   ▼ có cấu trúc
                                                  ┌────────────────────────┐
                                                  │     FlexSDS Engine     │
                                                  │  (Core / Server / vLLM)│
                                                  └────────────────────────┘
```

---

## 3. Các kịch bản Happy path tiêu chuẩn

### Kịch bản 1 (Kịch bản trọng tâm): Tự động phát hiện và khởi chạy cục bộ trên Laptop/PC có GPU NVIDIA

#### Bước 1.1: Người dùng đưa ra yêu cầu tự nhiên
Người dùng mở cửa sổ chat với AI Agent và yêu cầu:
> *"Tôi muốn dựng một chatbot dùng mô hình Qwen 7B trên máy tính này để thử nghiệm. Bạn kiểm tra xem cấu hình máy tôi có gánh nổi không và tối ưu tốc độ giúp tôi nhé."*

#### Bước 1.2: Tác tử kích hoạt kỹ năng và quét phần cứng ngầm
Tác tử nhận diện ý định triển khai LLM và kích hoạt kỹ năng `flexsds-assistant`. Tác tử thực thi lệnh kiểm tra:
```bash
flexsds detect --json
```
**Dữ liệu JSON thu được:**
```json
{
  "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "sm_architecture": "sm_120",
  "compute_capability": [12, 0],
  "total_vram_mb": 8151.0,
  "free_vram_mb": 7820.0,
  "cuda_version": "12.8",
  "is_cuda_available": true
}
```

#### Bước 1.3: Tác tử chạy Pre-download Guard và tính toán Speedup dự kiến
Người dùng muốn dùng `Qwen/Qwen2.5-7B-Instruct`. Tác tử gọi script kiểm tra an toàn:
```bash
flexsds check-compat --target Qwen/Qwen2.5-7B-Instruct --draft duckhoi/qwen2.5-7b-eagle3.1-speculator --json
```
**Kết quả phân tích từ động cơ FlexSDS:**
- Bản gốc FP16 cần: $14.5\text{ GB (weights)} + 0.6\text{ GB (EAGLE)} + 1.2\text{ GB (KV-cache)} = 16.3\text{ GB} > 7.8\text{ GB free} \implies$ **Nguy cơ OOM 100%**.
- Tác tử tự động chọn phiên bản tối ưu thay thế: `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit):
  $$\text{VRAM cần thiết: } 4.18\text{ GB (weights)} + 0.6\text{ GB (EAGLE)} + 1.2\text{ GB (KV-cache)} = 5.98\text{ GB} \le 7.8\text{ GB} \implies \text{\textbf{An toàn}}.$$
- Tác tử nạp mô hình chi phí của GPU, chạy thuật toán tìm điểm cân bằng phân tích $k^*$:
  $$\text{Tỷ lệ chấp nhận trung bình } \alpha \approx 0.78, \quad k^* = 3$$
  $$\text{Hệ số tăng tốc dự kiến } S = \frac{1 + \alpha + \alpha^2 + \alpha^3}{1 + c_{\text{eagle}}} \approx \mathbf{1.82\times} \text{ so với vLLM nguyên bản (vanilla)}.$$

#### Bước 1.4: Tác tử phản hồi người dùng với bảng khuyến nghị rõ ràng
Tác tử xuất hiện câu trả lời cho người dùng:
> "Tôi đã kiểm tra phần cứng thiết bị của bạn:
> - **GPU phát hiện:** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, 7.8 GB khả dụng).
> - **Cảnh báo VRAM:** Mô hình `Qwen2.5-7B-Instruct` dạng FP16 nguyên bản cần ~16.3 GB VRAM, sẽ gây tràn bộ nhớ (OOM) trên máy của bạn.
> - **Giải pháp tối ưu từ FlexSDS:** Chuyển sang mô hình lượng tử hóa **Qwen2.5-7B-Instruct-AWQ** kết hợp đầu nháp suy đoán **EAGLE-3.1**.
> - **Tốc độ dự kiến:** Nhanh hơn **~1.82 lần** so với chạy thông thường ($k^* = 3$, không suy giảm chất lượng đầu ra).
> - **Tài nguyên chiếm dụng:** ~5.98 GB VRAM (còn dư ~1.8 GB an toàn cho hệ thống).
> 
> Bạn có muốn tôi khởi chạy máy chủ phục vụ ở cổng `8000` ngay bây giờ không?"

#### Bước 1.5: Người dùng phê duyệt và tác tử khởi chạy máy chủ ngầm
Người dùng trả lời: *"Ok, chạy luôn đi bạn."*

Tác tử gọi lệnh khởi chạy nền với quản lý tiến trình độc lập:
```bash
flexsds serve \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --backend local \
  --port 8000 \
  --background
```
Máy chủ khởi chạy, ghi PID vào `.flexsds.pid`.

#### Bước 1.6: Tác tử xác thực sức khỏe endpoint và gửi truy vấn thử nghiệm
Tác tử gửi yêu cầu thăm dò endpoint:
```bash
curl -s http://localhost:8000/health
```
Khi nhận được HTTP 200, tác tử tự động gửi một prompt kiểm thử mẫu:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-adaptive",
    "messages": [{"role": "user", "content": "Viết 1 đoạn thơ ngắn về mùa thu Hà Nội."}],
    "max_tokens": 100
  }'
```

#### Bước 1.7: Báo cáo hoàn tất kèm mã mẫu cho người dùng
Tác tử thông báo:
> "✔ Máy chủ FlexSDS đã sẵn sàng phục vụ tại `http://localhost:8000/v1`!
> - **PID:** 14208 (đang chạy ngầm, không khóa terminal).
> - **Độ trễ phản hồi mẫu:** 380 ms (tốc độ ~52 tokens/giây, tăng tốc 1.8x).
> 
> Bạn có thể kết nối từ Python bằng thư viện OpenAI SDK như sau:
> ```python
> from openai import OpenAI
> client = OpenAI(base_url="http://localhost:8000/v1", api_key="flexsds")
> response = client.chat.completions.create(
>     model="qwen2.5-7b-adaptive",
>     messages=[{"role": "user", "content": "Xin chào!"}]
> )
> print(response.choices[0].message.content)
> ```
> Khi nào muốn dừng máy chủ, bạn chỉ cần bảo: *'Hãy dừng server FlexSDS'*, tôi sẽ dọn dẹp sạch tiến trình."

---

### Kịch bản 2: Máy tính không có GPU rời $\to$ Tự động chuyển hướng Modal Cloud (Serverless BYOC)

#### Bước 2.1: Người dùng yêu cầu
> *"Deploy giúp tôi mô hình Qwen 7B có suy đoán đầu cơ để test thử API."*

#### Bước 2.2: Tác tử phát hiện thiếu phần cứng phù hợp
Tác tử quét phần cứng cục bộ:
- Không phát hiện card đồ họa NVIDIA (hoặc chỉ có đồ họa tích hợp Intel/AMD).
- Tác tử ngay lập tức đề xuất giải pháp đám mây serverless:
> "Máy tính hiện tại không có card đồ họa NVIDIA CUDA rời đủ để chạy mô hình 7B. Tôi đề xuất bạn triển khai trên nền tảng **Modal.com** (dùng GPU NVIDIA L4 serverless, trả phí theo từng giây sử dụng và tặng sẵn $30 miễn phí hàng tháng)."

#### Bước 2.3: Tác tử hỗ trợ người dùng thiết lập khóa định danh (Secret guidance)
Nếu kiểm tra chưa thấy `~/.modal.toml` hoặc biến môi trường `MODAL_TOKEN_ID`:
> "Tôi thấy máy bạn chưa cấu hình khóa Modal. Bạn chỉ cần làm theo 2 bước sau:
> 1. Truy cập https://modal.com/settings/tokens và tạo một token mới.
> 2. Cung cấp cho tôi `MODAL_TOKEN_ID` và `MODAL_TOKEN_SECRET` (hoặc gõ lệnh `modal setup`), tôi sẽ tự động lưu bảo mật."

#### Bước 2.4: Tác tử khởi chạy bản serverless
Sau khi có token, tác tử kích hoạt lệnh:
```bash
flexsds serve --backend modal --target Qwen/Qwen2.5-7B-Instruct-AWQ --deploy
```
- Tác tử nhận được public URL của Modal: `https://duckhoi--ptit-vllm-serve.modal.run/v1`.
- Tác tử kiểm tra tính khả dụng và bàn giao đường dẫn kết nối công khai kèm hướng dẫn bảo mật cho người dùng.

---

### Kịch bản 3: Triển khai miễn phí trên Google Colab qua Cloudflare Tunnel

#### Bước 3.1: Người dùng yêu cầu
> *"Tôi không có thẻ tín dụng để dùng Modal và máy không có GPU, có cách nào chạy thử miễn phí không?"*

#### Bước 3.2: Tác tử gợi ý nền tảng Google Colab
> "Bạn hoàn toàn có thể chạy miễn phí 100% bằng cách sử dụng **Google Colab (GPU Tesla T4)** kết hợp đường hầm Cloudflare Tunnel mà FlexSDS hỗ trợ sẵn."

#### Bước 3.3: Tác tử cung cấp liên kết 1 chạm và nhận đường hầm
1. Tác tử gửi liên kết notebook mở sẵn:
   `https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb`
2. Tác tử hướng dẫn: *"Bạn chỉ cần nhấn 'Run all'. Khi ô cuối cùng chạy xong, Colab sẽ in ra một đường dẫn dạng `https://xxxx.trycloudflare.com`. Hãy dán đường dẫn đó vào đây."*
3. Người dùng gửi đường dẫn: `https://crisp-apples-taste.trycloudflare.com`
4. Tác tử âm thầm gọi hàm xác thực:
   ```bash
   python3 -c "from flexsds.server.colab_runner import ColabServingRunner; print(ColabServingRunner().verify_tunnel_health('https://crisp-apples-taste.trycloudflare.com'))"
   ```
5. Endpoint xác nhận sẵn sàng $\to$ Tác tử cấu hình client và báo người dùng có thể bắt đầu chat ngay lập tức.

---

### Kịch bản 4: Dừng máy chủ an toàn không để lại tiến trình rác (Zero-zombie cleanup)

#### Bước 4.1: Người dùng yêu cầu
> *"Thôi tôi test xong rồi, tắt server đi giúp tôi."*

#### Bước 4.2: Tác tử thực thi lệnh dừng
Tác tử không dùng các lệnh `kill -9` bừa bãi có thể làm treo driver GPU, mà gọi lệnh chuẩn:
```bash
flexsds stop
```
- Đọc PID từ `.flexsds.pid`.
- Gửi `os.killpg(pid, signal.SIGTERM)` dọn sạch tiến trình cha và toàn bộ tiến trình con vLLM engine, giải phóng 100% VRAM GPU.
- Xóa tệp PID.

#### Bước 4.3: Tác tử báo cáo kết quả
> "✔ Đã dừng máy chủ FlexSDS và giải phóng toàn bộ VRAM của GPU thành công. Thiết bị đã trở về trạng thái nghỉ an toàn."

---

## 4. Cài FlexSDS Assistant

Bundle gốc nằm tại `.agents/skills/flexsds-assistant/` trong repo FlexSDS. Có thể chia sẻ thư mục này qua GitHub hoặc phát hành ZIP chứa toàn bộ thư mục.

| Nền tảng | Cách cài |
|---|---|
| Codex | Mở repo có bundle trong `.agents/skills/`. Để dùng chung trên máy, sao chép bundle vào `~/.agents/skills/`. |
| Antigravity App, IDE và CLI | Mở workspace chứa repo có bundle trong `.agents/skills/`. Để dùng chung trên máy, sao chép bundle vào `~/.gemini/config/skills/`. |
| Claude Code | Sao chép bundle vào `.claude/skills/` của project hoặc `~/.claude/skills/` của người dùng. |
| Claude web hoặc Desktop | Nén thư mục `flexsds-assistant/` thành ZIP. Vào **Customize → Skills → + → Create skill → Upload a skill**. |

Khi người dùng muốn thực thi lệnh, môi trường của agent cần có FlexSDS CLI. Cài skill cung cấp workflow cho agent. Việc triển khai GPU hoặc endpoint serving được thực hiện riêng qua CLI.

---

## 5. Tổng kết giá trị

| Yếu tố đánh giá | Thao tác dòng lệnh thủ công (CLI) | Kỹ năng tác tử thông minh (Agent skill) |
| :--- | :--- | :--- |
| **Yêu cầu kiến thức người dùng** | Phải hiểu VRAM, AWQ, EAGLE, $k$, port, PID. | Chỉ cần diễn đạt mong muốn bằng ngôn ngữ tự nhiên. |
| **Rủi ro lỗi tràn bộ nhớ (OOM)** | Dễ gặp nếu người dùng chọn nhầm bản FP16. | Tác tử tự động chặn trước và đề xuất bản AWQ tối ưu. |
| **Ước lượng hiệu năng** | Người dùng phải tự đo và so sánh thủ công. | Tác tử tính toán và dự báo trước hệ số tăng tốc ($S$). |
| **Độ thân thiện (UX)** | Phải nhớ cú pháp và các cờ tham số. | Tác tử tự xử lý I/O, kiểm tra healthcheck và cung cấp code mẫu. |
