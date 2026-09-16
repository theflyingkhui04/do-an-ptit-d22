# QUY TẮC QUẢN TRỊ VÀ TẠO TASK TRÊN TRELLO (TRELLO RULES)
# DỰ ÁN ĐỒ ÁN TỐT NGHIỆP D22 PTIT

> **Tài liệu hướng dẫn chuẩn hóa quy trình tạo, quản lý và nghiệm thu thẻ công việc (Cards/Tasks)**  
> **Áp dụng cho:** Cả thành viên nhóm (Khôi, Đại, Dương) và AI Assistant / Agent (Antigravity, Cursor, LLMs).  
> **Bảng quản trị:** `🎯 Sprint Board · Đồ Án D22` (Trello Board ID: `6aa7c8900212fe787bbf59fa`)

---

## 1. NGUYÊN TẮC BẤT BIẾN (CORE PRINCIPLES)

1. **Không tạo task mồ côi:** Mọi task khi tạo ra bắt buộc phải có **Assignee**, **Start Date**, **Due Date**, **Estimated Time**, **PR Link** và **Checklist**. Tuyệt đối không tạo card rỗng (như `task`, `chưa có mô tả`).
2. **Quy tắc 3 điều kiện Done (Definition of Done - DoD):** Một thẻ chỉ được phép kéo sang cột `✅ Done` khi thỏa mãn đồng thời 3 điều kiện:
   - ✅ **Code chạy được:** Đã merge vào nhánh `main` (hoặc có Pull Request sẵn sàng review).
   - ✅ **Có số liệu thực nghiệm:** Đã lưu log JSONL hoặc kết quả đo vào thư mục `experiments/runs/` hoặc `docs/`.
   - ✅ **Có tài liệu báo cáo:** Đã viết tối thiểu 1–2 đoạn văn phân tích/giải thích vào bản thảo quyển đồ án (`docs/` hoặc file nháp).
3. **Giới hạn việc đang làm (WIP - Work In Progress):** Mỗi thành viên chỉ được để tối đa **1 task** (nhiều nhất 2 task nếu có task đang chờ review) trong cột `⏳ In Progress` tại một thời điểm để tránh phân tán nguồn lực.

---

## 2. BẢNG ĐỐI ỨNG THÀNH VIÊN VÀ PHÂN VAI (MEMBERS & ROLES)

Mọi task phải được gán đúng tài khoản Trello và nhãn vai trò chuyên trách:

| Thành viên | Trello Account | Trello Member ID | Vai trò & Nhãn Trello | Phụ trách chính trong đồ án |
| :--- | :--- | :--- | :--- | :--- |
| **Khôi** | `@canduckhoi` *(Can Duc Khoi)* | `6aa7bdc54e08e522e0de8ae0` | `Assignee: Khôi (Dev + PM)` | **Luồng A (Engine & Thuật toán):** Thuật toán điều khiển cây nháp (`src/proposer/`), tích hợp vLLM, API server (`src/server/`), quản lý dự án, review/merge `main`. Viết Chương 2 & 3. |
| **Đại** | `@buithien7` *(Bùi Thiên)* | `67da7e55b3c12bffa1c90b83` | `Assignee: Đại (Dev + IT)` | **Luồng B (Hạ tầng & Đo kiểm):** Xây dựng harness benchmark tự động, quản trị GPU T4 (`experiments/`), đo concurrency, log JSONL, vẽ biểu đồ F1–F6. Viết Chương 4 & 5. |
| **Dương** | `@dngnguynth` *(Dương Nguyễn Thế)* | `6aa7c7af7d1c825db392f1a0` | `Assignee: Dương (Dev + BA)` | **Luồng C (Ứng dụng & Dữ liệu):** Phân tích nghiệp vụ hóa đơn VAT, thiết kế schema JSON/SQLite, xây dựng pipeline 3 node MC-OCR (`src/pipeline/`), định dạng chuẩn quyển PTIT & slide. Viết Chương 1 & 6. |

---

## 3. QUY ƯỚC ĐẶT TÊN VÀ ĐỊNH DẠNG THẺ (CARD SPECIFICATION)

### 3.1. Quy ước đặt tiêu đề thẻ (Title)
Cú pháp chuẩn:
```text
[M{Sprint/Mốc}-{Số thứ tự}] {Động từ hành động} + {Module / Đối tượng} + {Mục tiêu cụ thể}
```
- **Ví dụ đúng:**
  - `[M0-01] Kiểm chứng vLLM + Qwen2.5-7B AWQ trên GPU T4 & Khởi tạo dự án`
  - `[M0-03] Thiết lập hạ tầng GPU T4 & Xây dựng Harness Benchmark ghi log JSONL`
  - `[M0-04] Phân tích nghiệp vụ hóa đơn VAT, thiết kế Schema & Lập trình xử lý MC-OCR`
- **Ví dụ sai (Cấm):**
  - ❌ `task`
  - ❌ `Làm benchmark` (quá chung chung, không có mã mốc)
  - ❌ `Fix bug` (không rõ bug ở đâu, mục tiêu là gì)

---

### 3.2. Cấu trúc nội dung Mô tả (Description Template)
Mọi card khi tạo mới PHẢI dùng khung Markdown sau:

```markdown
### 📌 Metadata
- **Status:** To-Do (hoặc In Progress)
- **Sprint:** Sprint {X} · {Tên Sprint} ({Ngày bắt đầu} – {Ngày kết thúc})
- **Assignee:** {Tên người phụ trách} ({Vai trò})
- **Estimated Time:** {Số ngày} ngày
- **Start Date:** {dd/mm/yyyy}
- **Due Date:** {dd/mm/yyyy}
- **Git Branch / PR:** {Link GitHub branch/PR}

---
### 🎯 Mục tiêu & Phân chia công việc theo Role
#### 💻 Phần Kỹ thuật (DEV / IT / BA):
- [Mô tả chi tiết mục tiêu kỹ thuật, công nghệ áp dụng]
- [Các rào cản kỹ thuật cần giải quyết]

#### 📋 Phần Quản trị & Sản phẩm (PM / Doc):
- [Yêu cầu về tài liệu, tiêu chuẩn code, tài liệu hóa]

---
### 🏁 Tiêu chí Hoàn thành (Definition of Done)
- [ ] Code/kết quả đã được test không lỗi trên môi trường mục tiêu.
- [ ] Đã có log/kết quả lưu tại `experiments/` hoặc commit Git.
- [ ] Đã cập nhật tài liệu hoặc viết vào bản thảo quyển báo cáo.
```

---

## 4. QUY ĐỊNH VỀ CÁC TRƯỜNG DỮ LIỆU BẮT BUỘC (CARD FIELDS)

Mỗi thẻ khi tạo ra PHẢI được cập nhật đầy đủ 5 trường sau:

1. **Start Date (Ngày bắt đầu):** Định dạng ISO 8601 UTC (vd: `2026-09-14T08:00:00.000Z`).
2. **Due Date (Hạn chót):** Định dạng ISO 8601 UTC (vd: `2026-09-17T23:59:59.000Z`).
3. **Custom Field `Estimated Time (Ngày)`:**
   - Định dạng: Số nguyên (`number`).
   - Giá trị: Bằng khoảng thời gian dự kiến từ `Start Date` đến `Due Date` (vd: `4`).
4. **Custom Field `PR / Git Branch Link`:**
   - Định dạng: Chuỗi liên kết (`text`).
   - Giá trị: Đường dẫn URL đến nhánh tính năng trên GitHub (vd: `https://github.com/theflyingkhui04/do-an-ptit-d22/tree/feat/engine-vllm-poc`).
5. **Nhãn dán (Labels - Tối thiểu 3 nhãn/card):**
   - 🟩 **Sprint:** `Sprint 1 · M0 PoC Nền tảng`, `Sprint 2 · M1`,...
   - ⬛ **Độ ưu tiên (MoSCoW):** `Priority: MUST` (Bắt buộc), `Priority: SHOULD` (Nên có), `Priority: COULD` (Có thể cắt giảm).
   - 🟦/🟧/🟪 **Người phụ trách:** `Assignee: Khôi (Dev + PM)`, `Assignee: Đại (Dev + IT)`, `Assignee: Dương (Dev + BA)`.
   - 🟨/🟩/🟪 **Loại công việc:** `Type: Core-Engine`, `Type: Benchmark & Data`, `Type: App & Pipeline`, `Type: Report & Doc`.
   - 🟥 **Cảnh báo kẹt (nếu có):** `Risk: Blocker (Chặn)`.

---

## 5. QUY ĐỊNH VỀ CHECKLIST

1. **Tên Checklist:** Luôn đặt theo định dạng:
   `Tiêu chí nghiệm thu & Checklist M{x}-{yy} ({Tên người phụ trách})`
2. **Số lượng mục:** Tối thiểu **3** và tối đa **7** mục hành động con (actionable sub-tasks).
3. **Nguyên tắc viết mục:** Phải cụ thể, đo lường được, rõ ràng hành động (bắt đầu bằng động từ: Cài đặt, Tải, Kiểm tra, Xây dựng, Đo đạc, Viết,...).
4. **Quy tắc tích chọn:** Người phụ trách tự tích chọn khi hoàn thành từng phần việc. Card chỉ được chuyển sang cột `Done` khi **100% mục checklist đã được tích xanh**.

---

## 6. QUY TẮC CHUYỂN DỊCH TRẠNG THÁI CỘT (WORKFLOW TRANSITIONS)

```text
[Backlog] ──> [To-Do] ──> [In Progress] ──> [Done]
                               │
                               ▼
                          [Blocked]
```

1. **Từ `To-Do` ──> `In Progress`:**
   - Thành viên tự kéo thẻ khi bắt đầu bắt tay vào làm.
   - Đảm bảo nhánh Git tương ứng đã được tạo theo quy tắc: `feat/{tên-tính-năng}` hoặc `fix/{tên-lỗi}`.
   - Không chuyển quá 1 task/người vào `In Progress`.
2. **Khi gặp sự cố ──> `Blocked`:**
   - Kéo thẻ sang `🛑 Sprint X · Blocked`.
   - Gắn nhãn màu đỏ `Risk: Blocker (Chặn)`.
   - Đăng comment nêu rõ: (1) Đang bị kẹt ở đâu? (2) Cần ai hỗ trợ? (3) Hạn chót cần gỡ trước khi ảnh hưởng toàn dự án.
3. **Từ `In Progress` ──> `Done`:**
   - Rà soát lại **3 điều kiện DoD**: Code đã merge/PR? Đã có số liệu? Đã viết báo cáo?
   - 100% checklist đã được đánh dấu hoàn thành.
   - Đánh dấu hoàn thành Due Date (`dueComplete: true`).

---

## 7. CẤU HÌNH TỰ ĐỘNG HÓA & POWER-UPS (AUTOMATION & POWER-UPS)

### 7.1. Power-Ups khuyến nghị
- **Custom Fields (Đã kích hoạt):** Dùng để hiển thị `Estimated Time (Ngày)` và `PR / Git Branch Link` ngay trên mặt trước thẻ.
- **Card Aging (Bật trên giao diện web):** Vào *Menu > Power-Ups > Tìm "Card Aging" > Add*. Giúp phát hiện các thẻ bị bỏ quên quá 7 ngày.
- **Gantt Chart (Planyway hoặc TeamGantt):** Vào *Menu > Power-Ups > Tìm "Planyway" > Add*. Tự động vẽ timeline tiến độ dựa trên trường `Start` và `Due` date đã điền.

### 7.2. Automations chuẩn (Butler Rules)
Thiết lập tại biểu tượng `⚡ Automation` trên thanh công cụ Trello:
1. **Rule Hoàn thành (Done Rule):**
   - *Trigger:* `when a card is moved into list "✅ Sprint X · Done"`
   - *Action:* `mark the due date as complete` và `post comment "🎉 Task đã hoàn thành nghiệm thu theo chuẩn Definition of Done!"`
2. **Rule Báo kẹt (Blocked Rule):**
   - *Trigger:* `when a card is moved into list "🛑 Sprint X · Blocked"`
   - *Action:* `add the red "Risk: Blocker (Chặn)" label to the card` và `post comment "@canduckhoi @buithien7 @dngnguynth Task đang bị chặn, cần họp tháo gỡ!"`
3. **Rule Nhắc Deadline (Due Date Rule):**
   - *Trigger:* `the day before a card is due`
   - *Action:* `post comment "⚠️ Cảnh báo: Task sắp đến hạn trong vòng 24 giờ nữa! Vui lòng cập nhật checklist."`
4. **Calendar Rule Họp tuần định kỳ:**
   - *Trigger:* `every week on Sunday at 8:30 pm` (20h30 tối Chủ Nhật)
   - *Action:* `create a new card with title "📅 Họp tổng kết tuần & Estimate task tuần sau" in list "📌 Sprint X · To-Do" at the top` và gắn checklist nội dung họp.

---

## 8. HƯỚNG DẪN DÀNH CHO AI AGENT KHI THAO TÁC QUA TRELLO MCP

Khi người dùng yêu cầu AI tạo hoặc cập nhật task trên Trello, AI Agent PHẢI tuân thủ nghiêm ngặt quy trình:
1. **Không bao giờ tạo card thiếu thông tin:** Luôn truyền đủ `name`, `desc` theo template mục 3.2.
2. **Luôn gọi API thiết lập đầy đủ:**
   - Gọi `update_card` thiết lập `start` và `due`.
   - Gán Custom Fields `Estimated Time` và `PR / Git Branch Link`.
   - Gọi `trello_create_checklist` và `trello_create_check_item` để tạo toàn bộ các mục kiểm tra nghiệm thu.
   - Gán đầy đủ các Labels tương ứng với Sprint, Role, Priority.
3. **Báo cáo kết quả rõ ràng:** Sau khi thao tác, luôn phản hồi bảng tổng hợp gồm Tên task, Người phụ trách, Start/Due Date, Estimated Days, Link PR và Checklist để người dùng kiểm chứng.
