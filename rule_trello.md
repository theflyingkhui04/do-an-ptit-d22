# QUY TẮC QUẢN TRỊ VÀ TẠO TASK TRÊN TRELLO (TRELLO RULES)
# DỰ ÁN ĐỒ ÁN TỐT NGHIỆP D22 PTIT

> **Tài liệu hướng dẫn chuẩn hóa quy trình tạo, quản lý và nghiệm thu thẻ công việc (Cards/Tasks)**  
> **Áp dụng cho:** Cả thành viên nhóm (Khôi, Đại, Dương) và AI Assistant / Agent (Antigravity, Cursor, LLMs).  
> **Bảng quản trị:** `🎯 Sprint Board · Đồ Án D22` (Board ID: `6aa7c8900212fe787bbf59fa`)  
> **Đường dẫn Board:** [🎯 Sprint Board · Đồ Án D22](https://trello.com/b/M2yyHLqg/%F0%9F%8E%AF-sprint-board-%C4%91%E1%BB%93-%C3%A1n-d22)

---

## 1. NGUYÊN TẮC BẤT BIẾN (CORE PRINCIPLES)

1. **Không tạo task mồ côi:** Mọi task khi tạo ra bắt buộc phải có:
   - **Assignee ghép cặp (`idMembers`):** Gán cả 2 thành viên (Lead 70% và Partner 30%).
   - **Start Date & Due Date:** Thời gian bắt đầu và hạn chót rõ ràng.
   - **Custom Fields:** `Estimated Time (Ngày)` và `PR / Git Branch Link`.
   - **Checklist tiêu chí nghiệm thu:** Gồm 5–6 subtasks, mỗi subtask có tiền tố `[{Tên người làm}]`, được gán `idMember` và deadline `due` tịnh tiến logic.  
   *Tuyệt đối không tạo card rỗng (như `task`, `chưa có mô tả`).*
2. **Quy tắc 3 điều kiện Done (Definition of Done - DoD):** Một thẻ chỉ được phép kéo sang cột `✅ Done` khi thỏa mãn đồng thời 3 điều kiện:
   - ✅ **Code chạy được:** Đã merge vào nhánh `main` (hoặc có Pull Request sẵn sàng review).
   - ✅ **Có số liệu thực nghiệm:** Đã lưu log JSONL hoặc kết quả đo vào thư mục `experiments/runs/` hoặc `docs/`.
   - ✅ **Có tài liệu báo cáo:** Đã viết tối thiểu 1–2 đoạn văn phân tích/giải thích vào bản thảo quyển đồ án (`docs/` hoặc file nháp chương).
3. **Giới hạn việc đang làm (WIP - Work In Progress):** Mỗi thành viên chỉ được để tối đa **1 task** (nhiều nhất 2 task nếu có task đang chờ review) trong cột `⏳ In Progress` tại một thời điểm để tránh phân tán nguồn lực.

---

## 2. BẢNG ĐỐI ỨNG THÀNH VIÊN VÀ CƠ CHẾ PHÂN VAI (MEMBERS & PAIR-ROTATION 70/30)

### 2.1. Bảng đối ứng tài khoản thành viên
Mỗi thành viên giữ vai trò Chủ trì (Lead) mảng cốt lõi của mình (70% effort), chịu trách nhiệm cao nhất về kiến trúc, kiểm soát chất lượng và review tích hợp vào nhánh `main`:

| Thành viên | Trello Account | Trello Member ID | Label Assignee trên Board | Phụ trách chính trong đồ án (70% effort) |
| :--- | :--- | :--- | :--- | :--- |
| **Khôi** | `@canduckhoi` *(Can Duc Khoi)* | `6aa7bdc54e08e522e0de8ae0` | `Assignee: Khôi (Dev + PM)` | **Luồng A (Core-Engine & Thuật toán):** Thuật toán điều khiển cây nháp (`src/proposer/`), tích hợp vLLM, API server (`src/server/`), quản lý dự án, review/merge `main`. Viết Chương 2 & 3. |
| **Đại** | `@buithien7` *(Bùi Thiên)* | `67da7e55b3c12bffa1c90b83` | `Assignee: Đại (Dev + IT)` | **Luồng B (Hạ tầng & Đo kiểm):** Xây dựng harness benchmark tự động, quản trị GPU T4 (`experiments/`), đo sweep concurrency, log JSONL, vẽ biểu đồ F1–F6. Viết Chương 4 & 5. |
| **Dương** | `@dngnguynth` *(Dương Nguyễn Thế)* | `6aa7c7af7d1c825db392f1a0` | `Assignee: Dương (Dev + BA)` | **Luồng C (Ứng dụng & Dữ liệu):** Phân tích nghiệp vụ hóa đơn VAT, thiết kế schema JSON/SQLite, xây dựng pipeline 3 node MC-OCR (`src/pipeline/`), định dạng chuẩn quyển PTIT & slide. Viết Chương 1 & 6. |

### 2.2. Cơ chế Ghép cặp Luân phiên 70/30 (Pair-Rotation System)
Đồ án không áp dụng cách chia biệt lập "nước sông không phạm nước giếng". Để **cả 3 thành viên đều nắm sâu toàn bộ dự án** và tự tin bảo vệ trước Hội đồng, nhóm áp dụng cơ chế:
1. **Mỗi task tại một thời điểm có đúng 2 người cùng làm:**
   - **1 Người Chủ trì (Lead - 70% effort):** Phụ trách kiến trúc chính, gánh ~70% checklist subtasks (thường là 4/6 mục).
   - **1 Người Đồng hành (Partner - 30% effort):** Tham gia trực tiếp code, viết test, hỗ trợ và gánh ~30% checklist subtasks (thường là 2/6 mục), đồng thời là reviewer chính cho Pull Request của Lead.
2. **Cân bằng tải tuyệt đối:**
   Mỗi tuần, tổng tải của mỗi thành viên luôn đạt chuẩn:  
   `70% (Lead task mảng mình) + 30% (Partner task mảng bạn) = 100% effort/tuần`.
3. **Ma trận Luân phiên theo tuần (Tuần A / Tuần B):**

| Luồng công việc | Tuần A (Tuần 1: 22/09 – 28/09) | Tuần B (Tuần 2: 29/09 – 04/10) |
|---|---|---|
| **AI Systems / Engine** | **Khôi (Lead 70%)** + Dương (Partner 30%) | **Khôi (Lead 70%)** + Đại (Partner 30%) |
| **Benchmark / Harness** | **Đại (Lead 70%)** + Khôi (Partner 30%) | **Đại (Lead 70%)** + Dương (Partner 30%) |
| **Data / Pipeline VAT** | **Dương (Lead 70%)** + Đại (Partner 30%) | **Dương (Lead 70%)** + Khôi (Partner 30%) |

*(Chu kỳ luân phiên tuần A/B đảo chéo đều đặn trong suốt các Sprint từ M1 đến M4. Đối với card checkpoint/nghiệm thu cổng cuối sprint, cả 3 thành viên cùng tham gia với tỉ lệ 40% - 30% - 30%).*

---

## 3. QUY ƯỚC ĐẶT TÊN VÀ ĐỊNH DẠNG THẺ (CARD SPECIFICATION)

### 3.1. Quy ước đặt tiêu đề thẻ (Title)
Cú pháp chuẩn:
```text
[M{Sprint/Mốc}-{Số thứ tự}] {Động từ hành động} + {Module / Đối tượng} + {Mục tiêu cụ thể}
```
- **Ví dụ đúng:**
  - `[M1-01] Đóng gói Server vLLM 2 làn (Vanilla & Static Draft) & Chuẩn hóa Endpoint Telemetry`
  - `[M1-02] Tự động hóa Benchmark Harness & Quét Concurrency 1→32 trên 2 làn Cơ sở (M1 Baseline)`
  - `[M1-03] Xây dựng Node 1 (VAT JSON Extractor) Pipeline MC-OCR & Chuẩn bị Dataset Workload W1`
  - `[M1-04] Đánh giá Cổng Kiểm Soát M1 (04/10): Nghiệm thu Bảng Số liệu 2 làn & Đồng bộ Báo cáo`
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
- **Assignee:** {Tên Lead} (Lead 70%) & {Tên Partner} (Partner 30%)
- **Estimated Time:** {Số ngày} ngày
- **Start Date:** {dd/mm/yyyy}
- **Due Date:** {dd/mm/yyyy}
- **Git Branch / PR:** {Link GitHub branch/PR}

---
### 🎯 Mục tiêu & Phân chia công việc theo Role
#### 💻 Phần Kỹ thuật (DEV / IT / BA):
- [Mô tả chi tiết mục tiêu kỹ thuật, công nghệ áp dụng, thông số mô hình/hệ thống]
- [Phân định ranh giới công việc kỹ thuật giữa Lead và Partner]

#### 📋 Phần Quản trị & Sản phẩm (PM / Doc):
- [Yêu cầu về log thực nghiệm JSONL, kết quả đo, lưu trữ dữ liệu]
- [Yêu cầu cập nhật tài liệu quyển đồ án docs/ và nhật ký kỹ thuật logs/]

---
### 🏁 Tiêu chí Hoàn thành (Definition of Done)
- [ ] Code/kết quả đã được test không lỗi trên môi trường mục tiêu (GPU T4).
- [ ] Đã có log/kết quả lưu tại `experiments/` hoặc commit Git.
- [ ] Đã cập nhật tài liệu hoặc viết vào bản thảo quyển báo cáo.
```

---

## 4. QUY ĐỊNH VỀ CÁC TRƯỜNG DỮ LIỆU BẮT BUỘC & BẢNG TRA CỨU ID (CARD FIELDS & BOARD REGISTRY)

Mỗi thẻ khi tạo ra PHẢI được cập nhật đầy đủ các trường sau:

1. **Assignee (`idMembers`):** Bắt buộc gán **cả 2 thành viên** của cặp (Lead và Partner) vào thẻ. (Với task nghiệm thu cổng mốc, gán cả 3 thành viên).
2. **Start Date (Ngày bắt đầu):** Định dạng ISO 8601 UTC (vd: `2026-09-22T00:00:00.000Z`) hoặc `YYYY-MM-DD`.
3. **Due Date (Hạn chót):** Định dạng ISO 8601 UTC (vd: `2026-09-27T23:59:59.000Z`).
4. **Custom Field `Estimated Time (Ngày)`:** Định dạng số nguyên (`number`).
5. **Custom Field `PR / Git Branch Link`:** Chuỗi URL trỏ đến branch hoặc PR trên GitHub (`text`).
6. **Nhãn dán (Labels - Tối thiểu 3–4 nhãn/card):** Gồm Nhãn Sprint, Nhãn Priority, Nhãn Assignee Lead, Nhãn Type.

### 4.1. Bảng Tra Cứu Custom Fields ID (Board Level)
| Custom Field Name | Custom Field ID | Kiểu dữ liệu (`type`) | Hiển thị mặt thẻ |
| :--- | :--- | :--- | :--- |
| **Estimated Time (Ngày)** | `6aaa133b020468454d4a87d2` | `number` | ✅ Có (cardFront) |
| **PR / Git Branch Link** | `6aaa1342ed62dfa352f692e4` | `text` | ✅ Có (cardFront) |

### 4.2. Bảng Tra Cứu Labels ID Chuẩn trên Board
| Tên Label | Label ID | Màu sắc | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| `Priority: MUST` | `6aa7c8d2457b570796e67062` | `black` | Bắt buộc phải hoàn thành trong Sprint |
| `Sprint 1 · M0 PoC Nền tảng` | `6aa7c8c79e52ece851ddf1d0` | `green` | Đánh dấu các task thuộc Sprint 1 |
| `Sprint 2 · M1 Baseline Benchmark` | `6ab00994a7877097b55a7a1f` | `green` | Đánh dấu các task thuộc Sprint 2 |
| `Assignee: Khôi (Dev + PM)` | `6aa7c8c273d654ca5bc8a797` | `blue` | Lead phụ trách chính: Khôi |
| `Assignee: Đại (Dev + IT)` | `6aa7c8c23c51f6283bde926e` | `orange` | Lead phụ trách chính: Đại |
| `Assignee: Dương (Dev + BA)` | `6aa7c8c70b9e3f974cf759c8` | `purple` | Lead phụ trách chính: Dương |
| `Type: Core-Engine` | `6aa7c8cb5354d3c37f0bb363` | `sky` | Luồng thuật toán & suy luận AI Engine |
| `Type: Benchmark & Data` | `6aa7c8ce59ae2e189fb5dbff` | `yellow` | Luồng đo kiểm, harness & số liệu |
| `Type: App & Pipeline` | `6aa7c8ced4dc68a49e01719b` | `lime` | Luồng pipeline OCR hóa đơn VAT |
| `Type: Report & Doc` | `6aa7c8d1ef84c70adf0f23fd` | `pink` | Báo cáo, quyển đồ án, slide nghiệm thu |
| `Risk: Blocker (Chặn)` | `6aa7c8ca36c3ec2b409cff90` | `red` | Cảnh báo task bị nghẽn, cần gỡ khẩn cấp |

### 4.3. Bảng Tra Cứu Danh Sách Cột (Lists ID)
| Tên List | List ID | Vai trò trong quy trình |
| :--- | :--- | :--- |
| `📊 Dashboard & Sprint Roadmap` | `6aa7c897befbe5760158e5a6` | Bảng điều khiển tiến độ mốc M0 → M5 (Cố định, không kéo thẻ đi) |
| `📌 Sprint 2 · To-Do` | `6aa7c899d2c7fe7028892ed5` | Danh sách công việc của Sprint hiện tại chờ thực hiện |
| `⏳ Sprint 2 · In Progress` | `6aa7c89b50c0507980f5efde` | Công việc đang làm trong tuần (giới hạn WIP <= 1 task/người) |
| `🛑 Sprint 2 · Blocked` | `6aa7c89e3a7848994ce24613` | Công việc bị chặn bởi sự cố môi trường/VRAM/phụ thuộc |
| `✅ Sprint 2 · Done` | `6aa7c8a3212bfe415052f687` | Công việc đã hoàn thành nghiệm thu theo chuẩn DoD |
| `📦 Backlog (Sprint 2 → 6)` | `6aa7c8a6d5d72bd1b83f0353` | Kho chứa danh sách công việc các Sprint tiếp theo |

---

## 5. QUY ĐỊNH VỀ CHECKLIST & SUBTASK NÂNG CAO (ADVANCED SUBTASKS)

1. **Tên Checklist:** Đặt theo định dạng phản ánh đúng mã task và cặp làm việc:
   `Tiêu chí nghiệm thu & Checklist M{x}-{yy} ({Tên Lead} 70% & {Tên Partner} 30%)`  
   *(hoặc ngắn gọn: `Tiêu chí nghiệm thu & Checklist M{x}-{yy} ({Tên Lead})`).*
2. **Quy chuẩn Tiền tố bắt buộc ở Tên Subtask:**  
   Mọi mục subtask trong checklist **BẮT BUỘC** phải có tiền tố `[{Tên người làm}]` ở ngay đầu dòng để nhận diện ngay lập tức ai làm phần nào:
   - `[Khôi] Viết script khởi chạy vLLM server Làn 1...`
   - `[Đại] Cấu hình benchmark harness sweep concurrency...`
   - `[Dương] Chuẩn hóa schema JSON VAT và tạo bộ prompt W1...`
3. **Phân bổ số lượng mục (Tỉ lệ 70/30):**  
   Mỗi card chuẩn hóa từ **5–6 subtasks**:
   - **Lead (Chủ trì 70%):** Nhận ~4 mục cốt lõi liên quan đến kiến trúc, code chính và đóng gói.
   - **Partner (Đồng hành 30%):** Nhận ~2 mục phối hợp (chuẩn hóa dữ liệu, viết test/smoke test, đối soát kết quả hoặc viết docs liên quan).
4. **Quy chuẩn Deadline (`due`) cho từng Subtask con:**  
   - Bắt buộc phải thiết lập hạn chót (`due`) định dạng ISO 8601 UTC cho **từng subtask con**.
   - Hạn chót của các subtask con phải **tịnh tiến logic theo ngày làm việc trong tuần** (ví dụ: ngày 23/09 -> 24/09 -> 25/09 -> 26/09 -> 27/09), không được để trống `due` và không dồn tất cả hạn chót vào ngày cuối cùng của card cha.
5. **Gán Thành viên (`idMember`) cho từng Subtask:**  
   Sử dụng tính năng Advanced Checklist của Trello để gán trực tiếp ID tài khoản thành viên (`idMember`) tương ứng với tiền tố `[Tên]`.
6. **Quy tắc tích chọn và nghiệm thu:**  
   - Từng cá nhân hoàn thành subtask nào thì tự tích xanh mục đó.
   - Card chỉ được phép chuyển sang cột `✅ Done` khi **100% mục subtask đã được tích xanh** và cả Lead lẫn Partner cùng nghiệm thu.

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
   - Rà soát lại **3 điều kiện DoD**: Code đã merge/PR? Đã có số liệu log JSONL? Đã viết báo cáo?
   - 100% checklist đã được đánh dấu hoàn thành.
   - Đánh dấu hoàn thành Due Date (`dueComplete: true`).

---

## 7. CẤU HÌNH TỰ ĐỘNG HÓA & POWER-UPS (AUTOMATION & POWER-UPS)

### 7.1. Power-Ups khuyến nghị
- **Custom Fields (Đã kích hoạt):** Dùng để hiển thị `Estimated Time (Ngày)` và `PR / Git Branch Link` ngay trên mặt trước thẻ.
- **Card Aging (Bật trên giao diện web):** Vào *Menu > Power-Ups > Tìm "Card Aging" > Add*. Giúp phát hiện các thẻ bị bỏ quên quá 7 ngày.
- **Gantt Chart (Planyway hoặc TeamGantt):** Tự động vẽ timeline tiến độ dựa trên trường `Start` và `Due` date đã điền.

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

Khi người dùng yêu cầu AI Agent tạo hoặc cập nhật task trên Trello, AI Agent **BẮT BUỘC** tuân thủ đúng danh sách công cụ và thứ tự thực thi chuẩn của `@delorenj/mcp-server-trello`:

### 8.1. Quy trình tạo mới một Task hoàn chỉnh (7 bước chuẩn)
1. **Bước 1 - Tạo Card:**  
   Gọi `add_card_to_list`:
   - `listId`: ID cột To-Do (vd: `6aa7c899d2c7fe7028892ed5`).
   - `name`: Tiêu đề theo chuẩn mục 3.1.
   - `description`: Khung nội dung theo template mục 3.2 (bao gồm cả Lead 70% & Partner 30%).
2. **Bước 2 - Gán cặp thành viên vào Card (`idMembers`):**  
   Gọi `assign_member_to_card`:
   - Gọi lần 1: `cardId` và `memberId` của Lead.
   - Gọi lần 2: `cardId` và `memberId` của Partner.
3. **Bước 3 - Cập nhật Timeline & Labels:**  
   Gọi `update_card_details`:
   - `cardId`: ID thẻ vừa tạo.
   - `start`: Định dạng `YYYY-MM-DD` (vd: `"2026-09-22"`).
   - `dueDate`: Định dạng ISO 8601 (vd: `"2026-09-27T23:59:59.000Z"`).
   - `labels`: Mảng chuỗi Label ID (vd: `["6aa7c8d2457b570796e67062", "6aa7c8c273d654ca5bc8a797", ...]`).
4. **Bước 4 - Gán Custom Field `Estimated Time`:**  
   Gọi `update_card_custom_field`:
   - `cardId`: ID thẻ.
   - `customFieldId`: `"6aaa133b020468454d4a87d2"`.
   - `type`: `"number"`.
   - `value`: Chuỗi số nguyên ngày (vd: `"5"`).
5. **Bước 5 - Gán Custom Field `PR / Git Branch Link`:**  
   Gọi `update_card_custom_field`:
   - `cardId`: ID thẻ.
   - `customFieldId`: `"6aaa1342ed62dfa352f692e4"`.
   - `type`: `"text"`.
   - `value`: URL link GitHub branch (vd: `"https://github.com/theflyingkhui04/do-an-ptit-d22/tree/feat/..."`).
6. **Bước 6 - Tạo Checklist tiêu chí nghiệm thu:**  
   Gọi `create_checklist`:
   - `cardId`: ID thẻ.
   - `name`: Tên checklist theo chuẩn mục 5.1.
7. **Bước 7 - Thêm và cấu hình từng Subtask con:**  
   - Gọi `add_checklist_item`:
     - `cardId`: ID thẻ.
     - `checkListName`: Tên checklist đã tạo ở Bước 6.
     - `text`: Nội dung subtask có tiền tố `[{Tên người làm}]`.
   - Lấy `checkItemId` trả về, gọi tiếp `update_checklist_item`:
     - `cardId`: ID thẻ.
     - `checkItemId`: ID của check-item con vừa tạo.
     - `idMember`: Member ID của người làm tương ứng với tiền tố.
     - `due`: Hạn chót định dạng ISO 8601 UTC tịnh tiến theo ngày làm việc.

### 8.2. Báo cáo kết quả sau khi thao tác
Sau khi hoàn tất thao tác trên Trello, AI Agent luôn xuất bảng tổng hợp phản hồi gồm:
- Mã & Tên Task
- Cặp phụ trách (Lead 70% + Partner 30%)
- Timeline (Start Date → Due Date, Số ngày ước lượng)
- Nhánh Git / PR
- Danh sách 5–6 subtasks kèm phân vai và deadline chi tiết từng mục.

