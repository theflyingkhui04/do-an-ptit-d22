# Tăng tốc suy luận LLM tự-host cho tự động hoá quy trình

> **Accelerating Self-Hosted LLM Inference for Workflow Automation: Hardware-Aware Speculative Decoding on Consumer GPUs**
>
> Đồ án tốt nghiệp — Học viện Công nghệ Bưu chính Viễn thông (PTIT), khoá D22.

Tăng tốc **độ trễ sinh (decode latency)** của mô hình ngôn ngữ lớn (LLM) mã nguồn mở **tự host trên GPU phổ thông**, bằng **giải mã suy đoán thích ứng phần cứng** (hardware-aware speculative decoding) trên nền EAGLE-2 — phục vụ nhu cầu tự động hoá quy trình (Dify / LangGraph) của người dùng hạn chế tài nguyên, **giữ nguyên đầu ra (lossless)**.

## Vì sao đề tài này?

- Nhu cầu tự động hoá tác vụ lặp lại bằng LLM (qua Dify/LangGraph) tăng nhanh, nhưng **API trả phí đắt** và nhiều tình huống yêu cầu **riêng tư / offline** → buộc **tự host mô hình mã nguồn mở trên phần cứng khiêm tốn**.
- "Model đóng đã mạnh, sao còn tối ưu?" → Chính các hệ phục vụ LLM lớn cũng chạy nhờ các kỹ thuật này; và khi tự host, *chi phí mỗi token* chuyển hoá thành **GPU-giây + VRAM + độ trễ** — thứ ta phải tối ưu.
- **Độ trễ sinh** là nút thắt của trải nghiệm tương tác và của **workflow agent nhiều lượt gọi** (độ trễ cộng dồn) — đây là trục đề tài tấn công.

Chi tiết đầy đủ: xem [docs/01-gioi-thieu.md](docs/01-gioi-thieu.md).

## Mục tiêu

1. Tái lập **EAGLE-2** làm baseline.
2. Xây thành phần **nháp thích ứng phần cứng** (điều chỉnh chiến lược cây nháp theo đặc tính GPU đích).
3. Đóng gói thành **endpoint tương thích OpenAI** — cắm thẳng vào Dify / LangGraph.
4. **Đánh giá end-to-end** trên nhiều workflow tương phản (Langfuse + instrument riêng) và dựng **bản đồ use-case**: khi nào tối ưu tốc độ thực sự đáng bật.

## Cấu trúc thư mục

```
do-an-ptit-d22/
├── README.md              # tài liệu này
├── .gitignore
├── requirements.txt       # phụ thuộc (bản khởi tạo)
├── docs/
│   ├── 01-gioi-thieu.md           # Giới thiệu · Sản phẩm · Đóng góp
│   ├── 02-tong-quan-tai-lieu.md   # Khảo sát 3 hướng + lý do chọn
│   ├── 03-quyet-dinh-thiet-ke.md  # Nhật ký quyết định (vì sao đi hướng này)
│   └── references.md              # Danh mục paper (kèm arXiv ID)
├── src/                   # mã nguồn engine + endpoint (giai đoạn thiết kế)
└── experiments/           # workflow đánh giá + pipeline đo + phân tích
```

## Cài đặt

> Giai đoạn thiết kế — môi trường sẽ được chốt (và pin phiên bản) khi bắt đầu hiện thực.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Nhóm thực hiện

| Vai trò | Họ tên |
|---|---|
| Thành viên | Cấn Đức Khôi |
| Thành viên | *(điền)* |
| Thành viên | *(điền)* |
| Giảng viên hướng dẫn | *(điền)* |

## Trạng thái

🟡 **Giai đoạn thiết kế / đề cương.** Hướng kỹ thuật đã khoá; đang hoàn thiện tài liệu trước khi hiện thực.
