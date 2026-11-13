# Web demo hoá đơn — frontend (Next.js trên Vercel)

Console FlexSDS của web demo [M5-01]. Các trang: Node 1 (ảnh → OCR → JSON → DB, có bảng hoá đơn đã lưu), Node 2 (câu hỏi → SQL đã kiểm tra), Node 3 (báo cáo Q1–Q6 và dashboard vẽ từ dòng SQL), Đo và so sánh (A: đo tốc độ nhiều endpoint với W1/W2/W3; B: chuỗi 1 → 2 → 3), Endpoint (dán địa chỉ do `flexsds serve` in ra, chọn mốc). Backend là FastAPI `src/api/` (Render); frontend chỉ biết URL backend và mật khẩu demo người dùng nhập — không có token Modal, khoá Supabase hay khoá API của endpoint.

Code chính: `src/components/Console.tsx` (khung, điều hướng theo `#node1|#node2|#node3|#result|#endpoints`), các trang `*Page.tsx`, `ui.tsx`, `charts.tsx` (SVG tự vẽ, 3 màu endpoint đã kiểm tra bằng bộ validate bảng màu), `src/lib/official.ts` (số chính thức M3-05, M3-06 hiển thị tách khỏi số đo trực tiếp).

| Lệnh | Việc |
|---|---|
| `npm install` | Cài thư viện (Node ≥ 22) |
| `npm run dev` | Chạy ở `http://localhost:3000` |
| `npm test` | Test component (Vitest + Testing Library) |
| `npm run lint` | ESLint |
| `npm run build && npm run check:bundle` | Build và quét bundle không có tên/khoá bí mật |

Biến môi trường (Vercel → Project Settings → Environment Variables): `NEXT_PUBLIC_API_URL` = URL backend Render, ví dụ `https://ptit-invoice-api.onrender.com`. Giá trị được nhúng lúc build, nên đổi URL thì phải build lại.

Trên Vercel: Root Directory = `web`, Framework = Next.js.
