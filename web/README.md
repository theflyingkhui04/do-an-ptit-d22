# Web demo hoá đơn — frontend (Next.js trên Vercel)

Giao diện của web demo [M5-01]: tải ảnh hoá đơn → OCR → Node 1 → DB, so sánh làn speculative decoding, hỏi dữ liệu bằng tiếng Việt. Backend là FastAPI `src/api/` (Render); frontend chỉ biết URL backend và mật khẩu demo người dùng nhập — không có token Modal hay khoá Supabase.

| Lệnh | Việc |
|---|---|
| `npm install` | Cài thư viện (Node ≥ 22) |
| `npm run dev` | Chạy ở `http://localhost:3000` |
| `npm test` | Test component (Vitest + Testing Library) |
| `npm run lint` | ESLint |
| `npm run build && npm run check:bundle` | Build và quét bundle không có tên/khoá bí mật |

Biến môi trường (Vercel → Project Settings → Environment Variables): `NEXT_PUBLIC_API_URL` = URL backend Render, ví dụ `https://ptit-invoice-api.onrender.com`. Giá trị được nhúng lúc build, nên đổi URL thì phải build lại.

Trên Vercel: Root Directory = `web`, Framework = Next.js.
