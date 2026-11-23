import type { Metadata } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import "./globals.css";

const sans = Be_Vietnam_Pro({
  variable: "--font-sans-vi",
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "FlexSDS Benchmark",
  description:
    "Chạy pipeline hoá đơn 3 node và đo tốc độ trên các endpoint do flexsds tạo ra (đồ án tốt nghiệp D22, PTIT)",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="vi" className={`${sans.variable} h-full antialiased`}>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
