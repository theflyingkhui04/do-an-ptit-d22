const integer = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 });

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number.isInteger(value) ? integer.format(value) : decimal.format(value);
}

export function formatMoney(value: number | null | undefined, currency?: string | null): string {
  if (value === null || value === undefined) return "—";
  return `${integer.format(value)} ${currency === "VND" || !currency ? "đ" : currency}`;
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${decimal.format(value / 1000)} s` : `${integer.format(value)} ms`;
}

export function formatSpeedup(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${decimal.format(value)}×`;
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

export const STEP_LABELS: Record<string, string> = {
  queued: "Đang chờ",
  ocr: "OCR ảnh",
  node1: "Node 1 trích xuất",
  saving: "Lưu DB",
  done: "Xong",
  failed: "Lỗi",
  interrupted: "Bị ngắt",
};
