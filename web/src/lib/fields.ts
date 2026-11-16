// Field-level view of a Node 1 extraction for the comparison table (design D13).
// leafFields() walks the payload exactly like leaf_fields() in src/api/compare.py, so the number of fields
// and the field paths match what the backend compares ("items[0].sku", "totals.total_amount.value").
import type { InvoicePayload } from "@/lib/api";

const SECTIONS = ["invoice", "seller", "buyer", "totals", "items", "taxes"] as const;

export type FieldValue = string | number | boolean | null | undefined | unknown[];

export function leafFields(payload: InvoicePayload | null | undefined): [string, FieldValue][] {
  const out: [string, FieldValue][] = [];
  if (!payload) return out;
  const walk = (value: unknown, path: string) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const key of Object.keys(value).sort()) walk((value as Record<string, unknown>)[key], `${path}.${key}`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) out.push([path, []]);
      value.forEach((item, index) => walk(item, `${path}[${index}]`));
    } else {
      out.push([path, value as FieldValue]);
    }
  };
  const record = payload as unknown as Record<string, unknown>;
  for (const section of SECTIONS) walk(record[section], section);
  return out;
}

const GROUPS: Record<string, string> = {
  invoice: "Hoá đơn", seller: "Người bán", buyer: "Người mua", totals: "Tổng tiền", taxes: "Thuế theo mức",
};
const NAMES: Record<string, string> = {
  currency: "Đơn vị tiền", invoice_number: "Số hoá đơn", invoice_series: "Ký hiệu", issued_at: "Thời điểm lập",
  issued_date: "Ngày lập", issued_time: "Giờ lập", payment_method: "Thanh toán", address: "Địa chỉ", name: "Tên",
  phone: "Điện thoại", tax_code: "Mã số thuế", discount: "Chiết khấu", subtotal: "Cộng tiền hàng",
  total_amount: "Tổng thanh toán", vat_amount: "Tiền thuế", description: "Mô tả", line_number: "STT",
  line_total: "Thành tiền", quantity: "Số lượng", sku: "Mã hàng", unit_price: "Đơn giá", vat_rate: "Thuế suất",
  rate: "Thuế suất", amount: "Tiền thuế", taxable_amount: "Tiền chịu thuế",
};
const PARTS: Record<string, string> = { raw: "chữ gốc", value: "giá trị", currency: "đơn vị tiền", unit: "đơn vị tính" };

export function describeField(path: string): { group: string; label: string } {
  const indexed = /^(items|taxes)\[(\d+)\]\.?(.*)$/.exec(path);
  const group = indexed
    ? `${indexed[1] === "items" ? "Dòng hàng" : "Thuế"} ${Number(indexed[2]) + 1}`
    : (GROUPS[path.split(".")[0]] ?? path.split(".")[0]);
  const rest = indexed ? indexed[3] : path.split(".").slice(1).join(".");
  if (!rest) return { group, label: indexed ? "Giá trị" : "Danh sách" };
  const [head, tail] = rest.split(".");
  const name = NAMES[head] ?? head;
  return { group, label: tail ? `${name}, ${PARTS[tail] ?? tail}` : name };
}

// The main fields, shown first; "Xem đủ" shows every field grouped (Dương, 16/11).
export const KEY_FIELDS: Record<string, string> = {
  "seller.name": "Người bán",
  "seller.address": "Địa chỉ người bán",
  "invoice.invoice_number": "Số hoá đơn",
  "invoice.issued_date": "Ngày lập",
  "invoice.issued_time": "Giờ lập",
  "totals.total_amount.value": "Tổng thanh toán",
  "items[0].description": "Dòng hàng 1, mô tả",
  "items[0].line_total.value": "Dòng hàng 1, thành tiền",
};

export function isEmptyValue(value: FieldValue): boolean {
  return value === null || value === undefined || (Array.isArray(value) && value.length === 0);
}

export function fieldText(value: FieldValue): string {
  if (isEmptyValue(value)) return "Không có";
  if (typeof value === "number") return value.toLocaleString("vi-VN");
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

export function sameValue(a: FieldValue, b: FieldValue): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}
