// Field-level view of a Node 1 extraction for the comparison table.
// leafFields() walks the payload identically to leaf_fields() in src/api/compare.py.
import type { InvoicePayload } from "@/lib/api";
import { localeTag } from "@/lib/format";
import type { Lang } from "@/lib/i18n";

const SECTIONS = ["invoice", "seller", "buyer", "totals", "items", "taxes"] as const;

export type FieldValue = string | number | boolean | null | undefined | unknown[];

export function leafFields(payload: InvoicePayload | null | undefined): [string, FieldValue][] {
  const out: [string, FieldValue][] = [];
  if (!payload) return out;
  const walk = (value: unknown, path: string) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const keys = Object.keys(value).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      for (const key of keys) walk((value as Record<string, unknown>)[key], `${path}.${key}`);
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

const GROUPS: Record<Lang, Record<string, string>> = {
  vi: { invoice: "Hóa đơn", seller: "Đơn vị bán", buyer: "Đơn vị mua", totals: "Tổng tiền", taxes: "Thuế" },
  en: { invoice: "Invoice", seller: "Seller", buyer: "Buyer", totals: "Totals", taxes: "Taxes" },
};

const NAMES: Record<Lang, Record<string, string>> = {
  vi: {
    currency: "Đơn vị tiền", invoice_number: "Số hóa đơn", invoice_series: "Ký hiệu", issued_at: "Thời điểm lập",
    issued_date: "Ngày lập", issued_time: "Giờ lập", payment_method: "Thanh toán", address: "Địa chỉ", name: "Tên",
    phone: "Điện thoại", tax_code: "Mã số thuế", discount: "Chiết khấu", subtotal: "Cộng tiền hàng",
    total_amount: "Tổng thanh toán", vat_amount: "Tiền thuế", description: "Mô tả", line_number: "STT",
    line_total: "Thành tiền", quantity: "Số lượng", sku: "Mã hàng", unit_price: "Đơn giá", vat_rate: "Thuế suất",
    rate: "Thuế suất", amount: "Tiền thuế", taxable_amount: "Tiền chịu thuế",
  },
  en: {
    currency: "Currency", invoice_number: "Invoice number", invoice_series: "Series", issued_at: "Issued at",
    issued_date: "Issue date", issued_time: "Issue time", payment_method: "Payment method", address: "Address", name: "Name",
    phone: "Phone", tax_code: "Tax ID", discount: "Discount", subtotal: "Subtotal",
    total_amount: "Total amount", vat_amount: "VAT amount", description: "Description", line_number: "Line #",
    line_total: "Line total", quantity: "Quantity", sku: "SKU", unit_price: "Unit price", vat_rate: "VAT rate",
    rate: "Tax rate", amount: "Tax amount", taxable_amount: "Taxable amount",
  },
};

const PARTS: Record<Lang, Record<string, string>> = {
  vi: { raw: "nguyên bản", value: "giá trị", currency: "đơn vị tiền", unit: "đơn vị tính" },
  en: { raw: "raw", value: "value", currency: "currency", unit: "unit" },
};

export function describeField(path: string, lang: Lang = "vi"): { group: string; label: string } {
  const indexed = /^(items|taxes)\[(\d+)\]\.?(.*)$/.exec(path);
  const group = indexed
    ? `${indexed[1] === "items" ? (lang === "en" ? "Item" : "Mặt hàng") : (lang === "en" ? "Tax" : "Thuế")} ${Number(indexed[2]) + 1}`
    : (GROUPS[lang][path.split(".")[0]] ?? path.split(".")[0]);
  const rest = indexed ? indexed[3] : path.split(".").slice(1).join(".");
  if (!rest) return { group, label: indexed ? (lang === "en" ? "Value" : "Giá trị") : (lang === "en" ? "List" : "Danh sách") };
  const [head, tail] = rest.split(".");
  const name = NAMES[lang][head] ?? head;
  return { group, label: tail ? `${name} (${PARTS[lang][tail] ?? tail})` : name };
}

export const KEY_FIELDS_BY_LANG: Record<Lang, Record<string, string>> = {
  vi: {
    "seller.name": "Đơn vị bán",
    "seller.address": "Địa chỉ",
    "invoice.invoice_number": "Số hóa đơn",
    "invoice.issued_date": "Ngày lập",
    "invoice.issued_time": "Giờ lập",
    "totals.total_amount.value": "Tổng thanh toán",
    "items[0].description": "Mặt hàng 1 · mô tả",
    "items[0].line_total.value": "Mặt hàng 1 · thành tiền",
  },
  en: {
    "seller.name": "Seller",
    "seller.address": "Seller address",
    "invoice.invoice_number": "Invoice number",
    "invoice.issued_date": "Issue date",
    "invoice.issued_time": "Issue time",
    "totals.total_amount.value": "Total amount",
    "items[0].description": "Item 1 · description",
    "items[0].line_total.value": "Item 1 · amount",
  },
};

export const KEY_FIELDS = KEY_FIELDS_BY_LANG.vi;

export function isEmptyValue(value: FieldValue): boolean {
  return value === null || value === undefined || (Array.isArray(value) && value.length === 0);
}

export function fieldText(value: FieldValue, lang: Lang = "vi"): string {
  if (isEmptyValue(value)) return lang === "en" ? "Empty" : "Trống";
  if (typeof value === "number") return value.toLocaleString(localeTag(lang));
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

export function sameValue(a: FieldValue, b: FieldValue): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}
