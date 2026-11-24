// The four read-only SQL views Node 2 may query and the six sample questions Q1-Q6.
import type { Lang } from "@/lib/i18n";

export const VIEWS: { name: string; columns: string[] }[] = [
  {
    name: "v_invoice_facts",
    columns: [
      "invoice_id", "issued_date", "issued_time", "seller_name", "seller_address", "seller_tax_code", "buyer_name",
      "payment_method", "subtotal_value", "discount_value", "vat_amount_value", "total_amount_value", "currency",
      "item_count", "tax_line_count",
    ],
  },
  {
    name: "v_item_facts",
    columns: [
      "invoice_id", "issued_date", "seller_name", "line_number", "description", "quantity_value", "quantity_unit",
      "unit_price_value", "line_total_value", "vat_rate",
    ],
  },
  {
    name: "v_tax_facts",
    columns: ["invoice_id", "issued_date", "seller_name", "tax_name", "tax_rate", "taxable_amount_value", "tax_amount_value"],
  },
  {
    name: "v_upload_facts",
    columns: ["invoice_id", "source", "uploaded_at", "uploaded_date"],
  },
];

export interface SampleQuestion {
  id: string;
  short: string;
  question: string;
}

const QUESTIONS_BY_LANG: Record<Lang, SampleQuestion[]> = {
  vi: [
    { id: "Q1", short: "Tổng chi tiêu tháng 08/2020", question: "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?" },
    { id: "Q2", short: "Mặt hàng mua nhiều nhất", question: "Mặt hàng nào được mua nhiều nhất?" },
    { id: "Q3", short: "Cửa hàng giá trị cao nhất", question: "Cửa hàng nào có tổng giá trị hóa đơn cao nhất?" },
    { id: "Q4", short: "Số hóa đơn có VAT", question: "Có bao nhiêu hóa đơn có phát sinh VAT?" },
    { id: "Q5", short: "Tổng tiền theo ngày", question: "Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu?" },
    { id: "Q6", short: "Thuế theo từng loại", question: "Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu?" },
  ],
  en: [
    { id: "Q1", short: "Total spend in 08/2020", question: "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?" },
    { id: "Q2", short: "Most purchased item", question: "Mặt hàng nào được mua nhiều nhất?" },
    { id: "Q3", short: "Highest-value store", question: "Cửa hàng nào có tổng giá trị hóa đơn cao nhất?" },
    { id: "Q4", short: "Invoices with VAT", question: "Có bao nhiêu hóa đơn có phát sinh VAT?" },
    { id: "Q5", short: "Daily spending total", question: "Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu?" },
    { id: "Q6", short: "Tax by category", question: "Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu?" },
  ],
};

export function getQuestions(lang: Lang = "vi"): SampleQuestion[] {
  return QUESTIONS_BY_LANG[lang];
}

export const QUESTIONS = QUESTIONS_BY_LANG.vi;
