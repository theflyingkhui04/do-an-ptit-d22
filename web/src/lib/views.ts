// The four read-only views Node 2 may query (src/api/questions.py view_columns: three mirrored from the
// SQLite v2 schema plus the demo-only upload view) and the six fixed questions Q1–Q6
// (src/pipeline/node2_questions.py), which Node 3 turns into the report.

export const VIEWS: { name: string; about: string; columns: string[] }[] = [
  {
    name: "v_invoice_facts",
    about: "Một dòng mỗi hoá đơn",
    columns: [
      "invoice_id", "issued_date", "issued_time", "seller_name", "seller_address", "seller_tax_code", "buyer_name",
      "payment_method", "subtotal_value", "discount_value", "vat_amount_value", "total_amount_value", "currency",
      "item_count", "tax_line_count",
    ],
  },
  {
    name: "v_item_facts",
    about: "Một dòng mỗi mặt hàng",
    columns: [
      "invoice_id", "issued_date", "seller_name", "line_number", "description", "quantity_value", "quantity_unit",
      "unit_price_value", "line_total_value", "vat_rate",
    ],
  },
  {
    name: "v_tax_facts",
    about: "Một dòng mỗi dòng thuế",
    columns: ["invoice_id", "issued_date", "seller_name", "tax_name", "tax_rate", "taxable_amount_value", "tax_amount_value"],
  },
  { name: "v_upload_facts", about: "Hoá đơn tải lên từ web", columns: ["invoice_id", "source", "uploaded_at", "uploaded_date"] },
];

export const QUESTIONS: { id: string; question: string }[] = [
  { id: "Q1", question: "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?" },
  { id: "Q2", question: "Mặt hàng nào được mua nhiều nhất?" },
  { id: "Q3", question: "Cửa hàng nào có tổng giá trị hóa đơn cao nhất?" },
  { id: "Q4", question: "Có bao nhiêu hóa đơn có phát sinh VAT?" },
  { id: "Q5", question: "Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu?" },
  { id: "Q6", question: "Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu?" },
];
