"use client";

import type { InvoicePayload } from "@/lib/api";
import { formatMoney, formatNumber } from "@/lib/format";

const FIELD_LABELS: [string, string, string][] = [
  ["seller", "name", "Người bán"],
  ["seller", "address", "Địa chỉ"],
  ["seller", "phone", "Điện thoại"],
  ["seller", "tax_code", "MST người bán"],
  ["buyer", "name", "Người mua"],
  ["invoice", "invoice_number", "Số hoá đơn"],
  ["invoice", "issued_date", "Ngày"],
  ["invoice", "issued_time", "Giờ"],
  ["invoice", "payment_method", "Thanh toán"],
];
const TOTAL_LABELS: [string, string][] = [
  ["subtotal", "Cộng tiền hàng"],
  ["discount", "Giảm giá"],
  ["vat_amount", "Thuế VAT"],
  ["total_amount", "Tổng tiền"],
];

interface Props {
  invoice: InvoicePayload;
  itemLines?: (boolean | null)[];
}

export default function InvoiceView({ invoice, itemLines }: Props) {
  const section = (name: string) => invoice[name as keyof InvoicePayload] as Record<string, unknown> | undefined;
  return (
    <div className="space-y-4" aria-label="Dữ liệu hoá đơn">
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        {FIELD_LABELS.map(([part, key, label]) => (
          <div key={`${part}.${key}`} className="flex gap-2">
            <dt className="w-28 shrink-0 text-slate-500">{label}</dt>
            <dd className="text-slate-900">{(section(part)?.[key] as string | null) ?? "—"}</dd>
          </div>
        ))}
        {TOTAL_LABELS.map(([key, label]) => (
          <div key={key} className="flex gap-2">
            <dt className="w-28 shrink-0 text-slate-500">{label}</dt>
            <dd className="font-medium text-slate-900">
              {formatMoney(invoice.totals[key]?.value, invoice.totals[key]?.currency)}
            </dd>
          </div>
        ))}
      </dl>

      <div>
        <h4 className="text-sm font-semibold text-slate-900">Mặt hàng ({invoice.items.length})</h4>
        {invoice.items.length === 0 ? (
          <p className="text-sm text-slate-500">Không có dòng mặt hàng.</p>
        ) : (
          <div className="mt-1 overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-sm" aria-label="Bảng mặt hàng">
              <thead className="text-xs text-slate-500">
                <tr>
                  <th className="py-1 pr-2 font-medium">#</th>
                  <th className="py-1 pr-2 font-medium">Mặt hàng</th>
                  <th className="py-1 pr-2 font-medium">SL</th>
                  <th className="py-1 pr-2 font-medium">Đơn giá</th>
                  <th className="py-1 pr-2 font-medium">Thành tiền</th>
                  <th className="py-1 pr-2 font-medium">SL×ĐG</th>
                </tr>
              </thead>
              <tbody>
                {invoice.items.map((item, index) => (
                  <tr key={item.line_number} className="border-t border-slate-100">
                    <td className="py-1 pr-2">{item.line_number}</td>
                    <td className="py-1 pr-2">{item.description ?? "—"}</td>
                    <td className="py-1 pr-2">{formatNumber(item.quantity?.value)}</td>
                    <td className="py-1 pr-2">{formatMoney(item.unit_price?.value, item.unit_price?.currency)}</td>
                    <td className="py-1 pr-2">{formatMoney(item.line_total?.value, item.line_total?.currency)}</td>
                    <td className="py-1 pr-2">
                      {itemLines?.[index] === true ? "khớp" : itemLines?.[index] === false ? "lệch" : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {invoice.taxes.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-slate-900">Thuế ({invoice.taxes.length})</h4>
          <ul className="text-sm">
            {invoice.taxes.map((tax, index) => (
              <li key={index}>
                {tax.tax_name ?? "Thuế"} {tax.tax_rate !== null ? `${formatNumber(tax.tax_rate)}%` : ""}:{" "}
                {formatMoney(tax.tax_amount?.value, tax.tax_amount?.currency)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
