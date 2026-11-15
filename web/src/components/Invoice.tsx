"use client";

import { Pill } from "@/components/ui";
import type { InvoicePayload } from "@/lib/api";
import { formatMoney, formatNumber } from "@/lib/format";

const FIELDS: [keyof InvoicePayload, string, string][] = [
  ["seller", "name", "Người bán"],
  ["seller", "address", "Địa chỉ"],
  ["seller", "phone", "Điện thoại"],
  ["seller", "tax_code", "MST người bán"],
  ["buyer", "name", "Người mua"],
  ["invoice", "invoice_number", "Số hoá đơn"],
  ["invoice", "issued_date", "Ngày lập"],
  ["invoice", "issued_time", "Giờ"],
  ["invoice", "payment_method", "Thanh toán"],
];
const TOTALS: [string, string][] = [
  ["subtotal", "Cộng tiền hàng"],
  ["discount", "Giảm giá"],
  ["vat_amount", "Thuế VAT"],
  ["total_amount", "Tổng tiền"],
];

function LineCheck({ value }: { value: boolean | null | undefined }) {
  if (value === true) return <Pill tone="good">khớp</Pill>;
  if (value === false) return <Pill tone="warn">lệch dòng</Pill>;
  return <span className="text-ink-3">—</span>;
}

export default function InvoiceDetail({ invoice, itemLines, invoiceSum }: {
  invoice: InvoicePayload;
  itemLines?: (boolean | null)[];
  invoiceSum?: boolean | null;
}) {
  const part = (name: keyof InvoicePayload) => invoice[name] as Record<string, unknown> | undefined;
  return (
    <div className="grid gap-5" aria-label="Dữ liệu hoá đơn">
      <dl className="grid gap-x-8 gap-y-1.5 text-[13.5px] sm:grid-cols-2">
        {FIELDS.map(([section, key, label]) => (
          <div key={`${section}.${key}`} className="flex min-w-0 gap-3">
            <dt className="w-28 shrink-0 text-ink-3">{label}</dt>
            <dd className="min-w-0 break-words">{(part(section)?.[key] as string | null) ?? "—"}</dd>
          </div>
        ))}
        {TOTALS.map(([key, label]) => (
          <div key={key} className="flex gap-3">
            <dt className="w-28 shrink-0 text-ink-3">{label}</dt>
            <dd className="num font-semibold">{formatMoney(invoice.totals[key]?.value, invoice.totals[key]?.currency)}</dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-[14px] font-semibold">Mặt hàng ({invoice.items.length})</h3>
          {invoiceSum === true && <Pill tone="good">Tổng các dòng khớp tổng hoá đơn</Pill>}
          {invoiceSum === false && <Pill tone="warn">Tổng các dòng lệch tổng hoá đơn</Pill>}
        </div>
        {invoice.items.length === 0 ? (
          <p className="text-ink-3">Không có dòng mặt hàng.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[600px] text-left text-[13.5px]" aria-label="Bảng mặt hàng">
              <thead className="text-ink-3">
                <tr className="border-b border-line">
                  <th className="py-1.5 pr-3 font-semibold">#</th>
                  <th className="py-1.5 pr-3 font-semibold">Mặt hàng</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">SL</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">Đơn giá</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">Thành tiền</th>
                  {itemLines && <th className="py-1.5 font-semibold">SL × đơn giá</th>}
                </tr>
              </thead>
              <tbody className="num">
                {invoice.items.map((item, index) => (
                  <tr key={`${item.line_number}-${index}`} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 text-ink-3">{item.line_number}</td>
                    <td className="py-1.5 pr-3">{item.description ?? "—"}</td>
                    <td className="py-1.5 pr-3 text-right">{formatNumber(item.quantity?.value)}</td>
                    <td className="py-1.5 pr-3 text-right">{formatMoney(item.unit_price?.value, item.unit_price?.currency)}</td>
                    <td className="py-1.5 pr-3 text-right">{formatMoney(item.line_total?.value, item.line_total?.currency)}</td>
                    {itemLines && <td className="py-1.5"><LineCheck value={itemLines[index]} /></td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {itemLines && <p className="text-[12.5px] text-ink-3">Chỉ đánh dấu chỗ lệch, không tự sửa số model trích ra.</p>}
      </div>

      {invoice.taxes.length > 0 && (
        <div className="grid gap-1">
          <h3 className="text-[14px] font-semibold">Thuế ({invoice.taxes.length})</h3>
          <ul className="num text-[13.5px]">
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
