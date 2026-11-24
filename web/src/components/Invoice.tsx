"use client";

import { Pill } from "@/components/ui";
import type { InvoicePayload } from "@/lib/api";
import { formatMoney, formatNumber } from "@/lib/format";
import { useI18n } from "@/lib/i18n";

function LineCheck({ value }: { value: boolean | null | undefined }) {
  const { t } = useI18n();
  if (value === true) return <Pill tone="good">{t.invLineMatch}</Pill>;
  if (value === false) return <Pill tone="warn">{t.invLineMismatch}</Pill>;
  return <span className="text-ink-3">—</span>;
}

export default function InvoiceDetail({ invoice, itemLines, invoiceSum }: {
  invoice: InvoicePayload;
  itemLines?: (boolean | null)[];
  invoiceSum?: boolean | null;
}) {
  const { lang, t } = useI18n();
  const fields: [keyof InvoicePayload, string, string][] = [
    ["seller", "name", t.invSeller],
    ["seller", "address", t.invAddress],
    ["seller", "phone", t.invPhone],
    ["seller", "tax_code", t.invTaxCode],
    ["buyer", "name", t.invBuyer],
    ["invoice", "invoice_number", t.invNumber],
    ["invoice", "issued_date", t.invDate],
    ["invoice", "issued_time", t.invTime],
    ["invoice", "payment_method", t.invPayment],
  ];
  const totals: [string, string][] = [
    ["subtotal", t.invSubtotal],
    ["discount", t.invDiscount],
    ["vat_amount", t.invVat],
    ["total_amount", t.invTotal],
  ];
  const part = (name: keyof InvoicePayload) => invoice[name] as Record<string, unknown> | undefined;
  return (
    <div className="grid gap-5" aria-label={t.invoiceDataAria}>
      <dl className="grid gap-x-8 gap-y-1.5 text-[13.5px] sm:grid-cols-2">
        {fields.map(([section, key, label]) => (
          <div key={`${section}.${key}`} className="flex min-w-0 gap-3">
            <dt className="w-32 shrink-0 text-ink-3">{label}</dt>
            <dd className="min-w-0 break-words">{(part(section)?.[key] as string | null) ?? "—"}</dd>
          </div>
        ))}
        {totals.map(([key, label]) => (
          <div key={key} className="flex gap-3">
            <dt className="w-32 shrink-0 text-ink-3">{label}</dt>
            <dd className="num font-semibold">{formatMoney(invoice.totals[key]?.value, invoice.totals[key]?.currency, lang)}</dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-[14px] font-semibold">{t.invItemsTitle(invoice.items.length)}</h3>
          {invoiceSum === true && <Pill tone="good">{t.invSumMatch}</Pill>}
          {invoiceSum === false && <Pill tone="warn">{t.invSumMismatch}</Pill>}
        </div>
        {invoice.items.length === 0 ? (
          <p className="text-ink-3">{t.invNoItems}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[600px] text-left text-[13.5px]" aria-label={t.invItemsTableAria}>
              <thead className="text-ink-3">
                <tr className="border-b border-line">
                  <th className="py-1.5 pr-3 font-semibold">#</th>
                  <th className="py-1.5 pr-3 font-semibold">{t.invItemCol}</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">{t.invQtyCol}</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">{t.invUnitPriceCol}</th>
                  <th className="py-1.5 pr-3 text-right font-semibold">{t.invLineTotalCol}</th>
                  {itemLines && <th className="py-1.5 font-semibold">{t.invCheckCol}</th>}
                </tr>
              </thead>
              <tbody className="num">
                {invoice.items.map((item, index) => (
                  <tr key={`${item.line_number}-${index}`} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 text-ink-3">{item.line_number}</td>
                    <td className="py-1.5 pr-3">{item.description ?? "—"}</td>
                    <td className="py-1.5 pr-3 text-right">{formatNumber(item.quantity?.value, lang)}</td>
                    <td className="py-1.5 pr-3 text-right">{formatMoney(item.unit_price?.value, item.unit_price?.currency, lang)}</td>
                    <td className="py-1.5 pr-3 text-right">{formatMoney(item.line_total?.value, item.line_total?.currency, lang)}</td>
                    {itemLines && <td className="py-1.5"><LineCheck value={itemLines[index]} /></td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {invoice.taxes.length > 0 && (
        <div className="grid gap-1">
          <h3 className="text-[14px] font-semibold">{t.invTaxesTitle(invoice.taxes.length)}</h3>
          <ul className="num text-[13.5px]">
            {invoice.taxes.map((tax, index) => (
              <li key={index}>
                {tax.tax_name ?? t.invTaxDefault} {tax.tax_rate !== null ? `${formatNumber(tax.tax_rate, lang)}%` : ""}:{" "}
                {formatMoney(tax.tax_amount?.value, tax.tax_amount?.currency, lang)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
