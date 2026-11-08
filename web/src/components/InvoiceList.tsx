"use client";

import { useCallback, useEffect, useState } from "react";

import InvoiceView from "@/components/InvoiceView";
import type { Api, InvoiceDetail, InvoiceRow } from "@/lib/api";
import { formatMoney } from "@/lib/format";

interface Props {
  api: Api;
  refreshKey?: number;
}

export default function InvoiceList({ api, refreshKey = 0 }: Props) {
  const [rows, setRows] = useState<InvoiceRow[] | null>(null);
  const [detail, setDetail] = useState<InvoiceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setRows((await api.invoices()).invoices);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }, [api]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load, refreshKey]);

  async function open(invoiceId: string) {
    try {
      setDetail(await api.invoice(invoiceId));
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white p-4">
        {error && (
          <p role="alert" className="text-sm text-red-600">
            {error}
          </p>
        )}
        <table className="w-full min-w-[560px] text-left text-sm" aria-label="Danh sách hoá đơn">
          <thead className="text-xs text-slate-500">
            <tr>
              <th className="py-1 pr-3 font-medium">Người bán</th>
              <th className="py-1 pr-3 font-medium">Ngày</th>
              <th className="py-1 pr-3 font-medium">Tổng tiền</th>
              <th className="py-1 pr-3 font-medium">Mặt hàng</th>
              <th className="py-1 pr-3 font-medium">Nguồn</th>
            </tr>
          </thead>
          <tbody>
            {(rows ?? []).map((row) => (
              <tr
                key={row.invoice_id}
                onClick={() => void open(row.invoice_id)}
                className="cursor-pointer border-t border-slate-100 hover:bg-slate-50"
              >
                <td className="py-1.5 pr-3">{row.seller_name ?? "—"}</td>
                <td className="py-1.5 pr-3">{row.issued_date ?? "—"}</td>
                <td className="py-1.5 pr-3">{formatMoney(row.total_amount_value, row.total_amount_currency)}</td>
                <td className="py-1.5 pr-3">{row.item_count}</td>
                <td className="py-1.5 pr-3">{row.source === "upload" ? "tải lên" : "devset"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows && rows.length === 0 && <p className="mt-2 text-sm text-slate-500">Chưa có hoá đơn.</p>}
      </div>
      {detail && (
        <section aria-label="Chi tiết hoá đơn" className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-900">
            {detail.invoice_id} <span className="font-normal text-slate-500">(làn {detail.lane ?? "—"})</span>
          </h3>
          {detail.image_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={detail.image_url} alt="Ảnh hoá đơn" className="max-h-80 w-full rounded border object-contain" />
          )}
          <InvoiceView invoice={detail.payload} />
        </section>
      )}
    </div>
  );
}
