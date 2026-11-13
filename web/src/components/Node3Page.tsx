"use client";

import { Database, FileText, LayoutDashboard, Play, ShieldCheck, Sparkles } from "lucide-react";
import { useState } from "react";

import { BarChart } from "@/components/charts";
import type { PageId } from "@/components/Console";
import {
  Button,
  EmptyState,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Panel,
  Pill,
  ResultTable,
  Stat,
  Trace,
  cx,
  defaultTarget,
} from "@/components/ui";
import { ApiError, type Api, type ChartSpec, type DataTable, type Endpoint, type ReportResult } from "@/lib/api";
import { formatMoney, formatMs, formatNumber } from "@/lib/format";

const EXAMPLES = [
  "Báo cáo chi tiêu và dashboard của các hoá đơn gần nhất",
  "Cửa hàng nào chi nhiều nhất? So sánh các cửa hàng",
  "Tình hình thuế VAT trong các hoá đơn",
];
const SCOPES: [number | null, string][] = [[5, "5 gần nhất"], [10, "10 gần nhất"], [25, "25 gần nhất"], [null, "Tất cả"]];
const LABELS: Record<string, string> = {
  so_hoa_don: "Số hoá đơn",
  tong_tien: "Tổng tiền",
  trung_binh: "Trung bình mỗi hoá đơn",
  hoa_don_co_vat: "Hoá đơn có VAT",
};
const MONEY = /tien|thue|amount|total/i;
const QUANTITY = /so_luong|quantity|qty/i;
const OCR_SUSPECT_QUANTITY = 1000; // a single shop receipt line above this is almost always an OCR misread

function quantityFlag(table: DataTable) {
  const index = table.columns.findIndex((c) => QUANTITY.test(c));
  if (index < 0) return undefined;
  return (row: unknown[]) => (typeof row[index] === "number" && (row[index] as number) > OCR_SUSPECT_QUANTITY ? "nghi lỗi OCR" : null);
}

function Chart({ spec, table }: { spec: ChartSpec; table: DataTable }) {
  if (spec.type === "kpi") {
    const row = table.rows[0] ?? [];
    return (
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label={spec.title}>
        {table.columns.map((c, i) =>
          typeof row[i] === "number" ? (
            <Stat key={c} label={LABELS[c] ?? c} value={MONEY.test(c) ? formatMoney(row[i] as number) : formatNumber(row[i] as number)} />
          ) : null,
        )}
      </div>
    );
  }
  if (spec.type === "table") return <ResultTable columns={table.columns} rows={table.rows} max={12} flag={quantityFlag(table)} />;
  const value = table.columns.indexOf(spec.value ?? "");
  // Label with one text column whose values are all different; otherwise join the text columns (date, store).
  const texts = table.columns.map((_, i) => i).filter((i) => i !== value && table.rows.some((r) => typeof r[i] === "string"));
  const unique = texts.find((i) => new Set(table.rows.map((r) => r[i])).size === table.rows.length);
  const labelOf = (r: unknown[]) =>
    unique !== undefined ? String(r[unique] ?? "—") : texts.map((i) => String(r[i] ?? "—")).join(", ");
  const rows = table.rows
    .filter((r) => typeof r[value] === "number")
    .map((r) => [labelOf(r), r[value] as number] as [string, number]);
  const sorted = spec.type === "bar" ? [...rows].sort((a, b) => b[1] - a[1]).slice(0, 8) : rows;
  return <BarChart rows={sorted} horizontal={spec.type === "bar"} unit={MONEY.test(spec.value ?? "") ? "đ" : ""} label={spec.title} />;
}

function Dashboard({ data }: { data: ReportResult }) {
  const tables = Object.fromEntries(data.tables.map((t) => [t.key, t]));
  const charts = data.dashboard.charts.filter((c) => tables[c.table]);
  return (
    <>
      {data.dashboard.status === "fallback" && (
        <Notice tone="warn">Model không chọn được biểu đồ ({data.dashboard.error}). Đang hiện bộ biểu đồ mặc định.</Notice>
      )}
      <div className="grid items-start gap-5 lg:grid-cols-2">
        {charts.map((c) => (
          <Panel
            key={`${c.table}-${c.type}-${c.value}`}
            title={c.title}
            meta={<span className="font-mono">{c.table}</span>}
            className={cx(c.type === "kpi" && "lg:col-span-2")}
          >
            <Chart spec={c} table={tables[c.table]} />
          </Panel>
        ))}
      </div>
    </>
  );
}

export default function Node3Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const [target, setTarget] = useState<string[]>(() => defaultTarget(endpoints));
  const [request, setRequest] = useState(EXAMPLES[0]);
  const [limit, setLimit] = useState<number | null>(10);
  const [data, setData] = useState<ReportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (endpoints.length === 0) return <><PageHead title="Báo cáo và dashboard" /><NeedEndpoint what="Node 3" onGo={() => go("endpoints")} /></>;

  async function run() {
    if (target.length === 0 || !request.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setData(await api.report(target[0], request.trim(), limit));
    } catch (exc) {
      setError(exc instanceof ApiError && exc.status === 429 ? "Đã chạy quá nhiều trong ít phút, thử lại sau." : (exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHead title="Báo cáo và dashboard">
        Node 3 nhận yêu cầu của bạn cùng số liệu lấy sẵn từ cơ sở dữ liệu bằng truy vấn cố định, không qua Node 2. Model chọn biểu đồ và viết báo cáo;
        mọi con số trên dashboard lấy từ các bảng số liệu.
      </PageHead>

      <Panel title="Yêu cầu báo cáo" icon={Sparkles}>
        <form
          className="grid gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            run();
          }}
        >
          <div className="grid gap-2">
            <label htmlFor="report-request" className="text-[13.5px] font-semibold">Yêu cầu</label>
            <textarea
              id="report-request"
              rows={2}
              maxLength={500}
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              className="rounded-lg border border-line bg-surface px-3 py-2"
            />
            <div className="flex flex-wrap gap-2" aria-label="Yêu cầu mẫu">
              {EXAMPLES.map((text) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => setRequest(text)}
                  className={cx("rounded-full border px-3 py-1 text-[13px] hover:bg-sunken", request === text ? "border-ink font-semibold" : "border-line")}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-x-8 gap-y-4">
            <fieldset className="grid gap-2">
              <legend className="mb-2 text-[13.5px] font-semibold">Phạm vi (theo ngày lập)</legend>
              <div className="flex flex-wrap gap-2">
                {SCOPES.map(([value, text]) => (
                  <label key={text} className={cx("flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-[13.5px]", limit === value ? "border-ink" : "border-line")}>
                    <input type="radio" name="scope" checked={limit === value} onChange={() => setLimit(value)} />
                    {text}
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="grid content-start gap-2">
              <span className="text-[13.5px] font-semibold">Chạy trên</span>
              <EndpointChips single endpoints={endpoints} selected={target} onToggle={(id) => setTarget([id])} />
            </div>
          </div>
          <div>
            <Button type="submit" icon={Play} disabled={busy || target.length === 0 || !request.trim()}>
              {busy ? "Đang chọn biểu đồ và viết báo cáo…" : "Tạo báo cáo"}
            </Button>
          </div>
        </form>
        {error && <Notice tone="bad">{error}</Notice>}
        {data && data.scope.invoice_count > 0 && (
          <Trace
            steps={[
              { label: "Lấy số liệu", title: "Lấy số liệu (truy vấn cố định)", ms: data.timings_ms.data, kind: "io" },
              { label: "Chọn biểu đồ", title: "Node 3 chọn biểu đồ", ms: data.timings_ms.dashboard, kind: "n3" },
              { label: "Viết báo cáo", title: "Node 3 viết báo cáo", ms: data.timings_ms.report, kind: "n3" },
            ]}
            extra={<span>trên <b className="text-ink">{data.endpoint.name}</b>, {data.scope.label} ({data.scope.invoice_count} hoá đơn)</span>}
          />
        )}
      </Panel>

      {!data ? (
        <EmptyState icon={LayoutDashboard} title="Chưa có báo cáo">
          <p>Gõ yêu cầu, chọn phạm vi rồi bấm “Tạo báo cáo”. Dashboard chỉ hiện số có trong cơ sở dữ liệu của lần chạy này.</p>
        </EmptyState>
      ) : data.scope.invoice_count === 0 ? (
        <Notice tone="info">Phạm vi này chưa có hoá đơn nào. Trích xuất hoá đơn ở trang Node 1 trước.</Notice>
      ) : (
        <>
          <Dashboard data={data} />
          <Panel
            title="Báo cáo của Node 3"
            icon={FileText}
            meta={data.report.metrics?.tpot_ms ? <span>TPOT {formatMs(data.report.metrics.tpot_ms)}</span> : undefined}
          >
            {data.report.status === "ok" && data.report.text ? (
              <div className="max-w-[75ch] whitespace-pre-wrap leading-relaxed">{data.report.text}</div>
            ) : (
              <Notice tone="bad">Node 3 không viết được báo cáo: {data.report.error ?? data.report.status}</Notice>
            )}
            {data.report.status === "ok" && (
              <div className="flex flex-wrap gap-2">
                {data.number_check.ungrounded.length === 0 ? (
                  <Pill tone="good" icon={ShieldCheck}>{data.number_check.numbers} số trong báo cáo đều có trong bảng số liệu</Pill>
                ) : (
                  <Pill tone="warn">Số không có trong bảng số liệu: {data.number_check.ungrounded.join(", ")}</Pill>
                )}
              </div>
            )}
          </Panel>
          <Panel title="Số liệu đầu vào của Node 3" icon={Database} meta={<span>{data.tables.length} bảng, truy vấn cố định</span>}>
            {data.tables.map((t) => (
              <details key={t.key} className="rounded-lg border border-line px-3 py-2">
                <summary className="cursor-pointer">
                  <b className="font-semibold">{t.title}</b>
                  <span className="ml-2 font-mono text-[12.5px] text-ink-3">{t.key}, {t.rows.length} dòng</span>
                </summary>
                <div className="mt-2"><ResultTable columns={t.columns} rows={t.rows} max={20} /></div>
              </details>
            ))}
          </Panel>
        </>
      )}
    </>
  );
}
