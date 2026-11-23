"use client";

import { Database, Play } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";

import { BarChart } from "@/components/charts";
import type { PageId } from "@/components/Console";
import {
  Button,
  Dot,
  Drawer,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Pill,
  ResultTable,
  Spinner,
  cx,
  defaultTarget,
  endpointColors,
  secondsSince,
  useNow,
} from "@/components/ui";
import { ApiError, type Api, type ChartSpec, type DataTable, type Endpoint, type ReportResult } from "@/lib/api";
import { formatMoney, formatMs, formatNumber, formatSpeedup } from "@/lib/format";

const SAMPLES = [
  { short: "Chi tiêu gần đây", text: "Báo cáo chi tiêu và dashboard của các hoá đơn gần nhất" },
  { short: "So sánh cửa hàng", text: "Cửa hàng nào chi nhiều nhất? So sánh các cửa hàng" },
  { short: "Tình hình VAT", text: "Tình hình thuế VAT trong các hoá đơn" },
];
const SCOPES: [number | null, string, string][] = [
  [5, "5", "5 hoá đơn gần nhất"], [10, "10", "10 hoá đơn gần nhất"], [25, "25", "25 hoá đơn gần nhất"], [null, "Tất cả", "Tất cả hoá đơn"],
];
const KPI_LABELS: Record<string, string> = {
  so_hoa_don: "Số hoá đơn",
  tong_tien: "Tổng tiền",
  trung_binh: "Trung bình mỗi hoá đơn",
  hoa_don_co_vat: "Hoá đơn có VAT",
};
const MONEY = /tien|thue|amount|total|trung_binh/i;
const QUANTITY = /so_luong|quantity|qty/i;
const OCR_SUSPECT_QUANTITY = 1000; // a single shop receipt line above this is almost always an OCR misread
const TYPE_NAMES: Record<ChartSpec["type"], string> = { kpi: "số tổng", bar: "biểu đồ thanh", column: "biểu đồ cột", table: "bảng" };
const MAX_ENDPOINTS = 6; // as on Node 1 and Node 2

interface Answer {
  started: number;
  result?: ReportResult;
  error?: string;
}

interface Round {
  request: string;
  order: string[]; // baseline first
  answers: Record<string, Answer>;
}

function runError(exc: unknown): string {
  if (exc instanceof ApiError && exc.status === 429) return "Đã chạy quá nhiều trong ít phút, thử lại sau.";
  return (exc as Error).message;
}

const totalMs = (r: ReportResult) => r.timings_ms.data + r.timings_ms.dashboard + r.timings_ms.report;
const drawn = (r: ReportResult) => r.dashboard.charts.filter((c) => c.type !== "kpi" && r.tables.some((t) => t.key === c.table));

function quantityFlag(table: DataTable) {
  const index = table.columns.findIndex((c) => QUANTITY.test(c));
  if (index < 0) return undefined;
  return (row: unknown[]) => (typeof row[index] === "number" && (row[index] as number) > OCR_SUSPECT_QUANTITY ? "nghi lỗi OCR" : null);
}

function Chart({ spec, table }: { spec: ChartSpec; table: DataTable }) {
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

// The fixed-query totals, shown once above the reports: they do not depend on the model.
function Totals({ table }: { table: DataTable }) {
  const row = table.rows[0] ?? [];
  return (
    <dl aria-label="Số tổng của phạm vi" className="grid overflow-hidden rounded-xl border border-line bg-surface [grid-template-columns:repeat(auto-fit,minmax(min(50%,150px),1fr))]">
      {table.columns.map((c, i) => (typeof row[i] === "number" ? (
        <div key={c} className="-ml-px -mt-px grid min-w-0 gap-0.5 border-l border-t border-line px-4 py-3.5">
          <dt className="text-[12.5px] text-ink-3">{KPI_LABELS[c] ?? c}</dt>
          <dd className="num m-0 text-[22px] font-bold tracking-tight">{MONEY.test(c) ? formatMoney(row[i] as number) : formatNumber(row[i] as number)}</dd>
        </div>
      ) : null))}
    </dl>
  );
}

// The report as written, with the numbers that are not in the data tables marked where they appear.
function ReportText({ text, ungrounded }: { text: string; ungrounded: string[] }) {
  const pattern = ungrounded.length ? new RegExp(`(${ungrounded.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`) : null;
  return (
    <div className="grid max-w-[68ch] gap-2.5 text-[15px] leading-relaxed">
      {text.split(/\n+/).filter((p) => p.trim()).map((p, i) => (
        <p key={i}>
          {pattern ? p.split(pattern).map((part, j) => (ungrounded.includes(part)
            ? <mark key={j} title="Không có trong bảng số liệu" className="rounded bg-warn-tint px-0.5 font-semibold text-warn">{part}</mark>
            : part)) : p}
        </p>
      ))}
    </div>
  );
}

function Stats({ result }: { result: ReportResult }) {
  const m = result.report.metrics;
  const cells: [string, string][] = [
    ["Thời gian", formatMs(totalMs(result))],
    ["TTFT", formatMs(m?.ttft_ms)],
    ["TPOT", formatMs(m?.tpot_ms)],
    ["Token", formatNumber(m?.completion_tokens)],
  ];
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-lg border border-line sm:grid-cols-4">
      {cells.map(([term, value], i) => (
        <div key={term} className={cx("grid min-w-0 gap-px border-line px-2.5 py-1.5", i > 0 && "sm:border-l", i % 2 === 1 && "border-l", i >= 2 && "border-t sm:border-t-0")}>
          <dt className="truncate text-[11.5px] text-ink-3">{term}</dt>
          <dd className="num m-0 whitespace-nowrap text-[14px] font-semibold">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Column({ endpoint, color, isBaseline, answer, base, viewing, onView, now }: {
  endpoint: Endpoint;
  color: string;
  isBaseline: boolean;
  answer: Answer;
  base?: ReportResult;
  viewing: boolean;
  onView: () => void;
  now: number;
}) {
  const r = answer.result;
  const pill = answer.error ? <Pill tone="bad">Lỗi</Pill>
    : !r ? <Pill tone="idle"><Spinner className="size-3.5" />Đang viết, {secondsSince(answer.started, now)} s</Pill>
    : r.report.status === "ok" ? <Pill tone="good">Đã viết</Pill> : <Pill tone="bad">Không viết được</Pill>;
  let body: ReactNode;
  if (answer.error) body = <p role="alert" className="text-bad">{answer.error}</p>;
  else if (!r) body = <p className="text-ink-3">Đang chọn biểu đồ rồi viết báo cáo…</p>;
  else {
    const ungrounded = r.number_check.ungrounded;
    const foreign = r.language_check?.foreign_words ?? [];
    const baseTables = base && base !== r ? new Set(drawn(base).map((c) => c.table)) : null;
    body = (
      <>
        {r.report.status === "ok" && r.report.text
          ? <ReportText text={r.report.text} ungrounded={ungrounded} />
          : <p role="alert" className="text-bad">Node 3 không viết được báo cáo: {r.report.error ?? r.report.status}</p>}
        <Stats result={r} />
        <div className="flex flex-wrap gap-1.5">
          {r.report.status === "ok" && (ungrounded.length === 0
            ? <Pill tone="good">{r.number_check.numbers}/{r.number_check.numbers} số khớp bảng số liệu</Pill>
            : <Pill tone="warn">{ungrounded.length} số không có trong bảng: {ungrounded.join(", ")}</Pill>)}
          {foreign.length > 0 && <Pill tone="warn">Chữ không phải tiếng Việt: {foreign.join(", ")}</Pill>}
          {r.dashboard.status === "fallback" && (
            <span title={r.dashboard.error ?? undefined} className="inline-flex items-center rounded-full bg-warn-tint px-2.5 py-0.5 text-[12.5px] font-semibold text-warn">
              Không chọn được biểu đồ, dùng bộ mặc định
            </span>
          )}
        </div>
        {drawn(r).length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-[12.5px] text-ink-3">
            <span>Biểu đồ đã chọn:</span>
            {drawn(r).map((c) => {
              const differs = baseTables !== null && !baseTables.has(c.table);
              return (
                <span
                  key={`${c.table}-${c.type}`}
                  data-diff={differs || undefined}
                  className={cx("rounded-md border px-2 py-px", differs ? "border-warn bg-warn-tint text-warn" : "border-line bg-surface text-ink-2")}
                >
                  {c.title}
                </span>
              );
            })}
            <button type="button" onClick={onView} disabled={viewing} className="font-semibold text-lane-1 disabled:text-ink-3">
              {viewing ? "Đang xem dashboard này" : "Xem dashboard này"}
            </button>
          </div>
        )}
      </>
    );
  }
  return (
    <article aria-label={endpoint.name} className="grid min-w-0 content-start overflow-hidden rounded-xl border border-line bg-surface">
      <header className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-b border-line bg-sunken px-3.5 py-2.5">
        <span className="flex min-w-0 items-center gap-2 font-bold"><Dot color={color} /><span className="truncate">{endpoint.name}</span></span>
        {isBaseline && <span className="rounded-md border border-line bg-surface px-1.5 text-[11.5px] font-semibold text-ink-2">Mốc</span>}
        <span className="ml-auto">{pill}</span>
      </header>
      <div className="grid min-w-0 gap-3 p-3.5">{body}</div>
    </article>
  );
}

function Compare({ base, others, colors }: {
  base: { endpoint: Endpoint; result: ReportResult };
  others: { endpoint: Endpoint; result: ReportResult }[];
  colors: Record<string, string>;
}) {
  const b = base.result;
  const baseCharts = drawn(b);
  return (
    <div className="grid gap-2 border-t border-line pt-4">
      <h3 className="text-[14px] font-bold">So với mốc {base.endpoint.name}</h3>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table role="table" aria-label={`So với mốc ${base.endpoint.name}`} className="w-full text-left text-[13.5px] [&_td]:[overflow-wrap:anywhere] max-sm:block">
          <thead className="bg-sunken text-[12px] text-ink-3 max-sm:hidden">
            <tr>{["Endpoint", "Biểu đồ", "Báo cáo", "Tốc độ"].map((h) => <th key={h} className="px-3.5 py-1.5 font-medium">{h}</th>)}</tr>
          </thead>
          <tbody className="max-sm:block">
            {others.map(({ endpoint, result: r }) => {
              const mine = drawn(r);
              const baseKeys = new Set(baseCharts.map((c) => c.table));
              const myKeys = new Set(mine.map((c) => c.table));
              const shared = mine.filter((c) => baseKeys.has(c.table)).length;
              const added = mine.filter((c) => !baseKeys.has(c.table)).map((c) => c.title);
              const dropped = baseCharts.filter((c) => !myKeys.has(c.table)).map((c) => c.title);
              const sameCharts = added.length === 0 && dropped.length === 0;
              const ungrounded = r.number_check.ungrounded.length;
              const mineTpot = r.report.metrics?.tpot_ms;
              const baseTpot = b.report.metrics?.tpot_ms;
              const speed = mineTpot && baseTpot ? baseTpot / mineTpot : null;
              return (
                <tr key={endpoint.endpoint_id} role="row" className="border-t border-line align-top max-sm:grid max-sm:gap-0.5 max-sm:py-1.5">
                  <td role="cell" data-label="Endpoint" className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]"><span className="flex items-center gap-2 font-bold"><Dot color={colors[endpoint.endpoint_id]} />{endpoint.name}</span></td>
                  <td role="cell" data-label="Biểu đồ" className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {sameCharts
                      ? <><b className="text-good">Giống mốc</b><small className="block text-[12.5px] text-ink-3">{shared}/{mine.length} biểu đồ giống</small></>
                      : (
                        <>
                          <b className="text-warn">Khác mốc</b>
                          <small className="block text-[12.5px] text-ink-3">
                            {shared}/{Math.max(mine.length, baseCharts.length)} biểu đồ giống
                            {added.length > 0 && `; chọn ${added.join(", ")}`}
                            {dropped.length > 0 && ` thay vì ${dropped.join(", ")}`}
                          </small>
                        </>
                      )}
                  </td>
                  <td role="cell" data-label="Báo cáo" className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {ungrounded === 0
                      ? <><b className="text-good">Số đều khớp</b><small className="block text-[12.5px] text-ink-3">{r.number_check.numbers}/{r.number_check.numbers} số khớp bảng</small></>
                      : <><b className="text-warn">{ungrounded} số không khớp</b><small className="block text-[12.5px] text-ink-3">{r.number_check.numbers - ungrounded}/{r.number_check.numbers} số khớp bảng</small></>}
                  </td>
                  <td role="cell" data-label="Tốc độ" className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {speed === null ? <span className="text-ink-3">Chưa đủ số đo</span> : (
                      <>
                        <b className={speed >= 1 ? "text-good" : "text-bad"}>{speed >= 1 ? `Nhanh hơn ${formatSpeedup(speed)}` : `Chậm hơn ${formatSpeedup(1 / speed)}`}</b>
                        <small className="block text-[12.5px] text-ink-3">theo TPOT; tổng {formatMs(totalMs(r))} so với {formatMs(totalMs(b))}</small>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DataDrawer({ result, onClose }: { result?: ReportResult; onClose: () => void }) {
  return (
    <Drawer title="Số liệu đầu vào" onClose={onClose}>
      <div className="grid gap-4">
        {result?.tables.map((t) => (
          <section key={t.key} className="grid gap-1.5">
            <h3 className="flex flex-wrap items-baseline gap-2 text-[14px] font-semibold">
              {t.title}<code className="font-mono text-[12px] font-normal text-ink-3">{t.key}, {t.rows.length} dòng</code>
            </h3>
            <ResultTable columns={t.columns} rows={t.rows} max={20} flag={quantityFlag(t)} />
          </section>
        ))}
      </div>
    </Drawer>
  );
}

export default function Node3Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const baseline = endpoints.find((e) => e.is_baseline);
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints).filter((id) => id !== baseline?.endpoint_id));
  const [request, setRequest] = useState("");
  const [limit, setLimit] = useState<number | null>(10);
  const [round, setRound] = useState<Round | null>(null);
  const [view, setView] = useState<string | null>(null);
  const [showData, setShowData] = useState(false);
  const latest = useRef(0);
  const busy = round !== null && round.order.some((id) => !round.answers[id].result && !round.answers[id].error);
  const now = useNow(busy);

  const colors = endpointColors(endpoints);
  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const baseLive = baseline && baseline.status !== "down" ? baseline : undefined;
  const chosen = selected.filter((id) => id !== baseline?.endpoint_id && byId.get(id)?.status !== "down");
  const order = [...(baseLive ? [baseLive.endpoint_id] : []), ...chosen].slice(0, MAX_ENDPOINTS);

  // `started` is read by the event handler that runs it.
  function run(text: string, started: number) {
    const value = text.trim();
    if (!value || order.length === 0) return;
    const id = ++latest.current;
    setRequest(value);
    setView(null);
    setRound({ request: value, order, answers: Object.fromEntries(order.map((e) => [e, { started }])) });
    const settle = (endpointId: string, answer: Partial<Answer>) => setRound((current) => (
      current && latest.current === id
        ? { ...current, answers: { ...current.answers, [endpointId]: { ...current.answers[endpointId], ...answer } } }
        : current));
    for (const endpointId of order) {
      api.report(endpointId, value, limit).then(
        (result) => settle(endpointId, { result }),
        (exc) => settle(endpointId, { error: runError(exc) }),
      );
    }
  }

  if (endpoints.length === 0) return <><PageHead title="Báo cáo và dashboard" /><NeedEndpoint what="Node 3" onGo={() => go("endpoints")} /></>;

  const results = round ? round.order.flatMap((id) => {
    const result = round.answers[id].result;
    const endpoint = byId.get(id);
    return result && endpoint ? [{ endpoint, result }] : [];
  }) : [];
  const first = results[0]?.result;
  const empty = first !== undefined && first.scope.invoice_count === 0;
  const kpis = first?.tables.find((t) => t.key === "kpis");
  const baseResult = baseLive ? results.find((r) => r.endpoint.endpoint_id === baseLive.endpoint_id)?.result : undefined;
  const viewId = view && results.some((r) => r.endpoint.endpoint_id === view) ? view
    : (results.find((r) => r.endpoint.endpoint_id !== baseLive?.endpoint_id) ?? results[0])?.endpoint.endpoint_id;
  const shown = results.find((r) => r.endpoint.endpoint_id === viewId)?.result;
  const tables = shown ? Object.fromEntries(shown.tables.map((t) => [t.key, t])) : {};
  const others = results.filter((r) => r.endpoint.endpoint_id !== baseLive?.endpoint_id);

  return (
    <>
      <PageHead title="Báo cáo và dashboard" actions={<Button variant="ghost" icon={Database} onClick={() => setShowData(true)}>Số liệu đầu vào</Button>} />

      <section aria-label="Yêu cầu báo cáo" className="grid gap-3.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            run(request, Date.now());
          }}
          className="rounded-xl border border-line bg-surface focus-within:border-lane-1 focus-within:ring-3 focus-within:ring-lane-1/20"
        >
          <label htmlFor="report-request" className="sr-only">Yêu cầu báo cáo</label>
          <textarea
            id="report-request"
            rows={2}
            maxLength={500}
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                run(request, Date.now());
              }
            }}
            placeholder="Bạn cần báo cáo gì"
            className="block min-h-[60px] w-full resize-none bg-transparent px-4 pb-1.5 pt-3.5 text-base leading-snug outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-2 px-3 pb-2.5 pl-4">
            <span className="text-[13px] text-ink-3">Hoá đơn gần nhất</span>
            <div role="radiogroup" aria-label="Phạm vi" className="inline-flex max-w-full flex-wrap rounded-full border border-line bg-sunken p-0.5">
              {SCOPES.map(([value, text, name]) => (
                <button
                  key={text}
                  type="button"
                  role="radio"
                  aria-checked={limit === value}
                  aria-label={name}
                  onClick={() => setLimit(value)}
                  className={cx("rounded-full px-3 py-0.5 text-[13px]", limit === value ? "bg-surface font-semibold text-ink shadow-sm" : "text-ink-2")}
                >
                  {text}
                </button>
              ))}
            </div>
            <span className="text-[13px] text-ink-3">Lựa chọn LLM</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              alwaysOn={baseLive ? [baseLive.endpoint_id] : []}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-(MAX_ENDPOINTS - 1))))}
            />
            <Button type="submit" icon={Play} loading={busy} disabled={busy || !request.trim() || order.length === 0} className="ml-auto">
              {busy ? "Đang tạo" : "Tạo báo cáo"}
            </Button>
          </div>
        </form>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Yêu cầu mẫu">
          <span className="mr-0.5 text-[13px] text-ink-3">Yêu cầu mẫu</span>
          {SAMPLES.map((s) => (
            <button
              key={s.short}
              type="button"
              title={s.text}
              disabled={busy}
              onClick={() => run(s.text, Date.now())}
              className={cx(
                "rounded-full border px-3 py-1 text-[13px] text-ink-2 hover:bg-sunken hover:text-ink disabled:opacity-50",
                round?.request === s.text ? "border-ink font-semibold text-ink" : "border-line",
              )}
            >
              {s.short}
            </button>
          ))}
        </div>
      </section>

      {round && (
        <section aria-label="Kết quả" className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 className="text-[15px] font-bold">Kết quả</h2>
            <span className="text-ink-2">“{round.request}”</span>
            {first && !empty && <span className="ml-auto text-[12.5px] text-ink-3">{first.scope.label}</span>}
          </div>
          {empty ? (
            <Notice tone="info">Phạm vi này chưa có hoá đơn nào. Trích xuất hoá đơn ở trang Node 1 trước.</Notice>
          ) : (
            <>
              {kpis && <Totals table={kpis} />}
              <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))]">
                {round.order.map((id) => {
                  const endpoint = byId.get(id);
                  return endpoint ? (
                    <Column
                      key={id}
                      endpoint={endpoint}
                      color={colors[id]}
                      isBaseline={id === baseLive?.endpoint_id}
                      answer={round.answers[id]}
                      base={baseResult}
                      viewing={id === viewId}
                      onView={() => setView(id)}
                      now={now}
                    />
                  ) : null;
                })}
              </div>
              {shown && drawn(shown).length > 0 && (
                <div className="grid gap-3">
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                    <h2 className="text-[15px] font-bold">Dashboard</h2>
                    {results.length > 1 && (
                      <div role="radiogroup" aria-label="Dashboard của endpoint" className="inline-flex max-w-full flex-wrap rounded-full border border-line bg-sunken p-0.5">
                        {results.map(({ endpoint }) => (
                          <button
                            key={endpoint.endpoint_id}
                            type="button"
                            role="radio"
                            aria-checked={endpoint.endpoint_id === viewId}
                            aria-label={`Dashboard của ${endpoint.name}`}
                            onClick={() => setView(endpoint.endpoint_id)}
                            className={cx("rounded-full px-3 py-0.5 text-[13px]", endpoint.endpoint_id === viewId ? "bg-surface font-semibold text-ink shadow-sm" : "text-ink-2")}
                          >
                            {endpoint.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))]">
                    {drawn(shown).map((c) => (
                      <figure key={`${c.table}-${c.type}-${c.value}`} className="m-0 grid min-w-0 content-start gap-2.5 rounded-xl border border-line bg-surface px-4 pb-3 pt-3.5">
                        <figcaption className="flex flex-wrap items-baseline gap-2 text-[14px] font-semibold">
                          {c.title}<span className="text-[12px] font-normal text-ink-3">{TYPE_NAMES[c.type]}</span>
                        </figcaption>
                        <Chart spec={c} table={tables[c.table]} />
                      </figure>
                    ))}
                  </div>
                </div>
              )}
              {baseResult && baseLive && others.length > 0 && !busy && (
                <Compare base={{ endpoint: baseLive, result: baseResult }} others={others} colors={colors} />
              )}
            </>
          )}
        </section>
      )}

      {showData && <DataDrawer result={first} onClose={() => setShowData(false)} />}
    </>
  );
}
