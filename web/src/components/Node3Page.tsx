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
  ReferencePicker,
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
import { useI18n, type Dict } from "@/lib/i18n";

const MONEY = /tien|thue|amount|total|trung_binh/i;
const QUANTITY = /so_luong|quantity|qty/i;
const OCR_SUSPECT_QUANTITY = 1000;
const MAX_ENDPOINTS = 6;

interface Answer {
  started: number;
  result?: ReportResult;
  error?: string;
}

interface Round {
  request: string;
  order: string[];
  answers: Record<string, Answer>;
}

function runError(exc: unknown, t: Dict): string {
  if (exc instanceof ApiError && exc.status === 429) return t.errRateLimit;
  return (exc as Error).message;
}

const totalMs = (r: ReportResult) => r.timings_ms.data + r.timings_ms.dashboard + r.timings_ms.report;
const drawn = (r: ReportResult) => r.dashboard.charts.filter((c) => c.type !== "kpi" && r.tables.some((t) => t.key === c.table));

function quantityFlag(table: DataTable, label: string) {
  const index = table.columns.findIndex((c) => QUANTITY.test(c));
  if (index < 0) return undefined;
  return (row: unknown[]) => (typeof row[index] === "number" && (row[index] as number) > OCR_SUSPECT_QUANTITY ? label : null);
}

function Chart({ spec, table }: { spec: ChartSpec; table: DataTable }) {
  const { t } = useI18n();
  if (spec.type === "table") return <ResultTable columns={table.columns} rows={table.rows} max={12} flag={quantityFlag(table, t.ocrSuspectQty)} />;
  const value = table.columns.indexOf(spec.value ?? "");
  const texts = table.columns.map((_, i) => i).filter((i) => i !== value && table.rows.some((r) => typeof r[i] === "string"));
  const unique = texts.find((i) => new Set(table.rows.map((r) => r[i])).size === table.rows.length);
  const labelOf = (r: unknown[]) =>
    unique !== undefined ? String(r[unique] ?? "—") : texts.map((i) => String(r[i] ?? "—")).join(", ");
  const rows = table.rows
    .filter((r) => typeof r[value] === "number")
    .map((r) => [labelOf(r), r[value] as number] as [string, number]);
  const sorted = spec.type === "bar" ? [...rows].sort((a, b) => b[1] - a[1]).slice(0, 8) : rows;
  return <BarChart rows={sorted} horizontal={spec.type === "bar"} unit={MONEY.test(spec.value ?? "") ? "VND" : ""} label={spec.title} />;
}

function Totals({ table }: { table: DataTable }) {
  const { lang, t } = useI18n();
  const row = table.rows[0] ?? [];
  return (
    <dl aria-label={t.totalsAria} className="grid overflow-hidden rounded-xl border border-line bg-surface [grid-template-columns:repeat(auto-fit,minmax(min(50%,150px),1fr))]">
      {table.columns.map((c, i) => (typeof row[i] === "number" ? (
        <div key={c} className="-ml-px -mt-px grid min-w-0 gap-0.5 border-l border-t border-line px-4 py-3.5">
          <dt className="text-[12.5px] text-ink-3">{t.kpiLabels[c] ?? c}</dt>
          <dd className="num m-0 text-[22px] font-bold tracking-tight">{MONEY.test(c) ? formatMoney(row[i] as number, "VND", lang) : formatNumber(row[i] as number, lang)}</dd>
        </div>
      ) : null))}
    </dl>
  );
}

function ReportText({ text, ungrounded }: { text: string; ungrounded: string[] }) {
  const { t } = useI18n();
  const pattern = ungrounded.length ? new RegExp(`(${ungrounded.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`) : null;
  return (
    <div className="grid max-w-[68ch] gap-2.5 text-[15px] leading-relaxed">
      {text.split(/\n+/).filter((p) => p.trim()).map((p, i) => (
        <p key={i}>
          {pattern ? p.split(pattern).map((part, j) => (ungrounded.includes(part)
            ? <mark key={j} title={t.ungroundedTooltip} className="rounded bg-warn-tint px-0.5 font-semibold text-warn">{part}</mark>
            : part)) : p}
        </p>
      ))}
    </div>
  );
}

function Stats({ result }: { result: ReportResult }) {
  const { lang, t } = useI18n();
  const m = result.report.metrics;
  const cells: [string, string][] = [
    [t.statLatency, formatMs(totalMs(result), lang)],
    ["TTFT", formatMs(m?.ttft_ms, lang)],
    ["TPOT", formatMs(m?.tpot_ms, lang)],
    [t.statTokens, formatNumber(m?.completion_tokens, lang)],
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

function Column({ endpoint, color, answer, base, viewing, onView, now }: {
  endpoint: Endpoint;
  color: string;
  answer: Answer;
  base?: ReportResult;
  viewing: boolean;
  onView: () => void;
  now: number;
}) {
  const { t } = useI18n();
  const r = answer.result;
  const pill = answer.error ? <Pill tone="bad">{t.statusError}</Pill>
    : !r ? <Pill tone="idle"><Spinner className="size-3.5" />{t.statusGenerating(secondsSince(answer.started, now))}</Pill>
    : r.report.status === "ok" ? <Pill tone="good">{t.statusGenerated}</Pill> : <Pill tone="bad">{t.statusGenFailed}</Pill>;
  let body: ReactNode;
  if (answer.error) body = <p role="alert" className="text-bad">{answer.error}</p>;
  else if (!r) body = <p className="text-ink-3">{t.node3RunningText}</p>;
  else {
    const ungrounded = r.number_check.ungrounded;
    const foreign = r.language_check?.foreign_words ?? [];
    const baseTables = base && base !== r ? new Set(drawn(base).map((c) => c.table)) : null;
    body = (
      <>
        {r.report.status === "ok" && r.report.text
          ? <ReportText text={r.report.text} ungrounded={ungrounded} />
          : <p role="alert" className="text-bad">{t.node3FailedPrefix} {r.report.error ?? r.report.status}</p>}
        <Stats result={r} />
        <div className="flex flex-wrap gap-1.5">
          {r.report.status === "ok" && (ungrounded.length === 0
            ? <Pill tone="good">{t.numbersVerified(r.number_check.numbers)}</Pill>
            : <Pill tone="warn">{t.numbersUnverified(ungrounded.length, ungrounded.join(", "))}</Pill>)}
          {foreign.length > 0 && <Pill tone="warn">{t.foreignWordsFlag(foreign.join(", "))}</Pill>}
          {r.dashboard.status === "fallback" && (
            <span title={r.dashboard.error ?? undefined} className="inline-flex items-center rounded-full bg-warn-tint px-2.5 py-0.5 text-[12.5px] font-semibold text-warn">
              {t.fallbackDashboard}
            </span>
          )}
        </div>
        {drawn(r).length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-[12.5px] text-ink-3">
            <span>{t.selectedCharts}</span>
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
              {viewing ? t.viewingDashboard : t.viewDashboard}
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
  const { lang, t } = useI18n();
  const b = base.result;
  const baseCharts = drawn(b);
  return (
    <div className="grid gap-2 border-t border-line pt-4">
      <h3 className="text-[14px] font-bold">{t.compareTableTitle(base.endpoint.name)}</h3>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table role="table" aria-label={t.compareTableTitle(base.endpoint.name)} className="w-full text-left text-[13.5px] [&_td]:[overflow-wrap:anywhere] max-sm:block">
          <thead className="bg-sunken text-[12px] text-ink-3 max-sm:hidden">
            <tr>{[t.colEndpoint, t.colCharts, t.colReport, t.colSpeed].map((h) => <th key={h} className="px-3.5 py-1.5 font-medium">{h}</th>)}</tr>
          </thead>
          <tbody className="max-sm:block">
            {others.map(({ endpoint, result: r }) => {
              const mine = drawn(r);
              const baseKeys = new Set(baseCharts.map((c) => c.table));
              const myKeys = new Set(mine.map((c) => c.table));
              const shared = mine.filter((c) => baseKeys.has(c.table)).length;
              const added = mine.filter((c) => !baseKeys.has(c.table));
              const dropped = baseCharts.filter((c) => !myKeys.has(c.table));
              const sameCharts = added.length === 0 && dropped.length === 0;
              const ungrounded = r.number_check.ungrounded.length;
              const mineTpot = r.report.metrics?.tpot_ms;
              const baseTpot = b.report.metrics?.tpot_ms;
              const speed = mineTpot && baseTpot ? baseTpot / mineTpot : null;
              return (
                <tr key={endpoint.endpoint_id} role="row" className="border-t border-line align-top max-sm:grid max-sm:gap-0.5 max-sm:py-1.5">
                  <td data-label={t.colEndpoint} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]"><span className="flex items-center gap-2 font-bold"><Dot color={colors[endpoint.endpoint_id]} />{endpoint.name}</span></td>
                  <td data-label={t.colCharts} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {sameCharts
                      ? <><b className="text-good">{t.matchesRef}</b><small className="block text-[12.5px] text-ink-3">{t.chartsMatchDetail(mine.length)}</small></>
                      : <><b className="text-warn">{t.differsRef}</b><small className="block text-[12.5px] text-ink-3">{t.chartsDiffDetail(shared, Math.max(mine.length, baseCharts.length))}</small></>}
                  </td>
                  <td data-label={t.colReport} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {ungrounded === 0
                      ? <><b className="text-good">{t.verifiedNumbers}</b><small className="block text-[12.5px] text-ink-3">{r.number_check.numbers}/{r.number_check.numbers}</small></>
                      : <><b className="text-warn">{t.unverifiedNumbers(ungrounded)}</b><small className="block text-[12.5px] text-ink-3">{r.number_check.numbers - ungrounded}/{r.number_check.numbers}</small></>}
                  </td>
                  <td data-label={t.colSpeed} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {speed === null ? <span className="text-ink-3">{t.insufficientMetrics}</span> : (
                      <>
                        <b className={speed >= 1 ? "text-good" : "text-bad"}>{speed >= 1 ? t.fasterBy(formatSpeedup(speed, lang)) : t.slowerBy(formatSpeedup(1 / speed, lang))}</b>
                        <small className="block text-[12.5px] text-ink-3">{t.totalTimeCompare(formatMs(totalMs(r), lang), formatMs(totalMs(b), lang))}</small>
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
  const { lang, t } = useI18n();
  return (
    <Drawer title={t.inputDataTitle} onClose={onClose}>
      <div className="grid gap-4">
        {result?.tables.map((tbl) => (
          <section key={tbl.key} className="grid gap-1.5">
            <h3 className="flex flex-wrap items-baseline gap-2 text-[14px] font-semibold">
              {tbl.title}
              <span className="text-[12px] font-normal text-ink-3">{t.sqlRowsCount(formatNumber(tbl.rows.length, lang))}</span>
            </h3>
            <ResultTable columns={tbl.columns} rows={tbl.rows} max={20} flag={quantityFlag(tbl, t.ocrSuspectQty)} />
          </section>
        ))}
      </div>
    </Drawer>
  );
}

export default function Node3Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints));
  const [request, setRequest] = useState("");
  const [limit, setLimit] = useState<number | null>(10);
  const [round, setRound] = useState<Round | null>(null);
  const [refChoice, setRefChoice] = useState<string | null>(null);
  const [view, setView] = useState<string | null>(null);
  const [showData, setShowData] = useState(false);
  const latest = useRef(0);
  const busy = round !== null && round.order.some((id) => !round.answers[id].result && !round.answers[id].error);
  const now = useNow(busy);

  const scopes: [number | null, string, string][] = [
    [5, "5", t.scopeRecent(5)],
    [10, "10", t.scopeRecent(10)],
    [25, "25", t.scopeRecent(25)],
    [null, t.scopeAll, t.scopeAllAria],
  ];

  const colors = endpointColors(endpoints);
  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const order = selected.filter((id) => byId.get(id)?.status !== "down").slice(0, MAX_ENDPOINTS);

  function run(text: string, started: number) {
    const value = text.trim();
    if (!value || order.length === 0) return;
    const id = ++latest.current;
    setRequest(value);
    setRefChoice(null);
    setView(null);
    setRound({ request: value, order, answers: Object.fromEntries(order.map((e) => [e, { started }])) });
    const settle = (endpointId: string, answer: Partial<Answer>) => setRound((current) => (
      current && latest.current === id
        ? { ...current, answers: { ...current.answers, [endpointId]: { ...current.answers[endpointId], ...answer } } }
        : current));
    for (const endpointId of order) {
      api.report(endpointId, value, limit).then(
        (result) => settle(endpointId, { result }),
        (exc) => settle(endpointId, { error: runError(exc, t) }),
      );
    }
  }

  if (endpoints.length === 0) return <><PageHead title={t.pageNode3} /><NeedEndpoint what="Node 3" onGo={() => go("endpoints")} /></>;

  const results = round ? round.order.flatMap((id) => {
    const result = round.answers[id].result;
    const endpoint = byId.get(id);
    return result && endpoint ? [{ endpoint, result }] : [];
  }) : [];
  const first = results[0]?.result;
  const empty = first !== undefined && first.scope.invoice_count === 0;
  const kpis = first?.tables.find((tbl) => tbl.key === "kpis");
  const roundEndpoints = round ? round.order.map((id) => byId.get(id)).filter((e): e is Endpoint => Boolean(e)) : [];
  const referenceId = roundEndpoints.length >= 2
    ? (refChoice && roundEndpoints.some((e) => e.endpoint_id === refChoice) ? refChoice : roundEndpoints[0].endpoint_id)
    : null;
  const refEndpoint = referenceId ? byId.get(referenceId) : undefined;
  const refResult = referenceId ? results.find((r) => r.endpoint.endpoint_id === referenceId)?.result : undefined;
  const viewId = view && results.some((r) => r.endpoint.endpoint_id === view) ? view : results[0]?.endpoint.endpoint_id;
  const shown = results.find((r) => r.endpoint.endpoint_id === viewId)?.result;
  const tables = shown ? Object.fromEntries(shown.tables.map((tbl) => [tbl.key, tbl])) : {};
  const others = referenceId ? results.filter((r) => r.endpoint.endpoint_id !== referenceId) : [];

  return (
    <>
      <PageHead title={t.pageNode3} actions={<Button variant="ghost" icon={Database} onClick={() => setShowData(true)}>{t.inputDataBtn}</Button>} />

      <section aria-label={t.reportSectionAria} className="grid gap-3.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            run(request, Date.now());
          }}
          className="rounded-xl border border-line bg-surface focus-within:border-lane-1 focus-within:ring-3 focus-within:ring-lane-1/20"
        >
          <label htmlFor="report-request" className="sr-only">{t.reportInputLabel}</label>
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
            placeholder={t.reportPlaceholder}
            className="block min-h-[60px] w-full resize-none bg-transparent px-4 pb-1.5 pt-3.5 text-base leading-snug outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-2 px-3 pb-2.5 pl-4">
            <span className="text-[13px] text-ink-3">{t.scopeLabel}</span>
            <div role="radiogroup" aria-label={t.scopeAria} className="inline-flex max-w-full flex-wrap rounded-full border border-line bg-sunken p-0.5">
              {scopes.map(([value, text, name]) => (
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
            <span className="text-[13px] text-ink-3">{t.selectLlm}</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-MAX_ENDPOINTS)))}
            />
            <Button type="submit" icon={Play} loading={busy} disabled={busy || !request.trim() || order.length === 0} className="ml-auto">
              {busy ? t.generatingBtn : t.generateReportBtn}
            </Button>
          </div>
        </form>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={t.samplePromptsLabel}>
          <span className="mr-0.5 text-[13px] text-ink-3">{t.samplePromptsLabel}</span>
          {t.sampleReports.map((s) => (
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
        <section aria-label={t.resultsPanel} className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h2 className="text-[15px] font-bold">{t.resultsPanel}</h2>
            <span className="text-ink-2">“{round.request}”</span>
            {referenceId && (
              <span className="ml-auto">
                <ReferencePicker
                  options={roundEndpoints.map((e) => ({ id: e.endpoint_id, name: e.name }))}
                  value={referenceId}
                  onChange={setRefChoice}
                  colors={colors}
                />
              </span>
            )}
          </div>
          {empty ? (
            <Notice tone="info">{t.emptyScopeNotice}</Notice>
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
                      answer={round.answers[id]}
                      base={refResult}
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
                    <h2 className="text-[15px] font-bold">{t.dashboardHeading}</h2>
                    {results.length > 1 && (
                      <div role="radiogroup" aria-label={t.dashboardPickerAria} className="inline-flex max-w-full flex-wrap rounded-full border border-line bg-sunken p-0.5">
                        {results.map(({ endpoint }) => (
                          <button
                            key={endpoint.endpoint_id}
                            type="button"
                            role="radio"
                            aria-checked={endpoint.endpoint_id === viewId}
                            aria-label={t.dashboardOf(endpoint.name)}
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
                        <figcaption className="text-[14px] font-semibold">{c.title}</figcaption>
                        <Chart spec={c} table={tables[c.table]} />
                      </figure>
                    ))}
                  </div>
                </div>
              )}
              {refResult && refEndpoint && others.length > 0 && !busy && (
                <Compare base={{ endpoint: refEndpoint, result: refResult }} others={others} colors={colors} />
              )}
            </>
          )}
        </section>
      )}

      {showData && <DataDrawer result={first} onClose={() => setShowData(false)} />}
    </>
  );
}
