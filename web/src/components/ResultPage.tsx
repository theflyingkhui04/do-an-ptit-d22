"use client";

import { ChevronRight, Play } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { LineChart, type Series } from "@/components/charts";
import type { PageId } from "@/components/Console";
import {
  Button,
  CodeBlock,
  Dot,
  Drawer,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Pill,
  ReferencePicker,
  Spinner,
  SqlBlock,
  baseModel,
  cx,
  defaultTarget,
  endpointColors,
  secondsSince,
  useNow,
} from "@/components/ui";
import type { Api, BenchCell, BenchConfig, BenchRun, ChainResult, ChainRow, Endpoint, SpeedResult } from "@/lib/api";
import { leafFields, sameValue } from "@/lib/fields";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { useI18n, type Dict, type Lang } from "@/lib/i18n";
import { M3_05_SPEEDUP, M3_06_CHAIN, OFFICIAL_LOADS, SAMPLES } from "@/lib/official";

type Workload = "W1" | "W2" | "W3";
type Mode = "speed" | "chain";
type PresetKey = "quick" | "full";
type MetricKey = "tpot" | "ttft" | "latency" | "throughput";

const PRESETS: Record<PresetKey, { requests: number; loads: number[]; repeats: number }> = {
  quick: { requests: 8, loads: [1, 8], repeats: 1 },
  full: { requests: 32, loads: [1, 2, 4, 8, 16, 32], repeats: 3 },
};
const FINAL = ["done", "failed", "interrupted"];
const OFFICIAL_COLORS = ["var(--lane-1)", "var(--lane-2)", "var(--lane-3)"];

const METRIC_META: Record<MetricKey, { lower: boolean; read: (c: BenchCell) => number | null; format: (v: number, lang: Lang) => string }> = {
  tpot: { lower: true, read: (c) => c.tpot_ms_p50, format: (v, lang) => `${formatNumber(Math.round(v * 10) / 10, lang)} ms` },
  ttft: { lower: true, read: (c) => c.ttft_ms_p50, format: (v, lang) => formatMs(v, lang) },
  latency: { lower: true, read: (c) => c.latency_ms_p50, format: (v, lang) => formatMs(v, lang) },
  throughput: { lower: false, read: (c) => c.throughput_tps, format: (v, lang) => `${formatNumber(Math.round(v), lang)} tok/s` },
};

function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const two = (n: number) => String(n).padStart(2, "0");
  return `${two(d.getDate())}/${two(d.getMonth() + 1)} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

function compared(metric: MetricKey, ratio: number, lang: Lang, t: Dict): string {
  if (Math.abs(ratio - 1) < 0.005) return t.equalShort;
  const sp = formatSpeedup(ratio > 1 ? ratio : 1 / ratio, lang);
  if (METRIC_META[metric].lower) return ratio > 1 ? t.fasterShort(sp) : t.slowerShort(sp);
  return ratio > 1 ? t.higherShort(sp) : t.lowerShort(sp);
}

const ratioOf = (metric: MetricKey, base: number, mine: number) => (METRIC_META[metric].lower ? base / mine : mine / base);

function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string;
  value: T;
  options: [T, ReactNode][];
  onChange: (value: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex max-w-full flex-wrap rounded-full border border-line bg-sunken p-0.5">
      {options.map(([key, text]) => (
        <button
          key={key}
          type="button"
          role="radio"
          aria-checked={value === key}
          onClick={() => onChange(key)}
          className={cx("rounded-full px-3 py-0.5 text-[13px]", value === key ? "bg-surface font-semibold text-ink shadow-sm" : "text-ink-2")}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function SpeedLive({ result, colors }: { result: SpeedResult; colors: Record<string, string> }) {
  const { lang, t } = useI18n();
  const [metric, setMetric] = useState<MetricKey>("tpot");
  const rawIds = [...new Set(result.cells.map((c) => c.endpoint_id))];
  const names = Object.fromEntries(result.cells.map((c) => [c.endpoint_id, c.name]));
  const [refChoice, setRefChoice] = useState<string | null>(null);
  const referenceId = rawIds.length >= 2
    ? (refChoice && rawIds.includes(refChoice) ? refChoice : (result.baseline && rawIds.includes(result.baseline) ? result.baseline : rawIds[0]))
    : null;
  const ids = [...rawIds].sort((a, b) => Number(b === referenceId) - Number(a === referenceId));
  const m = METRIC_META[metric];
  const cell = (id: string, load: number) => result.cells.find((c) => c.endpoint_id === id && c.load === load);
  const value = (id: string, load: number) => {
    const c = cell(id, load);
    return c ? m.read(c) : null;
  };
  const fmt = (v: number) => m.format(v, lang);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line">
        <div role="tablist" aria-label={t.metricTabsAria} className="flex flex-wrap gap-1">
          {(Object.keys(METRIC_META) as MetricKey[]).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={k === metric}
              onClick={() => setMetric(k)}
              className={cx("-mb-px border-b-2 px-3 py-2 text-[13.5px]", k === metric ? "border-ink font-semibold text-ink" : "border-transparent text-ink-2 hover:text-ink")}
            >
              {t.metricLabels[k]}
            </button>
          ))}
        </div>
        {referenceId && (
          <div className="pb-1.5">
            <ReferencePicker
              options={rawIds.map((id) => ({ id, name: names[id] }))}
              value={referenceId}
              onChange={setRefChoice}
              colors={colors}
            />
          </div>
        )}
      </div>

      {result.loads.length <= 3 ? (
        <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr))]">
          {result.loads.map((load) => {
            const max = Math.max(1, ...ids.map((id) => value(id, load) ?? 0));
            const base = referenceId ? value(referenceId, load) : null;
            return (
              <div key={load} role="group" aria-label={t.concurrencyGroup(load)} className="grid content-start gap-2.5 rounded-xl border border-line px-3.5 py-3">
                <h4 className="text-[13px] font-semibold text-ink-2">{t.concurrencyGroup(load)}</h4>
                {ids.map((id) => {
                  const v = value(id, load);
                  const ratio = v !== null && base ? ratioOf(metric, base, v) : null;
                  return (
                    <div key={id} className="grid grid-cols-[minmax(0,8.5em)_minmax(0,1fr)_auto] items-center gap-2.5 text-[13px]">
                      <span className="flex min-w-0 items-center gap-1.5 font-semibold" title={names[id]}>
                        <Dot color={colors[id] ?? "var(--ink-2)"} />
                        <span className="truncate">{names[id]}</span>
                      </span>
                      <span className="h-3.5 overflow-hidden rounded bg-sunken">
                        <span className="block h-full rounded" style={{ width: `${v === null ? 0 : Math.max(2, (v / max) * 100)}%`, background: colors[id] ?? "var(--ink-2)" }} />
                      </span>
                      <span className="grid min-w-[7.5em] justify-items-end leading-tight">
                        <b className="num">{v === null ? "—" : fmt(v)}</b>
                        {id !== referenceId && ratio !== null && (
                          <small className={cx("whitespace-nowrap text-[11.5px] font-semibold", ratio >= 1 ? "text-good" : "text-bad")}>{compared(metric, ratio, lang, t)}</small>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-line px-3.5 py-3">
          <LineChart
            label={t.metricLabels[metric]}
            loads={result.loads}
            format={fmt}
            series={ids.map((id) => ({
              name: names[id],
              color: colors[id] ?? "var(--ink-2)",
              dashed: id === referenceId,
              values: result.loads.map((load) => value(id, load)),
            }))}
          />
        </div>
      )}

      <details className="group rounded-lg border border-line">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-[13px] font-semibold text-ink-2 [&::-webkit-details-marker]:hidden">
          <ChevronRight aria-hidden className="size-3.5 transition-transform group-open:rotate-90" />{t.fullTableToggle}
        </summary>
        <div className="overflow-x-auto border-t border-line">
          <table className="w-full min-w-[720px] text-left text-[13.5px]" aria-label={t.fullTableAria}>
            <thead className="bg-sunken text-ink-3">
              <tr>
                {[t.colEndpoint, t.colConcurrency, "TTFT p50", "TPOT p50", t.colLatencyP50, t.colThroughput, t.colSuccess].map((h, i) => (
                  <th key={h} className={cx("px-3 py-2 font-semibold", i > 0 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="num">
              {result.cells.map((c) => (
                <tr key={`${c.endpoint_id}-${c.load}`} className="border-t border-line">
                  <td className="px-3 py-1.5"><span className="inline-flex items-center gap-2"><Dot color={colors[c.endpoint_id] ?? "var(--ink-2)"} />{c.name}</span></td>
                  <td className="px-3 py-1.5 text-right">{c.load}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.ttft_ms_p50, lang)}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.tpot_ms_p50, lang)}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.latency_ms_p50, lang)}</td>
                  <td className="px-3 py-1.5 text-right">{c.throughput_tps ? `${formatNumber(Math.round(c.throughput_tps), lang)} tok/s` : "—"}</td>
                  <td className={cx("px-3 py-1.5 text-right", c.n_ok < c.n && "text-bad")}>{c.n_ok}/{c.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function SpeedOfficial({ workload }: { workload: Workload }) {
  const { lang, t } = useI18n();
  const data = M3_05_SPEEDUP[workload];
  const series: Series[] = Object.entries(data).map(([name, values], i) => ({ name, color: OFFICIAL_COLORS[i], values }));
  return (
    <div className="grid gap-4">
      <div className="overflow-x-auto rounded-xl border border-line px-3.5 py-3">
        <LineChart
          label={t.officialSpeedupTitle}
          loads={OFFICIAL_LOADS}
          series={series}
          format={(v) => formatSpeedup(v, lang)}
          reference={1}
          referenceLabel={t.officialSpeedupRef}
        />
      </div>
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[560px] text-[13.5px]" aria-label={t.officialSpeedupTitle}>
          <thead className="bg-sunken text-[12px] text-ink-3">
            <tr>
              <th className="px-3.5 py-2 text-left font-medium">{t.colDecodingMethod}</th>
              {OFFICIAL_LOADS.map((l) => <th key={l} className="px-3.5 py-2 text-right font-medium">C={l}</th>)}
            </tr>
          </thead>
          <tbody className="num">
            <tr className="border-t border-line">
              <td className="px-3.5 py-2"><span className="inline-flex items-center gap-2 font-bold"><Dot color="var(--lane-ref)" />vanilla</span></td>
              {OFFICIAL_LOADS.map((l) => <td key={l} className="px-3.5 py-2 text-right font-semibold text-ink-3">1×</td>)}
            </tr>
            {series.map((s) => (
              <tr key={s.name} className="border-t border-line">
                <td className="px-3.5 py-2"><span className="inline-flex items-center gap-2 font-bold"><Dot color={s.color} />{s.name}</span></td>
                {s.values.map((v, i) => (
                  <td key={i} className={cx("px-3.5 py-2 text-right text-[15px] font-bold", v !== null && (v >= 1 ? "text-good" : "text-bad"))}>{v === null ? "—" : formatSpeedup(v, lang)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ChainBars({ rows }: { rows: { id: string; name: string; color: string; node1: number; node2: number; node3: number; isReference?: boolean; onDetail?: () => void }[] }) {
  const { lang, t } = useI18n();
  const nodes = [
    { key: "node1" as const, label: t.chainNode1, bg: "var(--node-1)", ink: "var(--node-1-ink)" },
    { key: "node2" as const, label: t.chainNode2, bg: "var(--node-2)", ink: "var(--node-2-ink)" },
    { key: "node3" as const, label: t.chainNode3, bg: "var(--node-3)", ink: "var(--node-3-ink)" },
  ];
  const max = Math.max(1, ...rows.map((r) => r.node1 + r.node2 + r.node3));
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        {nodes.map((n) => <span key={n.key} className="inline-flex items-center gap-1.5"><i aria-hidden className="inline-block size-3 rounded-sm" style={{ background: n.bg }} />{n.label}</span>)}
      </div>
      {rows.map((r) => {
        const total = r.node1 + r.node2 + r.node3;
        return (
          <div key={r.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3.5 gap-y-1.5 md:grid-cols-[minmax(150px,220px)_minmax(0,1fr)_auto]">
            <div className="grid min-w-0 justify-items-start gap-1">
              <span className="inline-flex max-w-full items-center gap-2 font-semibold"><Dot color={r.color} /><span className="truncate">{r.name}</span></span>
            </div>
            <div className="col-span-2 row-start-2 flex h-7 overflow-hidden rounded-md bg-sunken md:col-span-1 md:row-start-auto" role="img" aria-label={`${r.name}: ${t.chainNode1} ${formatMs(r.node1, lang)}, ${t.chainNode2} ${formatMs(r.node2, lang)}, ${t.chainNode3} ${formatMs(r.node3, lang)}`}>
              <div className="flex" style={{ width: `${(100 * total) / max}%` }}>
                {nodes.map((n, i) => (
                  <div
                    key={n.key}
                    title={`${n.label}: ${formatMs(r[n.key], lang)}`}
                    className={cx("flex items-center overflow-hidden whitespace-nowrap px-2 text-[12px] font-semibold", i > 0 && "border-l-2 border-surface")}
                    style={{ flex: Math.max(r[n.key], 1), background: n.bg, color: n.ink }}
                  >
                    {r[n.key] / max > 0.1 ? formatMs(r[n.key], lang) : ""}
                  </div>
                ))}
              </div>
            </div>
            <div className="grid justify-items-end gap-0.5 text-right">
              <b className="num">{formatMs(total, lang)}</b>
              {r.onDetail && (
                <button type="button" onClick={r.onDetail} aria-label={t.viewDetailAria(r.name)} className="text-[12.5px] font-semibold text-lane-1">{t.viewDetail}</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function chainQuality(row: ChainRow, base: ChainRow | undefined) {
  let fields: [number, number] | null = null;
  if (base && row !== base && row.invoice && base.invoice) {
    const mine = leafFields(row.invoice);
    const theirs = new Map(leafFields(base.invoice));
    const paths = new Set([...mine.map(([p]) => p), ...theirs.keys()]);
    const own = new Map(mine);
    fields = [[...paths].filter((p) => sameValue(own.get(p), theirs.get(p))).length, paths.size];
  }
  let questions: { same: number; total: number; differ: string[] } | null = null;
  if (base && row !== base) {
    const byId = new Map(base.questions.map((q) => [q.question_id, JSON.stringify(q.rows)]));
    const differ = row.questions.filter((q) => byId.get(q.question_id) !== JSON.stringify(q.rows)).map((q) => q.question_id);
    questions = { same: row.questions.length - differ.length, total: row.questions.length, differ };
  }
  return { fields, questions, numbers: row.number_check };
}

function ChainLive({ result, colors, onDetail }: { result: ChainResult; colors: Record<string, string>; onDetail: (row: ChainRow) => void }) {
  const { lang, t } = useI18n();
  const [refChoice, setRefChoice] = useState<string | null>(null);
  const referenceId = result.endpoints.length >= 2
    ? (refChoice && result.endpoints.some((e) => e.endpoint_id === refChoice)
      ? refChoice
      : (result.baseline && result.endpoints.some((e) => e.endpoint_id === result.baseline) ? result.baseline : result.endpoints[0].endpoint_id))
    : null;
  const rows = [...result.endpoints].sort((a, b) => Number(b.endpoint_id === referenceId) - Number(a.endpoint_id === referenceId));
  const base = rows.find((r) => r.endpoint_id === referenceId);
  const totalOf = (r: ChainRow) => r.timings_ms.node1 + r.timings_ms.node2 + r.timings_ms.node3;
  const good = (ok: boolean) => (ok ? "text-good" : "text-bad");
  return (
    <div className="grid gap-4">
      {referenceId && (
        <div className="flex justify-end">
          <ReferencePicker
            options={result.endpoints.map((r) => ({ id: r.endpoint_id, name: r.name }))}
            value={referenceId}
            onChange={setRefChoice}
            colors={colors}
          />
        </div>
      )}
      <ChainBars rows={rows.map((r) => ({
        id: r.endpoint_id,
        name: r.name,
        color: colors[r.endpoint_id] ?? "var(--ink-2)",
        ...r.timings_ms,
        isReference: r.endpoint_id === referenceId,
        onDetail: () => onDetail(r),
      }))} />
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[640px] text-left text-[13.5px]" aria-label={t.chainQualityAria}>
          <thead className="bg-sunken text-[12px] text-ink-3">
            <tr>
              {[t.colEndpoint, t.colChainN1, t.colChainN2, t.colChainN3, t.colChainTotal].map((h) => (
                <th key={h} className="px-3.5 py-2 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="num">
            {rows.map((r) => {
              const q = chainQuality(r, base);
              const isRef = r.endpoint_id === referenceId;
              const elapsed = totalOf(r);
              const baseElapsed = base ? totalOf(base) : null;
              const grounded = q.numbers.numbers - q.numbers.ungrounded.length;
              return (
                <tr key={r.endpoint_id} className="border-t border-line align-top">
                  <td className="px-3.5 py-2.5">
                    <span className="inline-flex items-center gap-2 font-bold"><Dot color={colors[r.endpoint_id] ?? "var(--ink-2)"} /><span>{r.name}</span></span>
                  </td>
                  <td className="px-3.5 py-2.5">
                    {q.fields ? <b className={cx("text-[15px]", good(q.fields[0] === q.fields[1]))}>{q.fields[0]}/{q.fields[1]}</b>
                      : <span className="text-ink-3">{isRef ? "—" : r.invoice ? "—" : t.node1Failed}</span>}
                  </td>
                  <td className="px-3.5 py-2.5">
                    {q.questions ? (
                      <>
                        <b className={cx("text-[15px]", good(q.questions.differ.length === 0))}>{q.questions.same}/{q.questions.total}</b>
                        <small className="block text-[12px] text-ink-3">{q.questions.differ.length ? t.diffAtQuestions(q.questions.differ) : t.matchesRef}</small>
                      </>
                    ) : <span className="text-ink-3">—</span>}
                  </td>
                  <td className="px-3.5 py-2.5"><b className={cx("text-[15px]", good(q.numbers.ungrounded.length === 0))}>{grounded}/{q.numbers.numbers}</b></td>
                  <td className="px-3.5 py-2.5">
                    <b className={cx("text-[15px]", !isRef && baseElapsed !== null && good(elapsed <= baseElapsed))}>{formatMs(elapsed, lang)}</b>
                    {!isRef && baseElapsed !== null && (
                      <small className="block text-[12px] text-ink-3">
                        {elapsed <= baseElapsed ? t.fasterBy(formatMs(baseElapsed - elapsed, lang)) : t.slowerBy(formatMs(elapsed - baseElapsed, lang))}
                      </small>
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

function ChainOfficial() {
  return (
    <div className="grid gap-4">
      <ChainBars rows={M3_06_CHAIN.map((r, i) => ({ ...r, id: r.name, color: i === 0 ? "var(--lane-ref)" : "var(--lane-2)" }))} />
    </div>
  );
}

function ChainDetail({ row, onClose }: { row: ChainRow; onClose: () => void }) {
  const { lang, t } = useI18n();
  const [tab, setTab] = useState<"json" | "sql" | "report">("json");
  return (
    <Drawer title={t.detailTitle(row.name)} onClose={onClose}>
      <div className="grid gap-3.5">
        <Segmented label={t.detailTabAria} value={tab} onChange={setTab} options={[["json", t.detailJson], ["sql", t.detailSql], ["report", t.detailReport]]} />
        {tab === "json" && (row.invoice ? <CodeBlock label={t.invoiceJsonLabel}>{JSON.stringify(row.invoice, null, 2)}</CodeBlock> : <Notice tone="bad">{row.node1.error ?? t.node1Failed}</Notice>)}
        {tab === "sql" && row.questions.map((q) => (
          <div key={q.question_id} className="grid gap-1.5">
            <div className="flex flex-wrap items-center gap-2 text-[13.5px]">
              <b>{q.question_id}</b><span className="text-ink-2">{q.question}</span>
              <Pill tone={q.status === "ok" ? "good" : q.status === "no_data" ? "idle" : "bad"}>
                {q.status === "ok" ? t.sqlRowsCount(formatNumber(q.row_count, lang)) : q.status === "no_data" ? t.statusNoData : t.statusBlocked}
              </Pill>
            </div>
            {q.sql && <SqlBlock sql={q.sql} />}
          </div>
        ))}
        {tab === "report" && <div className="max-w-[68ch] whitespace-pre-wrap leading-relaxed">{row.report.text || row.report.error || t.noReportText}</div>}
      </div>
    </Drawer>
  );
}

export default function ResultPage({ api, endpoints, pollMs, go }: { api: Api; endpoints: Endpoint[]; pollMs: number; go: (page: PageId) => void }) {
  const { t } = useI18n();
  const [mode, setMode] = useState<Mode>("speed");
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints, 6));
  const [workload, setWorkload] = useState<Workload>("W1");
  const [preset, setPreset] = useState<PresetKey>("quick");
  const [sample, setSample] = useState(SAMPLES[0].id);
  const [runs, setRuns] = useState<BenchRun[]>([]);
  const [current, setCurrent] = useState<BenchRun | null>(null);
  const [shownId, setShownId] = useState<string | null>(null);
  const [official, setOfficial] = useState(false);
  const [detail, setDetail] = useState<ChainRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);
  const colors = endpointColors(endpoints);
  const running = current !== null && !FINAL.includes(current.status);
  const now = useNow(running);

  const poll = useCallback(
    async (runId: string) => {
      for (;;) {
        const run = await api.bench(runId);
        if (!alive.current) return;
        setCurrent(run);
        if (FINAL.includes(run.status)) {
          setShownId(run.run_id);
          setOfficial(false);
          const { runs: list } = await api.benches();
          if (alive.current) setRuns(list);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
    },
    [api, pollMs],
  );

  useEffect(() => {
    alive.current = true;
    api.benches().then(
      ({ runs: list }) => {
        if (!alive.current) return;
        setRuns(list);
        const active = list.find((r) => !FINAL.includes(r.status));
        if (active) poll(active.run_id).catch((exc) => setError((exc as Error).message));
      },
      (exc) => alive.current && setError((exc as Error).message),
    );
    return () => {
      alive.current = false;
    };
  }, [api, poll]);

  if (endpoints.length === 0) return <><PageHead title={t.pageResult} /><NeedEndpoint what={t.pageResult} onGo={() => go("endpoints")} /></>;

  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const ids = selected.filter((id) => byId.get(id)?.status !== "down").slice(0, 6);
  const models = new Set(endpoints.filter((e) => ids.includes(e.endpoint_id)).map(baseModel));
  const p = PRESETS[preset];

  async function start() {
    setError(null);
    const config: BenchConfig = mode === "speed"
      ? { kind: "speed", endpoint_ids: ids, workload, loads: p.loads, requests: p.requests, repeats: p.repeats }
      : { kind: "chain", endpoint_ids: ids, sample_id: sample };
    try {
      const { run_id } = await api.startBench(config);
      setCurrent({ run_id, kind: config.kind, status: "queued", config, progress: {}, result: {}, error: null, created_at: new Date().toISOString() });
      await poll(run_id);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const modeRuns = runs.filter((r) => r.kind === mode);
  const shown = modeRuns.find((r) => r.run_id === shownId) ?? (current?.kind === mode && current.run_id === shownId ? current : null) ?? modeRuns[0] ?? null;
  const showOfficial = mode === "speed" ? (official || !shown) : !shown;

  let body: ReactNode = null;
  if (running && current?.kind === mode) {
    const c = current.config;
    const done = current.progress.done ?? 0;
    const total = current.progress.total ?? 0;
    const pct = total ? Math.round((100 * done) / total) : 0;
    body = (
      <div className="grid gap-2.5">
        <p className="flex items-center gap-2 text-[14px] text-ink-2">
          <Spinner />
          {c.kind === "speed" ? t.workloadNames[(c.workload ?? "W1") as Workload] : t.modeChain}
        </p>
        <div role="progressbar" aria-label={t.progressAria} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-2.5 overflow-hidden rounded-full bg-sunken">
          <div className="h-full rounded-full bg-ink transition-[width]" style={{ width: `${Math.max(pct, 3)}%` }} />
        </div>
        <p className="num flex flex-wrap gap-x-4 text-[13.5px] text-ink-2">
          <span>{total ? <><b className="text-ink">{done}/{total}</b> {c.kind === "speed" ? t.requestsUnit : t.endpointsUnit}</> : t.preparing}</span>
          <span>{t.elapsedLabel}: <b className="text-ink">{secondsSince(Date.parse(current.created_at), now)} s</b></span>
        </p>
      </div>
    );
  } else if (showOfficial) {
    body = mode === "speed" ? <SpeedOfficial workload={workload} /> : <ChainOfficial />;
  } else if (shown) {
    if (shown.status === "failed") body = <Notice tone="bad">{shown.error}</Notice>;
    else if (shown.status === "interrupted") body = <Notice tone="warn">{t.benchInterrupted}</Notice>;
    else if ("kind" in shown.result && shown.result.kind === "speed") {
      body = <SpeedLive key={shown.run_id} result={shown.result} colors={colors} />;
    } else if ("kind" in shown.result && shown.result.kind === "chain") {
      const chain = shown.result;
      body = <ChainLive key={shown.run_id} result={chain} colors={colors} onDetail={setDetail} />;
    }
  }

  const speedViews: [string, string][] = [
    ...(shown ? [["live", t.liveRunTab] as [string, string]] : []),
    ["official", t.officialM305],
  ];

  return (
    <>
      <PageHead
        title={t.pageResult}
        actions={<Segmented label={t.modeAria} value={mode} onChange={(v) => { setMode(v); setOfficial(false); setShownId(null); }} options={[["speed", t.modeSpeed], ["chain", t.modeChain]]} />}
      />

      <section aria-label={t.configAria} className="grid gap-2.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void start();
          }}
          className="grid gap-3 rounded-xl border border-line bg-surface px-4 py-3"
        >
          {mode === "speed" ? (
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-[13px] text-ink-3">{t.workloadLabel}</span>
                <Segmented label={t.workloadLabel} value={workload} onChange={setWorkload} options={(Object.keys(t.workloadNames) as Workload[]).map((w) => [w, t.workloadNames[w]])} />
              </span>
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-[13px] text-ink-3">{t.presetLabel}</span>
                <Segmented label={t.presetAria} value={preset} onChange={setPreset} options={[["quick", t.presetQuick], ["full", t.presetFull]]} />
              </span>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13px] text-ink-3">{t.sampleImageLabel}</span>
              <div role="radiogroup" aria-label={t.sampleImageLabel} className="flex flex-wrap gap-1.5">
                {SAMPLES.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    role="radio"
                    aria-checked={sample === s.id}
                    onClick={() => setSample(s.id)}
                    className={cx("inline-flex items-center gap-2 rounded-lg border bg-surface py-0.5 pl-0.5 pr-2.5 text-[13px]", sample === s.id ? "border-ink font-semibold text-ink" : "border-line text-ink-2")}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={s.file} alt="" className="size-8 rounded-md object-cover object-top" />
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-2">
            <span className="text-[13px] text-ink-3">{t.selectLlm}</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-6)))}
            />
            <Button type="submit" icon={Play} loading={running} disabled={running || ids.length === 0} className="ml-auto">
              {running ? t.measuringBtn : mode === "speed" ? t.startBench : t.startChain}
            </Button>
          </div>
        </form>
        {models.size > 1 && <Notice tone="warn">{t.diffModelsWarn(models.size)}</Notice>}
        {error && <Notice tone="bad">{error}</Notice>}
      </section>

      <section aria-label={t.resultsHeading} className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <h2 className="text-[15px] font-bold">{running && current?.kind === mode ? t.runningHeading : t.resultsHeading}</h2>
          {mode === "speed" && !(running && current?.kind === mode) && (
            <span className="ml-auto">
              <Segmented label={t.viewAria} value={showOfficial ? "official" : "live"} onChange={(v) => setOfficial(v === "official")} options={speedViews} />
            </span>
          )}
        </div>
        {body}
      </section>

      <section aria-label={t.historyTitle} className="overflow-hidden rounded-xl border border-line bg-surface">
        <h2 className="border-b border-line px-[18px] py-3.5 text-[15px] font-bold">{t.historyTitle}</h2>
        {modeRuns.length === 0 ? (
          <p className="px-[18px] py-3.5 text-[13px] text-ink-3">{t.historyEmpty}</p>
        ) : (
          <ul className="divide-y divide-line">
            {modeRuns.map((r) => (
              <li key={r.run_id}>
                <button
                  type="button"
                  onClick={() => { setShownId(r.run_id); setOfficial(false); }}
                  aria-current={!showOfficial && shown?.run_id === r.run_id}
                  className={cx("grid w-full grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-0.5 px-[18px] py-2.5 text-left hover:bg-sunken sm:grid-cols-[auto_minmax(0,1fr)_auto]", !showOfficial && shown?.run_id === r.run_id && "bg-sunken")}
                >
                  <Pill tone={r.status === "done" ? "good" : r.status === "failed" ? "bad" : r.status === "interrupted" ? "warn" : "idle"}>
                    {t.runStatusLabels[r.status] ?? r.status}
                  </Pill>
                  <span className="min-w-0">
                    <b className="block font-semibold">
                      {r.kind === "speed"
                        ? `${t.workloadNames[(r.config.workload ?? "W1") as Workload]} · C=${(r.config.loads ?? []).join(", ")}`
                        : `${t.modeChain} · ${SAMPLES.find((s) => s.id === r.config.sample_id)?.label ?? r.config.sample_id ?? "m0_04_001"}`}
                    </b>
                    <small className="text-[12.5px] text-ink-3">
                      {r.config.endpoint_ids.length} {t.endpointsUnit}
                      {r.kind === "speed" ? ` · ${r.config.requests ?? 8} ${t.requestsUnit}` : ""}
                    </small>
                  </span>
                  <span className="col-start-2 text-[12.5px] text-ink-3 sm:col-start-auto">{when(r.created_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {detail && <ChainDetail row={detail} onClose={() => setDetail(null)} />}
    </>
  );
}
