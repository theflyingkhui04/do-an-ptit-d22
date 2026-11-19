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
  Spinner,
  SqlBlock,
  baseModel,
  cx,
  endpointColors,
  secondsSince,
  useNow,
} from "@/components/ui";
import type { Api, BenchCell, BenchConfig, BenchRun, ChainResult, ChainRow, Endpoint, SpeedResult } from "@/lib/api";
import { setupLine } from "@/lib/endpoints";
import { leafFields, sameValue } from "@/lib/fields";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { M3_05_SPEEDUP, M3_06_CHAIN, OFFICIAL_LOADS, SAMPLES } from "@/lib/official";

type Workload = "W1" | "W2" | "W3";
type Mode = "speed" | "chain";
// The two presets of D8, kept as they were (Dương, 18/11); the line under them says what they measure.
const PRESETS = {
  quick: { label: "Nhanh", requests: 8, loads: [1, 8], repeats: 1 },
  full: { label: "Đầy đủ", requests: 32, loads: [1, 2, 4, 8, 16, 32], repeats: 3 },
};
// Rough seconds per request at low load on one GPU. W1 from the live run of 13/11 on A10G (latency p50 11.8-16.5 s,
// the "Nhanh" run took 2 min 46 s); W2 (short SQL) and W3 (prose) from the Node 2 / Node 3 pages (about 2 s and 6-8 s).
const SECONDS_PER_REQUEST: Record<Workload, number> = { W1: 18, W2: 3, W3: 8 };
const FINAL = ["done", "failed", "interrupted"];
const WORKLOADS: Record<Workload, string> = { W1: "Node 1 (W1)", W2: "Node 2 (W2)", W3: "Node 3 (W3)" };
const WORKLOAD_ABOUT: Record<Workload, string> = {
  W1: "Node 1 trích xuất hoá đơn (W1): mỗi request đưa văn bản OCR của một hoá đơn thật trong gói đã đóng băng và nhận lại JSON hoá đơn, đầu ra dài nhất trong ba bài.",
  W2: "Node 2 sinh SQL (W2): mỗi request là một câu hỏi tiếng Việt của gói đã đóng băng, model trả một câu SQL ngắn.",
  W3: "Node 3 viết báo cáo (W3): mỗi request gửi bảng số liệu của gói đã đóng băng, model viết một báo cáo vài đoạn.",
};
const OFFICIAL_SETUP = [
  { name: "vanilla", setup: "mốc, không model nháp" },
  { name: "static", setup: "EAGLE-3, K cố định" },
  { name: "adaptive", setup: "EAGLE-3, Hardware-Adaptive" },
  { name: "eagle", setup: "EAGLE gốc" },
];
const OFFICIAL_COLORS = ["var(--lane-1)", "var(--lane-2)", "var(--lane-3)"];

// What each measured number means, and which way is better (design D17).
type MetricKey = "tpot" | "ttft" | "latency" | "throughput";
const METRICS: Record<MetricKey, { label: string; lower: boolean; read: (c: BenchCell) => number | null; format: (v: number) => string; about: string }> = {
  tpot: {
    label: "Tốc độ sinh token (TPOT)", lower: true, read: (c) => c.tpot_ms_p50, format: (v) => `${formatNumber(Math.round(v * 10) / 10)} ms`,
    about: "Thời gian sinh thêm mỗi token sau token đầu, trung vị các request. Thấp hơn là nhanh hơn. Giải mã suy đoán nhắm vào chỉ số này, nên hệ số “nhanh gấp mấy lần” tính theo TPOT.",
  },
  ttft: {
    label: "Chờ token đầu (TTFT)", lower: true, read: (c) => c.ttft_ms_p50, format: (v) => formatMs(v),
    about: "Từ lúc gửi request đến lúc nhận token đầu tiên. Gồm cả đường mạng từ máy chủ web (Singapore) tới GPU, nên nhiễu hơn TPOT.",
  },
  latency: {
    label: "Thời gian trả lời", lower: true, read: (c) => c.latency_ms_p50, format: (v) => formatMs(v),
    about: "Từ lúc gửi đến khi nhận hết câu trả lời, trung vị các request. Đây là thời gian người dùng thật sự chờ.",
  },
  throughput: {
    label: "Thông lượng", lower: false, read: (c) => c.throughput_tps, format: (v) => `${formatNumber(Math.round(v))} token/s`,
    about: "Tổng số token cả endpoint sinh ra mỗi giây khi phục vụ cùng lúc nhiều người. Cao hơn là phục vụ được nhiều hơn.",
  },
};

function estimateMinutes(workload: Workload, preset: keyof typeof PRESETS): number {
  const p = PRESETS[preset];
  const waves = p.loads.reduce((sum, load) => sum + Math.ceil(p.requests / load), 0);
  return Math.max(1, Math.round((p.repeats * waves * SECONDS_PER_REQUEST[workload]) / 60));
}

function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const two = (n: number) => String(n).padStart(2, "0");
  return `${two(d.getDate())}/${two(d.getMonth() + 1)} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

// "nhanh hơn 1,42×" against the baseline; `ratio` > 1 means better.
function compared(metric: MetricKey, ratio: number): string {
  const [better, worse] = METRICS[metric].lower ? ["nhanh hơn", "chậm hơn"] : ["cao hơn", "thấp hơn"];
  if (Math.abs(ratio - 1) < 0.005) return "bằng mốc";
  return `${ratio > 1 ? better : worse} ${formatSpeedup(ratio > 1 ? ratio : 1 / ratio)}`;
}
const ratioOf = (metric: MetricKey, base: number, mine: number) => (METRICS[metric].lower ? base / mine : mine / base);

// Names in a narrow column lose their common start ("qwen2.5-7b-vanilla" -> "vanilla") so the part that tells them apart shows.
function shortNames(names: string[]): Map<string, string> {
  let prefix = names.length > 1 ? names[0] : "";
  for (const n of names) while (!n.startsWith(prefix)) prefix = prefix.slice(0, -1);
  const cut = Math.max(prefix.lastIndexOf("-"), prefix.lastIndexOf(" "), prefix.lastIndexOf("_")) + 1;
  return new Map(names.map((n) => [n, n.slice(cut) || n]));
}

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

// "What was measured, how, on what": the box that opens every result.
function About({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl aria-label="Lần đo này đo gì" className="m-0 grid gap-1.5 rounded-xl border border-line bg-sunken px-3.5 py-3 text-[13.5px]">
      {rows.map(([term, text]) => (
        <div key={term} className="grid gap-0.5 sm:grid-cols-[110px_minmax(0,1fr)] sm:gap-2.5">
          <dt className="pt-px text-[12.5px] font-semibold text-ink-3">{term}</dt>
          <dd className="m-0 min-w-0 text-ink-2">{text}</dd>
        </div>
      ))}
    </dl>
  );
}

function EndpointList({ items }: { items: { name: string; setup: string; color?: string }[] }) {
  return (
    <span className="flex flex-wrap gap-x-3.5 gap-y-1">
      {items.map((e) => (
        <span key={e.name} className="inline-flex items-center gap-1.5">
          {e.color && <Dot color={e.color} />}<b className="text-ink">{e.name}</b> {e.setup}
        </span>
      ))}
    </span>
  );
}

// ---------------------------------------------------------------- live speed run
function SpeedLive({ result, endpoints, colors }: { result: SpeedResult; endpoints: Endpoint[]; colors: Record<string, string> }) {
  const [metric, setMetric] = useState<MetricKey>("tpot");
  const m = METRICS[metric];
  const ids = [...new Set(result.cells.map((c) => c.endpoint_id))].sort((a, b) => Number(b === result.baseline) - Number(a === result.baseline));
  const names = Object.fromEntries(result.cells.map((c) => [c.endpoint_id, c.name]));
  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const cell = (id: string, load: number) => result.cells.find((c) => c.endpoint_id === id && c.load === load);
  const value = (id: string, load: number) => {
    const c = cell(id, load);
    return c ? m.read(c) : null;
  };
  const others = ids.filter((id) => id !== result.baseline);
  const short = shortNames(ids.map((id) => names[id]));
  const total = result.cells.reduce((sum, c) => sum + c.n, 0);
  const failed = result.cells.reduce((sum, c) => sum + (c.n - c.n_ok), 0);

  const best = (load: number) => {
    const base = result.baseline ? value(result.baseline, load) : null;
    if (!base) return null;
    return others
      .map((id) => ({ id, mine: value(id, load), ratio: 0 }))
      .filter((x): x is { id: string; mine: number; ratio: number } => x.mine !== null && x.mine > 0)
      .map((x) => ({ ...x, base, ratio: ratioOf(metric, base, x.mine) }))
      .sort((a, b) => b.ratio - a.ratio)[0] ?? null;
  };
  const low = result.loads[0];
  const high = result.loads[result.loads.length - 1];
  const bestLow = best(low);
  const bestHigh = best(high);
  const conclusions: ReactNode[] = [];
  if (bestLow) {
    conclusions.push(<>Ở <b className="text-ink">{low} người dùng</b>, {names[bestLow.id]} {compared(metric, bestLow.ratio)} so với mốc ({m.format(bestLow.mine)} so với {m.format(bestLow.base)}).</>);
  }
  if (bestHigh && high !== low) {
    conclusions.push(bestHigh.ratio >= 1
      ? <>Ở <b className="text-ink">{high} người dùng</b>, {names[bestHigh.id]} {compared(metric, bestHigh.ratio)} ({m.format(bestHigh.mine)} so với {m.format(bestHigh.base)}).</>
      : <>Ở <b className="text-ink">{high} người dùng</b>, mọi endpoint đều {m.lower ? "chậm" : "thấp"} hơn mốc; đỡ nhất là {names[bestHigh.id]}, {compared(metric, bestHigh.ratio)} ({m.format(bestHigh.mine)} so với {m.format(bestHigh.base)}).</>);
  }
  if ((metric === "tpot" || metric === "latency") && bestLow && bestHigh && high !== low && bestHigh.ratio < bestLow.ratio) {
    conclusions.push(<>Hệ số giảm khi thêm người dùng: GPU phải xử lý nhiều request cùng lúc nên phần đoán trước của model nháp lời ít hơn. Xu hướng này giống số chính thức M3-05.</>);
  }
  conclusions.push(failed ? <>Có {failed}/{total} request lỗi; không tính vào trung vị.</> : <>Mọi request đều thành công ({total}/{total}).</>);

  return (
    <div className="grid gap-4">
      <About rows={[
        ["Đo gì", WORKLOAD_ABOUT[result.workload as Workload] ?? result.workload],
        ["Đo thế nào", `${result.loads.length} mức người dùng cùng lúc (${result.loads.join(", ")}), mỗi mức ${result.requests} request${result.repeats > 1 ? `, lặp ${result.repeats} lần` : ""}; lấy trung vị (p50). Tổng ${total} request.`],
        ["Trên endpoint", <EndpointList key="eps" items={ids.map((id) => {
          const e = byId.get(id);
          const setup = e ? setupLine(e) : null;
          return { name: names[id], color: colors[id], setup: [id === result.baseline ? "mốc" : null, setup ?? "chưa ghi chú cấu hình"].filter(Boolean).join(", ") };
        })} />],
      ]} />
      {!result.baseline && <Notice tone="info">Lần đo này không có endpoint mốc nên chỉ có số tuyệt đối, không so được với mốc.</Notice>}

      <div role="tablist" aria-label="Chỉ số" className="flex flex-wrap gap-1 border-b border-line">
        {(Object.keys(METRICS) as MetricKey[]).map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={k === metric}
            onClick={() => setMetric(k)}
            className={cx("-mb-px border-b-2 px-3 py-2 text-[13.5px]", k === metric ? "border-ink font-semibold text-ink" : "border-transparent text-ink-2 hover:text-ink")}
          >
            {METRICS[k].label}
          </button>
        ))}
      </div>
      <p className="max-w-[90ch] text-[13.5px] text-ink-2">{m.about}</p>

      {result.loads.length <= 3 ? (
        <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr))]">
          {result.loads.map((load) => {
            const max = Math.max(1, ...ids.map((id) => value(id, load) ?? 0));
            const base = result.baseline ? value(result.baseline, load) : null;
            return (
              <div key={load} role="group" aria-label={`${load} người dùng cùng lúc`} className="grid content-start gap-2.5 rounded-xl border border-line px-3.5 py-3">
                <h4 className="text-[13px] font-semibold text-ink-2">{load} người dùng cùng lúc</h4>
                {ids.map((id) => {
                  const v = value(id, load);
                  const ratio = v !== null && base ? ratioOf(metric, base, v) : null;
                  return (
                    <div key={id} className="grid grid-cols-[minmax(0,8.5em)_minmax(0,1fr)_auto] items-center gap-2.5 text-[13px]">
                      <span className="flex min-w-0 items-center gap-1.5 font-semibold" title={names[id]}><Dot color={colors[id] ?? "var(--ink-2)"} /><span className="truncate">{short.get(names[id])}</span></span>
                      <span className="h-3.5 overflow-hidden rounded bg-sunken">
                        <span className="block h-full rounded" style={{ width: `${v === null ? 0 : Math.max(2, (v / max) * 100)}%`, background: colors[id] ?? "var(--ink-2)" }} />
                      </span>
                      <span className="grid min-w-[7.5em] justify-items-end leading-tight">
                        <b className="num">{v === null ? "—" : m.format(v)}</b>
                        {id === result.baseline
                          ? <small className="text-[11.5px] text-ink-3">mốc</small>
                          : ratio !== null && <small className={cx("whitespace-nowrap text-[11.5px] font-semibold", ratio >= 1 ? "text-good" : "text-bad")}>{compared(metric, ratio)}</small>}
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
            label={`${m.label} theo số người dùng`}
            loads={result.loads}
            format={m.format}
            series={ids.map((id) => ({ name: names[id], color: colors[id] ?? "var(--ink-2)", dashed: id === result.baseline, values: result.loads.map((load) => value(id, load)) }))}
          />
        </div>
      )}

      <div className="grid gap-1 border-l-[3px] border-ink py-0.5 pl-3.5">
        <h3 className="text-[14px] font-bold">Kết luận</h3>
        <ul className="grid list-disc gap-1 pl-5 text-[14px] text-ink-2">
          {conclusions.map((c, i) => <li key={i}>{c}</li>)}
        </ul>
      </div>

      <details className="group rounded-lg border border-line">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-[13px] font-semibold text-ink-2 [&::-webkit-details-marker]:hidden">
          <ChevronRight aria-hidden className="size-3.5 transition-transform group-open:rotate-90" />Bảng số đo đầy đủ
        </summary>
        <div className="overflow-x-auto border-t border-line">
          <table className="w-full min-w-[720px] text-left text-[13.5px]" aria-label="Bảng số đo đầy đủ">
            <thead className="bg-sunken text-ink-3">
              <tr>
                {["Endpoint", "Người dùng", "TTFT p50", "TPOT p50", "Thời gian p50", "Thông lượng", "Thành công"].map((h, i) => (
                  <th key={h} className={cx("px-3 py-2 font-semibold", i > 0 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="num">
              {result.cells.map((c) => (
                <tr key={`${c.endpoint_id}-${c.load}`} className="border-t border-line">
                  <td className="px-3 py-1.5"><span className="inline-flex items-center gap-2"><Dot color={colors[c.endpoint_id] ?? "var(--ink-2)"} />{c.name}</span></td>
                  <td className="px-3 py-1.5 text-right">{c.load}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.ttft_ms_p50)}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.tpot_ms_p50)}</td>
                  <td className="px-3 py-1.5 text-right">{formatMs(c.latency_ms_p50)}</td>
                  <td className="px-3 py-1.5 text-right">{c.throughput_tps ? `${formatNumber(Math.round(c.throughput_tps))} token/s` : "—"}</td>
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
  const data = M3_05_SPEEDUP[workload];
  const series: Series[] = Object.entries(data).map(([name, values], i) => ({ name, color: OFFICIAL_COLORS[i], values }));
  return (
    <div className="grid gap-4">
      <About rows={[
        ["Đo gì", WORKLOAD_ABOUT[workload]],
        ["Đo thế nào", `${OFFICIAL_LOADS.length} mức người dùng cùng lúc (${OFFICIAL_LOADS.join(", ")}), mỗi mức 32 request, lặp 5 lần; lấy trung vị (p50). GPU L4, campaign M3-05.`],
        ["Trên endpoint", <EndpointList key="eps" items={OFFICIAL_SETUP.map((e, i) => ({ ...e, color: i === 0 ? "var(--lane-ref)" : OFFICIAL_COLORS[i - 1] }))} />],
      ]} />
      <p className="max-w-[90ch] text-[13.5px] text-ink-2">
        Số chính thức chỉ lưu hệ số <b className="text-ink">nhanh gấp mấy lần mốc theo TPOT</b> (TPOT p50 của vanilla chia cho TPOT p50 của
        cách giải mã, cùng số người dùng). Trên 1× là nhanh hơn mốc.
      </p>
      <div className="overflow-x-auto rounded-xl border border-line px-3.5 py-3">
        <LineChart label="Nhanh gấp mấy lần mốc theo TPOT" loads={OFFICIAL_LOADS} series={series} format={(v) => formatSpeedup(v)} reference={1} referenceLabel="vanilla, mốc 1×" />
      </div>
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[560px] text-[13.5px]" aria-label="Nhanh gấp mấy lần mốc theo số người dùng">
          <thead className="bg-sunken text-[12px] text-ink-3">
            <tr>
              <th className="px-3.5 py-2 text-left font-medium">Cách giải mã</th>
              {OFFICIAL_LOADS.map((l) => <th key={l} className="px-3.5 py-2 text-right font-medium">{l} người dùng</th>)}
            </tr>
          </thead>
          <tbody className="num">
            <tr className="border-t border-line">
              <td className="px-3.5 py-2"><span className="inline-flex items-center gap-2 font-bold"><Dot color="var(--lane-ref)" />vanilla</span> <span className="ml-1 rounded border border-line px-1 text-[11.5px] text-ink-2">Mốc</span></td>
              {OFFICIAL_LOADS.map((l) => <td key={l} className="px-3.5 py-2 text-right font-semibold text-ink-3">1×</td>)}
            </tr>
            {series.map((s) => (
              <tr key={s.name} className="border-t border-line">
                <td className="px-3.5 py-2"><span className="inline-flex items-center gap-2 font-bold"><Dot color={s.color} />{s.name}</span></td>
                {s.values.map((v, i) => (
                  <td key={i} className={cx("px-3.5 py-2 text-right text-[15px] font-bold", v !== null && (v >= 1 ? "text-good" : "text-bad"))}>{v === null ? "—" : formatSpeedup(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- chain 1 -> 2 -> 3
const NODES = [
  { key: "node1", label: "Node 1, trích xuất", bg: "var(--node-1)", ink: "var(--node-1-ink)" },
  { key: "node2", label: "Node 2, 6 câu Q1–Q6", bg: "var(--node-2)", ink: "var(--node-2-ink)" },
  { key: "node3", label: "Node 3, báo cáo", bg: "var(--node-3)", ink: "var(--node-3-ink)" },
] as const;

function ChainBars({ rows }: { rows: { id: string; name: string; color: string; node1: number; node2: number; node3: number; isBaseline?: boolean; onDetail?: () => void }[] }) {
  const max = Math.max(1, ...rows.map((r) => r.node1 + r.node2 + r.node3));
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        {NODES.map((n) => <span key={n.key} className="inline-flex items-center gap-1.5"><i aria-hidden className="inline-block size-3 rounded-sm" style={{ background: n.bg }} />{n.label}</span>)}
      </div>
      {rows.map((r) => {
        const total = r.node1 + r.node2 + r.node3;
        return (
          <div key={r.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3.5 gap-y-1.5 md:grid-cols-[minmax(150px,220px)_minmax(0,1fr)_auto]">
            <div className="grid min-w-0 justify-items-start gap-1">
              <span className="inline-flex max-w-full items-center gap-2 font-semibold"><Dot color={r.color} /><span className="truncate">{r.name}</span></span>
              {r.isBaseline && <span className="rounded border border-line px-1 text-[11.5px] text-ink-2">Mốc</span>}
            </div>
            <div className="col-span-2 row-start-2 flex h-7 overflow-hidden rounded-md bg-sunken md:col-span-1 md:row-start-auto" role="img" aria-label={`${r.name}: Node 1 ${formatMs(r.node1)}, Node 2 ${formatMs(r.node2)}, Node 3 ${formatMs(r.node3)}`}>
              <div className="flex" style={{ width: `${(100 * total) / max}%` }}>
                {NODES.map((n, i) => (
                  <div
                    key={n.key}
                    title={`${n.label}: ${formatMs(r[n.key])}`}
                    className={cx("flex items-center overflow-hidden whitespace-nowrap px-2 text-[12px] font-semibold", i > 0 && "border-l-2 border-surface")}
                    style={{ flex: Math.max(r[n.key], 1), background: n.bg, color: n.ink }}
                  >
                    {r[n.key] / max > 0.1 ? formatMs(r[n.key]) : ""}
                  </div>
                ))}
              </div>
            </div>
            <div className="grid justify-items-end gap-0.5 text-right">
              <b className="num">{formatMs(total)}</b>
              {r.onDetail && (
                <button type="button" onClick={r.onDetail} aria-label={`Xem chi tiết ${r.name}`} className="text-[12.5px] font-semibold text-lane-1">Xem chi tiết</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Node 1 fields equal to the baseline's, Node 2 questions with the same rows, Node 3 numbers found in the tables.
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
  const numbers = row.number_check;
  return { fields, questions, numbers };
}

function ChainLive({ result, colors, onDetail }: { result: ChainResult; colors: Record<string, string>; onDetail: (row: ChainRow) => void }) {
  const rows = [...result.endpoints].sort((a, b) => Number(b.endpoint_id === result.baseline) - Number(a.endpoint_id === result.baseline));
  const base = rows.find((r) => r.endpoint_id === result.baseline);
  const totalOf = (r: ChainRow) => r.timings_ms.node1 + r.timings_ms.node2 + r.timings_ms.node3;
  const sample = SAMPLES.find((s) => s.id === result.sample_id)?.label ?? result.sample_id;
  const good = (ok: boolean) => (ok ? "text-good" : "text-bad");
  return (
    <div className="grid gap-4">
      <About rows={[
        ["Đo gì", `Cả pipeline trên ảnh ${sample}: OCR ảnh một lần${result.ocr_ms ? ` (${formatMs(result.ocr_ms)})` : ""}, rồi mỗi endpoint chạy Node 1 (đọc hoá đơn ra JSON) → Node 2 (6 câu hỏi Q1–Q6 thành SQL) → Node 3 (một báo cáo).`],
        ["So sánh", "Thời gian từng node và kết quả của từng node có giống endpoint mốc không."],
      ]} />
      <ChainBars rows={rows.map((r) => ({
        id: r.endpoint_id, name: r.name, color: colors[r.endpoint_id] ?? "var(--ink-2)", ...r.timings_ms,
        isBaseline: r.endpoint_id === result.baseline, onDetail: () => onDetail(r),
      }))} />
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[640px] text-left text-[13.5px]" aria-label="Kết quả từng node so với mốc">
          <thead className="bg-sunken text-[12px] text-ink-3">
            <tr>
              {["Endpoint", "Node 1: trường JSON giống mốc", "Node 2: câu có cùng kết quả", "Node 3: số khớp bảng", "Tổng thời gian"].map((h) => (
                <th key={h} className="px-3.5 py-2 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="num">
            {rows.map((r) => {
              const q = chainQuality(r, base);
              const isBase = r.endpoint_id === result.baseline;
              const t = totalOf(r);
              const bt = base ? totalOf(base) : null;
              const grounded = q.numbers.numbers - q.numbers.ungrounded.length;
              return (
                <tr key={r.endpoint_id} className="border-t border-line align-top">
                  <td className="px-3.5 py-2.5">
                    <span className="inline-flex items-center gap-2 font-bold"><Dot color={colors[r.endpoint_id] ?? "var(--ink-2)"} /><span>{r.name}</span></span>
                    {isBase && <span className="ml-1.5 rounded border border-line px-1 text-[11.5px] text-ink-2">Mốc</span>}
                  </td>
                  <td className="px-3.5 py-2.5">
                    {q.fields ? <b className={cx("text-[15px]", good(q.fields[0] === q.fields[1]))}>{q.fields[0]}/{q.fields[1]}</b>
                      : <span className="text-ink-3">{isBase ? "—" : r.invoice ? "—" : "Node 1 lỗi"}</span>}
                  </td>
                  <td className="px-3.5 py-2.5">
                    {q.questions ? (
                      <>
                        <b className={cx("text-[15px]", good(q.questions.differ.length === 0))}>{q.questions.same}/{q.questions.total}</b>
                        <small className="block text-[12px] text-ink-3">{q.questions.differ.length ? `khác ở ${q.questions.differ.join(", ")}` : "giống mốc"}</small>
                      </>
                    ) : <span className="text-ink-3">—</span>}
                  </td>
                  <td className="px-3.5 py-2.5"><b className={cx("text-[15px]", good(q.numbers.ungrounded.length === 0))}>{grounded}/{q.numbers.numbers}</b></td>
                  <td className="px-3.5 py-2.5">
                    <b className={cx("text-[15px]", !isBase && bt !== null && good(t <= bt))}>{formatMs(t)}</b>
                    {!isBase && bt !== null && <small className="block text-[12px] text-ink-3">{t <= bt ? `nhanh hơn ${formatMs(bt - t)}` : `chậm hơn ${formatMs(t - bt)}`}</small>}
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
      <About rows={[
        ["Đo gì", "Cả pipeline trên 30 hoá đơn của devset: Node 1 (trung vị mỗi hoá đơn), tổng 6 câu Q1–Q6, một báo cáo. GPU A10G, campaign M3-06."],
        ["So sánh", "vanilla (mốc) với static k=3."],
      ]} />
      <ChainBars rows={M3_06_CHAIN.map((r, i) => ({ ...r, id: r.name, color: i === 0 ? "var(--lane-ref)" : "var(--lane-2)", isBaseline: i === 0 }))} />
    </div>
  );
}

function ChainDetail({ row, onClose }: { row: ChainRow; onClose: () => void }) {
  const [tab, setTab] = useState<"json" | "sql" | "report">("json");
  return (
    <Drawer title={`Chi tiết ${row.name}`} onClose={onClose}>
      <div className="grid gap-3.5">
        <Segmented label="Phần" value={tab} onChange={setTab} options={[["json", "Node 1, JSON"], ["sql", "Node 2, SQL Q1–Q6"], ["report", "Node 3, báo cáo"]]} />
        {tab === "json" && (row.invoice ? <CodeBlock label="JSON hoá đơn">{JSON.stringify(row.invoice, null, 2)}</CodeBlock> : <Notice tone="bad">{row.node1.error ?? "Node 1 lỗi"}</Notice>)}
        {tab === "sql" && row.questions.map((q) => (
          <div key={q.question_id} className="grid gap-1.5">
            <div className="flex flex-wrap items-center gap-2 text-[13.5px]">
              <b>{q.question_id}</b><span className="text-ink-2">{q.question}</span>
              <Pill tone={q.status === "ok" ? "good" : q.status === "no_data" ? "idle" : "bad"}>{q.status === "ok" ? `${q.row_count} dòng` : q.status === "no_data" ? "không có dữ liệu" : "bị chặn"}</Pill>
            </div>
            {q.sql && <SqlBlock sql={q.sql} />}
          </div>
        ))}
        {tab === "report" && <div className="max-w-[68ch] whitespace-pre-wrap leading-relaxed">{row.report.text || row.report.error || "Không có báo cáo"}</div>}
      </div>
    </Drawer>
  );
}

// ---------------------------------------------------------------- page
export default function ResultPage({ api, endpoints, pollMs, go }: { api: Api; endpoints: Endpoint[]; pollMs: number; go: (page: PageId) => void }) {
  const baseline = endpoints.find((e) => e.is_baseline);
  const [mode, setMode] = useState<Mode>("speed");
  const [selected, setSelected] = useState<string[]>(() =>
    endpoints.filter((e) => e.status === "ok" && !e.is_baseline).slice(0, 5).map((e) => e.endpoint_id));
  const [workload, setWorkload] = useState<Workload>("W1");
  const [preset, setPreset] = useState<keyof typeof PRESETS>("quick");
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

  if (endpoints.length === 0) return <><PageHead title="Đo và so sánh" /><NeedEndpoint what="Phép đo" onGo={() => go("endpoints")} /></>;

  const baseLive = baseline && baseline.status === "ok" ? baseline : undefined;
  const ids = [...(baseLive ? [baseLive.endpoint_id] : []), ...selected.filter((id) => id !== baseline?.endpoint_id)].slice(0, 6);
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
  const showOfficial = official || !shown;

  let body: ReactNode = null;
  let badge: ReactNode = null;
  let meta = "";
  if (running && current?.kind === mode) {
    const c = current.config;
    const n = c.endpoint_ids.length;
    const plan = c.kind === "speed"
      ? `${n} endpoint × ${c.loads?.length ?? 0} mức người dùng (${(c.loads ?? []).join(", ")}) × ${c.requests ?? 0} request${(c.repeats ?? 1) > 1 ? ` × ${c.repeats} lần` : ""} = ${n * (c.loads?.length ?? 0) * (c.requests ?? 0) * (c.repeats ?? 1)} request`
      : `${n} endpoint × (Node 1 + 6 câu Q1–Q6 + 1 báo cáo)`;
    const done = current.progress.done ?? 0;
    const total = current.progress.total ?? 0;
    const pct = total ? Math.round((100 * done) / total) : 0;
    badge = <Pill tone="warn">đo trực tiếp</Pill>;
    body = (
      <div className="grid gap-2.5">
        <p className="flex items-center gap-2 text-[14px] text-ink-2"><Spinner />Đang {c.kind === "speed" ? `đo ${WORKLOADS[(c.workload ?? "W1") as Workload]}` : "chạy cả chuỗi"}. Kế hoạch: {plan}.</p>
        <div role="progressbar" aria-label="Tiến độ" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-2.5 overflow-hidden rounded-full bg-sunken">
          <div className="h-full rounded-full bg-ink transition-[width]" style={{ width: `${Math.max(pct, 3)}%` }} />
        </div>
        <p className="num flex flex-wrap gap-x-4 text-[13.5px] text-ink-2">
          <span>{total ? <><b className="text-ink">{done}/{total}</b> {c.kind === "speed" ? "request" : "endpoint"}</> : "Đang chuẩn bị…"}</span>
          <span>Đã chạy <b className="text-ink">{secondsSince(Date.parse(current.created_at), now)} s</b></span>
        </p>
      </div>
    );
  } else if (showOfficial) {
    badge = <span className="rounded-full border border-line bg-sunken px-2.5 py-0.5 text-[12px] font-semibold text-ink-2">số chính thức</span>;
    meta = mode === "speed" ? `${WORKLOADS[workload]}, GPU L4` : "GPU A10G, trung vị 30 hoá đơn";
    body = mode === "speed" ? <SpeedOfficial workload={workload} /> : <ChainOfficial />;
  } else if (shown) {
    badge = <Pill tone="warn">đo trực tiếp, có nhiễu</Pill>;
    if (shown.status === "failed") body = <Notice tone="bad">{shown.error}</Notice>;
    else if (shown.status === "interrupted") body = <Notice tone="warn">Máy chủ khởi động lại giữa lần đo; kết quả dở không được ghép. Chạy lại lần nữa.</Notice>;
    else if ("kind" in shown.result && shown.result.kind === "speed") {
      meta = `${WORKLOADS[shown.result.workload as Workload] ?? shown.result.workload}, ${shown.result.loads.join(", ")} người dùng`;
      body = <SpeedLive key={shown.run_id} result={shown.result} endpoints={endpoints} colors={colors} />;
    } else if ("kind" in shown.result && shown.result.kind === "chain") {
      meta = "cả chuỗi Node 1 → 2 → 3";
      body = <ChainLive result={shown.result} colors={colors} onDetail={setDetail} />;
    }
  }

  const views: [string, string][] = [
    ...(shown ? [["live", `Lần đo ${when(shown.created_at).split(" ")[1] ?? ""}`] as [string, string]] : []),
    ["official", mode === "speed" ? "Số chính thức M3-05" : "Số chính thức M3-06"],
  ];

  return (
    <>
      <PageHead
        title="Đo và so sánh"
        actions={<Segmented label="Chế độ" value={mode} onChange={(v) => { setMode(v); setOfficial(false); setShownId(null); }} options={[["speed", "Đo tốc độ"], ["chain", "Chạy cả chuỗi 1 → 2 → 3"]]} />}
      />

      <section aria-label="Cấu hình" className="grid gap-2.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void start();
          }}
          className="grid gap-2 rounded-xl border border-line bg-surface px-4 pb-2.5 pt-3"
        >
          {mode === "speed" ? (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <span className="flex flex-wrap items-center gap-2"><span className="text-[13px] text-ink-3">Bài đo</span>
                <Segmented label="Bài đo" value={workload} onChange={setWorkload} options={(Object.keys(WORKLOADS) as Workload[]).map((w) => [w, WORKLOADS[w]])} />
              </span>
              <span className="flex flex-wrap items-center gap-2"><span className="text-[13px] text-ink-3">Mức</span>
                <Segmented label="Mức đo" value={preset} onChange={setPreset} options={[["quick", "Nhanh"], ["full", "Đầy đủ"]]} />
              </span>
              <p className="w-full text-[12.5px] text-ink-3">
                {p.label}: {p.loads.join(", ")} người dùng cùng lúc, {p.requests} request mỗi mức{p.repeats > 1 ? `, lặp ${p.repeats} lần` : ""}; ước tính khoảng {estimateMinutes(workload, preset)} phút.
              </p>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13px] text-ink-3">Ảnh hoá đơn</span>
              <div role="radiogroup" aria-label="Ảnh hoá đơn" className="flex flex-wrap gap-1.5">
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
            <span className="text-[13px] text-ink-3">Lựa chọn LLM</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              alwaysOn={baseLive ? [baseLive.endpoint_id] : []}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-5)))}
            />
            <Button type="submit" icon={Play} loading={running} disabled={running || ids.length === 0} className="ml-auto">
              {running ? "Đang đo" : mode === "speed" ? "Bắt đầu đo" : "Chạy cả chuỗi"}
            </Button>
          </div>
        </form>
        <p className="text-[12.5px] text-ink-3">Giữ trang mở trong lúc đo, vì máy chủ miễn phí tự ngủ khi không ai hỏi tiến độ. Số đo trên web là đo trực tiếp, có nhiễu, khác số chính thức.</p>
        {models.size > 1 && <Notice tone="warn">Các endpoint đã chọn chạy {models.size} model khác nhau; so sánh tốc độ không công bằng.</Notice>}
        {error && <Notice tone="bad">{error}</Notice>}
      </section>

      <section aria-label="Kết quả" className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <h2 className="text-[15px] font-bold">{running && current?.kind === mode ? "Đang đo" : "Kết quả"}</h2>
          {badge}
          {meta && <span className="text-[12.5px] text-ink-3">{meta}</span>}
          {!(running && current?.kind === mode) && (
            <span className="ml-auto">
              <Segmented label="Xem" value={showOfficial ? "official" : "live"} onChange={(v) => setOfficial(v === "official")} options={views} />
            </span>
          )}
        </div>
        {body}
      </section>

      <section aria-label="Các lần đo trước" className="overflow-hidden rounded-xl border border-line bg-surface">
        <h2 className="border-b border-line px-[18px] py-3.5 text-[15px] font-bold">Các lần đo trước</h2>
        {modeRuns.length === 0 ? (
          <p className="px-[18px] py-3.5 text-[13px] text-ink-3">Chưa có lần đo nào ở chế độ này.</p>
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
                    {{ done: "xong", failed: "lỗi", interrupted: "bị gián đoạn", running: "đang chạy", queued: "đang chờ" }[r.status]}
                  </Pill>
                  <span className="min-w-0">
                    <b className="block font-semibold">{r.kind === "speed" ? `${WORKLOADS[(r.config.workload ?? "W1") as Workload]}, ${(r.config.loads ?? []).join(", ")} người dùng` : `Cả chuỗi, ảnh ${SAMPLES.find((s) => s.id === r.config.sample_id)?.label ?? r.config.sample_id ?? "m0_04_001"}`}</b>
                    <small className="text-[12.5px] text-ink-3">{r.config.endpoint_ids.length} endpoint{r.kind === "speed" ? `, ${r.config.requests ?? 8} request mỗi mức${(r.config.repeats ?? 1) > 1 ? `, ${r.config.repeats} lần` : ""}` : ", Node 1 → 2 → 3"}</small>
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
