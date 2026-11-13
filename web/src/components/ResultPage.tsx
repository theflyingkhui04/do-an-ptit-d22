"use client";

import { Clock, Gauge, History, Play, Workflow } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { SpeedChart, type Series } from "@/components/charts";
import type { PageId } from "@/components/Console";
import {
  Button,
  CodeBlock,
  Dot,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Panel,
  Pill,
  SqlBlock,
  Stat,
  Tabs,
  baseModel,
  cx,
  endpointColors,
} from "@/components/ui";
import type { Api, BenchConfig, BenchRun, ChainResult, ChainRow, Endpoint, SpeedResult } from "@/lib/api";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { M3_05_LABEL, M3_05_SPEEDUP, M3_06_CHAIN, M3_06_LABEL, OFFICIAL_LOADS, SAMPLES, WORKLOAD_LABELS } from "@/lib/official";

type Workload = "W1" | "W2" | "W3";
const PRESETS = {
  quick: { label: "Nhanh", requests: 8, loads: [1, 8], repeats: 1, about: "8 request đầu của gói, 1 và 8 người dùng, 1 lần" },
  full: { label: "Đầy đủ", requests: 32, loads: [1, 2, 4, 8, 16, 32], repeats: 3, about: "32 request, 1 đến 32 người dùng, 3 lần" },
};
// Rough seconds per request at low load on one GPU. W1 from the live run of 13/11 on A10G (latency p50 11.8-16.5 s,
// the "Nhanh" run took 2 min 46 s); W2 (short SQL) and W3 (prose) from the Node 2 / Node 3 pages (about 2 s and 6-8 s).
const SECONDS_PER_REQUEST: Record<Workload, number> = { W1: 18, W2: 3, W3: 8 };
const FINAL = ["done", "failed", "interrupted"];

function estimateMinutes(workload: Workload, preset: keyof typeof PRESETS): number {
  const p = PRESETS[preset];
  const waves = p.loads.reduce((sum, load) => sum + Math.ceil(p.requests / load), 0);
  return Math.max(1, Math.round((p.repeats * waves * SECONDS_PER_REQUEST[workload]) / 60));
}

function when(iso: string): string {
  try {
    return new Date(iso).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------- mode A: speed
function SpeedView({ result, colors }: { result: SpeedResult; colors: Record<string, string> }) {
  const ids = [...new Set(result.cells.map((c) => c.endpoint_id))];
  const names = Object.fromEntries(result.cells.map((c) => [c.endpoint_id, c.name]));
  const others = ids.filter((id) => id !== result.baseline);
  const cell = (id: string, load: number) => result.cells.find((c) => c.endpoint_id === id && c.load === load);
  const series: Series[] = others.map((id) => ({
    name: names[id],
    color: colors[id] ?? "var(--ink-2)",
    values: result.loads.map((load) => cell(id, load)?.speedup_vs_baseline ?? null),
  }));
  const low = result.loads[0];
  const high = result.loads[result.loads.length - 1];
  const best = (load: number) =>
    others
      .map((id) => ({ id, v: cell(id, load)?.speedup_vs_baseline ?? null }))
      .filter((x): x is { id: string; v: number } => x.v !== null)
      .sort((a, b) => b.v - a.v)[0];
  const errors = result.cells.reduce((sum, c) => sum + (c.n - c.n_ok), 0);
  const total = result.cells.reduce((sum, c) => sum + c.n, 0);
  const bestLow = best(low);
  const bestHigh = best(high);

  return (
    <div className="grid gap-5">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Stat
          label={`Tốt nhất ở ${low} người dùng`}
          value={bestLow ? formatSpeedup(bestLow.v) : "—"}
          sub={bestLow ? `${names[bestLow.id]}${bestLow.v < 1 ? ", chậm hơn mốc" : ""}` : "cần endpoint mốc"}
          tone={bestLow ? (bestLow.v >= 1 ? "good" : "bad") : undefined}
        />
        <Stat
          label={`Tốt nhất ở ${high} người dùng`}
          value={bestHigh ? formatSpeedup(bestHigh.v) : "—"}
          sub={bestHigh ? `${names[bestHigh.id]}${bestHigh.v < 1 ? ", chậm hơn mốc" : ""}` : "cần endpoint mốc"}
          tone={bestHigh ? (bestHigh.v >= 1 ? "good" : "bad") : undefined}
        />
        <Stat label="Request lỗi" value={`${errors}/${total}`} sub={errors ? "không tính vào trung vị" : "không có request lỗi"} tone={errors ? "bad" : undefined} />
      </div>

      {result.baseline && series.length > 0 && result.loads.length > 1 && (
        series.length <= 3 ? (
          <SpeedChart loads={result.loads} series={series} baselineLabel={names[result.baseline]} />
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {series.map((s) => <SpeedChart key={s.name} loads={result.loads} series={[s]} baselineLabel={names[result.baseline!]} />)}
          </div>
        )
      )}
      {!result.baseline && <Notice tone="info">Lần đo này không có endpoint mốc nên chỉ có số tuyệt đối, không có hệ số tăng tốc.</Notice>}

      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[760px] text-left text-[13.5px]" aria-label="Bảng kết quả đo">
          <thead className="bg-sunken text-ink-3">
            <tr>
              <th className="px-3 py-2 font-semibold">Endpoint</th>
              <th className="px-3 py-2 text-right font-semibold">Người dùng</th>
              <th className="px-3 py-2 text-right font-semibold">TTFT p50</th>
              <th className="px-3 py-2 text-right font-semibold">TPOT p50</th>
              <th className="px-3 py-2 text-right font-semibold">Thời gian p50</th>
              <th className="px-3 py-2 text-right font-semibold">Token/giây</th>
              <th className="px-3 py-2 text-right font-semibold">Thành công</th>
              <th className="px-3 py-2 text-right font-semibold">So với mốc</th>
            </tr>
          </thead>
          <tbody className="num">
            {result.cells.map((c) => (
              <tr key={`${c.endpoint_id}-${c.load}`} className="border-t border-line">
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-2"><Dot color={colors[c.endpoint_id] ?? "var(--ink-2)"} />{c.name}{c.endpoint_id === result.baseline && <span className="text-[12px] text-ink-3">mốc</span>}</span>
                </td>
                <td className="px-3 py-1.5 text-right">{c.load}</td>
                <td className="px-3 py-1.5 text-right">{formatMs(c.ttft_ms_p50)}</td>
                <td className="px-3 py-1.5 text-right">{formatMs(c.tpot_ms_p50)}</td>
                <td className="px-3 py-1.5 text-right">{formatMs(c.latency_ms_p50)}</td>
                <td className="px-3 py-1.5 text-right">{formatNumber(c.throughput_tps ? Math.round(c.throughput_tps) : null)}</td>
                <td className={cx("px-3 py-1.5 text-right", c.n_ok < c.n && "text-bad")}>{c.n_ok}/{c.n}</td>
                <td className={cx("px-3 py-1.5 text-right font-semibold", c.speedup_vs_baseline !== null && (c.speedup_vs_baseline >= 1 ? "text-good" : "text-bad"))}>
                  {c.endpoint_id === result.baseline ? "1×" : formatSpeedup(c.speedup_vs_baseline)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[12.5px] text-ink-3">
        Nhanh gấp mấy lần = TPOT p50 của mốc / TPOT p50 của endpoint ở cùng số người dùng. Máy chủ web ở Singapore, endpoint có thể ở nơi khác:
        độ trễ mạng cộng vào TTFT và thời gian, TPOT ít bị ảnh hưởng.
      </p>
    </div>
  );
}

function OfficialSpeed({ workload }: { workload: Workload }) {
  const data = M3_05_SPEEDUP[workload];
  const colors = ["var(--lane-1)", "var(--lane-2)", "var(--lane-3)"];
  const series: Series[] = Object.entries(data).map(([name, values], i) => ({ name, color: colors[i], values }));
  return (
    <div className="grid gap-3">
      <SpeedChart loads={OFFICIAL_LOADS} series={series} baselineLabel="vanilla" />
      <p className="text-[12.5px] text-ink-3">{M3_05_LABEL} Ở 32 người dùng cả ba cách giải mã suy đoán đều chậm hơn mốc.</p>
    </div>
  );
}

// ---------------------------------------------------------------- mode B: chain
const SEGMENTS = [
  { key: "node1", label: "Node 1", color: "var(--step-1)" },
  { key: "node2", label: "Node 2", color: "var(--step-2)" },
  { key: "node3", label: "Node 3", color: "var(--step-3)" },
] as const;

function ChainBars({ rows }: { rows: { name: string; color: string; node1: number; node2: number; node3: number; badge?: React.ReactNode }[] }) {
  const max = Math.max(1, ...rows.map((r) => r.node1 + r.node2 + r.node3));
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        {SEGMENTS.map((s) => <span key={s.key} className="inline-flex items-center gap-1.5"><i aria-hidden className="inline-block size-2.5 rounded-sm" style={{ background: s.color }} />{s.label}</span>)}
      </div>
      {rows.map((r) => {
        const total = r.node1 + r.node2 + r.node3;
        return (
          <div key={r.name} className="grid gap-1.5 md:grid-cols-[200px_minmax(0,1fr)_90px] md:items-center md:gap-3">
            <div className="grid min-w-0 justify-items-start gap-1">
              <span className="inline-flex max-w-full items-center gap-2 font-semibold"><Dot color={r.color} /><span className="truncate">{r.name}</span></span>
              {r.badge}
            </div>
            <div className="flex h-7 overflow-hidden rounded-md bg-sunken" role="img" aria-label={`${r.name}: Node 1 ${formatMs(r.node1)}, Node 2 ${formatMs(r.node2)}, Node 3 ${formatMs(r.node3)}`}>
              <div className="flex" style={{ width: `${(100 * total) / max}%` }}>
                {SEGMENTS.map((s, i) => (
                  <div
                    key={s.key}
                    title={`${s.label}: ${formatMs(r[s.key])}`}
                    className={cx("flex items-center overflow-hidden whitespace-nowrap px-2 text-[12px] font-semibold text-white", i > 0 && "border-l-2 border-surface")}
                    style={{ flex: Math.max(r[s.key], 1), background: s.color }}
                  >
                    {r[s.key] / max > 0.12 ? formatMs(r[s.key]) : ""}
                  </div>
                ))}
              </div>
            </div>
            <b className="num md:text-right">{formatMs(total)}</b>
          </div>
        );
      })}
    </div>
  );
}

function ChainDetail({ row }: { row: ChainRow }) {
  const [tab, setTab] = useState<"json" | "sql" | "report">("json");
  return (
    <div className="grid gap-3">
      <Tabs label={`Chi tiết ${row.name}`} value={tab} onChange={setTab} tabs={[["json", "Node 1, JSON"], ["sql", "Node 2, SQL Q1–Q6"], ["report", "Node 3, báo cáo"]]} />
      {tab === "json" && (row.invoice ? <CodeBlock label="JSON hoá đơn">{JSON.stringify(row.invoice, null, 2)}</CodeBlock> : <Notice tone="bad">{row.node1.error ?? "Node 1 lỗi"}</Notice>)}
      {tab === "sql" && (
        <div className="grid gap-3">
          {row.questions.map((q) => (
            <div key={q.question_id} className="grid gap-1.5">
              <div className="flex flex-wrap items-center gap-2 text-[13.5px]">
                <b>{q.question_id}</b><span className="text-ink-2">{q.question}</span>
                <Pill tone={q.status === "ok" ? "good" : q.status === "no_data" ? "idle" : "bad"}>{q.status === "ok" ? `${q.row_count} dòng` : q.status === "no_data" ? "không có dữ liệu" : "bị chặn"}</Pill>
              </div>
              {q.sql && <SqlBlock sql={q.sql} />}
            </div>
          ))}
        </div>
      )}
      {tab === "report" && <div className="max-w-[75ch] whitespace-pre-wrap leading-relaxed">{row.report.text || row.report.error || "Không có báo cáo"}</div>}
    </div>
  );
}

function ChainView({ result, colors }: { result: ChainResult; colors: Record<string, string> }) {
  const [open, setOpen] = useState<string | null>(null);
  const same = (row: ChainRow) => {
    if (!row.same_as_baseline) return row.endpoint_id === result.baseline ? <span className="text-[12px] font-normal text-ink-3">mốc</span> : null;
    const s = row.same_as_baseline;
    const differ = (["node1", "node2", "node3"] as const).filter((k) => !s[k]).map((k) => `Node ${k.slice(-1)}`);
    return (
      <Pill tone={differ.length === 0 ? "good" : "warn"}>
        {`giống mốc ${3 - differ.length}/3 node${differ.length ? `, khác ở ${differ.join(", ")}` : ""}`}
      </Pill>
    );
  };
  return (
    <div className="grid gap-4">
      <p className="text-[13.5px] text-ink-2">
        Ảnh mẫu <b className="text-ink">{result.sample_id}</b>, OCR một lần {formatMs(result.ocr_ms)} rồi mỗi endpoint chạy Node 1, sáu câu Q1–Q6 và một báo cáo.
      </p>
      <ChainBars rows={result.endpoints.map((r) => ({ name: r.name, color: colors[r.endpoint_id] ?? "var(--ink-2)", ...r.timings_ms, badge: same(r) }))} />
      <div className="flex flex-wrap gap-2">
        {result.endpoints.map((r) => (
          <Button key={r.endpoint_id} variant={open === r.endpoint_id ? "primary" : "ghost"} onClick={() => setOpen(open === r.endpoint_id ? null : r.endpoint_id)}>
            Xem {r.name}
          </Button>
        ))}
      </div>
      {open && result.endpoints.find((r) => r.endpoint_id === open) && <ChainDetail row={result.endpoints.find((r) => r.endpoint_id === open)!} />}
    </div>
  );
}

// ---------------------------------------------------------------- page
export default function ResultPage({ api, endpoints, pollMs, go }: { api: Api; endpoints: Endpoint[]; pollMs: number; go: (page: PageId) => void }) {
  const [mode, setMode] = useState<"speed" | "chain">("speed");
  const live = endpoints.filter((e) => e.status === "ok");
  const [selected, setSelected] = useState<string[]>(() =>
    [...live].sort((a, b) => Number(b.is_baseline) - Number(a.is_baseline)).slice(0, 4).map((e) => e.endpoint_id),
  );
  const [workload, setWorkload] = useState<Workload>("W1");
  const [preset, setPreset] = useState<keyof typeof PRESETS>("quick");
  const [sample, setSample] = useState(SAMPLES[0].id);
  const [runs, setRuns] = useState<BenchRun[]>([]);
  const [current, setCurrent] = useState<BenchRun | null>(null);
  const [view, setView] = useState<string | null>(null); // run id, or "official"
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);
  const colors = endpointColors(endpoints);

  const poll = useCallback(
    async (runId: string) => {
      for (;;) {
        const run = await api.bench(runId);
        if (!alive.current) return;
        setCurrent(run);
        if (FINAL.includes(run.status)) {
          setView(run.run_id);
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

  const running = current !== null && !FINAL.includes(current.status);
  const baseline = endpoints.find((e) => e.is_baseline);
  const chosen = endpoints.filter((e) => selected.includes(e.endpoint_id));
  const models = new Set(chosen.map(baseModel));

  async function start() {
    setError(null);
    const config: BenchConfig =
      mode === "speed"
        ? { kind: "speed", endpoint_ids: selected, workload, loads: PRESETS[preset].loads, requests: PRESETS[preset].requests, repeats: PRESETS[preset].repeats }
        : { kind: "chain", endpoint_ids: selected, sample_id: sample };
    try {
      const { run_id } = await api.startBench(config);
      setView(null);
      setCurrent({ run_id, kind: config.kind, status: "queued", config, progress: {}, result: {}, error: null, created_at: new Date().toISOString() });
      await poll(run_id);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const shown = view === "official" ? null : runs.find((r) => r.run_id === view) ?? (current?.run_id === view ? current : null);

  return (
    <>
      <PageHead title="Đo và so sánh">
        Đo cùng một bài trên nhiều endpoint cùng lúc, hoặc chạy cả chuỗi Node 1 → 2 → 3 trên từng endpoint để thấy kết quả giống nhau và thời gian khác nhau.
      </PageHead>

      <Tabs label="Chế độ" value={mode} onChange={setMode} tabs={[["speed", "Đo tốc độ nhiều endpoint"], ["chain", "Chạy cả chuỗi 1 → 2 → 3"]]} />

      <Panel title={mode === "speed" ? "Cấu hình lần đo" : "Cấu hình lần chạy"} icon={mode === "speed" ? Gauge : Workflow}>
        <div className="grid gap-2">
          <span className="text-[13.5px] font-semibold">Endpoint tham gia</span>
          <EndpointChips
            endpoints={endpoints}
            selected={selected}
            onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-6)))}
          />
          <small className="text-[12.5px] text-ink-3">
            {baseline ? (selected.includes(baseline.endpoint_id) ? `Mốc là ${baseline.name}.` : `Thêm mốc ${baseline.name} để có hệ số tăng tốc.`) : "Chưa có endpoint mốc; chọn mốc ở trang Endpoint để có hệ số tăng tốc."} Tối đa 6 endpoint.
          </small>
        </div>

        {mode === "speed" ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <fieldset className="grid gap-2">
              <legend className="mb-2 text-[13.5px] font-semibold">Bài đo (request của gói đã đóng băng)</legend>
              {(["W1", "W2", "W3"] as Workload[]).map((w) => (
                <label key={w} className={cx("flex cursor-pointer items-center gap-2.5 rounded-lg border px-3 py-2", workload === w ? "border-ink" : "border-line")}>
                  <input type="radio" name="workload" checked={workload === w} onChange={() => setWorkload(w)} />
                  {WORKLOAD_LABELS[w]}
                </label>
              ))}
            </fieldset>
            <fieldset className="grid gap-2">
              <legend className="mb-2 text-[13.5px] font-semibold">Mức đo</legend>
              {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((p) => (
                <label key={p} className={cx("flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2", preset === p ? "border-ink" : "border-line")}>
                  <input type="radio" name="preset" className="mt-1" checked={preset === p} onChange={() => setPreset(p)} />
                  <span className="grid">
                    <b className="font-semibold">{PRESETS[p].label}</b>
                    <span className="text-[13px] text-ink-3">{PRESETS[p].about}, ước tính thô khoảng {estimateMinutes(workload, p)} phút</span>
                  </span>
                </label>
              ))}
            </fieldset>
          </div>
        ) : (
          <fieldset className="grid gap-2">
            <legend className="mb-2 text-[13.5px] font-semibold">Ảnh hoá đơn cho Node 1</legend>
            <div className="flex flex-wrap gap-2">
              {SAMPLES.map((s) => (
                <label key={s.id} className={cx("flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2", sample === s.id ? "border-ink" : "border-line")}>
                  <input type="radio" name="sample" checked={sample === s.id} onChange={() => setSample(s.id)} />
                  {s.label}
                </label>
              ))}
            </div>
          </fieldset>
        )}

        {models.size > 1 && <Notice tone="warn">Các endpoint đã chọn chạy {models.size} model khác nhau; so sánh tốc độ không công bằng.</Notice>}
        <Notice tone="info" icon={Clock}>Giữ trang này mở trong lúc đo: máy chủ miễn phí tự ngủ khi không có ai hỏi tiến độ. Số đo trên web là đo trực tiếp, có nhiễu, khác số chính thức của campaign.</Notice>

        <div className="flex flex-wrap items-center gap-3">
          <Button icon={Play} disabled={running || selected.length === 0} onClick={start}>
            {running ? "Đang đo…" : mode === "speed" ? "Bắt đầu đo" : "Chạy cả chuỗi"}
          </Button>
          {current && running && (
            <div className="flex min-w-[240px] flex-1 items-center gap-3" aria-label="Tiến độ">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-sunken">
                <div className="h-full bg-ink" style={{ width: `${current.progress.total ? (100 * (current.progress.done ?? 0)) / current.progress.total : 3}%` }} />
              </div>
              <span className="num text-[13px] text-ink-2">{current.progress.total ? `${current.progress.done ?? 0}/${current.progress.total} request` : "Đang chuẩn bị…"}</span>
            </div>
          )}
        </div>
        {error && <Notice tone="bad">{error}</Notice>}
      </Panel>

      {(shown || view === "official") && (
        <Panel
          title={view === "official" ? (mode === "speed" ? `Số chính thức M3-05, ${WORKLOAD_LABELS[workload]}` : "Số chính thức M3-06") : `Kết quả lần ${shown?.run_id}`}
          meta={view === "official" ? <Pill tone="idle">số chính thức</Pill> : <Pill tone="warn">đo trực tiếp, có nhiễu</Pill>}
        >
          {view === "official" && mode === "speed" && <OfficialSpeed workload={workload} />}
          {view === "official" && mode === "chain" && (
            <div className="grid gap-3">
              <ChainBars rows={M3_06_CHAIN.map((r, i) => ({ ...r, color: i === 0 ? "var(--lane-ref)" : "var(--lane-1)" }))} />
              <p className="text-[12.5px] text-ink-3">{M3_06_LABEL}</p>
            </div>
          )}
          {shown?.status === "failed" && <Notice tone="bad">{shown.error}</Notice>}
          {shown?.status === "interrupted" && <Notice tone="warn">Máy chủ khởi động lại giữa lần đo; kết quả dở không được ghép. Chạy lại lần nữa.</Notice>}
          {shown?.status === "done" && shown.result && "kind" in shown.result && shown.result.kind === "speed" && <SpeedView result={shown.result} colors={colors} />}
          {shown?.status === "done" && shown.result && "kind" in shown.result && shown.result.kind === "chain" && <ChainView result={shown.result} colors={colors} />}
        </Panel>
      )}

      <Panel title="Lịch sử" icon={History} bodyClassName="p-0 gap-0">
        <ul className="divide-y divide-line">
          <li>
            <button type="button" onClick={() => setView("official")} className={cx("flex w-full flex-wrap items-center gap-3 px-[18px] py-2.5 text-left hover:bg-sunken", view === "official" && "bg-sunken")}>
              <Pill tone="idle">số chính thức</Pill>
              <span className="font-semibold">{mode === "speed" ? "M3-05, GPU L4, vanilla làm mốc" : "M3-06, GPU A10G, vanilla và static"}</span>
              <span className="text-[13px] text-ink-3">{mode === "speed" ? "5 lần × 32 request mỗi ô" : "trung vị 30 hoá đơn"}</span>
            </button>
          </li>
          {runs.filter((r) => r.kind === mode).map((r) => (
            <li key={r.run_id}>
              <button type="button" onClick={() => setView(r.run_id)} className={cx("flex w-full flex-wrap items-center gap-3 px-[18px] py-2.5 text-left hover:bg-sunken", view === r.run_id && "bg-sunken")}>
                <Pill tone={r.status === "done" ? "good" : r.status === "failed" ? "bad" : r.status === "interrupted" ? "warn" : "idle"}>
                  {{ done: "xong", failed: "lỗi", interrupted: "bị gián đoạn", running: "đang chạy", queued: "đang chờ" }[r.status]}
                </Pill>
                <span className="font-semibold">{r.kind === "speed" ? `${r.config.workload}, ${r.config.loads?.join("/")} người dùng` : `chuỗi, ảnh ${r.config.sample_id ?? "m0_04_001"}`}</span>
                <span className="text-[13px] text-ink-3">{r.config.endpoint_ids.length} endpoint, {when(r.created_at)}</span>
              </button>
            </li>
          ))}
        </ul>
      </Panel>
    </>
  );
}
