"use client";

import { CheckCircle2, Circle, Database, FileImage, ImageUp, Play, ScanText, ScrollText, XCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import InvoiceDetail from "@/components/Invoice";
import JobLogDrawer, { type LogPart } from "@/components/JobLogDrawer";
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
  Spinner,
  Tabs,
  cx,
  defaultTarget,
  endpointColors,
  useNow,
} from "@/components/ui";
import {
  ApiError,
  FINAL_JOB_STATUSES,
  type Api,
  type Endpoint,
  type InvoiceDetail as Detail,
  type InvoiceRow,
  type Job,
  type JobEvent,
  type RunReport,
} from "@/lib/api";
import { KEY_FIELDS, describeField, fieldText, isEmptyValue, leafFields, sameValue, type FieldValue } from "@/lib/fields";
import { formatMoney, formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { durationVi, endpointState, ocrState, silentLine, systemState, type PartState } from "@/lib/liveness";
import { SAMPLES } from "@/lib/official";

const MAX_BYTES = 10 * 1024 * 1024;
const TONE_TEXT: Record<PartState["tone"], string> = { good: "text-good", warn: "text-warn", bad: "text-bad", idle: "text-ink-3" };

function uploadError(exc: unknown): string {
  if (exc instanceof ApiError) {
    if (exc.status === 413) return "Ảnh lớn hơn 10 MB.";
    if (exc.status === 415) return "Tệp không phải ảnh JPEG hoặc PNG.";
    if (exc.status === 429) return "Đã chạy quá nhiều lần trong ít phút, thử lại sau.";
    return exc.message;
  }
  return (exc as Error).message;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function LiveState({ state, small }: { state: PartState; small?: boolean }) {
  const Icon = state.tone === "bad" ? XCircle : state.tone === "good" ? CheckCircle2 : Circle;
  return (
    <span className={cx("inline-flex items-center gap-1.5 font-medium", small ? "text-[12.5px]" : "text-[13px]", TONE_TEXT[state.tone])}>
      {state.busy && state.tone !== "bad" ? <Spinner className="size-3.5" /> : <Icon aria-hidden className="size-3.5" />}
      {state.text}
    </span>
  );
}

// Column head of one endpoint: what it is doing now, then its own speed once it answered (design D13).
function RunHead({ id, name, color, isBaseline, report, state, onLog }: {
  id: string;
  name: string;
  color: string;
  isBaseline: boolean;
  report?: RunReport;
  state: PartState;
  onLog: () => void;
}) {
  return (
    <div className="grid min-w-[210px] gap-1.5 text-left font-normal" data-testid={`run-${id}`}>
      <div className="flex items-center gap-2">
        <Dot color={color} />
        <b className="truncate text-[15px] font-bold">{name}</b>
        {isBaseline && <span className="rounded-md border border-line px-1.5 text-[11.5px] font-semibold text-ink-3">mốc</span>}
        <button type="button" onClick={onLog} aria-label={`Xem log ${name}`} title={`Xem log ${name}`} className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken">
          <ScrollText aria-hidden className="size-4" />
        </button>
      </div>
      <LiveState state={state} />
      {report?.valid && (
        <div className="num flex flex-wrap gap-x-3.5 gap-y-0.5 text-[12.5px] text-ink-2">
          <span>TTFT <b className="text-ink">{formatMs(report.ttft_ms)}</b></span>
          <span>TPOT <b className="text-ink">{formatMs(report.tpot_ms)}</b></span>
          <span><b className="text-ink">{formatNumber(report.completion_tokens)}</b> token</span>
        </div>
      )}
      {report && !report.valid && <p className="line-clamp-3 text-[12.5px] text-bad" title={report.error ?? undefined}>{report.error}</p>}
      {(report?.source_restored ?? []).length > 0 && (
        <p className="text-[12px] text-ink-3" title="Model chép sai giá trị nguồn mà hệ thống đã gửi; hệ thống điền lại giá trị đúng.">
          Hệ thống điền lại nguồn: {report!.source_restored!.join(", ")}
        </p>
      )}
    </div>
  );
}

function diffNote(paths: string[]): string {
  if (paths.length === 0) return "Mọi trường trùng với mốc";
  const named = paths.slice(0, 4).map((path) => {
    const { group, label } = describeField(path);
    return `${group}, ${label.toLowerCase()}`;
  });
  return `Khác ở ${named.join("; ")}${paths.length > 4 ? ` và ${paths.length - 4} trường khác` : ""}`;
}

// Under the table: how each endpoint compares with the baseline (Dương, 16/11).
function Conclusion({ run, base }: { run: RunReport; base?: RunReport }) {
  const speed = run.speedup_tpot_vs_baseline;
  const same = run.differing_fields.length === 0;
  return (
    <section className="grid gap-2" data-testid={`conclusion-${run.endpoint_id}`} aria-label={`${run.name} so với mốc`}>
      <h3 className="text-[13.5px] font-semibold text-ink-2">{run.name} so với mốc{base ? ` ${base.name}` : ""}</h3>
      <div className="grid gap-x-7 gap-y-3 sm:grid-cols-2">
        <div className="grid content-start gap-0.5">
          <span className="text-[13px] font-medium text-ink-3">Giống mốc</span>
          <b className={cx("num text-[24px] leading-tight", same ? "text-good" : "text-warn")}>{run.equal_fields}/{run.total_fields} trường</b>
          <small className="text-[13px] text-ink-2">{diffNote(run.differing_fields)}</small>
        </div>
        {speed !== null && (
          <div className="grid content-start gap-0.5">
            <span className="text-[13px] font-medium text-ink-3">{speed >= 1 ? "Nhanh hơn mốc" : "Chậm hơn mốc"}</span>
            <b className={cx("num text-[24px] leading-tight", speed >= 1 ? "text-good" : "text-bad")}>{formatSpeedup(speed)}</b>
            <small className="text-[13px] text-ink-2">Tính theo TPOT: {formatMs(run.tpot_ms)} so với {formatMs(base?.tpot_ms)}</small>
          </div>
        )}
      </div>
    </section>
  );
}

function SavedInvoices({ api, refresh }: { api: Api; refresh: number }) {
  const [rows, setRows] = useState<InvoiceRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Detail | null>(null);

  useEffect(() => {
    let alive = true;
    api.invoices().then(
      ({ invoices }) => alive && setRows(invoices),
      (exc) => alive && setError((exc as Error).message),
    );
    return () => {
      alive = false;
    };
  }, [api, refresh]);

  return (
    <Panel title="Hoá đơn đã lưu" icon={Database} meta={rows && <span>{rows.length} hoá đơn, ảnh tải lên hiện trước</span>} bodyClassName="p-0 gap-0">
      {error && <div className="p-[18px]"><Notice tone="bad">{error}</Notice></div>}
      {rows && rows.length === 0 && <p className="p-[18px] text-ink-3">Chưa có hoá đơn nào.</p>}
      {rows && rows.length > 0 && (
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full min-w-[720px] text-left text-[13.5px]">
            <thead className="sticky top-0 bg-surface text-ink-3">
              <tr className="border-b border-line">
                <th className="px-[18px] py-2.5 font-semibold">Cửa hàng</th>
                <th className="px-3 py-2.5 font-semibold">Ngày lập</th>
                <th className="px-3 py-2.5 text-right font-semibold">Tổng tiền</th>
                <th className="px-3 py-2.5 text-right font-semibold">Mặt hàng</th>
                <th className="px-3 py-2.5 font-semibold">Nguồn</th>
                <th className="px-[18px] py-2.5 font-semibold">Trích bởi</th>
              </tr>
            </thead>
            <tbody className="num">
              {rows.map((row) => (
                <tr
                  key={row.invoice_id}
                  className="cursor-pointer border-b border-line last:border-0 hover:bg-sunken"
                  onClick={() => api.invoice(row.invoice_id).then(setOpen, (exc) => setError((exc as Error).message))}
                >
                  <td className="px-[18px] py-2">{row.seller_name ?? "—"}</td>
                  <td className="px-3 py-2">{row.issued_date ?? "—"}</td>
                  <td className="px-3 py-2 text-right">{formatMoney(row.total_amount_value, row.total_amount_currency)}</td>
                  <td className="px-3 py-2 text-right">{row.item_count}</td>
                  <td className="px-3 py-2">{row.source === "upload" ? <Pill tone="idle">Tải lên</Pill> : "Devset"}</td>
                  <td className="px-[18px] py-2 text-ink-3">{row.lane ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {open && (
        <div className="grid gap-3 border-t border-line p-[18px]">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-semibold">{open.payload.seller?.name ?? open.invoice_id}</h3>
            <span className="text-[13px] text-ink-3">trích bởi {open.lane ?? "—"}, model {open.model_name ?? "—"}</span>
            <Button variant="ghost" className="ml-auto" onClick={() => setOpen(null)}>Đóng</Button>
          </div>
          <div className={cx("grid gap-5", open.image_url && "lg:grid-cols-[220px_minmax(0,1fr)]")}>
            {open.image_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={open.image_url} alt="Ảnh hoá đơn" className="max-h-[360px] w-full rounded-lg border border-line object-contain" />
            )}
            <InvoiceDetail invoice={open.payload} />
          </div>
        </div>
      )}
    </Panel>
  );
}

type StepState = "wait" | "run" | "ok" | "err";

export default function Node1Page({ api, endpoints, pollMs, go }: {
  api: Api;
  endpoints: Endpoint[];
  pollMs: number;
  go: (page: PageId) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints));
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState<"key" | "all">("key");
  const [tab, setTab] = useState<"info" | "json">("info");
  const [refresh, setRefresh] = useState(0);
  const [logPart, setLogPart] = useState<string | null>(null);
  const [webLines, setWebLines] = useState<JobEvent[]>([]);
  const [lostSince, setLostSince] = useState<number | null>(null);
  const [clock, setClock] = useState({ server: 0, client: 0 });
  const clockRef = useRef(clock);
  const alive = useRef(true);
  const nowMs = useNow(busy);
  const closeLog = useCallback(() => setLogPart(null), []); // stable, so the open drawer keeps its focus

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const choose = useCallback((chosen: File | null) => {
    setJob(null);
    setError(null);
    if (chosen && !["image/jpeg", "image/png"].includes(chosen.type)) {
      setError("Chỉ nhận ảnh JPEG hoặc PNG.");
      return;
    }
    if (chosen && chosen.size > MAX_BYTES) {
      setError("Ảnh lớn hơn 10 MB.");
      return;
    }
    setFile(chosen);
    setPreview((old) => {
      if (old) URL.revokeObjectURL?.(old);
      return chosen && typeof URL.createObjectURL === "function" ? URL.createObjectURL(chosen) : null;
    });
  }, []);

  async function pickSample(sample: (typeof SAMPLES)[number]) {
    try {
      const response = await fetch(sample.file);
      const blob = await response.blob();
      choose(new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" }));
    } catch {
      setError("Không tải được ảnh mẫu.");
    }
  }

  // Lines the web itself writes (it lost or found the backend again), on the server's clock.
  function addWebLine(level: JobEvent["level"], msg: string, key?: string) {
    const { server, client } = clockRef.current;
    const t = server ? server + (Date.now() - client) / 1000 : Date.now() / 1000;
    setWebLines((lines) => [...lines.filter((line) => !key || line.key !== key), { t, src: "web", level, msg, key }]);
  }

  async function run() {
    if (!file || selected.length === 0) return;
    setBusy(true);
    setError(null);
    setJob(null);
    setWebLines([]);
    setLostSince(null);
    setTab("info");
    let jobId: string;
    try {
      ({ job_id: jobId } = await api.upload(file, selected));
    } catch (exc) {
      setError(uploadError(exc));
      setBusy(false);
      return;
    }
    let lost: number | null = null;
    let tries = 0;
    try {
      for (;;) {
        let current: Job;
        try {
          current = await api.job(jobId);
        } catch (exc) {
          // 4xx (wrong key, unknown job) ends the run; a network error or 5xx is the backend asleep or gone.
          if (exc instanceof ApiError && exc.status < 500) throw exc;
          if (!alive.current) return;
          lost ??= Date.now();
          tries += 1;
          setLostSince(lost);
          addWebLine("error", `Không gọi được backend, đã thử ${tries} lần trong ${durationVi((Date.now() - lost) / 1000)}; `
            + "job có thể vẫn đang chạy trên máy chủ", "lost");
          await sleep(pollMs * 3);
          continue;
        }
        if (!alive.current) return;
        if (lost !== null) {
          addWebLine("ok", `Đã kết nối lại backend sau ${durationVi((Date.now() - lost) / 1000)}`);
          lost = null;
          tries = 0;
          setLostSince(null);
        }
        clockRef.current = { server: current.server_time ?? Date.now() / 1000, client: Date.now() };
        setClock(clockRef.current);
        setJob(current);
        if (FINAL_JOB_STATUSES.includes(current.status)) {
          if (current.status === "done") setRefresh((n) => n + 1);
          break;
        }
        await sleep(pollMs);
      }
    } catch (exc) {
      setError(uploadError(exc));
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  if (endpoints.length === 0) return <><PageHead title="Trích xuất hoá đơn" /><NeedEndpoint what="Node 1" onGo={() => go("endpoints")} /></>;

  const colors = endpointColors(endpoints);
  const baseline = endpoints.find((e) => e.is_baseline);
  const result = job?.result;
  const runs = result?.runs ?? {};
  const baselineId = result?.baseline ?? null;
  const columns = [...(result?.endpoints ?? [])].sort((a, b) =>
    Number(b.endpoint_id === baselineId) - Number(a.endpoint_id === baselineId));
  const nameOf = (id: string) => columns.find((c) => c.endpoint_id === id)?.name ?? endpoints.find((e) => e.endpoint_id === id)?.name ?? id;

  // Live state of every part, on the server's clock.
  const serverNow = clock.server + Math.max(0, (nowMs - clock.client) / 1000);
  const events = result?.events ?? [];
  const lostFor = lostSince === null ? null : Math.max(0, (nowMs - lostSince) / 1000);
  const states: Record<string, PartState> = {};
  if (job) {
    states.system = systemState(job, lostFor);
    states.ocr = ocrState(job, events, serverNow);
    for (const c of columns) states[c.endpoint_id] = endpointState(job, events, c.endpoint_id, serverNow);
  }
  const silent = ["ocr", ...columns.map((c) => c.endpoint_id)]
    .map((src) => (states[src] ? silentLine(src, states[src], serverNow) : null))
    .filter((line): line is JobEvent => line !== null);
  const lines = [...events, ...webLines, ...silent].sort((a, b) => a.t - b.t);
  const issues = lines.filter((line) => line.level === "warn" || line.level === "error");

  // Fields: each endpoint's own output, compared cell by cell with the baseline.
  const fields: Record<string, Map<string, FieldValue>> = {};
  for (const c of columns) {
    const payload = runs[c.endpoint_id]?.payload;
    if (payload) fields[c.endpoint_id] = new Map(leafFields(payload));
  }
  const allPaths: string[] = [];
  for (const c of columns) for (const path of fields[c.endpoint_id]?.keys() ?? []) if (!allPaths.includes(path)) allPaths.push(path);
  const showAll = view === "all" && allPaths.length > 0;
  const paths = showAll ? allPaths : Object.keys(KEY_FIELDS);
  const baseFields = baselineId ? fields[baselineId] : undefined;
  const comparisons = columns.map((c) => runs[c.endpoint_id])
    .filter((r): r is RunReport => Boolean(r) && r!.endpoint_id !== baselineId && r!.total_fields !== null);
  const stored = result?.stored_endpoint ? runs[result.stored_endpoint] : undefined;

  // Step track: OCR -> Node 1 -> Lưu, from the job status, its timings and the log.
  const firstAt = (match: (line: JobEvent) => boolean) => events.find(match)?.t;
  const failedAt = job?.status === "failed" ? (job.error ?? "").split(":")[0] : null;
  const order = ["ocr", "node1", "saving", "done"];
  const reached = job ? order.indexOf(job.status === "failed" || job.status === "interrupted" ? (failedAt ?? "") : job.status) : -1;
  const stepState = (name: string): StepState => {
    if (failedAt === name) return "err";
    if (job?.status === "done") return "ok";
    const index = order.indexOf(name);
    if (reached > index) return "ok";
    if (reached === index && !failedAt) return "run";
    return "wait";
  };
  const saveStart = firstAt((l) => l.src === "system" && l.msg.startsWith("Lưu hoá đơn"));
  const saveEnd = firstAt((l) => l.src === "system" && l.msg.startsWith("Đã lưu"));
  const steps: { name: string; label: string; state: StepState; meta: string }[] = [
    { name: "ocr", label: "OCR", state: stepState("ocr"), meta: "" },
    { name: "node1", label: "Node 1", state: Object.values(runs).length > 0 && !Object.values(runs).some((r) => r.valid) ? "err" : stepState("node1"), meta: "" },
    { name: "saving", label: "Lưu", state: stepState("saving"), meta: "" },
  ];
  const started: Record<string, number | undefined> = {
    ocr: firstAt((l) => l.src === "ocr"), node1: firstAt((l) => columns.some((c) => c.endpoint_id === l.src)), saving: saveStart,
  };
  for (const s of steps) {
    const ms = s.name === "saving" ? (saveStart && saveEnd ? (saveEnd - saveStart) * 1000 : undefined) : result?.timings_ms?.[s.name];
    if (s.state === "run" && started[s.name] !== undefined) s.meta = durationVi(serverNow - started[s.name]!);
    else if (ms !== undefined && s.state !== "wait") s.meta = formatMs(ms);
  }

  const parts: LogPart[] = job ? [
    { id: "all", label: "Tất cả" },
    { id: "ocr", label: "OCR", state: states.ocr },
    ...columns.map((c) => ({ id: c.endpoint_id, label: c.name, color: colors[c.endpoint_id], state: states[c.endpoint_id] })),
    { id: "system", label: "Hệ thống", state: states.system },
  ] : [];

  return (
    <>
      <PageHead title="Trích xuất hoá đơn" />

      <Panel title="Đầu vào" icon={ImageUp}>
        <div className="grid gap-4 lg:grid-cols-[170px_minmax(0,1fr)_auto] lg:items-center">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              choose(e.dataTransfer.files?.[0] ?? null);
            }}
            className={cx(
              "grid min-h-[150px] cursor-pointer place-items-center gap-1 rounded-xl border-2 border-dashed p-2 text-center",
              dragging ? "border-ink bg-sunken" : "border-line hover:bg-sunken",
            )}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Ảnh đã chọn" className="max-h-[150px] rounded-lg object-contain" />
            ) : (
              <>
                <FileImage aria-hidden className="size-7 text-ink-3" />
                <span className="text-[12.5px] text-ink-3">Kéo thả hoặc bấm để chọn ảnh</span>
              </>
            )}
            <input type="file" accept="image/jpeg,image/png" aria-label="Ảnh hoá đơn" className="sr-only" onChange={(e) => choose(e.target.files?.[0] ?? null)} />
          </label>

          <div className="grid min-w-0 gap-3">
            <div>
              <b className="block truncate text-[15px]">{file ? file.name : "Chưa chọn ảnh"}</b>
              <span className="text-[13px] text-ink-3">JPEG hoặc PNG, tối đa 10 MB. Hoặc dùng ảnh mẫu trong devset:</span>
              <div className="mt-1.5 flex flex-wrap gap-2">
                {SAMPLES.map((s) => (
                  <button key={s.id} type="button" onClick={() => pickSample(s)} className="flex items-center gap-2 rounded-lg border border-line py-1 pl-1 pr-2.5 text-left text-[12.5px] text-ink-2 hover:bg-sunken">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={s.file} alt="" className="size-9 rounded-md object-cover object-top" />
                    <span className="max-w-[160px] truncate">{s.label}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="grid gap-1.5">
              <span className="text-[13.5px] font-semibold">Chạy trên</span>
              <EndpointChips
                endpoints={endpoints}
                selected={selected}
                onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-6)))}
              />
              <small className="text-[12.5px] text-ink-3">
                {baseline
                  ? `Mốc ${baseline.name} luôn chạy kèm để so sánh. Hoá đơn lưu lấy kết quả của endpoint chọn đầu tiên.`
                  : "Chưa có endpoint mốc nên chỉ hiện số tuyệt đối. Hoá đơn lưu lấy kết quả của endpoint chọn đầu tiên."}
              </small>
            </div>
          </div>

          <div className="grid gap-2 lg:min-w-[190px]">
            <Button icon={Play} loading={busy} disabled={!file || selected.length === 0 || busy} onClick={run}>
              {busy ? "Đang chạy" : "Chạy Node 1"}
            </Button>
          </div>
        </div>
        {error && <Notice tone="bad">{error}</Notice>}
        {result?.ocr?.text && (
          <details className="border-t border-line pt-3">
            <summary className="flex cursor-pointer items-center gap-2 text-[13.5px] font-semibold">
              Chữ OCR đọc được ({result.ocr.text.split("\n").length} dòng, đã bỏ dấu)
              <button
                type="button"
                aria-label="Xem log OCR"
                title="Xem log OCR"
                onClick={(e) => {
                  e.preventDefault();
                  setLogPart("ocr");
                }}
                className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken"
              >
                <ScrollText aria-hidden className="size-4" />
              </button>
            </summary>
            <div className="mt-2.5"><CodeBlock label="Văn bản OCR">{result.ocr.text}</CodeBlock></div>
          </details>
        )}
      </Panel>

      <Panel title="Kết quả" icon={ScanText} meta={job && <span>Job {job.job_id}</span>} bodyClassName="p-0 gap-0">
        {!job ? (
          <p className="p-[18px] text-ink-3">
            {busy ? "Đang gửi ảnh lên máy chủ…" : "Chọn ảnh và bấm “Chạy Node 1”. Output của từng endpoint, so với mốc và log hiện ở đây."}
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b border-line px-[18px] py-3">
              <ol aria-label="Các bước" className="flex flex-wrap items-center gap-2">
                {steps.map((s, i) => (
                  <li key={s.name} className="flex items-center gap-2">
                    {i > 0 && <span aria-hidden className="h-px w-5 bg-line" />}
                    <span className={cx("inline-flex items-center gap-1.5 text-[13px] font-semibold",
                      s.state === "ok" ? "text-ink" : s.state === "run" ? "text-step-1" : s.state === "err" ? "text-bad" : "text-ink-3")}>
                      {s.state === "run" ? <Spinner className="size-3.5" /> : s.state === "ok" ? <CheckCircle2 aria-hidden className="size-3.5 text-good" />
                        : s.state === "err" ? <XCircle aria-hidden className="size-3.5" /> : <Circle aria-hidden className="size-3.5" />}
                      {s.label}
                      {s.meta && <small className="num font-medium">{s.meta}</small>}
                    </span>
                  </li>
                ))}
              </ol>
              <div className="ml-auto flex flex-wrap items-center gap-2.5">
                {allPaths.length > 0 && (
                  <div role="group" aria-label="Cách xem kết quả" className="inline-flex overflow-hidden rounded-lg border border-line text-[13px]">
                    {([["key", "Trường chính"], ["all", `Đủ ${allPaths.length} trường`]] as const).map(([value, label]) => (
                      <button key={value} type="button" aria-pressed={view === value} onClick={() => setView(value)}
                        className={cx("px-2.5 py-1", view === value ? "bg-ink font-semibold text-surface" : "hover:bg-sunken")}>
                        {label}
                      </button>
                    ))}
                  </div>
                )}
                <button type="button" onClick={() => setLogPart("all")} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] font-semibold hover:bg-sunken">
                  <ScrollText aria-hidden className="size-4" />
                  Xem log
                  {issues.length > 0 && (
                    <span aria-hidden className={cx("num grid min-w-[18px] place-items-center rounded-full px-1.5 text-[11.5px]",
                      issues.some((l) => l.level === "error") ? "bg-bad-tint text-bad" : "bg-warn-tint text-warn")}>
                      {issues.length}
                    </span>
                  )}
                </button>
              </div>
            </div>

            {lostFor !== null && (
              <div className="px-[18px] pt-3">
                <Notice tone="bad">Mất liên lạc với backend {durationVi(lostFor)}; web đang thử lại. Job có thể vẫn đang chạy trên máy chủ.</Notice>
              </div>
            )}
            {job.status === "failed" && (
              <div className="px-[18px] pt-3">
                <Notice tone="bad">
                  Lỗi ở bước {job.error}
                  {job.error?.includes("SOURCE_BINDING") && (
                    <p className="mt-1.5 text-ink-2">
                      Model chép sai thông tin nguồn của ảnh (mã hoặc đường dẫn) mà luồng Node 1 bắt phải chép y nguyên. Lỗi này
                      thỉnh thoảng xảy ra với model 7B; bấm Chạy Node 1 lần nữa.
                    </p>
                  )}
                </Notice>
              </div>
            )}
            {job.status === "interrupted" && <div className="px-[18px] pt-3"><Notice tone="warn">Máy chủ khởi động lại giữa chừng; chạy lại lần nữa.</Notice></div>}

            <div className="overflow-x-auto">
              <table aria-label="So sánh với mốc" className="w-full min-w-[620px] border-collapse text-left text-[14px]">
                <thead>
                  <tr className="border-b border-line">
                    <th scope="col" className="w-[170px] px-[18px] pb-2.5 pt-3.5 align-top text-[13px] font-medium text-ink-3">Trường</th>
                    {columns.map((c) => (
                      <th key={c.endpoint_id} scope="col" className="px-3.5 pb-2.5 pt-3.5 align-top">
                        <RunHead
                          id={c.endpoint_id}
                          name={c.name}
                          color={colors[c.endpoint_id] ?? "var(--ink-2)"}
                          isBaseline={c.endpoint_id === baselineId}
                          report={runs[c.endpoint_id]}
                          state={states[c.endpoint_id]}
                          onLog={() => setLogPart(c.endpoint_id)}
                        />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paths.map((path, index) => {
                    const { group, label } = describeField(path);
                    const newGroup = showAll && (index === 0 || describeField(paths[index - 1]).group !== group);
                    return [
                      newGroup && (
                        <tr key={`g-${group}`} className="border-t border-line bg-sunken">
                          <th scope="colgroup" colSpan={columns.length + 1} className="px-[18px] py-1.5 text-[12.5px] font-bold">{group}</th>
                        </tr>
                      ),
                      <tr key={path} className="border-t border-line first:border-t-0">
                        <th scope="row" className="whitespace-nowrap px-[18px] py-2.5 align-top font-medium text-ink-2">{showAll ? label : KEY_FIELDS[path]}</th>
                        {columns.map((c) => {
                          const mine = fields[c.endpoint_id];
                          if (!mine) {
                            const failed = runs[c.endpoint_id] && !runs[c.endpoint_id].valid;
                            return <td key={c.endpoint_id} className={cx("px-3.5 py-2.5 align-top", failed ? "italic text-ink-3" : "text-ink-3")}>{failed ? "Không có kết quả" : "—"}</td>;
                          }
                          const value = mine.get(path);
                          const differs = c.endpoint_id !== baselineId && baseFields !== undefined && !sameValue(value, baseFields.get(path));
                          return (
                            <td key={c.endpoint_id} title={differs ? "Khác mốc" : undefined}
                              className={cx("break-words px-3.5 py-2.5 align-top", differs && "bg-warn-tint font-semibold text-warn")}>
                              {isEmptyValue(value) ? <span className="italic text-ink-3">Không có</span> : fieldText(value)}
                            </td>
                          );
                        })}
                      </tr>,
                    ];
                  })}
                  {allPaths.length > 0 && (
                    <tr className="border-t border-line">
                      <td />
                      <td colSpan={columns.length} className="px-3.5 py-3">
                        {showAll ? (
                          <Button variant="ghost" onClick={() => setView("key")}>Thu gọn về trường chính</Button>
                        ) : (
                          <span className="flex flex-wrap items-center gap-2.5">
                            <Button variant="ghost" onClick={() => setView("all")}>{`Xem đủ ${allPaths.length} trường`}</Button>
                            <span className="text-[13px] text-ink-3">gồm mọi trường của hoá đơn, theo nhóm</span>
                          </span>
                        )}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {(comparisons.length > 0 || stored) && (
              <div className="grid gap-4 rounded-b-2xl border-t border-line bg-sunken px-[18px] py-4">
                {comparisons.map((r) => <Conclusion key={r.endpoint_id} run={r} base={baselineId ? runs[baselineId] : undefined} />)}
                {stored && (
                  <p className="text-[13px] text-ink-2">
                    Đã lưu hoá đơn từ <b className="text-ink">{stored.name}</b>
                    {result?.stored_from_fallback ? " (endpoint chọn đầu bị lỗi, lấy kết quả của mốc)" : ""}.
                  </p>
                )}
                {!baselineId && Object.keys(runs).length > 0 && <p className="text-[13px] text-ink-3">Chưa có endpoint mốc nên chưa so sánh.</p>}
              </div>
            )}

            {result?.invoice && (
              <div className="grid gap-4 border-t border-line px-[18px] py-4">
                <Tabs label="Hoá đơn đã lưu" value={tab} onChange={setTab} tabs={[["info", "Hoá đơn đã lưu"], ["json", "JSON"]]} />
                {tab === "info" && <InvoiceDetail invoice={result.invoice} itemLines={result.item_check?.lines} invoiceSum={result.item_check?.invoice_sum} />}
                {tab === "json" && <CodeBlock label="JSON hoá đơn">{JSON.stringify(result.invoice, null, 2)}</CodeBlock>}
                {result.grounding && result.grounding.ungrounded_fields.length > 0 && (
                  <Notice tone="warn">
                    {result.grounding.ungrounded_fields.length} trường không thấy trong văn bản OCR: {result.grounding.ungrounded_fields.join(", ")}
                  </Notice>
                )}
              </div>
            )}
          </>
        )}
      </Panel>

      <SavedInvoices api={api} refresh={refresh} />

      {logPart !== null && job && (
        <JobLogDrawer
          lines={lines}
          parts={parts}
          part={logPart}
          onPart={setLogPart}
          onClose={closeLog}
          sourceLabel={(src) => (src === "system" || src === "web" ? "Hệ thống" : src === "ocr" ? "OCR" : nameOf(src))}
          sourceColor={(src) => (src === "ocr" ? "var(--step-1)" : colors[src])}
          elapsed={`${lines.length} dòng`}
        />
      )}
    </>
  );
}
