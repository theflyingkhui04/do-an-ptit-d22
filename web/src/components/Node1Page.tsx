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
  ReferencePicker,
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
import { KEY_FIELDS_BY_LANG, describeField, fieldText, isEmptyValue, leafFields, sameValue, type FieldValue } from "@/lib/fields";
import { formatMoney, formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { useI18n, type Dict, type Lang } from "@/lib/i18n";
import { durationText, endpointState, ocrState, silentLine, systemState, type PartState } from "@/lib/liveness";
import { SAMPLES } from "@/lib/official";

const MAX_BYTES = 10 * 1024 * 1024;
const TONE_TEXT: Record<PartState["tone"], string> = { good: "text-good", warn: "text-warn", bad: "text-bad", idle: "text-ink-3" };

function uploadError(exc: unknown, t: Dict): string {
  if (exc instanceof ApiError) {
    if (exc.status === 413) return t.errTooLarge;
    if (exc.status === 415) return t.errInvalidType;
    if (exc.status === 429) return t.errRateLimit;
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

function RunHead({ id, name, color, isReference, report, state, onLog }: {
  id: string;
  name: string;
  color: string;
  isReference: boolean;
  report?: RunReport;
  state: PartState;
  onLog: () => void;
}) {
  const { lang, t } = useI18n();
  return (
    <div className="grid min-w-[210px] gap-1.5 text-left font-normal" data-testid={`run-${id}`}>
      <div className="flex items-center gap-2">
        <Dot color={color} />
        <b className="truncate text-[15px] font-bold">{name}</b>
        {isReference && <span className="rounded-md border border-line px-1.5 text-[11.5px] font-semibold text-ink-3">{t.referenceBadge}</span>}
        <button type="button" onClick={onLog} aria-label={t.viewLogAria(name)} title={t.viewLogAria(name)} className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken">
          <ScrollText aria-hidden className="size-4" />
        </button>
      </div>
      <LiveState state={state} />
      {report?.valid && (
        <div className="num flex flex-wrap gap-x-3.5 gap-y-0.5 text-[12.5px] text-ink-2">
          <span>TTFT <b className="text-ink">{formatMs(report.ttft_ms, lang)}</b></span>
          <span>TPOT <b className="text-ink">{formatMs(report.tpot_ms, lang)}</b></span>
          <span><b className="text-ink">{formatNumber(report.completion_tokens, lang)}</b> token</span>
        </div>
      )}
      {report && !report.valid && <p className="line-clamp-3 text-[12.5px] text-bad" title={report.error ?? undefined}>{report.error}</p>}
    </div>
  );
}

interface PairComparison {
  endpointId: string;
  name: string;
  equalFields: number | null;
  totalFields: number | null;
  differingPaths: string[];
  speedup: number | null;
  tpotMs: number | null | undefined;
  baseTpotMs: number | null | undefined;
}

function diffNote(paths: string[], lang: Lang, t: Dict): string {
  if (paths.length === 0) return t.allFieldsMatch;
  const named = paths.slice(0, 4).map((path) => {
    const { group, label } = describeField(path, lang);
    return `${group} · ${label.toLowerCase()}`;
  });
  return t.differsAt(named, Math.max(0, paths.length - 4));
}

function Conclusion({ cmp, refName }: { cmp: PairComparison; refName: string }) {
  const { lang, t } = useI18n();
  const same = cmp.differingPaths.length === 0;
  return (
    <section className="grid gap-2" data-testid={`conclusion-${cmp.endpointId}`} aria-label={`${cmp.name} · ${t.compareTableTitle(refName)}`}>
      <h3 className="text-[13.5px] font-semibold text-ink-2">{cmp.name} · {t.compareTableTitle(refName)}</h3>
      <div className="grid gap-x-7 gap-y-3 sm:grid-cols-2">
        {cmp.equalFields !== null && cmp.totalFields !== null && (
          <div className="grid content-start gap-0.5">
            <span className="text-[13px] font-medium text-ink-3">{t.matchFieldsLabel}</span>
            <b className={cx("num text-[24px] leading-tight", same ? "text-good" : "text-warn")}>{t.fieldsCount(cmp.equalFields, cmp.totalFields)}</b>
            <small className="text-[13px] text-ink-2">{diffNote(cmp.differingPaths, lang, t)}</small>
          </div>
        )}
        {cmp.speedup !== null && (
          <div className="grid content-start gap-0.5">
            <span className="text-[13px] font-medium text-ink-3">{cmp.speedup >= 1 ? t.fasterLabel : t.slowerLabel}</span>
            <b className={cx("num text-[24px] leading-tight", cmp.speedup >= 1 ? "text-good" : "text-bad")}>{formatSpeedup(cmp.speedup, lang)}</b>
            <small className="text-[13px] text-ink-2">{t.tpotDetail(formatMs(cmp.tpotMs, lang), formatMs(cmp.baseTpotMs, lang))}</small>
          </div>
        )}
      </div>
    </section>
  );
}

function SavedInvoices({ api, refresh }: { api: Api; refresh: number }) {
  const { lang, t } = useI18n();
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
    <Panel title={t.savedInvoicesTitle} icon={Database} bodyClassName="p-0 gap-0">
      {error && <div className="p-[18px]"><Notice tone="bad">{error}</Notice></div>}
      {rows && rows.length === 0 && <p className="p-[18px] text-ink-3">{t.noSavedInvoices}</p>}
      {rows && rows.length > 0 && (
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full min-w-[720px] text-left text-[13.5px]">
            <thead className="sticky top-0 bg-surface text-ink-3">
              <tr className="border-b border-line">
                <th className="px-[18px] py-2.5 font-semibold">{t.colStore}</th>
                <th className="px-3 py-2.5 font-semibold">{t.colDate}</th>
                <th className="px-3 py-2.5 text-right font-semibold">{t.colTotal}</th>
                <th className="px-3 py-2.5 text-right font-semibold">{t.colItems}</th>
                <th className="px-3 py-2.5 font-semibold">{t.colSource}</th>
                <th className="px-[18px] py-2.5 font-semibold">{t.colExtractedBy}</th>
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
                  <td className="px-3 py-2 text-right">{formatMoney(row.total_amount_value, row.total_amount_currency, lang)}</td>
                  <td className="px-3 py-2 text-right">{row.item_count}</td>
                  <td className="px-3 py-2">{row.source === "upload" ? <Pill tone="idle">{t.sourceUpload}</Pill> : t.sourceDevset}</td>
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
            {open.lane && <span className="text-[13px] text-ink-3">{open.lane}</span>}
            <Button variant="ghost" className="ml-auto" onClick={() => setOpen(null)}>{t.close}</Button>
          </div>
          <div className={cx("grid gap-5", open.image_url && "lg:grid-cols-[220px_minmax(0,1fr)]")}>
            {open.image_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={open.image_url} alt={t.invoiceImageAlt} className="max-h-[360px] w-full rounded-lg border border-line object-contain" />
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
  const { lang, t } = useI18n();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints));
  const [job, setJob] = useState<Job | null>(null);
  const [refChoice, setRefChoice] = useState<string | null>(null);
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
  const closeLog = useCallback(() => setLogPart(null), []);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const choose = useCallback((chosen: File | null) => {
    setJob(null);
    setRefChoice(null);
    setError(null);
    if (chosen && !["image/jpeg", "image/png"].includes(chosen.type)) {
      setError(t.errInvalidType);
      return;
    }
    if (chosen && chosen.size > MAX_BYTES) {
      setError(t.errTooLarge);
      return;
    }
    setFile(chosen);
    setPreview((old) => {
      if (old) URL.revokeObjectURL?.(old);
      return chosen && typeof URL.createObjectURL === "function" ? URL.createObjectURL(chosen) : null;
    });
  }, [t]);

  async function pickSample(sample: (typeof SAMPLES)[number]) {
    try {
      const response = await fetch(sample.file);
      const blob = await response.blob();
      choose(new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" }));
    } catch {
      setError(t.errSampleLoad);
    }
  }

  function addWebLine(level: JobEvent["level"], msg: string, key?: string) {
    const { server, client } = clockRef.current;
    const timestamp = server ? server + (Date.now() - client) / 1000 : Date.now() / 1000;
    setWebLines((lines) => [...lines.filter((line) => !key || line.key !== key), { t: timestamp, src: "web", level, msg, key }]);
  }

  async function run() {
    if (!file || selected.length === 0) return;
    setBusy(true);
    setError(null);
    setJob(null);
    setRefChoice(null);
    setWebLines([]);
    setLostSince(null);
    setTab("info");
    let jobId: string;
    try {
      ({ job_id: jobId } = await api.upload(file, selected));
    } catch (exc) {
      setError(uploadError(exc, t));
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
          if (exc instanceof ApiError && exc.status < 500) throw exc;
          if (!alive.current) return;
          lost ??= Date.now();
          tries += 1;
          setLostSince(lost);
          addWebLine("error", t.logLostBackend(tries, durationText((Date.now() - lost) / 1000, lang)), "lost");
          await sleep(pollMs * 3);
          continue;
        }
        if (!alive.current) return;
        if (lost !== null) {
          addWebLine("ok", t.logReconnected(durationText((Date.now() - lost) / 1000, lang)));
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
      setError(uploadError(exc, t));
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  if (endpoints.length === 0) return <><PageHead title={t.pageNode1} /><NeedEndpoint what="Node 1" onGo={() => go("endpoints")} /></>;

  const colors = endpointColors(endpoints);
  const result = job?.result;
  const runs = result?.runs ?? {};
  const rawColumns = result?.endpoints ?? [];
  const validRef = (id: string | null | undefined) => (id && rawColumns.some((c) => c.endpoint_id === id) ? id : null);
  const referenceId = rawColumns.length >= 2
    ? (validRef(refChoice) ?? validRef(result?.baseline) ?? rawColumns[0]?.endpoint_id ?? null)
    : null;
  const columns = [...rawColumns].sort((a, b) =>
    Number(b.endpoint_id === referenceId) - Number(a.endpoint_id === referenceId));
  const nameOf = (id: string) => columns.find((c) => c.endpoint_id === id)?.name ?? endpoints.find((e) => e.endpoint_id === id)?.name ?? id;

  const serverNow = clock.server + Math.max(0, (nowMs - clock.client) / 1000);
  const events = result?.events ?? [];
  const lostFor = lostSince === null ? null : Math.max(0, (nowMs - lostSince) / 1000);
  const states: Record<string, PartState> = {};
  if (job) {
    states.system = systemState(job, lostFor, lang);
    states.ocr = ocrState(job, events, serverNow, lang);
    for (const c of columns) states[c.endpoint_id] = endpointState(job, events, c.endpoint_id, serverNow, lang);
  }
  const silent = ["ocr", ...columns.map((c) => c.endpoint_id)]
    .map((src) => (states[src] ? silentLine(src, states[src], serverNow, lang) : null))
    .filter((line): line is JobEvent => line !== null);
  const lines = [...events, ...webLines, ...silent].sort((a, b) => a.t - b.t);
  const issues = lines.filter((line) => line.level === "warn" || line.level === "error");

  const fields: Record<string, Map<string, FieldValue>> = {};
  for (const c of columns) {
    const payload = runs[c.endpoint_id]?.payload;
    if (payload) fields[c.endpoint_id] = new Map(leafFields(payload));
  }
  const allPaths: string[] = [];
  for (const c of columns) for (const path of fields[c.endpoint_id]?.keys() ?? []) if (!allPaths.includes(path)) allPaths.push(path);
  const showAll = view === "all" && allPaths.length > 0;
  const keyFieldLabels = KEY_FIELDS_BY_LANG[lang];
  const paths = showAll ? allPaths : Object.keys(keyFieldLabels);
  const refFields = referenceId ? fields[referenceId] : undefined;
  const refRun = referenceId ? runs[referenceId] : undefined;

  const comparisons: PairComparison[] = [];
  if (referenceId && refRun?.valid) {
    for (const c of columns) {
      if (c.endpoint_id === referenceId) continue;
      const runReport = runs[c.endpoint_id];
      if (!runReport?.valid) continue;
      const mine = fields[c.endpoint_id];
      let equalFields: number | null = null;
      let totalFields: number | null = null;
      let differingPaths: string[] = [];
      if (mine && refFields) {
        const union = [...new Set([...mine.keys(), ...refFields.keys()])];
        differingPaths = union.filter((p) => !sameValue(mine.get(p), refFields.get(p)));
        totalFields = union.length;
        equalFields = union.length - differingPaths.length;
      } else if (referenceId === result?.baseline && runReport.total_fields !== null) {
        equalFields = runReport.equal_fields;
        totalFields = runReport.total_fields;
        differingPaths = runReport.differing_fields;
      }
      const speedup = runReport.tpot_ms && refRun.tpot_ms ? refRun.tpot_ms / runReport.tpot_ms : null;
      comparisons.push({
        endpointId: c.endpoint_id,
        name: c.name,
        equalFields,
        totalFields,
        differingPaths,
        speedup,
        tpotMs: runReport.tpot_ms,
        baseTpotMs: refRun.tpot_ms,
      });
    }
  }
  const stored = result?.stored_endpoint ? runs[result.stored_endpoint] : undefined;

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
    { name: "ocr", label: t.stepOcr, state: stepState("ocr"), meta: "" },
    { name: "node1", label: t.stepNode1, state: Object.values(runs).length > 0 && !Object.values(runs).some((r) => r.valid) ? "err" : stepState("node1"), meta: "" },
    { name: "saving", label: t.stepSaving, state: stepState("saving"), meta: "" },
  ];
  const started: Record<string, number | undefined> = {
    ocr: firstAt((l) => l.src === "ocr"),
    node1: firstAt((l) => columns.some((c) => c.endpoint_id === l.src)),
    saving: saveStart,
  };
  for (const s of steps) {
    const ms = s.name === "saving" ? (saveStart && saveEnd ? (saveEnd - saveStart) * 1000 : undefined) : result?.timings_ms?.[s.name];
    if (s.state === "run" && started[s.name] !== undefined) s.meta = durationText(serverNow - started[s.name]!, lang);
    else if (ms !== undefined && s.state !== "wait") s.meta = formatMs(ms, lang);
  }

  const parts: LogPart[] = job ? [
    { id: "all", label: t.logAll },
    { id: "ocr", label: t.stepOcr, state: states.ocr },
    ...columns.map((c) => ({ id: c.endpoint_id, label: c.name, color: colors[c.endpoint_id], state: states[c.endpoint_id] })),
    { id: "system", label: t.logSystem, state: states.system },
  ] : [];

  return (
    <>
      <PageHead title={t.pageNode1} />

      <Panel title={t.inputPanel} icon={ImageUp}>
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
              <img src={preview} alt={t.selectedImageAlt} className="max-h-[150px] rounded-lg object-contain" />
            ) : (
              <>
                <FileImage aria-hidden className="size-7 text-ink-3" />
                <span className="text-[12.5px] text-ink-3">{t.dropImage}</span>
              </>
            )}
            <input type="file" accept="image/jpeg,image/png" aria-label={t.imageInputAria} className="sr-only" onChange={(e) => choose(e.target.files?.[0] ?? null)} />
          </label>

          <div className="grid min-w-0 gap-3">
            <div>
              <b className="block truncate text-[15px]">{file ? file.name : t.selectImage}</b>
              <span className="text-[13px] text-ink-3">{t.imageConstraints}</span>
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
              <span className="text-[13.5px] font-semibold">{t.selectLlm}</span>
              <EndpointChips
                endpoints={endpoints}
                selected={selected}
                onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-6)))}
              />
            </div>
          </div>

          <div className="grid gap-2 lg:min-w-[170px]">
            <Button icon={Play} loading={busy} disabled={!file || selected.length === 0 || busy} onClick={run}>
              {busy ? t.runningBtn : t.startRun}
            </Button>
          </div>
        </div>
        {error && <Notice tone="bad">{error}</Notice>}
        {result?.ocr?.text && (
          <details className="border-t border-line pt-3">
            <summary className="flex cursor-pointer items-center gap-2 text-[13.5px] font-semibold">
              {t.ocrTextTitle(result.ocr.text.split("\n").length)}
              <button
                type="button"
                aria-label={t.viewOcrLog}
                title={t.viewOcrLog}
                onClick={(e) => {
                  e.preventDefault();
                  setLogPart("ocr");
                }}
                className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken"
              >
                <ScrollText aria-hidden className="size-4" />
              </button>
            </summary>
            <div className="mt-2.5"><CodeBlock label={t.ocrTextLabel}>{result.ocr.text}</CodeBlock></div>
          </details>
        )}
      </Panel>

      <Panel
        title={t.resultsPanel}
        icon={ScanText}
        meta={referenceId && (
          <ReferencePicker
            options={rawColumns.map((c) => ({ id: c.endpoint_id, name: c.name }))}
            value={referenceId}
            onChange={setRefChoice}
            colors={colors}
          />
        )}
        bodyClassName="p-0 gap-0"
      >
        {!job ? (
          busy && <p className="p-[18px] text-ink-3">{t.uploadingImage}</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b border-line px-[18px] py-3">
              <ol aria-label={t.stepsAria} className="flex flex-wrap items-center gap-2">
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
                  <div role="group" aria-label={t.viewModeAria} className="inline-flex overflow-hidden rounded-lg border border-line text-[13px]">
                    {([["key", t.keyFieldsBtn], ["all", t.allFieldsBtn(allPaths.length)]] as const).map(([value, label]) => (
                      <button key={value} type="button" aria-pressed={view === value} onClick={() => setView(value)}
                        className={cx("px-2.5 py-1", view === value ? "bg-ink font-semibold text-surface" : "hover:bg-sunken")}>
                        {label}
                      </button>
                    ))}
                  </div>
                )}
                <button type="button" onClick={() => setLogPart("all")} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] font-semibold hover:bg-sunken">
                  <ScrollText aria-hidden className="size-4" />
                  {t.viewLogsBtn}
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
                <Notice tone="bad">{t.lostBackendNotice(durationText(lostFor, lang))}</Notice>
              </div>
            )}
            {job.status === "failed" && (
              <div className="px-[18px] pt-3">
                <Notice tone="bad">{t.failedStepPrefix} {job.error}</Notice>
              </div>
            )}
            {job.status === "interrupted" && <div className="px-[18px] pt-3"><Notice tone="warn">{t.interruptedNotice}</Notice></div>}

            <div className="overflow-x-auto">
              <table aria-label={t.comparisonTableAria} className="w-full min-w-[620px] border-collapse text-left text-[14px]">
                <thead>
                  <tr className="border-b border-line">
                    <th scope="col" className="w-[170px] px-[18px] pb-2.5 pt-3.5 align-top text-[13px] font-medium text-ink-3">{t.fieldCol}</th>
                    {columns.map((c) => (
                      <th key={c.endpoint_id} scope="col" className="px-3.5 pb-2.5 pt-3.5 align-top">
                        <RunHead
                          id={c.endpoint_id}
                          name={c.name}
                          color={colors[c.endpoint_id] ?? "var(--ink-2)"}
                          isReference={c.endpoint_id === referenceId}
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
                    const { group, label } = describeField(path, lang);
                    const newGroup = showAll && (index === 0 || describeField(paths[index - 1], lang).group !== group);
                    return [
                      newGroup && (
                        <tr key={`g-${group}`} className="border-t border-line bg-sunken">
                          <th scope="colgroup" colSpan={columns.length + 1} className="px-[18px] py-1.5 text-[12.5px] font-bold">{group}</th>
                        </tr>
                      ),
                      <tr key={path} className="border-t border-line first:border-t-0">
                        <th scope="row" className="whitespace-nowrap px-[18px] py-2.5 align-top font-medium text-ink-2">{showAll ? label : keyFieldLabels[path]}</th>
                        {columns.map((c) => {
                          const mine = fields[c.endpoint_id];
                          if (!mine) {
                            const failed = runs[c.endpoint_id] && !runs[c.endpoint_id].valid;
                            return <td key={c.endpoint_id} className={cx("px-3.5 py-2.5 align-top", failed ? "italic text-ink-3" : "text-ink-3")}>{failed ? t.noResultCell : "—"}</td>;
                          }
                          const value = mine.get(path);
                          const differs = c.endpoint_id !== referenceId && refFields !== undefined && !sameValue(value, refFields.get(path));
                          return (
                            <td key={c.endpoint_id} title={differs ? t.diffTooltip : undefined}
                              className={cx("break-words px-3.5 py-2.5 align-top", differs && "bg-warn-tint font-semibold text-warn")}>
                              {isEmptyValue(value) ? <span className="italic text-ink-3">{t.emptyValue}</span> : fieldText(value, lang)}
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
                          <Button variant="ghost" onClick={() => setView("key")}>{t.collapseFields}</Button>
                        ) : (
                          <Button variant="ghost" onClick={() => setView("all")}>{t.expandFields(allPaths.length)}</Button>
                        )}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {(comparisons.length > 0 || stored) && (
              <div className="grid gap-4 rounded-b-2xl border-t border-line bg-sunken px-[18px] py-4">
                {referenceId && comparisons.map((cmp) => (
                  <Conclusion key={cmp.endpointId} cmp={cmp} refName={nameOf(referenceId)} />
                ))}
                {stored && (
                  <p className="text-[13px] text-ink-2">
                    {t.storedFrom} <b className="text-ink">{stored.name}</b>.
                  </p>
                )}
              </div>
            )}

            {result?.invoice && (
              <div className="grid gap-4 border-t border-line px-[18px] py-4">
                <Tabs label={t.storedInvoiceTab} value={tab} onChange={setTab} tabs={[["info", t.storedInvoiceTab], ["json", t.jsonTab]]} />
                {tab === "info" && <InvoiceDetail invoice={result.invoice} itemLines={result.item_check?.lines} invoiceSum={result.item_check?.invoice_sum} />}
                {tab === "json" && <CodeBlock label={t.invoiceJsonLabel}>{JSON.stringify(result.invoice, null, 2)}</CodeBlock>}
                {result.grounding && result.grounding.ungrounded_fields.length > 0 && (
                  <Notice tone="warn">
                    {t.ungroundedFieldsWarn(result.grounding.ungrounded_fields.length, result.grounding.ungrounded_fields.join(", "))}
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
          sourceLabel={(src) => (src === "system" || src === "web" ? t.logSystem : src === "ocr" ? t.stepOcr : nameOf(src))}
          sourceColor={(src) => (src === "ocr" ? "var(--step-1)" : colors[src])}
          elapsed={t.logLinesCount(lines.length)}
        />
      )}
    </>
  );
}
