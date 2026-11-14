"use client";

import { Database, FileImage, ImageUp, Play, ScanText } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import InvoiceDetail from "@/components/Invoice";
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
  Tabs,
  Trace,
  cx,
  defaultTarget,
  endpointColors,
} from "@/components/ui";
import {
  ApiError,
  FINAL_JOB_STATUSES,
  type Api,
  type Endpoint,
  type InvoiceDetail as Detail,
  type InvoiceRow,
  type Job,
  type RunReport,
} from "@/lib/api";
import { formatMoney, formatMs, formatNumber, formatSpeedup, STEP_LABELS } from "@/lib/format";
import { SAMPLES } from "@/lib/official";

const MAX_BYTES = 10 * 1024 * 1024;

function uploadError(exc: unknown): string {
  if (exc instanceof ApiError) {
    if (exc.status === 413) return "Ảnh lớn hơn 10 MB.";
    if (exc.status === 415) return "Tệp không phải ảnh JPEG hoặc PNG.";
    if (exc.status === 429) return "Đã chạy quá nhiều lần trong ít phút, thử lại sau.";
    return exc.message;
  }
  return (exc as Error).message;
}

function RunCard({ run, color }: { run: RunReport; color: string }) {
  const same = run.total_fields ? Math.round((100 * (run.equal_fields ?? 0)) / run.total_fields) : null;
  return (
    <div className="grid content-start gap-2 rounded-xl border border-line p-3.5" data-testid={`run-${run.endpoint_id}`}>
      <div className="flex items-center gap-2">
        <Dot color={color} />
        <b className="truncate">{run.name}</b>
        {run.is_baseline && <span className="rounded-md border border-line px-1.5 text-[11.5px] text-ink-3">mốc</span>}
        <span className="ml-auto">{run.valid ? <Pill tone="good">JSON hợp lệ</Pill> : <Pill tone="bad">Lỗi</Pill>}</span>
      </div>
      {run.valid ? (
        <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
          <dt className="text-ink-3">Thời gian</dt><dd className="text-right font-semibold">{formatMs(run.latency_ms)}</dd>
          <dt className="text-ink-3">TTFT</dt><dd className="text-right">{formatMs(run.ttft_ms)}</dd>
          <dt className="text-ink-3">TPOT</dt><dd className="text-right">{formatMs(run.tpot_ms)}</dd>
          <dt className="text-ink-3">Token/giây</dt><dd className="text-right">{formatNumber(run.output_tokens_per_s ? Math.round(run.output_tokens_per_s) : null)}</dd>
          {run.speedup_tpot_vs_baseline !== null && (
            <>
              <dt className="text-ink-3">So với mốc (TPOT)</dt>
              <dd className={cx("text-right font-semibold", run.speedup_tpot_vs_baseline >= 1 ? "text-good" : "text-bad")}>
                {formatSpeedup(run.speedup_tpot_vs_baseline)}
              </dd>
            </>
          )}
        </dl>
      ) : (
        <p className="text-[13px] text-bad">{run.error}</p>
      )}
      {(run.source_restored ?? []).length > 0 && (
        <p className="text-[12.5px] text-ink-3" title="Model chép sai giá trị nguồn mà hệ thống đã gửi; hệ thống điền lại giá trị đúng.">
          Hệ thống điền lại nguồn: {run.source_restored!.join(", ")}
        </p>
      )}
      {same !== null && (
        <div className="text-[13px]">
          {run.differing_fields.length === 0 ? (
            <Pill tone="good">Giống mốc {run.equal_fields}/{run.total_fields} trường</Pill>
          ) : (
            <details>
              <summary className="cursor-pointer"><Pill tone="warn">Giống mốc {same}%</Pill></summary>
              <p className="mt-1 break-words font-mono text-[12px] text-ink-3">{run.differing_fields.join(", ")}</p>
            </details>
          )}
        </div>
      )}
    </div>
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
  const [tab, setTab] = useState<"info" | "json" | "ocr">("info");
  const [refresh, setRefresh] = useState(0);
  const alive = useRef(true);

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

  async function run() {
    if (!file || selected.length === 0) return;
    setBusy(true);
    setError(null);
    setJob(null);
    setTab("info");
    try {
      const { job_id } = await api.upload(file, selected);
      for (;;) {
        const current = await api.job(job_id);
        if (!alive.current) return;
        setJob(current);
        if (FINAL_JOB_STATUSES.includes(current.status)) {
          if (current.status === "done") setRefresh((n) => n + 1);
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, pollMs));
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
  const runs = result?.runs ? Object.values(result.runs) : [];
  const stored = result?.stored_endpoint ? result.runs?.[result.stored_endpoint] : undefined;
  const steps = ["ocr", "node1", "saving", "done"] as const;

  return (
    <>
      <PageHead title="Trích xuất hoá đơn">
        Ảnh hoá đơn qua OCR (PaddleOCR + VietOCR, bỏ dấu), rồi Node 1 sinh JSON theo hợp đồng invoice_extraction.v1 và lưu vào cơ sở dữ liệu.
      </PageHead>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(320px,400px)_minmax(0,1fr)]">
        <Panel title="Đầu vào" icon={ImageUp}>
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
              "grid cursor-pointer justify-items-center gap-2 rounded-xl border-2 border-dashed px-4 py-6 text-center",
              dragging ? "border-ink bg-sunken" : "border-line hover:bg-sunken",
            )}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Ảnh đã chọn" className="max-h-[220px] rounded-lg object-contain" />
            ) : (
              <FileImage aria-hidden className="size-8 text-ink-3" />
            )}
            <span className="font-semibold">{file ? file.name : "Kéo thả ảnh hoá đơn vào đây"}</span>
            <span className="text-[13px] text-ink-3">hoặc bấm để chọn tệp, JPEG hoặc PNG, tối đa 10 MB</span>
            <input
              type="file"
              accept="image/jpeg,image/png"
              aria-label="Ảnh hoá đơn"
              className="sr-only"
              onChange={(e) => choose(e.target.files?.[0] ?? null)}
            />
          </label>

          <div className="grid gap-2">
            <span className="text-[13.5px] font-semibold">Hoặc dùng ảnh mẫu trong devset</span>
            <div className="grid grid-cols-3 gap-2">
              {SAMPLES.map((s) => (
                <button key={s.id} type="button" onClick={() => pickSample(s)} className="grid gap-1 rounded-lg border border-line p-1.5 text-left text-[12px] text-ink-2 hover:bg-sunken">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={s.file} alt="" className="h-20 w-full rounded-md object-cover object-top" />
                  <span className="line-clamp-2">{s.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-2">
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

          <Button icon={Play} disabled={!file || selected.length === 0 || busy} onClick={run}>
            {busy ? "Đang chạy…" : "Chạy Node 1"}
          </Button>
          {error && <Notice tone="bad">{error}</Notice>}
        </Panel>

        <Panel title="Kết quả" icon={ScanText} meta={job && <span>Job {job.job_id}</span>}>
          {!job && !busy && <p className="text-ink-3">Chọn ảnh và bấm “Chạy Node 1”. Kết quả, thời gian từng bước và so sánh với mốc hiện ở đây.</p>}
          {job && !FINAL_JOB_STATUSES.includes(job.status) && (
            <ol className="flex flex-wrap gap-2" aria-label="Tiến độ">
              {steps.map((s) => {
                const index = steps.indexOf(s);
                const at = steps.indexOf(job.status as (typeof steps)[number]);
                return (
                  <li key={s}>
                    <Pill tone={index < at ? "good" : index === at ? "warn" : "idle"}>{STEP_LABELS[s]}</Pill>
                  </li>
                );
              })}
            </ol>
          )}
          {job?.status === "failed" && (
            <Notice tone="bad">
              Lỗi ở bước {job.error}
              {job.error?.includes("SOURCE_BINDING") && (
                <p className="mt-1.5 text-ink-2">
                  Model chép sai thông tin nguồn của ảnh (mã hoặc đường dẫn) mà luồng Node 1 bắt phải chép y nguyên. Lỗi này
                  thỉnh thoảng xảy ra với model 7B; bấm Chạy Node 1 lần nữa.
                </p>
              )}
            </Notice>
          )}
          {job?.status === "interrupted" && <Notice tone="warn">Máy chủ khởi động lại giữa chừng; chạy lại lần nữa.</Notice>}

          {result?.timings_ms && (result.timings_ms.ocr || result.timings_ms.node1) && (
            <Trace
              steps={[
                { label: "OCR", title: "OCR thư viện", ms: result.timings_ms.ocr, kind: "io" },
                { label: "Node 1", title: `Node 1 (${runs.length} endpoint song song)`, ms: result.timings_ms.node1, kind: "n1" },
              ]}
              extra={stored && <span>Lưu từ <b className="text-ink">{stored.name}</b>{result.stored_from_fallback ? " (endpoint chọn đầu bị lỗi, lấy mốc)" : ""}</span>}
            />
          )}

          {runs.length > 0 && (
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {runs.map((r) => <RunCard key={r.endpoint_id} run={r} color={colors[r.endpoint_id] ?? "var(--ink-2)"} />)}
            </div>
          )}

          {result?.invoice && (
            <div className="grid gap-4">
              <Tabs
                label="Kết quả Node 1"
                value={tab}
                onChange={setTab}
                tabs={[["info", "Thông tin"], ["json", "JSON"], ["ocr", "Văn bản OCR"]]}
              />
              {tab === "info" && (
                <InvoiceDetail invoice={result.invoice} itemLines={result.item_check?.lines} invoiceSum={result.item_check?.invoice_sum} />
              )}
              {tab === "json" && <CodeBlock label="JSON hoá đơn">{JSON.stringify(result.invoice, null, 2)}</CodeBlock>}
              {tab === "ocr" && (
                <div className="grid gap-2">
                  <p className="text-[13px] text-ink-3">Văn bản đã bỏ dấu để tránh sai dấu khi OCR; câu hỏi ở Node 2 gõ có dấu hay không dấu đều được.</p>
                  <CodeBlock label="Văn bản OCR">{result.ocr?.text ?? ""}</CodeBlock>
                </div>
              )}
              {result.grounding && result.grounding.ungrounded_fields.length > 0 && (
                <Notice tone="warn">
                  {result.grounding.ungrounded_fields.length} trường không thấy trong văn bản OCR: {result.grounding.ungrounded_fields.join(", ")}
                </Notice>
              )}
            </div>
          )}
        </Panel>
      </div>

      <SavedInvoices api={api} refresh={refresh} />
    </>
  );
}
