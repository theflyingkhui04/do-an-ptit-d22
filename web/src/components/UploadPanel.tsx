"use client";

import { useEffect, useRef, useState } from "react";

import InvoiceView from "@/components/InvoiceView";
import LaneComparison from "@/components/LaneComparison";
import { ApiError, FINAL_JOB_STATUSES, type Api, type Job, type JobStatus } from "@/lib/api";
import { formatMs, STEP_LABELS } from "@/lib/format";

const STEPS: JobStatus[] = ["ocr", "node1", "saving", "done"];

interface Props {
  api: Api;
  pollMs?: number;
  onStored?: (invoiceId: string) => void;
}

function uploadError(exc: unknown): string {
  if (exc instanceof ApiError) {
    if (exc.status === 413) return "Ảnh quá lớn (tối đa 10 MB).";
    if (exc.status === 415) return "Tệp không phải ảnh JPEG/PNG.";
    if (exc.status === 429) return "Đã tải quá nhiều ảnh, hãy thử lại sau ít phút.";
    return exc.message;
  }
  return (exc as Error).message;
}

export default function UploadPanel({ api, pollMs = 1000, onStored }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [compareAll, setCompareAll] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    return () => {
      cancelled.current = true;
    };
  }, []);

  function choose(event: React.ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0] ?? null;
    setFile(chosen);
    setJob(null);
    setError(null);
    if (preview) URL.revokeObjectURL?.(preview);
    setPreview(chosen && typeof URL.createObjectURL === "function" ? URL.createObjectURL(chosen) : null);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    setJob(null);
    try {
      const { job_id } = await api.upload(file, compareAll);
      for (;;) {
        const current = await api.job(job_id);
        if (cancelled.current) return;
        setJob(current);
        if (FINAL_JOB_STATUSES.includes(current.status)) {
          if (current.status === "done" && current.invoice_id) onStored?.(current.invoice_id);
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
    } catch (exc) {
      setError(uploadError(exc));
    } finally {
      setBusy(false);
    }
  }

  const result = job?.result;
  const failedStep = job?.status === "failed" ? job.error?.split(":")[0] : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <form onSubmit={submit} className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
        <label htmlFor="invoice-file" className="block text-sm font-medium text-slate-700">
          Ảnh hoá đơn (JPEG/PNG, tối đa 10 MB)
        </label>
        <input id="invoice-file" type="file" accept="image/jpeg,image/png" onChange={choose} className="text-sm" />
        <label className="flex items-center gap-2 text-sm text-slate-700">
          <input type="checkbox" checked={compareAll} onChange={(e) => setCompareAll(e.target.checked)} />
          So sánh 4 làn (vanilla, static, adaptive, eagle)
        </label>
        <button
          type="submit"
          disabled={!file || busy}
          className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {busy ? "Đang xử lý…" : "Trích xuất"}
        </button>
        {error && (
          <p role="alert" className="text-sm text-red-600">
            {error}
          </p>
        )}
        {preview && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="Ảnh vừa chọn" className="max-h-96 w-full rounded border object-contain" />
        )}
      </form>

      <div className="space-y-4">
        {job && (
          <ol aria-label="Tiến độ" className="flex flex-wrap gap-2 text-xs">
            {STEPS.map((step) => {
              const reached = STEPS.indexOf(step) <= STEPS.indexOf(job.status as JobStatus);
              const failed = failedStep === step;
              return (
                <li
                  key={step}
                  data-testid={`step-${step}`}
                  data-state={failed ? "failed" : reached ? "reached" : "pending"}
                  className={`rounded px-2 py-1 ${failed ? "bg-red-100 text-red-700" : reached ? "bg-blue-100 text-blue-800" : "bg-slate-100 text-slate-500"}`}
                >
                  {STEP_LABELS[step]}
                  {failed ? " — lỗi" : ""}
                </li>
              );
            })}
          </ol>
        )}
        {job?.status === "failed" && (
          <p role="alert" className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {job.error}
          </p>
        )}
        {job?.status === "interrupted" && (
          <p role="alert" className="text-sm text-amber-700">Máy chủ khởi động lại giữa chừng; hãy tải lại ảnh.</p>
        )}
        {result?.lanes && <LaneComparison lanes={result.lanes} gpu={result.gpu} storedLane={result.stored_lane} />}
        {result?.stored_from_fallback && (
          <p className="text-sm text-amber-700">Làn chính không hợp lệ; hoá đơn được lưu từ làn vanilla.</p>
        )}
        {result?.grounding && result.grounding.ungrounded_fields.length > 0 && (
          <p role="status" className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            Không tìm thấy trong văn bản OCR: {result.grounding.ungrounded_fields.join(", ")}
          </p>
        )}
        {result?.invoice && (
          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-slate-900">
              Đã lưu hoá đơn {job?.invoice_id}
              {result.timings_ms && (
                <span className="ml-2 font-normal text-slate-500">
                  OCR {formatMs(result.timings_ms.ocr)} · Node 1 {formatMs(result.timings_ms.node1)}
                </span>
              )}
            </h3>
            <InvoiceView invoice={result.invoice} itemLines={result.item_check?.lines} />
          </section>
        )}
        {result?.ocr && (
          <details className="rounded-lg border border-slate-200 bg-white p-4" open={job?.status === "failed"}>
            <summary className="cursor-pointer text-sm font-semibold text-slate-900">
              Văn bản OCR{result.ocr.marks_stripped ? " · đã bỏ dấu" : ""}
            </summary>
            <pre data-testid="ocr-text" className="mt-2 whitespace-pre-wrap text-xs text-slate-700">
              {result.ocr.text}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}
