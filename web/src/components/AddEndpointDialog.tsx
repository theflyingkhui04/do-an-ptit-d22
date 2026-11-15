"use client";

import { CheckCircle2, Copy, FileJson, Plug, SearchCheck, X, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button, Notice, Spinner, secondsSince, useNow } from "@/components/ui";
import type { Api, DetectResult } from "@/lib/api";
import { formatMs } from "@/lib/format";

export const PLATFORM_LABELS: Record<string, string> = {
  modal: "Modal",
  tunnel: "Đường hầm (Colab hoặc máy cá nhân)",
  other: "Khác",
  local: "Máy cá nhân",
};

const EXAMPLE = "https://<tài-khoản>--flexsds-serving-serve.modal.run/v1";
// After this long a check is most likely waiting for a sleeping GPU to start.
export const SLOW_HINT_S = 10;

// One pasted address: checked on its own so a slow GPU does not hold back the others.
interface Check {
  input: string;
  done: boolean;
  ms?: number;
  result?: DetectResult;
}

function CopyLine({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2 rounded-lg bg-sunken px-3 py-2">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap font-mono text-[13px] text-ink">{text}</code>
      <button
        type="button"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
          } catch {
            setCopied(false);
          }
        }}
        className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-line bg-surface px-2 py-1 text-[12.5px] font-semibold text-ink"
      >
        <Copy aria-hidden className="size-3.5" /> {copied ? "Đã copy" : "Copy"}
      </button>
    </div>
  );
}

function Took({ ms }: { ms?: number }) {
  return ms === undefined ? null : <span className="num ml-auto shrink-0 pl-2 text-[12px] text-ink-3">{formatMs(ms)}</span>;
}

function RunningRow({ input }: { input: string }) {
  return (
    <li data-testid="check-running" className="flex animate-fade-in items-start gap-2.5 rounded-lg border border-dashed border-line px-3 py-2.5">
      <Spinner className="mt-0.5 text-ink-3" />
      <div className="grid min-w-0 gap-0.5">
        <span className="truncate font-mono text-[13px]">{input}</span>
        <span className="text-[13px] text-ink-3">Đang hỏi /v1/models…</span>
      </div>
    </li>
  );
}

function ResultRow({ result, ms }: { result: DetectResult; ms?: number }) {
  if (result.ok) {
    return (
      <li className="flex animate-fade-in items-start gap-2.5 rounded-lg border border-line px-3 py-2.5">
        <CheckCircle2 aria-hidden className="mt-0.5 size-4 shrink-0 text-good" />
        <div className="grid min-w-0 gap-0.5">
          <span className="truncate font-mono text-[13px]">{result.url}</span>
          <span className="text-[13px] text-ink-2">
            Model <b className="text-ink">{result.model_id}</b>, {PLATFORM_LABELS[result.platform ?? "other"]}
            {result.max_model_len ? `, ngữ cảnh ${result.max_model_len.toLocaleString("vi-VN")} token` : ""}
          </span>
        </div>
        <Took ms={ms} />
      </li>
    );
  }
  const command = result.code === "needs_tunnel" ? /`([^`]+)`/.exec(result.message ?? "")?.[1] : undefined;
  return (
    <li className="flex animate-fade-in items-start gap-2.5 rounded-lg border border-line px-3 py-2.5">
      {command ? (
        <Plug aria-hidden className="mt-0.5 size-4 shrink-0 text-warn" />
      ) : (
        <XCircle aria-hidden className="mt-0.5 size-4 shrink-0 text-bad" />
      )}
      <div className="grid min-w-0 flex-1 gap-1.5">
        <span className="truncate font-mono text-[13px]">{result.url ?? result.input}</span>
        {command ? (
          <>
            <span className="text-[13px] font-semibold text-warn">Cần mở đường hầm</span>
            <span className="text-[13px] text-ink-2">
              Web chạy trên mạng nên không gọi được địa chỉ trên máy của bạn. Chạy lệnh dưới đây trên máy đang chạy flexsds,
              rồi dán địa chỉ <code className="font-mono">https://….trycloudflare.com/v1</code> mà lệnh in ra.
            </span>
            <CopyLine text={command} />
          </>
        ) : (
          <span className="text-[13px] text-bad">{result.message}</span>
        )}
      </div>
      <Took ms={ms} />
    </li>
  );
}

function failed(input: string, message: string): DetectResult {
  return { input, url: null, ok: false, code: "request_failed", message, platform: null, model_id: null, max_model_len: null };
}

export default function AddEndpointDialog({ api, onClose, onSaved }: {
  api: Api;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [key, setKey] = useState("");
  const [config, setConfig] = useState<{ name: string; value: unknown } | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [busy, setBusy] = useState<"detect" | "add" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedCount, setSavedCount] = useState<number | null>(null);
  const first = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    first.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const urls = text.split(/\s+/).map((u) => u.trim()).filter(Boolean);
  const checking = checks?.some((c) => !c.done) ?? false;
  const now = useNow(checking);
  const waited = checking ? secondsSince(startedAt, now) : 0;
  const usable = (checks ?? []).flatMap((c) => (c.done && c.result?.ok ? [c.result] : []));

  async function detect() {
    const started = Date.now();
    const apiKey = key.trim() || undefined;
    setBusy("detect");
    setError(null);
    setSavedCount(null);
    setStartedAt(started);
    setChecks(urls.map((input) => ({ input, done: false })));
    const finish = (index: number, result: DetectResult) =>
      setChecks((current) => current && current.map((c, i) => (i === index ? { ...c, done: true, ms: Date.now() - started, result } : c)));
    await Promise.all(urls.map(async (input, index) => {
      try {
        const { results: found } = await api.detectEndpoints([input], apiKey);
        const result = found.find((r) => r.input === input) ?? found[0];
        finish(index, result ?? failed(input, "Máy chủ không trả kết quả cho địa chỉ này."));
      } catch (exc) {
        finish(index, failed(input, (exc as Error).message));
      }
    }));
    setBusy(null);
  }

  async function add() {
    setBusy("add");
    setError(null);
    try {
      const { results: checked, saved } = await api.addEndpoints(
        usable.map((r) => r.url ?? r.input),
        key.trim() || undefined,
        config?.value,
      );
      setChecks(checked.map((result) => ({ input: result.input, done: true, result })));
      setSavedCount(saved.length);
      await onSaved();
      if (checked.every((r) => r.ok)) onClose();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function readConfig(file: File | undefined) {
    setConfigError(null);
    if (!file) {
      setConfig(null);
      return;
    }
    try {
      const value = JSON.parse(await file.text());
      if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("not an object");
      setConfig({ name: file.name, value });
    } catch {
      setConfig(null);
      setConfigError("Tệp không phải JSON hợp lệ của serving_config.json.");
    }
  }

  return (
    <div className="fixed inset-0 z-30 grid place-items-start overflow-y-auto bg-[rgb(10_16_22/0.55)] px-4 py-10 sm:place-items-center">
      <div role="dialog" aria-modal="true" aria-labelledby="add-endpoint-title" className="grid w-full max-w-[640px] gap-5 rounded-2xl bg-surface p-6 text-[14.5px] text-ink">
        <div className="flex items-start gap-3">
          <div className="grid gap-1">
            <h2 id="add-endpoint-title" className="text-lg font-bold">Thêm endpoint</h2>
            <p className="text-ink-2">Dán địa chỉ ở dòng “API” mà lệnh <code className="font-mono">flexsds serve</code> in ra sau “Endpoint LLM created”. Dán nhiều địa chỉ thì mỗi dòng một địa chỉ.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Đóng" className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken">
            <X aria-hidden className="size-5" />
          </button>
        </div>

        <div className="grid gap-1.5">
          <label htmlFor="endpoint-urls" className="text-[13.5px] font-semibold">Endpoint URL</label>
          <textarea
            id="endpoint-urls"
            ref={first}
            rows={3}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setChecks(null);
            }}
            placeholder={EXAMPLE}
            className="rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[13px]"
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="grid content-start gap-1.5">
            <label htmlFor="endpoint-key" className="text-[13.5px] font-semibold">Khoá API (nếu có)</label>
            <input
              id="endpoint-key"
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              className="rounded-lg border border-line bg-surface px-3 py-2"
            />
            <small className="text-[12.5px] text-ink-3">Gửi dạng Bearer. Khoá chỉ lưu ở máy chủ, không hiện lại.</small>
          </div>
          <div className="grid content-start gap-1.5">
            <label htmlFor="endpoint-config" className="text-[13.5px] font-semibold">serving_config.json (tuỳ chọn)</label>
            <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-line px-3 py-2 text-ink-2 hover:bg-sunken">
              <FileJson aria-hidden className="size-4" />
              <span className="truncate">{config ? config.name : "Chọn tệp CLI tạo ra"}</span>
              <input id="endpoint-config" type="file" accept="application/json,.json" className="sr-only" onChange={(e) => readConfig(e.target.files?.[0])} />
            </label>
            <small className={configError ? "text-[12.5px] text-bad" : "text-[12.5px] text-ink-3"}>
              {configError ?? "Cho biết GPU, model nháp, có bật Adaptive hay không."}
            </small>
          </div>
        </div>

        {checks && checks.length > 0 && (
          <ul className="grid gap-2" aria-label="Kết quả kiểm tra" aria-live="polite">
            {checks.map((c, i) => (c.done && c.result
              ? <ResultRow key={`${c.input}-${i}`} result={c.result} ms={c.ms} />
              : <RunningRow key={`${c.input}-${i}`} input={c.input} />))}
          </ul>
        )}
        {checking && waited >= SLOW_HINT_S && (
          <Notice tone="info">
            GPU trên Modal đang ngủ có thể cần vài phút để khởi động. Mỗi lần kiểm tra, web chờ tối đa khoảng 1 phút;
            nếu báo chưa sẵn sàng, đợi 1–2 phút rồi bấm Kiểm tra lại.
          </Notice>
        )}
        {savedCount !== null && <Notice tone="info">Đã lưu {savedCount} endpoint.</Notice>}
        {error && <Notice tone="bad">{error}</Notice>}

        <div className="flex flex-wrap justify-end gap-2.5">
          <Button variant="ghost" onClick={onClose}>Huỷ</Button>
          {!checking && usable.length > 0 && savedCount === null ? (
            <Button icon={CheckCircle2} loading={busy === "add"} disabled={busy !== null} onClick={add}>
              {busy === "add" ? "Đang lưu…" : `Thêm ${usable.length} endpoint`}
            </Button>
          ) : (
            <Button icon={SearchCheck} loading={checking} disabled={busy !== null || urls.length === 0} onClick={detect}>
              {checking ? `Đang kiểm tra · ${waited} s` : "Kiểm tra"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
