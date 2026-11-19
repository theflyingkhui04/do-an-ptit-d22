"use client";

import { CheckCircle2, CircleCheck, KeyRound, Plug, RefreshCw, SearchCheck, XCircle } from "lucide-react";
import { useState } from "react";

import { Button, CopyButton, Drawer, Notice, Spinner, secondsSince, useNow } from "@/components/ui";
import type { Api, DetectResult } from "@/lib/api";
import { PLATFORM_LABELS } from "@/lib/endpoints";
import { formatMs } from "@/lib/format";

const EXAMPLE = "https://<tài-khoản>--flexsds-serving-serve.modal.run/v1";
// After this long a check is most likely waiting for a sleeping GPU to start.
export const SLOW_HINT_S = 10;

// One pasted address, checked on its own so a slow GPU does not hold back the others (design D14).
interface Check {
  input: string;
  started: number;
  ms?: number;
  result?: DetectResult;
  name?: string; // typed on its row once it answered
  key?: string; // typed on its row when the endpoint asked for one
  triedKey?: string; // the key sent with the last check
}

const usable = (c: Check) => c.result?.ok === true;
const skipped = (c: Check) => c.result !== undefined && !c.result.ok;

function failed(input: string, message: string): DetectResult {
  return { input, url: null, ok: false, code: "request_failed", message, platform: null, model_id: null, max_model_len: null };
}

function Took({ ms }: { ms?: number }) {
  return ms === undefined ? null : <span className="num ml-auto shrink-0 pl-2 text-[12px] text-ink-3">{formatMs(ms)}</span>;
}

function Row({ check, onName, onKey, onRecheck }: {
  check: Check;
  onName: (name: string) => void;
  onKey: (key: string) => void;
  onRecheck: () => void;
}) {
  const { input, result: r } = check;
  const shell = "flex animate-fade-in items-start gap-2.5 rounded-lg border border-line px-3 py-2.5";
  const address = <span className="truncate font-mono text-[13px]" title={input}>{input}</span>;
  if (!r) {
    return (
      <li data-testid="check-running" className={`${shell} border-dashed`}>
        <Spinner className="mt-0.5 text-ink-3" />
        <div className="grid min-w-0 gap-0.5">
          {address}
          <span className="text-[13px] text-ink-3">Đang hỏi /v1/models…</span>
        </div>
      </li>
    );
  }
  if (r.ok) {
    return (
      <li className={shell}>
        <CheckCircle2 aria-hidden className="mt-0.5 size-4 shrink-0 text-good" />
        <div className="grid min-w-0 flex-1 gap-1.5">
          {address}
          <span className="text-[13px] text-ink-2">
            Model <b className="text-ink">{r.model_id}</b>, {PLATFORM_LABELS[r.platform ?? "other"]}
            {r.max_model_len ? `, ngữ cảnh ${r.max_model_len.toLocaleString("vi-VN")} token` : ""}
            {check.triedKey ? ", có khoá API" : ""}
          </span>
          <label className="grid gap-1 text-[13px] font-semibold text-ink-2">
            Tên hiển thị
            <input
              aria-label={`Tên hiển thị cho ${input}`}
              value={check.name ?? r.model_id ?? ""}
              maxLength={80}
              onChange={(e) => onName(e.target.value)}
              className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[13.5px] font-normal text-ink"
            />
          </label>
        </div>
        <Took ms={check.ms} />
      </li>
    );
  }
  if (r.code === "exists") {
    return (
      <li className={shell}>
        <CircleCheck aria-hidden className="mt-0.5 size-4 shrink-0 text-ink-2" />
        <div className="grid min-w-0 gap-0.5">
          {address}
          <span className="text-[13px] text-ink-2">{r.message}</span>
        </div>
        <Took ms={check.ms} />
      </li>
    );
  }
  if (r.code === "auth_failed") {
    const status = /HTTP \d+/.exec(r.message ?? "")?.[0];
    return (
      <li className={shell}>
        <KeyRound aria-hidden className="mt-0.5 size-4 shrink-0 text-warn" />
        <div className="grid min-w-0 flex-1 gap-1.5">
          {address}
          <span className="text-[13px] font-semibold text-warn">
            {check.triedKey ? "Khoá API không đúng" : "Endpoint này cần khoá API"}{status ? ` (${status})` : ""}
          </span>
          <span className="text-[13px] text-ink-2">
            Nhập khoá đã đặt khi chạy vLLM với <code className="font-mono">--api-key</code>. Khoá chỉ lưu ở máy chủ, không hiện lại.
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="password"
              autoComplete="off"
              aria-label={`Khoá API cho ${input}`}
              placeholder="Khoá API"
              value={check.key ?? ""}
              onChange={(e) => onKey(e.target.value)}
              className="min-w-0 flex-[1_1_160px] rounded-lg border border-line bg-surface px-2.5 py-1.5"
            />
            <Button variant="ghost" icon={RefreshCw} className="px-2.5 py-1.5 text-[13px]" disabled={!check.key?.trim()} onClick={onRecheck}>
              Kiểm tra lại với khoá này
            </Button>
          </div>
        </div>
        <Took ms={check.ms} />
      </li>
    );
  }
  const command = r.code === "needs_tunnel" ? /`([^`]+)`/.exec(r.message ?? "")?.[1] : undefined;
  if (command) {
    return (
      <li className={shell}>
        <Plug aria-hidden className="mt-0.5 size-4 shrink-0 text-warn" />
        <div className="grid min-w-0 flex-1 gap-1.5">
          {address}
          <span className="text-[13px] font-semibold text-warn">Cần mở đường hầm</span>
          <span className="text-[13px] text-ink-2">
            Web chạy trên mạng nên không gọi được địa chỉ trên máy của bạn. Chạy lệnh dưới đây trên máy đang chạy flexsds,
            rồi dán địa chỉ <code className="font-mono">https://….trycloudflare.com/v1</code> mà lệnh in ra.
          </span>
          <div className="flex items-center gap-2 rounded-lg bg-sunken px-3 py-2">
            <code className="min-w-0 flex-1 break-all font-mono text-[13px] text-ink">{command}</code>
            <CopyButton text={command} label="Copy lệnh" />
          </div>
        </div>
        <Took ms={check.ms} />
      </li>
    );
  }
  return (
    <li className={shell}>
      <XCircle aria-hidden className="mt-0.5 size-4 shrink-0 text-bad" />
      <div className="grid min-w-0 flex-1 gap-1.5">
        {address}
        <span className="text-[13px] text-bad">{r.message}</span>
        {r.code === "not_ready" && (
          <Button variant="ghost" icon={RefreshCw} className="w-max px-2.5 py-1.5 text-[13px]" onClick={onRecheck}>
            Kiểm tra lại địa chỉ này
          </Button>
        )}
      </div>
      <Took ms={check.ms} />
    </li>
  );
}

export default function AddEndpointDrawer({ api, onClose, onSaved }: {
  api: Api;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // One endpoint per add (Canva review 19/11). A pasted "- API: https://…" line keeps only the address.
  const url = text.trim().split(/\s+/).find((t) => /^https?:\/\//i.test(t)) ?? text.trim();
  const urls = url ? [url] : [];
  const running = (checks ?? []).filter((c) => !c.result);
  const busy = running.length > 0;
  const now = useNow(busy);
  const waited = busy ? secondsSince(Math.min(...running.map((c) => c.started)), now) : 0;
  const ready = (checks ?? []).filter(usable);
  const known = (checks ?? []).filter((c) => c.result?.code === "exists").length;
  const bad = (checks ?? []).filter(skipped).length - known;

  const patch = (input: string, change: Partial<Check>) =>
    setChecks((current) => current && current.map((c) => (c.input === input ? { ...c, ...change } : c)));

  async function run(input: string, key: string | undefined) {
    const started = Date.now();
    patch(input, { started, result: undefined, ms: undefined, triedKey: key?.trim() || undefined });
    let result: DetectResult;
    try {
      const { results } = await api.detectEndpoints([input], key?.trim() || undefined);
      result = results.find((r) => r.input === input) ?? results[0] ?? failed(input, "Máy chủ không trả kết quả cho địa chỉ này.");
    } catch (exc) {
      result = failed(input, (exc as Error).message);
    }
    patch(input, { result, ms: Date.now() - started });
  }

  function checkAll() {
    setError(null);
    const before = new Map((checks ?? []).map((c) => [c.input, c]));
    const next = urls.map((input) => ({
      input, started: Date.now(), name: before.get(input)?.name, key: before.get(input)?.key,
    }));
    setChecks(next);
    void Promise.all(next.map((c) => run(c.input, c.key)));
  }

  async function add() {
    setSaving(true);
    setError(null);
    try {
      const items = ready.map((c) => ({
        url: c.result?.url ?? c.input, name: c.name?.trim() || undefined, api_key: c.triedKey,
      }));
      const { results, saved } = await api.addEndpoints(items);
      await onSaved();
      if (saved.length === items.length) {
        onClose();
        return;
      }
      // Something changed since the check (a GPU went to sleep, the limit was reached): show why on its row.
      setChecks((current) => current && current.map((c) => {
        const now = results.find((r) => r.input === (c.result?.url ?? c.input));
        return now && !now.ok ? { ...c, result: now } : c;
      }));
      setError(`Đã thêm ${saved.length}/${items.length} endpoint; xem lý do ở các dòng còn lại.`);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const skip = [known && `${known} địa chỉ đã có`, bad && `${bad} địa chỉ lỗi`].filter(Boolean).join(", ");
  const main = !busy && ready.length > 0
    ? <Button icon={CheckCircle2} loading={saving} disabled={saving} onClick={add}>{saving ? "Đang thêm" : `Thêm ${ready.length} endpoint`}</Button>
    : (
      <Button icon={SearchCheck} loading={busy} disabled={busy || urls.length === 0} onClick={checkAll}>
        {busy ? `Đang kiểm tra, ${waited} s` : checks?.length ? "Kiểm tra lại" : "Kiểm tra"}
      </Button>
    );

  return (
    <Drawer
      title="Thêm endpoint"
      onClose={onClose}
      footer={(
        <>
          {!busy && ready.length > 0 && skip && <span className="mr-auto text-[12.5px] text-ink-3">Bỏ qua {skip}</span>}
          <Button variant="ghost" onClick={onClose}>Huỷ</Button>
          {main}
        </>
      )}
    >
      <div className="grid content-start gap-4">
        <div className="grid gap-1.5">
          <label htmlFor="endpoint-urls" className="text-[13.5px] font-semibold">Địa chỉ API</label>
          <input
            id="endpoint-urls"
            type="text"
            inputMode="url"
            autoFocus
            spellCheck={false}
            autoComplete="off"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setChecks(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy && urls.length > 0) {
                e.preventDefault();
                checkAll();
              }
            }}
            placeholder={EXAMPLE}
            className="w-full min-w-0 rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[13px]"
          />
        </div>

        {checks && checks.length > 0 && (
          <ul className="grid gap-2" aria-label="Kết quả kiểm tra" aria-live="polite">
            {checks.map((c) => (
              <Row
                key={c.input}
                check={c}
                onName={(name) => patch(c.input, { name })}
                onKey={(key) => patch(c.input, { key })}
                onRecheck={() => void run(c.input, c.key)}
              />
            ))}
          </ul>
        )}
        {busy && waited >= SLOW_HINT_S && (
          <Notice tone="info">
            GPU trên Modal đang ngủ có thể cần vài phút để khởi động. Mỗi lần kiểm tra, web chờ tối đa khoảng 1 phút;
            nếu báo chưa sẵn sàng, đợi 1–2 phút rồi bấm Kiểm tra lại.
          </Notice>
        )}
        {error && <Notice tone="bad">{error}</Notice>}
      </div>
    </Drawer>
  );
}
