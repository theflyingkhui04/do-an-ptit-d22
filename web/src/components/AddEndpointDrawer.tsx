"use client";

import { CheckCircle2, CircleCheck, KeyRound, Plug, RefreshCw, SearchCheck, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button, CopyButton, Drawer, Notice, Spinner, secondsSince, useNow } from "@/components/ui";
import type { Api, DetectResult, Endpoint } from "@/lib/api";
import { nextAlias, platformLabel } from "@/lib/endpoints";
import { formatMs, localeTag } from "@/lib/format";
import { detectErrorText, useI18n } from "@/lib/i18n";

export const SLOW_HINT_S = 10;

interface Check {
  input: string;
  started: number;
  ms?: number;
  result?: DetectResult;
  name?: string;
  key?: string;
  triedKey?: string;
}

const usable = (c: Check) => c.result?.ok === true;

function failed(input: string, message: string): DetectResult {
  return { input, url: null, ok: false, code: "request_failed", message, platform: null, model_id: null, max_model_len: null };
}

function Took({ ms }: { ms?: number }) {
  const { lang } = useI18n();
  return ms === undefined ? null : <span className="num ml-auto shrink-0 pl-2 text-[12px] text-ink-3">{formatMs(ms, lang)}</span>;
}

function Row({ check, suggestedAlias, onName, onKey, onRecheck }: {
  check: Check;
  suggestedAlias: string;
  onName: (name: string) => void;
  onKey: (key: string) => void;
  onRecheck: () => void;
}) {
  const { lang, t } = useI18n();
  const { input, result: r } = check;
  const shell = "flex animate-fade-in items-start gap-2.5 rounded-lg border border-line px-3 py-2.5";
  const address = <span className="truncate font-mono text-[13px]" title={input}>{input}</span>;
  if (!r) {
    return (
      <li data-testid="check-running" className={`${shell} border-dashed`}>
        <Spinner className="mt-0.5 text-ink-3" />
        <div className="grid min-w-0 gap-0.5">
          {address}
          <span className="text-[13px] text-ink-3">{t.checkingModels}</span>
        </div>
      </li>
    );
  }
  if (r.ok) {
    return (
      <li className={shell}>
        <CheckCircle2 aria-hidden className="mt-0.5 size-4 shrink-0 text-good" />
        <div className="grid min-w-0 flex-1 gap-2">
          {address}
          <span className="text-[12.5px] text-ink-2">
            <b className="text-ink">{r.model_id}</b> · {platformLabel(r.platform ?? "other", lang)}
            {r.max_model_len ? ` · ${t.contextTokens(r.max_model_len.toLocaleString(localeTag(lang)))}` : ""}
            {check.triedKey ? ` · ${t.hasApiKey}` : ""}
          </span>
          <label className="grid gap-1 text-[13px] font-semibold text-ink-2">
            <span>{t.aliasLabel}</span>
            <input
              aria-label={t.aliasAria(input)}
              placeholder={suggestedAlias}
              value={check.name ?? suggestedAlias}
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
          <span className="text-[13px] text-ink-2">{detectErrorText(lang, r.code, r.message, r.name)}</span>
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
            {check.triedKey ? t.keyInvalid : t.keyRequired}{status ? ` (${status})` : ""}
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="password"
              autoComplete="off"
              aria-label={t.apiKeyAria(input)}
              placeholder={t.apiKeyLabel}
              value={check.key ?? ""}
              onChange={(e) => onKey(e.target.value)}
              className="min-w-0 flex-[1_1_160px] rounded-lg border border-line bg-surface px-2.5 py-1.5"
            />
            <Button variant="ghost" icon={RefreshCw} className="px-2.5 py-1.5 text-[13px]" disabled={!check.key?.trim()} onClick={onRecheck}>
              {t.recheckWithKey}
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
          <span className="text-[13px] font-semibold text-warn">{t.tunnelNeededTitle}</span>
          <span className="text-[13px] text-ink-2">{t.tunnelNeededBody}</span>
          <div className="flex items-center gap-2 rounded-lg bg-sunken px-3 py-2">
            <code className="min-w-0 flex-1 break-all font-mono text-[13px] text-ink">{command}</code>
            <CopyButton text={command} label={t.copyCommand} />
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
        <span className="text-[13px] text-bad">{detectErrorText(lang, r.code, r.message)}</span>
        {r.code === "not_ready" && (
          <Button variant="ghost" icon={RefreshCw} className="w-max px-2.5 py-1.5 text-[13px]" onClick={onRecheck}>
            {t.recheckAddress}
          </Button>
        )}
      </div>
      <Took ms={check.ms} />
    </li>
  );
}

export default function AddEndpointDrawer({ api, existingEndpoints = [], onClose, onSaved }: {
  api: Api;
  existingEndpoints?: Endpoint[];
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const { t } = useI18n();
  const suggestedAlias = nextAlias(existingEndpoints);
  const [text, setText] = useState("");
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => field.current?.focus(), []);
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const url = text.trim().split(/\s+/).find((tok) => /^https?:\/\//i.test(tok)) ?? text.trim();
  const urls = url ? [url] : [];
  const running = (checks ?? []).filter((c) => !c.result);
  const busy = running.length > 0;
  const now = useNow(busy);
  const waited = busy ? secondsSince(Math.min(...running.map((c) => c.started)), now) : 0;
  const ready = (checks ?? []).filter(usable);

  const patch = (input: string, change: Partial<Check>) =>
    setChecks((current) => current && current.map((c) => (c.input === input ? { ...c, ...change } : c)));

  async function run(input: string, key: string | undefined) {
    const started = Date.now();
    patch(input, { started, result: undefined, ms: undefined, triedKey: key?.trim() || undefined });
    let result: DetectResult;
    try {
      const { results } = await api.detectEndpoints([input], key?.trim() || undefined);
      result = results.find((r) => r.input === input) ?? results[0] ?? failed(input, t.noServerResult);
    } catch (exc) {
      result = failed(input, (exc as Error).message);
    }
    patch(input, { result, ms: Date.now() - started });
  }

  function checkAll() {
    setError(null);
    const before = new Map((checks ?? []).map((c) => [c.input, c]));
    const next = urls.map((input) => ({
      input,
      started: Date.now(),
      name: before.get(input)?.name,
      key: before.get(input)?.key,
    }));
    setChecks(next);
    void Promise.all(next.map((c) => run(c.input, c.key)));
  }

  async function add() {
    setSaving(true);
    setError(null);
    try {
      const items = ready.map((c) => ({
        url: c.result?.url ?? c.input,
        name: (c.name ?? suggestedAlias).trim() || suggestedAlias,
        api_key: c.triedKey,
      }));
      const { results, saved } = await api.addEndpoints(items);
      await onSaved();
      if (saved.length === items.length) {
        onClose();
        return;
      }
      setChecks((current) => current && current.map((c) => {
        const latest = results.find((r) => r.input === (c.result?.url ?? c.input));
        return latest && !latest.ok ? { ...c, result: latest } : c;
      }));
      setError(t.partialAddError(saved.length, items.length));
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const main = !busy && ready.length > 0
    ? <Button icon={CheckCircle2} loading={saving} disabled={saving} onClick={add}>{saving ? t.addingBtn : t.addBtn}</Button>
    : (
      <Button icon={SearchCheck} loading={busy} disabled={busy || urls.length === 0} onClick={checkAll}>
        {busy ? t.checkingElapsed(waited) : checks?.length ? t.checkAgain : t.checkBtn}
      </Button>
    );

  return (
    <Drawer
      title={t.addEndpoint}
      onClose={onClose}
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>{t.cancel}</Button>
          {main}
        </>
      )}
    >
      <div className="grid content-start gap-4">
        <div className="grid gap-1.5">
          <label htmlFor="endpoint-urls" className="text-[13.5px] font-semibold">{t.apiUrlLabel}</label>
          <input
            id="endpoint-urls"
            ref={field}
            type="text"
            inputMode="url"
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
            placeholder={t.apiUrlExample}
            className="w-full min-w-0 rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[13px]"
          />
        </div>

        {checks && checks.length > 0 && (
          <ul className="grid gap-2" aria-label={t.checkResultsAria} aria-live="polite">
            {checks.map((c) => (
              <Row
                key={c.input}
                check={c}
                suggestedAlias={suggestedAlias}
                onName={(name) => patch(c.input, { name })}
                onKey={(key) => patch(c.input, { key })}
                onRecheck={() => void run(c.input, c.key)}
              />
            ))}
          </ul>
        )}
        {busy && waited >= SLOW_HINT_S && (
          <Notice tone="info">{t.slowGpuNotice}</Notice>
        )}
        {error && <Notice tone="bad">{error}</Notice>}
      </div>
    </Drawer>
  );
}
