"use client";

import { Check, Copy, Languages, Loader2, X, type LucideIcon } from "lucide-react";
import Image from "next/image";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import type { Endpoint } from "@/lib/api";
import { formatCell, formatMs, localeTag } from "@/lib/format";
import { useI18n, type Lang } from "@/lib/i18n";

export function BrandMark({ size, className }: { size: number; className?: string }) {
  return (
    <Image src="/brand/flexsds-icon-light.png" alt="" width={size} height={size} unoptimized priority
      className={cx("rounded-[13%]", className)} />
  );
}

export function cx(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ");
}

export function LanguageSwitcher({ className }: { className?: string }) {
  const { lang, setLang, t } = useI18n();
  return (
    <div
      role="group"
      aria-label={t.langToggleLabel}
      className={cx("inline-flex items-center rounded-lg border border-line bg-sunken p-0.5 text-[12.5px]", className)}
    >
      <Languages aria-hidden className="mx-1.5 size-3.5 text-ink-3" />
      {(["vi", "en"] as Lang[]).map((code) => (
        <button
          key={code}
          type="button"
          aria-pressed={lang === code}
          onClick={() => setLang(code)}
          className={cx(
            "rounded-md px-2 py-0.5 font-semibold uppercase transition-colors",
            lang === code ? "bg-surface text-ink shadow-sm" : "text-ink-3 hover:text-ink",
          )}
        >
          {code}
        </button>
      ))}
    </div>
  );
}

export function ReferencePicker({ options, value, onChange, colors }: {
  options: { id: string; name: string }[];
  value: string;
  onChange: (id: string) => void;
  colors?: Record<string, string>;
}) {
  const { t } = useI18n();
  const id = useId();
  if (options.length < 2) return null;
  return (
    <label htmlFor={id} className="inline-flex items-center gap-2 text-[13px] text-ink-3">
      <span>{t.compareAgainst}</span>
      <span className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1 text-ink">
        {colors?.[value] && <Dot color={colors[value]} />}
        <select
          id={id}
          aria-label={t.compareAgainst}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="cursor-pointer bg-transparent pr-1 text-[13px] font-semibold text-ink outline-none"
        >
          {options.map((opt) => (
            <option key={opt.id} value={opt.id}>{opt.name}</option>
          ))}
        </select>
      </span>
    </label>
  );
}

export function Panel({ title, icon: Icon, meta, children, className, bodyClassName }: {
  title?: ReactNode;
  icon?: LucideIcon;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("min-w-0 rounded-xl border border-line bg-surface", className)}>
      {title && (
        <header className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-line px-[18px] py-3.5">
          {Icon && <Icon aria-hidden className="size-[18px] text-ink-2" />}
          <h2 className="text-[15px] font-semibold">{title}</h2>
          {meta && <div className="ml-auto flex flex-wrap items-center gap-3 text-[13px] text-ink-3">{meta}</div>}
        </header>
      )}
      <div className={cx("grid gap-4 p-[18px]", bodyClassName)}>{children}</div>
    </section>
  );
}

type Tone = "good" | "warn" | "bad" | "idle";
const PILL: Record<Tone, string> = {
  good: "bg-good-tint text-good",
  warn: "bg-warn-tint text-warn",
  bad: "bg-bad-tint text-bad",
  idle: "bg-sunken text-ink-2",
};

export function Pill({ tone = "idle", icon: Icon, children }: { tone?: Tone; icon?: LucideIcon; children: ReactNode }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12.5px] font-semibold", PILL[tone])}>
      {Icon && <Icon aria-hidden className="size-3.5" />}
      {children}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 aria-hidden data-testid="spinner" className={cx("size-4 shrink-0 animate-spin", className)} />;
}

export function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(timer);
  }, [active, everyMs]);
  return now;
}

export function secondsSince(start: number, now: number): number {
  return Math.max(0, Math.floor((now - start) / 1000));
}

export function clockTime(date: Date, lang: Lang = "vi"): string {
  return date.toLocaleTimeString(localeTag(lang), { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

export function Button({ variant = "primary", icon: Icon, loading = false, children, className, ...props }: {
  variant?: "primary" | "ghost" | "danger" | "danger-solid";
  icon?: LucideIcon;
  loading?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const look = {
    primary: "bg-ink text-surface hover:opacity-90",
    ghost: "border border-line bg-surface text-ink hover:bg-sunken",
    danger: "border border-line bg-surface text-bad hover:bg-bad-tint",
    "danger-solid": "bg-bad text-white hover:opacity-90",
  }[variant];
  return (
    <button
      type="button"
      aria-busy={loading || undefined}
      {...props}
      className={cx("inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50", loading && "disabled:cursor-progress", look, className)}
    >
      {loading ? <Spinner /> : Icon && <Icon aria-hidden className="size-4" />}
      {children}
    </button>
  );
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return (
    <button
      type="button"
      aria-label={copied ? t.copied : label}
      title={copied ? t.copied : label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
        } catch {
          setCopied(false);
        }
      }}
      className="grid size-6 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-sunken hover:text-ink"
    >
      {copied ? <Check aria-hidden className="size-3.5 text-good" /> : <Copy aria-hidden className="size-3.5" />}
    </button>
  );
}

export function Drawer({ title, aside, onClose, footer, children, paused = false }: {
  title: string;
  aside?: ReactNode;
  onClose: () => void;
  footer?: ReactNode;
  children: ReactNode;
  paused?: boolean;
}) {
  const { t } = useI18n();
  const titleId = useId();
  const close = useRef(onClose);
  const hold = useRef(paused);
  useEffect(() => {
    close.current = onClose;
    hold.current = paused;
  });
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !hold.current) close.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, []);
  return (
    <>
      <div aria-hidden className="fixed inset-0 z-30 bg-[rgb(10_16_22/0.45)]" onClick={() => !hold.current && close.current()} />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[560px] animate-fade-in flex-col border-l border-line bg-surface text-[14px] text-ink shadow-[-12px_0_32px_rgb(0_0_0/0.18)]"
      >
        <header className="flex items-center gap-2.5 border-b border-line px-5 py-3.5">
          <h2 id={titleId} className="text-base font-bold">{title}</h2>
          {aside}
          <button type="button" onClick={onClose} aria-label={t.close} className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken">
            <X aria-hidden className="size-5" />
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-[18px]">{children}</div>
        {footer && <footer className="flex flex-wrap items-center justify-end gap-2.5 border-t border-line px-5 py-3">{footer}</footer>}
      </aside>
    </>
  );
}

export function ConfirmDialog({ title, children, confirmLabel, confirmIcon, busy = false, onConfirm, onCancel }: {
  title: string;
  children: ReactNode;
  confirmLabel: string;
  confirmIcon?: LucideIcon;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const titleId = useId();
  const keep = useRef<HTMLButtonElement>(null);
  const cancel = useRef(onCancel);
  useEffect(() => {
    cancel.current = onCancel;
  });
  useEffect(() => {
    keep.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") cancel.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <>
      <div aria-hidden className="fixed inset-0 z-50 bg-[rgb(10_16_22/0.45)]" onClick={onCancel} />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="fixed left-1/2 top-1/2 z-[60] w-[min(460px,calc(100vw-32px))] -translate-x-1/2 -translate-y-1/2 animate-fade-in rounded-2xl bg-surface text-ink shadow-[0_20px_48px_rgb(0_0_0/0.25)]"
      >
        <header className="border-b border-line px-5 py-3.5">
          <h2 id={titleId} className="text-base font-bold">{title}</h2>
        </header>
        <div className="grid gap-2.5 px-5 py-4 text-[14px]">{children}</div>
        <footer className="flex flex-wrap justify-end gap-2.5 border-t border-line px-5 py-3">
          <button
            ref={keep}
            type="button"
            onClick={onCancel}
            className="inline-flex items-center justify-center rounded-lg border border-line bg-surface px-3.5 py-2 text-sm font-semibold text-ink hover:bg-sunken"
          >
            {t.cancel}
          </button>
          <Button variant="danger-solid" icon={confirmIcon} loading={busy} disabled={busy} onClick={onConfirm}>{confirmLabel}</Button>
        </footer>
      </div>
    </>
  );
}

export function Notice({ tone = "info", icon: Icon, children }: { tone?: "info" | "warn" | "bad"; icon?: LucideIcon; children: ReactNode }) {
  const look = { info: "bg-sunken text-ink-2", warn: "bg-warn-tint text-warn", bad: "bg-bad-tint text-bad" }[tone];
  return (
    <div role={tone === "bad" ? "alert" : "status"} className={cx("flex items-start gap-2.5 rounded-xl px-3.5 py-3 text-[13.5px]", look)}>
      {Icon && <Icon aria-hidden className="mt-0.5 size-4 shrink-0" />}
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action }: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="grid justify-items-start gap-2.5 rounded-xl border border-line bg-surface p-7">
      <div className="grid size-11 place-items-center rounded-xl bg-sunken">
        <Icon aria-hidden className="size-5 text-ink-2" />
      </div>
      <h2 className="text-[17px] font-bold">{title}</h2>
      {children && <div className="grid w-full text-ink-2 [&>p]:max-w-[62ch]">{children}</div>}
      {action}
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="grid gap-0.5 rounded-xl border border-line bg-surface px-3.5 py-3">
      <span className="text-[13px] text-ink-3">{label}</span>
      <b className={cx("num text-xl font-bold", tone === "good" && "text-good", tone === "bad" && "text-bad")}>{value}</b>
      {sub && <small className="text-[12.5px] text-ink-3">{sub}</small>}
    </div>
  );
}

export function PageHead({ title, children, actions }: { title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
      <div className="grid max-w-[70ch] gap-1">
        <h1 className="text-[22px] font-bold leading-tight">{title}</h1>
        {children && <p className="text-ink-2">{children}</p>}
      </div>
      {actions && <div className="ml-auto flex flex-wrap items-center gap-2.5">{actions}</div>}
    </div>
  );
}

export interface TraceStep {
  label: string;
  ms: number | null | undefined;
  kind: "io" | "n1" | "n2" | "n3";
  title?: string;
}
const STEP_COLOR = { io: "var(--step-io)", n1: "var(--step-1)", n2: "var(--step-2)", n3: "var(--step-3)" };

export function Trace({ steps, extra }: { steps: TraceStep[]; extra?: ReactNode }) {
  const { lang, t } = useI18n();
  const shown = steps.filter((s) => s.ms !== null && s.ms !== undefined) as (TraceStep & { ms: number })[];
  const total = shown.reduce((sum, s) => sum + s.ms, 0);
  return (
    <div className="grid gap-2">
      <div className="flex h-[30px] overflow-hidden rounded-lg bg-sunken" role="img" aria-label={shown.map((s) => `${s.label} ${formatMs(s.ms, lang)}`).join(", ")}>
        {shown.map((s, i) => (
          <div
            key={s.label}
            title={`${s.title ?? s.label}: ${formatMs(s.ms, lang)}`}
            className={cx("flex min-w-[42px] items-center overflow-hidden whitespace-nowrap px-2.5 text-[12.5px] font-semibold text-white", i > 0 && "border-l-2 border-surface")}
            style={{ flex: Math.max(s.ms, total * 0.03), background: STEP_COLOR[s.kind] }}
          >
            {s.ms >= total * 0.08 && s.label}
            {s.ms >= total * 0.12 && <span className="ml-1.5 font-normal">{formatMs(s.ms, lang)}</span>}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-[18px] gap-y-1.5 text-[13px] text-ink-2">
        {shown.map((s) => (
          <span key={s.label}>{s.title ?? s.label} <b className="num text-ink">{formatMs(s.ms, lang)}</b></span>
        ))}
        <span>{t.total} <b className="num text-ink">{formatMs(total, lang)}</b></span>
        {extra}
      </div>
    </div>
  );
}

export function baseModel(e: Endpoint): string {
  return e.meta?.target_model || e.meta?.base_model || e.model_id;
}

// Distinct validated colours assigned in endpoint order (no special grey baseline lock).
export function endpointColors(endpoints: Endpoint[]): Record<string, string> {
  const colors: Record<string, string> = {};
  const slots = ["var(--lane-1)", "var(--lane-2)", "var(--lane-3)", "var(--lane-ref)"];
  endpoints.forEach((e, i) => {
    colors[e.endpoint_id] = slots[i % slots.length] ?? "var(--ink-2)";
  });
  return colors;
}

export function Dot({ color, className }: { color: string; className?: string }) {
  return <i aria-hidden className={cx("inline-block size-2.5 shrink-0 rounded-full", className)} style={{ background: color }} />;
}

export function EndpointChips({ endpoints, selected, onToggle, single }: {
  endpoints: Endpoint[];
  selected: string[];
  onToggle: (endpointId: string) => void;
  single?: boolean;
}) {
  const { t } = useI18n();
  const colors = endpointColors(endpoints);
  return (
    <div className="flex flex-wrap gap-2" role={single ? "radiogroup" : "group"} aria-label={t.selectLlm}>
      {endpoints.map((e) => {
        const on = selected.includes(e.endpoint_id) && e.status !== "down";
        return (
          <button
            key={e.endpoint_id}
            type="button"
            role={single ? "radio" : "checkbox"}
            aria-checked={on}
            disabled={e.status === "down"}
            onClick={() => onToggle(e.endpoint_id)}
            title={e.status === "down" ? (e.status_message ?? t.endpointOffline) : e.base_url}
            className={cx(
              "inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1.5 text-[13.5px]",
              e.status === "down" && "cursor-not-allowed text-ink-3",
              on ? "border-ink font-semibold shadow-[0_0_0_1px_var(--ink)]" : "border-line bg-surface",
            )}
          >
            <Dot color={e.status === "down" ? "var(--bad)" : colors[e.endpoint_id]} />
            <span className="truncate">{e.name}</span>
            {e.status === "down" && <span className="sr-only">{t.offlineSr}</span>}
          </button>
        );
      })}
    </div>
  );
}

export function CodeBlock({ children, label }: { children: string; label?: string }) {
  return (
    <pre aria-label={label} className="max-h-[440px] overflow-auto whitespace-pre rounded-lg bg-sunken px-4 py-3.5 font-mono text-[13px] leading-relaxed">
      {children}
    </pre>
  );
}

export function NeedEndpoint({ onGo, what }: { onGo: () => void; what: string }) {
  const { t } = useI18n();
  return (
    <div className="grid justify-items-start gap-2.5 rounded-xl border border-line bg-surface p-7">
      <h2 className="text-[17px] font-bold">{t.needEndpointTitle}</h2>
      <p className="max-w-[62ch] text-ink-2">{t.needEndpointBody(what)}</p>
      <button type="button" onClick={onGo} className="rounded-lg bg-ink px-3.5 py-2 text-sm font-semibold text-surface">{t.openEndpoints}</button>
    </div>
  );
}

// Default selection: up to 2 active endpoints (no baseline locking).
export function defaultTarget(endpoints: Endpoint[], limit = 2): string[] {
  const live = endpoints.filter((e) => e.status === "ok");
  const pool = live.length > 0 ? live : endpoints.filter((e) => e.status !== "down");
  return pool.slice(0, limit).map((e) => e.endpoint_id);
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: {
  tabs: [T, string][];
  value: T;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex flex-wrap gap-1 border-b border-line">
      {tabs.map(([id, text]) => (
        <button
          key={id}
          type="button"
          role="tab"
          aria-selected={value === id}
          onClick={() => onChange(id)}
          className={cx(
            "-mb-px border-b-2 px-3 py-2 text-[13.5px]",
            value === id ? "border-ink font-semibold text-ink" : "border-transparent text-ink-3 hover:text-ink",
          )}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

export function ResultTable({ columns, rows, max = 50, flag, differs }: {
  columns: string[];
  rows: unknown[][];
  max?: number;
  flag?: (row: unknown[]) => string | null;
  differs?: (index: number) => boolean;
}) {
  const { lang, t } = useI18n();
  if (rows.length === 0) return <p className="text-ink-3">{t.noRows}</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="w-full text-left text-[13.5px]" aria-label={t.sqlResult}>
        <thead className="bg-sunken text-ink-3">
          <tr>
            {columns.map((c) => <th key={c} className="px-3 py-2 font-mono text-[12.5px] font-semibold">{c}</th>)}
            {flag && <th className="px-3 py-2" />}
          </tr>
        </thead>
        <tbody className="num">
          {rows.slice(0, max).map((row, i) => {
            const note = flag?.(row) ?? null;
            return (
              <tr key={i} data-diff={differs?.(i) || undefined} className={cx("border-t border-line", differs?.(i) && "bg-warn-tint")}>
                {row.map((cell, j) => (
                  <td key={j} className={cx("px-3 py-1.5", typeof cell === "number" && "text-right")}>{formatCell(cell, lang)}</td>
                ))}
                {flag && <td className="px-3 py-1.5">{note && <Pill tone="warn">{note}</Pill>}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > max && <p className="border-t border-line px-3 py-2 text-[12.5px] text-ink-3">{t.showingRows(max, rows.length)}</p>}
    </div>
  );
}

const SQL_KEYWORDS = /\b(SELECT|FROM|WHERE|GROUP|BY|ORDER|LIMIT|AS|AND|OR|NOT|IN|IS|NULL|JOIN|LEFT|INNER|ON|HAVING|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|DESC|ASC|LIKE|ILIKE|BETWEEN|WITH|ROUND|COALESCE|EXTRACT|CAST|DATE_TRUNC|LOWER|UPPER)\b/gi;

export function SqlBlock({ sql }: { sql: string }) {
  const { t } = useI18n();
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of sql.matchAll(SQL_KEYWORDS)) {
    const at = match.index ?? 0;
    if (at > last) parts.push(sql.slice(last, at));
    parts.push(<b key={at} className="font-semibold text-lane-1">{match[0]}</b>);
    last = at + match[0].length;
  }
  parts.push(sql.slice(last));
  return (
    <pre aria-label={t.sqlQuery} className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-sunken px-4 py-3.5 font-mono text-[13px] leading-relaxed">
      {parts}
    </pre>
  );
}
