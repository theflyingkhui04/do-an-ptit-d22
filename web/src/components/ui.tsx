"use client";

import type { LucideIcon } from "lucide-react";
import Image from "next/image";
import type { ReactNode } from "react";

import type { Endpoint } from "@/lib/api";
import { formatCell, formatMs } from "@/lib/format";

// FlexSDS white tile from logo/icon_white.webp: its bold red F stays legible at 16–36 px,
// unlike the thin glow of the black tile. Decorative (empty alt): the name is written next to it.
export function BrandMark({ size, className }: { size: number; className?: string }) {
  return (
    <Image src="/brand/flexsds-icon-light.png" alt="" width={size} height={size} unoptimized priority
      className={cx("rounded-[13%]", className)} />
  );
}

export function cx(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ");
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

export function Button({ variant = "primary", icon: Icon, children, className, ...props }: {
  variant?: "primary" | "ghost" | "danger";
  icon?: LucideIcon;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const look = {
    primary: "bg-ink text-surface hover:opacity-90",
    ghost: "border border-line bg-surface text-ink hover:bg-sunken",
    danger: "border border-line bg-surface text-bad hover:bg-bad-tint",
  }[variant];
  return (
    <button
      type="button"
      {...props}
      className={cx("inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50", look, className)}
    >
      {Icon && <Icon aria-hidden className="size-4" />}
      {children}
    </button>
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

// ---------------------------------------------------------------- trace strip: timed steps of one run
export interface TraceStep {
  label: string;
  ms: number | null | undefined;
  kind: "io" | "n1" | "n2" | "n3";
  title?: string;
}
const STEP_COLOR = { io: "var(--step-io)", n1: "var(--step-1)", n2: "var(--step-2)", n3: "var(--step-3)" };

export function Trace({ steps, extra }: { steps: TraceStep[]; extra?: ReactNode }) {
  const shown = steps.filter((s) => s.ms !== null && s.ms !== undefined) as (TraceStep & { ms: number })[];
  const total = shown.reduce((sum, s) => sum + s.ms, 0);
  return (
    <div className="grid gap-2">
      <div className="flex h-[30px] overflow-hidden rounded-lg bg-sunken" role="img" aria-label={shown.map((s) => `${s.label} ${formatMs(s.ms)}`).join(", ")}>
        {shown.map((s, i) => (
          <div
            key={s.label}
            title={`${s.title ?? s.label}: ${formatMs(s.ms)}`}
            className={cx("flex min-w-[42px] items-center overflow-hidden whitespace-nowrap px-2.5 text-[12.5px] font-semibold text-white", i > 0 && "border-l-2 border-surface")}
            style={{ flex: Math.max(s.ms, total * 0.03), background: STEP_COLOR[s.kind] }}
          >
            {/* very short steps stay unlabeled here; the legend below names every step */}
            {s.ms >= total * 0.08 && s.label}
            {s.ms >= total * 0.12 && <span className="ml-1.5 font-normal">{formatMs(s.ms)}</span>}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-[18px] gap-y-1.5 text-[13px] text-ink-2">
        {shown.map((s) => (
          <span key={s.label}>{s.title ?? s.label} <b className="num text-ink">{formatMs(s.ms)}</b></span>
        ))}
        <span>Tổng <b className="num text-ink">{formatMs(total)}</b></span>
        {extra}
      </div>
    </div>
  );
}

// The model behind an endpoint: serving_config's target model, else vLLM's ``root``, else the served name.
// Lanes served as qwen2.5-7b-vanilla / -adaptive share one base model, so comparing them is fair.
export function baseModel(e: Endpoint): string {
  return e.meta?.target_model || e.meta?.base_model || e.model_id;
}

// ---------------------------------------------------------------- endpoint colours and chips
// Three validated colours, in the order endpoints were added; the baseline is grey; extra endpoints share ink.
export function endpointColors(endpoints: Endpoint[]): Record<string, string> {
  const colors: Record<string, string> = {};
  const slots = ["var(--lane-1)", "var(--lane-2)", "var(--lane-3)"];
  let next = 0;
  for (const e of endpoints) {
    colors[e.endpoint_id] = e.is_baseline ? "var(--lane-ref)" : (slots[next++] ?? "var(--ink-2)");
  }
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
  const colors = endpointColors(endpoints);
  return (
    <div className="flex flex-wrap gap-2" role={single ? "radiogroup" : "group"} aria-label="Chạy trên endpoint">
      {endpoints.map((e) => {
        const on = selected.includes(e.endpoint_id);
        return (
          <button
            key={e.endpoint_id}
            type="button"
            role={single ? "radio" : "checkbox"}
            aria-checked={on}
            disabled={e.status === "down"}
            onClick={() => onToggle(e.endpoint_id)}
            title={e.status === "down" ? (e.status_message ?? "Endpoint đã dừng") : e.base_url}
            className={cx(
              "inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1.5 text-[13.5px] disabled:opacity-50",
              on ? "border-ink font-semibold shadow-[0_0_0_1px_var(--ink)]" : "border-line bg-surface",
            )}
          >
            <Dot color={colors[e.endpoint_id]} />
            <span className="truncate">{e.name}</span>
            {e.is_baseline && <span className="rounded-md border border-line px-1.5 text-[11.5px] font-normal text-ink-3">mốc</span>}
            {e.status === "down" && <span className="text-[11.5px] font-normal text-ink-3">đã dừng</span>}
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
  return (
    <div className="grid justify-items-start gap-2.5 rounded-xl border border-line bg-surface p-7">
      <h2 className="text-[17px] font-bold">Cần ít nhất một endpoint</h2>
      <p className="max-w-[62ch] text-ink-2">{what} chạy trên endpoint do CLI FlexSDS tạo ra. Dán địa chỉ endpoint ở trang Endpoint rồi quay lại đây.</p>
      <button type="button" onClick={onGo} className="rounded-lg bg-ink px-3.5 py-2 text-sm font-semibold text-surface">Mở trang Endpoint</button>
    </div>
  );
}

// First endpoint that is not the baseline and still running; the baseline joins every run on its own.
export function defaultTarget(endpoints: Endpoint[]): string[] {
  const live = endpoints.filter((e) => e.status === "ok");
  const pick = live.find((e) => !e.is_baseline) ?? live[0] ?? endpoints[0];
  return pick ? [pick.endpoint_id] : [];
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

export function ResultTable({ columns, rows, max = 50, flag }: {
  columns: string[];
  rows: unknown[][];
  max?: number;
  flag?: (row: unknown[]) => string | null;
}) {
  if (rows.length === 0) return <p className="text-ink-3">Không có dòng nào.</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="w-full text-left text-[13.5px]" aria-label="Kết quả SQL">
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
              <tr key={i} className="border-t border-line">
                {row.map((cell, j) => (
                  <td key={j} className={cx("px-3 py-1.5", typeof cell === "number" && "text-right")}>{formatCell(cell)}</td>
                ))}
                {flag && <td className="px-3 py-1.5">{note && <Pill tone="warn">{note}</Pill>}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > max && <p className="border-t border-line px-3 py-2 text-[12.5px] text-ink-3">Hiện {max}/{rows.length} dòng.</p>}
    </div>
  );
}

const SQL_KEYWORDS = /\b(SELECT|FROM|WHERE|GROUP|BY|ORDER|LIMIT|AS|AND|OR|NOT|IN|IS|NULL|JOIN|LEFT|INNER|ON|HAVING|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|DESC|ASC|LIKE|ILIKE|BETWEEN|WITH|ROUND|COALESCE|EXTRACT|CAST|DATE_TRUNC|LOWER|UPPER)\b/gi;

export function SqlBlock({ sql }: { sql: string }) {
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
    <pre aria-label="Câu SQL" className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-sunken px-4 py-3.5 font-mono text-[13px] leading-relaxed">
      {parts}
    </pre>
  );
}
