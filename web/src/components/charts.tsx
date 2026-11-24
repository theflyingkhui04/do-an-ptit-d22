"use client";

import { formatNumber, formatShortMoney } from "@/lib/format";
import { useI18n } from "@/lib/i18n";

export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
  dashed?: boolean;
}

function ticksFor(lo: number, hi: number): number[] {
  const raw = (hi - lo) / 4 || Math.abs(hi) / 4 || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((v) => v >= raw) ?? raw;
  const ticks: number[] = [];
  for (let t = Math.floor(lo / step) * step; t <= Math.ceil(hi / step) * step + step / 2; t += step) ticks.push(Number(t.toPrecision(6)));
  return ticks;
}

export function LineChart({ label, loads, series, format, reference, referenceLabel }: {
  label: string;
  loads: number[];
  series: Series[];
  format: (value: number) => string;
  reference?: number;
  referenceLabel?: string;
}) {
  const { t } = useI18n();
  const W = 720, H = 320, L = 74, R = 230, T = 18, B = 52;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null)).concat(reference === undefined ? [] : [reference]);
  const ticks = ticksFor(Math.min(...all), Math.max(...all));
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x = (i: number) => (loads.length === 1 ? (L + W - R) / 2 : L + (i * (W - L - R)) / (loads.length - 1));
  const y = (v: number) => T + ((hi - v) * (H - T - B)) / (hi - lo || 1);
  const ends = series
    .map((s) => ({ s, last: [...s.values].reverse().find((v) => v !== null) ?? null }))
    .filter((e): e is { s: Series; last: number } => e.last !== null)
    .sort((a, b) => b.last - a.last);
  const labelY: number[] = [];
  for (const e of ends) labelY.push(Math.max(y(e.last) + 5, (labelY[labelY.length - 1] ?? -Infinity) + 18));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full min-w-[520px] max-w-[820px]" role="img" aria-label={label}>
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={L} x2={W - R} y1={y(tick)} y2={y(tick)} stroke="var(--line)" />
          <text x={L - 10} y={y(tick) + 5} textAnchor="end" fontSize={14} fill="var(--ink-3)">{format(tick)}</text>
        </g>
      ))}
      {reference !== undefined && (
        <>
          <line x1={L} x2={W - R} y1={y(reference)} y2={y(reference)} stroke="var(--ink)" strokeWidth={2} />
          {referenceLabel && <text x={L + 8} y={y(reference) - 8} fontSize={14} fill="var(--ink-2)">{referenceLabel}</text>}
        </>
      )}
      {loads.map((c, i) => (
        <text key={c} x={x(i)} y={H - B + 22} textAnchor="middle" fontSize={14} fill="var(--ink-3)">{c}</text>
      ))}
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={13} fill="var(--ink-3)">{t.concurrencyAxis}</text>
      {series.map((s) => {
        const points = s.values.map((v, i) => (v === null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(" ");
        return (
          <g key={s.name}>
            <polyline points={points} fill="none" stroke={s.color} strokeWidth={2.5} strokeLinejoin="round" strokeDasharray={s.dashed ? "6 5" : undefined} />
            {s.values.map((v, i) => v !== null && (
              <circle key={i} cx={x(i)} cy={y(v)} r={5} fill={s.color} stroke="var(--surface)" strokeWidth={2}>
                <title>{`${s.name}, Concurrency ${loads[i]}: ${format(v)}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
      {ends.map((e, i) => (
        <text key={e.s.name} x={W - R + 14} y={labelY[i]} fontSize={14} fontWeight={600} fill="var(--ink)">
          {`${e.s.name.length > 22 ? `${e.s.name.slice(0, 21)}…` : e.s.name} ${format(e.last)}`}
        </text>
      ))}
    </svg>
  );
}

export function BarChart({ rows, horizontal, color = "var(--step-3)", unit = "VND", label }: {
  rows: [string, number][];
  horizontal?: boolean;
  color?: string;
  unit?: string;
  label: string;
}) {
  const { lang } = useI18n();
  const max = Math.max(1, ...rows.map((r) => r[1]));
  const fmt = (v: number) => `${formatNumber(v, lang)}${unit ? ` ${unit}` : ""}`;
  if (horizontal) {
    return (
      <div role="img" aria-label={label} className="grid gap-2">
        {rows.map(([k, v], i) => (
          <div key={`${k}-${i}`} className="grid grid-cols-[minmax(0,13em)_minmax(0,1fr)_auto] items-center gap-2.5 text-[13px] max-sm:grid-cols-[minmax(0,7.5em)_minmax(0,1fr)_auto]">
            <span className="truncate text-right text-ink-2" title={k}>{k}</span>
            <span className="h-4">
              <span className="block h-full rounded" style={{ width: `${Math.max(1, (v / max) * 100)}%`, background: color }} title={`${k}: ${fmt(v)}`} />
            </span>
            <span className="num whitespace-nowrap font-semibold">{fmt(v)}</span>
          </div>
        ))}
      </div>
    );
  }
  const short = (v: number) => (!unit ? formatNumber(v, lang) : formatShortMoney(v, lang));
  const peak = rows.reduce((best, r, i) => (r[1] > rows[best][1] ? i : best), 0);
  const crowded = rows.length > 8;
  return (
    <div role="img" aria-label={label} className="flex h-[220px] items-end gap-1.5 border-b border-line pt-1 sm:gap-2.5">
      {rows.map(([k, v], i) => (
        <div key={`${k}-${i}`} className="relative flex h-full min-w-0 flex-1 flex-col items-center justify-end gap-1 pb-6">
          {(!crowded || i === peak) && <span className="num whitespace-nowrap text-[12px] font-semibold">{short(v)}</span>}
          <span
            className="w-[70%] max-w-14 rounded-t-[5px]"
            style={{ height: `calc((100% - 20px) * ${Math.max(0.01, v / max)})`, background: color }}
            title={`${k}: ${fmt(v)}`}
          />
          <span className="absolute bottom-0.5 max-w-full truncate text-[11.5px] text-ink-3 sm:text-[12px]">{k}</span>
        </div>
      ))}
    </div>
  );
}
