"use client";

import { formatNumber, formatSpeedup } from "@/lib/format";

// Line chart of speed-up versus load, one line per endpoint, with the bold 1x line of the baseline.
// Text uses theme ink tokens; marks use the endpoint colours. Labels sit at the line ends (no legend lookup).
export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
}

export function SpeedChart({ loads, series, baselineLabel = "endpoint mốc" }: {
  loads: number[];
  series: Series[];
  baselineLabel?: string;
}) {
  const W = 720, H = 330, L = 56, R = 220, T = 18, B = 52;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const lo = Math.min(0.6, Math.floor((Math.min(1, ...all) - 0.05) * 10) / 10);
  const hi = Math.max(1.6, Math.ceil((Math.max(1, ...all) + 0.05) * 10) / 10);
  const x = (i: number) => (loads.length === 1 ? (L + W - R) / 2 : L + (i * (W - L - R)) / (loads.length - 1));
  const y = (v: number) => T + ((hi - v) * (H - T - B)) / (hi - lo);
  const ticks: number[] = [];
  for (let t = Math.ceil(lo * 5) / 5; t <= hi + 1e-9; t += 0.2) ticks.push(Math.round(t * 10) / 10);
  const ends = series
    .map((s) => ({ s, last: [...s.values].reverse().find((v) => v !== null) ?? null }))
    .filter((e): e is { s: Series; last: number } => e.last !== null)
    .sort((a, b) => b.last - a.last);
  let prev = -Infinity;
  const labelY = ends.map((e) => {
    const yy = Math.max(y(e.last) + 5, prev + 18);
    prev = yy;
    return yy;
  });
  return (
    <div className="overflow-x-auto">
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full min-w-[540px] max-w-[800px]" role="img" aria-label="Nhanh gấp mấy lần endpoint mốc theo số người dùng cùng lúc">
      {ticks.filter((t) => Math.abs(t - 1) > 1e-9).map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <text x={L - 10} y={y(t) + 5} textAnchor="end" fontSize={15} fill="var(--ink-3)">{formatSpeedup(t)}</text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={y(1)} y2={y(1)} stroke="var(--ink)" strokeWidth={2} />
      <text x={L - 10} y={y(1) + 5} textAnchor="end" fontSize={15} fontWeight={700} fill="var(--ink)">1×</text>
      <text x={L + 8} y={y(1) - 8} fontSize={14} fill="var(--ink-2)">{baselineLabel} (1×)</text>
      {loads.map((c, i) => (
        <text key={c} x={x(i)} y={H - B + 22} textAnchor="middle" fontSize={15} fill="var(--ink-3)">{c}</text>
      ))}
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={14} fill="var(--ink-3)">Số người dùng cùng lúc</text>
      {series.map((s) => {
        const points = s.values.map((v, i) => (v === null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(" ");
        return (
          <g key={s.name}>
            <polyline points={points} fill="none" stroke={s.color} strokeWidth={2.5} strokeLinejoin="round" />
            {s.values.map((v, i) => v !== null && (
              <circle key={i} cx={x(i)} cy={y(v)} r={5} fill={s.color} stroke="var(--surface)" strokeWidth={2}>
                <title>{`${s.name}, ${loads[i]} người dùng: ${formatSpeedup(v)}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
      {ends.map((e, i) => (
        <text key={e.s.name} x={W - R + 14} y={labelY[i]} fontSize={14} fill="var(--ink)">
          <tspan fontWeight={600}>{e.s.name.length > 22 ? `${e.s.name.slice(0, 21)}…` : e.s.name}</tspan>
          <tspan fill="var(--ink-2)"> {formatSpeedup(e.last)}</tspan>
        </text>
      ))}
    </svg>
    </div>
  );
}

// Bars drawn straight from SQL rows: vertical for dates, horizontal for names. HTML, not SVG, so the text keeps
// its size on a phone (design D16); values are written on the bars, so there is no axis.
export function BarChart({ rows, horizontal, color = "var(--step-3)", unit = "đ", label }: {
  rows: [string, number][];
  horizontal?: boolean;
  color?: string;
  unit?: string;
  label: string;
}) {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  const fmt = (v: number) => `${formatNumber(v)}${unit ? ` ${unit}` : ""}`;
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
  // Short values fit above narrow columns: 1,27 tr and 420 k for money.
  const short = (v: number) => (!unit ? formatNumber(v)
    : v >= 1e6 ? `${formatNumber(Math.round(v / 1e4) / 100)} tr` : v >= 1e3 ? `${formatNumber(Math.round(v / 1e3))} k` : fmt(v));
  const peak = rows.reduce((best, r, i) => (r[1] > rows[best][1] ? i : best), 0);
  const crowded = rows.length > 8; // then only the highest column is labelled
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
