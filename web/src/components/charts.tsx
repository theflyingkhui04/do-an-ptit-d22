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

// Bars drawn straight from SQL rows: vertical for dates, horizontal for names.
export function BarChart({ rows, horizontal, color = "var(--lane-1)", unit = "đ", label }: {
  rows: [string, number][];
  horizontal?: boolean;
  color?: string;
  unit?: string;
  label: string;
}) {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  const fmt = (v: number) => `${formatNumber(v)}${unit ? ` ${unit}` : ""}`;
  if (horizontal) {
    // Label column wide enough for "2020-08-16, MINIMART ANAN" (date + store) at 13px.
    const W = 560, L = 240, R = 110, rowH = 34, H = rows.length * rowH + 10;
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={label}>
        {rows.map(([k, v], i) => {
          const w = ((W - L - R) * v) / max, yy = 6 + i * rowH;
          return (
            <g key={`${k}-${i}`}>
              <text x={L - 10} y={yy + 18} textAnchor="end" fontSize={13} fill="var(--ink-2)">{k.length > 28 ? `${k.slice(0, 27)}…` : k}</text>
              <rect x={L} y={yy + 5} width={Math.max(w, 2)} height={18} rx={4} fill={color}><title>{`${k}: ${fmt(v)}`}</title></rect>
              <text x={L + w + 8} y={yy + 18} fontSize={13} fill="var(--ink)">{fmt(v)}</text>
            </g>
          );
        })}
      </svg>
    );
  }
  const W = 520, H = 240, L = 64, B = 34, T = 22;
  const bw = (W - L - 10) / Math.max(rows.length, 1);
  const top = Math.ceil(max / 4 / 100000) * 100000 * 4 || max;
  const y = (v: number) => T + (1 - v / top) * (H - T - B);
  const peak = rows.reduce((best, r, i) => (r[1] > rows[best][1] ? i : best), 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={label}>
      {[0, top / 4, top / 2, (3 * top) / 4, top].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - 10} y1={y(g)} y2={y(g)} stroke="var(--line)" />
          <text x={L - 8} y={y(g) + 4} textAnchor="end" fontSize={12} fill="var(--ink-3)">{g >= 1000 ? `${formatNumber(g / 1000)}k` : formatNumber(g)}</text>
        </g>
      ))}
      {rows.map(([k, v], i) => {
        const xx = L + i * bw + bw * 0.18, w = bw * 0.64;
        return (
          <g key={`${k}-${i}`}>
            <rect x={xx} y={y(v)} width={w} height={H - B - y(v)} rx={4} fill={color}><title>{`${k}: ${fmt(v)}`}</title></rect>
            <text x={xx + w / 2} y={H - B + 18} textAnchor="middle" fontSize={12} fill="var(--ink-3)">{k}</text>
            {i === peak && <text x={xx + w / 2} y={y(v) - 7} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink)">{fmt(v)}</text>}
          </g>
        );
      })}
    </svg>
  );
}
