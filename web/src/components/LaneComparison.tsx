"use client";

import type { LaneReport } from "@/lib/api";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";

const ORDER = ["vanilla", "static", "adaptive", "eagle"];

interface Props {
  lanes: Record<string, LaneReport>;
  gpu?: string;
  storedLane?: string;
}

export default function LaneComparison({ lanes, gpu, storedLane }: Props) {
  const rows = Object.values(lanes).sort((a, b) => ORDER.indexOf(a.lane) - ORDER.indexOf(b.lane));
  return (
    <section aria-label="So sánh làn" className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">Tốc độ Node 1 theo làn</h3>
        <p className="text-xs text-slate-500">
          GPU {gpu ?? "—"} · số đo của lượt demo này, không phải số benchmark
        </p>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="text-xs text-slate-500">
            <tr>
              <th className="py-1 pr-3 font-medium">Làn</th>
              <th className="py-1 pr-3 font-medium">Độ trễ</th>
              <th className="py-1 pr-3 font-medium">TPOT</th>
              <th className="py-1 pr-3 font-medium">Token/s</th>
              <th className="py-1 pr-3 font-medium">Tăng tốc so với vanilla</th>
              <th className="py-1 pr-3 font-medium">Giống vanilla</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((lane) => (
              <tr key={lane.lane} data-testid={`lane-${lane.lane}`} className="border-t border-slate-100">
                <td className="py-1.5 pr-3 font-medium text-slate-900">
                  {lane.lane}
                  {lane.lane === storedLane && <span className="ml-1 text-xs text-blue-600">(đã lưu)</span>}
                </td>
                {lane.valid ? (
                  <>
                    <td className="py-1.5 pr-3">{formatMs(lane.latency_ms)}</td>
                    <td className="py-1.5 pr-3">{formatMs(lane.tpot_ms)}</td>
                    <td className="py-1.5 pr-3">{formatNumber(lane.output_tokens_per_s)}</td>
                    <td className="py-1.5 pr-3" data-testid={`speedup-${lane.lane}`}>
                      {formatSpeedup(lane.speedup_tpot_vs_vanilla)}
                    </td>
                    <td className="py-1.5 pr-3" title={lane.differing_fields.join(", ") || undefined}>
                      {lane.total_fields === null ? "—" : `giống vanilla ${lane.equal_fields}/${lane.total_fields} trường`}
                    </td>
                  </>
                ) : (
                  <td colSpan={5} className="py-1.5 pr-3 text-red-600">
                    Không hợp lệ: {lane.error}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
