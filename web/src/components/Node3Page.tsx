"use client";

import { FileText, LayoutDashboard, Play, ShieldCheck } from "lucide-react";
import { useState } from "react";

import { BarChart } from "@/components/charts";
import type { PageId } from "@/components/Console";
import {
  Button,
  EmptyState,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Panel,
  Pill,
  ResultTable,
  Stat,
  Trace,
  defaultTarget,
} from "@/components/ui";
import { ApiError, type Api, type Endpoint, type QuestionResult, type ReportResult } from "@/lib/api";
import { formatMoney, formatMs, formatNumber } from "@/lib/format";

type Q = QuestionResult & { question_id: string };

const MONEY = /amount|total|value|tien|spend|chi_tieu|revenue/i;
const QUANTITY = /quantity|qty|so_luong/i;
const OCR_SUSPECT_QUANTITY = 1000; // a single shop receipt line above this is almost always an OCR misread

function single(q: Q | undefined): number | null {
  if (!q || q.status !== "ok" || q.rows.length !== 1) return null;
  const nums = q.rows[0].filter((c): c is number => typeof c === "number");
  return nums.length === 1 ? nums[0] : null;
}

// [label, value] pairs when the rows have one text-like column and a numeric column; otherwise null.
function pairs(q: Q | undefined): { rows: [string, number][]; money: boolean } | null {
  if (!q || q.status !== "ok" || q.rows.length < 2) return null;
  const sample = q.rows[0];
  const label = sample.findIndex((c) => typeof c === "string");
  let value = -1;
  for (let i = sample.length - 1; i >= 0; i -= 1) if (typeof sample[i] === "number") { value = i; break; }
  if (label < 0 || value < 0) return null;
  const rows = q.rows
    .filter((r) => typeof r[label] === "string" && typeof r[value] === "number")
    .map((r) => [String(r[label]), r[value] as number] as [string, number]);
  return rows.length >= 2 ? { rows, money: MONEY.test(q.columns[value] ?? "") } : null;
}

function quantityFlag(q: Q) {
  const index = q.columns.findIndex((c) => QUANTITY.test(c));
  if (index < 0) return undefined;
  return (row: unknown[]) => (typeof row[index] === "number" && (row[index] as number) > OCR_SUSPECT_QUANTITY ? "nghi lỗi OCR" : null);
}

function QuestionTile({ q, title, money }: { q: Q | undefined; title: string; money?: boolean }) {
  const value = single(q);
  if (value !== null) return <Stat label={title} value={money ? formatMoney(value) : formatNumber(value)} sub={`${q?.question_id}, từ kết quả SQL`} />;
  return <Stat label={title} value="—" sub={q ? (q.status === "no_data" ? "Không có dữ liệu" : q.status === "error" ? "Câu SQL bị chặn" : "Kết quả không phải một số") : "Chưa chạy"} />;
}

function ChartOrTable({ q, title, horizontal }: { q: Q | undefined; title: string; horizontal?: boolean }) {
  if (!q) return null;
  const data = pairs(q);
  const top = q.status === "ok" && q.rows.length === 1 ? q.rows[0] : null;
  return (
    <Panel title={title} meta={<span>{q.question_id}</span>}>
      {data ? (
        <BarChart
          rows={horizontal ? [...data.rows].sort((a, b) => b[1] - a[1]).slice(0, 8) : data.rows}
          horizontal={horizontal}
          unit={data.money ? "đ" : ""}
          label={q.question}
        />
      ) : top ? (
        <p className="text-[17px]"><b>{top.map((c) => (typeof c === "number" ? formatNumber(c) : String(c ?? "—"))).join(", ")}</b></p>
      ) : q.status === "ok" ? (
        <ResultTable columns={q.columns} rows={q.rows} max={10} />
      ) : (
        <p className="text-ink-3">{q.status === "no_data" ? "Không có dữ liệu khớp." : q.error}</p>
      )}
    </Panel>
  );
}

function Dashboard({ data }: { data: ReportResult }) {
  const by = (id: string) => data.questions.find((q) => q.question_id === id);
  const q2 = by("Q2");
  const q6 = by("Q6");
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Stat label="Hoá đơn trong dữ liệu" value={formatNumber(data.invoice_count)} sub="đếm trực tiếp trong cơ sở dữ liệu" />
        <QuestionTile q={by("Q1")} title="Tổng chi tiêu tháng 08/2020" money />
        <QuestionTile q={by("Q4")} title="Hoá đơn có VAT" />
      </div>
      <div className="grid items-start gap-5 lg:grid-cols-2">
        <ChartOrTable q={by("Q5")} title="Tổng tiền theo ngày lập" />
        <ChartOrTable q={by("Q3")} title="Cửa hàng có tổng giá trị cao nhất" horizontal />
      </div>
      <div className="grid items-start gap-5 lg:grid-cols-2">
        {q2 && (
          <Panel title="Mặt hàng mua nhiều nhất" meta={<span>Q2</span>}>
            {q2.status === "ok" ? <ResultTable columns={q2.columns} rows={q2.rows} max={10} flag={quantityFlag(q2)} /> : <p className="text-ink-3">{q2.status === "no_data" ? "Không có dữ liệu khớp." : q2.error}</p>}
          </Panel>
        )}
        {q6 && (
          <Panel title="Thuế theo loại" meta={<span>Q6</span>}>
            {q6.status === "ok" ? <ResultTable columns={q6.columns} rows={q6.rows} max={10} /> : <p className="text-ink-3">{q6.status === "no_data" ? "Không có dữ liệu khớp." : q6.error}</p>}
          </Panel>
        )}
      </div>
    </>
  );
}

export default function Node3Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const [target, setTarget] = useState<string[]>(() => defaultTarget(endpoints));
  const [data, setData] = useState<ReportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (endpoints.length === 0) return <><PageHead title="Báo cáo và dashboard" /><NeedEndpoint what="Node 3" onGo={() => go("endpoints")} /></>;

  async function run() {
    if (target.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      setData(await api.report(target[0]));
    } catch (exc) {
      setError(exc instanceof ApiError && exc.status === 429 ? "Đã chạy quá nhiều trong ít phút, thử lại sau." : (exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const answered = data ? data.questions.filter((q) => q.status === "ok").length : 0;

  return (
    <>
      <PageHead title="Báo cáo và dashboard">
        Node 2 chạy sáu câu Q1–Q6, rồi Node 3 viết báo cáo tiếng Việt từ kết quả. Biểu đồ vẽ thẳng từ các dòng SQL; model chỉ viết phần nhận xét.
      </PageHead>

      <Panel title="Tạo báo cáo" icon={Play}>
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="grid gap-2">
            <span className="text-[13.5px] font-semibold">Chạy trên</span>
            <EndpointChips single endpoints={endpoints} selected={target} onToggle={(id) => setTarget([id])} />
          </div>
          <Button icon={Play} disabled={busy || target.length === 0} onClick={run}>
            {busy ? "Đang chạy Q1–Q6 và viết báo cáo…" : "Tạo báo cáo"}
          </Button>
        </div>
        {error && <Notice tone="bad">{error}</Notice>}
        {data && (
          <Trace
            steps={[
              { label: "Node 2", title: "Node 2, sáu câu Q1–Q6", ms: data.timings_ms.node2, kind: "n2" },
              { label: "Node 3", title: "Node 3, viết báo cáo", ms: data.timings_ms.node3, kind: "n3" },
            ]}
            extra={<span>trên <b className="text-ink">{data.endpoint.name}</b>, {answered}/6 câu có dữ liệu</span>}
          />
        )}
      </Panel>

      {!data ? (
        <EmptyState icon={LayoutDashboard} title="Chưa có báo cáo">
          <p>Bấm “Tạo báo cáo”. Dashboard chỉ hiện số có trong kết quả SQL của lần chạy này, không có số dựng sẵn.</p>
        </EmptyState>
      ) : (
        <>
          <Dashboard data={data} />
          <Panel
            title="Báo cáo của Node 3"
            icon={FileText}
            meta={data.report.metrics?.tpot_ms ? <span>TPOT {formatMs(data.report.metrics.tpot_ms)}</span> : undefined}
          >
            {data.report.status === "ok" && data.report.text ? (
              <div className="max-w-[75ch] whitespace-pre-wrap leading-relaxed">{data.report.text}</div>
            ) : (
              <Notice tone="bad">Node 3 không viết được báo cáo: {data.report.error ?? data.report.status}</Notice>
            )}
            <div className="flex flex-wrap gap-2">
              {data.number_check.ungrounded.length === 0 ? (
                <Pill tone="good" icon={ShieldCheck}>{data.number_check.numbers} số trong báo cáo đều có trong kết quả SQL</Pill>
              ) : (
                <Pill tone="warn">Số không có trong kết quả SQL: {data.number_check.ungrounded.join(", ")}</Pill>
              )}
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
