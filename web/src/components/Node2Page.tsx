"use client";

import { DatabaseZap, MessageSquareText, Play, ShieldCheck, Table2 } from "lucide-react";
import { useState } from "react";

import type { PageId } from "@/components/Console";
import {
  Button,
  EndpointChips,
  NeedEndpoint,
  Notice,
  PageHead,
  Panel,
  Pill,
  ResultTable,
  SqlBlock,
  Stat,
  Trace,
  cx,
  defaultTarget,
  type TraceStep,
} from "@/components/ui";
import { ApiError, type Api, type Endpoint, type QuestionResult } from "@/lib/api";
import { formatMs, formatNumber } from "@/lib/format";
import { QUESTIONS, VIEWS } from "@/lib/views";

const CHECKS = ["Chỉ đọc (một câu SELECT)", "Chỉ 4 view được phép", "Không cộng trùng hoá đơn khi nối bảng", "Tối đa 200 dòng"];

function askError(exc: unknown): string {
  if (exc instanceof ApiError && exc.status === 429) return "Đã hỏi quá nhiều trong ít phút, thử lại sau.";
  return (exc as Error).message;
}

export function traceOf(result: QuestionResult): TraceStep[] {
  let node2 = 0;
  return result.calls.map((c) => {
    if (c.step === "node2") {
      node2 += 1;
      return { label: node2 === 1 ? "Sinh SQL" : "Sửa SQL", ms: c.wall_ms ?? c.latency_ms, kind: "n2" as const };
    }
    return { label: "Trả lời", title: "Câu trả lời ngắn", ms: c.wall_ms ?? c.latency_ms, kind: "io" as const };
  });
}

function Outcome({ result }: { result: QuestionResult }) {
  const first = result.calls.find((c) => c.step === "node2");
  return (
    <div className="grid gap-4">
      <Trace steps={traceOf(result)} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="TTFT (sinh SQL)" value={formatMs(first?.ttft_ms)} />
        <Stat label="TPOT (sinh SQL)" value={formatMs(first?.tpot_ms)} />
        <Stat label="Token SQL" value={formatNumber(first?.completion_tokens)} />
        <Stat label="Số dòng" value={formatNumber(result.row_count)} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {result.status === "ok" && <Pill tone="good" icon={ShieldCheck}>Đã trả lời</Pill>}
        {result.status === "no_data" && <Pill tone="idle">Không có dữ liệu khớp</Pill>}
        {result.status === "error" && <Pill tone="bad">Bị chặn</Pill>}
        {result.repairs > 0 && <Pill tone="warn">Câu SQL đầu bị lỗi, đã sửa {result.repairs} lần</Pill>}
      </div>

      {result.sql && <SqlBlock sql={result.sql} />}
      {result.status !== "error" ? (
        <ul className="flex flex-wrap gap-2" aria-label="Kiểm tra SQL">
          {CHECKS.map((c) => <li key={c}><Pill tone="good" icon={ShieldCheck}>{c}</Pill></li>)}
        </ul>
      ) : (
        <Notice tone="bad">{result.error}</Notice>
      )}
      {result.first_error && result.status !== "error" && (
        <p className="text-[13px] text-ink-3">Lỗi lần đầu: <span className="font-mono">{result.first_error}</span></p>
      )}

      {result.status !== "error" && <ResultTable columns={result.columns} rows={result.rows} />}
      {result.answer && (
        <div className="rounded-xl border border-line p-3.5">
          <div className="mb-1 text-[13px] text-ink-3">Câu trả lời (chỉ dùng số có trong kết quả SQL)</div>
          <p>{result.answer}</p>
        </div>
      )}
      {result.warning === "fabricated_number" && (
        <Notice tone="warn">Câu trả lời của model có số không nằm trong kết quả SQL nên không hiện. Xem bảng kết quả ở trên.</Notice>
      )}
    </div>
  );
}

export default function Node2Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const [target, setTarget] = useState<string[]>(() => defaultTarget(endpoints));
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QuestionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (endpoints.length === 0) return <><PageHead title="Hỏi bằng SQL" /><NeedEndpoint what="Node 2" onGo={() => go("endpoints")} /></>;

  async function ask(text: string) {
    if (!text.trim() || target.length === 0) return;
    setQuestion(text);
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.ask(text.trim(), target[0]));
    } catch (exc) {
      setError(askError(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHead title="Hỏi bằng SQL">
        Node 2 đổi câu hỏi tiếng Việt thành một câu SQL chỉ đọc, kiểm tra câu SQL rồi chạy trên dữ liệu hoá đơn đã lưu.
      </PageHead>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="grid min-w-0 gap-5">
          <Panel title="Câu hỏi" icon={MessageSquareText}>
            <form
              className="grid gap-3"
              onSubmit={(event) => {
                event.preventDefault();
                ask(question);
              }}
            >
              <label htmlFor="question" className="sr-only">Câu hỏi</label>
              <textarea
                id="question"
                rows={2}
                maxLength={300}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ví dụ: Cửa hàng nào có nhiều hoá đơn nhất?"
                className="rounded-lg border border-line bg-surface px-3 py-2"
              />
              <div className="flex flex-wrap gap-2" aria-label="Câu hỏi mẫu Q1–Q6">
                {QUESTIONS.map((q) => (
                  <button
                    key={q.id}
                    type="button"
                    title={q.question}
                    onClick={() => ask(q.question)}
                    disabled={busy}
                    className={cx("rounded-full border border-line px-3 py-1 text-[13px] hover:bg-sunken disabled:opacity-50", question === q.question && "border-ink font-semibold")}
                  >
                    {q.id}
                  </button>
                ))}
              </div>
              <div className="grid gap-2">
                <span className="text-[13.5px] font-semibold">Chạy trên</span>
                <EndpointChips single endpoints={endpoints} selected={target} onToggle={(id) => setTarget([id])} />
              </div>
              <div>
                <Button type="submit" icon={Play} disabled={busy || !question.trim() || target.length === 0}>
                  {busy ? "Đang hỏi…" : "Hỏi"}
                </Button>
              </div>
            </form>
            <ol className="grid gap-1 text-[13.5px] text-ink-2">
              {QUESTIONS.map((q) => <li key={q.id}><b className="text-ink">{q.id}.</b> {q.question}</li>)}
            </ol>
          </Panel>

          <Panel title="Kết quả" icon={Table2} meta={result?.endpoint && <span>trên {result.endpoint.name}</span>}>
            {!result && !busy && !error && <p className="text-ink-3">Gõ câu hỏi hoặc chọn Q1–Q6. Câu SQL, kết quả kiểm tra và bảng dữ liệu hiện ở đây.</p>}
            {busy && <p className="text-ink-3">Đang sinh và kiểm tra câu SQL…</p>}
            {error && <Notice tone="bad">{error}</Notice>}
            {result && <Outcome result={result} />}
          </Panel>
        </div>

        <Panel title="Dữ liệu được phép hỏi" icon={DatabaseZap}>
          <p className="text-[13px] text-ink-3">Node 2 chỉ đọc 4 view này. Mọi câu lệnh ghi, xoá hoặc bảng khác đều bị chặn.</p>
          {VIEWS.map((v) => (
            <details key={v.name} className="rounded-lg border border-line px-3 py-2" open={v.name === "v_invoice_facts"}>
              <summary className="cursor-pointer">
                <span className="font-mono text-[13px] font-semibold">{v.name}</span>
                <span className="ml-2 text-[12.5px] text-ink-3">{v.about}</span>
              </summary>
              <p className="mt-1.5 break-words font-mono text-[12px] leading-relaxed text-ink-2">{v.columns.join(", ")}</p>
            </details>
          ))}
        </Panel>
      </div>
    </>
  );
}
