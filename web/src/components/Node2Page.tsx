"use client";

import { ChevronRight, DatabaseZap, Play, ShieldCheck } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";

import type { PageId } from "@/components/Console";
import {
  Button,
  Drawer,
  EndpointChips,
  NeedEndpoint,
  PageHead,
  Pill,
  ResultTable,
  Spinner,
  SqlBlock,
  cx,
  defaultTarget,
  endpointColors,
  secondsSince,
  useNow,
  Dot,
} from "@/components/ui";
import { ApiError, type Api, type Endpoint, type QuestionResult } from "@/lib/api";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { QUESTIONS, VIEWS } from "@/lib/views";

const CHECKS = ["Chỉ đọc (một câu SELECT)", "Chỉ 4 view được phép", "Không cộng trùng hoá đơn khi nối bảng", "Tối đa 200 dòng"];
const MAX_ENDPOINTS = 6; // as on Node 1

// What one endpoint answered for the current question (design D15).
interface Answer {
  started: number;
  result?: QuestionResult;
  error?: string;
}

interface Round {
  question: string;
  order: string[]; // baseline first
  answers: Record<string, Answer>;
}

function askError(exc: unknown): string {
  if (exc instanceof ApiError && exc.status === 429) return "Đã hỏi quá nhiều trong ít phút, thử lại sau.";
  return (exc as Error).message;
}

const firstSql = (r: QuestionResult) => r.calls.find((c) => c.step === "node2");
const totalMs = (r: QuestionResult) => r.calls.reduce((sum, c) => sum + (c.wall_ms ?? c.latency_ms ?? 0), 0);
const sameRow = (a: unknown[] | undefined, b: unknown[] | undefined) => JSON.stringify(a) === JSON.stringify(b);

// Rows of `r` that are not the same as the baseline's row at the same place.
function differentRows(r: QuestionResult, base: QuestionResult | undefined): Set<number> | null {
  if (!base || base === r || r.status !== "ok" || base.status !== "ok") return null;
  const rows = new Set<number>();
  r.rows.forEach((row, i) => {
    if (!sameRow(row, base.rows[i])) rows.add(i);
  });
  return rows;
}

function StatusPill({ answer, now }: { answer: Answer; now: number }) {
  if (answer.error) return <Pill tone="bad">Lỗi</Pill>;
  const r = answer.result;
  if (!r) return <Pill tone="idle"><Spinner className="size-3.5" />Đang hỏi, {secondsSince(answer.started, now)} s</Pill>;
  if (r.status === "ok") return <Pill tone="good">Đã trả lời</Pill>;
  if (r.status === "no_data") return <Pill tone="idle">Không có dữ liệu khớp</Pill>;
  return <Pill tone="bad">Bị chặn</Pill>;
}

function Stats({ result }: { result: QuestionResult }) {
  const first = firstSql(result);
  const cells: [string, string][] = [
    ["Thời gian", formatMs(totalMs(result))],
    ["TTFT", formatMs(first?.ttft_ms)],
    ["TPOT", formatMs(first?.tpot_ms)],
    ["Token", formatNumber(first?.completion_tokens)],
  ];
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-lg border border-line sm:grid-cols-4">
      {cells.map(([term, value], i) => (
        <div key={term} className={cx("grid min-w-0 gap-px px-2.5 py-1.5", i > 0 && "sm:border-l", i % 2 === 1 && "border-l", i >= 2 && "border-t sm:border-t-0", "border-line")}>
          <dt className="truncate text-[11.5px] text-ink-3">{term}</dt>
          <dd className="num m-0 whitespace-nowrap text-[14px] font-semibold">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function SqlDetails({ result }: { result: QuestionResult }) {
  const blocked = result.status === "error";
  const [open, setOpen] = useState(blocked);
  if (!result.sql) return null;
  return (
    <details open={open} className="rounded-lg border border-line">
      <summary
        onClick={(event) => {
          event.preventDefault();
          setOpen(!open);
        }}
        className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-[13px] font-semibold text-ink-2 [&::-webkit-details-marker]:hidden"
      >
        <ChevronRight aria-hidden className={cx("size-3.5 transition-transform", open && "rotate-90")} />
        <span>Câu SQL</span>
        {!blocked && (
          <span className="ml-auto inline-flex items-center gap-1 font-medium text-good"><ShieldCheck aria-hidden className="size-3.5" />qua 4 kiểm tra</span>
        )}
      </summary>
      <div className="grid gap-2 border-t border-line p-2.5">
        <SqlBlock sql={result.sql} />
        {!blocked && (
          <ul className="grid list-disc gap-0.5 pl-5 text-[12.5px] text-ink-2">
            {CHECKS.map((c) => <li key={c}>{c}</li>)}
          </ul>
        )}
        {result.first_error && <p className="text-[12.5px] text-ink-3">Lỗi lần đầu: <span className="font-mono">{result.first_error}</span></p>}
      </div>
    </details>
  );
}

function Column({ endpoint, color, isBaseline, answer, base, now }: {
  endpoint: Endpoint;
  color: string;
  isBaseline: boolean;
  answer: Answer;
  base?: QuestionResult;
  now: number;
}) {
  const r = answer.result;
  const diff = r ? differentRows(r, base) : null;
  const foreign = r?.foreign_words ?? [];
  const ungrounded = r?.ungrounded_numbers ?? [];
  let body: ReactNode;
  if (answer.error) {
    body = <p role="alert" className="text-[14px] text-bad">{answer.error}</p>;
  } else if (!r) {
    body = <p className="text-ink-3">Đang sinh câu SQL, kiểm tra rồi chạy trên dữ liệu đã lưu…</p>;
  } else {
    body = (
      <>
        {r.status === "error" ? (
          <p role="alert" className="min-h-[3em] text-[14px] text-bad">{r.error}</p>
        ) : r.answer ? (
          <p className="min-h-[3em] text-base leading-normal">{r.answer}</p>
        ) : (
          <p className="min-h-[3em] text-ink-3">
            {r.status === "no_data" ? "Câu SQL chạy đúng nhưng không có dòng nào khớp, nên không có câu trả lời." : "Model không trả về câu trả lời nào."}
          </p>
        )}
        <Stats result={r} />
        {(r.repairs > 0 || foreign.length > 0 || ungrounded.length > 0) && (
          <div className="flex flex-wrap gap-1.5">
            {r.repairs > 0 && <Pill tone="warn">Câu SQL đầu lỗi, đã sửa {r.repairs} lần</Pill>}
            {foreign.length > 0 && <Pill tone="warn">Chữ không phải tiếng Việt: {foreign.join(", ")}</Pill>}
            {ungrounded.length > 0 && <Pill tone="warn">Số không có trong kết quả SQL: {ungrounded.join(", ")}</Pill>}
          </div>
        )}
        {(r.text_rewrites ?? []).length > 0 && (
          <p className="rounded-lg bg-sunken px-3 py-2 text-[12.5px] text-ink-2">
            Khớp chữ không dấu (dữ liệu hoá đơn lưu không dấu): {r.text_rewrites!.map((w) => `${w.from} → ${w.to}`).join("; ")}
          </p>
        )}
        {r.status === "ok" && (
          <div className="grid gap-1.5">
            <div className="flex flex-wrap justify-between gap-2 text-[12.5px] text-ink-3">
              <span>Kết quả SQL, {formatNumber(r.row_count)} dòng</span>
              {diff && diff.size > 0 && <span className="font-semibold text-warn">{diff.size} dòng khác mốc</span>}
            </div>
            <ResultTable columns={r.columns} rows={r.rows} differs={diff ? (i) => diff.has(i) : undefined} />
          </div>
        )}
        <SqlDetails result={r} />
      </>
    );
  }
  return (
    <article aria-label={endpoint.name} className="grid min-w-0 content-start overflow-hidden rounded-xl border border-line bg-surface">
      <header className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-b border-line bg-sunken px-3.5 py-2.5">
        <span className="flex min-w-0 items-center gap-2 font-bold"><Dot color={color} /><span className="truncate">{endpoint.name}</span></span>
        {isBaseline && <span className="rounded-md border border-line bg-surface px-1.5 text-[11.5px] font-semibold text-ink-2">Mốc</span>}
        <span className="ml-auto"><StatusPill answer={answer} now={now} /></span>
      </header>
      <div className="grid min-w-0 gap-3 p-3.5">{body}</div>
    </article>
  );
}

// Under the columns, as on Node 1: each endpoint against the baseline (design D15).
function Compare({ base, others, colors }: {
  base: { endpoint: Endpoint; result: QuestionResult };
  others: { endpoint: Endpoint; result: QuestionResult }[];
  colors: Record<string, string>;
}) {
  const b = base.result;
  return (
    <div className="grid gap-2 border-t border-line pt-4">
      <h3 className="text-[14px] font-bold">So với mốc {base.endpoint.name}</h3>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table aria-label={`So với mốc ${base.endpoint.name}`} className="w-full text-left text-[13.5px] [&_td]:[overflow-wrap:anywhere]">
          <thead className="bg-sunken text-[12px] text-ink-3">
            <tr><th className="px-3.5 py-1.5 font-medium">Endpoint</th><th className="px-3.5 py-1.5 font-medium">Kết quả</th><th className="px-3.5 py-1.5 font-medium">Tốc độ</th></tr>
          </thead>
          <tbody>
            {others.map(({ endpoint, result: r }) => {
              let verdict: ReactNode;
              if (r.status === "error" || b.status === "error") {
                verdict = r.status === b.status ? <b className="text-good">Cùng bị chặn</b> : <b className="text-bad">Khác mốc</b>;
              } else if (r.status === b.status && sameRow(r.rows as unknown[], b.rows as unknown[])) {
                verdict = <><b className="text-good">Giống mốc</b><small className="block text-[12.5px] text-ink-3">{r.rows.length ? `${r.rows.length}/${r.rows.length} dòng kết quả giống` : "cùng không có dữ liệu"}</small></>;
              } else {
                const total = Math.max(r.rows.length, b.rows.length);
                const differ = Array.from({ length: total }, (_, i) => !sameRow(r.rows[i], b.rows[i])).filter(Boolean).length;
                verdict = <><b className="text-warn">Khác mốc</b><small className="block text-[12.5px] text-ink-3">{differ}/{total} dòng khác</small></>;
              }
              const mine = firstSql(r)?.tpot_ms;
              const theirs = firstSql(b)?.tpot_ms;
              const speed = mine && theirs ? theirs / mine : null;
              return (
                <tr key={endpoint.endpoint_id} className="border-t border-line align-top">
                  <td className="px-3.5 py-2.5"><span className="flex items-center gap-2 font-bold"><Dot color={colors[endpoint.endpoint_id]} />{endpoint.name}</span></td>
                  <td className="px-3.5 py-2.5">{verdict}</td>
                  <td className="px-3.5 py-2.5">
                    {speed === null ? <span className="text-ink-3">Chưa đủ số đo</span> : (
                      <>
                        <b className={speed >= 1 ? "text-good" : "text-bad"}>{speed >= 1 ? `Nhanh hơn ${formatSpeedup(speed)}` : `Chậm hơn ${formatSpeedup(1 / speed)}`}</b>
                        <small className="block text-[12.5px] text-ink-3">theo TPOT{r.repairs > 0 ? `; phải sửa SQL ${r.repairs} lần` : ""}</small>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ViewsDrawer({ onClose }: { onClose: () => void }) {
  return (
    <Drawer title="Dữ liệu có thể hỏi" onClose={onClose}>
      <div className="grid gap-4">
        <p className="text-[13.5px] text-ink-2">Node 2 chỉ đọc 4 view dưới đây; câu lệnh ghi, xoá hoặc bảng khác đều bị chặn. Chữ trong dữ liệu được lưu không dấu.</p>
        {VIEWS.map((v) => (
          <section key={v.name} className="grid gap-1.5">
            <h3 className="flex flex-wrap items-baseline gap-2 text-[14px]">
              <code className="font-mono text-[13px] font-semibold">{v.name}</code>
              <span className="text-[13px] font-normal text-ink-3">{v.about}</span>
            </h3>
            <div className="flex flex-wrap gap-1">
              {v.columns.map((c) => <code key={c} className="rounded-md bg-sunken px-1.5 font-mono text-[12px] text-ink-2">{c}</code>)}
            </div>
          </section>
        ))}
      </div>
    </Drawer>
  );
}

export default function Node2Page({ api, endpoints, go }: { api: Api; endpoints: Endpoint[]; go: (page: PageId) => void }) {
  const baseline = endpoints.find((e) => e.is_baseline);
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints).filter((id) => id !== baseline?.endpoint_id));
  const [question, setQuestion] = useState("");
  const [round, setRound] = useState<Round | null>(null);
  const [views, setViews] = useState(false);
  const latest = useRef(0);
  const busy = round !== null && round.order.some((id) => !round.answers[id].result && !round.answers[id].error);
  const now = useNow(busy);

  const colors = endpointColors(endpoints);
  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const baseLive = baseline && baseline.status !== "down" ? baseline : undefined;
  const chosen = selected.filter((id) => id !== baseline?.endpoint_id && byId.get(id)?.status !== "down");
  const order = [...(baseLive ? [baseLive.endpoint_id] : []), ...chosen].slice(0, MAX_ENDPOINTS);

  // `started` is read by the event handler that asks.
  function ask(text: string, started: number) {
    const q = text.trim();
    if (!q || order.length === 0) return;
    const id = ++latest.current;
    setQuestion(q);
    setRound({ question: q, order, answers: Object.fromEntries(order.map((e) => [e, { started }])) });
    const settle = (endpointId: string, answer: Partial<Answer>) => setRound((current) => (
      current && latest.current === id
        ? { ...current, answers: { ...current.answers, [endpointId]: { ...current.answers[endpointId], ...answer } } }
        : current));
    for (const endpointId of order) {
      api.ask(q, endpointId).then(
        (result) => settle(endpointId, { result }),
        (exc) => settle(endpointId, { error: askError(exc) }),
      );
    }
  }

  const baseResult = round && baseLive && round.order.includes(baseLive.endpoint_id) ? round.answers[baseLive.endpoint_id].result : undefined;
  const others = round
    ? round.order.filter((id) => id !== baseLive?.endpoint_id).flatMap((id) => {
      const result = round.answers[id].result;
      const endpoint = byId.get(id);
      return result && endpoint ? [{ endpoint, result }] : [];
    })
    : [];

  if (endpoints.length === 0) return <><PageHead title="Hỏi bằng SQL" /><NeedEndpoint what="Node 2" onGo={() => go("endpoints")} /></>;

  return (
    <>
      <PageHead title="Hỏi bằng SQL" actions={<Button variant="ghost" icon={DatabaseZap} onClick={() => setViews(true)}>Dữ liệu có thể hỏi</Button>} />

      <section aria-label="Đặt câu hỏi" className="grid gap-3.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            ask(question, Date.now());
          }}
          className="rounded-xl border border-line bg-surface focus-within:border-lane-1 focus-within:ring-3 focus-within:ring-lane-1/20"
        >
          <label htmlFor="question" className="sr-only">Câu hỏi</label>
          <textarea
            id="question"
            rows={2}
            maxLength={300}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                ask(question, Date.now());
              }
            }}
            placeholder="Hỏi về dữ liệu hoá đơn, ví dụ: cửa hàng nào có tổng giá trị hoá đơn cao nhất?"
            className="block min-h-[60px] w-full resize-none bg-transparent px-4 pb-1.5 pt-3.5 text-base leading-snug outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-2 px-3 pb-2.5 pl-4">
            <span className="text-[13px] text-ink-3">Chạy trên</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              alwaysOn={baseLive ? [baseLive.endpoint_id] : []}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-(MAX_ENDPOINTS - 1))))}
            />
            <Button type="submit" icon={Play} loading={busy} disabled={busy || !question.trim() || order.length === 0} className="ml-auto">
              {busy ? "Đang hỏi" : "Hỏi"}
            </Button>
          </div>
        </form>
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Câu hỏi mẫu" role="group">
          <span className="mr-0.5 text-[13px] text-ink-3">Câu hỏi mẫu</span>
          {QUESTIONS.map((q) => (
            <button
              key={q.id}
              type="button"
              title={q.question}
              onClick={() => ask(q.question, Date.now())}
              disabled={busy}
              className={cx(
                "rounded-full border px-3 py-1 text-left text-[13px] text-ink-2 hover:bg-sunken hover:text-ink disabled:opacity-50",
                round?.question === q.question ? "border-ink font-semibold text-ink" : "border-line",
              )}
            >
              <b className="mr-1 font-semibold text-ink-3">{q.id}</b>{q.short}
            </button>
          ))}
        </div>
        {baseline && !baseLive && (
          <p className="text-[12.5px] text-ink-3">Mốc {baseline.name} đang dừng nên lượt hỏi này không có số so sánh.</p>
        )}
      </section>

      {round && (
        <section aria-label="Kết quả" className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 className="text-[15px] font-bold">Kết quả</h2>
            <span className="text-ink-2">“{round.question}”</span>
          </div>
          <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))]">
            {round.order.map((id) => {
              const endpoint = byId.get(id);
              return endpoint ? (
                <Column
                  key={id}
                  endpoint={endpoint}
                  color={colors[id]}
                  isBaseline={id === baseLive?.endpoint_id}
                  answer={round.answers[id]}
                  base={baseResult}
                  now={now}
                />
              ) : null;
            })}
          </div>
          {baseResult && baseLive && others.length > 0 && (
            <Compare base={{ endpoint: baseLive, result: baseResult }} others={others} colors={colors} />
          )}
        </section>
      )}

      {views && <ViewsDrawer onClose={() => setViews(false)} />}
    </>
  );
}
