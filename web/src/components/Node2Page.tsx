"use client";

import { ChevronRight, DatabaseZap, Play, ShieldCheck } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";

import type { PageId } from "@/components/Console";
import {
  Button,
  Dot,
  Drawer,
  EndpointChips,
  NeedEndpoint,
  PageHead,
  Pill,
  ReferencePicker,
  ResultTable,
  Spinner,
  SqlBlock,
  cx,
  defaultTarget,
  endpointColors,
  secondsSince,
  useNow,
} from "@/components/ui";
import { ApiError, type Api, type Endpoint, type QuestionResult } from "@/lib/api";
import { formatMs, formatNumber, formatSpeedup } from "@/lib/format";
import { useI18n, type Dict } from "@/lib/i18n";
import { VIEWS, getQuestions } from "@/lib/views";

const MAX_ENDPOINTS = 6;

interface Answer {
  started: number;
  result?: QuestionResult;
  error?: string;
}

interface Round {
  question: string;
  order: string[];
  answers: Record<string, Answer>;
}

function askError(exc: unknown, t: Dict): string {
  if (exc instanceof ApiError && exc.status === 429) return t.errRateLimit;
  return (exc as Error).message;
}

const firstSql = (r: QuestionResult) => r.calls.find((c) => c.step === "node2");
const totalMs = (r: QuestionResult) => r.calls.reduce((sum, c) => sum + (c.wall_ms ?? c.latency_ms ?? 0), 0);
const sameRow = (a: unknown[] | undefined, b: unknown[] | undefined) => JSON.stringify(a) === JSON.stringify(b);

function differentRows(r: QuestionResult, base: QuestionResult | undefined): Set<number> | null {
  if (!base || base === r || r.status !== "ok" || base.status !== "ok") return null;
  const rows = new Set<number>();
  r.rows.forEach((row, i) => {
    if (!sameRow(row, base.rows[i])) rows.add(i);
  });
  return rows;
}

function StatusPill({ answer, now }: { answer: Answer; now: number }) {
  const { t } = useI18n();
  if (answer.error) return <Pill tone="bad">{t.statusError}</Pill>;
  const r = answer.result;
  if (!r) return <Pill tone="idle"><Spinner className="size-3.5" />{t.statusAsking(secondsSince(answer.started, now))}</Pill>;
  if (r.status === "ok") return <Pill tone="good">{t.statusAnswered}</Pill>;
  if (r.status === "no_data") return <Pill tone="idle">{t.statusNoData}</Pill>;
  return <Pill tone="bad">{t.statusBlocked}</Pill>;
}

function Stats({ result }: { result: QuestionResult }) {
  const { lang, t } = useI18n();
  const first = firstSql(result);
  const cells: [string, string][] = [
    [t.statLatency, formatMs(totalMs(result), lang)],
    ["TTFT", formatMs(first?.ttft_ms, lang)],
    ["TPOT", formatMs(first?.tpot_ms, lang)],
    [t.statTokens, formatNumber(first?.completion_tokens, lang)],
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
  const { t } = useI18n();
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
        <span>{t.sqlQuery}</span>
        {!blocked && (
          <span className="ml-auto inline-flex items-center gap-1 font-medium text-good"><ShieldCheck aria-hidden className="size-3.5" />{t.sqlValidated}</span>
        )}
      </summary>
      <div className="grid gap-2 border-t border-line p-2.5">
        <SqlBlock sql={result.sql} />
        {result.first_error && <p className="text-[12.5px] text-ink-3">{t.sqlFirstError} <span className="font-mono">{result.first_error}</span></p>}
      </div>
    </details>
  );
}

function Column({ endpoint, color, answer, base, now }: {
  endpoint: Endpoint;
  color: string;
  answer: Answer;
  base?: QuestionResult;
  now: number;
}) {
  const { lang, t } = useI18n();
  const r = answer.result;
  const diff = r ? differentRows(r, base) : null;
  const foreign = r?.foreign_words ?? [];
  const ungrounded = r?.ungrounded_numbers ?? [];
  let body: ReactNode;
  if (answer.error) {
    body = <p role="alert" className="text-[14px] text-bad">{answer.error}</p>;
  } else if (!r) {
    body = <p className="text-ink-3">{t.node2RunningText}</p>;
  } else {
    body = (
      <>
        {r.status === "error" ? (
          <p role="alert" className="min-h-[3em] text-[14px] text-bad">{r.error}</p>
        ) : r.answer ? (
          <p className="min-h-[3em] text-base leading-normal">{r.answer}</p>
        ) : (
          <p className="min-h-[3em] text-ink-3">
            {r.status === "no_data" ? t.node2NoDataText : t.node2NoAnswerText}
          </p>
        )}
        <Stats result={r} />
        {(r.repairs > 0 || foreign.length > 0 || ungrounded.length > 0) && (
          <div className="flex flex-wrap gap-1.5">
            {r.repairs > 0 && <Pill tone="warn">{t.sqlRepaired(r.repairs)}</Pill>}
            {foreign.length > 0 && <Pill tone="warn">{t.foreignWordsFlag(foreign.join(", "))}</Pill>}
            {ungrounded.length > 0 && <Pill tone="warn">{t.ungroundedNumbersFlag(ungrounded.join(", "))}</Pill>}
          </div>
        )}
        {r.status === "ok" && (
          <div className="grid gap-1.5">
            <div className="flex flex-wrap justify-between gap-2 text-[12.5px] text-ink-3">
              <span>{t.sqlRowsCount(formatNumber(r.row_count, lang))}</span>
              {diff && diff.size > 0 && <span className="font-semibold text-warn">{t.diffRowsCount(diff.size)}</span>}
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
        <span className="ml-auto"><StatusPill answer={answer} now={now} /></span>
      </header>
      <div className="grid min-w-0 gap-3 p-3.5">{body}</div>
    </article>
  );
}

function Compare({ base, others, colors }: {
  base: { endpoint: Endpoint; result: QuestionResult };
  others: { endpoint: Endpoint; result: QuestionResult }[];
  colors: Record<string, string>;
}) {
  const { lang, t } = useI18n();
  const b = base.result;
  return (
    <div className="grid gap-2 border-t border-line pt-4">
      <h3 className="text-[14px] font-bold">{t.compareTableTitle(base.endpoint.name)}</h3>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table role="table" aria-label={t.compareTableTitle(base.endpoint.name)} className="w-full text-left text-[13.5px] [&_td]:[overflow-wrap:anywhere] max-sm:block">
          <thead className="bg-sunken text-[12px] text-ink-3 max-sm:hidden">
            <tr><th className="px-3.5 py-1.5 font-medium">{t.colEndpoint}</th><th className="px-3.5 py-1.5 font-medium">{t.colResult}</th><th className="px-3.5 py-1.5 font-medium">{t.colSpeed}</th></tr>
          </thead>
          <tbody className="max-sm:block">
            {others.map(({ endpoint, result: r }) => {
              let verdict: ReactNode;
              if (r.status === "error" || b.status === "error") {
                verdict = r.status === b.status ? <b className="text-good">{t.bothBlocked}</b> : <b className="text-bad">{t.differsRef}</b>;
              } else if (r.status === b.status && sameRow(r.rows as unknown[], b.rows as unknown[])) {
                verdict = <><b className="text-good">{t.matchesRef}</b><small className="block text-[12.5px] text-ink-3">{r.rows.length ? t.rowsMatchDetail(r.rows.length) : t.bothEmptyDetail}</small></>;
              } else {
                const total = Math.max(r.rows.length, b.rows.length);
                const differ = Array.from({ length: total }, (_, i) => !sameRow(r.rows[i], b.rows[i])).filter(Boolean).length;
                verdict = <><b className="text-warn">{t.differsRef}</b><small className="block text-[12.5px] text-ink-3">{t.rowsDiffDetail(differ, total)}</small></>;
              }
              const mine = firstSql(r)?.tpot_ms;
              const theirs = firstSql(b)?.tpot_ms;
              const speed = mine && theirs ? theirs / mine : null;
              return (
                <tr key={endpoint.endpoint_id} role="row" className="border-t border-line align-top max-sm:grid max-sm:gap-0.5 max-sm:py-1.5">
                  <td data-label={t.colEndpoint} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]"><span className="flex items-center gap-2 font-bold"><Dot color={colors[endpoint.endpoint_id]} />{endpoint.name}</span></td>
                  <td data-label={t.colResult} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">{verdict}</td>
                  <td data-label={t.colSpeed} className="px-3.5 py-2.5 max-sm:block max-sm:py-1 max-sm:before:block max-sm:before:text-[11.5px] max-sm:before:text-ink-3 max-sm:before:content-[attr(data-label)]">
                    {speed === null ? <span className="text-ink-3">{t.insufficientMetrics}</span> : (
                      <>
                        <b className={speed >= 1 ? "text-good" : "text-bad"}>{speed >= 1 ? t.fasterBy(formatSpeedup(speed, lang)) : t.slowerBy(formatSpeedup(1 / speed, lang))}</b>
                        {r.repairs > 0 && <small className="block text-[12.5px] text-ink-3">{t.repairedNote(r.repairs)}</small>}
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
  const { t } = useI18n();
  return (
    <Drawer title={t.dataSchemaTitle} onClose={onClose}>
      <div className="grid gap-4">
        <p className="text-[13.5px] text-ink-2">{t.dataSchemaIntro}</p>
        {VIEWS.map((v) => (
          <section key={v.name} className="grid gap-1.5">
            <h3 className="text-[14px]">
              <code className="font-mono text-[13px] font-semibold">{v.name}</code>
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
  const { lang, t } = useI18n();
  const [selected, setSelected] = useState<string[]>(() => defaultTarget(endpoints));
  const [question, setQuestion] = useState("");
  const [round, setRound] = useState<Round | null>(null);
  const [refChoice, setRefChoice] = useState<string | null>(null);
  const [views, setViews] = useState(false);
  const latest = useRef(0);
  const busy = round !== null && round.order.some((id) => !round.answers[id].result && !round.answers[id].error);
  const now = useNow(busy);

  const colors = endpointColors(endpoints);
  const byId = new Map(endpoints.map((e) => [e.endpoint_id, e]));
  const order = selected.filter((id) => byId.get(id)?.status !== "down").slice(0, MAX_ENDPOINTS);
  const questions = getQuestions(lang);

  function ask(text: string, started: number) {
    const q = text.trim();
    if (!q || order.length === 0) return;
    const id = ++latest.current;
    setQuestion(q);
    setRefChoice(null);
    setRound({ question: q, order, answers: Object.fromEntries(order.map((e) => [e, { started }])) });
    const settle = (endpointId: string, answer: Partial<Answer>) => setRound((current) => (
      current && latest.current === id
        ? { ...current, answers: { ...current.answers, [endpointId]: { ...current.answers[endpointId], ...answer } } }
        : current));
    for (const endpointId of order) {
      api.ask(q, endpointId).then(
        (result) => settle(endpointId, { result }),
        (exc) => settle(endpointId, { error: askError(exc, t) }),
      );
    }
  }

  const roundEndpoints = round ? round.order.map((id) => byId.get(id)).filter((e): e is Endpoint => Boolean(e)) : [];
  const referenceId = roundEndpoints.length >= 2
    ? (refChoice && roundEndpoints.some((e) => e.endpoint_id === refChoice) ? refChoice : roundEndpoints[0].endpoint_id)
    : null;
  const refEndpoint = referenceId ? byId.get(referenceId) : undefined;
  const refResult = round && referenceId ? round.answers[referenceId]?.result : undefined;
  const others = round && referenceId
    ? round.order.filter((id) => id !== referenceId).flatMap((id) => {
      const result = round.answers[id]?.result;
      const endpoint = byId.get(id);
      return result && endpoint ? [{ endpoint, result }] : [];
    })
    : [];

  if (endpoints.length === 0) return <><PageHead title={t.pageNode2} /><NeedEndpoint what="Node 2" onGo={() => go("endpoints")} /></>;

  return (
    <>
      <PageHead title={t.pageNode2} actions={<Button variant="ghost" icon={DatabaseZap} onClick={() => setViews(true)}>{t.dataSchemaBtn}</Button>} />

      <section aria-label={t.questionSectionAria} className="grid gap-3.5 rounded-xl border border-line bg-surface p-[18px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            ask(question, Date.now());
          }}
          className="rounded-xl border border-line bg-surface focus-within:border-lane-1 focus-within:ring-3 focus-within:ring-lane-1/20"
        >
          <label htmlFor="question" className="sr-only">{t.questionLabel}</label>
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
            placeholder={t.questionPlaceholder}
            className="block min-h-[60px] w-full resize-none bg-transparent px-4 pb-1.5 pt-3.5 text-base leading-snug outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-2 px-3 pb-2.5 pl-4">
            <span className="text-[13px] text-ink-3">{t.selectLlm}</span>
            <EndpointChips
              endpoints={endpoints}
              selected={selected}
              onToggle={(id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-MAX_ENDPOINTS)))}
            />
            <Button type="submit" icon={Play} loading={busy} disabled={busy || !question.trim() || order.length === 0} className="ml-auto">
              {busy ? t.runningQueryBtn : t.runQueryBtn}
            </Button>
          </div>
        </form>
        <div className="flex flex-wrap items-center gap-1.5" aria-label={t.sampleQuestionsLabel} role="group">
          <span className="mr-0.5 text-[13px] text-ink-3">{t.sampleQuestionsLabel}</span>
          {questions.map((q) => (
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
      </section>

      {round && (
        <section aria-label={t.resultsPanel} className="grid gap-4 rounded-xl border border-line bg-surface p-[18px]">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h2 className="text-[15px] font-bold">{t.resultsPanel}</h2>
            <span className="text-ink-2">“{round.question}”</span>
            {referenceId && (
              <span className="ml-auto">
                <ReferencePicker
                  options={roundEndpoints.map((e) => ({ id: e.endpoint_id, name: e.name }))}
                  value={referenceId}
                  onChange={setRefChoice}
                  colors={colors}
                />
              </span>
            )}
          </div>
          <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))]">
            {round.order.map((id) => {
              const endpoint = byId.get(id);
              return endpoint ? (
                <Column
                  key={id}
                  endpoint={endpoint}
                  color={colors[id]}
                  answer={round.answers[id]}
                  base={refResult}
                  now={now}
                />
              ) : null;
            })}
          </div>
          {refResult && refEndpoint && others.length > 0 && (
            <Compare base={{ endpoint: refEndpoint, result: refResult }} others={others} colors={colors} />
          )}
        </section>
      )}

      {views && <ViewsDrawer onClose={() => setViews(false)} />}
    </>
  );
}
