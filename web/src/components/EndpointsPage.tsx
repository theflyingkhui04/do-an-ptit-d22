"use client";

import { Pencil, Plus, RefreshCw, Server, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import EndpointEditDrawer from "@/components/EndpointEditDrawer";
import { Button, ConfirmDialog, CopyButton, Dot, EmptyState, Notice, PageHead, Pill, baseModel, cx, endpointColors, secondsSince, useNow } from "@/components/ui";
import type { Api, Endpoint } from "@/lib/api";
import { agoVi, detailLine, setupLine, shortReason, shortUrl } from "@/lib/endpoints";
import { formatMs } from "@/lib/format";
import { durationVi } from "@/lib/liveness";

// A sleeping GPU on Modal answers 503 for the 2-3 minutes it takes to start: ask again until it is up.
export const WAKE_RETRY_MS = 20_000;
export const WAKE_LIMIT_MS = 5 * 60_000;

// What the last check of a row found in this session; fresher than the list, which is reloaded once at the end.
interface CheckOutcome {
  at: number;
  ms: number;
  status: Endpoint["status"] | "error"; // error: the backend itself could not be called
  message: string | null;
  gaveUp?: boolean; // still waking after WAKE_LIMIT_MS
}

function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record };
  delete next[key];
  return next;
}

// Kept in step with the real CLI by tests/test_web_cli_sample.py (argparse of `flexsds serve`,
// the header printed by serving_session.py and the Modal app name in modal_app.py).
export const SERVE_SAMPLE =
  "flexsds serve --backend modal --gpu A10G --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft duckhoi/qwen2.5-7b-eagle3.1-speculator";
export const SERVE_SAMPLE_HEADER = "Endpoint LLM created:";
export const SERVE_SAMPLE_API = "https://…--flexsds-serving-serve.modal.run/v1";

function Onboarding({ onAdd }: { onAdd: () => void }) {
  const steps = [
    { title: "Chạy flexsds serve", body: "Trên Modal, Colab hoặc GPU của máy. CLI dựng vLLM và chọn cách giải mã hợp với GPU." },
    { title: "Copy dòng API", body: `Khi CLI báo “${SERVE_SAMPLE_HEADER.replace(/:$/, "")}”, copy địa chỉ ở dòng API. Trên máy cá nhân, mở đường hầm cloudflared trước.` },
    { title: "Dán vào web", body: "Web tự nhận model và độ dài ngữ cảnh. Dán nhiều endpoint để so sánh với nhau." },
  ];
  return (
    <EmptyState
      icon={Server}
      title="Chưa có endpoint nào"
      action={<Button icon={Plus} onClick={onAdd}>Thêm endpoint đầu tiên</Button>}
    >
      <p>Mọi node của pipeline chạy trên endpoint do CLI FlexSDS tạo ra. Thêm ít nhất một endpoint để bắt đầu.</p>
      <ol className="mt-4 grid gap-3 sm:grid-cols-3">
        {steps.map((s, i) => (
          <li key={s.title} className="grid content-start gap-1 rounded-xl bg-sunken p-3.5">
            <span className="grid size-6 place-items-center rounded-md bg-ink text-xs font-bold text-surface">{i + 1}</span>
            <b className="text-ink">{s.title}</b>
            <span className="text-[13.5px]">{s.body}</span>
          </li>
        ))}
      </ol>
      <pre data-testid="serve-sample" className="mt-4 whitespace-pre-wrap break-words rounded-lg bg-rail px-4 py-3 font-mono text-[13px] leading-relaxed text-rail-ink">
        <span className="text-[#7d8a97]">$ </span>{SERVE_SAMPLE}{"\n"}
        <span className="text-[#7d8a97]">{SERVE_SAMPLE_HEADER}{"\n"}- API: </span><span className="text-white">{SERVE_SAMPLE_API}</span>
      </pre>
    </EmptyState>
  );
}

function StatusCell({ endpoint: e, checkingFor, waking, outcome, now }: {
  endpoint: Endpoint;
  checkingFor: number | null; // seconds this check has been waiting, or null when idle
  waking: boolean; // the GPU answered "starting" and is being asked again
  outcome?: CheckOutcome;
  now: number;
}) {
  if (checkingFor !== null && waking) {
    return (
      <>
        <Pill tone="warn">
          <span aria-hidden className="size-2 animate-pulse rounded-full bg-warn" />
          Đang khởi động GPU, {durationVi(checkingFor)}
        </Pill>
        <span className="text-[12.5px] text-ink-3">Tự kiểm tra lại mỗi {WAKE_RETRY_MS / 1000} s tới khi GPU lên, tối đa {WAKE_LIMIT_MS / 60_000} phút.</span>
      </>
    );
  }
  if (checkingFor !== null) {
    return (
      <Pill tone="idle">
        <span aria-hidden className="size-2 animate-pulse rounded-full bg-ink-3" />
        Đang kiểm tra, {checkingFor} s
      </Pill>
    );
  }
  const fresh = outcome && outcome.status !== "error" ? outcome : null;
  const status = fresh ? fresh.status : e.status;
  const reason = fresh ? fresh.message : e.status_message;
  const checkedAt = outcome ? new Date(outcome.at).toISOString() : e.checked_at;
  const failedCall = outcome?.status === "error" ? outcome.message : null;
  return (
    <>
      {status === "ok" && <Pill tone="good">Đang chạy</Pill>}
      {status === "starting" && !fresh?.gaveUp && <Pill tone="warn">Chưa sẵn sàng</Pill>}
      {(status === "down" || fresh?.gaveUp) && <Pill tone="bad">Đã dừng</Pill>}
      {fresh?.gaveUp ? (
        <span className="text-[12.5px] text-bad">
          GPU chưa lên sau {WAKE_LIMIT_MS / 60_000} phút. Xem lại phiên flexsds serve rồi bấm Kiểm tra.
        </span>
      ) : status !== "ok" && reason && (
        <span className={cx("text-[12.5px]", status === "down" ? "text-bad" : "text-ink-2")} title={reason}>{shortReason(reason)}</span>
      )}
      <span className="num text-[12.5px] text-ink-3">
        Kiểm tra {agoVi(checkedAt, now)}{outcome ? `, mất ${formatMs(outcome.ms)}` : ""}
      </span>
      {failedCall && <span className="animate-fade-in text-[12.5px] text-bad">{failedCall}</span>}
    </>
  );
}

export default function EndpointsPage({ api, endpoints, onChange, onAdd, onCheckingChange }: {
  api: Api;
  endpoints: Endpoint[];
  onChange: () => Promise<unknown>;
  onAdd: () => void;
  // How many checks are running, for the spinner in the top bar.
  onCheckingChange?: (count: number) => void;
}) {
  const [checking, setChecking] = useState<Record<string, number>>({});
  const [waking, setWaking] = useState<Record<string, true>>({});
  const [outcome, setOutcome] = useState<Record<string, CheckOutcome>>({});
  // Waking GPUs are asked again only while this page is open.
  const open = useRef(true);
  useEffect(() => {
    open.current = true;
    return () => {
      open.current = false;
    };
  }, []);
  const [flash, setFlash] = useState<Record<string, "good" | "bad">>({});
  const [editing, setEditing] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const running = Object.keys(checking).length;
  // Every second while a check runs (its counter), else every 30 s ("Kiểm tra 3 phút trước").
  const now = useNow(true, running > 0 ? 1000 : 30_000);

  useEffect(() => {
    onCheckingChange?.(running);
  }, [running, onCheckingChange]);
  useEffect(() => () => onCheckingChange?.(0), [onCheckingChange]);

  const colors = endpointColors(endpoints);
  const models = new Set(endpoints.map(baseModel));
  const ordered = [...endpoints].sort((a, b) => Number(b.is_baseline) - Number(a.is_baseline)); // baseline first
  const editingEndpoint = endpoints.find((e) => e.endpoint_id === editing);
  const confirming = endpoints.find((e) => e.endpoint_id === confirm);
  const closeEditor = useCallback(() => setEditing(null), []);

  // Asks again every WAKE_RETRY_MS while the GPU says it is starting, for at most WAKE_LIMIT_MS.
  async function checkOne(id: string) {
    const started = Date.now();
    setChecking((current) => ({ ...current, [id]: started }));
    let found: CheckOutcome;
    for (;;) {
      let checked: Endpoint;
      try {
        checked = await api.checkEndpoint(id);
      } catch (exc) {
        found = { at: Date.now(), ms: Date.now() - started, status: "error", message: (exc as Error).message };
        break;
      }
      const at = Date.now();
      if (checked.status === "starting" && at - started < WAKE_LIMIT_MS && open.current) {
        setWaking((current) => ({ ...current, [id]: true }));
        await new Promise((resolve) => setTimeout(resolve, WAKE_RETRY_MS));
        if (!open.current) return;
        continue;
      }
      found = { at, ms: at - started, status: checked.status, message: checked.status === "ok" ? null : checked.status_message,
                gaveUp: checked.status === "starting" };
      break;
    }
    if (!open.current) return;
    setOutcome((current) => ({ ...current, [id]: found }));
    setFlash((current) => ({ ...current, [id]: found.status === "ok" ? "good" : "bad" }));
    setWaking((current) => without(current, id));
    setChecking((current) => without(current, id));
  }

  // Checks run side by side; the list is reloaded once, after the last one answers.
  async function check(ids: string[]) {
    await Promise.all(ids.map(checkOne));
    if (open.current) await onChange();
  }

  async function remove(id: string) {
    setDeleting(true);
    setError(null);
    try {
      await api.deleteEndpoint(id);
      setConfirm(null);
      setEditing(null);
      await onChange();
    } catch (exc) {
      setConfirm(null);
      setError((exc as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <>
      <PageHead
        title="Endpoint"
        actions={endpoints.length > 0 && (
          <Button
            variant="ghost"
            icon={RefreshCw}
            loading={running > 0}
            disabled={running > 0}
            onClick={() => void check(endpoints.map((e) => e.endpoint_id))}
          >
            {running > 0 ? `Đang kiểm tra ${running} endpoint` : "Kiểm tra tất cả"}
          </Button>
        )}
      />

      {endpoints.length === 0 ? (
        <Onboarding onAdd={onAdd} />
      ) : (
        <section aria-label="Danh sách endpoint" className="overflow-hidden rounded-xl border border-line bg-surface">
          {ordered.map((e) => {
            const setup = setupLine(e);
            const busy = checking[e.endpoint_id] !== undefined;
            return (
              <article
                key={e.endpoint_id}
                aria-label={e.name}
                className={cx(
                  "grid gap-x-5 gap-y-3 border-line px-[18px] py-4 [&+&]:border-t",
                  "md:grid-cols-2 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1.2fr)_minmax(190px,0.85fr)_auto]",
                  flash[e.endpoint_id] === "good" && "animate-row-good", flash[e.endpoint_id] === "bad" && "animate-row-bad",
                )}
                onAnimationEnd={() => setFlash((current) => without(current, e.endpoint_id))}
              >
                <div className="grid min-w-0 content-start gap-1">
                  <div className="flex min-w-0 items-center gap-2">
                    <Dot color={colors[e.endpoint_id]} />
                    <button
                      type="button"
                      title="Xem và sửa"
                      onClick={() => setEditing(e.endpoint_id)}
                      className="truncate text-left text-[15px] font-bold hover:underline"
                    >
                      {e.name}
                    </button>
                    {e.is_baseline && <span className="shrink-0 rounded-md border border-line px-1.5 text-[12px] font-semibold text-ink-2">Mốc</span>}
                  </div>
                  <div className="flex min-w-0 items-center gap-1 font-mono text-[12.5px] text-ink-3">
                    <span className="truncate" title={e.base_url}>{shortUrl(e.base_url)}</span>
                    <CopyButton text={e.base_url} label={`Copy địa chỉ ${e.name}`} />
                  </div>
                  {e.model_id !== e.name && (
                    <small className="text-[12.5px] text-ink-3">Model endpoint báo: <span className="text-ink-2">{e.model_id}</span></small>
                  )}
                </div>

                <div className="grid min-w-0 content-start gap-0.5 text-[13.5px]">
                  {setup && <span className="font-semibold">{setup}</span>}
                  <span className={setup ? "text-[12.5px] text-ink-3" : "text-ink-2"}>{detailLine(e, baseModel(e))}</span>
                </div>

                <div className="grid content-start justify-items-start gap-1" aria-live="polite">
                  <StatusCell
                    endpoint={e}
                    checkingFor={busy ? secondsSince(checking[e.endpoint_id], now) : null}
                    waking={waking[e.endpoint_id] === true}
                    outcome={outcome[e.endpoint_id]}
                    now={now}
                  />
                </div>

                <div className="flex flex-wrap items-start gap-1.5 md:col-span-2 lg:col-span-1 lg:justify-end">
                  <Button variant="ghost" icon={RefreshCw} loading={busy} disabled={busy} aria-label={`Kiểm tra ${e.name}`}
                    className="px-2.5 py-1.5 text-[13px]" onClick={() => void check([e.endpoint_id])}>
                    Kiểm tra
                  </Button>
                  <Button variant="ghost" icon={Pencil} aria-label={`Sửa ${e.name}`} className="px-2.5 py-1.5 text-[13px]"
                    onClick={() => setEditing(e.endpoint_id)}>
                    Sửa
                  </Button>
                  <Button variant="danger" icon={Trash2} aria-label={`Xoá ${e.name}`} disabled={busy} className="px-2.5 py-1.5 text-[13px]"
                    onClick={() => setConfirm(e.endpoint_id)}>
                    Xoá
                  </Button>
                </div>
              </article>
            );
          })}
        </section>
      )}

      {error && <Notice tone="bad">{error}</Notice>}
      {endpoints.length > 0 && !endpoints.some((e) => e.is_baseline) && (
        <Notice tone="info">Chưa chọn endpoint mốc: các trang chỉ hiện số tuyệt đối (thời gian, TTFT, TPOT), không hiện hệ số tăng tốc. Chọn mốc trong ngăn Sửa.</Notice>
      )}
      {models.size > 1 && (
        <Notice tone="warn">Các endpoint đang chạy {models.size} model khác nhau. So sánh tốc độ giữa hai model khác nhau không công bằng.</Notice>
      )}

      {editingEndpoint && (
        <EndpointEditDrawer
          key={editingEndpoint.endpoint_id}
          api={api}
          endpoint={editingEndpoint}
          paused={confirm !== null}
          onClose={closeEditor}
          onSaved={onChange}
          onDelete={() => setConfirm(editingEndpoint.endpoint_id)}
        />
      )}
      {confirming && (
        <ConfirmDialog
          title={`Xoá ${confirming.name}?`}
          confirmLabel="Xoá endpoint"
          confirmIcon={Trash2}
          busy={deleting}
          onConfirm={() => void remove(confirming.endpoint_id)}
          onCancel={() => setConfirm(null)}
        >
          <ul className="grid list-disc gap-1 pl-5 text-ink-2">
            <li>Endpoint biến mất khỏi mọi trang của web.</li>
            <li>GPU trên Modal và phiên <code className="font-mono">flexsds serve</code> không bị tắt; muốn dùng lại thì thêm lại địa chỉ.</li>
            <li>Hoá đơn và các lần đo đã lưu vẫn giữ nguyên.</li>
            {confirming.is_baseline && (
              <li><b className="text-ink">Đây là endpoint mốc:</b> các trang không có số so sánh cho tới khi bạn chọn mốc mới.</li>
            )}
          </ul>
        </ConfirmDialog>
      )}
    </>
  );
}
