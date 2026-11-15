"use client";

import { Check, Pencil, Plus, RefreshCw, Server, Terminal, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { PLATFORM_LABELS } from "@/components/AddEndpointDialog";
import { Button, Dot, EmptyState, Notice, PageHead, Panel, Pill, baseModel, clockTime, cx, endpointColors, secondsSince, useNow } from "@/components/ui";
import type { Api, Endpoint } from "@/lib/api";
import { formatMs } from "@/lib/format";

// What the last check of a row found, kept on the row (not as a page-wide error).
interface CheckOutcome {
  at: Date;
  ms: number;
  ok: boolean;
  message: string | null;
}

function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record };
  delete next[key];
  return next;
}

function metaLine(e: Endpoint): string {
  const m = e.meta ?? {};
  const parts: string[] = [];
  if (m.gpu) parts.push(m.gpu);
  if (m.draft_method) parts.push(`nháp ${m.draft_method}${m.draft_model ? ` (${m.draft_model})` : ""}`);
  else if (m.draft_method === null) parts.push("không dùng model nháp");
  if (m.adaptive !== undefined) parts.push(m.adaptive ? "Adaptive bật" : "Adaptive tắt");
  return parts.length ? parts.join(", ") : "Chưa rõ GPU và cách giải mã";
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
      action={<Button icon={Plus} onClick={onAdd}>Thêm endpoint</Button>}
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

function NameCell({ endpoint, color, onRename }: { endpoint: Endpoint; color: string; onRename: (name: string) => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(endpoint.name);
  if (editing) {
    return (
      <form
        className="flex items-center gap-1.5"
        onSubmit={async (event) => {
          event.preventDefault();
          if (name.trim() && name.trim() !== endpoint.name) await onRename(name.trim());
          setEditing(false);
        }}
      >
        <input
          aria-label="Tên endpoint"
          value={name}
          autoFocus
          onChange={(e) => setName(e.target.value)}
          className="w-44 rounded-md border border-line bg-surface px-2 py-1"
        />
        <button type="submit" aria-label="Lưu tên" className="rounded-md p-1 hover:bg-sunken"><Check aria-hidden className="size-4" /></button>
      </form>
    );
  }
  return (
    <div className="flex items-center gap-2">
      <Dot color={color} />
      <b className="whitespace-nowrap font-semibold">{endpoint.name}</b>
      <button type="button" aria-label={`Đổi tên ${endpoint.name}`} onClick={() => setEditing(true)} className="rounded-md p-1 text-ink-3 hover:bg-sunken">
        <Pencil aria-hidden className="size-3.5" />
      </button>
    </div>
  );
}

function StatusCell({ endpoint, checkingFor, outcome }: {
  endpoint: Endpoint;
  checkingFor: number | null; // seconds this check has been waiting, or null when idle
  outcome?: CheckOutcome;
}) {
  if (checkingFor !== null) {
    return (
      <Pill tone="idle">
        <span aria-hidden className="size-2 animate-pulse rounded-full bg-ink-3" />
        Đang kiểm tra · {checkingFor} s
      </Pill>
    );
  }
  const repeated = outcome?.message && outcome.message === endpoint.status_message;
  return (
    <>
      {endpoint.status === "ok" ? <Pill tone="good">Đang chạy</Pill> : <Pill tone="bad">Đã dừng</Pill>}
      {endpoint.status === "down" && endpoint.status_message && <div className="mt-1 max-w-[220px] text-[12px] text-ink-3">{endpoint.status_message}</div>}
      {outcome && (
        <div className={cx("mt-1 max-w-[240px] animate-fade-in text-[12px]", outcome.ok ? "text-ink-3" : "text-bad")}>
          Vừa kiểm tra <span className="whitespace-nowrap">{clockTime(outcome.at)} · {formatMs(outcome.ms)}</span>
          {!outcome.ok && outcome.message && !repeated ? `: ${outcome.message}` : ""}
        </div>
      )}
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
  const [busy, setBusy] = useState<string | null>(null);
  const [checking, setChecking] = useState<Record<string, number>>({});
  const [outcome, setOutcome] = useState<Record<string, CheckOutcome>>({});
  const [flash, setFlash] = useState<Record<string, "good" | "bad">>({});
  const running = Object.keys(checking).length;
  const now = useNow(running > 0);

  useEffect(() => {
    onCheckingChange?.(running);
  }, [running, onCheckingChange]);
  useEffect(() => () => onCheckingChange?.(0), [onCheckingChange]);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const colors = endpointColors(endpoints);
  const models = new Set(endpoints.map(baseModel));

  async function checkOne(id: string) {
    const started = Date.now();
    setChecking((current) => ({ ...current, [id]: started }));
    let found: CheckOutcome;
    try {
      const checked = await api.checkEndpoint(id);
      found = { at: new Date(), ms: Date.now() - started, ok: checked.status === "ok", message: checked.status === "ok" ? null : checked.status_message };
    } catch (exc) {
      found = { at: new Date(), ms: Date.now() - started, ok: false, message: (exc as Error).message };
    }
    setOutcome((current) => ({ ...current, [id]: found }));
    setFlash((current) => ({ ...current, [id]: found.ok ? "good" : "bad" }));
    setChecking((current) => without(current, id));
  }

  // Checks run side by side; the list is reloaded once, after the last one answers.
  async function check(ids: string[]) {
    await Promise.all(ids.map(checkOne));
    await onChange();
  }

  async function act(id: string, action: () => Promise<unknown>) {
    setBusy(id);
    setError(null);
    try {
      await action();
      await onChange();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHead
        title="Endpoint"
        actions={endpoints.length > 0 && (
          <>
            <Button
              variant="ghost"
              icon={RefreshCw}
              loading={running > 0}
              disabled={running > 0 || busy !== null}
              onClick={() => void check(endpoints.map((e) => e.endpoint_id))}
            >
              {running > 0 ? `Đang kiểm tra ${running} endpoint` : "Kiểm tra tất cả"}
            </Button>
            <Button icon={Plus} variant="ghost" onClick={onAdd}>Thêm endpoint</Button>
          </>
        )}
      >
        Các endpoint chuẩn OpenAI do CLI FlexSDS tạo ra. Endpoint mốc là chuẩn để tính “nhanh gấp mấy lần” ở mọi trang.
      </PageHead>

      {endpoints.length === 0 ? (
        <Onboarding onAdd={onAdd} />
      ) : (
        <Panel
          title="Danh sách endpoint"
          icon={Server}
          meta={<span>Trạng thái chỉ kiểm tra khi bạn bấm, để không đánh thức GPU ngoài ý muốn</span>}
          bodyClassName="p-0 gap-0"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] border-collapse text-left text-[13.5px]">
              <thead className="text-ink-3">
                <tr className="border-b border-line">
                  <th className="px-[18px] py-2.5 font-semibold">Tên</th>
                  <th className="px-3 py-2.5 font-semibold">Model và địa chỉ</th>
                  <th className="px-3 py-2.5 font-semibold">Nền tảng và cấu hình</th>
                  <th className="px-3 py-2.5 font-semibold">Trạng thái</th>
                  <th className="px-3 py-2.5 font-semibold">Mốc</th>
                  <th className="px-[18px] py-2.5 text-right font-semibold">Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {endpoints.map((e) => (
                  <tr
                    key={e.endpoint_id}
                    className={cx("border-b border-line last:border-0 align-top",
                      flash[e.endpoint_id] === "good" && "animate-row-good", flash[e.endpoint_id] === "bad" && "animate-row-bad")}
                    onAnimationEnd={() => setFlash((current) => without(current, e.endpoint_id))}
                  >
                    <td className="px-[18px] py-3">
                      <NameCell
                        endpoint={e}
                        color={colors[e.endpoint_id]}
                        onRename={(name) => act(e.endpoint_id, () => api.updateEndpoint(e.endpoint_id, { name }))}
                      />
                      {e.has_key && <span className="ml-[18px] text-[12px] text-ink-3">đã đặt khoá API</span>}
                    </td>
                    <td className="max-w-[320px] px-3 py-3">
                      <div className="font-semibold">{e.model_id}</div>
                      <div className="truncate font-mono text-[12.5px] text-ink-3" title={e.base_url}>{e.base_url}</div>
                    </td>
                    <td className="px-3 py-3">
                      <div>{PLATFORM_LABELS[e.platform] ?? e.platform}{e.max_model_len ? `, ngữ cảnh ${e.max_model_len.toLocaleString("vi-VN")}` : ""}</div>
                      <div className="text-[12.5px] text-ink-3">{metaLine(e)}</div>
                    </td>
                    <td className="px-3 py-3">
                      <StatusCell
                        endpoint={e}
                        checkingFor={checking[e.endpoint_id] === undefined ? null : secondsSince(checking[e.endpoint_id], now)}
                        outcome={outcome[e.endpoint_id]}
                      />
                    </td>
                    <td className="px-3 py-3">
                      <label className="inline-flex cursor-pointer items-center gap-2">
                        <input
                          type="radio"
                          name="baseline"
                          checked={e.is_baseline}
                          disabled={busy !== null}
                          onChange={() => act(e.endpoint_id, () => api.updateEndpoint(e.endpoint_id, { is_baseline: true }))}
                          aria-label={`Chọn ${e.name} làm mốc`}
                        />
                        {e.is_baseline ? "Mốc" : ""}
                      </label>
                    </td>
                    <td className="px-[18px] py-3">
                      <div className="flex justify-end gap-2">
                        {confirmDelete === e.endpoint_id ? (
                          <>
                            <span className="self-center text-[13px] text-ink-2">Xoá endpoint này?</span>
                            <Button variant="danger" disabled={busy !== null} onClick={() => act(e.endpoint_id, () => api.deleteEndpoint(e.endpoint_id)).then(() => setConfirmDelete(null))}>
                              Xoá
                            </Button>
                            <Button variant="ghost" onClick={() => setConfirmDelete(null)}>Giữ lại</Button>
                          </>
                        ) : (
                          <>
                            <Button
                              variant="ghost"
                              icon={RefreshCw}
                              loading={checking[e.endpoint_id] !== undefined}
                              disabled={checking[e.endpoint_id] !== undefined || busy !== null}
                              onClick={() => void check([e.endpoint_id])}
                            >
                              {checking[e.endpoint_id] === undefined ? "Kiểm tra lại" : "Đang kiểm tra"}
                            </Button>
                            <Button
                              variant="danger"
                              icon={Trash2}
                              aria-label={`Xoá ${e.name}`}
                              disabled={checking[e.endpoint_id] !== undefined}
                              onClick={() => setConfirmDelete(e.endpoint_id)}
                            />
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {error && <Notice tone="bad">{error}</Notice>}
      {endpoints.length > 0 && !endpoints.some((e) => e.is_baseline) && (
        <Notice tone="info">Chưa chọn endpoint mốc: các trang chỉ hiện số tuyệt đối (thời gian, TTFT, TPOT), không hiện hệ số tăng tốc.</Notice>
      )}
      {models.size > 1 && (
        <Notice tone="warn">Các endpoint đang chạy {models.size} model khác nhau. So sánh tốc độ giữa hai model khác nhau không công bằng.</Notice>
      )}
      {endpoints.length > 0 && (
        <Panel title="Endpoint trên máy cá nhân" icon={Terminal}>
          <p className="text-ink-2">
            Web chạy trên mạng nên không gọi được <code className="font-mono">http://127.0.0.1</code>. Mở đường hầm trên máy đang chạy flexsds rồi dán địa chỉ
            {" "}<code className="font-mono">https://….trycloudflare.com/v1</code> mà lệnh in ra. Colab dùng cùng cách này.
          </p>
          <pre className="overflow-x-auto rounded-lg bg-sunken px-4 py-3 font-mono text-[13px]">cloudflared tunnel --url http://127.0.0.1:8000</pre>
        </Panel>
      )}
    </>
  );
}
