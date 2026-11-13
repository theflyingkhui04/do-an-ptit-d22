"use client";

import { Check, Pencil, Plus, RefreshCw, Server, Terminal, Trash2 } from "lucide-react";
import { useState } from "react";

import { PLATFORM_LABELS } from "@/components/AddEndpointDialog";
import { Button, Dot, EmptyState, Notice, PageHead, Panel, Pill, endpointColors } from "@/components/ui";
import type { Api, Endpoint } from "@/lib/api";

function metaLine(e: Endpoint): string {
  const m = e.meta ?? {};
  const parts: string[] = [];
  if (m.gpu) parts.push(m.gpu);
  if (m.draft_method) parts.push(`nháp ${m.draft_method}${m.draft_model ? ` (${m.draft_model})` : ""}`);
  else if (m.draft_method === null) parts.push("không dùng model nháp");
  if (m.adaptive !== undefined) parts.push(m.adaptive ? "Adaptive bật" : "Adaptive tắt");
  return parts.length ? parts.join(", ") : "Chưa rõ GPU và cách giải mã";
}

function Onboarding({ onAdd }: { onAdd: () => void }) {
  const steps = [
    { title: "Chạy flexsds serve", body: "Trên Modal, Colab hoặc GPU của máy. CLI dựng vLLM và chọn cách giải mã hợp với GPU." },
    { title: "Copy dòng Endpoint URL", body: "Ví dụ https://…--flexsds-serve.modal.run/v1. Trên máy cá nhân, mở đường hầm cloudflared trước." },
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
      <pre className="mt-4 overflow-x-auto rounded-lg bg-rail px-4 py-3 font-mono text-[13px] leading-relaxed text-rail-ink">
        <span className="text-[#7d8a97]">$ </span>flexsds serve --model Qwen/Qwen2.5-7B-Instruct{"\n"}
        <span className="text-[#7d8a97]">  Endpoint URL: </span><span className="text-white">https://…--flexsds-serve.modal.run/v1</span>
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

export default function EndpointsPage({ api, endpoints, onChange, onAdd }: {
  api: Api;
  endpoints: Endpoint[];
  onChange: () => Promise<unknown>;
  onAdd: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const colors = endpointColors(endpoints);
  const models = new Set(endpoints.map((e) => e.model_id));

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
        actions={endpoints.length > 0 && <Button icon={Plus} variant="ghost" onClick={onAdd}>Thêm endpoint</Button>}
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
                  <tr key={e.endpoint_id} className="border-b border-line last:border-0 align-top">
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
                      {e.status === "ok" ? <Pill tone="good">Đang chạy</Pill> : <Pill tone="bad">Đã dừng</Pill>}
                      {e.status === "down" && e.status_message && <div className="mt-1 max-w-[220px] text-[12px] text-ink-3">{e.status_message}</div>}
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
                              disabled={busy !== null}
                              onClick={() => act(e.endpoint_id, () => api.checkEndpoint(e.endpoint_id))}
                            >
                              {busy === e.endpoint_id ? "Đang kiểm tra…" : "Kiểm tra lại"}
                            </Button>
                            <Button variant="danger" icon={Trash2} aria-label={`Xoá ${e.name}`} onClick={() => setConfirmDelete(e.endpoint_id)} />
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
