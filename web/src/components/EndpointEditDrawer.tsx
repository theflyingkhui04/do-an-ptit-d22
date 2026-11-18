"use client";

import { FileJson, KeyRound, Trash2 } from "lucide-react";
import { useId, useState, type FormEvent } from "react";

import { Button, Drawer, Notice, baseModel, cx } from "@/components/ui";
import type { Api, Endpoint, EndpointChange, EndpointNotes } from "@/lib/api";
import {
  DECODING_LABELS, GPU_SUGGESTIONS, PLATFORM_LABELS, decodingOf, notesFor, notesFromServingConfig, type Decoding,
} from "@/lib/endpoints";

type KeyMode = "keep" | "set" | "clear";

const field = "w-full min-w-0 rounded-lg border border-line bg-surface px-3 py-2";
const label = "text-[13px] font-semibold text-ink-2";
const hint = "text-[12.5px] text-ink-3";
const STATUS_WORDS: Record<Endpoint["status"], string> = {
  ok: "đang chạy",
  starting: "chưa sẵn sàng (GPU đang ngủ hoặc khởi động)",
  down: "đã dừng",
};

function Fact({ term, children, tone }: { term: string; children: React.ReactNode; tone?: "bad" }) {
  return (
    <>
      <dt className="text-ink-3">{term}</dt>
      <dd className={cx("m-0 [overflow-wrap:anywhere]", tone === "bad" && "text-bad")}>{children}</dd>
    </>
  );
}

// Edit one endpoint (design D14): only the fields that changed are sent; a new address is checked by the backend.
export default function EndpointEditDrawer({ api, endpoint: e, paused, onClose, onSaved, onDelete }: {
  api: Api;
  endpoint: Endpoint;
  paused: boolean; // the delete confirmation is open on top
  onClose: () => void;
  onSaved: () => Promise<unknown>;
  onDelete: () => void;
}) {
  const id = useId();
  const meta = e.meta ?? {};
  const [name, setName] = useState(e.name);
  const [url, setUrl] = useState(e.base_url);
  const [baseline, setBaseline] = useState(e.is_baseline);
  const [keyMode, setKeyMode] = useState<KeyMode>("keep");
  const [newKey, setNewKey] = useState("");
  const [gpu, setGpu] = useState(meta.gpu ?? "");
  const [decoding, setDecoding] = useState<Decoding>(decodingOf(meta));
  const [draft, setDraft] = useState(meta.draft_model ?? "");
  const [target, setTarget] = useState<string | undefined>(undefined); // only from serving_config.json
  const [loaded, setLoaded] = useState<{ tone: "info" | "bad"; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function notes(): EndpointNotes {
    const want: EndpointNotes = { gpu: gpu.trim() || null };
    if (decoding !== decodingOf(meta) || draft.trim() !== (meta.draft_model ?? "")) Object.assign(want, notesFor(decoding, draft));
    if (target !== undefined) want.target_model = target || null;
    const stored = meta as Record<string, unknown>;
    return Object.fromEntries(Object.entries(want).filter(([k, v]) => v !== (stored[k] ?? null)));
  }

  function changes(): EndpointChange | string {
    const change: EndpointChange = {};
    if (!name.trim()) return "Tên không được để trống.";
    if (name.trim() !== e.name) change.name = name.trim();
    if (!url.trim()) return "Địa chỉ API không được để trống.";
    if (url.trim() !== e.base_url) change.base_url = url.trim();
    if (keyMode === "set") {
      if (!newKey.trim()) return "Nhập khoá mới, hoặc bấm Giữ khoá cũ.";
      change.api_key = newKey.trim();
    }
    if (keyMode === "clear") change.api_key = null;
    if (baseline !== e.is_baseline) change.is_baseline = baseline;
    const noted = notes();
    if (Object.keys(noted).length) change.meta = noted;
    return change;
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    const change = changes();
    if (typeof change === "string") {
      setError(change);
      return;
    }
    if (!Object.keys(change).length) {
      onClose();
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await api.updateEndpoint(e.endpoint_id, change);
      await onSaved();
      onClose();
    } catch (exc) {
      setError((exc as Error).message);
      setSaving(false);
    }
  }

  async function readConfig(file: File | undefined) {
    if (!file) return;
    let found: ReturnType<typeof notesFromServingConfig> = null;
    try {
      found = notesFromServingConfig(JSON.parse(await file.text()));
    } catch {
      found = null;
    }
    if (!found) {
      setLoaded({ tone: "bad", text: "Tệp không phải serving_config.json hợp lệ (cần một đối tượng JSON)." });
      return;
    }
    if (found.gpu) setGpu(found.gpu);
    setDecoding(found.decoding);
    setDraft(found.draft_model ?? "");
    if (found.target_model) setTarget(found.target_model);
    const what = [found.gpu, found.decoding ? DECODING_LABELS[found.decoding as Exclude<Decoding, "">] : null].filter(Boolean).join(", ");
    setLoaded({ tone: "info", text: `Đã đọc serving_config.json${what ? `: ${what}` : ""}. Bấm Lưu thay đổi để áp dụng.` });
  }

  const keyText = { keep: e.has_key ? "Đã đặt khoá, không hiện lại" : "Không có khoá", set: "Sẽ đặt khoá mới khi lưu", clear: "Sẽ bỏ khoá khi lưu" }[keyMode];
  const urlChanged = url.trim() !== e.base_url;
  const small = "px-2.5 py-1.5 text-[13px]";

  return (
    <Drawer
      title="Sửa endpoint"
      aside={<span className="num text-[13px] text-ink-3">{e.endpoint_id}</span>}
      onClose={onClose}
      paused={paused || saving}
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>Huỷ</Button>
          <Button type="submit" form={`${id}-form`} loading={saving} disabled={saving}>
            {saving ? (urlChanged ? "Đang kiểm tra địa chỉ mới" : "Đang lưu") : "Lưu thay đổi"}
          </Button>
        </>
      )}
    >
      <form id={`${id}-form`} onSubmit={save} noValidate className="grid content-start gap-6">
        <fieldset className="grid gap-2">
          <legend className="mb-2 text-[13.5px] font-bold">Tình trạng</legend>
          <dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[13.5px]">
            <Fact term="Nền tảng">{PLATFORM_LABELS[e.platform] ?? e.platform}</Fact>
            <Fact term="Model endpoint báo">{e.model_id}</Fact>
            <Fact term="Model gốc">{baseModel(e)}</Fact>
            <Fact term="Ngữ cảnh tối đa">{e.max_model_len ? `${e.max_model_len.toLocaleString("vi-VN")} token` : "Chưa rõ"}</Fact>
            <Fact term="Thêm lúc">{new Date(e.created_at).toLocaleString("vi-VN")}</Fact>
            <Fact term="Kiểm tra lần cuối">
              {new Date(e.checked_at).toLocaleString("vi-VN")}, {STATUS_WORDS[e.status]}
            </Fact>
            {e.status !== "ok" && e.status_message && (
              <Fact term="Kết quả" tone={e.status === "down" ? "bad" : undefined}>{e.status_message}</Fact>
            )}
          </dl>
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-2 text-[13.5px] font-bold">Nhận diện</legend>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-name`} className={label}>Tên hiển thị</label>
            <input id={`${id}-name`} autoFocus value={name} maxLength={80} onChange={(ev) => setName(ev.target.value)} className={field} />
            <small className={hint}>Tên hiện ở mọi trang, ví dụ “adaptive A10G”.</small>
          </div>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-url`} className={label}>Địa chỉ API</label>
            <input id={`${id}-url`} value={url} spellCheck={false} onChange={(ev) => setUrl(ev.target.value)} className={cx(field, "font-mono text-[13px]")} />
            <small className={hint}>Đổi địa chỉ thì máy chủ kiểm tra địa chỉ mới trước khi lưu.</small>
          </div>
          <label className="inline-flex items-center gap-2.5">
            <input type="checkbox" checked={baseline} onChange={(ev) => setBaseline(ev.target.checked)} className="size-4" />
            Dùng làm endpoint mốc để so sánh tốc độ và kết quả
          </label>
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-2 text-[13.5px] font-bold">Khoá API</legend>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 font-semibold"><KeyRound aria-hidden className="size-4 text-ink-3" />{keyText}</span>
            {keyMode === "keep" ? (
              <>
                <Button variant="ghost" className={small} onClick={() => setKeyMode("set")}>{e.has_key ? "Đổi khoá" : "Đặt khoá"}</Button>
                {e.has_key && <Button variant="ghost" className={small} onClick={() => setKeyMode("clear")}>Bỏ khoá</Button>}
              </>
            ) : (
              <Button variant="ghost" className={small} onClick={() => { setKeyMode("keep"); setNewKey(""); }}>
                {e.has_key ? "Giữ khoá cũ" : "Không đặt khoá"}
              </Button>
            )}
          </div>
          {keyMode === "set" && (
            <div className="grid gap-1.5">
              <label htmlFor={`${id}-key`} className={label}>Khoá mới</label>
              <input id={`${id}-key`} type="password" autoComplete="off" value={newKey} onChange={(ev) => setNewKey(ev.target.value)} className={field} />
              <small className={hint}>Gửi dạng Bearer; chỉ lưu ở máy chủ.</small>
            </div>
          )}
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-1 text-[13.5px] font-bold">Ghi chú cấu hình (tuỳ chọn)</legend>
          <small className={hint}>Chỉ để hiện trên trang này cho dễ phân biệt; không ảnh hưởng cách web gọi endpoint.</small>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="grid min-w-0 gap-1.5">
              <label htmlFor={`${id}-gpu`} className={label}>GPU</label>
              <input id={`${id}-gpu`} list={`${id}-gpus`} value={gpu} placeholder="Chưa rõ" onChange={(ev) => setGpu(ev.target.value)} className={field} />
              <datalist id={`${id}-gpus`}>{GPU_SUGGESTIONS.map((g) => <option key={g} value={g} />)}</datalist>
            </div>
            <div className="grid min-w-0 gap-1.5">
              <label htmlFor={`${id}-decoding`} className={label}>Cách giải mã</label>
              <select id={`${id}-decoding`} value={decoding} onChange={(ev) => setDecoding(ev.target.value as Decoding)} className={field}>
                <option value="">Chưa rõ</option>
                {Object.entries(DECODING_LABELS).map(([value, text]) => <option key={value} value={value}>{text}</option>)}
              </select>
            </div>
          </div>
          {decoding && decoding !== "vanilla" && (
            <div className="grid gap-1.5">
              <label htmlFor={`${id}-draft`} className={label}>Model nháp</label>
              <input id={`${id}-draft`} value={draft} spellCheck={false} placeholder="duckhoi/qwen2.5-7b-eagle3.1-speculator"
                onChange={(ev) => setDraft(ev.target.value)} className={cx(field, "font-mono text-[13px]")} />
            </div>
          )}
          <div className="grid justify-items-start gap-1.5">
            <label htmlFor={`${id}-file`} className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-[13px] font-semibold hover:bg-sunken has-[+input:focus-visible]:outline-2">
              <FileJson aria-hidden className="size-4" /> Nạp từ serving_config.json
            </label>
            <input id={`${id}-file`} type="file" accept="application/json,.json" className="sr-only" onChange={(ev) => void readConfig(ev.target.files?.[0])} />
            <small className={hint}>Tệp do wizard của <code className="font-mono">flexsds serve</code> ghi ra; điền sẵn GPU, cách giải mã và model nháp.</small>
          </div>
          {loaded && <Notice tone={loaded.tone}>{loaded.text}</Notice>}
        </fieldset>

        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-bad-tint px-3.5 py-3">
          <p className="min-w-[200px] flex-1 text-[13px] text-ink-2">Gỡ endpoint này khỏi web. GPU trên Modal không bị tắt.</p>
          <Button variant="danger" icon={Trash2} onClick={onDelete}>Xoá endpoint</Button>
        </div>

        {error && <Notice tone="bad">{error}</Notice>}
      </form>
    </Drawer>
  );
}
