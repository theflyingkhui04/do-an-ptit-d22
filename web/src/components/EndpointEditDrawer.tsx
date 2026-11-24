"use client";

import { FileJson, KeyRound, Trash2 } from "lucide-react";
import { useEffect, useId, useRef, useState, type FormEvent } from "react";

import { Button, Drawer, Notice, baseModel, cx } from "@/components/ui";
import type { Api, Endpoint, EndpointChange, EndpointNotes } from "@/lib/api";
import {
  DECODING_LABELS_BY_LANG,
  GPU_SUGGESTIONS,
  decodingLabel,
  decodingOf,
  notesFor,
  notesFromServingConfig,
  platformLabel,
  type Decoding,
} from "@/lib/endpoints";
import { localeTag } from "@/lib/format";
import { useI18n } from "@/lib/i18n";

type KeyMode = "keep" | "set" | "clear";

const field = "w-full min-w-0 rounded-lg border border-line bg-surface px-3 py-2";
const label = "text-[13px] font-semibold text-ink-2";

function Fact({ term, children, tone }: { term: string; children: React.ReactNode; tone?: "bad" }) {
  return (
    <>
      <dt className="text-ink-3">{term}</dt>
      <dd className={cx("m-0 [overflow-wrap:anywhere]", tone === "bad" && "text-bad")}>{children}</dd>
    </>
  );
}

export default function EndpointEditDrawer({ api, endpoint: e, paused, onClose, onSaved, onDelete }: {
  api: Api;
  endpoint: Endpoint;
  paused: boolean;
  onClose: () => void;
  onSaved: () => Promise<unknown>;
  onDelete: () => void;
}) {
  const { lang, t } = useI18n();
  const id = useId();
  const meta = e.meta ?? {};
  const [name, setName] = useState(e.name);
  const first = useRef<HTMLInputElement>(null);
  useEffect(() => first.current?.focus(), []);
  const [url, setUrl] = useState(e.base_url);
  const [keyMode, setKeyMode] = useState<KeyMode>("keep");
  const [newKey, setNewKey] = useState("");
  const [gpu, setGpu] = useState(meta.gpu ?? "");
  const [decoding, setDecoding] = useState<Decoding>(decodingOf(meta));
  const [draft, setDraft] = useState(meta.draft_model ?? "");
  const [target, setTarget] = useState(meta.target_model ?? "");
  const [loaded, setLoaded] = useState<{ tone: "info" | "bad"; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const statusLabel: Record<Endpoint["status"], string> = {
    ok: t.statusOnline,
    starting: t.statusStarting,
    down: t.statusOffline,
  };

  function notes(): EndpointNotes {
    const want: EndpointNotes = { gpu: gpu.trim() || null };
    if (decoding !== decodingOf(meta)) {
      Object.assign(want, notesFor(decoding, draft));
    }
    if (draft.trim() !== (meta.draft_model ?? "")) {
      want.draft_model = draft.trim() || null;
    }
    if (target.trim() !== (meta.target_model ?? "")) want.target_model = target.trim() || null;
    const stored = meta as Record<string, unknown>;
    return Object.fromEntries(Object.entries(want).filter(([k, v]) => v !== (stored[k] ?? null)));
  }

  function changes(): EndpointChange | string {
    const change: EndpointChange = {};
    if (!name.trim()) return t.aliasEmptyError;
    if (name.trim() !== e.name) change.name = name.trim();
    if (!url.trim()) return t.urlEmptyError;
    if (url.trim() !== e.base_url) change.base_url = url.trim();
    if (keyMode === "set") {
      if (!newKey.trim()) return t.newKeyEmptyError;
      change.api_key = newKey.trim();
    }
    if (keyMode === "clear") change.api_key = null;
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
      setLoaded({ tone: "bad", text: t.invalidConfigFile });
      return;
    }
    if (found.gpu) setGpu(found.gpu);
    setDecoding(found.decoding);
    setDraft(found.draft_model ?? "");
    if (found.target_model) setTarget(found.target_model);
    const what = [found.gpu, found.decoding ? decodingLabel(found.decoding as Exclude<Decoding, "">, lang) : null].filter(Boolean).join(" · ");
    setLoaded({ tone: "info", text: t.loadedConfigFile(what) });
  }

  const keyText = { keep: e.has_key ? t.keySavedHidden : t.keyNone, set: t.keyWillSet, clear: t.keyWillClear }[keyMode];
  const urlChanged = url.trim() !== e.base_url;
  const small = "px-2.5 py-1.5 text-[13px]";

  return (
    <Drawer
      title={t.editEndpointTitle}
      onClose={onClose}
      paused={paused || saving}
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>{t.cancel}</Button>
          <Button type="submit" form={`${id}-form`} loading={saving} disabled={saving}>
            {saving ? (urlChanged ? t.verifyingUrl : t.saving) : t.saveChanges}
          </Button>
        </>
      )}
    >
      <form id={`${id}-form`} onSubmit={save} noValidate className="grid content-start gap-6">
        <fieldset className="grid gap-2">
          <legend className="mb-2 text-[13.5px] font-bold">{t.sectionInfo}</legend>
          <dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[13.5px]">
            <Fact term={t.factPlatform}>{platformLabel(e.platform, lang)}</Fact>
            <Fact term={t.factModel}>{baseModel(e)}</Fact>
            <Fact term={t.factContext}>{e.max_model_len ? t.contextTokens(e.max_model_len.toLocaleString(localeTag(lang))) : t.unknown}</Fact>
            <Fact term={t.factStatus}>{statusLabel[e.status]}</Fact>
            {e.status !== "ok" && e.status_message && (
              <Fact term={t.factDetail} tone={e.status === "down" ? "bad" : undefined}>{e.status_message}</Fact>
            )}
          </dl>
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-2 text-[13.5px] font-bold">{t.sectionIdentity}</legend>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-name`} className={label}>{t.aliasLabel}</label>
            <input id={`${id}-name`} ref={first} value={name} maxLength={80} placeholder={t.aliasPlaceholder} onChange={(ev) => setName(ev.target.value)} className={field} />
          </div>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-url`} className={label}>{t.apiUrlLabel}</label>
            <input id={`${id}-url`} value={url} spellCheck={false} onChange={(ev) => setUrl(ev.target.value)} className={cx(field, "font-mono text-[13px]")} />
          </div>
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-2 text-[13.5px] font-bold">{t.sectionAuth}</legend>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 font-semibold"><KeyRound aria-hidden className="size-4 text-ink-3" />{keyText}</span>
            {keyMode === "keep" ? (
              <>
                <Button variant="ghost" className={small} onClick={() => setKeyMode("set")}>{e.has_key ? t.changeKey : t.setKey}</Button>
                {e.has_key && <Button variant="ghost" className={small} onClick={() => setKeyMode("clear")}>{t.removeKey}</Button>}
              </>
            ) : (
              <Button variant="ghost" className={small} onClick={() => { setKeyMode("keep"); setNewKey(""); }}>
                {e.has_key ? t.keepKey : t.noKeyAction}
              </Button>
            )}
          </div>
          {keyMode === "set" && (
            <div className="grid gap-1.5">
              <label htmlFor={`${id}-key`} className={label}>{t.newKeyLabel}</label>
              <input id={`${id}-key`} type="password" autoComplete="off" value={newKey} onChange={(ev) => setNewKey(ev.target.value)} className={field} />
            </div>
          )}
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="mb-1 text-[13.5px] font-bold">{t.sectionConfig}</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="grid min-w-0 gap-1.5">
              <label htmlFor={`${id}-gpu`} className={label}>{t.gpuLabel}</label>
              <input id={`${id}-gpu`} list={`${id}-gpus`} value={gpu} placeholder={t.unknown} onChange={(ev) => setGpu(ev.target.value)} className={field} />
              <datalist id={`${id}-gpus`}>{GPU_SUGGESTIONS.map((g) => <option key={g} value={g} />)}</datalist>
            </div>
            <div className="grid min-w-0 gap-1.5">
              <label htmlFor={`${id}-decoding`} className={label}>{t.decodingLabel}</label>
              <select id={`${id}-decoding`} value={decoding} onChange={(ev) => setDecoding(ev.target.value as Decoding)} className={field}>
                <option value="">{t.unknown}</option>
                {Object.entries(DECODING_LABELS_BY_LANG[lang]).map(([value, text]) => <option key={value} value={value}>{text}</option>)}
              </select>
            </div>
          </div>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-target`} className={label}>{t.targetModelLabel}</label>
            <input id={`${id}-target`} value={target} spellCheck={false} placeholder={baseModel(e)}
              onChange={(ev) => setTarget(ev.target.value)} className={cx(field, "font-mono text-[13px]")} />
          </div>
          <div className="grid gap-1.5">
            <label htmlFor={`${id}-draft`} className={label}>{t.draftModelLabel}</label>
            <input id={`${id}-draft`} value={draft} spellCheck={false} placeholder="duckhoi/qwen2.5-7b-eagle3.1-speculator"
              onChange={(ev) => setDraft(ev.target.value)} className={cx(field, "font-mono text-[13px]")} />
          </div>
          <div className="grid justify-items-start gap-1.5">
            <label htmlFor={`${id}-file`} className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-[13px] font-semibold hover:bg-sunken has-[+input:focus-visible]:outline-2">
              <FileJson aria-hidden className="size-4" /> {t.importConfig}
            </label>
            <input id={`${id}-file`} type="file" accept="application/json,.json" className="sr-only" onChange={(ev) => void readConfig(ev.target.files?.[0])} />
          </div>
          {loaded && <Notice tone={loaded.tone}>{loaded.text}</Notice>}
        </fieldset>

        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-bad-tint px-3.5 py-3">
          <p className="min-w-[200px] flex-1 text-[13px] text-ink-2">{t.removeFromWeb}</p>
          <Button variant="danger" icon={Trash2} onClick={onDelete}>{t.deleteConfirm}</Button>
        </div>

        {error && <Notice tone="bad">{error}</Notice>}
      </form>
    </Drawer>
  );
}
