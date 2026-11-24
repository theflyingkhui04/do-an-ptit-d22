"use client";

import { Pencil, Plus, RefreshCw, Server, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import EndpointEditDrawer from "@/components/EndpointEditDrawer";
import { Button, ConfirmDialog, CopyButton, Dot, EmptyState, Notice, PageHead, Pill, baseModel, cx, endpointColors, secondsSince, useNow } from "@/components/ui";
import type { Api, Endpoint } from "@/lib/api";
import { agoText, isHardwareAdaptive, shortModelName, shortReason, shortUrl } from "@/lib/endpoints";
import { useI18n } from "@/lib/i18n";
import { durationText } from "@/lib/liveness";

export const WAKE_RETRY_MS = 20_000;
// A scaled-to-zero GPU endpoint (flexsds serve on Modal) was measured at 7.5 to 15 minutes from cold.
export const WAKE_LIMIT_MS = 15 * 60_000;

interface CheckOutcome {
  at: number;
  ms: number;
  status: Endpoint["status"] | "error";
  message: string | null;
  gaveUp?: boolean;
}

function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record };
  delete next[key];
  return next;
}

// Kept in step with the real CLI by tests/test_web_cli_sample.py.
export const SERVE_SAMPLE =
  "flexsds serve --backend modal --gpu A10G --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft duckhoi/qwen2.5-7b-eagle3.1-speculator";
export const SERVE_SAMPLE_HEADER = "Endpoint LLM created:";
export const SERVE_SAMPLE_API = "https://…--flexsds-serving-serve.modal.run/v1";

function Onboarding({ onAdd }: { onAdd: () => void }) {
  const { t } = useI18n();
  const steps = [
    { title: t.step1Title, body: t.step1Body },
    { title: t.step2Title, body: t.step2Body },
    { title: t.step3Title, body: t.step3Body },
  ];
  return (
    <EmptyState
      icon={Server}
      title={t.emptyEndpointsTitle}
      action={<Button icon={Plus} onClick={onAdd}>{t.addFirstEndpoint}</Button>}
    >
      <p>{t.emptyEndpointsBody}</p>
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
  checkingFor: number | null;
  waking: boolean;
  outcome?: CheckOutcome;
  now: number;
}) {
  const { lang, t } = useI18n();
  if (checkingFor !== null && waking) {
    return (
      <Pill tone="warn">
        <span aria-hidden className="size-2 animate-pulse rounded-full bg-warn" />
        {t.wakingElapsed(durationText(checkingFor, lang))}
      </Pill>
    );
  }
  if (checkingFor !== null) {
    return (
      <Pill tone="idle">
        <span aria-hidden className="size-2 animate-pulse rounded-full bg-ink-3" />
        {t.checkingElapsed(checkingFor)}
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
      <span className="inline-flex" title={status !== "ok" && reason ? shortReason(reason, lang) : undefined}>
        {status === "ok" && <Pill tone="good">{t.statusOnline}</Pill>}
        {status === "starting" && !fresh?.gaveUp && <Pill tone="warn">{t.statusStarting}</Pill>}
        {(status === "down" || fresh?.gaveUp) && <Pill tone="bad">{t.statusOffline}</Pill>}
      </span>
      {fresh?.gaveUp && (
        <span className="text-[12.5px] text-bad">{t.wakingTimeout(WAKE_LIMIT_MS / 60_000)}</span>
      )}
      <span className="num text-[12px] text-ink-3">{agoText(checkedAt, now, lang)}</span>
      {failedCall && <span className="animate-fade-in text-[12.5px] text-bad">{failedCall}</span>}
    </>
  );
}

export default function EndpointsPage({ api, endpoints, onChange, onAdd, onCheckingChange }: {
  api: Api;
  endpoints: Endpoint[];
  onChange: () => Promise<unknown>;
  onAdd: () => void;
  onCheckingChange?: (count: number) => void;
}) {
  const { t } = useI18n();
  const [checking, setChecking] = useState<Record<string, number>>({});
  const [waking, setWaking] = useState<Record<string, true>>({});
  const [outcome, setOutcome] = useState<Record<string, CheckOutcome>>({});
  const open = useRef(true);
  useEffect(() => {
    open.current = true;
    return () => {
      open.current = false;
    };
  }, []);
  const [flash, setFlash] = useState<Record<string, "good" | "bad" | "info">>({});
  const [editing, setEditing] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const running = Object.keys(checking).length;
  const now = useNow(true, running > 0 ? 1000 : 30_000);

  useEffect(() => {
    onCheckingChange?.(running);
  }, [running, onCheckingChange]);
  useEffect(() => () => onCheckingChange?.(0), [onCheckingChange]);

  const colors = endpointColors(endpoints);
  const models = new Set(endpoints.map(baseModel));
  const editingEndpoint = endpoints.find((e) => e.endpoint_id === editing);
  const confirming = endpoints.find((e) => e.endpoint_id === confirm);
  const closeEditor = useCallback(() => setEditing(null), []);

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
      found = {
        at,
        ms: at - started,
        status: checked.status,
        message: checked.status === "ok" ? null : checked.status_message,
        gaveUp: checked.status === "starting",
      };
      break;
    }
    if (!open.current) return;
    setOutcome((current) => ({ ...current, [id]: found }));
    setFlash((current) => ({ ...current, [id]: found.status === "ok" ? "good" : "bad" }));
    setWaking((current) => without(current, id));
    setChecking((current) => without(current, id));
  }

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
        title={t.pageEndpoints}
        actions={endpoints.length > 0 && (
          <Button
            variant="ghost"
            icon={RefreshCw}
            loading={running > 0}
            disabled={running > 0}
            onClick={() => void check(endpoints.map((e) => e.endpoint_id))}
          >
            {running > 0 ? t.checkingCount(running) : t.checkAll}
          </Button>
        )}
      />

      {endpoints.length === 0 ? (
        <Onboarding onAdd={onAdd} />
      ) : (
        <section aria-label={t.endpointsList} className="overflow-hidden rounded-xl border border-line bg-surface">
          {endpoints.map((e) => {
            const busy = checking[e.endpoint_id] !== undefined;
            const targetName = baseModel(e);
            const draftName = e.meta?.draft_model ?? null;
            const adaptiveOn = isHardwareAdaptive(e.meta);
            return (
              <article
                key={e.endpoint_id}
                aria-label={e.name}
                className={cx(
                  "grid gap-x-5 gap-y-3 border-line px-[18px] py-4 [&+&]:border-t",
                  "md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]",
                  "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)_minmax(140px,0.7fr)_auto]",
                  flash[e.endpoint_id] === "good" && "animate-row-good",
                  flash[e.endpoint_id] === "bad" && "animate-row-bad",
                  flash[e.endpoint_id] === "info" && "animate-row-info",
                )}
                onAnimationEnd={() => setFlash((current) => without(current, e.endpoint_id))}
              >
                <div className="grid min-w-0 content-start gap-1">
                  <div className="flex min-w-0 items-center gap-2">
                    <Dot color={colors[e.endpoint_id]} />
                    <button
                      type="button"
                      onClick={() => setEditing(e.endpoint_id)}
                      className="truncate text-left text-[15px] font-bold hover:underline"
                    >
                      {e.name}
                    </button>
                  </div>
                  <div className="flex min-w-0 items-center gap-1 font-mono text-[12.5px] text-ink-3">
                    <span className="truncate" title={e.base_url}>{shortUrl(e.base_url)}</span>
                    <CopyButton text={e.base_url} label={t.copyAddress(e.name)} />
                  </div>
                </div>

                <dl className="m-0 grid min-w-0 grid-cols-[max-content_minmax(0,1fr)] content-start gap-x-3 gap-y-1 text-[13px]">
                  <dt className="text-ink-3">{t.epFieldChip}</dt>
                  <dd className="m-0 truncate font-medium text-ink">{e.meta?.gpu || "—"}</dd>
                  <dt className="text-ink-3">{t.epFieldDraft}</dt>
                  <dd className="m-0 truncate text-ink-2" title={draftName ?? undefined}>{draftName ? shortModelName(draftName) : "—"}</dd>
                  <dt className="text-ink-3">{t.epFieldTarget}</dt>
                  <dd className="m-0 truncate text-ink-2" title={targetName}>{shortModelName(targetName)}</dd>
                  <dt className="text-ink-3">{t.epFieldAdaptive}</dt>
                  <dd className="m-0">
                    <span className={cx(
                      "inline-flex items-center rounded-md px-1.5 py-px text-[11.5px] font-bold",
                      adaptiveOn ? "bg-good-tint text-good" : "bg-sunken text-ink-3",
                    )}>
                      {adaptiveOn ? t.onLabel : t.offLabel}
                    </span>
                  </dd>
                </dl>

                <div className="grid content-start justify-items-start gap-1" aria-live="polite">
                  <StatusCell
                    endpoint={e}
                    checkingFor={busy ? secondsSince(checking[e.endpoint_id], now) : null}
                    waking={waking[e.endpoint_id] === true}
                    outcome={outcome[e.endpoint_id]}
                    now={now}
                  />
                </div>

                <div className="flex flex-wrap items-start gap-1.5 lg:justify-end">
                  <Button variant="ghost" icon={RefreshCw} loading={busy} disabled={busy} aria-label={t.checkEndpointAria(e.name)}
                    className="px-2.5 py-1.5 text-[13px]" onClick={() => void check([e.endpoint_id])}>
                    {t.actionCheck}
                  </Button>
                  <Button variant="ghost" icon={Pencil} aria-label={t.editEndpointAria(e.name)} className="px-2.5 py-1.5 text-[13px]"
                    onClick={() => setEditing(e.endpoint_id)}>
                    {t.actionEdit}
                  </Button>
                  <Button variant="danger" icon={Trash2} aria-label={t.deleteEndpointAria(e.name)} disabled={busy} className="px-2.5 py-1.5 text-[13px]"
                    onClick={() => setConfirm(e.endpoint_id)}>
                    {t.actionDelete}
                  </Button>
                </div>
              </article>
            );
          })}
        </section>
      )}

      {error && <Notice tone="bad">{error}</Notice>}
      {models.size > 1 && (
        <Notice tone="warn">{t.diffModelsWarn(models.size)}</Notice>
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
          title={t.deleteTitle(confirming.name)}
          confirmLabel={t.deleteConfirm}
          confirmIcon={Trash2}
          busy={deleting}
          onConfirm={() => void remove(confirming.endpoint_id)}
          onCancel={() => setConfirm(null)}
        >
          <p className="text-ink-2">{t.deleteBody}</p>
        </ConfirmDialog>
      )}
    </>
  );
}
