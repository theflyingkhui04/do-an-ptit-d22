"use client";

import { Activity, ChevronRight, Gauge, LogOut, Plus, Server } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import AddEndpointDrawer from "@/components/AddEndpointDrawer";
import EndpointsPage from "@/components/EndpointsPage";
import KeyGate from "@/components/KeyGate";
import Node1Page from "@/components/Node1Page";
import Node2Page from "@/components/Node2Page";
import Node3Page from "@/components/Node3Page";
import ResultPage from "@/components/ResultPage";
import { BrandMark, Button, Spinner, cx } from "@/components/ui";
import { createApi, readKey, saveKey, withUnauthorizedHandler, type Api, type Endpoint } from "@/lib/api";

export type PageId = "node1" | "node2" | "node3" | "result" | "endpoints";

const PAGES: { id: PageId; label: string; step?: number; group: string }[] = [
  { id: "node1", label: "Trích xuất hoá đơn", step: 1, group: "Pipeline hoá đơn" },
  { id: "node2", label: "Hỏi bằng SQL", step: 2, group: "Pipeline hoá đơn" },
  { id: "node3", label: "Báo cáo và dashboard", step: 3, group: "Pipeline hoá đơn" },
  { id: "result", label: "Đo và so sánh", group: "Đánh giá" },
  { id: "endpoints", label: "Endpoint", group: "Hệ thống" },
];
const CRUMB: Record<PageId, string> = {
  node1: "Node 1",
  node2: "Node 2",
  node3: "Node 3",
  result: "Đánh giá",
  endpoints: "Hệ thống",
};

interface Props {
  // Tests inject a fake API; the page builds the real one from NEXT_PUBLIC_API_URL.
  makeApi?: (getKey: () => string | null) => Api;
  pollMs?: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function pageFromHash(): PageId | null {
  const id = window.location.hash.replace(/^#/, "");
  return PAGES.some((p) => p.id === id) ? (id as PageId) : null;
}

export default function Console({ makeApi, pollMs = 1500 }: Props) {
  const [key, setKey] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const build = useMemo(() => makeApi ?? ((getKey: () => string | null) => createApi(API_URL, getKey)), [makeApi]);
  const api = useMemo(
    () =>
      withUnauthorizedHandler(build(() => key), () => {
        saveKey(null);
        setKey(null);
      }),
    [build, key],
  );

  useEffect(() => {
    // The key lives in sessionStorage, which only exists in the browser.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setKey(readKey());
    setReady(true);
  }, []);

  if (!ready) return null;
  if (!key) {
    return (
      <KeyGate
        verify={async (candidate) => {
          await build(() => candidate).endpoints();
        }}
        onAccepted={(accepted) => {
          saveKey(accepted);
          setKey(accepted);
        }}
      />
    );
  }
  return (
    <Shell
      api={api}
      pollMs={pollMs}
      onSignOut={() => {
        saveKey(null);
        setKey(null);
      }}
    />
  );
}

function Shell({ api, pollMs, onSignOut }: { api: Api; pollMs: number; onSignOut: () => void }) {
  const [endpoints, setEndpoints] = useState<Endpoint[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [page, setPage] = useState<PageId | null>(null);
  const [adding, setAdding] = useState(false);
  const [checking, setChecking] = useState(0);

  const reload = useCallback(async () => {
    try {
      const { endpoints: list } = await api.endpoints();
      setEndpoints(list);
      setLoadError(null);
      return list;
    } catch (exc) {
      setLoadError((exc as Error).message);
      return null;
    }
  }, [api]);

  useEffect(() => {
    let alive = true;
    api.endpoints().then(
      ({ endpoints: list }) => {
        if (!alive) return;
        setEndpoints(list);
        setPage(pageFromHash() ?? (list.length > 0 ? "node1" : "endpoints"));
      },
      (exc) => alive && setLoadError((exc as Error).message),
    );
    const onHash = () => {
      const next = pageFromHash();
      if (next) setPage(next);
    };
    window.addEventListener("hashchange", onHash);
    return () => {
      alive = false;
      window.removeEventListener("hashchange", onHash);
    };
  }, [api]);

  const go = useCallback((next: PageId) => {
    setPage(next);
    if (window.location.hash !== `#${next}`) window.history.replaceState(null, "", `#${next}`);
    window.scrollTo?.({ top: 0 });
  }, []);

  const list = endpoints ?? [];
  const running = list.filter((e) => e.status === "ok").length;
  const current = PAGES.find((p) => p.id === page);
  const groups = [...new Set(PAGES.map((p) => p.group))];

  return (
    // Side bar (design D15): full from 1100px, a 68px icon rail from 768px, a scrolling top row on phones.
    <div className="min-h-screen text-[14.5px] leading-[1.55] md:grid md:grid-cols-[68px_minmax(0,1fr)] md:bg-[linear-gradient(to_right,var(--rail)_68px,transparent_68px)] wide:grid-cols-[248px_minmax(0,1fr)] wide:bg-[linear-gradient(to_right,var(--rail)_248px,transparent_248px)]">
      <nav aria-label="Điều hướng" className="flex items-center gap-3 overflow-x-auto bg-rail px-3 py-2.5 text-rail-ink md:sticky md:top-0 md:h-screen md:flex-col md:items-stretch md:gap-5 md:overflow-x-visible md:overflow-y-auto md:px-2.5 md:py-[18px] wide:px-3">
        <div className="flex shrink-0 items-center gap-2.5 md:justify-center wide:justify-start wide:px-2">
          <BrandMark size={32} />
          <div className="hidden leading-tight wide:block">
            <b className="block text-base text-white">FlexSDS</b>
            <span className="text-xs">Console đồ án D22, PTIT</span>
          </div>
        </div>
        <div className="flex gap-1 md:grid md:gap-4">
          {groups.map((group) => (
            <div key={group} className="flex gap-1 md:grid md:gap-0.5">
              <div className="hidden px-2.5 pb-1.5 text-xs text-[#7d8a97] wide:block">{group}</div>
              {PAGES.filter((p) => p.group === group).map((p) => (
                <a
                  key={p.id}
                  href={`#${p.id}`}
                  aria-current={page === p.id ? "page" : undefined}
                  title={p.label}
                  onClick={(event) => {
                    event.preventDefault();
                    go(p.id);
                  }}
                  className={cx(
                    "group flex shrink-0 items-center gap-2.5 whitespace-nowrap rounded-lg px-2.5 py-2 hover:bg-rail-active hover:text-white",
                    "md:h-10 md:justify-center md:px-0 wide:h-auto wide:justify-start wide:px-2.5",
                    page === p.id && "bg-rail-active font-semibold text-white",
                  )}
                >
                  {p.step ? (
                    <span
                      className={cx(
                        "grid size-5 shrink-0 place-items-center rounded-md border text-xs font-bold",
                        page === p.id ? "border-white bg-white text-rail" : "border-[#3a4a5a]",
                      )}
                    >
                      {p.step}
                    </span>
                  ) : p.id === "result" ? (
                    <Gauge aria-hidden className="size-5 shrink-0" />
                  ) : (
                    <Server aria-hidden className="size-5 shrink-0" />
                  )}
                  <span className="md:sr-only wide:not-sr-only">{p.label}</span>
                  {p.id === "endpoints" && endpoints && (
                    <span className="ml-auto hidden rounded-md border border-[#2c3a48] px-1.5 text-[11.5px] text-[#8796a5] wide:inline">{list.length}</span>
                  )}
                </a>
              ))}
            </div>
          ))}
        </div>
        <div className="mt-auto hidden gap-2 rounded-xl bg-rail-active p-3 text-[13px] wide:grid">
          <p>Web không bật hay tắt GPU. Endpoint do lệnh <code className="font-mono text-white">flexsds serve</code> tạo ra.</p>
          <button type="button" onClick={onSignOut} className="flex items-center gap-2 text-[#8796a5] hover:text-white">
            <LogOut aria-hidden className="size-4" /> Thoát
          </button>
        </div>
        <button
          type="button"
          onClick={onSignOut}
          title="Thoát"
          aria-label="Thoát"
          className="mt-auto hidden size-10 place-items-center self-center rounded-lg text-[#8796a5] hover:bg-rail-active hover:text-white md:grid wide:hidden"
        >
          <LogOut aria-hidden className="size-4" />
        </button>
      </nav>

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b border-line bg-surface px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2 text-[13.5px] text-ink-3">
            <span>{current ? CRUMB[current.id] : ""}</span>
            <ChevronRight aria-hidden className="size-4" />
            <b className="font-semibold text-ink">{current?.label ?? ""}</b>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2.5">
            {endpoints && (
              <button
                type="button"
                onClick={() => go("endpoints")}
                className="inline-flex items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-[13.5px] hover:bg-sunken"
                data-testid="endpoint-summary"
              >
                {checking > 0
                  ? <Spinner className="text-ink-3" />
                  : <Activity aria-hidden className={cx("size-4", running > 0 ? "text-good" : "text-ink-3")} />}
                {list.length === 0 ? "Chưa có endpoint" : `${running}/${list.length} endpoint đang chạy`}
              </button>
            )}
            <Button icon={Plus} onClick={() => setAdding(true)}>Thêm endpoint</Button>
          </div>
        </header>

        <main className="grid w-full max-w-[1360px] gap-5 px-4 pb-10 pt-6 sm:px-6">
          {loadError && !endpoints && (
            <div role="alert" className="rounded-xl bg-bad-tint px-4 py-3 text-bad">
              Không tải được danh sách endpoint: {loadError}
            </div>
          )}
          {endpoints && page === "endpoints" && (
            <EndpointsPage api={api} endpoints={list} onChange={reload} onAdd={() => setAdding(true)} onCheckingChange={setChecking} />
          )}
          {endpoints && page === "node1" && <Node1Page api={api} endpoints={list} pollMs={pollMs} go={go} />}
          {endpoints && page === "node2" && <Node2Page api={api} endpoints={list} go={go} />}
          {endpoints && page === "node3" && <Node3Page api={api} endpoints={list} go={go} />}
          {endpoints && page === "result" && <ResultPage api={api} endpoints={list} pollMs={pollMs} go={go} />}
        </main>
      </div>

      {adding && (
        <AddEndpointDrawer
          api={api}
          onClose={() => setAdding(false)}
          onSaved={async () => {
            await reload();
          }}
        />
      )}
    </div>
  );
}
