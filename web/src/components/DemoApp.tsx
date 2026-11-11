"use client";

import { useEffect, useMemo, useState } from "react";

import InvoiceList from "@/components/InvoiceList";
import KeyGate from "@/components/KeyGate";
import QuestionPanel from "@/components/QuestionPanel";
import StatusBar from "@/components/StatusBar";
import UploadPanel from "@/components/UploadPanel";
import { createApi, readKey, saveKey, withUnauthorizedHandler, type Api } from "@/lib/api";

type Tab = "extract" | "ask" | "list";
const TABS: [Tab, string][] = [
  ["extract", "Trích xuất hoá đơn"],
  ["ask", "Hỏi dữ liệu"],
  ["list", "Danh sách hoá đơn"],
];

interface Props {
  // Tests inject a fake API; the page builds the real one from NEXT_PUBLIC_API_URL.
  makeApi?: (getKey: () => string | null) => Api;
  pollMs?: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function DemoApp({ makeApi, pollMs }: Props) {
  const [key, setKey] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [tab, setTab] = useState<Tab>("extract");
  const [stored, setStored] = useState(0);
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
          await build(() => candidate).invoices();
        }}
        onAccepted={(accepted) => {
          saveKey(accepted);
          setKey(accepted);
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <h1 className="text-base font-semibold text-slate-900">PTIT · Hệ thống đa tác vụ trên hoá đơn</h1>
            <p className="text-xs text-slate-500">
              OCR → Node 1 trích xuất → DB → hỏi dữ liệu · speculative decoding (vanilla / static / adaptive / eagle)
            </p>
            <p data-testid="marks-note" className="text-xs text-slate-500">
              Văn bản đã bỏ dấu để tránh sai dấu (OCR thư viện); câu hỏi gõ có dấu hay không dấu đều được.
            </p>
          </div>
          <StatusBar api={api} />
        </div>
        <nav className="mx-auto flex max-w-7xl gap-1 px-4" role="tablist">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`rounded-t-md px-3 py-2 text-sm ${tab === id ? "bg-slate-50 font-medium text-blue-700" : "text-slate-600"}`}
            >
              {label}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              saveKey(null);
              setKey(null);
            }}
            className="ml-auto px-3 py-2 text-xs text-slate-500"
          >
            Thoát
          </button>
        </nav>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-4">
        {tab === "extract" && <UploadPanel api={api} pollMs={pollMs} onStored={() => setStored((n) => n + 1)} />}
        {tab === "ask" && <QuestionPanel api={api} />}
        {tab === "list" && <InvoiceList api={api} refreshKey={stored} />}
      </main>
    </div>
  );
}
