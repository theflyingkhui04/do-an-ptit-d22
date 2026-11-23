"use client";

import { KeyRound } from "lucide-react";
import { useState } from "react";

import { BrandMark } from "@/components/ui";
import { ApiError } from "@/lib/api";

interface Props {
  // Checks the key against the backend (an authenticated request); resolves when it is accepted.
  verify: (key: string) => Promise<void>;
  onAccepted: (key: string) => void;
}

export default function KeyGate({ verify, onAccepted }: Props) {
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await verify(key.trim());
      onAccepted(key.trim());
    } catch (exc) {
      setError(exc instanceof ApiError && exc.status === 401 ? "Sai mật khẩu demo" : (exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-screen place-items-center bg-rail px-4 text-[14.5px]">
      <form onSubmit={submit} className="grid w-full max-w-sm gap-4 rounded-2xl bg-surface p-7">
        <div className="flex items-center gap-2.5">
          <BrandMark size={36} className="ring-1 ring-line" />
          <div className="leading-tight">
            <h1 className="text-lg font-bold">FlexSDS Benchmark</h1>
            <p className="text-[13px] text-ink-3">Đồ án tốt nghiệp D22, PTIT</p>
          </div>
        </div>
        <p className="text-ink-2">Nhập mật khẩu demo để dùng hệ thống.</p>
        <div className="grid gap-1.5">
          <label htmlFor="demo-key" className="text-[13.5px] font-semibold">Mật khẩu demo</label>
          <input
            id="demo-key"
            type="password"
            autoComplete="off"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="rounded-lg border border-line bg-surface px-3 py-2"
          />
        </div>
        {error && <p role="alert" className="text-[13.5px] text-bad">{error}</p>}
        <button
          type="submit"
          disabled={busy || !key.trim()}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-ink px-3.5 py-2.5 font-semibold text-surface disabled:opacity-50"
        >
          <KeyRound aria-hidden className="size-4" />
          {busy ? "Đang kiểm tra…" : "Vào console"}
        </button>
      </form>
    </main>
  );
}
