"use client";

import { useState } from "react";

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
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h1 className="text-lg font-semibold text-slate-900">Demo hoá đơn · Speculative Decoding</h1>
        <p className="mt-1 text-sm text-slate-600">Nhập mật khẩu demo để dùng hệ thống.</p>
        <label htmlFor="demo-key" className="mt-4 block text-sm font-medium text-slate-700">
          Mật khẩu demo
        </label>
        <input
          id="demo-key"
          type="password"
          autoComplete="off"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
        />
        {error && (
          <p role="alert" className="mt-2 text-sm text-red-600">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || !key.trim()}
          className="mt-4 w-full rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {busy ? "Đang kiểm tra…" : "Vào demo"}
        </button>
      </form>
    </main>
  );
}
