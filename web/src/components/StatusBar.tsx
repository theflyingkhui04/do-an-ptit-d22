"use client";

import { useCallback, useEffect, useState } from "react";

import type { Api, ServiceStateName, Status } from "@/lib/api";

const STATE_LABEL: Record<ServiceStateName, string> = {
  ready: "sẵn sàng",
  starting: "đang khởi động",
  cold: "đang ngủ",
  error: "lỗi",
};
const STATE_STYLE: Record<ServiceStateName, string> = {
  ready: "bg-emerald-100 text-emerald-800",
  starting: "bg-amber-100 text-amber-800",
  cold: "bg-slate-100 text-slate-600",
  error: "bg-red-100 text-red-700",
};
const SERVICE_LABEL: Record<string, string> = { database: "DB", ocr: "OCR" };

interface Props {
  api: Api;
  refreshMs?: number;
}

export default function StatusBar({ api, refreshMs = 10000 }: Props) {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warming, setWarming] = useState(false);

  const load = useCallback(async () => {
    try {
      setStatus(await api.status());
      setError(null);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }, [api]);

  useEffect(() => {
    // First load and periodic refresh of the service states.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
    if (refreshMs <= 0) return;
    const timer = setInterval(() => void load(), refreshMs);
    return () => clearInterval(timer);
  }, [load, refreshMs]);

  async function warm() {
    setWarming(true);
    try {
      const body = await api.warmup();
      setStatus((current) => (current ? { ...current, services: body.services } : current));
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setWarming(false);
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" aria-label="Trạng thái dịch vụ">
      <span className={`rounded px-2 py-1 ${error ? STATE_STYLE.error : STATE_STYLE.ready}`}>
        Backend: {error ? "lỗi" : "sẵn sàng"}
      </span>
      {status &&
        Object.entries(status.services).map(([name, service]) => (
          <span
            key={name}
            title={service.message ?? undefined}
            data-testid={`service-${name}`}
            className={`rounded px-2 py-1 ${STATE_STYLE[service.state]}`}
          >
            {SERVICE_LABEL[name] ?? name}: {STATE_LABEL[service.state]}
          </span>
        ))}
      <button
        type="button"
        onClick={warm}
        disabled={warming}
        className="rounded bg-slate-800 px-2 py-1 font-medium text-white disabled:opacity-50"
      >
        {warming ? "Đang làm nóng…" : "Làm nóng hệ thống"}
      </button>
    </div>
  );
}
