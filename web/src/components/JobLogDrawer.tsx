"use client";

import { Copy, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { clockTime, cx } from "@/components/ui";
import type { JobEvent } from "@/lib/api";
import type { PartState } from "@/lib/liveness";

// A part of the job whose log can be shown on its own: OCR, an endpoint, or the system lines.
export interface LogPart {
  id: string; // "all", "ocr", "system" or an endpoint id
  label: string;
  color?: string;
  state?: PartState;
}

const TONE: Record<string, string> = {
  good: "text-good", warn: "text-warn", bad: "text-bad", idle: "text-ink-3",
};
const LEVEL_ROW: Record<JobEvent["level"], string> = {
  info: "", ok: "", warn: "border-l-warn bg-warn-tint", error: "border-l-bad bg-bad-tint",
};
const LEVEL_TEXT: Record<JobEvent["level"], string> = {
  info: "text-ink", ok: "text-good", warn: "text-warn", error: "text-bad",
};

export const partOf = (src: string) => (src === "system" || src === "web" ? "system" : src);

// Log lines live in a drawer opened on demand, filtered to one part when opened from that part (Dương, 16/11).
export default function JobLogDrawer({ lines, parts, part, onPart, onClose, sourceLabel, sourceColor, elapsed }: {
  lines: JobEvent[];
  parts: LogPart[];
  part: string;
  onPart: (id: string) => void;
  onClose: () => void;
  sourceLabel: (src: string) => string;
  sourceColor: (src: string) => string | undefined;
  elapsed?: string;
}) {
  const [onlyIssues, setOnlyIssues] = useState(false);
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState<string | null>(null);
  const list = useRef<HTMLOListElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  const shown = (line: JobEvent) =>
    (part === "all" || partOf(line.src) === part) && (!onlyIssues || line.level === "warn" || line.level === "error");

  useEffect(() => {
    if (follow && list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [lines, follow, part, onlyIssues]);

  async function copy() {
    const text = lines.filter(shown)
      .map((line) => `${clockTime(new Date(line.t * 1000))} [${sourceLabel(line.src)}] ${line.msg}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied("Đã copy");
    } catch {
      setCopied("Không copy được, hãy chọn chữ rồi bấm Ctrl+C");
    }
  }

  return (
    <>
      <div className="fixed inset-0 z-30 bg-[rgb(10_16_22/0.45)]" onClick={onClose} aria-hidden />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="job-log-title"
        className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[600px] animate-fade-in flex-col border-l border-line bg-surface text-[14px] text-ink shadow-[-12px_0_32px_rgb(0_0_0/0.18)]"
      >
        <header className="flex items-center gap-2.5 border-b border-line px-[18px] py-3.5">
          <h2 id="job-log-title" className="text-base font-bold">Log lượt chạy</h2>
          {elapsed && <span className="num text-[13px] text-ink-3">{elapsed}</span>}
          <button ref={closeButton} type="button" onClick={onClose} aria-label="Đóng log" className="ml-auto rounded-md p-1 text-ink-3 hover:bg-sunken">
            <X aria-hidden className="size-5" />
          </button>
        </header>

        <div className="flex flex-wrap gap-1.5 px-[18px] pt-3" role="group" aria-label="Xem log của phần">
          {parts.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-pressed={part === p.id}
              onClick={() => onPart(p.id)}
              className={cx(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[13px] font-semibold",
                part === p.id ? "border-ink bg-ink text-surface" : "border-line bg-surface hover:bg-sunken",
              )}
            >
              {p.state && <span aria-hidden className={cx("size-2 rounded-full bg-current", part === p.id ? "" : TONE[p.state.tone], p.state.busy && p.state.tone !== "idle" && "animate-pulse")} />}
              {p.label}
              {p.state && <span className="font-medium opacity-80">{p.state.text}</span>}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-x-3.5 gap-y-2 border-b border-line px-[18px] py-2.5 text-[13px] text-ink-2">
          <label className="inline-flex items-center gap-1.5">
            <input type="checkbox" checked={onlyIssues} onChange={(e) => setOnlyIssues(e.target.checked)} /> Chỉ cảnh báo và lỗi
          </label>
          <label className="inline-flex items-center gap-1.5">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Tự cuộn
          </label>
          <button type="button" onClick={copy} className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1 font-semibold text-ink hover:bg-sunken">
            <Copy aria-hidden className="size-3.5" /> {copied ?? "Copy log"}
          </button>
        </div>

        <ol ref={list} role="log" aria-live="polite" aria-label="Các dòng log" className="flex-1 overflow-y-auto py-1.5 font-mono text-[12.8px] leading-[1.5]">
          {lines.length === 0 && <li className="px-[18px] py-4 font-sans text-ink-3">Chưa có log.</li>}
          {lines.map((line, index) => (
            <li
              key={`${line.key ?? index}-${line.src}`}
              hidden={!shown(line)}
              className={cx("grid grid-cols-[62px_86px_minmax(0,1fr)] gap-2.5 border-l-[3px] border-transparent px-[18px] py-[3px]", LEVEL_ROW[line.level])}
            >
              <span className="num text-ink-3">{clockTime(new Date(line.t * 1000))}</span>
              <span className="flex min-w-0 items-center gap-1.5 text-ink-2">
                <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: sourceColor(line.src) ?? "var(--ink-3)" }} />
                <span className="truncate">{sourceLabel(line.src)}</span>
              </span>
              <span className={cx("min-w-0 break-words", LEVEL_TEXT[line.level])}>{line.msg}</span>
            </li>
          ))}
        </ol>
      </aside>
    </>
  );
}
