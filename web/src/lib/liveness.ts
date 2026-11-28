// Live status of each part of a Node 1 job, derived from server log events.
import type { Job, JobEvent } from "@/lib/api";
import { localeTag } from "@/lib/format";
import type { Lang } from "@/lib/i18n";

export const SILENT_S = 60;
const BACKEND_WAIT_S: Record<string, number> = { ocr: 300, endpoint: 600 };

export type Tone = "idle" | "warn" | "good" | "bad";
export interface PartState {
  tone: Tone;
  text: string;
  busy: boolean;
  silentFor?: number;
}

export function durationText(seconds: number, lang: Lang = "vi"): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  if (lang === "en") {
    if (!minutes) return `${rest} s`;
    return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
  }
  if (!minutes) return `${rest} s`;
  return rest ? `${minutes} phút ${rest} s` : `${minutes} phút`;
}

export const durationVi = (seconds: number) => durationText(seconds, "vi");

const num = (n: number | null | undefined, lang: Lang) => (n ?? 0).toLocaleString(localeTag(lang));
const tookText = (seconds: number, lang: Lang) =>
  seconds < 10
    ? `${seconds.toLocaleString(localeTag(lang), { maximumFractionDigits: 1 })} s`
    : durationText(seconds, lang);

function lastOf(events: JobEvent[], src: string): JobEvent | undefined {
  let last: JobEvent | undefined;
  for (const event of events) if (event.src === src && (!last || event.t >= last.t)) last = event;
  return last;
}

const isFinal = (job: Job) => ["done", "failed", "interrupted"].includes(job.status);

export function systemState(job: Job, lostFor: number | null, lang: Lang = "vi"): PartState {
  if (lostFor !== null) {
    return {
      tone: "bad",
      text: lang === "en" ? `Reconnecting (${durationText(lostFor, lang)})` : `Mất kết nối (${durationText(lostFor, lang)})`,
      busy: true,
    };
  }
  if (job.status === "failed" || job.status === "interrupted") {
    return { tone: "bad", text: lang === "en" ? "Failed" : "Lỗi xử lý", busy: false };
  }
  if (job.status === "done") {
    return { tone: "good", text: lang === "en" ? "Completed" : "Hoàn tất", busy: false };
  }
  return { tone: "good", text: lang === "en" ? "Active" : "Đang hoạt động", busy: true };
}

export function ocrState(job: Job, events: JobEvent[], now: number, lang: Lang = "vi"): PartState {
  const last = lastOf(events, "ocr");
  if (!last) {
    return {
      tone: "idle",
      text: isFinal(job) ? (lang === "en" ? "Skipped" : "Không chạy") : (lang === "en" ? "Pending" : "Đang chờ"),
      busy: false,
    };
  }
  if (last.level === "ok") {
    const ms = job.result.timings_ms?.ocr;
    return {
      tone: "good",
      text: ms ? (lang === "en" ? `Done (${tookText(ms / 1000, lang)})` : `Hoàn tất (${tookText(ms / 1000, lang)})`) : (lang === "en" ? "Done" : "Hoàn tất"),
      busy: false,
    };
  }
  if (job.status === "failed" && (job.error ?? "").startsWith("ocr")) {
    return { tone: "bad", text: lang === "en" ? "Error" : "Lỗi", busy: false };
  }
  if (isFinal(job)) return { tone: "idle", text: lang === "en" ? "Stopped" : "Đã dừng", busy: false };
  return waiting(last, now, "OCR", lang === "en" ? "Reading image" : "Đang nhận dạng", lang);
}

export function endpointState(job: Job, events: JobEvent[], endpointId: string, now: number, lang: Lang = "vi"): PartState {
  const report = job.result.runs?.[endpointId];
  if (report) {
    return report.valid
      ? { tone: "good", text: lang === "en" ? "Completed" : "Hoàn tất", busy: false }
      : { tone: "bad", text: lang === "en" ? "Error" : "Lỗi", busy: false };
  }
  const last = lastOf(events, endpointId);
  if (!last) {
    return {
      tone: "idle",
      text: isFinal(job) ? (lang === "en" ? "Skipped" : "Không chạy") : (lang === "en" ? "Waiting for OCR" : "Chờ OCR"),
      busy: false,
    };
  }
  if (last.level === "error") return { tone: "bad", text: lang === "en" ? "Error" : "Lỗi", busy: false };
  if (isFinal(job)) return { tone: "idle", text: lang === "en" ? "Stopped" : "Đã dừng", busy: false };
  return waiting(last, now, "GPU", lang === "en" ? "Waiting for first token" : "Chờ token đầu", lang);
}

function waiting(last: JobEvent, now: number, who: string, firstWait: string, lang: Lang): PartState {
  const age = now - last.t;
  if (last.key?.endsWith("-wait")) {
    return {
      tone: "warn",
      text: lang === "en"
        ? `${who} starting (attempt ${num(last.data?.tries, lang)})`
        : `${who} đang khởi động (lần thử ${num(last.data?.tries, lang)})`,
      busy: true,
    };
  }
  if (age >= SILENT_S) {
    return {
      tone: "bad",
      text: lang === "en" ? `No signal (${durationText(age, lang)})` : `Không có phản hồi (${durationText(age, lang)})`,
      busy: true,
      silentFor: age,
    };
  }
  if (last.key?.endsWith("-recv")) {
    return {
      tone: "good",
      text: lang === "en" ? `Streaming (${num(last.data?.chars, lang)} chars)` : `Đang nhận (${num(last.data?.chars, lang)} ký tự)`,
      busy: true,
    };
  }
  if (last.level === "ok") return { tone: "good", text: lang === "en" ? "Streaming" : "Đang sinh văn bản", busy: true };
  return { tone: "warn", text: `${firstWait} (${durationText(age, lang)})`, busy: true };
}

export function silentLine(src: string, state: PartState, now: number, lang: Lang = "vi"): JobEvent | null {
  if (state.silentFor === undefined) return null;
  const isOcr = src === "ocr";
  const waitMax = durationText(BACKEND_WAIT_S[isOcr ? "ocr" : "endpoint"], lang);
  return {
    t: now,
    src,
    level: "warn",
    key: `${src}-silent`,
    msg: lang === "en"
      ? `No response for ${durationText(state.silentFor, lang)}; timeout is ${waitMax}.`
      : `Không có phản hồi trong ${durationText(state.silentFor, lang)}; thời gian chờ tối đa ${waitMax}.`,
  };
}
