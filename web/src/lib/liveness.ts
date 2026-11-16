// "Alive or dead" for each part of a Node 1 job, read from its live log (design D13).
import type { Job, JobEvent } from "@/lib/api";

export const SILENT_S = 60; // a waiting part with no new log line for this long is flagged
// How long the backend keeps waiting before it gives up (httpx read timeouts in src/api).
const BACKEND_WAIT_S: Record<string, number> = { ocr: 300, endpoint: 600 };

export type Tone = "idle" | "warn" | "good" | "bad";
export interface PartState {
  tone: Tone;
  text: string;
  busy: boolean; // something is still in flight: show a spinner
  silentFor?: number; // seconds without a new line, once over SILENT_S
}

export function durationVi(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  if (!minutes) return `${rest} s`;
  return rest ? `${minutes} phút ${rest} s` : `${minutes} phút`;
}

const vi = (n: number | null | undefined) => (n ?? 0).toLocaleString("vi-VN");
// Short durations keep one decimal (2,7 s); longer ones read in minutes and seconds.
const tookVi = (seconds: number) =>
  seconds < 10 ? `${seconds.toLocaleString("vi-VN", { maximumFractionDigits: 1 })} s` : durationVi(seconds);

function lastOf(events: JobEvent[], src: string): JobEvent | undefined {
  let last: JobEvent | undefined;
  for (const event of events) if (event.src === src && (!last || event.t >= last.t)) last = event;
  return last;
}

const isFinal = (job: Job) => ["done", "failed", "interrupted"].includes(job.status);

export function systemState(job: Job, lostFor: number | null): PartState {
  if (lostFor !== null) return { tone: "bad", text: `Mất liên lạc ${durationVi(lostFor)}`, busy: true };
  if (job.status === "failed" || job.status === "interrupted") return { tone: "bad", text: "Job dừng vì lỗi", busy: false };
  if (job.status === "done") return { tone: "good", text: "Đã xong", busy: false };
  return { tone: "good", text: "Sống", busy: true };
}

export function ocrState(job: Job, events: JobEvent[], now: number): PartState {
  const last = lastOf(events, "ocr");
  if (!last) return { tone: "idle", text: isFinal(job) ? "Không chạy" : "Chưa chạy", busy: false };
  if (last.level === "ok") {
    const ms = job.result.timings_ms?.ocr;
    return { tone: "good", text: ms ? `Xong trong ${tookVi(ms / 1000)}` : "Xong", busy: false };
  }
  if (job.status === "failed" && (job.error ?? "").startsWith("ocr")) return { tone: "bad", text: "Lỗi", busy: false };
  if (isFinal(job)) return { tone: "idle", text: "Đã dừng", busy: false };
  return waiting(last, now, "Máy OCR", "Đang đọc ảnh");
}

export function endpointState(job: Job, events: JobEvent[], endpointId: string, now: number): PartState {
  const report = job.result.runs?.[endpointId];
  if (report) return report.valid ? { tone: "good", text: "Xong", busy: false } : { tone: "bad", text: "Lỗi", busy: false };
  const last = lastOf(events, endpointId);
  if (!last) return { tone: "idle", text: isFinal(job) ? "Không chạy" : "Chờ OCR xong", busy: false };
  if (last.level === "error") return { tone: "bad", text: "Lỗi", busy: false };
  if (isFinal(job)) return { tone: "idle", text: "Đã dừng", busy: false };
  return waiting(last, now, "GPU", "Chờ token đầu");
}

function waiting(last: JobEvent, now: number, who: string, firstWait: string): PartState {
  const age = now - last.t;
  if (last.key?.endsWith("-wait")) {
    return { tone: "warn", text: `${who} đang khởi động, đã thử ${vi(last.data?.tries)} lần`, busy: true };
  }
  if (age >= SILENT_S) return { tone: "bad", text: `Không có tín hiệu ${durationVi(age)}`, busy: true, silentFor: age };
  if (last.key?.endsWith("-recv")) return { tone: "good", text: `Đang nhận ${vi(last.data?.chars)} ký tự`, busy: true };
  if (last.level === "ok") return { tone: "good", text: "Đang trả lời", busy: true };
  return { tone: "warn", text: `${firstWait}, ${durationVi(age)}`, busy: true };
}

// A warning line the web adds for a part that went quiet; it is not stored on the server.
export function silentLine(src: string, state: PartState, now: number): JobEvent | null {
  if (state.silentFor === undefined) return null;
  const isOcr = src === "ocr";
  return {
    t: now, src, level: "warn", key: `${src}-silent`,
    msg: `Chưa có tín hiệu trong ${durationVi(state.silentFor)}; ${isOcr ? "máy OCR" : "endpoint"} có thể đã dừng. `
      + `Backend chờ tối đa ${durationVi(BACKEND_WAIT_S[isOcr ? "ocr" : "endpoint"])}`,
  };
}
