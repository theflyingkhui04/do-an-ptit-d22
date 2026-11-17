// Text and notes for the Endpoint page (design D14): how an endpoint is served, short list text, serving_config.json.
import type { Endpoint, EndpointMeta, EndpointNotes } from "@/lib/api";

export const PLATFORM_LABELS: Record<string, string> = {
  modal: "Modal",
  tunnel: "Đường hầm (Colab hoặc máy cá nhân)",
  other: "Khác",
  local: "Máy cá nhân",
};

export type Decoding = "" | "vanilla" | "static" | "adaptive" | "draft";

export const DECODING_LABELS: Record<Exclude<Decoding, "">, string> = {
  vanilla: "Không nháp (vanilla)",
  static: "EAGLE-3, K cố định",
  adaptive: "EAGLE-3, Hardware-Adaptive",
  draft: "Model nháp riêng",
};

// Suggestions for the GPU note; anything else can be typed.
export const GPU_SUGGESTIONS = ["NVIDIA A10G", "NVIDIA L4", "NVIDIA T4", "NVIDIA A100", "NVIDIA H100"];

export function decodingOf(meta: EndpointMeta): Decoding {
  const method = meta.draft_method;
  if (method === undefined) return "";
  if (method === null || method === "none") return "vanilla"; // null: written by the server-side parser before D14
  if (method === "draft_model") return "draft";
  return meta.adaptive ? "adaptive" : "static";
}

// Notes to send for a decoding choice; null removes a note that no longer applies.
export function notesFor(decoding: Decoding, draftModel: string): Required<Pick<EndpointNotes, "draft_method" | "adaptive" | "draft_model">> {
  const draft = draftModel.trim() || null;
  switch (decoding) {
    case "vanilla": return { draft_method: "none", adaptive: null, draft_model: null };
    case "static": return { draft_method: "eagle3", adaptive: false, draft_model: draft };
    case "adaptive": return { draft_method: "eagle3", adaptive: true, draft_model: draft };
    case "draft": return { draft_method: "draft_model", adaptive: false, draft_model: draft };
    default: return { draft_method: null, adaptive: null, draft_model: null };
  }
}

export interface ServingNotes {
  gpu?: string;
  decoding: Decoding;
  draft_model?: string;
  target_model?: string;
}

const text = (value: unknown) => (typeof value === "string" && value.trim() ? value.trim() : undefined);
const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// serving_config.json as the CLI writes it: the wizard puts the fields at the top level
// (src/flexsds/cli/wizard.py), the serve command keeps them under "serving" (src/flexsds/cli/main.py).
export function notesFromServingConfig(value: unknown): ServingNotes | null {
  if (!isObject(value)) return null;
  const serving = isObject(value.serving) ? value.serving : {};
  const hardware = isObject(value.hardware) ? value.hardware : {};
  const pick = (key: string) => (key in value ? value[key] : serving[key]);
  const draftModel = text(pick("draft_model"));
  const method = text(pick("draft_method"));
  const adaptive = pick("enable_adaptive") ?? pick("adaptive_enabled");
  const decoding: Decoding = !draftModel ? "vanilla" : method === "draft_model" ? "draft" : adaptive ? "adaptive" : "static";
  const notes: ServingNotes = { decoding };
  const gpu = text(hardware.device_name) ?? text(pick("target_gpu"));
  if (gpu) notes.gpu = gpu;
  if (draftModel) notes.draft_model = draftModel;
  const target = text(pick("target_model"));
  if (target) notes.target_model = target;
  return notes;
}

const lastPart = (name: string) => name.replace(/\/+$/, "").split("/").pop() || name;

// "NVIDIA A10G, EAGLE-3, Hardware-Adaptive" when noted, else null.
export function setupLine(e: Endpoint): string | null {
  const decoding = decodingOf(e.meta ?? {});
  const parts = [e.meta?.gpu, decoding ? DECODING_LABELS[decoding] : undefined].filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}

export function detailLine(e: Endpoint, baseModel: string): string {
  const parts = [PLATFORM_LABELS[e.platform] ?? e.platform, lastPart(baseModel)];
  if (e.max_model_len) parts.push(`ngữ cảnh ${e.max_model_len.toLocaleString("vi-VN")}`);
  const decoding = decodingOf(e.meta ?? {});
  if (e.meta?.draft_model && decoding && decoding !== "vanilla") parts.push(`nháp ${lastPart(e.meta.draft_model)}`);
  if (e.has_key) parts.push("có khoá API");
  return parts.join(", ");
}

// Long Modal addresses keep the part that tells them apart: the first host label and the path.
export function shortUrl(url: string): string {
  const bare = url.replace(/^https?:\/\//, "");
  if (bare.length <= 40) return bare;
  const dot = bare.indexOf(".");
  const slash = bare.indexOf("/");
  return dot > 0 ? `${bare.slice(0, dot)}…${slash > 0 ? bare.slice(slash) : ""}` : `${bare.slice(0, 24)}…${bare.slice(-12)}`;
}

export function agoVi(iso: string, now: number): string {
  const seconds = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (seconds < 5) return "vừa xong";
  if (seconds < 60) return `${seconds} giây trước`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} phút trước`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} giờ trước`;
  return `${Math.floor(seconds / 86400)} ngày trước`;
}

// The full message stays in the edit drawer; the list shows the gist.
export function shortReason(message: string | null): string {
  if (!message) return "";
  const busy = /HTTP (50[234])/.exec(message);
  return busy ? `GPU đang ngủ hoặc khởi động (HTTP ${busy[1]})` : message;
}
