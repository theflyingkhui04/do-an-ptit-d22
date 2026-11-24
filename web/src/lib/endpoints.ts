// Endpoint metadata helpers, alias suggestions, serving_config.json parser, and short display lines.
import type { Endpoint, EndpointMeta, EndpointNotes } from "@/lib/api";
import { localeTag } from "@/lib/format";
import type { Lang } from "@/lib/i18n";

export const PLATFORM_LABELS_BY_LANG: Record<Lang, Record<string, string>> = {
  vi: {
    modal: "Modal",
    tunnel: "Tunnel (Colab / Local)",
    other: "Khác",
    local: "Local",
  },
  en: {
    modal: "Modal",
    tunnel: "Tunnel (Colab / Local)",
    other: "Other",
    local: "Local",
  },
};

export function platformLabel(platform: string, lang: Lang = "vi"): string {
  return PLATFORM_LABELS_BY_LANG[lang][platform] ?? platform;
}

export type Decoding = "" | "vanilla" | "static" | "adaptive" | "draft";

export const DECODING_LABELS_BY_LANG: Record<Lang, Record<Exclude<Decoding, "">, string>> = {
  vi: {
    vanilla: "Tiêu chuẩn (Vanilla)",
    static: "EAGLE-3 (K cố định)",
    adaptive: "EAGLE-3 (Hardware-Adaptive)",
    draft: "Draft model độc lập",
  },
  en: {
    vanilla: "Standard (Vanilla)",
    static: "EAGLE-3 (Static K)",
    adaptive: "EAGLE-3 (Hardware-Adaptive)",
    draft: "Standalone draft model",
  },
};

export function decodingLabel(decoding: Exclude<Decoding, "">, lang: Lang = "vi"): string {
  return DECODING_LABELS_BY_LANG[lang][decoding];
}

export const GPU_SUGGESTIONS = ["NVIDIA A10G", "NVIDIA L4", "NVIDIA T4", "NVIDIA A100", "NVIDIA H100"];

/** Suggests the next unused short alias (#1, #2, ...) for a new endpoint. */
export function nextAlias(endpoints: Pick<Endpoint, "name">[]): string {
  const taken = new Set(endpoints.map((e) => e.name.trim().toLowerCase()));
  let n = 1;
  while (taken.has(`#${n}`)) n += 1;
  return `#${n}`;
}

export function decodingOf(meta: EndpointMeta): Decoding {
  const method = meta.draft_method;
  if (method === undefined) return "";
  if (method === null || method === "none") return "vanilla";
  if (method === "draft_model") return "draft";
  return meta.adaptive ? "adaptive" : "static";
}

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

function withoutTrailingSlashes(raw: string): string {
  let end = raw.length;
  while (end > 0 && raw[end - 1] === "/") end -= 1;
  return raw.slice(0, end);
}

const lastPart = (name: string) => withoutTrailingSlashes(name).split("/").pop() || name;

export function setupLine(e: Endpoint, lang: Lang = "vi"): string | null {
  const decoding = decodingOf(e.meta ?? {});
  const parts = [e.meta?.gpu, decoding ? decodingLabel(decoding, lang) : undefined].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

export function detailLine(e: Endpoint, baseModel: string, lang: Lang = "vi"): string {
  const parts = [platformLabel(e.platform, lang), lastPart(baseModel)];
  if (e.max_model_len) {
    const formatted = e.max_model_len.toLocaleString(localeTag(lang));
    parts.push(lang === "en" ? `${formatted} tokens` : `${formatted} token`);
  }
  if (e.has_key) parts.push(lang === "en" ? "API key" : "có API key");
  return parts.join(" · ");
}

export function shortUrl(url: string): string {
  const bare = url.replace(/^https?:\/\//, "");
  if (bare.length <= 40) return bare;
  const dot = bare.indexOf(".");
  const slash = bare.indexOf("/");
  return dot > 0 ? `${bare.slice(0, dot)}…${slash > 0 ? bare.slice(slash) : ""}` : `${bare.slice(0, 24)}…${bare.slice(-12)}`;
}

export function agoText(iso: string, now: number, lang: Lang = "vi"): string {
  const seconds = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (lang === "en") {
    if (seconds < 5) return "just now";
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }
  if (seconds < 5) return "vừa xong";
  if (seconds < 60) return `${seconds} giây trước`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} phút trước`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} giờ trước`;
  return `${Math.floor(seconds / 86400)} ngày trước`;
}

export const agoVi = (iso: string, now: number) => agoText(iso, now, "vi");

export function shortReason(message: string | null, lang: Lang = "vi"): string {
  if (!message) return "";
  const busy = /HTTP (50[234])/.exec(message);
  if (busy) {
    return lang === "en"
      ? `GPU starting up (HTTP ${busy[1]})`
      : `GPU đang khởi động (HTTP ${busy[1]})`;
  }
  return message;
}
