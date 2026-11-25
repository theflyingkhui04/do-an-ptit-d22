// Endpoint metadata helpers, alias suggestions, serving_config.json parser, and short display lines.
import type { Endpoint, EndpointMeta, EndpointNotes } from "@/lib/api";
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

export type Decoding =
  | ""
  | "vanilla"
  | "eagle1_static"
  | "eagle1_adaptive"
  | "eagle2_static"
  | "eagle2_adaptive"
  | "eagle3_static"
  | "eagle3_adaptive"
  | "eagle31_static"
  | "eagle31_adaptive";

export const DECODING_LABELS_BY_LANG: Record<Lang, Record<Exclude<Decoding, "">, string>> = {
  vi: {
    vanilla: "Tiêu chuẩn (Vanilla)",
    eagle1_static: "EAGLE-1 (K cố định)",
    eagle1_adaptive: "EAGLE-1 (Hardware-Adaptive)",
    eagle2_static: "EAGLE-2 (K cố định)",
    eagle2_adaptive: "EAGLE-2 (Hardware-Adaptive)",
    eagle3_static: "EAGLE-3 (K cố định)",
    eagle3_adaptive: "EAGLE-3 (Hardware-Adaptive)",
    eagle31_static: "EAGLE-3.1 (K cố định)",
    eagle31_adaptive: "EAGLE-3.1 (Hardware-Adaptive)",
  },
  en: {
    vanilla: "Standard (Vanilla)",
    eagle1_static: "EAGLE-1 (Static K)",
    eagle1_adaptive: "EAGLE-1 (Hardware-Adaptive)",
    eagle2_static: "EAGLE-2 (Static K)",
    eagle2_adaptive: "EAGLE-2 (Hardware-Adaptive)",
    eagle3_static: "EAGLE-3 (Static K)",
    eagle3_adaptive: "EAGLE-3 (Hardware-Adaptive)",
    eagle31_static: "EAGLE-3.1 (Static K)",
    eagle31_adaptive: "EAGLE-3.1 (Hardware-Adaptive)",
  },
};

export function decodingLabel(decoding: Exclude<Decoding, "">, lang: Lang = "vi"): string {
  return DECODING_LABELS_BY_LANG[lang][decoding];
}

export const GPU_SUGGESTIONS = ["NVIDIA A10", "NVIDIA L40S", "NVIDIA A100", "NVIDIA T4", "NVIDIA H100"];

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
  const adaptive = Boolean(meta.adaptive);
  const norm = method.toLowerCase().replace(/[\s_-]+/g, "");
  if (norm === "eagle1") return adaptive ? "eagle1_adaptive" : "eagle1_static";
  if (norm === "eagle2") return adaptive ? "eagle2_adaptive" : "eagle2_static";
  if (norm === "eagle3.1" || norm === "eagle31") return adaptive ? "eagle31_adaptive" : "eagle31_static";
  if (norm === "eagle3" || norm === "eagle") return adaptive ? "eagle3_adaptive" : "eagle3_static";
  return adaptive ? "eagle3_adaptive" : "eagle3_static";
}

export function isHardwareAdaptive(meta: EndpointMeta | undefined): boolean {
  return Boolean(meta?.adaptive);
}

export function notesFor(decoding: Decoding, draftModel: string): Required<Pick<EndpointNotes, "draft_method" | "adaptive" | "draft_model">> {
  const draft = draftModel.trim() || null;
  switch (decoding) {
    case "vanilla": return { draft_method: "none", adaptive: false, draft_model: null };
    case "eagle1_static": return { draft_method: "eagle1", adaptive: false, draft_model: draft };
    case "eagle1_adaptive": return { draft_method: "eagle1", adaptive: true, draft_model: draft };
    case "eagle2_static": return { draft_method: "eagle2", adaptive: false, draft_model: draft };
    case "eagle2_adaptive": return { draft_method: "eagle2", adaptive: true, draft_model: draft };
    case "eagle3_static": return { draft_method: "eagle3", adaptive: false, draft_model: draft };
    case "eagle3_adaptive": return { draft_method: "eagle3", adaptive: true, draft_model: draft };
    case "eagle31_static": return { draft_method: "eagle3.1", adaptive: false, draft_model: draft };
    case "eagle31_adaptive": return { draft_method: "eagle3.1", adaptive: true, draft_model: draft };
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

function versionedEagleMethod(eagleVersion: string | undefined): string | undefined {
  if (!eagleVersion) return undefined;
  const norm = eagleVersion.toLowerCase().replace(/^eagle[\s_-]*/i, "").replace(/[\s_-]+/g, "");
  if (norm === "1") return "eagle1";
  if (norm === "2") return "eagle2";
  if (norm === "3") return "eagle3";
  if (norm === "3.1" || norm === "31") return "eagle3.1";
  return undefined;
}

function normalizeGpuLabel(raw: string | undefined): string | undefined {
  if (!raw) return undefined;
  let cleaned = raw.replace(/^remote\s+/i, "").trim();
  if (!cleaned) return undefined;
  if (/^a10g$/i.test(cleaned)) cleaned = "A10";
  if (/^nvidia\s+a10g$/i.test(cleaned)) return "NVIDIA A10";
  if (/^(t4|a10|l4|l40s|a100|h100)$/i.test(cleaned)) {
    return `NVIDIA ${cleaned.toUpperCase()}`;
  }
  return cleaned;
}

export function notesFromServingConfig(value: unknown): ServingNotes | null {
  if (!isObject(value)) return null;
  const serving = isObject(value.serving) ? value.serving : {};
  const hardware = isObject(value.hardware) ? value.hardware : {};
  const pick = (key: string) => (key in value ? value[key] : serving[key]);
  const draftModel = text(pick("draft_model"));
  const method = versionedEagleMethod(text(pick("eagle_version"))) ?? text(pick("draft_method"));
  const adaptive = Boolean(pick("enable_adaptive") ?? pick("adaptive_enabled"));
  let decoding: Decoding;
  if (!draftModel) {
    decoding = "vanilla";
  } else {
    decoding = decodingOf({ draft_method: method ?? "eagle3", adaptive });
  }
  const notes: ServingNotes = { decoding };
  const gpu = normalizeGpuLabel(text(hardware.device_name) ?? text(pick("target_gpu")));
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

export function shortModelName(name: string): string {
  return withoutTrailingSlashes(name).split("/").pop() || name;
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
