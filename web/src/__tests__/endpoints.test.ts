import { describe, expect, it } from "vitest";

import {
  GPU_SUGGESTIONS,
  agoText,
  agoVi,
  decodingOf,
  nextAlias,
  notesFor,
  notesFromServingConfig,
  shortReason,
  shortUrl,
} from "@/lib/endpoints";

describe("serving_config.json from the CLI", () => {
  it("reads the file the wizard writes (fields at the top level, GPU under hardware)", () => {
    const wizard = {
      flexsds_version: "0.3.4", action: "serve", backend: "modal", target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ",
      draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator", draft_method: "eagle3", eagle_version: "3.1",
      enable_adaptive: true, target_gpu: "A10",
      hardware: { device_name: "NVIDIA A10", sm_architecture: "sm_86", free_vram_mb: 22000 },
    };
    expect(notesFromServingConfig(wizard)).toEqual({
      gpu: "NVIDIA A10", decoding: "eagle31_adaptive", draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator",
      target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ",
    });
  });

  it("maps EAGLE-1, EAGLE-2, EAGLE-3, and EAGLE-3.1 from eagle_version in CLI JSON", () => {
    expect(notesFromServingConfig({
      draft_model: "org/e1", draft_method: "eagle", eagle_version: "1", enable_adaptive: false,
      hardware: { device_name: "Remote A10" },
    })).toEqual({ gpu: "NVIDIA A10", decoding: "eagle1_static", draft_model: "org/e1" });

    expect(notesFromServingConfig({
      draft_model: "org/e2", draft_method: "eagle", eagle_version: "2", enable_adaptive: true,
      hardware: { device_name: "NVIDIA L40S" },
    })).toEqual({ gpu: "NVIDIA L40S", decoding: "eagle2_adaptive", draft_model: "org/e2" });

    expect(notesFromServingConfig({
      draft_model: "org/e3", draft_method: "eagle3", eagle_version: "3", enable_adaptive: false,
      hardware: { device_name: "NVIDIA A100" },
    })).toEqual({ gpu: "NVIDIA A100", decoding: "eagle3_static", draft_model: "org/e3" });

    expect(notesFromServingConfig({
      draft_model: "org/e31", draft_method: "eagle3", eagle_version: "EAGLE-3.1", enable_adaptive: false,
      target_gpu: "T4",
    })).toEqual({ gpu: "NVIDIA T4", decoding: "eagle31_static", draft_model: "org/e31" });
  });

  it("normalizes GPU suggestions to A10, L40S, A100, and T4", () => {
    expect(GPU_SUGGESTIONS).toContain("NVIDIA A10");
    expect(GPU_SUGGESTIONS).toContain("NVIDIA L40S");
    expect(GPU_SUGGESTIONS).toContain("NVIDIA A100");
    expect(GPU_SUGGESTIONS).toContain("NVIDIA T4");
    expect(GPU_SUGGESTIONS).not.toContain("NVIDIA A10G");
  });

  it("also reads the nested serving block and its older adaptive_enabled name", () => {
    expect(notesFromServingConfig({ serving: { draft_method: "eagle3", draft_model: "d", adaptive_enabled: false } }))
      .toEqual({ decoding: "eagle3_static", draft_model: "d" });
    expect(notesFromServingConfig({ serving: { draft_method: "eagle3.1", draft_model: "d", adaptive_enabled: true } }))
      .toEqual({ decoding: "eagle31_adaptive", draft_model: "d" });
    expect(notesFromServingConfig({ draft_model: null, enable_adaptive: false })).toEqual({ decoding: "vanilla" });
  });

  it("refuses something that is not a JSON object", () => {
    expect(notesFromServingConfig(["a"])).toBeNull();
    expect(notesFromServingConfig("text")).toBeNull();
    expect(notesFromServingConfig(null)).toBeNull();
  });
});

describe("decoding notes", () => {
  it("reads every EAGLE-1/2/3/3.1 decoding mode from stored notes", () => {
    expect(decodingOf({})).toBe("");
    expect(decodingOf({ draft_method: "none" })).toBe("vanilla");
    expect(decodingOf({ draft_method: null })).toBe("vanilla");
    expect(decodingOf({ draft_method: "eagle1", adaptive: false })).toBe("eagle1_static");
    expect(decodingOf({ draft_method: "eagle1", adaptive: true })).toBe("eagle1_adaptive");
    expect(decodingOf({ draft_method: "eagle2", adaptive: false })).toBe("eagle2_static");
    expect(decodingOf({ draft_method: "eagle2", adaptive: true })).toBe("eagle2_adaptive");
    expect(decodingOf({ draft_method: "eagle3", adaptive: false })).toBe("eagle3_static");
    expect(decodingOf({ draft_method: "eagle3", adaptive: true })).toBe("eagle3_adaptive");
    expect(decodingOf({ draft_method: "eagle3.1", adaptive: false })).toBe("eagle31_static");
    expect(decodingOf({ draft_method: "eagle3.1", adaptive: true })).toBe("eagle31_adaptive");
  });

  it("writes notes that match the chosen decoding method", () => {
    expect(notesFor("vanilla", "ignored")).toEqual({ draft_method: "none", adaptive: false, draft_model: null });
    expect(notesFor("eagle1_static", " d ")).toEqual({ draft_method: "eagle1", adaptive: false, draft_model: "d" });
    expect(notesFor("eagle2_adaptive", " d ")).toEqual({ draft_method: "eagle2", adaptive: true, draft_model: "d" });
    expect(notesFor("eagle3_adaptive", " d ")).toEqual({ draft_method: "eagle3", adaptive: true, draft_model: "d" });
    expect(notesFor("eagle31_static", "")).toEqual({ draft_method: "eagle3.1", adaptive: false, draft_model: null });
    expect(notesFor("eagle31_adaptive", "e31")).toEqual({ draft_method: "eagle3.1", adaptive: true, draft_model: "e31" });
    expect(notesFor("", "m")).toEqual({ draft_method: null, adaptive: null, draft_model: null });
  });
});

describe("alias suggestions and list formatting", () => {
  it("suggests the next free #N alias", () => {
    expect(nextAlias([])).toBe("#1");
    expect(nextAlias([{ name: "#1" }])).toBe("#2");
    expect(nextAlias([{ name: "#1" }, { name: "#3" }])).toBe("#2");
    expect(nextAlias([{ name: "vanilla" }, { name: "#1" }, { name: "#2" }])).toBe("#3");
  });

  it("shortens a long address to its first host label and path", () => {
    expect(shortUrl("https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1")).toBe("u--ptit-vllm-demo-vanilla…/v1");
    expect(shortUrl("https://abc.trycloudflare.com/v1")).toBe("abc.trycloudflare.com/v1");
  });

  it("formats relative check times in both VI and EN", () => {
    const now = Date.parse("2026-11-17T10:00:00Z");
    expect(agoVi("2026-11-17T09:59:58Z", now)).toBe("vừa xong");
    expect(agoVi("2026-11-17T09:59:20Z", now)).toBe("40 giây trước");
    expect(agoVi("2026-11-17T09:35:00Z", now)).toBe("25 phút trước");
    expect(agoText("2026-11-17T09:59:58Z", now, "en")).toBe("just now");
    expect(agoText("2026-11-17T09:35:00Z", now, "en")).toBe("25m ago");
    expect(agoText("2026-11-15T09:00:00Z", now, "en")).toBe("2d ago");
  });

  it("gives a concise reason for a starting/stopped endpoint in VI and EN", () => {
    expect(shortReason("Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động", "vi")).toBe("GPU đang khởi động (HTTP 503)");
    expect(shortReason("Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động", "en")).toBe("GPU starting up (HTTP 503)");
    expect(shortReason(null)).toBe("");
  });
});
