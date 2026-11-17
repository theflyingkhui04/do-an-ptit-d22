import { describe, expect, it } from "vitest";

import { agoVi, decodingOf, notesFor, notesFromServingConfig, shortReason, shortUrl } from "@/lib/endpoints";

describe("serving_config.json from the CLI", () => {
  it("reads the file the wizard writes (fields at the top level, GPU under hardware)", () => {
    // src/flexsds/cli/wizard.py: target_model, draft_model, draft_method, enable_adaptive at the top level
    const wizard = {
      flexsds_version: "0.2.1", action: "serve", backend: "modal", target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ",
      draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator", draft_method: "eagle3", enable_adaptive: true,
      target_gpu: "A10G", hardware: { device_name: "NVIDIA A10G", sm_architecture: "sm_86", free_vram_mb: 22000 },
    };
    expect(notesFromServingConfig(wizard)).toEqual({
      gpu: "NVIDIA A10G", decoding: "adaptive", draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator",
      target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ",
    });
  });

  it("also reads the nested serving block and its older adaptive_enabled name", () => {
    expect(notesFromServingConfig({ serving: { draft_model: "x/draft", draft_method: "draft_model", target_gpu: "L4" } }))
      .toEqual({ gpu: "L4", decoding: "draft", draft_model: "x/draft" });
    expect(notesFromServingConfig({ serving: { draft_method: "eagle3", draft_model: "d", adaptive_enabled: false } }))
      .toEqual({ decoding: "static", draft_model: "d" });
    expect(notesFromServingConfig({ draft_model: null, enable_adaptive: false })).toEqual({ decoding: "vanilla" });
  });

  it("refuses something that is not a JSON object", () => {
    expect(notesFromServingConfig(["a"])).toBeNull();
    expect(notesFromServingConfig("text")).toBeNull();
    expect(notesFromServingConfig(null)).toBeNull();
  });
});

describe("decoding notes", () => {
  it("reads the decoding from the stored notes", () => {
    expect(decodingOf({})).toBe("");
    expect(decodingOf({ draft_method: "none" })).toBe("vanilla");
    expect(decodingOf({ draft_method: null })).toBe("vanilla"); // written by the old parser
    expect(decodingOf({ draft_method: "eagle3", adaptive: true })).toBe("adaptive");
    expect(decodingOf({ draft_method: "eagle", adaptive: false })).toBe("static");
    expect(decodingOf({ draft_method: "draft_model" })).toBe("draft");
  });

  it("writes notes that remove what does not apply", () => {
    expect(notesFor("vanilla", "ignored")).toEqual({ draft_method: "none", adaptive: null, draft_model: null });
    expect(notesFor("adaptive", " d ")).toEqual({ draft_method: "eagle3", adaptive: true, draft_model: "d" });
    expect(notesFor("static", "")).toEqual({ draft_method: "eagle3", adaptive: false, draft_model: null });
    expect(notesFor("draft", "m")).toEqual({ draft_method: "draft_model", adaptive: false, draft_model: "m" });
    expect(notesFor("", "m")).toEqual({ draft_method: null, adaptive: null, draft_model: null });
  });
});

describe("list text", () => {
  it("shortens a long address to its first host label and path", () => {
    expect(shortUrl("https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1")).toBe("u--ptit-vllm-demo-vanilla…/v1");
    expect(shortUrl("https://abc.trycloudflare.com/v1")).toBe("abc.trycloudflare.com/v1");
  });

  it("says how long ago an endpoint was checked", () => {
    const now = Date.parse("2026-11-17T10:00:00Z");
    expect(agoVi("2026-11-17T09:59:58Z", now)).toBe("vừa xong");
    expect(agoVi("2026-11-17T09:59:20Z", now)).toBe("40 giây trước");
    expect(agoVi("2026-11-17T09:35:00Z", now)).toBe("25 phút trước");
    expect(agoVi("2026-11-17T07:00:00Z", now)).toBe("3 giờ trước");
    expect(agoVi("2026-11-15T09:00:00Z", now)).toBe("2 ngày trước");
  });

  it("gives a short reason for a stopped endpoint", () => {
    expect(shortReason("Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động hoặc đang ngủ, …")).toBe("GPU đang ngủ hoặc khởi động (HTTP 503)");
    expect(shortReason("Không kết nối được (ConnectError). Kiểm tra phiên flexsds serve còn chạy không."))
      .toBe("Không kết nối được (ConnectError). Kiểm tra phiên flexsds serve còn chạy không.");
    expect(shortReason(null)).toBe("");
  });
});
