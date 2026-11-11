import { vi } from "vitest";

import { ApiError, type Api, type InvoicePayload, type Job, type LaneReport } from "@/lib/api";

export const INVOICE: InvoicePayload = {
  document_type: "retail_receipt",
  seller: { name: "MINIMART ANAN", address: "Chợ Sủi Phú Thị Gia Lâm", tax_code: null, phone: null },
  buyer: { name: null, address: null, tax_code: null, phone: null },
  invoice: { invoice_number: null, issued_date: "2020-08-10", issued_time: "17:12:00", payment_method: null },
  totals: {
    subtotal: { raw: null, value: null, currency: "VND" },
    discount: { raw: null, value: null, currency: "VND" },
    vat_amount: { raw: null, value: null, currency: "VND" },
    total_amount: { raw: "42,000", value: 42000, currency: "VND" },
  },
  items: [
    {
      line_number: 1,
      description: "Sữa tươi",
      sku: "8934673",
      quantity: { raw: "2", value: 2, unit: "hộp" },
      unit_price: { raw: "10.000", value: 10000, currency: "VND" },
      line_total: { raw: "20.000", value: 20000, currency: "VND" },
      vat_rate: null,
    },
  ],
  taxes: [],
  extraction: { status: "partial", confidence: 0.8, missing_fields: [], warnings: [] },
};

export function lane(name: string, extra: Partial<LaneReport> = {}): LaneReport {
  return {
    lane: name,
    model: `model-${name}`,
    valid: true,
    latency_ms: 12000,
    ttft_ms: 300,
    tpot_ms: name === "vanilla" ? 20 : 14,
    output_tokens_per_s: 100,
    error: null,
    speedup_tpot_vs_vanilla: name === "vanilla" ? 1 : 1.43,
    speedup_latency_vs_vanilla: 1,
    equal_fields: 40,
    total_fields: 40,
    differing_fields: [],
    ...extra,
  };
}

export function doneJob(extra: Partial<Job["result"]> = {}): Job {
  return {
    job_id: "up_0123456789ab",
    status: "done",
    error: null,
    invoice_id: "up_0123456789ab:up_0123456789ab",
    result: {
      gpu: "A10G",
      ocr: { text: "MINIMART ANAN\nTong tien\n42,000", kind: "library", engine: "A", marks_stripped: true },
      lanes: { vanilla: lane("vanilla"), adaptive: lane("adaptive") },
      stored_lane: "adaptive",
      stored_from_fallback: false,
      invoice: INVOICE,
      grounding: { checked_fields: 5, grounded_fields: 5, ungrounded_fields: [] },
      item_check: { lines: [true], invoice_sum: false },
      timings_ms: { ocr: 9000, node1: 13000 },
      ...extra,
    },
  };
}

export function fakeApi(overrides: Partial<Record<keyof Api, unknown>> = {}): Api {
  const api: Api = {
    status: vi.fn(async () => ({
      backend: "ready",
      database: "ready",
      gpu: "A10G",
      lane_map: { node1: "adaptive" },
      services: { ocr: { state: "cold" as const, message: null }, adaptive: { state: "cold" as const, message: null } },
    })),
    warmup: vi.fn(async () => ({
      services: { ocr: { state: "ready" as const, message: null }, adaptive: { state: "ready" as const, message: null } },
    })),
    upload: vi.fn(async () => ({ job_id: "up_0123456789ab" })),
    job: vi.fn(async () => doneJob()),
    invoices: vi.fn(async () => ({ invoices: [] })),
    invoice: vi.fn(async () => ({ invoice_id: "x", payload: INVOICE, source: "upload" as const, image_url: null, lane: "adaptive" })),
    ask: vi.fn(async () => {
      throw new ApiError(500, "not scripted");
    }),
  };
  return { ...api, ...overrides } as Api;
}

export function jpegFile(name = "hoadon.jpg", type = "image/jpeg"): File {
  return new File([new Uint8Array([0xff, 0xd8, 0xff, 0xe0])], name, { type });
}
