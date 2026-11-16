import { vi } from "vitest";

import {
  ApiError,
  type Api,
  type BenchRun,
  type Endpoint,
  type InvoicePayload,
  type Job,
  type JobEvent,
  type QuestionResult,
  type ReportResult,
  type RunReport,
} from "@/lib/api";

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

export const BASE = "ep_0000000a";
export const FAST = "ep_0000000b";

export function endpoint(id: string, name: string, extra: Partial<Endpoint> = {}): Endpoint {
  return {
    endpoint_id: id,
    name,
    base_url: `https://acct--${name}.modal.run/v1`,
    platform: "modal",
    model_id: "Qwen/Qwen2.5-7B-Instruct",
    max_model_len: 8192,
    meta: {},
    is_baseline: false,
    status: "ok",
    status_message: null,
    checked_at: "2026-11-12T00:00:00Z",
    created_at: "2026-11-12T00:00:00Z",
    has_key: true,
    ...extra,
  };
}

export const ENDPOINTS: Endpoint[] = [
  endpoint(BASE, "vanilla", { is_baseline: true }),
  endpoint(FAST, "adaptive", { meta: { gpu: "L4", draft_method: "eagle", adaptive: true } }),
];

export function run(id: string, name: string, extra: Partial<RunReport> = {}): RunReport {
  const baseline = id === BASE;
  return {
    endpoint_id: id,
    name,
    model: "Qwen/Qwen2.5-7B-Instruct",
    valid: true,
    is_baseline: baseline,
    latency_ms: 12000,
    ttft_ms: 300,
    tpot_ms: baseline ? 20 : 14,
    output_tokens_per_s: 100,
    completion_tokens: 900,
    error: null,
    speedup_tpot_vs_baseline: baseline ? null : 1.43,
    speedup_latency_vs_baseline: baseline ? null : 1.2,
    equal_fields: baseline ? null : 40,
    total_fields: baseline ? null : 40,
    differing_fields: [],
    payload: INVOICE,
    ...extra,
  };
}

// The live log of a finished job, as the backend writes it (design D13); times are server epoch seconds.
export function jobEvents(): JobEvent[] {
  return [
    { t: 100, src: "system", level: "info", msg: "Nhận ảnh JPG (1 KB); chạy Node 1 trên 2 endpoint" },
    { t: 100.2, src: "ocr", level: "info", msg: "Gửi ảnh tới máy OCR (PaddleOCR tách dòng, VietOCR đọc chữ)" },
    { t: 109.2, src: "ocr", level: "ok", msg: "Đọc xong 3 dòng chữ trong 9,0 s; đã bỏ dấu tiếng Việt" },
    { t: 109.3, src: FAST, level: "info", msg: "Gửi yêu cầu Node 1 (prompt node1_extractor.v2, trả về tối đa 3.072 token)" },
    { t: 109.3, src: BASE, level: "info", msg: "Gửi yêu cầu Node 1 (prompt node1_extractor.v2, trả về tối đa 3.072 token)" },
    { t: 109.6, src: FAST, level: "ok", msg: "Token đầu tiên sau 0,3 s, endpoint đang trả lời", data: { ttft_s: 0.3 } },
    { t: 121.4, src: FAST, level: "ok", msg: "Xong: 900 token, TPOT 14 ms, JSON hợp lệ", data: { tokens: 900, tpot_ms: 14 } },
    { t: 122.2, src: BASE, level: "ok", msg: "Xong: 900 token, TPOT 20 ms, JSON hợp lệ", data: { tokens: 900, tpot_ms: 20 } },
    { t: 122.3, src: "system", level: "ok", msg: "adaptive so với mốc vanilla: 40/40 trường giống, nhanh hơn 1,43× theo TPOT" },
    { t: 122.4, src: "system", level: "info", msg: "Lưu hoá đơn vào cơ sở dữ liệu từ adaptive" },
    { t: 122.6, src: "system", level: "ok", msg: "Đã lưu hoá đơn, mã up_0123456789ab:up_0123456789ab" },
    { t: 122.7, src: "system", level: "ok", msg: "Hoàn tất sau 22,7 s" },
  ];
}

export function doneJob(extra: Partial<Job["result"]> = {}): Job {
  return {
    job_id: "up_0123456789ab",
    status: "done",
    error: null,
    invoice_id: "up_0123456789ab:up_0123456789ab",
    server_time: 123,
    result: {
      endpoints: [
        { endpoint_id: FAST, name: "adaptive", model_id: "Qwen/Qwen2.5-7B-Instruct" },
        { endpoint_id: BASE, name: "vanilla", model_id: "Qwen/Qwen2.5-7B-Instruct" },
      ],
      events: jobEvents(),
      primary: FAST,
      baseline: BASE,
      ocr: { text: "MINIMART ANAN\nTong tien\n42,000", kind: "library", engine: "A", marks_stripped: true },
      runs: { [FAST]: run(FAST, "adaptive"), [BASE]: run(BASE, "vanilla") },
      stored_endpoint: FAST,
      stored_from_fallback: false,
      invoice: INVOICE,
      grounding: { checked_fields: 5, grounded_fields: 5, ungrounded_fields: [] },
      item_check: { lines: [true], invoice_sum: false },
      timings_ms: { ocr: 9000, node1: 13000 },
      ...extra,
    },
  };
}

export function question(extra: Partial<QuestionResult> = {}): QuestionResult {
  return {
    question: "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?",
    status: "ok",
    sql: "SELECT SUM(total_amount_value) AS tong FROM v_invoice_facts WHERE issued_date LIKE '2020-08%'",
    columns: ["tong"],
    rows: [[3912548]],
    row_count: 1,
    answer: "Tổng chi tiêu tháng 08/2020 là 3.912.548 đồng.",
    warning: null,
    error: null,
    repairs: 0,
    first_error: null,
    calls: [
      { step: "node2", endpoint: "adaptive", wall_ms: 2100, ttft_ms: 400, tpot_ms: 15, completion_tokens: 40 },
      { step: "answer", endpoint: "adaptive", wall_ms: 900 },
    ],
    endpoint: { endpoint_id: FAST, name: "adaptive" },
    ...extra,
  };
}

const q = (id: string, columns: string[], rows: unknown[][], extra: Partial<QuestionResult> = {}) => ({
  ...question({ question: id, columns, rows, row_count: rows.length, calls: [], answer: null, ...extra }),
  question_id: id,
});

// Q1-Q6 results and report of the chain mode (Node 1 -> 2 -> 3, as in M3-06).
export const CHAIN = {
  questions: [
    q("Q1", ["tong"], [[3912548]]),
    q("Q2", ["description", "total_quantity"], [["Sữa chua", 12], ["Bánh mì", 6500]]),
    q("Q3", ["seller_name", "total"], [["MINIMART ANAN", 2500000], ["Nhà sách Cẩm Phả", 75000]]),
    q("Q4", ["so_hoa_don"], [[3]]),
    q("Q5", ["issued_date", "total"], [["2020-08-10", 420000], ["2020-08-11", 980000], ["2020-08-12", 150000]]),
    q("Q6", [], [], { status: "no_data" }),
  ],
  report: { status: "ok", text: "Tháng 08/2020 có tổng chi tiêu 3.912.548 đồng trên 25 hoá đơn.", metrics: { tpot_ms: 16 } },
  number_check: { numbers: 4, ungrounded: [] },
};

// Node 3 on its own: request + fixed tables, charts chosen by the model.
export const REPORT: ReportResult = {
  request: "Báo cáo chi tiêu và dashboard của các hoá đơn gần nhất",
  scope: { limit: 10, invoice_count: 10, label: "10 hoá đơn có ngày lập gần nhất" },
  tables: [
    { key: "kpis", title: "Số tổng của phạm vi", columns: ["so_hoa_don", "tong_tien", "trung_binh", "hoa_don_co_vat"], rows: [[10, 3912548, 391255, 3]] },
    { key: "invoices", title: "Danh sách hoá đơn", columns: ["issued_date", "seller_name", "tong_tien", "so_mat_hang"], rows: [["2020-08-16", "MINIMART ANAN", 161500, 8]] },
    { key: "by_store", title: "Tổng tiền theo cửa hàng", columns: ["seller_name", "so_hoa_don", "tong_tien"], rows: [["MINIMART ANAN", 7, 2500000], ["Nhà sách Cẩm Phả", 1, 75000]] },
    { key: "by_date", title: "Tổng tiền theo ngày lập", columns: ["issued_date", "so_hoa_don", "tong_tien"], rows: [["2020-08-10", 2, 420000], ["2020-08-11", 3, 980000]] },
    { key: "top_items", title: "Mặt hàng mua nhiều nhất", columns: ["description", "tong_so_luong", "tong_thanh_tien"], rows: [["Nước suối", 6500, 39000], ["Sữa chua", 12, 78000]] },
    { key: "taxes", title: "Thuế theo loại và thuế suất", columns: ["tax_name", "tax_rate", "so_hoa_don", "tong_thue"], rows: [] },
  ],
  dashboard: {
    status: "ok",
    error: null,
    charts: [
      { table: "kpis", type: "kpi", value: null, title: "Tổng quan 10 hoá đơn" },
      { table: "by_date", type: "column", value: "tong_tien", title: "Chi tiêu theo ngày" },
      { table: "top_items", type: "table", value: null, title: "Mặt hàng nổi bật" },
    ],
  },
  report: { status: "ok", text: "10 hoá đơn gần nhất có tổng chi tiêu 3.912.548 đồng.", metrics: { tpot_ms: 16 } },
  number_check: { numbers: 2, ungrounded: [] },
  timings_ms: { data: 120, dashboard: 1500, report: 6000 },
  endpoint: { endpoint_id: FAST, name: "adaptive" },
};

export function speedRun(extra: Partial<BenchRun> = {}): BenchRun {
  const cell = (id: string, name: string, load: number, tpot: number, speedup: number | null) => ({
    endpoint_id: id, name, load, n: 8, n_ok: 8, ttft_ms_p50: 350, tpot_ms_p50: tpot, latency_ms_p50: 20000,
    throughput_tps: 80, errors: [], speedup_vs_baseline: speedup,
  });
  return {
    run_id: "bn_0123456789ab",
    kind: "speed",
    status: "done",
    config: { kind: "speed", endpoint_ids: [BASE, FAST], workload: "W1", loads: [1, 8], requests: 8, repeats: 1 },
    progress: { done: 32, total: 32 },
    result: {
      kind: "speed", live: true, workload: "W1", loads: [1, 8], requests: 8, repeats: 1, baseline: BASE,
      cells: [cell(BASE, "vanilla", 1, 20, 1), cell(FAST, "adaptive", 1, 14, 1.43), cell(BASE, "vanilla", 8, 22, 1), cell(FAST, "adaptive", 8, 25, 0.88)],
    },
    error: null,
    created_at: "2026-11-12T03:00:00Z",
    ...extra,
  };
}

export function fakeApi(overrides: Partial<Record<keyof Api, unknown>> = {}): Api {
  const api: Api = {
    status: vi.fn(async () => ({ backend: "ready", database: "ready", endpoints: { total: 2, ok: 2, down: 0 } })),
    endpoints: vi.fn(async () => ({ endpoints: ENDPOINTS })),
    detectEndpoints: vi.fn(async () => ({ results: [] })),
    addEndpoints: vi.fn(async () => ({ results: [], saved: [] })),
    updateEndpoint: vi.fn(async (id: string) => ENDPOINTS.find((e) => e.endpoint_id === id)!),
    deleteEndpoint: vi.fn(async (id: string) => ({ deleted: id })),
    checkEndpoint: vi.fn(async (id: string) => ENDPOINTS.find((e) => e.endpoint_id === id)!),
    upload: vi.fn(async () => ({ job_id: "up_0123456789ab", endpoint_ids: [FAST, BASE] })),
    job: vi.fn(async () => doneJob()),
    invoices: vi.fn(async () => ({ invoices: [] })),
    invoice: vi.fn(async () => ({
      invoice_id: "x", payload: INVOICE, source: "upload" as const, created_at: "2026-11-12", latency_ms: 12000,
      lane: "adaptive", model_name: "Qwen/Qwen2.5-7B-Instruct", prompt_version: "node1_extractor.v2", image_url: null,
    })),
    ask: vi.fn(async () => question()),
    report: vi.fn(async () => REPORT),
    startBench: vi.fn(async () => ({ run_id: "bn_0123456789ab" })),
    bench: vi.fn(async () => speedRun()),
    benches: vi.fn(async () => ({ runs: [] })),
  };
  return { ...api, ...overrides } as Api;
}

export function notScripted(): never {
  throw new ApiError(500, "not scripted");
}

export function jpegFile(name = "hoadon.jpg", type = "image/jpeg"): File {
  return new File([new Uint8Array([0xff, 0xd8, 0xff, 0xe0])], name, { type });
}
