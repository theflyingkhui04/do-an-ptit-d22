// Client of the FastAPI backend (src/api). The browser only knows the backend URL and the demo key the user
// typed. Endpoint API keys are sent once when an endpoint is added and are never returned by the backend.

export interface Status {
  backend: string;
  database: string;
  endpoints: { total: number; ok: number; down: number };
}

export type Platform = "modal" | "tunnel" | "other" | "local";

export interface EndpointMeta {
  gpu?: string;
  sm?: string;
  backend?: string;
  target_model?: string;
  base_model?: string; // vLLM ``root`` read from /v1/models
  draft_model?: string | null;
  draft_method?: string | null;
  adaptive?: boolean;
  max_model_len?: number;
  flexsds_version?: string;
}

export interface Endpoint {
  endpoint_id: string;
  name: string;
  base_url: string;
  platform: Platform;
  model_id: string;
  max_model_len: number | null;
  meta: EndpointMeta;
  is_baseline: boolean;
  status: "ok" | "down";
  status_message: string | null;
  checked_at: string;
  created_at: string;
  has_key: boolean;
}

export interface DetectResult {
  input: string;
  url: string | null;
  ok: boolean;
  code: string | null;
  message: string | null;
  platform: Platform | null;
  model_id: string | null;
  max_model_len: number | null;
}

export interface Money {
  raw: string | null;
  value: number | null;
  currency: string | null;
}
export interface InvoiceItem {
  line_number: number;
  description: string | null;
  sku: string | null;
  quantity: { raw: string | null; value: number | null; unit: string | null };
  unit_price: Money;
  line_total: Money;
  vat_rate: number | null;
}
export interface InvoicePayload {
  document_type: string;
  seller: Record<string, string | null>;
  buyer: Record<string, string | null>;
  invoice: Record<string, string | null>;
  totals: Record<string, Money>;
  items: InvoiceItem[];
  taxes: { tax_name: string | null; tax_rate: number | null; taxable_amount: Money; tax_amount: Money }[];
  extraction: { status: string; confidence: number | null; missing_fields: string[]; warnings: string[] };
}

export interface RunReport {
  endpoint_id: string;
  name: string;
  model: string;
  valid: boolean;
  is_baseline?: boolean;
  latency_ms?: number | null;
  ttft_ms?: number | null;
  tpot_ms?: number | null;
  output_tokens_per_s?: number | null;
  completion_tokens?: number | null;
  error: string | null;
  speedup_tpot_vs_baseline: number | null;
  speedup_latency_vs_baseline: number | null;
  equal_fields: number | null;
  total_fields: number | null;
  differing_fields: string[];
}

export interface JobResult {
  endpoints?: { endpoint_id: string; name: string; model_id: string }[];
  primary?: string;
  baseline?: string | null;
  ocr?: { text: string; kind?: string; engine?: string; marks_stripped?: boolean; latency_ms?: number };
  runs?: Record<string, RunReport>;
  stored_endpoint?: string;
  stored_from_fallback?: boolean;
  invoice?: InvoicePayload;
  grounding?: { checked_fields: number; grounded_fields: number; ungrounded_fields: string[] };
  item_check?: { lines: (boolean | null)[]; invoice_sum: boolean | null };
  invoice_id?: string;
  timings_ms?: Record<string, number>;
}

export type JobStatus = "queued" | "ocr" | "node1" | "saving" | "done" | "failed" | "interrupted";
export const FINAL_JOB_STATUSES: JobStatus[] = ["done", "failed", "interrupted"];
export interface Job {
  job_id: string;
  status: JobStatus;
  error: string | null;
  result: JobResult;
  invoice_id: string | null;
}

export interface CallInfo {
  step: string;
  endpoint: string;
  wall_ms?: number;
  latency_ms?: number;
  ttft_ms?: number | null;
  tpot_ms?: number | null;
  completion_tokens?: number | null;
}

export interface QuestionResult {
  question: string;
  question_id?: string;
  status: "ok" | "no_data" | "error";
  sql: string | null;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  answer: string | null;
  warning: string | null;
  error: string | null;
  repairs: number;
  first_error: string | null;
  calls: CallInfo[];
  endpoint?: { endpoint_id: string; name: string };
}

export interface CallMetrics {
  ttft_ms?: number | null;
  tpot_ms?: number | null;
  output_tokens_per_s?: number | null;
  completion_tokens?: number | null;
}

export interface Node3Report {
  status: string;
  text: string;
  error?: string | null;
  latency_ms?: number | null;
  completion_tokens?: number | null;
  metrics?: CallMetrics;
}

// One table of the fixed Node 3 queries (src/api/report_data.py); every number on the dashboard comes from these rows.
export interface DataTable {
  key: string;
  title: string;
  columns: string[];
  rows: unknown[][];
}

export interface ChartSpec {
  table: string;
  type: "kpi" | "bar" | "column" | "table";
  value: string | null;
  title: string;
}

export interface ReportResult {
  request: string;
  scope: { limit: number | null; invoice_count: number; label: string };
  tables: DataTable[];
  dashboard: { status: "ok" | "fallback" | "skipped"; charts: ChartSpec[]; error: string | null; metrics?: CallMetrics };
  report: Node3Report;
  number_check: { numbers: number; ungrounded: string[] };
  timings_ms: { data: number; dashboard: number; report: number };
  endpoint: { endpoint_id: string; name: string };
}

export interface InvoiceDetail {
  invoice_id: string;
  payload: InvoicePayload;
  source: "devset" | "upload";
  created_at: string;
  latency_ms: number | null;
  lane: string | null;
  model_name: string | null;
  prompt_version: string | null;
  image_url: string | null;
}

export interface InvoiceRow {
  invoice_id: string;
  seller_name: string | null;
  issued_date: string | null;
  total_amount_value: number | null;
  total_amount_currency: string | null;
  item_count: number;
  source: "devset" | "upload";
  created_at: string;
  lane: string | null;
}

export interface BenchCell {
  endpoint_id: string;
  name: string;
  load: number;
  n: number;
  n_ok: number;
  ttft_ms_p50: number | null;
  tpot_ms_p50: number | null;
  latency_ms_p50: number | null;
  throughput_tps: number | null;
  errors: string[];
  speedup_vs_baseline: number | null;
}

export interface SpeedResult {
  kind: "speed";
  live: boolean;
  workload: string;
  loads: number[];
  requests: number;
  repeats: number;
  baseline: string | null;
  cells: BenchCell[];
}

export interface ChainRow {
  endpoint_id: string;
  name: string;
  timings_ms: { node1: number; node2: number; node3: number };
  total_ms: number;
  node1: RunReport;
  invoice: InvoicePayload | null;
  questions: (QuestionResult & { question_id: string })[];
  report: Node3Report;
  number_check: { numbers: number; ungrounded: string[] };
  same_as_baseline: { node1: boolean; node2: boolean; node3: boolean } | null;
}

export interface ChainResult {
  kind: "chain";
  live: boolean;
  sample_id: string;
  ocr_ms: number | null;
  baseline: string | null;
  endpoints: ChainRow[];
}

export type BenchStatus = "queued" | "running" | "done" | "failed" | "interrupted";

export interface BenchConfig {
  kind: "speed" | "chain";
  endpoint_ids: string[];
  workload?: "W1" | "W2" | "W3";
  loads?: number[];
  requests?: number;
  repeats?: number;
  sample_id?: string;
}

export interface BenchRun {
  run_id: string;
  kind: "speed" | "chain";
  status: BenchStatus;
  config: BenchConfig;
  progress: { done?: number; total?: number };
  result: SpeedResult | ChainResult | Record<string, never>;
  error: string | null;
  created_at: string;
}

export interface Api {
  status(): Promise<Status>;
  endpoints(): Promise<{ endpoints: Endpoint[] }>;
  detectEndpoints(urls: string[], apiKey?: string): Promise<{ results: DetectResult[] }>;
  addEndpoints(
    urls: string[],
    apiKey?: string,
    servingConfig?: unknown,
  ): Promise<{ results: DetectResult[]; saved: Endpoint[] }>;
  updateEndpoint(endpointId: string, change: { name?: string; is_baseline?: boolean }): Promise<Endpoint>;
  deleteEndpoint(endpointId: string): Promise<{ deleted: string }>;
  checkEndpoint(endpointId: string): Promise<Endpoint>;
  upload(file: File, endpointIds: string[]): Promise<{ job_id: string; endpoint_ids: string[] }>;
  job(jobId: string): Promise<Job>;
  invoices(): Promise<{ invoices: InvoiceRow[] }>;
  invoice(invoiceId: string): Promise<InvoiceDetail>;
  ask(question: string, endpointId: string): Promise<QuestionResult>;
  report(endpointId: string, request: string, limit: number | null): Promise<ReportResult>;
  startBench(config: BenchConfig): Promise<{ run_id: string }>;
  bench(runId: string): Promise<BenchRun>;
  benches(): Promise<{ runs: BenchRun[] }>;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const KEY_STORAGE = "ptit-demo-key";

export function readKey(): string | null {
  try {
    return window.sessionStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

export function saveKey(key: string | null): void {
  try {
    if (key) window.sessionStorage.setItem(KEY_STORAGE, key);
    else window.sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // private mode: the key lives only in memory for this page
  }
}

export function createApi(baseUrl: string, getKey: () => string | null): Api {
  const base = baseUrl.replace(/\/+$/, "");

  async function call<T>(path: string, init: RequestInit = {}, needsKey = true): Promise<T> {
    const headers = new Headers(init.headers);
    const key = getKey();
    if (needsKey && key) headers.set("X-Demo-Key", key);
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, { ...init, headers });
    } catch {
      throw new ApiError(0, "Không kết nối được máy chủ");
    }
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (typeof body?.detail === "string") detail = body.detail;
      } catch {
        // keep the HTTP status text
      }
      throw new ApiError(response.status, detail);
    }
    return (await response.json()) as T;
  }

  const json = (method: string, body: unknown): RequestInit => ({
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const id = encodeURIComponent;

  return {
    status: () => call<Status>("/api/status", {}, false),
    endpoints: () => call("/api/endpoints"),
    detectEndpoints: (urls, apiKey) => call("/api/endpoints/detect", json("POST", { urls, api_key: apiKey || null })),
    addEndpoints: (urls, apiKey, servingConfig) =>
      call("/api/endpoints", json("POST", { urls, api_key: apiKey || null, serving_config: servingConfig ?? null })),
    updateEndpoint: (endpointId, change) => call(`/api/endpoints/${id(endpointId)}`, json("PATCH", change)),
    deleteEndpoint: (endpointId) => call(`/api/endpoints/${id(endpointId)}`, { method: "DELETE" }),
    checkEndpoint: (endpointId) => call(`/api/endpoints/${id(endpointId)}/check`, { method: "POST" }),
    upload: (file, endpointIds) => {
      const form = new FormData();
      form.append("file", file);
      form.append("endpoint_ids", endpointIds.join(","));
      return call("/api/invoices", { method: "POST", body: form });
    },
    job: (jobId) => call<Job>(`/api/jobs/${id(jobId)}`),
    invoices: () => call("/api/invoices"),
    invoice: (invoiceId) => call<InvoiceDetail>(`/api/invoices/${id(invoiceId)}`),
    ask: (question, endpointId) => call("/api/questions", json("POST", { question, endpoint_id: endpointId })),
    report: (endpointId, request, limit) =>
      call("/api/reports", json("POST", { endpoint_id: endpointId, request, limit })),
    startBench: (config) => call("/api/bench", json("POST", config)),
    bench: (runId) => call<BenchRun>(`/api/bench/${id(runId)}`),
    benches: () => call("/api/bench"),
  };
}

// Any 401 means the demo key is no longer valid: forget it and show the key prompt again.
export function withUnauthorizedHandler(api: Api, onUnauthorized: () => void): Api {
  const wrapped = {} as Api;
  for (const name of Object.keys(api) as (keyof Api)[]) {
    const method = api[name] as (...args: unknown[]) => Promise<unknown>;
    (wrapped as unknown as Record<string, unknown>)[name] = async (...args: unknown[]) => {
      try {
        return await method(...args);
      } catch (exc) {
        if (exc instanceof ApiError && exc.status === 401) onUnauthorized();
        throw exc;
      }
    };
  }
  return wrapped;
}
