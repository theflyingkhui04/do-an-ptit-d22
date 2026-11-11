// Client of the FastAPI backend (src/api). The browser only knows the backend URL and the
// demo key the user typed; Modal and Supabase credentials never reach the frontend.

export type ServiceStateName = "cold" | "starting" | "ready" | "error";
export interface ServiceState {
  state: ServiceStateName;
  message: string | null;
}
export interface Status {
  backend: string;
  database: string;
  services: Record<string, ServiceState>;
  gpu: string;
  lane_map: Record<string, string>;
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
export interface InvoiceTax {
  tax_name: string | null;
  tax_rate: number | null;
  taxable_amount: Money;
  tax_amount: Money;
}
export interface InvoicePayload {
  document_type: string;
  seller: Record<string, string | null>;
  buyer: Record<string, string | null>;
  invoice: Record<string, string | null>;
  totals: Record<string, Money>;
  items: InvoiceItem[];
  taxes: InvoiceTax[];
  extraction: { status: string; confidence: number | null; missing_fields: string[]; warnings: string[] };
}

export interface LaneReport {
  lane: string;
  model: string;
  valid: boolean;
  latency_ms?: number | null;
  ttft_ms?: number | null;
  tpot_ms?: number | null;
  output_tokens_per_s?: number | null;
  completion_tokens?: number | null;
  error: string | null;
  speedup_tpot_vs_vanilla: number | null;
  speedup_latency_vs_vanilla: number | null;
  equal_fields: number | null;
  total_fields: number | null;
  differing_fields: string[];
}

export interface JobResult {
  gpu?: string;
  // Library OCR (canonical flow): text without Vietnamese marks.
  ocr?: { text: string; kind?: string; engine?: string; marks_stripped?: boolean; latency_ms?: number };
  lanes?: Record<string, LaneReport>;
  stored_lane?: string;
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

export interface QuestionResult {
  question: string;
  status: "ok" | "no_data" | "error";
  sql: string | null;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  answer: string | null;
  warning: string | null;
  error: string | null;
  repairs: number;
  calls: { step: string; lane: string; latency_ms?: number; tpot_ms?: number | null }[];
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

export interface InvoiceDetail {
  invoice_id: string;
  payload: InvoicePayload;
  source: "devset" | "upload";
  image_url: string | null;
  lane: string | null;
}

export interface Api {
  status(): Promise<Status>;
  warmup(): Promise<{ services: Record<string, ServiceState> }>;
  upload(file: File, compareAll: boolean): Promise<{ job_id: string }>;
  job(jobId: string): Promise<Job>;
  invoices(): Promise<{ invoices: InvoiceRow[] }>;
  invoice(invoiceId: string): Promise<InvoiceDetail>;
  ask(question: string): Promise<QuestionResult>;
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

  return {
    status: () => call<Status>("/api/status", {}, false),
    warmup: () => call("/api/warmup", { method: "POST" }),
    upload: (file, compareAll) => {
      const form = new FormData();
      form.append("file", file);
      form.append("compare", compareAll ? "all" : "vanilla");
      return call("/api/invoices", { method: "POST", body: form });
    },
    job: (jobId) => call<Job>(`/api/jobs/${encodeURIComponent(jobId)}`),
    invoices: () => call("/api/invoices"),
    invoice: (invoiceId) => call<InvoiceDetail>(`/api/invoices/${encodeURIComponent(invoiceId)}`),
    ask: (question) =>
      call<QuestionResult>("/api/questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }),
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
