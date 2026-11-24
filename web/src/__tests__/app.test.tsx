import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import Console from "@/components/Console";
import { ApiError, type Api, type BenchRun, type DetectResult, type Endpoint, type Job, type ReportResult } from "@/lib/api";
import { leafFields } from "@/lib/fields";

import { BASE, CHAIN, ENDPOINTS, FAST, INVOICE, REPORT, doneJob, endpoint, fakeApi, jobEvents, jpegFile, question, speedRun } from "./fakes";

function open(page: string, api: Api) {
  window.sessionStorage.setItem("ptit-demo-key", "good-key-123");
  window.history.replaceState(null, "", page ? `#${page}` : "/");
  render(<Console makeApi={() => api} pollMs={0} />);
}

describe("demo key gate and language switcher", () => {
  it("shows an error for a wrong key and opens the console for a good one", async () => {
    window.history.replaceState(null, "", "/");
    const endpoints = vi.fn(async () => {
      throw new ApiError(401, "Invalid key");
    });
    const user = userEvent.setup();
    const { unmount } = render(<Console makeApi={() => fakeApi({ endpoints })} />);
    await user.type(await screen.findByLabelText("Mật khẩu truy cập"), "wrong-key");
    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Mật khẩu không đúng.");
    unmount();

    render(<Console makeApi={() => fakeApi()} />);
    await user.type(await screen.findByLabelText("Mật khẩu truy cập"), "good-key-123");
    await user.click(screen.getByRole("button", { name: "Đăng nhập" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Trích xuất hóa đơn" })).toBeInTheDocument();
    expect(window.sessionStorage.getItem("ptit-demo-key")).toBe("good-key-123");
  });

  it("opens straight on the console with a preset key and hides sign out", async () => {
    window.sessionStorage.clear();
    window.history.replaceState(null, "", "/");
    let sent: (() => string | null) | null = null;
    render(<Console makeApi={(getKey) => { sent = getKey; return fakeApi(); }} presetKey="preset-key-1" />);
    expect(await screen.findByRole("heading", { level: 1, name: "Trích xuất hóa đơn" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Mật khẩu truy cập")).toBeNull();
    expect(sent!()).toBe("preset-key-1");
    expect(screen.queryByRole("button", { name: "Đăng xuất" })).toBeNull();
  });

  it("falls back to the key prompt when the preset key is refused", async () => {
    window.sessionStorage.clear();
    window.history.replaceState(null, "", "/");
    const endpoints = vi.fn(async () => {
      throw new ApiError(401, "Unauthorized");
    });
    render(<Console makeApi={() => fakeApi({ endpoints })} presetKey="old-key" />);
    expect(await screen.findByLabelText("Mật khẩu truy cập")).toBeInTheDocument();
  });

  it("switches between Vietnamese and English via the header toggle and persists the choice", async () => {
    open("result", fakeApi());
    const user = userEvent.setup();
    expect(await screen.findByRole("heading", { level: 1, name: "Đo hiệu năng" })).toBeInTheDocument();
    const switcher = screen.getByRole("group", { name: "Ngôn ngữ giao diện" });
    await user.click(within(switcher).getByRole("button", { name: "en" }));
    expect(screen.getByRole("heading", { level: 1, name: "Benchmark" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add endpoint" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run benchmark" })).toBeInTheDocument();
    expect(window.localStorage.getItem("flexsds-lang")).toBe("en");
    expect(document.documentElement.lang).toBe("en");

    await user.click(screen.getByRole("button", { name: "vi" }));
    expect(screen.getByRole("heading", { level: 1, name: "Đo hiệu năng" })).toBeInTheDocument();
    expect(window.localStorage.getItem("flexsds-lang")).toBe("vi");
  });
});

describe("brand", () => {
  const logo = (variant: "light" | "dark") => document.querySelector(`img[src*="flexsds-icon-${variant}"]`);

  it("uses the white tile on the key card and sidebar", async () => {
    window.sessionStorage.clear();
    window.history.replaceState(null, "", "/");
    const { unmount } = render(<Console makeApi={() => fakeApi()} />);
    await screen.findByLabelText("Mật khẩu truy cập");
    expect(logo("light")).toHaveAttribute("alt", "");
    expect(logo("light")).toHaveClass("ring-1");
    unmount();

    open("", fakeApi());
    await screen.findByRole("navigation", { name: "Điều hướng" });
    expect(logo("light")).toHaveAttribute("alt", "");
    expect(logo("dark")).toBeNull();
  });

  it("ships small icons", () => {
    for (const file of ["public/brand/flexsds-icon-light.png",
      "src/app/icon.png", "src/app/apple-icon.png", "src/app/favicon.ico"]) {
      expect(statSync(join(process.cwd(), file)).size).toBeLessThan(20_000);
    }
  });
});

describe("endpoints and alias flow", () => {
  const ok = (url: string, model = "Qwen/Qwen2.5-7B-Instruct"): DetectResult =>
    ({ input: url, url, ok: true, code: null, message: null, platform: "modal", model_id: model, max_model_len: 8192 });
  const refused = (url: string, code: string, message: string): DetectResult =>
    ({ input: url, url, ok: false, code, message, platform: "modal", model_id: null, max_model_len: null });
  const rowOf = (name: string) => screen.getByRole("article", { name });

  it("opens on the onboarding page when no endpoint has been added", async () => {
    open("", fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })) }));
    expect(await screen.findByRole("heading", { name: "Chưa có endpoint nào" })).toBeInTheDocument();
    const sample = screen.getByTestId("serve-sample");
    expect(sample).toHaveTextContent(/flexsds serve .*--target \S+ .*--draft \S+/);
    expect(sample).toHaveTextContent("Endpoint LLM created:");
    expect(screen.getByTestId("endpoint-summary")).toHaveTextContent("Chưa có endpoint");
  });

  it("lists endpoints with Chip, Draft model, Target model, and Hardware-Adaptive ON/OFF only", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [
      endpoint(FAST, "#1", {
        model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        meta: { gpu: "NVIDIA L4", draft_method: "eagle3.1", adaptive: true, draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator" },
      }),
      endpoint(BASE, "#2", {
        model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        has_key: false,
        status: "down",
        status_message: "Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động",
      }),
    ] }));
    open("endpoints", fakeApi({ endpoints }));
    const rows = await screen.findAllByRole("article");
    expect(rows.map((r) => r.getAttribute("aria-label"))).toEqual(["#1", "#2"]);
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByText(/8\.192 token/)).toBeNull();
    expect(screen.queryByText(/Modal/)).toBeNull();

    const first = rowOf("#1");
    expect(within(first).getByText("NVIDIA L4")).toBeInTheDocument();
    expect(within(first).getByText("qwen2.5-7b-eagle3.1-speculator")).toBeInTheDocument();
    expect(within(first).getByText("Qwen2.5-7B-Instruct-AWQ")).toBeInTheDocument();
    expect(within(first).getByText("ON")).toBeInTheDocument();

    const second = rowOf("#2");
    expect(within(second).getByText("OFF")).toBeInTheDocument();
    expect(within(second).getByText("Ngưng kết nối")).toBeInTheDocument();
    expect(within(second).getByTitle("GPU đang khởi động (HTTP 503)")).toBeInTheDocument();
  });

  it("suggests #1 / #2 alias when adding an endpoint and saves the custom alias", async () => {
    const modal = "https://acct--flexsds-serve.modal.run/v1";
    const local = "http://127.0.0.1:8000/v1";
    const found: Record<string, DetectResult> = {
      [modal]: ok(modal),
      [local]: { ...refused(local, "needs_tunnel", "Run `cloudflared tunnel --url http://127.0.0.1:8000`"), platform: "local" },
    };
    const detectEndpoints = vi.fn(async (urls: string[]) => ({ results: urls.map((u) => found[u]) }));
    const addEndpoints = vi.fn(async () => ({ results: [ok(modal)], saved: [endpoint("ep_0000000c", "#1")] }));
    const api = fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })), detectEndpoints, addEndpoints });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
    const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
    const address = within(drawer).getByLabelText("Địa chỉ API");

    await user.type(address, local);
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(drawer).findByText("Cần địa chỉ công khai (Tunnel)")).toBeInTheDocument();
    expect(within(drawer).getByText("cloudflared tunnel --url http://127.0.0.1:8000")).toBeInTheDocument();

    await user.clear(address);
    await user.type(address, `- API: ${modal}{Enter}`);
    expect(detectEndpoints).toHaveBeenLastCalledWith([modal], undefined);
    const aliasInput = await within(drawer).findByLabelText(`Alias cho ${modal}`);
    expect(aliasInput).toHaveValue("#1");
    await user.clear(aliasInput);
    await user.type(aliasInput, "#1 Adaptive");
    await user.click(within(drawer).getByRole("button", { name: "Thêm endpoint" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(addEndpoints).toHaveBeenCalledWith([{ url: modal, name: "#1 Adaptive", api_key: undefined }]);
  });

  it("handles API key prompt and falls back to suggested alias if left untouched", async () => {
    const url = "https://llm.example.com/v1";
    const detectEndpoints = vi.fn(async (_urls: string[], key?: string) => ({
      results: [key === "k-1" ? ok(url, "meta/llama") : refused(url, "auth_failed", "HTTP 401")],
    }));
    const addEndpoints = vi.fn(async () => ({ results: [ok(url, "meta/llama")], saved: [endpoint("ep_0000000c", "#1")] }));
    open("endpoints", fakeApi({ detectEndpoints, addEndpoints }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
    const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
    await user.type(within(drawer).getByLabelText("Địa chỉ API"), url);
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(drawer).findByText("Endpoint yêu cầu API key (HTTP 401)")).toBeInTheDocument();
    await user.type(within(drawer).getByLabelText(`API key cho ${url}`), "k-1");
    await user.click(within(drawer).getByRole("button", { name: "Thử lại với key" }));
    expect(await within(drawer).findByLabelText(`Alias cho ${url}`)).toHaveValue("#1");
    await user.click(within(drawer).getByRole("button", { name: "Thêm endpoint" }));
    expect(addEndpoints).toHaveBeenCalledWith([{ url, name: "#1", api_key: "k-1" }]);
  });

  it("edits alias, URL, key and notes without baseline checkboxes or verbose hints", async () => {
    const updateEndpoint = vi.fn(async () => ENDPOINTS[1]);
    const api = fakeApi({ updateEndpoint });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Sửa adaptive" }));
    const drawer = screen.getByRole("dialog", { name: "Chỉnh sửa endpoint" });
    expect(within(drawer).queryByText(/Model gốc/)).toBeNull();
    expect(within(drawer).queryByText(/endpoint mốc/i)).toBeNull();
    expect(within(drawer).getByLabelText("Alias")).toHaveFocus();

    await user.clear(within(drawer).getByLabelText("Alias"));
    await user.type(within(drawer).getByLabelText("Alias"), "#2 L4");
    await user.clear(within(drawer).getByLabelText("Chip"));
    await user.type(within(drawer).getByLabelText("Chip"), "NVIDIA A10G");
    const decodingSelect = within(drawer).getByLabelText("Phương pháp giải mã");
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-1 (K cố định)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-1 (Hardware-Adaptive)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-2 (K cố định)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-2 (Hardware-Adaptive)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-3 (K cố định)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-3 (Hardware-Adaptive)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-3.1 (K cố định)" })).toBeInTheDocument();
    expect(within(decodingSelect).getByRole("option", { name: "EAGLE-3.1 (Hardware-Adaptive)" })).toBeInTheDocument();
    expect(within(decodingSelect).queryByRole("option", { name: /độc lập/i })).toBeNull();
    await user.selectOptions(decodingSelect, "EAGLE-3.1 (Hardware-Adaptive)");
    await user.click(within(drawer).getByRole("button", { name: "Lưu thay đổi" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(updateEndpoint).toHaveBeenCalledWith(FAST, {
      name: "#2 L4",
      meta: { gpu: "NVIDIA A10G", draft_method: "eagle3.1" },
    });
  });

  it("keeps polling a waking GPU until it is online", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const waking: Endpoint = { ...ENDPOINTS[0], status: "starting", status_message: "HTTP 503" };
      const answers: Endpoint[] = [waking, { ...ENDPOINTS[0], checked_at: new Date().toISOString() }];
      const checkEndpoint = vi.fn(async () => answers.shift() ?? waking);
      open("endpoints", fakeApi({ checkEndpoint }));
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
      const row = rowOf("vanilla");
      expect(await within(row).findByText(/Đang khởi động GPU/)).toBeInTheDocument();
      await act(async () => {
        vi.advanceTimersByTime(20_000);
      });
      expect(await within(row).findByText("Đang chạy")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("confirms before deleting an endpoint", async () => {
    const api = fakeApi();
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Xóa vanilla" }));
    const confirm = screen.getByRole("alertdialog", { name: "Xóa endpoint vanilla?" });
    await user.click(within(confirm).getByRole("button", { name: "Hủy" }));
    expect(api.deleteEndpoint).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Xóa adaptive" }));
    await user.click(within(screen.getByRole("alertdialog", { name: "Xóa endpoint adaptive?" })).getByRole("button", { name: "Xóa endpoint" }));
    await waitFor(() => expect(api.deleteEndpoint).toHaveBeenCalledWith(FAST));
  });
});

describe("node 1", () => {
  async function runNode1(job: ReturnType<typeof vi.fn>) {
    const api = fakeApi({ job });
    open("node1", api);
    const user = userEvent.setup();
    await user.upload(await screen.findByLabelText("Ảnh hóa đơn"), jpegFile());
    await user.click(screen.getByRole("button", { name: "Bắt đầu" }));
    return { api, user };
  }

  it("runs selected endpoints freely, compares against a switchable reference, and opens logs", async () => {
    const queued: Job = { ...doneJob(), status: "node1", invoice_id: null, result: {} };
    const job = vi.fn().mockResolvedValueOnce(queued).mockResolvedValueOnce(doneJob());
    const { api, user } = await runNode1(job);
    const table = await screen.findByRole("table", { name: "Bảng kết quả trích xuất" });
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), [BASE, FAST]);

    const refSelect = screen.getByLabelText("Đối chiếu với");
    expect(refSelect).toHaveValue(BASE);
    const conclusion = screen.getByTestId(`conclusion-${FAST}`);
    expect(conclusion).toHaveTextContent("1,43×");

    // Switching reference endpoint flips comparison direction dynamically
    await user.selectOptions(refSelect, FAST);
    expect(screen.getByTestId(`conclusion-${BASE}`)).toHaveTextContent("0,7×");

    // Expand all fields and open log drawer
    const total = leafFields(INVOICE).length;
    await user.click(screen.getByRole("button", { name: `Xem tất cả ${total} trường` }));
    expect(within(table).getAllByRole("rowheader")).toHaveLength(total);
    await user.click(screen.getByRole("button", { name: "Nhật ký" }));
    expect(screen.getByRole("dialog", { name: "Nhật ký xử lý" })).toBeInTheDocument();
  });

  it("flags silent endpoints without showing informal 'Sống' status", async () => {
    const running: Job = {
      ...doneJob(), status: "node1", invoice_id: null, server_time: 200,
      result: { ...doneJob().result, runs: undefined, invoice: undefined, stored_endpoint: undefined,
        events: jobEvents().slice(0, 5).map((e) => ({ ...e, t: e.t <= 109.2 ? e.t : 110 })) },
    };
    const { user } = await runNode1(vi.fn(async () => running));
    const fast = await screen.findByTestId(`run-${FAST}`);
    expect(fast).toHaveTextContent("Không có phản hồi (1 phút 30 s)");
    await user.click(screen.getByRole("button", { name: "Nhật ký" }));
    expect(screen.queryByText("Sống")).toBeNull();
    expect(screen.getByText("Đang hoạt động")).toBeInTheDocument();
  });
});

describe("node 2", () => {
  const timed = (endpointId: string, extra: Parameters<typeof question>[0] = {}) => question({
    calls: [
      { step: "node2", endpoint: endpointId, wall_ms: endpointId === BASE ? 2600 : 2100, ttft_ms: 400, tpot_ms: endpointId === BASE ? 20 : 15, completion_tokens: 40 },
      { step: "answer", endpoint: endpointId, wall_ms: 900 },
    ],
    endpoint: { endpoint_id: endpointId, name: endpointId === BASE ? "vanilla" : "adaptive" },
    ...extra,
  });

  it("allows toggling any endpoint freely and compares against the chosen reference", async () => {
    const ask = vi.fn(async (_q: string, endpointId: string) =>
      endpointId === BASE ? timed(endpointId) : timed(endpointId, { rows: [[3912000]], answer: "Tổng chi tiêu là 3.912.000 đồng." }));
    open("node2", fakeApi({ ask }));
    const user = userEvent.setup();
    // Both endpoints are selectable (none locked)
    expect(await screen.findByRole("checkbox", { name: "vanilla" })).not.toBeDisabled();
    await user.click(screen.getByRole("button", { name: /^Q1/ }));
    await screen.findByText("Tổng chi tiêu là 3.912.000 đồng.");

    const compare = screen.getByRole("table", { name: "So sánh với vanilla" });
    expect(compare).toHaveTextContent("Khác biệt");
    expect(compare).toHaveTextContent("Nhanh hơn 1,33×");

    // Switch comparison reference to adaptive
    await user.selectOptions(screen.getByLabelText("Đối chiếu với"), FAST);
    expect(screen.getByRole("table", { name: "So sánh với adaptive" })).toHaveTextContent("Chậm hơn 1,33×");
  });

  it("opens data schema drawer", async () => {
    open("node2", fakeApi());
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Lược đồ dữ liệu" }));
    const drawer = screen.getByRole("dialog", { name: "Lược đồ dữ liệu" });
    expect(drawer).toHaveTextContent("v_invoice_facts");
  });
});

describe("node 3", () => {
  const written = (endpointId: string, extra: Partial<ReportResult> = {}): ReportResult => ({
    ...REPORT,
    report: { ...REPORT.report, metrics: { ttft_ms: 600, tpot_ms: endpointId === BASE ? 20 : 16, completion_tokens: 300 } },
    timings_ms: endpointId === BASE ? { data: 120, dashboard: 1900, report: 7800 } : { data: 120, dashboard: 1500, report: 6000 },
    endpoint: { endpoint_id: endpointId, name: endpointId === BASE ? "vanilla" : "adaptive" },
    ...extra,
  });

  it("generates reports and dashboard without redundant chart type suffixes", async () => {
    const report = vi.fn(async (endpointId: string) => written(endpointId));
    open("node3", fakeApi({ report }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Chi tiêu gần đây" }));
    const totals = await screen.findByLabelText("Tổng hợp số liệu");
    expect(totals).toHaveTextContent("3.912.548 VND");
    expect(screen.getByRole("table", { name: "So sánh với vanilla" })).toHaveTextContent("Số liệu khớp");
    expect(screen.queryByText("biểu đồ cột")).toBeNull();
    expect(screen.queryByText("Số đều khớp")).toBeNull();
  });
});

describe("benchmark page", () => {
  const chainRun = (): BenchRun => ({
    run_id: "bn_0000000000ff",
    kind: "chain",
    status: "done",
    config: { kind: "chain", endpoint_ids: [BASE, FAST], sample_id: "m0_04_001" },
    progress: { done: 2, total: 2 },
    result: {
      kind: "chain", live: true, sample_id: "m0_04_001", ocr_ms: 8000, baseline: BASE,
      endpoints: [BASE, FAST].map((id) => ({
        endpoint_id: id, name: id === BASE ? "vanilla" : "adaptive",
        timings_ms: { node1: id === BASE ? 8000 : 6000, node2: 7000, node3: 4500 }, total_ms: 0,
        node1: doneJob().result.runs![id], invoice: doneJob().result.invoice!, questions: CHAIN.questions,
        report: CHAIN.report, number_check: CHAIN.number_check,
        same_as_baseline: id === BASE ? null : { node1: true, node2: true, node3: false },
      })),
    },
    error: null,
    created_at: "2026-11-12T03:00:00Z",
  });

  it("uses Task 1/2/3, dataset/single-invoice tabs, and clean result headers", async () => {
    let finish: (run: BenchRun) => void = () => undefined;
    const running: BenchRun = { ...speedRun(), status: "running", progress: { done: 8, total: 32 }, result: {} };
    const bench = vi.fn().mockResolvedValueOnce(running).mockImplementationOnce(() => new Promise<BenchRun>((resolve) => {
      finish = resolve;
    }));
    const benches = vi.fn().mockResolvedValueOnce({ runs: [] }).mockResolvedValue({ runs: [speedRun()] });
    const api = fakeApi({ bench, benches });
    open("result", api);
    const user = userEvent.setup();

    expect(await screen.findByRole("radio", { name: "Trên tập hóa đơn" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Trên một hóa đơn" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Task 1" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Task 2" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Task 3" })).toBeInTheDocument();
    expect(screen.queryByText(/Node 1 \(W1\)/)).toBeNull();
    expect(screen.getByRole("radiogroup", { name: "Chế độ đo" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Trên lý thuyết" })).toBeInTheDocument();
    expect(screen.queryByText("Chuẩn thực nghiệm")).toBeNull();
    expect(screen.queryByText("Đo trực tiếp")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Bắt đầu đo" }));
    expect(api.startBench).toHaveBeenCalledWith({ kind: "speed", endpoint_ids: [BASE, FAST], workload: "W1", loads: [1, 8], requests: 8, repeats: 1 });
    expect(await screen.findByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");

    await act(async () => finish(speedRun()));
    expect(await screen.findByRole("radio", { name: "Trên lần đo này" })).toBeInTheDocument();
    expect(screen.queryByText("Đo trực tiếp")).toBeNull();

    const heavy = screen.getByRole("group", { name: "Concurrency 8" });
    expect(heavy).toHaveTextContent("25 ms");
    expect(heavy).toHaveTextContent("Chậm hơn 1,14×");
    expect(within(heavy).queryByText("Đối chiếu")).toBeNull();
    expect(within(screen.getByRole("group", { name: "Concurrency 1" })).getByText("Nhanh hơn 1,43×")).toBeInTheDocument();

    // Switch reference endpoint dynamically
    await user.selectOptions(screen.getByLabelText("Đối chiếu với"), FAST);
    expect(within(screen.getByRole("group", { name: "Concurrency 8" })).getByText("Nhanh hơn 1,14×")).toBeInTheDocument();
  });

  it("runs single-invoice benchmark with Task 1/2/3 and no M3-06 badges", async () => {
    const api = fakeApi({ bench: vi.fn(async () => chainRun()) });
    open("result", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: "Trên một hóa đơn" }));
    expect(screen.queryByText("Chuẩn thực nghiệm")).toBeNull();
    expect(screen.queryByText(/M3-06/)).toBeNull();
    expect(screen.getByText("Task 1")).toBeInTheDocument();
    expect(screen.getByText("Task 2")).toBeInTheDocument();
    expect(screen.getByText("Task 3")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Bắt đầu đo" }));
    const quality = await screen.findByRole("table", { name: "Bảng đối chiếu chất lượng" });
    expect(quality).toHaveTextContent("6/6");
    await user.click(screen.getByRole("button", { name: "Xem chi tiết adaptive" }));
    expect(screen.getByRole("dialog", { name: "Chi tiết · adaptive" })).toBeInTheDocument();
  });
});

describe("no secret in the frontend", () => {
  it("never references backend credentials", () => {
    const forbidden = /MODAL_PROXY_TOKEN|SUPABASE_SERVICE_KEY|DEMO_DATABASE_URL|DEMO_READER_DATABASE_URL|DEMO_KEY\b/;
    const walk = (dir: string): string[] =>
      readdirSync(dir).flatMap((name) => {
        const path = join(dir, name);
        return statSync(path).isDirectory() ? walk(path) : [path];
      });
    const offenders = walk(join(__dirname, "..")).filter(
      (path) => !path.includes("__tests__") && forbidden.test(readFileSync(path, "utf-8")),
    );
    expect(offenders).toEqual([]);
  });
});
