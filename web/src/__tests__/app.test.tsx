import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import Console from "@/components/Console";
import { ApiError, type Api, type BenchRun, type Job } from "@/lib/api";

import { BASE, CHAIN, ENDPOINTS, FAST, REPORT, doneJob, endpoint, fakeApi, jpegFile, question, speedRun } from "./fakes";

function open(page: string, api: Api) {
  window.sessionStorage.setItem("ptit-demo-key", "good-key-123");
  window.history.replaceState(null, "", page ? `#${page}` : "/");
  render(<Console makeApi={() => api} pollMs={0} />);
}

describe("demo key gate", () => {
  it("shows 'Sai mật khẩu demo' for a wrong key and opens the console for a good one", async () => {
    window.history.replaceState(null, "", "/");
    const endpoints = vi.fn(async () => {
      throw new ApiError(401, "Sai mật khẩu demo");
    });
    const user = userEvent.setup();
    const { unmount } = render(<Console makeApi={() => fakeApi({ endpoints })} />);
    await user.type(await screen.findByLabelText("Mật khẩu demo"), "wrong-key");
    await user.click(screen.getByRole("button", { name: "Vào console" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Sai mật khẩu demo");
    unmount();

    render(<Console makeApi={() => fakeApi()} />);
    await user.type(await screen.findByLabelText("Mật khẩu demo"), "good-key-123");
    await user.click(screen.getByRole("button", { name: "Vào console" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Trích xuất hoá đơn" })).toBeInTheDocument();
    expect(window.sessionStorage.getItem("ptit-demo-key")).toBe("good-key-123");
  });

  it("returns to the key prompt when the API answers 401", async () => {
    const ask = vi.fn(async () => {
      throw new ApiError(401, "Sai mật khẩu demo");
    });
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Q1" }));
    expect(await screen.findByLabelText("Mật khẩu demo")).toBeInTheDocument();
    expect(window.sessionStorage.getItem("ptit-demo-key")).toBeNull();
  });
});

describe("endpoints", () => {
  it("opens on the onboarding page when no endpoint has been added", async () => {
    open("", fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })) }));
    expect(await screen.findByRole("heading", { name: "Chưa có endpoint nào" })).toBeInTheDocument();
    expect(screen.getByText(/flexsds serve --model/)).toBeInTheDocument();
    expect(screen.getByTestId("endpoint-summary")).toHaveTextContent("Chưa có endpoint");
  });

  it("detects pasted addresses, explains the tunnel for a local one and saves the usable ones", async () => {
    const detectEndpoints = vi.fn(async () => ({
      results: [
        { input: "https://acct--flexsds-serve.modal.run/v1", url: "https://acct--flexsds-serve.modal.run/v1", ok: true, code: null, message: null, platform: "modal" as const, model_id: "Qwen/Qwen2.5-7B-Instruct", max_model_len: 8192 },
        { input: "http://127.0.0.1:8000/v1", url: "http://127.0.0.1:8000/v1", ok: false, code: "needs_tunnel", platform: "local" as const, model_id: null, max_model_len: null,
          message: "Đây là địa chỉ trên máy của bạn; chạy lệnh `cloudflared tunnel --url http://127.0.0.1:8000` rồi dán lại." },
      ],
    }));
    const added = endpoint("ep_0000000c", "Qwen/Qwen2.5-7B-Instruct");
    const addEndpoints = vi.fn(async () => ({
      results: [{ input: added.base_url, url: added.base_url, ok: true, code: null, message: null, platform: "modal" as const, model_id: added.model_id, max_model_len: 8192 }],
      saved: [added],
    }));
    const api = fakeApi({ detectEndpoints, addEndpoints });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click((await screen.findAllByRole("button", { name: "Thêm endpoint" }))[0]);
    const dialog = screen.getByRole("dialog", { name: "Thêm endpoint" });
    await user.type(within(dialog).getByLabelText("Endpoint URL"), "https://acct--flexsds-serve.modal.run/v1\nhttp://127.0.0.1:8000/v1");
    await user.type(within(dialog).getByLabelText("Khoá API (nếu có)"), "tok.secret");
    await user.click(within(dialog).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(dialog).findByText("Cần mở đường hầm")).toBeInTheDocument();
    expect(within(dialog).getByText("cloudflared tunnel --url http://127.0.0.1:8000")).toBeInTheDocument();
    expect(within(dialog).getByText("Qwen/Qwen2.5-7B-Instruct")).toBeInTheDocument();
    expect(detectEndpoints).toHaveBeenCalledWith(["https://acct--flexsds-serve.modal.run/v1", "http://127.0.0.1:8000/v1"], "tok.secret");

    await user.click(within(dialog).getByRole("button", { name: "Thêm 1 endpoint" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(addEndpoints).toHaveBeenCalledWith(["https://acct--flexsds-serve.modal.run/v1"], "tok.secret", undefined);
    expect(api.endpoints).toHaveBeenCalledTimes(2);
  });

  it("sets the baseline, checks on demand and asks before deleting", async () => {
    const api = fakeApi();
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: "Chọn adaptive làm mốc" }));
    expect(api.updateEndpoint).toHaveBeenCalledWith(FAST, { is_baseline: true });
    await user.click(screen.getAllByRole("button", { name: "Kiểm tra lại" })[0]);
    expect(api.checkEndpoint).toHaveBeenCalledWith(BASE);
    await user.click(screen.getByRole("button", { name: "Xoá adaptive" }));
    expect(api.deleteEndpoint).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Xoá" }));
    expect(api.deleteEndpoint).toHaveBeenCalledWith(FAST);
  });

  it("warns when the endpoints serve different models", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [...ENDPOINTS, endpoint("ep_0000000c", "llama", { model_id: "meta/llama" })] }));
    open("endpoints", fakeApi({ endpoints }));
    expect(await screen.findByText(/2 model khác nhau/)).toBeInTheDocument();
  });
});

describe("node 1", () => {
  it("runs the chosen endpoint with the baseline and compares them", async () => {
    const queued: Job = { ...doneJob(), status: "node1", invoice_id: null, result: {} };
    const job = vi.fn().mockResolvedValueOnce(queued).mockResolvedValueOnce(doneJob());
    const api = fakeApi({ job });
    open("node1", api);
    const user = userEvent.setup();
    await user.upload(await screen.findByLabelText("Ảnh hoá đơn"), jpegFile());
    await user.click(screen.getByRole("button", { name: "Chạy Node 1" }));
    expect(await screen.findByTestId(`run-${FAST}`)).toHaveTextContent("Giống mốc 40/40 trường");
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), [FAST]);
    expect(job).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId(`run-${FAST}`)).toHaveTextContent("1,43×");
    expect(screen.getByRole("img", { name: /OCR 9 s, Node 1 13 s/ })).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Bảng mặt hàng" })).getByText("Sữa tươi")).toBeInTheDocument();
    expect(screen.getByText("Tổng các dòng lệch tổng hoá đơn")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Văn bản OCR" }));
    expect(screen.getByLabelText("Văn bản OCR")).toHaveTextContent("MINIMART ANAN");
    expect(api.invoices).toHaveBeenCalledTimes(2);
  });

  it("rejects a file that is not a JPEG or PNG before uploading", async () => {
    const api = fakeApi();
    open("node1", api);
    await userEvent.setup({ applyAccept: false }).upload(await screen.findByLabelText("Ảnh hoá đơn"), jpegFile("a.pdf", "application/pdf"));
    expect(await screen.findByText("Chỉ nhận ảnh JPEG hoặc PNG.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Chạy Node 1" })).toBeDisabled();
  });

  it("asks for an endpoint when none exists", async () => {
    window.sessionStorage.setItem("ptit-demo-key", "k");
    open("node1", fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })) }));
    expect(await screen.findByRole("heading", { name: "Cần ít nhất một endpoint" })).toBeInTheDocument();
  });
});

describe("node 2", () => {
  it("asks a fixed question on the chosen endpoint and shows the checked SQL and rows", async () => {
    const api = fakeApi();
    open("node2", api);
    await userEvent.setup().click(await screen.findByRole("button", { name: "Q1" }));
    expect(await screen.findByLabelText("Câu SQL")).toHaveTextContent("SUM(total_amount_value)");
    expect(api.ask).toHaveBeenCalledWith("Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?", FAST);
    expect(screen.getByText("Đã trả lời")).toBeInTheDocument();
    expect(screen.getByText("Chỉ đọc (một câu SELECT)")).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Kết quả SQL" })).getByText("3.912.548")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Sinh SQL 2,1 s/ })).toBeInTheDocument();
  });

  it("shows a blocked query with its reason", async () => {
    const ask = vi.fn(async () => question({ status: "error", error: "JOIN_FANOUT: cộng trùng hoá đơn", rows: [], columns: [], repairs: 1 }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Q3" }));
    expect(await screen.findByText("Bị chặn")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("JOIN_FANOUT");
  });
});

describe("node 3", () => {
  it("sends the request and scope, draws the charts the model chose from the tables and checks the numbers", async () => {
    const api = fakeApi();
    open("node3", api);
    expect(await screen.findByRole("heading", { name: "Chưa có báo cáo" })).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Cửa hàng nào chi nhiều nhất? So sánh các cửa hàng" }));
    await user.click(screen.getByRole("radio", { name: "25 gần nhất" }));
    await user.click(screen.getByRole("button", { name: "Tạo báo cáo" }));
    expect(await screen.findByText("3.912.548 đ")).toBeInTheDocument();
    expect(api.report).toHaveBeenCalledWith(FAST, "Cửa hàng nào chi nhiều nhất? So sánh các cửa hàng", 25);
    expect(screen.getByLabelText("Tổng quan 10 hoá đơn")).toHaveTextContent("Hoá đơn có VAT");
    expect(screen.getByRole("img", { name: "Chi tiêu theo ngày" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /theo cửa hàng/ })).not.toBeInTheDocument();
    expect(screen.getByText("nghi lỗi OCR")).toBeInTheDocument();
    expect(screen.getByText(/2 số trong báo cáo đều có trong bảng số liệu/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Lấy số liệu .*Chọn biểu đồ 1,5 s, Viết báo cáo 6 s/ })).toBeInTheDocument();
    expect(screen.getByText("Số liệu đầu vào của Node 3")).toBeInTheDocument();
  });

  it("says when the model chose no usable chart and the default set is shown", async () => {
    const report = vi.fn(async () => ({
      ...REPORT,
      dashboard: { status: "fallback" as const, error: "Model không trả JSON hợp lệ cho dashboard.", charts: [{ table: "by_store", type: "bar" as const, value: "tong_tien", title: "Tổng tiền theo cửa hàng" }] },
    }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    expect(await screen.findByText(/Đang hiện bộ biểu đồ mặc định/)).toHaveTextContent("Model không trả JSON hợp lệ");
    expect(screen.getByRole("img", { name: "Tổng tiền theo cửa hàng" })).toBeInTheDocument();
  });

  it("labels bars with enough columns to tell rows apart", async () => {
    const invoices = {
      key: "invoices", title: "Danh sách hoá đơn", columns: ["issued_date", "seller_name", "tong_tien", "so_mat_hang"],
      rows: [["2020-08-16", "MINIMART ANAN", 161500, 8], ["2020-08-16", "CUA HANG X", 52000, 2], ["2020-08-15", "MINIMART ANAN", 30000, 1]],
    };
    const report = vi.fn(async () => ({
      ...REPORT,
      tables: REPORT.tables.map((t) => (t.key === "invoices" ? invoices : t)),
      dashboard: { status: "ok" as const, error: null, charts: [
        { table: "invoices", type: "bar" as const, value: "tong_tien", title: "Từng hoá đơn" },
        { table: "by_date", type: "column" as const, value: "tong_tien", title: "Theo ngày" },
      ] },
    }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    const bars = await screen.findByRole("img", { name: "Từng hoá đơn" });
    expect(bars).toHaveTextContent("2020-08-16, MINIMART ANAN");
    expect(bars).toHaveTextContent("2020-08-16, CUA HANG X");
    const days = screen.getByRole("img", { name: "Theo ngày" });
    expect(days).toHaveTextContent("2020-08-10");
    expect(days).not.toHaveTextContent("2020-08-10,");
  });

  it("explains an empty scope instead of drawing anything", async () => {
    const report = vi.fn(async () => ({ ...REPORT, scope: { limit: 10, invoice_count: 0, label: "x" }, dashboard: { ...REPORT.dashboard, charts: [] } }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    expect(await screen.findByText(/Phạm vi này chưa có hoá đơn nào/)).toBeInTheDocument();
  });
});

describe("measure and compare", () => {
  it("starts a quick speed run, follows its progress and shows the live result", async () => {
    const running: BenchRun = { ...speedRun(), status: "running", progress: { done: 8, total: 32 }, result: {} };
    const bench = vi.fn().mockResolvedValueOnce(running).mockResolvedValueOnce(speedRun());
    const benches = vi.fn().mockResolvedValueOnce({ runs: [] }).mockResolvedValue({ runs: [speedRun()] });
    const api = fakeApi({ bench, benches });
    open("result", api);
    await userEvent.setup().click(await screen.findByRole("button", { name: "Bắt đầu đo" }));
    expect(await screen.findByRole("table", { name: "Bảng kết quả đo" })).toBeInTheDocument();
    expect(api.startBench).toHaveBeenCalledWith({ kind: "speed", endpoint_ids: [BASE, FAST], workload: "W1", loads: [1, 8], requests: 8, repeats: 1 });
    expect(screen.getByText("đo trực tiếp, có nhiễu")).toBeInTheDocument();
    expect(screen.getByText("0,88×", { selector: "td" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Nhanh gấp mấy lần/ })).toBeInTheDocument();
  });

  it("keeps the official campaign numbers apart from live runs", async () => {
    open("result", fakeApi());
    await userEvent.setup().click(await screen.findByRole("button", { name: /M3-05, GPU L4/ }));
    expect(screen.getAllByText("số chính thức").length).toBeGreaterThan(1);
    expect(screen.getByText(/Ở 32 người dùng cả ba cách giải mã suy đoán đều chậm hơn mốc/)).toBeInTheDocument();
  });

  it("runs the whole chain per endpoint and compares each node with the baseline", async () => {
    const chain: BenchRun = {
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
    };
    const api = fakeApi({ bench: vi.fn(async () => chain) });
    open("result", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("tab", { name: "Chạy cả chuỗi 1 → 2 → 3" }));
    await user.click(screen.getByRole("button", { name: "Chạy cả chuỗi" }));
    expect(await screen.findByText("giống mốc 2/3 node")).toBeInTheDocument();
    expect(api.startBench).toHaveBeenCalledWith({ kind: "chain", endpoint_ids: [BASE, FAST], sample_id: "m0_04_001" });
    expect(screen.getByRole("img", { name: /adaptive: Node 1 6 s, Node 2 7 s, Node 3 4,5 s/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Xem adaptive" }));
    expect(screen.getByLabelText("JSON hoá đơn")).toHaveTextContent("MINIMART ANAN");
  });

  it("reports that another run is in progress", async () => {
    const startBench = vi.fn(async () => {
      throw new ApiError(409, "Đang có một lần đo chạy, đợi lần đó xong.");
    });
    open("result", fakeApi({ startBench }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Bắt đầu đo" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Đang có một lần đo chạy");
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
