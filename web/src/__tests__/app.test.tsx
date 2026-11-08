import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import DemoApp from "@/components/DemoApp";
import InvoiceList from "@/components/InvoiceList";
import QuestionPanel from "@/components/QuestionPanel";
import StatusBar from "@/components/StatusBar";
import UploadPanel from "@/components/UploadPanel";
import { ApiError, type Job, type QuestionResult } from "@/lib/api";

import { doneJob, fakeApi, jpegFile, lane } from "./fakes";

async function chooseAndSubmit(file = jpegFile()) {
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText(/Ảnh hoá đơn/), file);
  await user.click(screen.getByRole("button", { name: "Trích xuất" }));
  return user;
}

describe("demo key gate", () => {
  it("shows 'Sai mật khẩu demo' for a wrong key and opens the app for a good one", async () => {
    const invoices = vi.fn(async () => {
      throw new ApiError(401, "Sai mật khẩu demo");
    });
    const user = userEvent.setup();
    const { unmount } = render(<DemoApp makeApi={() => fakeApi({ invoices })} />);
    await user.type(await screen.findByLabelText("Mật khẩu demo"), "wrong-key");
    await user.click(screen.getByRole("button", { name: "Vào demo" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Sai mật khẩu demo");
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    unmount();

    render(<DemoApp makeApi={() => fakeApi()} />);
    await user.type(await screen.findByLabelText("Mật khẩu demo"), "good-key-123");
    await user.click(screen.getByRole("button", { name: "Vào demo" }));
    expect(await screen.findByRole("tab", { name: "Trích xuất hoá đơn" })).toBeInTheDocument();
    expect(window.sessionStorage.getItem("ptit-demo-key")).toBe("good-key-123");
  });

  it("returns to the key prompt when the API answers 401", async () => {
    window.sessionStorage.setItem("ptit-demo-key", "expired-key");
    const upload = vi.fn(async () => {
      throw new ApiError(401, "Sai mật khẩu demo");
    });
    render(<DemoApp makeApi={() => fakeApi({ upload })} pollMs={0} />);
    await screen.findByRole("tab", { name: "Trích xuất hoá đơn" });
    await chooseAndSubmit();
    expect(await screen.findByLabelText("Mật khẩu demo")).toBeInTheDocument();
    expect(window.sessionStorage.getItem("ptit-demo-key")).toBeNull();
  });
});

describe("upload flow", () => {
  it("polls the job and shows OCR text, fields, items and the lane comparison", async () => {
    const queued: Job = { ...doneJob(), status: "node1", invoice_id: null, result: {} };
    const job = vi.fn().mockResolvedValueOnce(queued).mockResolvedValueOnce(doneJob());
    const api = fakeApi({ job });
    render(<UploadPanel api={api} pollMs={0} />);
    await chooseAndSubmit();
    expect(await screen.findByText(/Đã lưu hoá đơn/)).toBeInTheDocument();
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), false);
    expect(job).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("ocr-text")).toHaveTextContent("MINIMART ANAN");
    expect(screen.getAllByText("MINIMART ANAN").length).toBeGreaterThan(0);
    expect(within(screen.getByRole("table", { name: "Bảng mặt hàng" })).getByText("Sữa tươi")).toBeInTheDocument();
    const adaptive = screen.getByTestId("lane-adaptive");
    expect(adaptive).toHaveTextContent("giống vanilla 40/40 trường");
    expect(screen.getByTestId("speedup-adaptive")).toHaveTextContent("1,43×");
    expect(screen.getByText(/không phải số benchmark/)).toHaveTextContent("A10G");
    expect(screen.getByTestId("step-done")).toHaveAttribute("data-state", "reached");
  });

  it("sends compare=all when the four-lane toggle is on", async () => {
    const api = fakeApi();
    render(<UploadPanel api={api} pollMs={0} />);
    const user = userEvent.setup();
    await user.click(screen.getByLabelText(/So sánh 4 làn/));
    await chooseAndSubmit();
    await screen.findByText(/Đã lưu hoá đơn/);
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), true);
  });

  it.each([
    [413, "Ảnh quá lớn"],
    [415, "không phải ảnh"],
  ])("explains HTTP %i without showing progress", async (status, text) => {
    const upload = vi.fn(async () => {
      throw new ApiError(status, "x");
    });
    render(<UploadPanel api={fakeApi({ upload })} pollMs={0} />);
    await chooseAndSubmit();
    expect(await screen.findByRole("alert")).toHaveTextContent(text);
    expect(screen.queryByLabelText("Tiến độ")).not.toBeInTheDocument();
  });

  it("marks the failed step and still shows the OCR text", async () => {
    const failed: Job = {
      ...doneJob(),
      status: "failed",
      invoice_id: null,
      error: "node1: adaptive: SOURCE_BINDING_ERROR at SOURCE_BINDING: mismatch",
      result: { ocr: { text: "CỬA HÀNG NAM OÁNH", finish_reason: "stop" } },
    };
    render(<UploadPanel api={fakeApi({ job: vi.fn(async () => failed) })} pollMs={0} />);
    await chooseAndSubmit();
    expect(await screen.findByText(/SOURCE_BINDING_ERROR/)).toBeInTheDocument();
    expect(screen.getByTestId("step-node1")).toHaveAttribute("data-state", "failed");
    expect(screen.getByTestId("ocr-text")).toHaveTextContent("CỬA HÀNG NAM OÁNH");
  });

  it("shows a slower lane's speed-up as it is", async () => {
    const slow = doneJob({ lanes: { vanilla: lane("vanilla"), adaptive: lane("adaptive", { speedup_tpot_vs_vanilla: 0.8 }) } });
    render(<UploadPanel api={fakeApi({ job: vi.fn(async () => slow) })} pollMs={0} />);
    await chooseAndSubmit();
    expect(await screen.findByTestId("speedup-adaptive")).toHaveTextContent("0,8×");
  });

  it("warns about values that are not in the OCR text", async () => {
    const job = doneJob({ grounding: { checked_fields: 5, grounded_fields: 4, ungrounded_fields: ["seller.phone"] } });
    render(<UploadPanel api={fakeApi({ job: vi.fn(async () => job) })} pollMs={0} />);
    await chooseAndSubmit();
    expect(await screen.findByRole("status")).toHaveTextContent("seller.phone");
  });
});

describe("question panel", () => {
  const base: QuestionResult = {
    question: "Tổng tiền?",
    status: "ok",
    sql: "SELECT SUM(total_amount_value) AS total FROM v_invoice_facts LIMIT 200",
    columns: ["total"],
    rows: [[4930094]],
    row_count: 1,
    answer: "Tổng tiền là 4.930.094 đồng.",
    warning: null,
    error: null,
    repairs: 0,
    calls: [{ step: "node2", lane: "static", latency_ms: 900 }],
  };

  async function ask(result: QuestionResult) {
    render(<QuestionPanel api={fakeApi({ ask: vi.fn(async () => result) })} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Câu hỏi về dữ liệu hoá đơn"), "Tổng tiền?");
    await user.click(screen.getByRole("button", { name: "Hỏi" }));
  }

  it("shows the answer, the SQL and the table", async () => {
    await ask(base);
    expect(await screen.findByText("Tổng tiền là 4.930.094 đồng.")).toBeInTheDocument();
    expect(screen.getByTestId("sql")).toHaveTextContent("SELECT SUM(total_amount_value)");
    expect(within(screen.getByRole("table", { name: "Bảng kết quả" })).getByText("4.930.094")).toBeInTheDocument();
  });

  it("shows only the table and the warning when the answer was withheld", async () => {
    await ask({ ...base, answer: null, warning: "fabricated_number" });
    expect(await screen.findByRole("status")).toHaveTextContent("đã bị ẩn");
    expect(screen.queryByText(/Tổng tiền là/)).not.toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Bảng kết quả" })).toBeInTheDocument();
  });

  it("shows the no-data sentence", async () => {
    await ask({ ...base, status: "no_data", rows: [], row_count: 0, answer: "Không có dữ liệu khớp câu hỏi." });
    expect(await screen.findByText("Không có dữ liệu khớp câu hỏi.")).toBeInTheDocument();
  });
});

describe("invoice list and status", () => {
  it("lists uploads first and opens an invoice", async () => {
    const invoices = vi.fn(async () => ({
      invoices: [
        { invoice_id: "up_1:up_1", seller_name: "CỬA HÀNG MỚI", issued_date: "2020-08-20", total_amount_value: 50000,
          total_amount_currency: "VND", item_count: 2, source: "upload" as const, created_at: "2026-11-08", lane: "adaptive" },
        { invoice_id: "m5_01_adaptive:m0_04_001", seller_name: "MINIMART ANAN", issued_date: "2020-08-10",
          total_amount_value: 42000, total_amount_currency: "VND", item_count: 1, source: "devset" as const,
          created_at: "2026-11-01", lane: "adaptive" },
      ],
    }));
    const api = fakeApi({ invoices });
    render(<InvoiceList api={api} />);
    await screen.findByText("CỬA HÀNG MỚI");
    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("CỬA HÀNG MỚI");
    expect(rows[1]).toHaveTextContent("tải lên");
    await userEvent.setup().click(rows[1]);
    expect(await screen.findByRole("region", { name: "Chi tiết hoá đơn" })).toBeInTheDocument();
    expect(api.invoice).toHaveBeenCalledWith("up_1:up_1");
  });

  it("moves services from cold to ready after warm-up", async () => {
    const api = fakeApi();
    render(<StatusBar api={api} refreshMs={0} />);
    expect(await screen.findByTestId("service-ocr")).toHaveTextContent("đang ngủ");
    await userEvent.setup().click(screen.getByRole("button", { name: "Làm nóng hệ thống" }));
    await waitFor(() => expect(screen.getByTestId("service-ocr")).toHaveTextContent("sẵn sàng"));
    expect(api.warmup).toHaveBeenCalledTimes(1);
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
