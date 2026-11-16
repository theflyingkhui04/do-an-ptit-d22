import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import Console from "@/components/Console";
import { ApiError, type Api, type BenchRun, type DetectResult, type Endpoint, type Job } from "@/lib/api";

import { BASE, CHAIN, ENDPOINTS, FAST, INVOICE, REPORT, doneJob, endpoint, fakeApi, jobEvents, jpegFile, question, run, speedRun } from "./fakes";
import { leafFields } from "@/lib/fields";

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

describe("brand", () => {
  const logo = (variant: "light" | "dark") => document.querySelector(`img[src*="flexsds-icon-${variant}"]`);

  // The white tile's bold red F stays legible at 16–36 px; the black tile's thin glow does not.
  it("uses the white tile on the key card (outlined on white) and on the dark side bar", async () => {
    window.sessionStorage.clear();
    window.history.replaceState(null, "", "/");
    const { unmount } = render(<Console makeApi={() => fakeApi()} />);
    await screen.findByLabelText("Mật khẩu demo");
    expect(logo("light")).toHaveAttribute("alt", "");
    expect(logo("light")).toHaveClass("ring-1");
    unmount();

    open("", fakeApi());
    await screen.findByRole("navigation", { name: "Điều hướng" });
    expect(logo("light")).toHaveAttribute("alt", "");
    expect(logo("dark")).toBeNull();
  });

  it("ships small icons, not the 1600 px source tiles", () => {
    for (const file of ["public/brand/flexsds-icon-light.png",
      "src/app/icon.png", "src/app/apple-icon.png", "src/app/favicon.ico"]) {
      expect(statSync(join(process.cwd(), file)).size).toBeLessThan(20_000);
    }
  });
});

describe("endpoints", () => {
  it("opens on the onboarding page when no endpoint has been added", async () => {
    open("", fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })) }));
    expect(await screen.findByRole("heading", { name: "Chưa có endpoint nào" })).toBeInTheDocument();
    const sample = screen.getByTestId("serve-sample");
    expect(sample).toHaveTextContent(/flexsds serve .*--target \S+ .*--draft \S+/);
    expect(sample).not.toHaveTextContent("--model");
    expect(sample).toHaveTextContent("Endpoint LLM created:");
    expect(screen.getByTestId("endpoint-summary")).toHaveTextContent("Chưa có endpoint");
  });

  it("detects pasted addresses, explains the tunnel for a local one and saves the usable ones", async () => {
    const found: Record<string, DetectResult> = {
      "https://acct--flexsds-serve.modal.run/v1": { input: "https://acct--flexsds-serve.modal.run/v1", url: "https://acct--flexsds-serve.modal.run/v1", ok: true, code: null, message: null, platform: "modal", model_id: "Qwen/Qwen2.5-7B-Instruct", max_model_len: 8192 },
      "http://127.0.0.1:8000/v1": { input: "http://127.0.0.1:8000/v1", url: "http://127.0.0.1:8000/v1", ok: false, code: "needs_tunnel", platform: "local", model_id: null, max_model_len: null,
        message: "Đây là địa chỉ trên máy của bạn; chạy lệnh `cloudflared tunnel --url http://127.0.0.1:8000` rồi dán lại." },
    };
    const detectEndpoints = vi.fn(async (urls: string[]) => ({ results: urls.map((u) => found[u]) }));
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
    // Each address is checked on its own, so one slow GPU does not hold back the others.
    expect(detectEndpoints).toHaveBeenCalledTimes(2);
    expect(detectEndpoints).toHaveBeenCalledWith(["https://acct--flexsds-serve.modal.run/v1"], "tok.secret");
    expect(detectEndpoints).toHaveBeenCalledWith(["http://127.0.0.1:8000/v1"], "tok.secret");

    await user.click(within(dialog).getByRole("button", { name: "Thêm 1 endpoint" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(addEndpoints).toHaveBeenCalledWith(["https://acct--flexsds-serve.modal.run/v1"], "tok.secret", undefined);
    expect(api.endpoints).toHaveBeenCalledTimes(2);
  });

  it("shows each address while it is checked, its own result as soon as it answers, and a hint for slow GPUs", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const a = "https://acct--a.modal.run/v1";
      const b = "https://acct--b.modal.run/v1";
      const answer: Record<string, (value: { results: DetectResult[] }) => void> = {};
      const detectEndpoints = vi.fn((urls: string[]) => new Promise<{ results: DetectResult[] }>((resolve) => {
        answer[urls[0]] = resolve;
      }));
      open("endpoints", fakeApi({ detectEndpoints }));
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click((await screen.findAllByRole("button", { name: "Thêm endpoint" }))[0]);
      const dialog = screen.getByRole("dialog", { name: "Thêm endpoint" });
      await user.type(within(dialog).getByLabelText("Endpoint URL"), `${a}\n${b}`);
      await user.click(within(dialog).getByRole("button", { name: "Kiểm tra" }));

      const button = within(dialog).getByRole("button", { name: /Đang kiểm tra/ });
      expect(button).toHaveAttribute("aria-busy", "true");
      expect(within(button).getByTestId("spinner")).toBeInTheDocument();
      expect(within(dialog).getAllByTestId("check-running")).toHaveLength(2);
      expect(within(dialog).queryByText(/đang ngủ/)).toBeNull();

      await act(async () => {
        vi.advanceTimersByTime(11_000);
      });
      expect(button).toHaveTextContent(/1[01] s/);
      expect(within(dialog).getByText(/đang ngủ/)).toBeInTheDocument();

      await act(async () => {
        answer[a]({ results: [{ input: a, url: a, ok: true, code: null, message: null, platform: "modal", model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len: 8192 }] });
      });
      expect(within(dialog).getByText("Qwen/Qwen2.5-7B-Instruct-AWQ")).toBeInTheDocument();
      expect(within(dialog).getAllByTestId("check-running")).toHaveLength(1); // b is still waking up
    } finally {
      vi.useRealTimers();
    }
  });

  it("marks a row while it is checked, then says when and how fast it answered", async () => {
    let finish: (value: Endpoint) => void = () => undefined;
    const checkEndpoint = vi.fn(() => new Promise<Endpoint>((resolve) => {
      finish = resolve;
    }));
    open("endpoints", fakeApi({ checkEndpoint }));
    const user = userEvent.setup();
    const recheck = (await screen.findAllByRole("button", { name: "Kiểm tra lại" }))[0];
    await user.click(recheck);
    const row = recheck.closest("tr") as HTMLElement;
    expect(within(row).getByText(/Đang kiểm tra ·/)).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: /Kiểm tra lại|Đang kiểm tra/ })).toHaveAttribute("aria-busy", "true");
    expect(within(screen.getByTestId("endpoint-summary")).getByTestId("spinner")).toBeInTheDocument();

    await act(async () => finish(ENDPOINTS[0]));
    expect(await within(row).findByText(/Vừa kiểm tra/)).toBeInTheDocument();
    expect(within(row).queryByText(/Đang kiểm tra ·/)).toBeNull();
    expect(within(screen.getByTestId("endpoint-summary")).queryByTestId("spinner")).toBeNull();
  });

  it("checks every endpoint at once and reloads the list once", async () => {
    const answers: Array<() => void> = [];
    const checkEndpoint = vi.fn((id: string) => new Promise<Endpoint>((resolve) => {
      answers.push(() => resolve(ENDPOINTS.find((e) => e.endpoint_id === id)!));
    }));
    const api = fakeApi({ checkEndpoint });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Kiểm tra tất cả" }));
    expect(checkEndpoint).toHaveBeenCalledTimes(ENDPOINTS.length); // all started before any answered
    expect(screen.getAllByText(/Đang kiểm tra ·/)).toHaveLength(ENDPOINTS.length);

    await act(async () => answers.forEach((answerNow) => answerNow()));
    await waitFor(() => expect(screen.getAllByText(/Vừa kiểm tra/)).toHaveLength(ENDPOINTS.length));
    expect(api.endpoints).toHaveBeenCalledTimes(2); // first load + one reload
  });

  it("keeps a failed check on its row instead of a page-wide error", async () => {
    const checkEndpoint = vi.fn(async () => {
      throw new Error("Không gọi được máy chủ");
    });
    open("endpoints", fakeApi({ checkEndpoint }));
    const user = userEvent.setup();
    const recheck = (await screen.findAllByRole("button", { name: "Kiểm tra lại" }))[0];
    await user.click(recheck);
    const row = recheck.closest("tr") as HTMLElement;
    expect(await within(row).findByText(/Không gọi được máy chủ/)).toBeInTheDocument();
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

  it("does not warn when the served names differ but the model behind them is the same", async () => {
    const same = { base_model: "/models/qwen2.5-7b-awq" };
    const endpoints = vi.fn(async () => ({ endpoints: [
      endpoint(BASE, "vanilla", { is_baseline: true, model_id: "qwen2.5-7b-vanilla", meta: same }),
      endpoint(FAST, "adaptive", { model_id: "qwen2.5-7b-adaptive", meta: same }),
    ] }));
    open("endpoints", fakeApi({ endpoints }));
    await screen.findByText("qwen2.5-7b-adaptive");
    expect(screen.queryByText(/model khác nhau/)).not.toBeInTheDocument();
  });

  it("warns when the endpoints serve different models", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [...ENDPOINTS, endpoint("ep_0000000c", "llama", { model_id: "meta/llama" })] }));
    open("endpoints", fakeApi({ endpoints }));
    expect(await screen.findByText(/2 model khác nhau/)).toBeInTheDocument();
  });
});

describe("node 1", () => {
  async function runNode1(job: ReturnType<typeof vi.fn>) {
    const api = fakeApi({ job });
    open("node1", api);
    const user = userEvent.setup();
    await user.upload(await screen.findByLabelText("Ảnh hoá đơn"), jpegFile());
    await user.click(screen.getByRole("button", { name: "Chạy Node 1" }));
    return { api, user };
  }

  it("shows each endpoint's output side by side, the comparison under the table and the saved invoice", async () => {
    const queued: Job = { ...doneJob(), status: "node1", invoice_id: null, result: {} };
    const job = vi.fn().mockResolvedValueOnce(queued).mockResolvedValueOnce(doneJob());
    const { api } = await runNode1(job);
    const table = await screen.findByRole("table", { name: "So sánh với mốc" });
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), [FAST]);
    expect(job).toHaveBeenCalledTimes(2);

    const base = within(table).getByTestId(`run-${BASE}`);
    const fast = within(table).getByTestId(`run-${FAST}`);
    expect(base).toHaveTextContent("vanilla");
    expect(base).toHaveTextContent("mốc");
    expect(fast).toHaveTextContent("TTFT 300 ms");
    expect(fast).toHaveTextContent("TPOT 14 ms");
    const seller = within(table).getByRole("rowheader", { name: "Người bán" }).closest("tr") as HTMLElement;
    expect(within(seller).getAllByText("MINIMART ANAN")).toHaveLength(2); // both columns show their full output

    const conclusion = screen.getByTestId(`conclusion-${FAST}`);
    expect(conclusion).toHaveTextContent("Giống mốc");
    expect(conclusion).toHaveTextContent("40/40 trường");
    expect(conclusion).toHaveTextContent("Nhanh hơn mốc");
    expect(conclusion).toHaveTextContent("1,43×");
    expect(screen.getByRole("list", { name: "Các bước" })).toHaveTextContent(/OCR.*9 s.*Node 1.*13 s/);
    expect(within(screen.getByRole("table", { name: "Bảng mặt hàng" })).getByText("Sữa tươi")).toBeInTheDocument();
    expect(screen.getByText("Tổng các dòng lệch tổng hoá đơn")).toBeInTheDocument();
    expect(api.invoices).toHaveBeenCalledTimes(2);
  });

  it("opens on the main fields and can show every field, grouped", async () => {
    const { user } = await runNode1(vi.fn(async () => doneJob()));
    const table = await screen.findByRole("table", { name: "So sánh với mốc" });
    expect(within(table).queryByRole("rowheader", { name: "Mã hàng" })).toBeNull();
    const total = leafFields(INVOICE).length;
    await user.click(screen.getAllByRole("button", { name: `Xem đủ ${total} trường` })[0]);
    expect(within(table).getByRole("columnheader", { name: "Dòng hàng 1" })).toBeInTheDocument();
    expect(within(table).getByRole("rowheader", { name: "Mã hàng" })).toBeInTheDocument();
    expect(within(table).getAllByRole("rowheader")).toHaveLength(total);
    await user.click(screen.getByRole("button", { name: "Trường chính" }));
    expect(within(table).queryByRole("rowheader", { name: "Mã hàng" })).toBeNull();
  });

  it("keeps the log closed until asked, then opens the part that was clicked", async () => {
    const { user } = await runNode1(vi.fn(async () => doneJob()));
    await screen.findByRole("table", { name: "So sánh với mốc" });
    expect(screen.queryByRole("dialog", { name: "Log lượt chạy" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Xem log" }));
    let drawer = screen.getByRole("dialog", { name: "Log lượt chạy" });
    expect(within(drawer).getByText("Hoàn tất sau 22,7 s")).toBeVisible();
    expect(within(drawer).getByRole("button", { name: /^Tất cả/ })).toHaveAttribute("aria-pressed", "true");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Log lượt chạy" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Xem log adaptive" }));
    drawer = screen.getByRole("dialog", { name: "Log lượt chạy" });
    expect(within(drawer).getByRole("button", { name: /^adaptive/ })).toHaveAttribute("aria-pressed", "true");
    expect(within(drawer).getByText("Token đầu tiên sau 0,3 s, endpoint đang trả lời")).toBeVisible();
    expect(within(drawer).getByText("Đọc xong 3 dòng chữ trong 9,0 s; đã bỏ dấu tiếng Việt")).not.toBeVisible();
  });

  it("flags a part that has had no signal for a minute", async () => {
    const running: Job = { ...doneJob(), status: "node1", invoice_id: null, server_time: 200,
      result: { ...doneJob().result, runs: undefined, invoice: undefined, stored_endpoint: undefined,
        events: jobEvents().slice(0, 5).map((e) => ({ ...e, t: e.t <= 109.2 ? e.t : 110 })) } };
    const { user } = await runNode1(vi.fn(async () => running));
    const fast = await screen.findByTestId(`run-${FAST}`);
    expect(fast).toHaveTextContent("Không có tín hiệu 1 phút 30 s");
    await user.click(screen.getByRole("button", { name: "Xem log adaptive" }));
    // vanilla went quiet too; with the drawer on adaptive only adaptive's warning is shown
    const quiet = within(screen.getByRole("dialog", { name: "Log lượt chạy" })).getAllByText(/Chưa có tín hiệu trong 1 phút 30 s/);
    expect(quiet.filter((line) => !(line.closest("li") as HTMLElement).hidden)).toHaveLength(1);
  });

  it("keeps asking when the backend cannot be reached and says so in the log", async () => {
    const job = vi.fn().mockRejectedValueOnce(new Error("Failed to fetch")).mockResolvedValue(doneJob());
    const { user } = await runNode1(job);
    await screen.findByRole("table", { name: "So sánh với mốc" });
    await waitFor(() => expect(screen.getByTestId(`conclusion-${FAST}`)).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Xem log" }));
    const drawer = screen.getByRole("dialog", { name: "Log lượt chạy" });
    expect(within(drawer).getByText(/Không gọi được backend, đã thử 1 lần/)).toBeInTheDocument();
    expect(within(drawer).getByText(/Đã kết nối lại backend/)).toBeInTheDocument();
  });

  it("notes when the system put back a source value the model copied wrong", async () => {
    await runNode1(vi.fn(async () => doneJob({ runs: { [FAST]: run(FAST, "adaptive", { source_restored: ["image_path"] }), [BASE]: run(BASE, "vanilla") } })));
    expect(await screen.findByTestId(`run-${FAST}`)).toHaveTextContent("Hệ thống điền lại nguồn: image_path");
    expect(screen.getByTestId(`run-${BASE}`)).not.toHaveTextContent("điền lại nguồn");
  });

  it("explains a source copy error and suggests running again", async () => {
    const failed: Job = { ...doneJob(), status: "failed", invoice_id: null,
      error: "node1: adaptive: SOURCE_BINDING_ERROR at SOURCE_BINDING: Source binding mismatch: value mismatch: image_path" };
    open("node1", fakeApi({ job: vi.fn(async () => failed) }));
    const user = userEvent.setup();
    await user.upload(await screen.findByLabelText("Ảnh hoá đơn"), jpegFile());
    await user.click(screen.getByRole("button", { name: "Chạy Node 1" }));
    expect(await screen.findByText(/Model chép sai thông tin nguồn/)).toHaveTextContent("Chạy Node 1 lần nữa");
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

  it("says how a text filter was matched against the unaccented data", async () => {
    const ask = vi.fn(async () => question({ text_rewrites: [{ column: "seller_name", from: "%Nhà sách Cẩm Phả%", to: "%nha%sach%cam%pha%" }] }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Q1" }));
    expect(await screen.findByText(/Khớp chữ không dấu/)).toHaveTextContent("%Nhà sách Cẩm Phả% → %nha%sach%cam%pha%");
  });

  it("shows a flawed answer as written, with its flags", async () => {
    const ask = vi.fn(async () => question({
      answer: "Tổng tiền là 76.000. beğen", warning: "foreign_text", foreign_words: ["beğen"], ungrounded_numbers: ["76.000"],
    }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Q1" }));
    expect(await screen.findByText("Tổng tiền là 76.000. beğen")).toBeInTheDocument();
    expect(screen.getByText(/Chữ không phải tiếng Việt/)).toHaveTextContent("beğen");
    expect(screen.getByText(/Số không có trong kết quả SQL/)).toHaveTextContent("76.000");
    expect(screen.queryByText(/nên không hiện/)).not.toBeInTheDocument();
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

  it("flags foreign words in the report", async () => {
    const report = vi.fn(async () => ({ ...REPORT, language_check: { foreign_words: ["城市"] } }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    expect(await screen.findByText(/Chữ không phải tiếng Việt/)).toHaveTextContent("城市");
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
    // at 8 users the best endpoint is still slower than the baseline: the tile must say so
    expect(screen.getByText("Tốt nhất ở 8 người dùng").parentElement).toHaveTextContent("adaptive, chậm hơn mốc");
    expect(screen.getByText(/khoảng 3 phút/)).toBeInTheDocument();
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
    expect(await screen.findByText("giống mốc 2/3 node, khác ở Node 3")).toBeInTheDocument();
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
