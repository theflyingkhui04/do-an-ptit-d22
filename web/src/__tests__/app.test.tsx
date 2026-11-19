import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import Console from "@/components/Console";
import { ApiError, type Api, type BenchRun, type DetectResult, type Endpoint, type Job, type ReportResult } from "@/lib/api";

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
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
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
    expect(sample).not.toHaveTextContent("--model");
    expect(sample).toHaveTextContent("Endpoint LLM created:");
    expect(screen.getByTestId("endpoint-summary")).toHaveTextContent("Chưa có endpoint");
  });

  it("lists each endpoint on one row without the parts it does not need", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [
      endpoint(FAST, "adaptive", { model_id: "qwen2.5-7b-adaptive", meta: { gpu: "NVIDIA L4", draft_method: "eagle3", adaptive: true } }),
      endpoint(BASE, "vanilla", { is_baseline: true, model_id: "vanilla", has_key: false, status: "down",
        status_message: "Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động hoặc đang ngủ, hoặc phiên flexsds serve đã dừng." }),
    ] }));
    open("endpoints", fakeApi({ endpoints }));
    const rows = await screen.findAllByRole("article");
    expect(rows.map((r) => r.getAttribute("aria-label"))).toEqual(["vanilla", "adaptive"]); // the baseline first
    const vanilla = rowOf("vanilla");
    expect(within(vanilla).getByText("Mốc")).toBeInTheDocument();
    expect(within(vanilla).getByText("Đã dừng")).toBeInTheDocument();
    expect(within(vanilla).queryByText("GPU đang ngủ hoặc khởi động (HTTP 503)")).toBeNull(); // only in the tooltip
    expect(within(vanilla).getByTitle("GPU đang ngủ hoặc khởi động (HTTP 503)")).toBeInTheDocument();
    expect(within(vanilla).queryByText(/Model endpoint báo/)).toBeNull(); // same as its name
    expect(within(vanilla).queryByText(/Chưa rõ GPU/)).toBeNull();
    const adaptive = rowOf("adaptive");
    expect(within(adaptive).getByText("NVIDIA L4, EAGLE-3, Hardware-Adaptive")).toBeInTheDocument();
    expect(within(adaptive).getByText("qwen2.5-7b-adaptive")).toBeInTheDocument();
    expect(within(adaptive).getByText(/có khoá API/)).toBeInTheDocument();
    for (const action of ["Kiểm tra adaptive", "Sửa adaptive", "Xoá adaptive"]) {
      expect(within(adaptive).getByRole("button", { name: action })).toBeVisible();
    }
    // One "Thêm endpoint" (top bar), no radio column, no tunnel panel
    expect(screen.getAllByRole("button", { name: "Thêm endpoint" })).toHaveLength(1);
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByText("Endpoint trên máy cá nhân")).toBeNull();
  });

  it("adds one address at a time: explains a local one, then saves a usable one under the typed name", async () => {
    const modal = "https://acct--flexsds-serve.modal.run/v1";
    const local = "http://127.0.0.1:8000/v1";
    const found: Record<string, DetectResult> = {
      [modal]: ok(modal),
      [local]: { ...refused(local, "needs_tunnel", "Đây là địa chỉ trên máy của bạn; chạy lệnh `cloudflared tunnel --url http://127.0.0.1:8000` rồi dán lại."), platform: "local" },
    };
    const detectEndpoints = vi.fn(async (urls: string[]) => ({ results: urls.map((u) => found[u]) }));
    const addEndpoints = vi.fn(async () => ({ results: [ok(modal)], saved: [endpoint("ep_0000000c", "adaptive L4")] }));
    const api = fakeApi({ detectEndpoints, addEndpoints });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
    const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
    expect(within(drawer).queryByLabelText(/Khoá API/)).toBeNull();
    expect(within(drawer).queryByText(/serving_config/)).toBeNull();
    const address = within(drawer).getByLabelText("Địa chỉ API");
    expect(address.tagName).toBe("INPUT"); // one endpoint per add (Canva review 19/11)
    expect(within(drawer).queryByText(/mỗi dòng/)).toBeNull();

    await user.type(address, local);
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(drawer).findByText("Cần mở đường hầm")).toBeInTheDocument();
    expect(within(drawer).getByText("cloudflared tunnel --url http://127.0.0.1:8000")).toBeInTheDocument();
    expect(within(drawer).queryByRole("button", { name: /Thêm \d endpoint/ })).toBeNull();

    // The whole "- API: …" line printed by the CLI can be pasted; Enter checks it.
    await user.clear(address);
    await user.type(address, `- API: ${modal}{Enter}`);
    expect(detectEndpoints).toHaveBeenLastCalledWith([modal], undefined);
    const name = await within(drawer).findByLabelText(`Tên hiển thị cho ${modal}`);
    expect(name).toHaveValue("Qwen/Qwen2.5-7B-Instruct");
    await user.clear(name);
    await user.type(name, "adaptive L4");
    await user.click(within(drawer).getByRole("button", { name: "Thêm 1 endpoint" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(addEndpoints).toHaveBeenCalledWith([{ url: modal, name: "adaptive L4", api_key: undefined }]);
    expect(api.endpoints).toHaveBeenCalledTimes(2);
  });

  it("asks for a key on the row of an endpoint that refuses the call", async () => {
    const url = "https://llm.other-team.example.com/v1";
    const detectEndpoints = vi.fn(async (_urls: string[], key?: string) => ({
      results: [key === "k-1" ? ok(url, "meta/llama") : refused(url, "auth_failed", "Endpoint từ chối khoá API (HTTP 401).")],
    }));
    const addEndpoints = vi.fn(async () => ({ results: [ok(url, "meta/llama")], saved: [endpoint("ep_0000000c", "meta/llama")] }));
    open("endpoints", fakeApi({ detectEndpoints, addEndpoints }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
    const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
    await user.type(within(drawer).getByLabelText("Địa chỉ API"), url);
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(drawer).findByText("Endpoint này cần khoá API (HTTP 401)")).toBeInTheDocument();
    await user.type(within(drawer).getByLabelText(`Khoá API cho ${url}`), "k-1");
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra lại với khoá này" }));
    expect(await within(drawer).findByLabelText(`Tên hiển thị cho ${url}`)).toHaveValue("meta/llama");
    expect(detectEndpoints).toHaveBeenLastCalledWith([url], "k-1");
    await user.click(within(drawer).getByRole("button", { name: "Thêm 1 endpoint" }));
    expect(addEndpoints).toHaveBeenCalledWith([{ url, name: undefined, api_key: "k-1" }]);
  });

  it("says an address is already in the list", async () => {
    const saved = "https://acct--vanilla.modal.run/v1";
    const detectEndpoints = vi.fn(async () => ({
      results: [{ ...refused(saved, "exists", "Đã có trong danh sách với tên vanilla, không cần thêm lại."), endpoint_id: BASE, name: "vanilla" }],
    }));
    open("endpoints", fakeApi({ detectEndpoints }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
    const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
    await user.type(within(drawer).getByLabelText("Địa chỉ API"), saved);
    await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));
    expect(await within(drawer).findByText("Đã có trong danh sách với tên vanilla, không cần thêm lại.")).toBeInTheDocument();
    expect(within(drawer).queryByRole("button", { name: /Thêm \d endpoint/ })).toBeNull();
  });

  it("shows the address while it is checked, its result as soon as it answers, and a hint for slow GPUs", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const a = "https://acct--a.modal.run/v1";
      let answer: (value: { results: DetectResult[] }) => void = () => undefined;
      const detectEndpoints = vi.fn(() => new Promise<{ results: DetectResult[] }>((resolve) => {
        answer = resolve;
      }));
      open("endpoints", fakeApi({ detectEndpoints }));
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click(await screen.findByRole("button", { name: "Thêm endpoint" }));
      const drawer = screen.getByRole("dialog", { name: "Thêm endpoint" });
      await user.type(within(drawer).getByLabelText("Địa chỉ API"), a);
      await user.click(within(drawer).getByRole("button", { name: "Kiểm tra" }));

      const button = within(drawer).getByRole("button", { name: /Đang kiểm tra/ });
      expect(button).toHaveAttribute("aria-busy", "true");
      expect(within(button).getByTestId("spinner")).toBeInTheDocument();
      expect(within(drawer).getAllByTestId("check-running")).toHaveLength(1);
      expect(within(drawer).queryByText(/đang ngủ/)).toBeNull();

      await act(async () => {
        vi.advanceTimersByTime(11_000);
      });
      expect(button).toHaveTextContent(/1[01] s/);
      expect(within(drawer).getByText(/đang ngủ/)).toBeInTheDocument();

      await act(async () => {
        answer({ results: [ok(a, "Qwen/Qwen2.5-7B-Instruct-AWQ")] });
      });
      expect(within(drawer).getByLabelText(`Tên hiển thị cho ${a}`)).toHaveValue("Qwen/Qwen2.5-7B-Instruct-AWQ");
      expect(within(drawer).queryAllByTestId("check-running")).toHaveLength(0);
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
    await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
    const row = rowOf("vanilla");
    expect(within(row).getByText(/Đang kiểm tra, \d+ s/)).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: "Kiểm tra vanilla" })).toHaveAttribute("aria-busy", "true");
    expect(within(screen.getByTestId("endpoint-summary")).getByTestId("spinner")).toBeInTheDocument();

    await act(async () => finish({ ...ENDPOINTS[0], checked_at: new Date().toISOString() }));
    expect(await within(row).findByText(/Kiểm tra vừa xong, mất/)).toBeInTheDocument();
    expect(within(row).queryByText(/Đang kiểm tra,/)).toBeNull();
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
    expect(screen.getAllByText(/Đang kiểm tra, \d+ s/)).toHaveLength(ENDPOINTS.length);

    await act(async () => answers.forEach((answerNow) => answerNow()));
    await waitFor(() => expect(screen.getAllByText(/, mất /)).toHaveLength(ENDPOINTS.length));
    expect(api.endpoints).toHaveBeenCalledTimes(2); // first load + one reload
  });

  it("keeps a failed check on its row instead of a page-wide error", async () => {
    const checkEndpoint = vi.fn(async () => {
      throw new Error("Không gọi được máy chủ");
    });
    open("endpoints", fakeApi({ checkEndpoint }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
    expect(await within(rowOf("vanilla")).findByText(/Không gọi được máy chủ/)).toBeInTheDocument();
  });

  const WAKING: Endpoint = { ...ENDPOINTS[0], status: "starting",
    status_message: "Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động hoặc đang ngủ, hoặc phiên flexsds serve đã dừng." };

  it("keeps asking a waking GPU every 20 s until it answers", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const answers: Endpoint[] = [WAKING, { ...ENDPOINTS[0], checked_at: new Date().toISOString() }];
      const checkEndpoint = vi.fn(async () => answers.shift() ?? WAKING);
      const api = fakeApi({ checkEndpoint });
      open("endpoints", api);
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
      const row = rowOf("vanilla");
      expect(await within(row).findByText(/Đang khởi động GPU/)).toBeInTheDocument();
      expect(within(row).getByText(/tối đa 5 phút/)).toBeInTheDocument();
      expect(within(row).queryByText("Đã dừng")).toBeNull();
      expect(within(row).queryByText("Đang chạy")).toBeNull();
      expect(checkEndpoint).toHaveBeenCalledTimes(1);

      await act(async () => {
        vi.advanceTimersByTime(20_000);
      });
      expect(checkEndpoint).toHaveBeenCalledTimes(2);
      expect(await within(row).findByText("Đang chạy")).toBeInTheDocument();
      expect(within(row).getByText(/Kiểm tra vừa xong, mất/)).toBeInTheDocument();
      expect(within(row).queryByText(/Đang khởi động GPU/)).toBeNull();
      expect(api.endpoints).toHaveBeenCalledTimes(2); // first load + one reload
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops asking after 5 minutes and says the GPU never came up", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const checkEndpoint = vi.fn(async () => WAKING);
      open("endpoints", fakeApi({ checkEndpoint }));
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
      const row = rowOf("vanilla");
      await within(row).findByText(/Đang khởi động GPU/);
      for (let i = 0; i < 16; i++) {
        await act(async () => {
          vi.advanceTimersByTime(20_000);
        });
      }
      expect(await within(row).findByText("Đã dừng")).toBeInTheDocument();
      expect(within(row).getByText(/chưa lên sau 5 phút/)).toBeInTheDocument();
      const calls = checkEndpoint.mock.calls.length;
      expect(calls).toBeGreaterThanOrEqual(10);
      expect(calls).toBeLessThanOrEqual(16);
      await act(async () => {
        vi.advanceTimersByTime(60_000);
      });
      expect(checkEndpoint).toHaveBeenCalledTimes(calls);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops asking a waking GPU once the page is closed", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const checkEndpoint = vi.fn(async () => WAKING);
      open("endpoints", fakeApi({ checkEndpoint }));
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      await user.click(await screen.findByRole("button", { name: "Kiểm tra vanilla" }));
      await within(rowOf("vanilla")).findByText(/Đang khởi động GPU/);
      cleanup();
      await act(async () => {
        vi.advanceTimersByTime(60_000);
      });
      expect(checkEndpoint).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a saved waking endpoint as not ready yet, not as stopped", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [WAKING, ENDPOINTS[1]] }));
    open("endpoints", fakeApi({ endpoints }));
    const row = await screen.findByRole("article", { name: "vanilla" });
    expect(within(row).getByText("Chưa sẵn sàng")).toBeInTheDocument();
    expect(within(row).getByTitle("GPU đang ngủ hoặc khởi động (HTTP 503)")).toBeInTheDocument();
    expect(within(row).queryByText("Đã dừng")).toBeNull();
    const user = userEvent.setup();
    await user.click(within(row).getByRole("button", { name: "Sửa vanilla" }));
    const drawer = screen.getByRole("dialog", { name: "Sửa endpoint" });
    expect(within(drawer).getByText(/, chưa sẵn sàng \(GPU đang ngủ hoặc khởi động\)/)).toBeInTheDocument();
    expect(within(drawer).getByText(WAKING.status_message!)).toBeInTheDocument();
  });

  it("edits name, address, key, notes and baseline in the drawer, sending only what changed", async () => {
    const updateEndpoint = vi.fn(async () => ENDPOINTS[1]);
    const api = fakeApi({ updateEndpoint });
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Sửa adaptive" }));
    const drawer = screen.getByRole("dialog", { name: "Sửa endpoint" });
    expect(within(drawer).getByText("Đã đặt khoá, không hiện lại")).toBeInTheDocument();
    await user.clear(within(drawer).getByLabelText("Tên hiển thị"));
    await user.type(within(drawer).getByLabelText("Tên hiển thị"), "adaptive L4");
    await user.clear(within(drawer).getByLabelText("Địa chỉ API"));
    await user.type(within(drawer).getByLabelText("Địa chỉ API"), "https://acct--adaptive-l4.modal.run/v1");
    await user.click(within(drawer).getByRole("checkbox", { name: /endpoint mốc/ }));
    await user.click(within(drawer).getByRole("button", { name: "Đổi khoá" }));
    await user.type(within(drawer).getByLabelText("Khoá mới"), "k-2");
    await user.clear(within(drawer).getByLabelText("GPU"));
    await user.type(within(drawer).getByLabelText("GPU"), "NVIDIA A10G");
    await user.selectOptions(within(drawer).getByLabelText("Cách giải mã"), "Không nháp (vanilla)");
    await user.click(within(drawer).getByRole("button", { name: "Lưu thay đổi" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(updateEndpoint).toHaveBeenCalledWith(FAST, {
      name: "adaptive L4", base_url: "https://acct--adaptive-l4.modal.run/v1", api_key: "k-2", is_baseline: true,
      meta: { gpu: "NVIDIA A10G", draft_method: "none", adaptive: null },
    });
    expect(api.endpoints).toHaveBeenCalledTimes(2);
  });

  it("keeps the drawer open with the reason when a new address does not answer", async () => {
    const updateEndpoint = vi.fn(async () => {
      throw new ApiError(422, "Endpoint chưa sẵn sàng (HTTP 503): GPU đang khởi động hoặc đang ngủ.");
    });
    open("endpoints", fakeApi({ updateEndpoint }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "vanilla" }));
    const drawer = screen.getByRole("dialog", { name: "Sửa endpoint" });
    await user.type(within(drawer).getByLabelText("Địa chỉ API"), "x");
    await user.click(within(drawer).getByRole("button", { name: "Lưu thay đổi" }));
    expect(await within(drawer).findByText(/HTTP 503/)).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Sửa endpoint" })).toBeInTheDocument();
  });

  it("removes a key and sends nothing else", async () => {
    const api = fakeApi();
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Sửa vanilla" }));
    const drawer = screen.getByRole("dialog", { name: "Sửa endpoint" });
    await user.click(within(drawer).getByRole("button", { name: "Bỏ khoá" }));
    expect(within(drawer).getByText("Sẽ bỏ khoá khi lưu")).toBeInTheDocument();
    await user.click(within(drawer).getByRole("button", { name: "Lưu thay đổi" }));
    expect(api.updateEndpoint).toHaveBeenCalledWith(BASE, { api_key: null });
  });

  it("fills the notes from the serving_config.json the CLI wrote", async () => {
    const api = fakeApi();
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Sửa vanilla" }));
    const drawer = screen.getByRole("dialog", { name: "Sửa endpoint" });
    const config = { target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ", draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator",
      draft_method: "eagle3", enable_adaptive: true, hardware: { device_name: "NVIDIA A10G" } };
    await user.upload(within(drawer).getByLabelText("Nạp từ serving_config.json"),
      new File([JSON.stringify(config)], "serving_config.json", { type: "application/json" }));
    expect(await within(drawer).findByText(/Đã đọc serving_config.json/)).toBeInTheDocument();
    expect(within(drawer).getByLabelText("GPU")).toHaveValue("NVIDIA A10G");
    expect(within(drawer).getByLabelText("Model nháp")).toHaveValue("duckhoi/qwen2.5-7b-eagle3.1-speculator");
    await user.click(within(drawer).getByRole("button", { name: "Lưu thay đổi" }));
    expect(api.updateEndpoint).toHaveBeenCalledWith(BASE, { meta: {
      gpu: "NVIDIA A10G", draft_method: "eagle3", adaptive: true, draft_model: "duckhoi/qwen2.5-7b-eagle3.1-speculator",
      target_model: "Qwen/Qwen2.5-7B-Instruct-AWQ" } });
  });

  it("asks before deleting and says what deleting does not stop", async () => {
    const api = fakeApi();
    open("endpoints", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Xoá vanilla" }));
    let confirm = screen.getByRole("alertdialog", { name: "Xoá vanilla?" });
    expect(confirm).toHaveTextContent("không bị tắt");
    expect(confirm).toHaveTextContent("Đây là endpoint mốc");
    await user.click(within(confirm).getByRole("button", { name: "Giữ lại" }));
    expect(api.deleteEndpoint).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Sửa adaptive" }));
    await user.click(within(screen.getByRole("dialog", { name: "Sửa endpoint" })).getByRole("button", { name: "Xoá endpoint" }));
    confirm = screen.getByRole("alertdialog", { name: "Xoá adaptive?" });
    expect(confirm).not.toHaveTextContent("Đây là endpoint mốc");
    await user.click(within(confirm).getByRole("button", { name: "Xoá endpoint" }));
    await waitFor(() => expect(api.deleteEndpoint).toHaveBeenCalledWith(FAST));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
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
    await user.click(screen.getByRole("button", { name: "Bắt đầu" }));
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
    await user.click(screen.getByRole("button", { name: "Bắt đầu" }));
    expect(await screen.findByText(/Model chép sai thông tin nguồn/)).toHaveTextContent("Bắt đầu lần nữa");
  });

  it("rejects a file that is not a JPEG or PNG before uploading", async () => {
    const api = fakeApi();
    open("node1", api);
    await userEvent.setup({ applyAccept: false }).upload(await screen.findByLabelText("Ảnh hoá đơn"), jpegFile("a.pdf", "application/pdf"));
    expect(await screen.findByText("Chỉ nhận ảnh JPEG hoặc PNG.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bắt đầu" })).toBeDisabled();
  });

  it("asks for an endpoint when none exists", async () => {
    window.sessionStorage.setItem("ptit-demo-key", "k");
    open("node1", fakeApi({ endpoints: vi.fn(async () => ({ endpoints: [] })) }));
    expect(await screen.findByRole("heading", { name: "Cần ít nhất một endpoint" })).toBeInTheDocument();
  });
});

describe("node 2", () => {
  const Q1 = "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?";
  // The baseline (vanilla) is slower; the chosen endpoint (adaptive) answers the same unless told otherwise.
  const timed = (endpointId: string, extra: Parameters<typeof question>[0] = {}) => question({
    calls: [
      { step: "node2", endpoint: endpointId, wall_ms: endpointId === BASE ? 2600 : 2100, ttft_ms: 400, tpot_ms: endpointId === BASE ? 20 : 15, completion_tokens: 40 },
      { step: "answer", endpoint: endpointId, wall_ms: 900 },
    ],
    endpoint: { endpoint_id: endpointId, name: endpointId === BASE ? "vanilla" : "adaptive" },
    ...extra,
  });
  const column = (name: string) => screen.getByRole("article", { name });

  it("asks on the chosen endpoint and the baseline, shows each answer first and folds the SQL", async () => {
    const ask = vi.fn(async (_q: string, endpointId: string) => timed(endpointId));
    open("node2", fakeApi({ ask }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /^Q1/ }));
    expect(ask).toHaveBeenCalledWith(Q1, FAST);
    expect(ask).toHaveBeenCalledWith(Q1, BASE); // the baseline always runs too
    expect(ask).toHaveBeenCalledTimes(2);
    await screen.findAllByText("Tổng chi tiêu tháng 08/2020 là 3.912.548 đồng.");
    expect(screen.getAllByRole("article").map((a) => a.getAttribute("aria-label"))).toEqual(["vanilla", "adaptive"]);

    const adaptive = column("adaptive");
    expect(within(adaptive).getByText("Đã trả lời")).toBeInTheDocument();
    expect(within(adaptive).getByText("3 s")).toBeInTheDocument(); // 2,1 s + 0,9 s
    expect(within(adaptive).getByText("15 ms")).toBeInTheDocument();
    expect(within(within(adaptive).getByRole("table", { name: "Kết quả SQL" })).getByText("3.912.548")).toBeInTheDocument();
    const sql = within(adaptive).getByLabelText("Câu SQL");
    expect(sql).not.toBeVisible();
    await user.click(within(adaptive).getByText("Câu SQL"));
    expect(sql).toBeVisible();
    expect(sql).toHaveTextContent("SUM(total_amount_value)");
    expect(within(adaptive).getByText("Chỉ đọc (một câu SELECT)")).toBeVisible();

    const compare = screen.getByRole("table", { name: "So với mốc vanilla" });
    expect(compare).toHaveTextContent("Giống mốc");
    expect(compare).toHaveTextContent("1/1 dòng kết quả giống");
    expect(compare).toHaveTextContent("Nhanh hơn 1,33×");
    // gone: the lead text and the question list repeated under the sample buttons
    expect(screen.queryByText(/Node 2 đổi câu hỏi/)).toBeNull();
    expect(screen.getAllByText(Q1, { exact: false }).filter((el) => el.tagName === "LI")).toHaveLength(0);
  });

  it("marks the rows that differ from the baseline", async () => {
    const ask = vi.fn(async (_q: string, endpointId: string) =>
      endpointId === BASE ? timed(endpointId) : timed(endpointId, { rows: [[3912000]], answer: "Tổng chi tiêu là 3.912.000 đồng." }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
    await screen.findByText("Tổng chi tiêu là 3.912.000 đồng.");
    const adaptive = column("adaptive");
    expect(within(adaptive).getByText("1 dòng khác mốc")).toBeInTheDocument();
    expect(within(adaptive).getByText("3.912.000").closest("tr")).toHaveAttribute("data-diff", "true");
    expect(within(column("vanilla")).getByText("3.912.548").closest("tr")).not.toHaveAttribute("data-diff");
    const compare = screen.getByRole("table", { name: "So với mốc vanilla" });
    expect(compare).toHaveTextContent("Khác mốc");
    expect(compare).toHaveTextContent("1/1 dòng khác");
  });

  it("asks a typed question with Enter", async () => {
    const api = fakeApi();
    open("node2", api);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Câu hỏi"), "Có bao nhiêu hoá đơn?{Enter}");
    expect(api.ask).toHaveBeenCalledWith("Có bao nhiêu hoá đơn?", FAST);
  });

  it("runs only the chosen endpoint when there is no baseline", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [endpoint(FAST, "adaptive")] }));
    const api = fakeApi({ endpoints });
    open("node2", api);
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
    await screen.findByText("Đã trả lời");
    expect(api.ask).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("table", { name: /So với mốc/ })).toBeNull();
  });

  it("keeps the other column when one endpoint fails", async () => {
    const ask = vi.fn(async (_q: string, endpointId: string) => {
      if (endpointId === BASE) throw new ApiError(429, "Too many");
      return timed(endpointId);
    });
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
    expect(await within(await screen.findByRole("article", { name: "vanilla" })).findByText("Đã hỏi quá nhiều trong ít phút, thử lại sau.")).toBeInTheDocument();
    expect(await within(column("adaptive")).findByText("Đã trả lời")).toBeInTheDocument();
  });

  it("says how a text filter was matched against the unaccented data", async () => {
    const ask = vi.fn(async () => question({ text_rewrites: [{ column: "seller_name", from: "%Nhà sách Cẩm Phả%", to: "%nha%sach%cam%pha%" }] }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
    expect((await screen.findAllByText(/Khớp chữ không dấu/))[0]).toHaveTextContent("%Nhà sách Cẩm Phả% → %nha%sach%cam%pha%");
  });

  it("shows a flawed answer as written, with its flags", async () => {
    const ask = vi.fn(async () => question({
      answer: "Tổng tiền là 76.000. beğen", warning: "foreign_text", foreign_words: ["beğen"], ungrounded_numbers: ["76.000"],
    }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q1/ }));
    expect((await screen.findAllByText("Tổng tiền là 76.000. beğen"))[0]).toBeInTheDocument();
    expect(screen.getAllByText(/Chữ không phải tiếng Việt/)[0]).toHaveTextContent("beğen");
    expect(screen.getAllByText(/Số không có trong kết quả SQL/)[0]).toHaveTextContent("76.000");
  });

  it("shows a blocked query with its reason and its SQL open", async () => {
    const ask = vi.fn(async () => question({ status: "error", error: "JOIN_FANOUT: cộng trùng hoá đơn", rows: [], columns: [], repairs: 1 }));
    open("node2", fakeApi({ ask }));
    await userEvent.setup().click(await screen.findByRole("button", { name: /^Q3/ }));
    const adaptive = await screen.findByRole("article", { name: "adaptive" });
    expect(await within(adaptive).findByText("Bị chặn")).toBeInTheDocument();
    expect(within(adaptive).getByRole("alert")).toHaveTextContent("JOIN_FANOUT");
    expect(within(adaptive).getByLabelText("Câu SQL")).toBeVisible();
    expect(screen.getByRole("table", { name: "So với mốc vanilla" })).toHaveTextContent("Cùng bị chặn");
  });

  it("lists the views that can be asked in a side panel", async () => {
    open("node2", fakeApi());
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Dữ liệu có thể hỏi" }));
    const drawer = screen.getByRole("dialog", { name: "Dữ liệu có thể hỏi" });
    expect(drawer).toHaveTextContent("v_invoice_facts");
    expect(drawer).toHaveTextContent("total_amount_value");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("node 3", () => {
  const STORES = "Cửa hàng nào chi nhiều nhất? So sánh các cửa hàng";
  // The baseline (vanilla) writes slower; both pick the same charts unless told otherwise.
  const written = (endpointId: string, extra: Partial<ReportResult> = {}): ReportResult => ({
    ...REPORT,
    report: { ...REPORT.report, metrics: { ttft_ms: 600, tpot_ms: endpointId === BASE ? 20 : 16, completion_tokens: 300 } },
    timings_ms: endpointId === BASE ? { data: 120, dashboard: 1900, report: 7800 } : { data: 120, dashboard: 1500, report: 6000 },
    endpoint: { endpoint_id: endpointId, name: endpointId === BASE ? "vanilla" : "adaptive" },
    ...extra,
  });
  const column = (name: string) => screen.getByRole("article", { name });

  it("runs a sample on the chosen endpoint and the baseline, shows the totals once and each report", async () => {
    const report = vi.fn(async (endpointId: string) => written(endpointId));
    open("node3", fakeApi({ report }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: "25 hoá đơn gần nhất" }));
    await user.click(screen.getByRole("button", { name: "So sánh cửa hàng" }));
    expect(report).toHaveBeenCalledWith(FAST, STORES, 25);
    expect(report).toHaveBeenCalledWith(BASE, STORES, 25);
    const totals = await screen.findByLabelText("Số tổng của phạm vi");
    expect(totals).toHaveTextContent("3.912.548 đ");
    expect(totals).toHaveTextContent("391.255 đ"); // the average is money too
    expect(await screen.findAllByRole("article")).toHaveLength(2);
    expect(screen.getAllByRole("article").map((a) => a.getAttribute("aria-label"))).toEqual(["vanilla", "adaptive"]);
    const adaptive = column("adaptive");
    await within(adaptive).findByText("Đã viết");
    expect(within(adaptive).getByText("10 hoá đơn gần nhất có tổng chi tiêu 3.912.548 đồng.")).toBeInTheDocument();
    expect(within(adaptive).getByText("7,62 s")).toBeInTheDocument(); // data + charts + report
    expect(within(adaptive).getByText("2/2 số khớp bảng số liệu")).toBeInTheDocument();
    // the dashboard of the chosen endpoint, without repeating the totals
    expect(screen.getByRole("img", { name: "Chi tiêu theo ngày" })).toBeInTheDocument();
    expect(screen.getByText("nghi lỗi OCR")).toBeInTheDocument();
    expect(screen.queryByLabelText("Tổng quan 10 hoá đơn")).toBeNull();
    const compare = screen.getByRole("table", { name: "So với mốc vanilla" });
    expect(compare).toHaveTextContent("Giống mốc");
    expect(compare).toHaveTextContent("Số đều khớp");
    expect(compare).toHaveTextContent("Nhanh hơn 1,25×");
    // gone: the lead text, the empty state and the table keys on the charts
    expect(screen.queryByText(/Node 3 nhận yêu cầu/)).toBeNull();
    expect(screen.queryByText("top_items")).toBeNull();
  });

  it("marks a number the report made up, right in the sentence", async () => {
    const report = vi.fn(async (endpointId: string) => (endpointId === BASE ? written(endpointId) : written(endpointId, {
      report: { status: "ok", text: "MINIMART ANAN chi nhiều nhất, 1.270.000 đồng.", metrics: { tpot_ms: 16 } },
      number_check: { numbers: 2, ungrounded: ["1.270.000"] },
    })));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    const adaptive = await screen.findByRole("article", { name: "adaptive" });
    expect(await within(adaptive).findByText("1.270.000", { selector: "mark" })).toBeInTheDocument();
    expect(within(adaptive).getByText(/1 số không có trong bảng/)).toHaveTextContent("1.270.000");
    expect(screen.getByRole("table", { name: "So với mốc vanilla" })).toHaveTextContent("1 số không khớp");
  });

  it("says which charts differ from the baseline and switches the dashboard", async () => {
    const report = vi.fn(async (endpointId: string) => (endpointId === BASE ? written(endpointId) : written(endpointId, {
      dashboard: { status: "ok", error: null, charts: [REPORT.dashboard.charts[0], REPORT.dashboard.charts[1], { table: "taxes", type: "table", value: null, title: "Thuế" }] },
    })));
    open("node3", fakeApi({ report }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    const adaptive = await screen.findByRole("article", { name: "adaptive" });
    expect(await within(adaptive).findByText("Thuế")).toHaveAttribute("data-diff", "true");
    expect(within(adaptive).getByText("Chi tiêu theo ngày")).not.toHaveAttribute("data-diff");
    expect(screen.getByRole("table", { name: "So với mốc vanilla" })).toHaveTextContent("Khác mốc");
    expect(screen.queryByText("nghi lỗi OCR")).toBeNull(); // adaptive's dashboard has no item table
    await user.click(screen.getByRole("radio", { name: "Dashboard của vanilla" }));
    expect(screen.getByText("nghi lỗi OCR")).toBeInTheDocument();
  });

  it("flags foreign words in the report", async () => {
    const report = vi.fn(async (endpointId: string) => written(endpointId, { language_check: { foreign_words: ["城市"] } }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    expect((await screen.findAllByText(/Chữ không phải tiếng Việt/))[0]).toHaveTextContent("城市");
  });

  it("says when the model chose no usable chart and the default set is shown", async () => {
    const report = vi.fn(async (endpointId: string) => written(endpointId, {
      dashboard: { status: "fallback" as const, error: "Model không trả JSON hợp lệ cho dashboard.", charts: [{ table: "by_store", type: "bar" as const, value: "tong_tien", title: "Tổng tiền theo cửa hàng" }] },
    }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    const flag = (await within(await screen.findByRole("article", { name: "adaptive" })).findByText("Không chọn được biểu đồ, dùng bộ mặc định"));
    expect(flag).toHaveAttribute("title", "Model không trả JSON hợp lệ cho dashboard.");
    expect(screen.getByRole("img", { name: "Tổng tiền theo cửa hàng" })).toBeInTheDocument();
  });

  it("labels bars with enough columns to tell rows apart", async () => {
    const invoices = {
      key: "invoices", title: "Danh sách hoá đơn", columns: ["issued_date", "seller_name", "tong_tien", "so_mat_hang"],
      rows: [["2020-08-16", "MINIMART ANAN", 161500, 8], ["2020-08-16", "CUA HANG X", 52000, 2], ["2020-08-15", "MINIMART ANAN", 30000, 1]],
    };
    const report = vi.fn(async (endpointId: string) => written(endpointId, {
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
    const report = vi.fn(async (endpointId: string) => written(endpointId, { scope: { limit: 10, invoice_count: 0, label: "x" }, dashboard: { ...REPORT.dashboard, charts: [] } }));
    open("node3", fakeApi({ report }));
    await userEvent.setup().click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    expect(await screen.findByText(/Phạm vi này chưa có hoá đơn nào/)).toBeInTheDocument();
    expect(screen.queryByRole("article")).toBeNull();
  });

  it("lists the input tables of the last report in a side panel", async () => {
    open("node3", fakeApi());
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Số liệu đầu vào" }));
    expect(screen.getByRole("dialog", { name: "Số liệu đầu vào" })).not.toHaveTextContent(/truy vấn|báo cáo/); // empty before a report
    await user.click(screen.getByRole("button", { name: "Đóng" }));
    await user.click(await screen.findByRole("button", { name: "Tạo báo cáo" }));
    await screen.findByLabelText("Số tổng của phạm vi");
    await user.click(screen.getByRole("button", { name: "Số liệu đầu vào" }));
    const drawer = screen.getByRole("dialog", { name: "Số liệu đầu vào" });
    expect(drawer).toHaveTextContent("Tổng tiền theo cửa hàng");
    expect(drawer).not.toHaveTextContent("truy vấn cố định");
  });
});

describe("Canva review 19/11", () => {
  it("names the app FlexSDS Benchmark and keeps the side bar to the pages and Thoát", async () => {
    open("node1", fakeApi());
    const nav = await screen.findByRole("navigation", { name: "Điều hướng" });
    expect(within(nav).getByText("FlexSDS Benchmark")).toBeInTheDocument();
    expect(within(nav).queryByText(/Console đồ án/)).toBeNull();
    expect(within(nav).queryByText(/Web không bật hay tắt GPU/)).toBeNull();
    expect(within(nav).getByRole("link", { name: /Truy vấn SQL/ })).toBeInTheDocument();
    expect(within(nav).queryByRole("link", { name: /Hỏi bằng SQL/ })).toBeNull();
    expect(within(nav).getAllByRole("button", { name: "Thoát" }).length).toBeGreaterThan(0);
  });

  it("keeps Node 1 to the choices it needs and marks a stopped LLM with a red dot", async () => {
    const endpoints = vi.fn(async () => ({ endpoints: [ENDPOINTS[0], endpoint(FAST, "adaptive", { status: "down", status_message: "HTTP 503" })] }));
    open("node1", fakeApi({ endpoints }));
    expect(await screen.findByText("Chọn ảnh")).toBeInTheDocument();
    expect(screen.getByText("JPEG hoặc PNG, tối đa 10 MB.")).toBeInTheDocument();
    expect(screen.getByText("Lựa chọn LLM")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bắt đầu" })).toBeInTheDocument();
    for (const gone of [/ảnh mẫu trong devset/, /luôn chạy kèm/, /Output của từng endpoint/, /ảnh tải lên hiện trước/, /Chạy trên/]) {
      expect(screen.queryByText(gone)).toBeNull();
    }
    const stopped = screen.getByRole("checkbox", { name: /adaptive/ });
    expect(stopped).toBeDisabled();
    expect(within(stopped).getByText("đã dừng")).toHaveClass("sr-only"); // read aloud, not shown
    expect(stopped.querySelector("i")?.getAttribute("style")).toContain("var(--bad)");
    const running = screen.getByRole("checkbox", { name: /vanilla/ });
    expect(running.querySelector("i")?.getAttribute("style")).not.toContain("var(--bad)");
  });

  it("asks without examples on Node 2 and lists the views without notes", async () => {
    open("node2", fakeApi());
    expect(await screen.findByRole("heading", { name: "Truy vấn SQL" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Hỏi về dữ liệu hoá đơn")).toBeInTheDocument();
    expect(screen.getByText("Lựa chọn LLM")).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Dữ liệu có thể hỏi" }));
    const drawer = screen.getByRole("dialog", { name: "Dữ liệu có thể hỏi" });
    expect(within(drawer).getByText("Các SQL view có thể truy vấn bao gồm")).toBeInTheDocument();
    expect(within(drawer).getByText("v_invoice_facts")).toBeInTheDocument();
    expect(drawer).not.toHaveTextContent(/Một dòng mỗi|bị chặn/);
  });

  it("asks without an example on Node 3", async () => {
    open("node3", fakeApi());
    expect(await screen.findByPlaceholderText("Bạn cần báo cáo gì")).toBeInTheDocument();
    expect(screen.getByText("Lựa chọn LLM")).toBeInTheDocument();
  });
});

describe("measure and compare", () => {
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

  it("starts a quick speed run, says the plan while it runs, then explains each metric", async () => {
    let finish: (run: BenchRun) => void = () => undefined;
    const running: BenchRun = { ...speedRun(), status: "running", progress: { done: 8, total: 32 }, result: {} };
    const bench = vi.fn().mockResolvedValueOnce(running).mockImplementationOnce(() => new Promise<BenchRun>((resolve) => {
      finish = resolve;
    }));
    const benches = vi.fn().mockResolvedValueOnce({ runs: [] }).mockResolvedValue({ runs: [speedRun()] });
    const api = fakeApi({ bench, benches });
    open("result", api);
    const user = userEvent.setup();
    expect(await screen.findByText(/Nhanh: 1, 8 người dùng cùng lúc, 8 request mỗi mức; ước tính khoảng 3 phút/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Bắt đầu đo" }));
    expect(api.startBench).toHaveBeenCalledWith({ kind: "speed", endpoint_ids: [BASE, FAST], workload: "W1", loads: [1, 8], requests: 8, repeats: 1 });
    expect(await screen.findByText(/Kế hoạch: 2 endpoint × 2 mức người dùng \(1, 8\) × 8 request = 32 request/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");

    await act(async () => finish(speedRun()));
    expect(await screen.findByText("đo trực tiếp, có nhiễu")).toBeInTheDocument();
    const about = screen.getByLabelText("Lần đo này đo gì");
    expect(about).toHaveTextContent("Node 1 trích xuất hoá đơn (W1)");
    expect(about).toHaveTextContent("2 mức người dùng cùng lúc (1, 8), mỗi mức 8 request; lấy trung vị (p50). Tổng 32 request.");
    expect(screen.getByRole("tab", { name: "Tốc độ sinh token (TPOT)" })).toHaveAttribute("aria-selected", "true");
    const heavy = screen.getByRole("group", { name: "8 người dùng cùng lúc" });
    expect(heavy).toHaveTextContent("25 ms");
    expect(heavy).toHaveTextContent("chậm hơn 1,14×");
    expect(within(screen.getByRole("group", { name: "1 người dùng cùng lúc" })).getByText("nhanh hơn 1,43×")).toBeInTheDocument();
    // at 8 users even the best endpoint is slower than the baseline: the conclusion says so
    expect(screen.getByText(/mọi endpoint đều chậm hơn mốc; đỡ nhất là adaptive, chậm hơn 1,14×/)).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Thông lượng" }));
    expect(screen.getByText(/Tổng số token cả endpoint sinh ra mỗi giây/)).toBeInTheDocument();
    expect(screen.getAllByText("80 token/s").length).toBeGreaterThan(0);
    // gone: the lead text and the three stat tiles
    expect(screen.queryByText(/Đo cùng một bài trên nhiều endpoint/)).toBeNull();
    expect(screen.queryByText(/Tốt nhất ở 8 người dùng/)).toBeNull();
  });

  it("starts a full run with the six loads of the old preset", async () => {
    const api = fakeApi();
    open("result", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: "Đầy đủ" }));
    expect(screen.getByText(/Đầy đủ: 1, 2, 4, 8, 16, 32 người dùng cùng lúc, 32 request mỗi mức, lặp 3 lần/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Bắt đầu đo" }));
    expect(api.startBench).toHaveBeenCalledWith({ kind: "speed", endpoint_ids: [BASE, FAST], workload: "W1", loads: [1, 2, 4, 8, 16, 32], requests: 32, repeats: 3 });
  });

  it("shows the official campaign numbers apart from live runs", async () => {
    open("result", fakeApi());
    expect(await screen.findByText("số chính thức")).toBeInTheDocument();
    expect(screen.getByText(/Số chính thức chỉ lưu hệ số/)).toBeInTheDocument();
    expect(screen.getByLabelText("Lần đo này đo gì")).toHaveTextContent("GPU L4");
    expect(screen.getByRole("table", { name: "Nhanh gấp mấy lần mốc theo số người dùng" })).toHaveTextContent("0,95×");
    expect(screen.getByRole("img", { name: /Nhanh gấp mấy lần mốc/ })).toBeInTheDocument();
  });

  it("lists earlier runs and opens one", async () => {
    const benches = vi.fn(async () => ({ runs: [speedRun()] }));
    open("result", fakeApi({ benches }));
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Node 1 \(W1\), 1, 8 người dùng/ }));
    expect(screen.getByText("đo trực tiếp, có nhiễu")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "8 người dùng cùng lúc" })).toHaveTextContent("25 ms");
    await user.click(screen.getByRole("radio", { name: "Số chính thức M3-05" }));
    expect(screen.getByText("số chính thức")).toBeInTheDocument();
  });

  it("runs the whole chain per endpoint and compares each node with the baseline", async () => {
    const api = fakeApi({ bench: vi.fn(async () => chainRun()) });
    open("result", api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: "Chạy cả chuỗi 1 → 2 → 3" }));
    await user.click(screen.getByRole("button", { name: "Chạy cả chuỗi" }));
    expect(api.startBench).toHaveBeenCalledWith({ kind: "chain", endpoint_ids: [BASE, FAST], sample_id: "m0_04_001" });
    const quality = await screen.findByRole("table", { name: "Kết quả từng node so với mốc" });
    const fields = leafFields(INVOICE).length;
    const adaptiveRow = within(quality).getByText("adaptive").closest("tr") as HTMLElement;
    expect(adaptiveRow).toHaveTextContent(`${fields}/${fields}`);
    expect(adaptiveRow).toHaveTextContent("6/6");
    expect(adaptiveRow).toHaveTextContent("4/4");
    expect(screen.getByRole("img", { name: /adaptive: Node 1 6 s, Node 2 7 s, Node 3 4,5 s/ })).toBeInTheDocument();
    expect(screen.getByLabelText("Lần đo này đo gì")).toHaveTextContent("OCR ảnh một lần");
    await user.click(screen.getByRole("button", { name: "Xem chi tiết adaptive" }));
    const drawer = screen.getByRole("dialog", { name: "Chi tiết adaptive" });
    expect(within(drawer).getByLabelText("JSON hoá đơn")).toHaveTextContent("MINIMART ANAN");
    await user.click(within(drawer).getByRole("radio", { name: "Node 2, SQL Q1–Q6" }));
    expect(within(drawer).getAllByLabelText("Câu SQL")).toHaveLength(6);
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
