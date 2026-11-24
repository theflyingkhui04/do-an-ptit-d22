import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarChart, LineChart } from "@/components/charts";
import { Trace } from "@/components/ui";
import { I18nProvider } from "@/lib/i18n";

describe("charts and trace keep their labels readable", () => {
  it("shows a whole 'date, store' label beside a horizontal bar", () => {
    render(<BarChart horizontal label="Từng hóa đơn" rows={[["2020-08-16, MINIMART ANAN", 161500], ["2020-08-16, CUA HANG X", 52000]]} />);
    const chart = screen.getByRole("img", { name: "Từng hóa đơn" });
    expect(within(chart).getByText("2020-08-16, MINIMART ANAN")).toBeInTheDocument();
    expect(chart.tagName).not.toBe("svg");
    expect(within(chart).getByText("161.500 VND")).toBeInTheDocument();
  });

  it("writes each column's value on it and keeps whole dates under the columns", () => {
    render(<BarChart label="Theo ngày" rows={[["2020-08-10", 420000], ["2020-08-13", 1268000]]} />);
    const chart = screen.getByRole("img", { name: "Theo ngày" });
    expect(chart.tagName).not.toBe("svg");
    expect(within(chart).getByText("1,27 tr")).toBeInTheDocument();
    expect(within(chart).getByText("420K")).toBeInTheDocument();
    expect(within(chart).getByText("2020-08-10")).toBeInTheDocument();
  });

  it("formats compact values in English locale when EN is active", () => {
    render(
      <I18nProvider initialLang="en">
        <BarChart label="By day" rows={[["2020-08-10", 420000], ["2020-08-13", 1268000]]} />
      </I18nProvider>,
    );
    const chart = screen.getByRole("img", { name: "By day" });
    expect(within(chart).getByText("1.27M")).toBeInTheDocument();
    expect(within(chart).getByText("420K")).toBeInTheDocument();
  });

  it("names each line at its end with its last value, and labels the 1x reference", () => {
    render(<LineChart label="Hệ số tăng tốc" loads={[1, 8]} format={(v) => `${v}×`} reference={1} referenceLabel="Vanilla (1×)"
      series={[{ name: "#1", color: "red", values: [1.29, 0.93] }]} />);
    const chart = screen.getByRole("img", { name: "Hệ số tăng tốc" });
    expect(within(chart).getByText("#1 0.93×")).toBeInTheDocument();
    expect(within(chart).getByText("Vanilla (1×)")).toBeInTheDocument();
  });

  it("leaves a very short step unlabeled inside the bar but keeps it in the legend", () => {
    render(<Trace steps={[{ label: "Lấy số liệu", ms: 300, kind: "io" }, { label: "Tạo báo cáo", ms: 8000, kind: "n3" }]} />);
    const bar = screen.getByRole("img");
    expect(within(bar).queryByText("Lấy số liệu")).not.toBeInTheDocument();
    expect(within(bar).getByText("Tạo báo cáo")).toBeInTheDocument();
    expect(screen.getByText(/Lấy số liệu/)).toHaveTextContent("300 ms");
  });
});
