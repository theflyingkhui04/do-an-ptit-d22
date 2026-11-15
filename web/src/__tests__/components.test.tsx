import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarChart, SpeedChart } from "@/components/charts";
import { Trace } from "@/components/ui";

describe("charts and trace keep their labels readable", () => {
  it("shows a whole 'date, store' label beside a horizontal bar", () => {
    render(<BarChart horizontal label="Từng hoá đơn" rows={[["2020-08-16, MINIMART ANAN", 161500], ["2020-08-16, CUA HANG X", 52000]]} />);
    const chart = screen.getByRole("img", { name: "Từng hoá đơn" });
    expect(within(chart).getByText("2020-08-16, MINIMART ANAN")).toBeInTheDocument();
  });

  it("names an endpoint in full at the end of its speed-up line", () => {
    render(<SpeedChart loads={[1, 8]} series={[{ name: "qwen2.5-7b-adaptive", color: "red", values: [1.29, 0.93] }]} />);
    expect(screen.getByText("qwen2.5-7b-adaptive")).toBeInTheDocument();
  });

  it("leaves a very short step unlabeled inside the bar but keeps it in the legend", () => {
    render(<Trace steps={[{ label: "Lấy số liệu", ms: 300, kind: "io" }, { label: "Viết báo cáo", ms: 8000, kind: "n3" }]} />);
    const bar = screen.getByRole("img");
    expect(within(bar).queryByText("Lấy số liệu")).not.toBeInTheDocument();
    expect(within(bar).getByText("Viết báo cáo")).toBeInTheDocument();
    expect(screen.getByText(/Lấy số liệu/)).toHaveTextContent("300 ms");
  });
});
