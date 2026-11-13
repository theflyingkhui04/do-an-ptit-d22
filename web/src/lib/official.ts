// Official campaign numbers shown next to live measurements, so the two are never confused.
// M3-05: experiments/results/m3_sweep_4lanes_summary.json (campaign m305-confirmatory-colab-v1, NVIDIA L4,
//        5 repeats x 32 requests per cell); speed-up = vanilla TPOT p50 / lane TPOT p50.
// M3-06: experiments/runs/m3_06/20261104-m306-e2e-a10g-{vanilla,static}/e2e_summary.json (A10G), medians.

export const OFFICIAL_LOADS = [1, 2, 4, 8, 16, 32];

export const M3_05_SPEEDUP: Record<"W1" | "W2" | "W3", Record<string, number[]>> = {
  W1: {
    static: [1.43, 1.39, 1.38, 1.29, 1.16, 0.95],
    adaptive: [1.45, 1.35, 1.31, 1.2, 1.02, 0.86],
    eagle: [1.41, 1.41, 1.3, 1.12, 0.91, 0.69],
  },
  W2: {
    static: [1.35, 1.32, 1.31, 1.27, 1.18, 0.93],
    adaptive: [1.25, 1.11, 0.87, 0.76, 0.73, 0.67],
    eagle: [1.25, 1.21, 1.17, 1.06, 0.87, 0.65],
  },
  W3: {
    static: [1.02, 1.03, 1.03, 0.99, 0.93, 0.75],
    adaptive: [0.75, 0.71, 0.64, 0.58, 0.52, 0.47],
    eagle: [1.24, 1.22, 1.18, 1.09, 0.93, 0.67],
  },
};

export const M3_05_LABEL = "Số chính thức M3-05: GPU L4, 5 lần đo, mỗi lần 32 yêu cầu; mốc là vanilla.";

export const M3_06_CHAIN = [
  { name: "vanilla", node1: 7955, node2: 7046, node3: 4719 },
  { name: "static k=3", node1: 6170, node2: 7189, node3: 3832 },
];

export const M3_06_LABEL = "Số chính thức M3-06, GPU A10G: trung vị Node 1 trên 30 hoá đơn, tổng 6 câu Q1–Q6, 1 báo cáo.";

export const WORKLOAD_LABELS: Record<string, string> = {
  W1: "Node 1, trích xuất hoá đơn (W1)",
  W2: "Node 2, sinh SQL (W2)",
  W3: "Node 3, viết báo cáo (W3)",
};

export const SAMPLES = [
  { id: "m0_04_001", file: "/samples/m0_04_001.jpg", label: "MINIMART ANAN, 10/08/2020" },
  { id: "m0_04_015", file: "/samples/m0_04_015.jpg", label: "MINIMART ANAN, 16/08/2020" },
  { id: "m0_04_021", file: "/samples/m0_04_021.jpg", label: "Nhà sách Cẩm Phả, 14/08/2020" },
];
