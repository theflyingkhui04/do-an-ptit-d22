# [M1-02] Tự Động Hóa Benchmark Harness & Quét Concurrency 1→32 trên 2 Làn Cơ Sở

Tài liệu nhật ký kỹ thuật tổng hợp của task M1-02 được lưu tại:
👉 **[M1-02_worklogs.md](./M1-02_worklogs.md)**

### Tài liệu Chuyên đề & Nhật ký Chi tiết:
- Worklog Subtask 3 (Đối soát kỷ luật đo lường & Viết `aggregate_results.py`): 👉 **[Task 3/README.md](./Task%203/README.md)**

### Mã nguồn Thực nghiệm & Trực quan hóa:
- Bộ đo lường tự động: [`experiments/benchmark_harness.py`](../../../experiments/benchmark_harness.py)
- Module tính toán thống kê (Median, Min-Max, Variance): [`experiments/aggregate_results.py`](../../../experiments/aggregate_results.py)
- Script sinh biểu đồ khoa học F1 & F2: [`experiments/plot_f1.py`](../../../experiments/plot_f1.py)
- Bảng số liệu tổng hợp 36 cells: [`experiments/results/summary.csv`](../../../experiments/results/summary.csv)
- Biểu đồ F1 (Throughput & TPOT vs Concurrency): [`experiments/results/figures/F1_tpot_throughput_concurrency.png`](../../../experiments/results/figures/F1_tpot_throughput_concurrency.png)
- Biểu đồ F2 (Speedup vs Concurrency): [`experiments/results/figures/F2_speedup_vs_concurrency.png`](../../../experiments/results/figures/F2_speedup_vs_concurrency.png)
