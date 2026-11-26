# Báo cáo benchmark đa GPU — smoke-a10-w2

- **Chế độ:** `fast` (concurrency `[1, 8, 32]`, 3 warm-up + 5 × 32 request/ô)
- **Phần cứng Modal:** A10
- **3 node (tác vụ):** W2
- **3 làn:** vanilla, static, adaptive
- **Mô hình:** target `Qwen/Qwen2.5-7B-Instruct-AWQ`, draft `duckhoi/qwen2.5-7b-eagle3.1-speculator` (`eagle3`, static `K=3`, adaptive `max_k=4`)
- **Quy tắc tăng tốc:** `speedup_vs_vanilla` luôn tính trên **cùng GPU, cùng workload, cùng concurrency**.

## GPU A10

| Workload | Concurrency | Làn | TPOT p50 (ms) | Throughput (tok/s) | Speedup vs vanilla | α | τ | VRAM peak (MB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| W2 | 1 | vanilla | 16.67 | 56.9 | 1.000 | — | — | 19639.4 |
| W2 | 1 | static | 11.16 | 82.0 | 1.493 | 0.256 | 1.768 | 19675.4 |
| W2 | 1 | adaptive | 12.72 | 73.8 | 1.311 | 0.312 | 1.703 | 19677.4 |
| W2 | 8 | vanilla | 21.06 | 306.4 | 1.000 | — | — | 19915.4 |
| W2 | 8 | static | 16.94 | 373.9 | 1.243 | 0.267 | 1.802 | 20027.4 |
| W2 | 8 | adaptive | 18.04 | 355.0 | 1.167 | 0.325 | 1.670 | 20027.4 |
| W2 | 32 | vanilla | 29.72 | 616.2 | 1.000 | — | — | 19915.4 |
| W2 | 32 | static | 38.48 | 510.0 | 0.772 | 0.264 | 1.791 | 20107.4 |
| W2 | 32 | adaptive | 36.85 | 521.4 | 0.806 | 0.293 | 1.517 | 20103.4 |
