# Báo cáo benchmark đa GPU — flexsds-fast-a10-2611b

- **Chế độ:** `fast` (concurrency `[1, 8, 32]`, 3 warm-up + 5 × 32 request/ô)
- **Phần cứng Modal:** A10
- **3 node (tác vụ):** W1_V3, W2, W3
- **3 làn:** vanilla, static, adaptive
- **Mô hình:** target `Qwen/Qwen2.5-7B-Instruct-AWQ`, draft `duckhoi/qwen2.5-7b-eagle3.1-speculator` (`eagle3`, static `K=3`, adaptive `max_k=4`)
- **Quy tắc tăng tốc:** `speedup_vs_vanilla` luôn tính trên **cùng GPU, cùng workload, cùng concurrency**.

## GPU A10

| Workload | Concurrency | Làn | TPOT p50 (ms) | Throughput (tok/s) | Speedup vs vanilla | α | τ | VRAM peak (MB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| W1_V3 | 1 | vanilla | 17.86 | 55.1 | 1.000 | — | — | 20008.4 |
| W1_V3 | 1 | static | 9.64 | 99.9 | 1.853 | 0.414 | 2.117 | 20192.4 |
| W1_V3 | 1 | adaptive | 10.53 | 91.7 | 1.695 | 0.396 | 2.090 | 20192.4 |
| W1_V3 | 8 | vanilla | 20.97 | 338.2 | 1.000 | — | — | 20008.4 |
| W1_V3 | 8 | static | 17.04 | 426.3 | 1.231 | 0.414 | 2.115 | 20192.4 |
| W1_V3 | 8 | adaptive | 18.46 | 390.2 | 1.136 | 0.414 | 2.112 | 20192.4 |
| W1_V3 | 32 | vanilla | 27.06 | 736.6 | 1.000 | — | — | 20008.4 |
| W1_V3 | 32 | static | 46.56 | 520.7 | 0.581 | 0.414 | 2.117 | 20192.4 |
| W1_V3 | 32 | adaptive | 31.47 | 655.0 | 0.860 | 0.445 | 1.907 | 20192.4 |
| W2 | 1 | vanilla | 16.49 | 57.1 | 1.000 | — | — | 19640.4 |
| W2 | 1 | static | 11.97 | 77.0 | 1.378 | 0.256 | 1.768 | 19676.4 |
| W2 | 1 | adaptive | 12.20 | 75.6 | 1.352 | 0.312 | 1.703 | 19678.4 |
| W2 | 8 | vanilla | 22.28 | 288.2 | 1.000 | — | — | 19916.4 |
| W2 | 8 | static | 17.81 | 365.2 | 1.251 | 0.269 | 1.807 | 20028.4 |
| W2 | 8 | adaptive | 18.30 | 352.9 | 1.218 | 0.326 | 1.663 | 20028.4 |
| W2 | 32 | vanilla | 30.87 | 596.7 | 1.000 | — | — | 19916.4 |
| W2 | 32 | static | 39.19 | 504.4 | 0.788 | 0.265 | 1.795 | 20108.4 |
| W2 | 32 | adaptive | 35.36 | 517.7 | 0.873 | 0.343 | 1.599 | 20104.4 |
| W3 | 1 | vanilla | 16.21 | 61.2 | 1.000 | — | — | 19640.4 |
| W3 | 1 | static | 11.16 | 88.8 | 1.453 | 0.258 | 1.774 | 19676.4 |
| W3 | 1 | adaptive | 12.27 | 80.9 | 1.321 | 0.314 | 1.616 | 19678.4 |
| W3 | 8 | vanilla | 16.80 | 427.7 | 1.000 | — | — | 19872.4 |
| W3 | 8 | static | 11.89 | 589.4 | 1.413 | 0.262 | 1.786 | 19912.4 |
| W3 | 8 | adaptive | 13.61 | 530.5 | 1.234 | 0.344 | 1.584 | 19910.4 |
| W3 | 32 | vanilla | 17.64 | 1195.6 | 1.000 | — | — | 20022.4 |
| W3 | 32 | static | 26.84 | 901.9 | 0.657 | 0.261 | 1.782 | 20066.4 |
| W3 | 32 | adaptive | 23.32 | 994.8 | 0.756 | 0.393 | 1.599 | 20064.4 |

## Ghi chú bộ đếm speculative

Các request dưới đây bị loại khỏi α và τ; TPOT, throughput, độ trễ và nội dung trả lời không bị ảnh hưởng. `grammar_post_stop_overcount` là lỗi đếm của vLLM 0.30 khi có ràng buộc JSON: nháp nằm sau EOS bị tính là invalid nhưng vẫn được verify và tính là accepted, trước khi đầu ra bị cắt ở stop token (`logs/bugs/vllmGrammarPostStopOvercount/`).

| Ô | Request bị loại khỏi α/τ | Lý do |
|---|---|---|
| A10__adaptive__W1_V3__c32 | `req-05a67192-4f84-44a0-b9d7-ad4f062add0f` (r004, accepted 4 / drafted 3) | `grammar_post_stop_overcount` |
| A10__adaptive__W1_V3__c32 | `req-bc8827c1-20d7-48e2-a52f-abec2868f29a` (r005, accepted 3 / drafted 2) | `grammar_post_stop_overcount` |
