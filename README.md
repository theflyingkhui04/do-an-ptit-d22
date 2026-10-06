# Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference

> **VN:** *Giải mã suy đoán thích ứng phần cứng nhằm tăng tốc luồng trích xuất & phân tích dữ liệu hóa đơn tiếng Việt.*
> Graduation thesis — Posts and Telecommunications Institute of Technology (PTIT), cohort D22.

**Speed up self-hosted open-source LLMs on consumer GPUs — losslessly.** A custom **Adaptive Proposer** plugged into **vLLM** dynamically resizes the EAGLE-2 draft tree at runtime from real-time acceptance rate, hardware profiling and free VRAM — killing the *negative-acceleration* and *out-of-memory* failure modes that make static speculative decoding unusable on 8–12 GB GPUs.

`🟡 Status: design locked, implementation in progress · defense Dec 2026`

---

## TL;DR

- **Problem:** Multi-step LLM workflows (agents, data pipelines) are bottlenecked by **decode latency**, which *compounds* across sequential calls. Static speculative decoding (even vLLM's default) uses a **fixed draft tree** → it *slows down* on hard/high-entropy steps, *under-uses* the GPU on easy steps, and can **OOM** under concurrency on consumer GPUs.
- **Contribution:** A **hardware-aware, adaptive draft-tree controller** integrated into vLLM via its `custom_class` proposer backend — lossless, training-free at serve time, and tuned to the *actual* GPU it runs on.
- **Demonstrated on** a real end-to-end **Vietnamese receipt-analytics pipeline** (MC-OCR dataset) and benchmarked with vLLM `benchmark_serving` + Langfuse.

## The core idea — Adaptive Proposer

```
        ┌───────────────────────────────────────────────┐
        │  Startup: WARM-UP PROFILING                   │
        │  measure t_draft, t_verify, VRAM_free on GPU  │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  Serving: REAL-TIME TELEMETRY                 │
        │  track rolling acceptance α, live VRAM        │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  Decision: DYNAMIC TREE BUDGETING             │
        │  tree = f(α, t_draft, t_verify, VRAM_free)    │
        └───────────────────────────────────────────────┘
```

- **α > 0.8** (easy, low-entropy e.g. JSON extraction) → grow the tree (depth 4, width 8) for max speedup (~3–4×).
- **0.4 ≤ α ≤ 0.8** → balanced tree (depth 3, width 4).
- **α < 0.3 or VRAM tight** (hard, high-entropy e.g. free-form analysis) → shrink to a short chain or **disable SD** → removes negative acceleration and prevents OOM.

## Why it's real work (novelty vs. vLLM built-in)

| | vLLM default (built-in speculative) | This thesis (Adaptive Custom Proposer) |
|---|---|---|
| Draft tree | **Static** for the whole server run | **Dynamic** per step, by measured cost |
| Hardware awareness | ❌ manual static params | ✅ auto profiler tunes to the running GPU |
| Memory safety | ❌ OOM-prone under load | ✅ prunes the tree when VRAM runs low |
| Negative acceleration | ❌ can drop below 1.0× | ✅ falls back to plain decoding — never slower |
| Effort | CLI flags only | Custom controller algorithm + startup hardware profiler + vLLM integration |

> Draft-head training is explicitly **out of scope** — published EAGLE-2 checkpoints are used instead. The contribution is the *controller*, not the draft head ([QĐ-10](docs/03-quyet-dinh-thiet-ke.md)).

## Demo pipeline — Vietnamese receipt analytics (MC-OCR)

```
[Raw VN receipt text — MC-OCR]
        │
        ▼  NODE 1 · EXTRACTOR      → structured JSON (store, items, prices, tax) → SQLite
        ▼  NODE 2 · TEXT-TO-SQL    → Vietnamese question → SQL over the receipts DB
        ▼  NODE 3 · SQL-TO-BI      → analytic report (HTML/Markdown) → Streamlit dashboard
```

E2E latency = Lat(Node1) + Lat(Node2) + Lat(Node3) — a workload where compounding decode latency, and the adaptive controller's per-step tuning, both show up clearly.

## Benchmarking

Measured with vLLM `benchmark_serving` + **Langfuse**, across a concurrency sweep (batch 1 → 32):

- **Acceptance rate (α)** — draft-head quality
- **TTFT** (time to first token) — kept within budget
- **TPOT / ITL** (inter-token latency) — the main optimization target
- **E2E latency** — full 3-stage pipeline
- **VRAM** — proving the memory safety of the Adaptive Proposer

Three lanes are compared on the same hardware: **vanilla autoregressive** · **stock EAGLE-2** · **adaptive**, to build a **use-case map** (when speed optimization is worth enabling — and when it isn't, e.g. large-batch throughput or long-input/short-output).

## Tech stack

Python · **vLLM** (`custom_class` proposer backend) · **EAGLE-2** speculative decoding · PyTorch · **Qwen2.5-7B-Instruct 4-bit (AWQ/GPTQ)** as target on a T4 16GB ([QĐ-09](docs/03-quyet-dinh-thiet-ke.md)) · SQLite · **Langfuse** · OpenAI-compatible `/v1/chat/completions` endpoint (plugs into Dify / LangGraph)

## Repository layout

```
do-an-ptit-d22/
├── README.md
├── requirements.txt
├── docs/
│   ├── README.md                   # ← DOC MAP: what every file is for. Start here.
│   ├── 01-gioi-thieu.md            # Introduction · research questions · scope
│   ├── 02-tong-quan-tai-lieu.md    # Literature survey (3 directions) + rationale
│   ├── 03-quyet-dinh-thiet-ke.md   # Decision log (ADR) + open questions
│   ├── 04-quy-trinh-va-ke-hoach.md # Process + milestone plan + risk register
│   ├── 05-thiet-ke-he-thong.md     # Architecture + the two frozen interfaces
│   ├── 06-ke-hoach-thuc-nghiem.md  # Measurement protocol + result matrix
│   ├── 07-thuat-ngu.md             # Glossary & notation
│   ├── 08-outline-quyen.md         # Thesis outline + defense Q&A prep
│   ├── paper_summary/              # Deep summaries of the 6 foundational papers
│   └── references.md               # Papers (with arXiv IDs)
├── src/                            # inference engine + adaptive proposer + endpoint
└── experiments/                    # eval workflows + measurement pipeline + analysis
```

**New here?** Read [docs/README.md](docs/README.md) — it maps every document and tells you which ones to read for your role.

## Team

Nhóm áp dụng mô hình **Ghép cặp Luân phiên 70/30 (Pair-Rotation System)**: Mỗi thành viên chủ trì (Lead) 70% mảng chuyên môn cốt lõi, đồng thời luân phiên đồng hành (Partner) 30% ở các mảng còn lại để cả 3 người đều nắm chắc toàn diện hệ thống khi bảo vệ đồ án.

| Chuyên môn cốt lõi (Lead 70%) | Thành viên | Vai trò |
|---|---|---|
| AI Systems & PM (Core-Engine, vLLM) | Cấn Đức Khôi ([GitHub](https://github.com/theflyingkhui04) · | Dev + PM |
| MLOps & Infrastructure (Benchmark Harness, GPU T4) | Bùi Đức Đại | Dev + IT |
| AI Application & Data (Pipeline MC-OCR, VAT Invoices) | Dương Nguyễn Thế | Dev + BA |
| Advisor | *(Cập nhật theo GVHD)* | GVHD |

## Status

🟡 **Design locked; implementation in progress** (thesis defense Dec 2026). Minimal-viable fallback if draft-head training is delayed: demonstrate the adaptive tree on a published EAGLE-2 checkpoint + full benchmark suite — still a complete, measurable contribution.
