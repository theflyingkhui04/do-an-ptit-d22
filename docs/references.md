# Danh mục tài liệu tham khảo

> Ưu tiên nguồn sơ cấp (2023–2025). arXiv ID để tra cứu nhanh. Kiểm chứng qua khảo sát có đối chiếu (deep-research, 2026-07).

## Speculative decoding (hướng chính — C)

- **EAGLE-2** — *Faster Inference of Language Models with Dynamic Draft Trees.* Li et al., EMNLP 2024. `arXiv:2406.16858` — **anchor để tái lập**.
- **EAGLE** — *Speculative Sampling Requires Rethinking Feature Uncertainty.* Li et al., ICML 2024. `arXiv:2401.15077` — nền của EAGLE-2 (nháp ở mức feature).
- **EAGLE-3** — *Scaling up Inference Acceleration via Training-Time Test.* Li et al., 2025. `arXiv:2503.01840` — bản mạnh nhất (speedup tới ~6.5×), training-based.
- **Medusa** — *Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* Cai et al., ICML 2024. `arXiv:2401.10774`.
- **Lookahead Decoding** — *Break the Sequential Dependency of LLM Inference.* Fu et al., ICML 2024. `arXiv:2402.02057` — hướng training-free.
- **LayerSkip** — self-speculative decoding. Elhoushi et al., ACL 2024. `arXiv:2404.16710`.
- **Speculative Decoding (gốc)** — Leviathan et al., ICML 2023. `arXiv:2211.17192`.
- **Spec-Bench** — benchmark trung lập cho speculative decoding. `github.com/hemingkx/Spec-Bench`.

## Nén KV-cache (hướng dự phòng — A)

- **SnapKV** — *LLM Knows What You are Looking for Before Generation.* Li et al., NeurIPS 2024. `arXiv:2404.14469`.
- **KeyDiff** — nén KV training-free, attention-free, tương thích FlashAttention. Qualcomm AI Research, NeurIPS 2025. `arXiv:2504.15364`.
- **Ada-KV** — *Adaptive Budget Allocation for KV Cache Eviction.* Feng et al., NeurIPS 2025. `arXiv:2407.11550`.
- **H2O** — *Heavy-Hitter Oracle.* Zhang et al., NeurIPS 2023. `arXiv:2306.14048`.
- **StreamingLLM** — *Attention Sinks.* Xiao et al., ICLR 2024. `arXiv:2309.17453`.
- **PyramidKV** — Zhang et al., 2024. `arXiv:2406.02069`.
- **SCBench** — *A KV Cache-Centric Analysis of Long-Context Methods.* Li et al., ICLR 2025. `arXiv:2412.10319` — harness so sánh chéo.

## Nén prompt/context (đã cân nhắc — B)

- **LLMLingua-2** — *Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* Pan et al., ACL 2024 Findings. `arXiv:2403.12968`.
- **LongLLMLingua** — nén prompt query-aware. Jiang et al., ACL 2024. `arXiv:2310.06839`.
- **LLMLingua** — Jiang et al., EMNLP 2023. `arXiv:2310.05736`.
- **AttentionRAG** — nén context training-free, attention-guided. Fang et al., 2025. `arXiv:2503.10720`.
- **Prompt Compression Survey** — Li et al., NAACL 2025. `arXiv:2410.12388`.

## Công cụ & hạ tầng

- **vLLM** — engine phục vụ hiệu năng cao, endpoint tương thích OpenAI, hỗ trợ speculative decoding. `github.com/vllm-project/vllm`.
- **Langfuse** — quan trắc/tracing cho ứng dụng LLM (LangChain/LangGraph). `langfuse.com`.
- **Dify**, **LangGraph** — nền tảng orchestration workflow LLM (đích tích hợp).
- **Ollama** / **llama.cpp** — chạy model local (dùng cho làn *vanilla* baseline).

> Các số liệu speedup/nén trích từ paper phần lớn là **tác giả tự báo, best-case**; cần đối chiếu và tự đo lại trên phần cứng của nhóm.
