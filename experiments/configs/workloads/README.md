# M1 Benchmark Workload Prompt Sets

This directory stores source prompt sets for M1 benchmark workload packages.

- `w2_text_to_sql.jsonl`: medium-entropy Vietnamese Text-to-SQL prompts over a VAT invoice schema.
- `w3_free_analysis.jsonl`: high-entropy Vietnamese free-form VAT analysis prompts.

W1 is frozen as a Node 1 extraction package at `experiments/configs/w1/`.

Each JSONL row has:

- `id`: stable prompt id.
- `workload`: canonical workload name.
- `prompt`: full user prompt used when building OpenAI-compatible chat requests.

Build W2/W3 packages with:

```bash
python scripts/build_w2_w3_workloads.py
```

The canonical replay artifacts are written to `experiments/configs/w2/chat_requests.jsonl` and `experiments/configs/w3/chat_requests.jsonl`.
