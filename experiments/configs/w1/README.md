# Workload W1 - Node 1 Invoice Extraction

- Workload ID: `w1_node1_invoice_extraction`
- Dataset version: `w1.m1_03.v1`
- Sample count: `30`
- Package SHA-256: `e7cba647035f8834920d523ce8f6a70ef68896069ee71d3a862fac2fc42eff66`
- Prompt SHA-256: `3452c77da3a6485c8e92e371f2861acab9ec6424bbe52477a579345a7e90f8d9`

Canonical files for M1-02/M1-03:

- `samples.jsonl`: model-facing sample metadata and OCR text, without evaluator labels.
- `provenance.jsonl`: audit-only upstream provenance and annotation label counts.
- `node1_chat_requests.jsonl`: replayable OpenAI-compatible chat requests.
- `package_manifest.json`: hashes and freeze metadata.

Do not rebuild prompts inside the benchmark harness; consume `node1_chat_requests.jsonl`.
