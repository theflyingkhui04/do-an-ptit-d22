# Workload W1_v2 - Node 1 Invoice Extraction on full-receipt OCR text

- Workload ID: `w1_node1_invoice_extraction` (label `W1_V2`)
- Dataset version: `w1.m5_01.v2`
- Sample count: `30`
- Package SHA-256: `cd048f71ba2a94db732a1a61d9d4e1105419998c8db48bb45efa5bf66001a41a`
- Prompt: `node1_extractor.v2`, SHA-256 `4609cddd127fc429aa19d73f824751f43b30353eab95bd3b2a2f98b10587a13a`
- OCR source: `experiments/runs/m5_01/20261108-m501-ocr-a10g/ocr_texts.jsonl`

Same 30 M0-04 samples and request shape as W1 v1, but `raw_text` is the OCR text of
the whole receipt (Qwen2.5-VL, `src/pipeline/ocr.py`) and `max_tokens` is 3072.

Requests need a server with `--max-model-len` >= 8192:
deploy the Modal lanes with `PTIT_MAX_MODEL_LEN=8192`.

Check: `python scripts/check_w1_dataset.py --w1-dir experiments/configs/w1_v2`.
