# Workload W1_v3 - Node 1 on library OCR text without Vietnamese marks

- Workload ID: `w1_node1_invoice_extraction` (label `W1_V3`)
- Dataset version: `w1.m5_01.v3`
- Sample count: `30`
- Package SHA-256: `5e22c3cbdfd1252181d913480caad79fea1e9dc4323c6f16d67f4066ba245005`
- Prompt: `node1_extractor.v2`, SHA-256 `4609cddd127fc429aa19d73f824751f43b30353eab95bd3b2a2f98b10587a13a`
- OCR: engine A (PaddleOCR text detection + VietOCR vgg_transformer recognition), source `experiments/runs/m5_01/20261108-m501-ocrlib-choice/ocr_lib_A.jsonl`
- Marks: strip_vietnamese_marks (src/pipeline/text_normalize.py): NFD, drop combining marks, đ->d, Đ->D, NFC

Requests need a server with `--max-model-len` >= 8192.
Check: `python scripts/check_w1_dataset.py --w1-dir experiments/configs/w1_v3`.
