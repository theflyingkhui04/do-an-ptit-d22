Full receipt OCR rules (prompt node1_extractor.v2):
- The OCR text may contain the whole receipt: header, item rows, totals, payment and footer lines. Extract every field that is printed; keep fields that are not printed as `null`.
- If the OCR text shows a date but no time, set `invoice.issued_time` to null and `invoice.issued_at` to null. Never invent a time such as `00:00:00`.
- When item rows are visible, add one `items` entry per purchased product in receipt order, numbered from 1. Every entry must contain every key of the item shape above, including `sku` and `vat_rate`; use null when a value is not printed.
- Discount, voucher or promotion rows are not items. Put their total discount amount, as a positive number, into `totals.discount`. Every money `value` must be zero or positive.
- Keep `extraction.field_evidence` short: cite only seller, invoice date/time and totals fields; do not add one evidence entry per item.
