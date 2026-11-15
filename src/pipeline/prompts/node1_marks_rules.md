Vietnamese mark rules (prompt node1_extractor.v3):
- The OCR text keeps Vietnamese marks, but the OCR engine sometimes reads a correct letter with a wrong mark, for example "Cám ơn quy khách" for "Cám ơn quý khách" or "Hen gặp lại" for "Hẹn gặp lại".
- In the extracted text fields (`seller.name`, `seller.address`, `buyer.name`, `buyer.address`, `items[].description`, `invoice.payment_method`), write the Vietnamese words with their correct marks.
- Only change marks: never add, remove or replace letters, digits, spaces or words, never translate, and never expand abbreviations. Keep the letter case of the OCR text. When unsure which mark is right, keep the OCR spelling.
- `source.raw_text` and every `extraction.field_evidence[].raw_text` stay verbatim copies of the OCR text, wrong marks included.
