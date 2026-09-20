# Pipeline Prompts

This directory stores prompt contracts used by `src/pipeline/`.

## Node 1 Extractor

Prompt:

```text
src/pipeline/prompts/node1_extractor_system.md
```

The frozen prompt version is `node1_extractor.v1`. It includes the complete
nested output shape and asks the model to emit one JSON object matching
`invoice_extraction.v1`, making W1 a suitable best-case workload for
speculative decoding. The request builder feeds only pipeline-owned source
metadata and OCR text into the user message; MC-OCR labels are kept for
audit/debug data and are not included in the model prompt.
