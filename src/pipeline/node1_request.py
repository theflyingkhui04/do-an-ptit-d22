"""Node 1 chat-completion requests built from OCR text (shared by workload scripts and the web API)."""

from __future__ import annotations

import json

SCHEMA_VERSION = "invoice_extraction.v1"
PROMPT_VERSION = "node1_extractor.v1"


def repair_mojibake(text: str) -> str:
    markers = ("Ã", "Â", "Ä", "Æ", "Ð", "áº", "á»")
    has_c1_control = any("\u0080" <= char <= "\u009f" for char in text)
    if not has_c1_control and not any(marker in text for marker in markers):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    lines = []
    for line in str(text).splitlines():
        cleaned = repair_mojibake(line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def build_node1_input_text(sample: dict) -> str:
    raw_text = normalize_text(sample.get("raw_text"))
    source_metadata = {
        "sample_id": sample.get("sample_id"),
        "img_id": sample.get("img_id"),
        "image_path": sample.get("image_path"),
    }
    return (
        "SOURCE_METADATA:\n"
        f"{json.dumps(source_metadata, ensure_ascii=False)}\n"
        "\n"
        "OCR_TEXT:\n"
        f"{raw_text}\n"
    )


def build_chat_request(
    sample: dict,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict:
    user_content = build_node1_input_text(sample)
    expected_source = {
        "sample_id": sample.get("sample_id"),
        "img_id": sample.get("img_id"),
        "image_path": sample.get("image_path"),
        "raw_text": normalize_text(sample.get("raw_text")),
    }
    return {
        "sample_id": sample.get("sample_id"),
        "img_id": sample.get("img_id"),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "expected_source": expected_source,
        "request": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": {"type": "json_object"},
        },
    }
