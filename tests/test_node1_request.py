"""[M5-01 web demo] Node 1 request building shared by the workload scripts and the web API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_node1_requests as script

from src.pipeline import node1_request

UPLOAD = {"sample_id": "up_1a2b", "img_id": "up_1a2b", "image_path": "uploads/up_1a2b.jpg",
          "raw_text": "  CỬA HÀNG ANAN \n\n Tổng tiền: 42.000  "}


def test_scripts_reuse_the_shared_functions():
    for name in ("SCHEMA_VERSION", "PROMPT_VERSION", "normalize_text", "build_node1_input_text",
                 "build_chat_request", "repair_mojibake"):
        assert getattr(script, name) is getattr(node1_request, name)


def test_request_for_an_uploaded_image():
    row = node1_request.build_chat_request(UPLOAD, "SYSTEM", "qwen2.5-7b-adaptive", 3072, 0.0, 42)
    assert row["expected_source"] == {"sample_id": "up_1a2b", "img_id": "up_1a2b",
                                      "image_path": "uploads/up_1a2b.jpg",
                                      "raw_text": "CỬA HÀNG ANAN\nTổng tiền: 42.000"}
    request = row["request"]
    assert (request["model"], request["max_tokens"], request["temperature"], request["seed"]) == (
        "qwen2.5-7b-adaptive", 3072, 0.0, 42)
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0] == {"role": "system", "content": "SYSTEM"}
    metadata = {"sample_id": "up_1a2b", "img_id": "up_1a2b", "image_path": "uploads/up_1a2b.jpg"}
    assert request["messages"][1]["content"] == (
        f"SOURCE_METADATA:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        "OCR_TEXT:\nCỬA HÀNG ANAN\nTổng tiền: 42.000\n")
