"""Build Node 1 Extractor inputs and chat-completion requests from MC-OCR samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.node1_request import (  # noqa: E402,F401  (re-exported for the workload scripts)
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_chat_request,
    build_node1_input_text,
    normalize_text,
    repair_mojibake,
)


DEFAULT_SAMPLES_JSONL = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/m0_04_mcocr_devset/node1")
DEFAULT_REPORT_DIR = Path("experiments/results")
DEFAULT_SYSTEM_PROMPT = Path("src/pipeline/prompts/node1_extractor_system.md")
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            records.append(record)
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_text_inputs(input_text_dir: Path, samples: list[dict]) -> None:
    input_text_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample_id = sample.get("sample_id") or f"line_{sample['_line_number']:04d}"
        path = input_text_dir / f"{sample_id}.txt"
        path.write_text(build_node1_input_text(sample), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, default=DEFAULT_SAMPLES_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-text-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    samples = load_jsonl(args.samples_jsonl)
    if not samples:
        raise RuntimeError(f"No samples found in {args.samples_jsonl}")
    invalid_sample_lines = [
        sample["_line_number"]
        for sample in samples
        if not isinstance(sample.get("sample_id"), str) or not sample["sample_id"].strip()
    ]
    if invalid_sample_lines:
        raise ValueError(f"Missing non-empty sample_id at JSONL lines: {invalid_sample_lines}")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    input_text_dir = args.output_dir / "inputs"
    inputs_jsonl = args.output_dir / "node1_inputs.jsonl"
    requests_jsonl = args.output_dir / "node1_chat_requests.jsonl"
    local_summary = args.output_dir / "summary.json"
    tracked_summary = args.report_dir / "m0_04_node1_requests_summary.json"

    input_rows = [
        {
            "sample_id": sample.get("sample_id"),
            "img_id": sample.get("img_id"),
            "image_path": sample.get("image_path"),
            "source_csv": sample.get("source_csv"),
            "node1_input_text": build_node1_input_text(sample),
        }
        for sample in samples
    ]
    request_rows = [
        build_chat_request(
            sample,
            system_prompt,
            args.model,
            args.max_tokens,
            args.temperature,
            args.seed,
        )
        for sample in samples
    ]

    write_jsonl(inputs_jsonl, input_rows)
    request_count = write_jsonl(requests_jsonl, request_rows)
    if not args.no_text_files:
        write_text_inputs(input_text_dir, samples)

    summary = {
        "task": "M0-04",
        "step": "7-build-node1-inputs",
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "samples_jsonl": repo_relative(args.samples_jsonl, repo_root),
        "system_prompt": repo_relative(args.system_prompt, repo_root),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "response_format": "json_object",
        "request_count": request_count,
        "outputs": {
            "inputs_jsonl": repo_relative(inputs_jsonl, repo_root),
            "chat_requests_jsonl": repo_relative(requests_jsonl, repo_root),
            "input_text_dir": repo_relative(input_text_dir, repo_root) if not args.no_text_files else None,
            "local_summary": repo_relative(local_summary, repo_root),
            "tracked_summary": repo_relative(tracked_summary, repo_root),
        },
        "prompt_policy": "MC-OCR labels are not included in the user prompt; they remain audit/debug metadata.",
    }
    local_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tracked_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
