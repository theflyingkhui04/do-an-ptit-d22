"""Promote the tracked M0-04 MC-OCR devset into frozen Workload W1 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_node1_requests import (  # noqa: E402
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_chat_request,
    build_node1_input_text,
    load_jsonl,
    normalize_text,
)
from src.pipeline.w1_workload import (  # noqa: E402
    W1_DATASET_VERSION,
    W1_WORKLOAD_ID,
    compute_package_sha256,
    file_sha256,
    repo_relative,
    sha256_json,
    sha256_text,
    write_jsonl,
)


DEFAULT_SOURCE_SAMPLES = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
DEFAULT_SOURCE_MANIFEST = Path("experiments/results/m0_04_mcocr_devset_manifest.csv")
DEFAULT_SYSTEM_PROMPT = Path("src/pipeline/prompts/node1_extractor_system.md")
DEFAULT_OUTPUT_DIR = Path("experiments/configs/w1")
DEFAULT_REPORT_DIR = Path("experiments/results")
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def make_sample_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_DATASET_VERSION,
        "sample_id": sample["sample_id"],
        "source_row_idx": sample.get("source_row_idx"),
        "img_id": sample.get("img_id"),
        "image_path": sample.get("image_path"),
        "source_image_path": sample.get("source_image_path"),
        "source_csv": sample.get("source_csv"),
        "anno_image_quality": sample.get("anno_image_quality"),
        "raw_text": normalize_text(sample.get("raw_text")),
    }


def label_counter(sample: dict[str, Any]) -> Counter[str]:
    labels = sample.get("labels")
    if isinstance(labels, list):
        return Counter(str(label) for label in labels)
    pairs = sample.get("key_text_pairs")
    if isinstance(pairs, list):
        return Counter(str(pair.get("label")) for pair in pairs if isinstance(pair, dict))
    return Counter()


def make_provenance_row(sample: dict[str, Any], source_samples_path: Path) -> dict[str, Any]:
    original = {key: value for key, value in sample.items() if key != "_line_number"}
    counts = label_counter(sample)
    return {
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_DATASET_VERSION,
        "sample_id": sample["sample_id"],
        "upstream_task": "M0-04",
        "upstream_samples_jsonl": source_samples_path.as_posix(),
        "upstream_jsonl_line_number": sample.get("_line_number"),
        "upstream_sample_sha256": sha256_json(original),
        "source_row_idx": sample.get("source_row_idx"),
        "source_csv": sample.get("source_csv"),
        "source_image_path": sample.get("source_image_path"),
        "tracked_image_path": sample.get("image_path"),
        "annotation_label_counts": dict(sorted(counts.items())),
        "annotation_labels_audit_only": True,
        "model_facing_artifacts_exclude_labels": True,
    }


def write_manifest(path: Path, sample_rows: list[dict[str, Any]], provenance_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "img_id",
        "source_row_idx",
        "image_path",
        "source_csv",
        "anno_image_quality",
        "raw_text_sha256",
        "label_types",
        "label_count",
    ]
    provenance_by_id = {row["sample_id"]: row for row in provenance_rows}
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in sample_rows:
            provenance = provenance_by_id[sample["sample_id"]]
            label_counts = provenance["annotation_label_counts"]
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "img_id": sample["img_id"],
                    "source_row_idx": sample["source_row_idx"],
                    "image_path": sample["image_path"],
                    "source_csv": sample["source_csv"],
                    "anno_image_quality": sample["anno_image_quality"],
                    "raw_text_sha256": sha256_text(sample["raw_text"]),
                    "label_types": "|".join(sorted(label_counts)),
                    "label_count": sum(label_counts.values()),
                }
            )


def read_git_sha(ref: str) -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-samples", type=Path, default=DEFAULT_SOURCE_SAMPLES)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.source_samples)
    if len(samples) < 30:
        raise ValueError(f"W1 requires at least 30 samples; found {len(samples)}")

    sample_rows = [make_sample_row(sample) for sample in samples]
    sample_ids = [sample["sample_id"] for sample in sample_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("W1 sample_id values must be unique")

    provenance_rows = [make_provenance_row(sample, args.source_samples) for sample in samples]
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    samples_path = args.output_dir / "samples.jsonl"
    provenance_path = args.output_dir / "provenance.jsonl"
    manifest_path = args.output_dir / "manifest.csv"
    requests_path = args.output_dir / "node1_chat_requests.jsonl"
    inputs_path = args.output_dir / "node1_inputs.jsonl"
    prompt_path = args.output_dir / "system_prompt.md"
    package_manifest_path = args.output_dir / "package_manifest.json"
    readme_path = args.output_dir / "README.md"

    prompt_path.write_text(system_prompt + "\n", encoding="utf-8", newline="\n")
    write_jsonl(samples_path, sample_rows)
    write_jsonl(provenance_path, provenance_rows)
    write_manifest(manifest_path, sample_rows, provenance_rows)

    input_rows = [
        {
            "workload_id": W1_WORKLOAD_ID,
            "dataset_version": W1_DATASET_VERSION,
            "sample_id": sample["sample_id"],
            "img_id": sample["img_id"],
            "image_path": sample["image_path"],
            "node1_input_text": build_node1_input_text(sample),
        }
        for sample in sample_rows
    ]
    request_rows = [
        {
            "workload_id": W1_WORKLOAD_ID,
            "dataset_version": W1_DATASET_VERSION,
            **build_chat_request(
                sample,
                system_prompt,
                args.model,
                args.max_tokens,
                args.temperature,
                args.seed,
            ),
        }
        for sample in sample_rows
    ]
    write_jsonl(inputs_path, input_rows)
    write_jsonl(requests_path, request_rows)

    label_totals: Counter[str] = Counter()
    for provenance in provenance_rows:
        label_totals.update(provenance["annotation_label_counts"])

    artifact_paths = {
        "samples_jsonl": samples_path,
        "provenance_jsonl": provenance_path,
        "manifest_csv": manifest_path,
        "node1_inputs_jsonl": inputs_path,
        "node1_chat_requests_jsonl": requests_path,
        "system_prompt": prompt_path,
        "json_schema": Path("src/pipeline/schemas/invoice_extraction.schema.json"),
    }
    file_hashes = {
        name: file_sha256(path if path.is_absolute() else REPO_ROOT / path)
        for name, path in artifact_paths.items()
    }
    metadata = {
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "sample_count": len(sample_rows),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "response_format": {"type": "json_object"},
        "upstream_m0_04_ref": "origin/feature/m0-04-vat-invoice-schema-mc-ocr",
        "upstream_m0_04_commit": read_git_sha("origin/feature/m0-04-vat-invoice-schema-mc-ocr"),
        "label_totals_audit_only": dict(sorted(label_totals.items())),
        "no_label_leakage_policy": (
            "samples.jsonl and node1_chat_requests.jsonl exclude labels/key_text_pairs; "
            "annotation labels are kept only in provenance.jsonl for audit."
        ),
        "freeze_decision": "freeze W1 v1 from the tracked 30-sample M0-04 devset",
        "coverage_note": (
            "M0-04 provides seller/address/timestamp/total evidence for all samples; "
            "buyer/items/VAT fields remain nullable and should not be over-claimed."
        ),
    }
    package_sha256 = compute_package_sha256(file_hashes, metadata)
    package_manifest = {
        **metadata,
        "package_sha256": package_sha256,
        "artifacts": {
            name: repo_relative(path if path.is_absolute() else REPO_ROOT / path, REPO_ROOT)
            for name, path in artifact_paths.items()
        },
        "file_hashes": file_hashes,
        "source_artifacts": {
            "source_samples": repo_relative(args.source_samples, REPO_ROOT),
            "source_manifest": repo_relative(args.source_manifest, REPO_ROOT),
            "source_system_prompt": repo_relative(args.system_prompt, REPO_ROOT),
        },
    }
    package_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    readme_path.write_text(
        "\n".join(
            [
                "# Workload W1 - Node 1 Invoice Extraction",
                "",
                f"- Workload ID: `{W1_WORKLOAD_ID}`",
                f"- Dataset version: `{W1_DATASET_VERSION}`",
                f"- Sample count: `{len(sample_rows)}`",
                f"- Package SHA-256: `{package_sha256}`",
                f"- Prompt SHA-256: `{file_hashes['system_prompt']}`",
                "",
                "Canonical files for M1-02/M1-03:",
                "",
                "- `samples.jsonl`: model-facing sample metadata and OCR text, without evaluator labels.",
                "- `provenance.jsonl`: audit-only upstream provenance and annotation label counts.",
                "- `node1_chat_requests.jsonl`: replayable OpenAI-compatible chat requests.",
                "- `package_manifest.json`: hashes and freeze metadata.",
                "",
                "Do not rebuild prompts inside the benchmark harness; consume `node1_chat_requests.jsonl`.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    dataset_summary = {
        "task": "M1-03",
        "step": "build-w1-workload",
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_DATASET_VERSION,
        "sample_count": len(sample_rows),
        "minimum_required_samples": 30,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "package_sha256": package_sha256,
        "label_totals_audit_only": dict(sorted(label_totals.items())),
        "artifacts": package_manifest["artifacts"],
        "upstream_m0_04_commit": metadata["upstream_m0_04_commit"],
        "freeze_decision": metadata["freeze_decision"],
        "coverage_note": metadata["coverage_note"],
    }
    request_summary = {
        "task": "M1-03",
        "step": "build-w1-node1-requests",
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": file_hashes["system_prompt"],
        "request_count": len(request_rows),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "response_format": {"type": "json_object"},
        "requests_jsonl": repo_relative(requests_path, REPO_ROOT),
        "package_sha256": package_sha256,
        "prompt_policy": metadata["no_label_leakage_policy"],
    }
    (args.report_dir / "m1_03_w1_dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.report_dir / "m1_03_node1_requests_summary.json").write_text(
        json.dumps(request_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
