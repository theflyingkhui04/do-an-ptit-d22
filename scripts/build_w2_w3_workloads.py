"""Build frozen W2/W3 workload handoff packages from committed prompt packs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.w1_workload import (  # noqa: E402
    canonical_text_file_sha256,
    compute_package_sha256,
    repo_relative,
    sha256_text,
    write_jsonl,
)


DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SEED = 42
SOURCE_REF = "origin/feat/m1-01-vllm-dual-lane-server"


@dataclass(frozen=True)
class WorkloadSpec:
    key: str
    workload_id: str
    dataset_version: str
    prompt_version: str
    source_prompt_path: Path
    output_dir: Path
    task_name: str
    entropy_profile: str
    max_tokens: int
    system_prompt: str


WORKLOADS = {
    "w2": WorkloadSpec(
        key="w2",
        workload_id="w2_text_to_sql",
        dataset_version="w2.m1_03.v1",
        prompt_version="w2_text_to_sql.v1",
        source_prompt_path=Path("experiments/configs/workloads/w2_text_to_sql.jsonl"),
        output_dir=Path("experiments/configs/w2"),
        task_name="Text-to-SQL over Vietnamese VAT invoice schema",
        entropy_profile="medium",
        max_tokens=96,
        system_prompt=(
            "You are Node 2 Text-to-SQL in the MC-OCR VAT analytics pipeline.\n"
            "Convert the Vietnamese business question into one read-only SQL SELECT query.\n"
            "Use only the schema and table names provided by the user prompt.\n"
            "Return SQL text only: no Markdown fences, no commentary, no explanation.\n"
            "Do not write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE."
        ),
    ),
    "w3": WorkloadSpec(
        key="w3",
        workload_id="w3_free_analysis",
        dataset_version="w3.m1_03.v1",
        prompt_version="w3_free_analysis.v1",
        source_prompt_path=Path("experiments/configs/workloads/w3_free_analysis.jsonl"),
        output_dir=Path("experiments/configs/w3"),
        task_name="Free-form Vietnamese VAT/business analysis",
        entropy_profile="high",
        max_tokens=192,
        system_prompt=(
            "You are Node 3 business-analysis writer in the MC-OCR VAT analytics pipeline.\n"
            "Write concise Vietnamese analytical prose that directly answers the user prompt.\n"
            "Prefer grounded, cautious reasoning and avoid over-claiming facts not provided.\n"
            "Do not return JSON unless the user explicitly asks for JSON."
        ),
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "workload",
        "source_line_number",
        "prompt_sha256",
        "prompt_chars",
        "prompt_words",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prompt = str(row["prompt"])
            writer.writerow(
                {
                    "sample_id": row["id"],
                    "workload": row["workload"],
                    "source_line_number": row["_line_number"],
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_chars": len(prompt),
                    "prompt_words": len(prompt.split()),
                }
            )


def artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: canonical_text_file_sha256(
            path if path.is_absolute() else REPO_ROOT / path,
            line_ending="\n",
        )
        for name, path in paths.items()
    }


def validate_prompt_rows(spec: WorkloadSpec, rows: list[dict[str, Any]]) -> None:
    if len(rows) < 12:
        raise ValueError(f"{spec.key.upper()} requires at least 12 prompts; found {len(rows)}")
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{spec.key.upper()} prompt ids must be unique")
    for row in rows:
        if row.get("workload") != spec.workload_id:
            raise ValueError(f"Wrong workload in {spec.source_prompt_path}:{row.get('_line_number')}")
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"Empty prompt in {spec.source_prompt_path}:{row.get('_line_number')}")


def build_package(spec: WorkloadSpec, model: str, temperature: float, seed: int) -> dict[str, Any]:
    source_path = REPO_ROOT / spec.source_prompt_path
    rows = load_jsonl(source_path)
    validate_prompt_rows(spec, rows)

    output_dir = REPO_ROOT / spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    provenance_path = output_dir / "provenance.jsonl"
    manifest_path = output_dir / "manifest.csv"
    requests_path = output_dir / "chat_requests.jsonl"
    prompt_path = output_dir / "system_prompt.md"
    package_manifest_path = output_dir / "package_manifest.json"
    readme_path = output_dir / "README.md"

    prompt_path.write_text(spec.system_prompt.strip() + "\n", encoding="utf-8", newline="\n")

    sample_rows = [
        {
            "workload_id": spec.workload_id,
            "dataset_version": spec.dataset_version,
            "sample_id": row["id"],
            "prompt": row["prompt"],
            "entropy_profile": spec.entropy_profile,
        }
        for row in rows
    ]
    provenance_rows = [
        {
            "workload_id": spec.workload_id,
            "dataset_version": spec.dataset_version,
            "sample_id": row["id"],
            "source_ref": SOURCE_REF,
            "source_prompt_path": repo_relative(source_path, REPO_ROOT),
            "source_line_number": row["_line_number"],
            "source_prompt_sha256": sha256_text(str(row["prompt"])),
            "model_facing_artifacts_exclude_labels": True,
            "annotation_labels_audit_only": False,
        }
        for row in rows
    ]
    request_rows = [
        {
            "workload_id": spec.workload_id,
            "dataset_version": spec.dataset_version,
            "sample_id": row["id"],
            "prompt_version": spec.prompt_version,
            "request": {
                "model": model,
                "messages": [
                    {"role": "system", "content": spec.system_prompt.strip()},
                    {"role": "user", "content": row["prompt"]},
                ],
                "temperature": temperature,
                "max_tokens": spec.max_tokens,
                "seed": seed,
            },
        }
        for row in rows
    ]

    write_jsonl(samples_path, sample_rows)
    write_jsonl(provenance_path, provenance_rows)
    write_jsonl(requests_path, request_rows)
    write_csv(manifest_path, rows)

    artifacts = {
        "samples_jsonl": samples_path,
        "provenance_jsonl": provenance_path,
        "manifest_csv": manifest_path,
        "chat_requests_jsonl": requests_path,
        "system_prompt": prompt_path,
        "source_prompts_jsonl": source_path,
    }
    file_hashes = artifact_hashes(artifacts)
    metadata = {
        "workload_id": spec.workload_id,
        "dataset_version": spec.dataset_version,
        "prompt_version": spec.prompt_version,
        "task_name": spec.task_name,
        "entropy_profile": spec.entropy_profile,
        "sample_count": len(rows),
        "model": model,
        "temperature": temperature,
        "max_tokens": spec.max_tokens,
        "seed": seed,
        "source_ref": SOURCE_REF,
        "freeze_decision": f"freeze {spec.key.upper()} v1 prompt pack for M1 benchmark handoff",
        "handoff_note": "Consume chat_requests.jsonl directly; concurrency, warm-up, and repeats stay in M1-02 harness.",
    }
    package_sha256 = compute_package_sha256(file_hashes, metadata)
    package_manifest = {
        **metadata,
        "package_sha256": package_sha256,
        "artifacts": {
            name: repo_relative(path if path.is_absolute() else REPO_ROOT / path, REPO_ROOT)
            for name, path in artifacts.items()
        },
        "file_hashes": file_hashes,
    }
    package_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    readme_path.write_text(
        "\n".join(
            [
                f"# Workload {spec.key.upper()} - {spec.task_name}",
                "",
                f"- Workload ID: `{spec.workload_id}`",
                f"- Dataset version: `{spec.dataset_version}`",
                f"- Prompt version: `{spec.prompt_version}`",
                f"- Entropy profile: `{spec.entropy_profile}`",
                f"- Sample count: `{len(rows)}`",
                f"- Package SHA-256: `{package_sha256}`",
                f"- Prompt SHA-256: `{file_hashes['system_prompt']}`",
                f"- Max tokens: `{spec.max_tokens}`",
                "",
                "Canonical files for M1-02/M1-03:",
                "",
                "- `samples.jsonl`: model-facing prompt rows.",
                "- `provenance.jsonl`: audit trail for the prompt source and row hashes.",
                "- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.",
                "- `package_manifest.json`: hashes and freeze metadata.",
                "",
                "Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    return {
        "workload_id": spec.workload_id,
        "dataset_version": spec.dataset_version,
        "sample_count": len(rows),
        "package_sha256": package_sha256,
        "artifacts": package_manifest["artifacts"],
        "file_hashes": file_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [build_package(spec, args.model, args.temperature, args.seed) for spec in WORKLOADS.values()]

    summary = {
        "task": "M1-03",
        "step": "build-w2-w3-workload-packages",
        "status": "PASS",
        "source_ref": SOURCE_REF,
        "workloads": summaries,
    }
    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "m1_03_w2_w3_workload_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
