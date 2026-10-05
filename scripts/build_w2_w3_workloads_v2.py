"""Build PM-approved W2/W3 v2 benchmark workload packages.

W2 v2 decision:
- pipeline schema-aligned Text-to-SQL over SQLite stable views only;
- diverse SQL patterns while keeping benchmark output SQL-only.

W3 v2 decision:
- Node 3 business-analysis only;
- diverse Vietnamese VAT/business scenarios;
- no thesis/speculative-decoding meta prompts.

The frozen v1 packages are left untouched. This script writes new packages to
``experiments/configs/w2_v2`` and ``experiments/configs/w3_v2``.
"""

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
DEFAULT_TEMPERATURE = 0.0
PM_DECISION = "PM-approved W2-C pipeline-aligned diverse SQL + W3-D business-analysis-only diverse scenarios"


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
    metadata_fields: tuple[str, ...]


W2_SYSTEM_PROMPT = """You are Node 2 Text-to-SQL in the MC-OCR VAT analytics pipeline.
Convert the Vietnamese business question into exactly one read-only SQLite SELECT query.
Use only these stable read-only views and columns:
- v_invoice_facts(invoice_id, run_id, sample_id, img_id, document_type, invoice_number, invoice_series, issued_at, issued_date, issued_time, seller_name, seller_address, seller_tax_code, seller_phone, buyer_name, buyer_address, buyer_tax_code, buyer_phone, currency, payment_method, subtotal_value, subtotal_currency, discount_value, discount_currency, vat_amount_value, vat_amount_currency, total_amount_value, total_amount_currency, extraction_status, extraction_confidence, item_count, tax_line_count)
- v_item_facts(invoice_id, run_id, sample_id, img_id, issued_date, seller_name, line_number, description, sku, quantity_value, quantity_unit, unit_price_value, unit_price_currency, line_total_value, line_total_currency, vat_rate)
- v_tax_facts(invoice_id, run_id, sample_id, img_id, issued_date, seller_name, buyer_name, tax_name, tax_rate, taxable_amount_value, taxable_amount_currency, tax_amount_value, tax_amount_currency)
Do not query base tables or raw_json.
Return SQL text only: no Markdown fences, no commentary, no explanation.
Do not write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, PRAGMA, ATTACH, DETACH, or TRUNCATE."""

W3_SYSTEM_PROMPT = """You are Node 3 business-analysis writer in the MC-OCR VAT analytics pipeline.
Write concise Vietnamese analytical prose that directly answers the business scenario.
Use only facts supplied by the user; separate observations from hypotheses and recommendations when relevant.
Prefer grounded, cautious reasoning and explicitly note important uncertainty or missing data.
Do not discuss the thesis, benchmark design, speculative decoding, or model performance unless the business scenario explicitly contains such information.
Do not return JSON unless the user explicitly asks for JSON."""


WORKLOADS = {
    "w2": WorkloadSpec(
        key="w2",
        workload_id="w2_text_to_sql",
        dataset_version="w2.m1_03.v2",
        prompt_version="w2_text_to_sql.v2",
        source_prompt_path=Path("experiments/configs/workloads/w2_text_to_sql_v2.jsonl"),
        output_dir=Path("experiments/configs/w2_v2"),
        task_name="Pipeline-aligned Vietnamese VAT Text-to-SQL over SQLite stable views",
        entropy_profile="medium",
        max_tokens=96,
        system_prompt=W2_SYSTEM_PROMPT,
        metadata_fields=("sql_features", "target_views"),
    ),
    "w3": WorkloadSpec(
        key="w3",
        workload_id="w3_free_analysis",
        dataset_version="w3.m1_03.v2",
        prompt_version="w3_business_analysis.v2",
        source_prompt_path=Path("experiments/configs/workloads/w3_business_analysis_v2.jsonl"),
        output_dir=Path("experiments/configs/w3_v2"),
        task_name="Vietnamese VAT business-analysis free-form generation",
        entropy_profile="high",
        # PM-approved before the first v2 freeze after diagnostic evidence:
        # all 12/12 W3 v2 prompts hit the 192-token cap with finish_reason=length.
        # Freeze v2 with enough headroom for natural business-analysis completion.
        max_tokens=768,
        system_prompt=W3_SYSTEM_PROMPT,
        metadata_fields=("scenario_category",),
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], spec: WorkloadSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "workload",
        "source_line_number",
        "prompt_sha256",
        "prompt_chars",
        "prompt_words",
        *spec.metadata_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prompt = str(row["prompt"])
            record: dict[str, Any] = {
                "sample_id": row["id"],
                "workload": row["workload"],
                "source_line_number": row["_line_number"],
                "prompt_sha256": sha256_text(prompt),
                "prompt_chars": len(prompt),
                "prompt_words": len(prompt.split()),
            }
            for field in spec.metadata_fields:
                value = row.get(field)
                if isinstance(value, list):
                    value = "|".join(str(item) for item in value)
                record[field] = value
            writer.writerow(record)


def artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: canonical_text_file_sha256(
            path if path.is_absolute() else REPO_ROOT / path,
            line_ending="\n",
        )
        for name, path in paths.items()
    }


def validate_prompt_rows(spec: WorkloadSpec, rows: list[dict[str, Any]]) -> None:
    if len(rows) != 12:
        raise ValueError(f"{spec.key.upper()} v2 requires exactly 12 prompts; found {len(rows)}")

    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{spec.key.upper()} prompt ids must be unique")

    for row in rows:
        line = row.get("_line_number")
        if row.get("workload") != spec.workload_id:
            raise ValueError(f"Wrong workload in {spec.source_prompt_path}:{line}")
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"Empty prompt in {spec.source_prompt_path}:{line}")
        for field in spec.metadata_fields:
            if field not in row:
                raise ValueError(f"Missing {field!r} in {spec.source_prompt_path}:{line}")

    if spec.key == "w2":
        all_text = "\n".join(str(row["prompt"]) for row in rows).lower()
        forbidden = ("postgresql", " sellers", " buyers", " invoices", " invoice_items", " invoice_taxes")
        for token in forbidden:
            if token in all_text:
                raise ValueError(f"W2 v2 prompt pack contains forbidden legacy schema token: {token!r}")

        expected_views = {"v_invoice_facts", "v_item_facts", "v_tax_facts"}
        covered_views = {
            view
            for row in rows
            for view in row.get("target_views", [])
        }
        if covered_views != expected_views:
            raise ValueError(
                "W2 v2 must cover exactly the three stable views; "
                f"got {sorted(covered_views)}"
            )

        covered_features = {
            feature
            for row in rows
            for feature in row.get("sql_features", [])
        }
        required_features = {
            "filter",
            "aggregate",
            "group-by",
            "order-by",
            "top-k",
            "having",
            "join",
            "cte",
            "count-distinct",
            "ratio",
            "multi-condition",
        }
        missing = required_features - covered_features
        if missing:
            raise ValueError(f"W2 v2 SQL pattern coverage missing: {sorted(missing)}")

    if spec.key == "w3":
        categories = {str(row["scenario_category"]) for row in rows}
        required_categories = {
            "trend-summary",
            "concentration",
            "product-mix",
            "anomaly-risk",
            "compliance-risk",
            "data-quality",
            "reconciliation",
            "audit-prioritization",
            "dashboard-decision",
            "decision-recommendation",
        }
        missing = required_categories - categories
        if missing:
            raise ValueError(f"W3 v2 scenario coverage missing: {sorted(missing)}")

        banned = (
            "speculative decoding",
            "vanilla",
            "static draft",
            "benchmark",
            "luận văn",
            "đồ án",
        )
        for row in rows:
            prompt_lower = str(row["prompt"]).lower()
            hit = [term for term in banned if term in prompt_lower]
            if hit:
                raise ValueError(
                    f"W3 v2 must be business-analysis-only; {row['id']} contains {hit}"
                )


def _row_metadata(row: dict[str, Any], spec: WorkloadSpec) -> dict[str, Any]:
    return {field: row[field] for field in spec.metadata_fields}


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
            **_row_metadata(row, spec),
        }
        for row in rows
    ]

    provenance_rows = [
        {
            "workload_id": spec.workload_id,
            "dataset_version": spec.dataset_version,
            "sample_id": row["id"],
            "decision": PM_DECISION,
            "source_prompt_path": repo_relative(source_path, REPO_ROOT),
            "source_line_number": row["_line_number"],
            "source_prompt_sha256": sha256_text(str(row["prompt"])),
            "model_facing_artifacts_exclude_labels": True,
            "annotation_labels_audit_only": False,
            **_row_metadata(row, spec),
        }
        for row in rows
    ]

    request_rows = [
        {
            "workload_id": spec.workload_id,
            "dataset_version": spec.dataset_version,
            "sample_id": row["id"],
            "prompt_version": spec.prompt_version,
            **_row_metadata(row, spec),
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
    write_csv(manifest_path, rows, spec)

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
        "decision": PM_DECISION,
        "freeze_decision": f"freeze {spec.key.upper()} v2 after PM-approved workload redesign",
        "handoff_note": "Consume chat_requests.jsonl directly; concurrency, warm-up, repeats, generation-config, and sample reuse remain M1-02 harness/server protocol.",
    }
    if spec.key == "w2":
        metadata["benchmark_contract_note"] = (
            "Schema/task aligned with Node 2 stable SQLite views; benchmark response remains SQL-only to preserve the medium-entropy Text-to-SQL workload boundary."
        )
    if spec.key == "w3":
        metadata["benchmark_contract_note"] = (
            "Node 3 business-analysis only; diverse VAT/business scenarios; v2 freezes max_tokens=768 after diagnostics showed 12/12 prompts truncated at 192 and 5/12 still truncated at 512; a 1024 diagnostic probe observed natural stops up to 681 tokens."
        )

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

    readme_lines = [
        f"# Workload {spec.key.upper()} v2 - {spec.task_name}",
        "",
        f"- Workload ID: `{spec.workload_id}`",
        f"- Dataset version: `{spec.dataset_version}`",
        f"- Prompt version: `{spec.prompt_version}`",
        f"- Entropy profile: `{spec.entropy_profile}`",
        f"- Sample count: `{len(rows)}`",
        f"- Package SHA-256: `{package_sha256}`",
        f"- Prompt SHA-256: `{file_hashes['system_prompt']}`",
        f"- Max tokens: `{spec.max_tokens}`",
        f"- Decision: `{PM_DECISION}`",
        "",
        "The frozen v1 package remains unchanged. This directory is a new v2 package.",
        "",
        "Canonical files:",
        "",
        "- `samples.jsonl`: model-facing prompt rows and scenario/query-pattern metadata.",
        "- `provenance.jsonl`: decision/source audit trail and prompt hashes.",
        "- `chat_requests.jsonl`: replayable OpenAI-compatible chat requests.",
        "- `package_manifest.json`: hashes and freeze metadata.",
        "",
        "Do not rebuild prompts inside the benchmark harness; consume `chat_requests.jsonl`.",
    ]
    if spec.key == "w2":
        readme_lines += [
            "",
            "W2 v2 rules:",
            "- SQLite only; no PostgreSQL dialect request.",
            "- Read-only stable views only: `v_invoice_facts`, `v_item_facts`, `v_tax_facts`.",
            "- SQL-only benchmark response to keep W2 semi-structured/medium-entropy.",
            "- Prompt pack covers filters, aggregates, GROUP BY, ORDER BY/TOP-K, HAVING, CTE/join, count-distinct, ratio, and multi-condition queries.",
        ]
    if spec.key == "w3":
        readme_lines += [
            "",
            "W3 v2 rules:",
            "- Business-analysis scenarios only; no thesis/speculative-decoding/meta-benchmark prompts.",
            "- Diverse scenarios cover trends, concentration, product mix, anomaly/risk, compliance, data quality, reconciliation, audit prioritization, dashboards, and recommendations.",
            "- v2 freezes `max_tokens=768`, PM-approved after diagnostics showed 12/12 truncation at 192, 5/12 truncation at 512, and a diagnostic-only 1024 probe observed natural completions up to 681 tokens.",
        ]

    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8", newline="\n")

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
    parser.add_argument("--workload", choices=("w2", "w3", "all"), default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = WORKLOADS.values() if args.workload == "all" else (WORKLOADS[args.workload],)
    summaries = [build_package(spec, args.model, args.temperature, args.seed) for spec in selected]

    summary = {
        "task": "M1-03/M1-02 workload contract update",
        "step": "build-w2-w3-v2-packages",
        "status": "PASS",
        "decision": PM_DECISION,
        "workloads": summaries,
    }
    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "m1_w2_w3_v2_workload_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
