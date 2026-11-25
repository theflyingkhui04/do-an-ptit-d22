"""FlexSDS CLI entrypoint and wizard flow for multi-GPU 3-node 3-lane benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from flexsds.cli.output import print_output as print
    from flexsds.core.hardware import MODAL_GPU_CHOICES, get_modal_gpu_spec
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
    from src.flexsds.core.hardware import MODAL_GPU_CHOICES, get_modal_gpu_spec

from experiments import run_benchmark_campaign as camp


def _default_campaign_id(mode: str, *, preview: bool) -> str:
    if preview:
        return f"flexsds-bench-{mode}-preview"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"flexsds-bench-{mode}-{ts}"


def run_benchmark_command(args: Any) -> int:
    """Execute `flexsds benchmark` subcommand actions safely."""
    action = getattr(args, "bench_action", None) or ("plan" if getattr(args, "dry_run", False) else "plan")
    if getattr(args, "dry_run", False) and action in ("run", "plan"):
        action = "plan"

    if action == "stop":
        from experiments import modal_benchmark_runner as runner

        try:
            runner.stop_app()
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Benchmark stop failed: {exc}")
            return 1
        print("Stopped Modal benchmark app (flexsds-benchmark).")
        return 0

    mode = getattr(args, "mode", "full")
    campaign_id = getattr(args, "campaign_id", None) or _default_campaign_id(
        mode, preview=(action == "plan" or bool(getattr(args, "dry_run", False)))
    )
    try:
        sha = getattr(args, "execution_sha", None) or camp.resolve_git_head()
    except Exception:
        sha = "0" * 40

    try:
        record = camp.plan(
            campaign_id,
            sha,
            mode=mode,
            gpus=getattr(args, "gpus", "all"),
            workloads=getattr(args, "workloads", "all"),
            lanes_list=getattr(args, "lanes", "all"),
            target_model=getattr(args, "target", camp.DEFAULT_TARGET_MODEL),
            draft_model=getattr(args, "draft", camp.DEFAULT_DRAFT_MODEL),
            draft_method=getattr(args, "draft_method", camp.DEFAULT_DRAFT_METHOD),
            static_k=getattr(args, "static_k", camp.DEFAULT_STATIC_K),
            adaptive_max_k=getattr(args, "adaptive_max_k", camp.DEFAULT_ADAPTIVE_MAX_K),
            match_adaptive_runtime=bool(getattr(args, "match_adaptive_runtime", False)),
            profile_dir=getattr(args, "profile_dir", None),
        )
    except ValueError as exc:
        print(f"Invalid benchmark configuration: {exc}")
        return 2

    if action == "plan" or getattr(args, "dry_run", False):
        if getattr(args, "json_output", False):
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print("\n" + camp.format_plan_overview(record) + "\n")
        return 0

    try:
        camp.save_plan(record)
        if action == "run":
            camp.drive(
                record,
                retry_failed=bool(getattr(args, "retry_failed", False)),
                profile_dir=getattr(args, "profile_dir", None),
                auto_calibrate=bool(getattr(args, "auto_calibrate", False)),
                calibrate_mode=getattr(args, "calibrate_mode", "fast"),
            )
            return 0
        if action == "status":
            from experiments import modal_benchmark_runner as runner

            att_root = camp.attempts_dir(record)
            for cell in record["cells"]:
                history = camp.attempts(record, cell)
                state = (
                    "COLLECTED"
                    if camp.selected_attempt(record, cell)
                    else (runner.bench_status(history[-1].name, evidence_root=att_root) if history else "NOT_SUBMITTED")
                )
                print(f"{cell['key']} {state}")
            return 0
        if action == "validate":
            cells = camp.validate_campaign(record)
            print(f"VALID: {len(cells)} cells, {len(cells) * camp.REPEATS * camp.NUM_REQUESTS} measured requests")
            return 0
        if action == "aggregate":
            camp.aggregate(record)
            print(f"Aggregated campaign {campaign_id} successfully.")
            return 0
    except KeyboardInterrupt:
        print("Benchmark interrupted; completed cells and remote Modal calls are preserved.")
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Benchmark {action} failed: {exc}")
        return 1
    return 0


def run_benchmark_wizard() -> dict[str, Any]:
    """Interactive multi-GPU 3-node 3-lane benchmark flow (entered via Measure -> Measure now)."""
    try:
        from flexsds.cli import wizard
    except ModuleNotFoundError:
        from src.flexsds.cli import wizard

    mode_choice = wizard.prompt_choice(
        "Benchmark mode:",
        [
            "Full benchmark (concurrency 1, 2, 4, 8, 16, 32 — 216 cells)",
            "Fast benchmark (concurrency 1, 8, 32 — 108 cells)",
        ],
        default_idx=0,
    )
    mode = "fast" if mode_choice.startswith("Fast benchmark") else "full"

    hw_choice = wizard.prompt_choice(
        "Benchmark hardware:",
        ["All 4 Modal GPUs (T4, A10, L40S, A100)", *MODAL_GPU_CHOICES],
        default_idx=0,
    )
    gpus: tuple[str, ...] = (
        camp.BENCHMARK_GPUS
        if hw_choice.startswith("All 4 Modal GPUs")
        else (get_modal_gpu_spec(hw_choice).name,)
    )

    model_choice = wizard.prompt_choice(
        "Model pair:",
        [
            "Default: Qwen2.5-7B-AWQ + EAGLE-3.1",
            "Custom Hugging Face model",
        ],
        default_idx=0,
    )
    target_model = camp.DEFAULT_TARGET_MODEL
    draft_model = camp.DEFAULT_DRAFT_MODEL
    draft_method = camp.DEFAULT_DRAFT_METHOD
    static_k = camp.DEFAULT_STATIC_K
    if model_choice.startswith("Custom"):
        target_model = wizard.prompt_string("Target model ID or path:", target_model)
        draft_model = wizard.prompt_string("Draft speculator ID or path:", draft_model)
        draft_method = wizard.prompt_choice("Draft method:", ["eagle3", "eagle", "draft_model"], default_idx=0)
        static_k = wizard.prompt_positive_int("Static K (1..8):", static_k)

    try:
        sha = camp.resolve_git_head()
    except Exception:
        sha = "0" * 40
    campaign_id = _default_campaign_id(mode, preview=False)
    record = camp.plan(
        campaign_id,
        sha,
        mode=mode,
        gpus=gpus,
        workloads=camp.BENCHMARK_WORKLOADS,
        lanes_list=camp.BENCHMARK_LANES,
        target_model=target_model,
        draft_model=draft_model,
        draft_method=draft_method,
        static_k=static_k,
    )
    print("\n" + camp.format_plan_overview(record) + "\n")

    missing = [g for g, info in record["profile_readiness"].items() if info["status"] != "ready"]
    actions = ["Preview plan (dry-run)", "Run benchmark now"]
    if missing:
        actions.append("Auto-calibrate missing GPUs & run")
    actions.append("Back")

    selected_action = wizard.prompt_choice("Benchmark action:", actions, default_idx=0)
    if selected_action == "Back":
        return {"action": "profile", "back": True}
    if selected_action.startswith("Preview plan"):
        return {
            "action": "benchmark",
            "mode": mode,
            "gpus": list(gpus),
            "total_cells": record["total_cells"],
            "dry_run": True,
            "back": True,
        }

    auto_calibrate = selected_action.startswith("Auto-calibrate")
    calibrate_mode = "fast"
    if auto_calibrate:
        calibrate_mode = wizard.prompt_measurement_mode()

    try:
        camp.drive(record, auto_calibrate=auto_calibrate, calibrate_mode=calibrate_mode)
    except KeyboardInterrupt:
        print("Benchmark cancelled.")
        return {"action": "abort", "exit_code": 130}
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Benchmark failed: {exc}")
        return {"action": "abort", "exit_code": 1}

    return {
        "action": "benchmark",
        "campaign_id": campaign_id,
        "mode": mode,
        "gpus": list(gpus),
        "total_cells": record["total_cells"],
        "dry_run": False,
    }
