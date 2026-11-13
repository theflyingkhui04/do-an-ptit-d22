"""Local preparation and USER-RUN A100 diagnostics/campaign. No import side effects.

All inference actions require exact clean source, real A100, pinned runtime and
a validated measured profile. The sweep re-evaluates raw diagnostic evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from experiments.m4_a100_profile import hardware_gate, validate_profile_file, inspect_environment
from experiments.m4_git_evidence import publish, copy_immutable, git_auth
from experiments.m4_lossless import compare_captures
from experiments.m4_protocol import (ROOT, WORKLOADS, build_manifest, git, sha256, source_gate,
                                      validate_manifest, validate_working_workloads, write_new)
from experiments.m4_readiness import evaluate, read_lines
from experiments.m4_runtime import serving, http_json
from experiments.m4_treatments import (TREATMENTS, TARGET, TARGET_REVISION, SPECULATOR,
                                       SPECULATOR_REVISION, definition)

STATES = ("PENDING", "RUNNING", "COMPLETE_VALID", "FAILED", "OOM", "TIMEOUT", "INVALID")


def environment(source_commit):
    source_gate(source_commit)
    import platform
    env = dict(inspect_environment(), python_version=platform.python_version(),
               source_commit=source_commit, execution_backend="google_colab",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if not Path("/content").is_dir() or "google.colab" not in sys.modules and not os.environ.get("COLAB_RELEASE_TAG"):
        # Normal subprocesses need not import google.colab; Colab sets its release tag.
        raise ValueError("Canonical runtime must execute in Google Colab")
    hardware_gate(env)
    return env


def profile_gate(path: Path, source_commit, env=None):
    profile = validate_profile_file(path, source_commit)
    if env is not None:
        for field in ("gpu", "gpu_compute_capability", "gpu_memory_total_mb", "driver_version",
                      "torch_version", "cuda_version", "vllm_version"):
            if profile["metadata"].get(field) != env.get(field):
                raise ValueError(f"Profile does not describe current runtime: {field}")
    return sha256(path.read_bytes())


def model_gate():
    from huggingface_hub import HfApi, snapshot_download
    result = []
    for model, revision in ((TARGET, TARGET_REVISION), (SPECULATOR, SPECULATOR_REVISION)):
        actual = HfApi().model_info(model, revision=revision).sha
        if actual != revision:
            raise ValueError("MODEL_GATE FAIL: resolved revision differs")
        snapshot_download(model, revision=revision)
        result.append(dict(model=model, revision=actual))
    return result


def expected_prompts(manifest: dict, full: bool):
    expected, rows = [], []
    for workload, identity in manifest["workloads"].items():
        corpus = [json.loads(line) for line in (ROOT / identity["path"]).read_text(encoding="utf-8").splitlines()]
        for index, row in enumerate(corpus if full else corpus[:2]):
            expected.append(dict(workload=workload, prompt_id=row["sample_id"],
                                 prompt_hash=identity["prompt_hashes"][index],
                                 max_tokens=row["request"]["max_tokens"],
                                 stopping="vllm generation_config=vllm; default EOS; original response_format",
                                 tokenizer_revision=TARGET_REVISION))
            rows.append(row)
    return expected, rows


def diagnostic(args):
    env = environment(args.source_commit)
    manifest = build_manifest(args.source_commit)
    validate_working_workloads(manifest)
    profile_hash = profile_gate(args.profile, args.source_commit, env)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "environment.json", env)
    shutil.copyfile(args.profile, args.out / "profile_sm80.json")
    copy_immutable(args.profile.parent, args.out / "profile_inputs")
    expected, rows = expected_prompts(manifest, args.full_lossless)
    write_new(args.out / "diagnostic.json", dict(source_commit=args.source_commit,
              profile_sha256=profile_hash, expected_prompts=expected, full_lossless=args.full_lossless))
    captures = []
    try:
        from experiments.workload_adapter import build_request_payload
        for treatment in TREATMENTS:
            with serving(treatment, args.out / treatment, args.source_commit) as base:
                with (args.out / treatment / "responses.jsonl").open("x", encoding="utf-8") as raw:
                    for repeat in range(2 if treatment == "vanilla" else 1):
                        for identity, row in zip(expected, rows):
                            payload = build_request_payload(row, runtime_model=definition(treatment)["served_model_name"])
                            payload.update(stream=False, return_token_ids=True)
                            payload.pop("stream_options")
                            response = http_json(base + "/chat/completions", payload, timeout=180)
                            raw.write(json.dumps(response) + "\n"); raw.flush(); os.fsync(raw.fileno())
                            choice = response["choices"][0]
                            record = dict(identity, treatment=treatment, repeat=repeat,
                                          source_commit=args.source_commit, target_model=TARGET,
                                          target_revision=TARGET_REVISION, temperature=0, seed=42,
                                          token_ids=choice.get("token_ids"), finish_reason=choice.get("finish_reason"),
                                          request_id=response["id"], complete=True)
                            captures.append(record)
                            with (args.out / "captures.jsonl").open("a", encoding="utf-8") as cap:
                                cap.write(json.dumps(record) + "\n"); cap.flush(); os.fsync(cap.fileno())
        write_new(args.out / "lossless.json", compare_captures(captures, expected, args.source_commit))
    except BaseException as exc:
        write_new(args.out / "failure.json", dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
        raise
    finally:
        # A failed diagnostic is still publishable. This report is informational;
        # campaign launch always recomputes the gate from the raw files.
        write_new(args.out / "readiness.json", evaluate(args.out, args.source_commit))


def attempt_state(directory: Path) -> str:
    terminal = directory / "finished.json"
    if terminal.exists():
        state = json.loads(terminal.read_text())["state"]
        if state not in STATES[2:]:
            raise ValueError("Invalid terminal attempt state")
        return state
    return "RUNNING" if (directory / "started.json").exists() else "PENDING"


def validate_cell(cell: dict, directory: Path, manifest: dict) -> dict:
    from experiments.aggregate_results import aggregate_raw_jsonls
    started = json.loads((directory / "started.json").read_text())
    if started["source_commit"] != manifest["source_commit"] or started["cell"] != cell:
        raise ValueError("Cell/source provenance mismatch")
    records = read_lines(directory / "raw.jsonl")
    measured = [r for r in records if r.get("phase") == "measured"]
    if len(measured) != cell["repeats"] * cell["measured_requests"]:
        raise ValueError("Incomplete measured request count")
    ids, repeats = set(), {}
    identity = manifest["workloads"][cell["workload"]]
    for record in measured:
        rid = record["request_id"]
        if rid in ids:
            raise ValueError("Duplicate request identity")
        ids.add(rid)
        repeats.setdefault(record["repeat_id"], []).append(record)
        if (record["lane"] != cell["treatment"] or record["workload"] != cell["workload"]
                or record["concurrency"] != cell["concurrency"] or record["commit"] != manifest["source_commit"]
                or record["workload_file_sha256"] != identity["sha256"]
                or record["temperature"] != 0 or record["seed"] != 42
                or record["sample_policy"] != "cycle-rotate"):
            raise ValueError("Raw cell metadata differs from frozen protocol")
        index = record["sample_index"]
        if record["sample_id"] != identity["prompt_ids"][index]:
            raise ValueError("Wrong prompt identity")
        if record["max_completion_tokens"] != identity["request_settings"][index]["max_tokens"]:
            raise ValueError("Output cap drift")
        if record["experiment_id"] != started["campaign_id"]:
            raise ValueError("Campaign identity mismatch")
    if set(repeats) != {f"r{i:03d}" for i in range(1, 6)}:
        raise ValueError("Repeat identity coverage differs")
    for repeat, group in repeats.items():
        number = int(repeat[1:])
        if sorted(r["sample_index"] for r in group) != sorted(cell["sample_indices_by_repeat"][number - 1]):
            raise ValueError("Repeat sampling identities differ")
    launch = json.loads((directory / "server/launch.json").read_text())
    if launch["definition"] != definition(cell["treatment"]) or launch["source_commit"] != manifest["source_commit"]:
        raise ValueError("Executed server launch identity differs")
    log = (directory / "server/server.log").read_text(encoding="utf-8").lower()
    if any(needle in log for needle in ("planning failed", "feedback failed", "lineage registration failed",
            "provenance unavailable", "adaptive feedback skipped", "falling back", "traceback (most recent", "out of memory")):
        raise ValueError("Runtime error/fallback in cell server log")
    if cell["treatment"] == "eagle_adaptive":
        steps = read_lines(directory / "server/steps.jsonl")
        if not any(s["record_type"] == "planned_decision" for s in steps):
            raise ValueError("Adaptive cell has no project controller decisions")
        if any(s["record_type"] == "verification_feedback_skipped" for s in steps):
            raise ValueError("Adaptive cell has missing feedback lineage")
        cleanup = [e for e in read_lines(directory / "server/lineage.jsonl") if e["event"] == "request_cleanup"]
        if not cleanup or cleanup[-1]["active_sessions"] != 0 or cleanup[-1]["active_lineage"] != 0:
            raise ValueError("Adaptive request cleanup is incomplete")
    summaries = aggregate_raw_jsonls([directory / "raw.jsonl"])
    if len(summaries) != 1 or summaries[0]["n_runs"] != 5:
        raise ValueError("Wrong repeat coverage")
    if started.get("raw_sha256") and started["raw_sha256"] != sha256((directory / "raw.jsonl").read_bytes()):
        raise ValueError("Raw evidence changed")
    return summaries[0]


def completed_attempt(cell, directory, manifest):
    attempts = sorted(directory.glob("attempt-*"))
    complete = [p for p in attempts if attempt_state(p) == "COMPLETE_VALID"]
    if len(complete) > 1:
        raise ValueError("Multiple valid attempts: explicit review required")
    if complete:
        validate_cell(cell, complete[0], manifest)
        final = json.loads((complete[0] / "finished.json").read_text())
        if final["raw_sha256"] != sha256((complete[0] / "raw.jsonl").read_bytes()):
            raise ValueError("Completed raw hash mismatch")
        return complete[0]
    if attempts:
        raise ValueError("Failed/abandoned attempt requires root-cause assessment and explicit rerun record")
    return None


def validate_rerun_assessment(directory: Path, assessment: dict):
    attempts = sorted(directory.glob("attempt-*"))
    if not attempts or any(attempt_state(p) == "COMPLETE_VALID" for p in attempts):
        raise ValueError("Rerun requires prior failed/abandoned evidence and no valid completion")
    if set(assessment.get("prior_attempts", [])) != {p.name for p in attempts}:
        raise ValueError("Rerun assessment must identify every prior attempt")
    for field in ("previous_failure", "root_cause_hypothesis", "offline_fix_evidence", "expected_pass_fail_signal"):
        if not isinstance(assessment.get(field), str) or not assessment[field].strip():
            raise ValueError(f"Rerun assessment missing {field}")


def cell_status(directory: Path, manifest: dict):
    result = {}
    for cell in manifest["cells"]:
        attempts = sorted((directory / cell["cell_id"]).glob("attempt-*"))
        result[cell["cell_id"]] = [{"attempt_id": p.name, "state": attempt_state(p)} for p in attempts] or [{"state": "PENDING"}]
    return result


def verify_diagnostic_persisted(directory, evidence_repo):
    relative = Path("experiments/runs/m4_02_06") / directory.name
    for path in directory.rglob("*"):
        if path.is_file():
            data = subprocess.check_output(["git", "show", f"HEAD:{(relative / path.relative_to(directory)).as_posix()}"], cwd=evidence_repo)
            if sha256(data) != sha256(path.read_bytes()):
                raise ValueError("Diagnostic not preserved exactly in evidence checkout")
    # Require that the evidence commit is reachable from refreshed remote branch.
    with git_auth() as auth_env:
        subprocess.run(["git", "fetch", "origin"], cwd=evidence_repo, check=True, env=auth_env)
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/feat/m4-02-06-lossless-sweep-freeze"],
                   cwd=evidence_repo, check=True)


def campaign(args):
    env = environment(args.source_commit)
    profile_hash = profile_gate(args.profile, args.source_commit, env)
    if evaluate(args.diagnostic, args.source_commit)["ADAPTIVE_READY"] != "PASS":
        raise ValueError("ADAPTIVE_READY FAIL: full sweep prohibited")
    verify_diagnostic_persisted(args.diagnostic, args.evidence_repo)
    manifest = build_manifest(args.source_commit)
    validate_manifest(manifest, args.source_commit)
    validate_working_workloads(manifest)
    if args.action == "campaign":
        args.out.mkdir(parents=True, exist_ok=False)
        write_new(args.out / "campaign_manifest.json", dict(manifest, profile_sha256=profile_hash,
                  diagnostic_id=args.diagnostic.name, diagnostic_evidence_commit=git("rev-parse", "HEAD", root=args.evidence_repo)))
        write_new(args.out / "environment.json", env)
        shutil.copyfile(args.profile, args.out / "profile_sm80.json")
        publish(args.out, args.evidence_repo, args.source_commit)
    stored = json.loads((args.out / "campaign_manifest.json").read_text())
    if stored != dict(manifest, profile_sha256=profile_hash, diagnostic_id=args.diagnostic.name,
                      diagnostic_evidence_commit=stored["diagnostic_evidence_commit"]):
        raise ValueError("Resume manifest/profile/source drift")
    for cell in manifest["cells"]:
        cell_dir = args.out / cell["cell_id"]
        assessment = None
        if args.rerun_cell == cell["cell_id"]:
            if args.action != "resume" or args.rerun_assessment is None:
                raise ValueError("Rerun requires resume with explicit assessment")
            assessment = json.loads(args.rerun_assessment.read_text())
            validate_rerun_assessment(cell_dir, assessment)
        elif completed_attempt(cell, cell_dir, manifest):
            continue
        attempt = cell_dir / ("attempt-" + uuid.uuid4().hex)
        attempt.mkdir(parents=True, exist_ok=False)
        write_new(attempt / "started.json", dict(source_commit=args.source_commit, cell=cell,
                  campaign_id=args.out.name, attempt_id=attempt.name, state="RUNNING",
                  environment_sha256=sha256((args.out / "environment.json").read_bytes()), profile_sha256=profile_hash,
                  rerun_assessment=assessment,
                  timestamp=time.time()))
        publish(args.out, args.evidence_repo, args.source_commit)
        state, error = "FAILED", None
        try:
            with serving(cell["treatment"], attempt / "server", args.source_commit) as base:
                command = [sys.executable, "-m", "experiments.benchmark_harness",
                    "--lane", cell["treatment"], "--workload", cell["workload"],
                    "--endpoint", base + "/chat/completions", "--model", cell["served_model_name"],
                    "--workload-file", str(ROOT / WORKLOADS[cell["workload"]]),
                    "--sample-policy", "cycle-rotate", "--concurrency", str(cell["concurrency"]),
                    "--repeats", "5", "--num-requests", "32", "--warmup-requests", "3",
                    "--warmup-policy", "m3-fixed3", "--environment-profile", "m4-a100-colab-v1",
                    "--experiment-id", args.out.name, "--raw-out", str(attempt / "raw.jsonl"),
                    "--summary-csv", str(attempt / "harness_summary.csv")]
                with (attempt / "harness.log").open("x", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=7200)
            validate_cell(cell, attempt, manifest)
            state = "COMPLETE_VALID"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            text = error + "\n" + "\n".join(p.read_text(errors="replace") for p in attempt.rglob("*.log"))
            state = "OOM" if "out of memory" in text.lower() else "TIMEOUT" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "INVALID" if isinstance(exc, ValueError) else "FAILED"
            write_new(attempt / "failure.json", dict(error=error, traceback=traceback.format_exc()))
        finally:
            write_new(attempt / "finished.json", dict(state=state, error=error, timestamp=time.time(),
                      raw_sha256=sha256((attempt / "raw.jsonl").read_bytes()) if (attempt / "raw.jsonl").exists() else None))
            write_new(args.out / ("cell_status-" + uuid.uuid4().hex + ".json"), cell_status(args.out, manifest))
            publish(args.out, args.evidence_repo, args.source_commit)
        if state != "COMPLETE_VALID":
            raise RuntimeError(f"{cell['cell_id']}: {state}; evidence preserved, no automatic retry")


def aggregate(directory: Path, source_commit: str):
    from experiments.aggregate_results import aggregate_raw_jsonls, apply_speedup_vs_vanilla, write_summary_csv
    manifest = json.loads((directory / "campaign_manifest.json").read_text())
    base = {k: v for k, v in manifest.items() if k not in ("profile_sha256", "diagnostic_id", "diagnostic_evidence_commit")}
    validate_manifest(base, source_commit)
    raw = []
    for cell in base["cells"]:
        path = completed_attempt(cell, directory / cell["cell_id"], base)
        if path is None:
            raise ValueError("Full campaign incomplete; no canonical summary generated")
        raw.append(path / "raw.jsonl")
    destination = directory / ("analysis-" + uuid.uuid4().hex)
    destination.mkdir()
    summaries = apply_speedup_vs_vanilla(aggregate_raw_jsonls(raw))
    write_summary_csv(summaries, destination / "summary_eagle_adaptive_m4.csv")
    write_new(destination / "validation.json", dict(valid_cells=54, source_commit=source_commit,
              raw_sha256={str(p.relative_to(directory)): sha256(p.read_bytes()) for p in raw},
              speedup_denominator="paired vanilla TPOT p50; ratio vanilla/treatment",
              variance="repeat TPOT p50 range / cell median >15% flagged"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("source-gate", "freeze", "models", "diagnostic", "campaign", "resume", "aggregate"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--evidence-repo", type=Path)
    parser.add_argument("--full-lossless", action="store_true")
    parser.add_argument("--rerun-cell")
    parser.add_argument("--rerun-assessment", type=Path)
    args = parser.parse_args()
    source_gate(args.source_commit)
    if args.action == "source-gate":
        print("SOURCE_GATE=PASS"); return
    if args.action == "models":
        print(json.dumps(model_gate())); return
    if args.out is None:
        parser.error("--out required")
    if args.action == "freeze":
        manifest = build_manifest(args.source_commit)
        validate_working_workloads(manifest)
        write_new(args.out, manifest)
    elif args.action == "diagnostic":
        diagnostic(args)
    elif args.action in ("campaign", "resume"):
        campaign(args)
    else:
        aggregate(args.out, args.source_commit)


if __name__ == "__main__":
    main()
