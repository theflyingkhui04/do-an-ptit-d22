#!/usr/bin/env python3
"""M2-03 diagnostic-only prefix-cache ablation replications R2/R3.

Commands (from repo root after copying to scripts/):
  python scripts/m2_03_cache_replication.py prepare
  python scripts/m2_03_cache_replication.py run-cell R2 on --allow-stop-vllm
  python scripts/m2_03_cache_replication.py run-cell R2 off --allow-stop-vllm
  python scripts/m2_03_cache_replication.py run-cell R3 off --allow-stop-vllm
  python scripts/m2_03_cache_replication.py run-cell R3 on --allow-stop-vllm
  python scripts/m2_03_cache_replication.py summarize

Uses prior validated launch/switch/probe code unchanged except unique file names
and DIAG paths. Never edits canonical launchers, the adaptive scheduler, or R1.
Every run-cell switches/restarts ALL vLLM lanes (same as existing switch_lane).
Do not use for throughput benchmarking or for canonical M2-03 acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import unified_diff
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNS = ROOT / "experiments" / "runs" / "m2_03"
DIAGNOSTICS = RUNS / "diagnostics"
LOG_FILE = ROOT / "logs" / "vllm_lane3.log"
MODEL = "qwen2.5-7b-adaptive"
SAMPLE_ID = "w3_business_v2_004"
EXPECTED_WORKLOAD_SHA = "57341b0fc8c2ebbdfcfc579a6343aea66b08ca11fe2bfed5bc53a2bd49f5d83b"
EXPECTED_VANILLA_SHA = "248bbb600d6e7e378501b7b0f08ae3561e2ef6f2ee953469e9efe07aeca02803"
WORKLOAD = ROOT / "experiments" / "configs" / "w3_v2" / "chat_requests.jsonl"
VANILLA = RUNS / "m2-03-20261013-r03" / "vanilla.jsonl"
REPLICATIONS = ("R2", "R3")
CONDITIONS = ("on", "off")
R1_DIRS = {
    "on": DIAGNOSTICS / "k4_all_output_coldstart_w3_004",
    "off": DIAGNOSTICS / "k4_no_prefix_coldstart_w3_004",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError("STOP: " + message)


def json_lines(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, obj: Any, *, exclusive: bool = False) -> None:
    # State and manifest files are derived metadata; only the first write uses x.
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def cmd_output(args: list[str], *, timeout: int = 30) -> str:
    result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout)
    require(result.returncode == 0, f"command failed: {args!r}: {result.stdout[-1500:]}")
    return result.stdout.strip()


def diag_dir(rep: str, cond: str) -> Path:
    return DIAGNOSTICS / f"k4_cache_{cond}_coldstart_w3_004_{rep.lower()}"


def generated_names(rep: str, cond: str) -> tuple[Path, Path, Path]:
    suffix = f"{rep.lower()}_cache_{cond}"
    return (
        SCRIPTS / f"launch_lane3_adaptive_{suffix}_diag.sh",
        SCRIPTS / f"switch_lane_{suffix}_diag.sh",
        SCRIPTS / f"m2_03_{suffix}_two_probes.py",
    )


def source_names(cond: str) -> tuple[Path, Path, Path]:
    tag = "all_output" if cond == "on" else "no_prefix"
    return (
        SCRIPTS / f"launch_lane3_adaptive_{tag}_diag.sh",
        SCRIPTS / f"switch_lane_{tag}_diag.sh",
        SCRIPTS / f"m2_03_{tag}_two_probes.py",
    )


def checked_replace(text: str, old: str, new: str, count: int, context: str) -> str:
    found = text.count(old)
    require(found == count, f"{context}: expected {count} occurrences of {old!r}; found {found}")
    return text.replace(old, new)


def planned_files(rep: str, cond: str) -> list[tuple[Path, Path, str]]:
    src_launch, src_switch, src_probe = source_names(cond)
    dst_launch, dst_switch, dst_probe = generated_names(rep, cond)
    for path in (src_launch, src_switch, src_probe):
        require(path.is_file(), f"missing tested source script: {path}")

    launch = src_launch.read_text(encoding="utf-8")
    switch = src_switch.read_text(encoding="utf-8")
    probe = src_probe.read_text(encoding="utf-8")

    require("TraceAllOutputScheduler" in launch, "launcher no longer uses all-output scheduler")
    require("--per-request-spec-decode-metrics detailed" in launch,
            "launcher not configured for detailed metrics")
    require(("--no-enable-prefix-caching" in launch) == (cond == "off"),
            f"unexpected cache flag in {src_launch}")
    require("--enable-prefix-caching" not in launch,
            "unexpected explicit positive cache flag in tested launcher")
    require("ADAPTIVE_ALL_OUTPUT_TRACE_PATH" not in launch,
            "unexpected environment override inside launcher")

    switch = checked_replace(switch, src_launch.name, dst_launch.name, 2, str(src_switch))
    old_diag = (
        "diagnostics/k4_all_output_coldstart_w3_004" if cond == "on"
        else "diagnostics/k4_no_prefix_coldstart_w3_004"
    )
    new_diag = f"diagnostics/{diag_dir(rep, cond).name}"
    probe = checked_replace(probe, old_diag, new_diag, 1, str(src_probe))
    require('PROBES = DIAG / "probes.jsonl"' in probe,
            "probe evidence contract changed")
    require('TRACE = DIAG / "step_trace.jsonl"' in probe,
            "probe trace contract changed")
    require('TELEMETRY = DIAG / "telemetry.jsonl"' in probe,
            "probe telemetry contract changed")
    compile(probe, str(dst_probe), "exec")
    return [(src_launch, dst_launch, launch),
            (src_switch, dst_switch, switch),
            (src_probe, dst_probe, probe)]


def ensure_frozen_inputs() -> None:
    require(WORKLOAD.is_file(), f"missing frozen workload: {WORKLOAD}")
    require(VANILLA.is_file(), f"missing canonical Vanilla reference: {VANILLA}")
    require(digest(WORKLOAD) == EXPECTED_WORKLOAD_SHA,
            "W3 workload SHA changed; DO NOT compare across changed inputs")
    require(digest(VANILLA) == EXPECTED_VANILLA_SHA,
            "Vanilla reference SHA changed; DO NOT compare across changed references")
    records = [x for x in json_lines(WORKLOAD) if x.get("sample_id") == SAMPLE_ID]
    require(len(records) == 1, "expected exactly one W3-004 workload row")
    generation = records[0].get("request") or {}
    require((generation.get("max_tokens"), generation.get("seed"), generation.get("temperature"))
            == (768, 42, 0.0), "frozen generation params changed")


def validate_prepared(rep: str, cond: str) -> None:
    for _, dst, desired in planned_files(rep, cond):
        require(dst.is_file(), f"run prepare first; missing {dst}")
        require(dst.read_text(encoding="utf-8") == desired,
                f"generated file drifted after preparation: {dst}")
    launch, switch, probe = generated_names(rep, cond)
    for p in (launch, switch):
        cmd_output(["bash", "-n", str(p)])
    compile(probe.read_text(encoding="utf-8"), str(probe), "exec")


def prepare() -> None:
    ensure_frozen_inputs()
    plan = []
    for rep in REPLICATIONS:
        for cond in CONDITIONS:
            plan.extend(planned_files(rep, cond))
    for _, dest, _ in plan:
        require(not dest.exists(), f"refusing to overwrite prepared file {dest}")
    for _, dest, content in plan:
        compile(content, str(dest), "exec") if dest.suffix == ".py" else None
    print(f"PREPARE: creating {len(plan)} new diagnostic-only scripts")
    for src, dest, content in plan:
        with dest.open("x", encoding="utf-8") as stream:
            stream.write(content)
        os.chmod(dest, src.stat().st_mode & 0o777)
        print(f"\nCREATED: {dest.relative_to(ROOT)} SHA256={digest(dest)}")
        diff = "\n".join(unified_diff(
            src.read_text(encoding="utf-8").splitlines(), content.splitlines(),
            fromfile=str(src.relative_to(ROOT)), tofile=str(dest.relative_to(ROOT)),
            lineterm=""))
        if diff:
            print(diff)
    for rep in REPLICATIONS:
        for cond in CONDITIONS:
            validate_prepared(rep, cond)
    cmd_output(["git", "diff", "--check"])
    print("\nPREPARE_REPLICATIONS: PASS")
    print("CANONICAL_AND_R1_SOURCES_UNCHANGED")
    print("ORDER: R2 on -> R2 off -> R3 off -> R3 on")


def shell_capture(args: list[str], logfile: Path) -> None:
    """Stream stdout to terminal and immutable per-cell text file."""
    with logfile.open("x", encoding="utf-8") as output:
        with subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
            code = proc.wait()
    require(code == 0, f"command returned {code}; inspect {logfile}")


def live_vllm_command() -> list[str]:
    """Read the actual live vLLM serve argv for port 8003 on Linux."""
    result = []
    proc_root = Path("/proc")
    require(proc_root.is_dir(), "Linux /proc is required for live server checks")
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        args = [x for x in args if x]
        if len(args) < 3:
            continue
        if not any(Path(x).name == "vllm" for x in args[:3]) or "serve" not in args:
            continue
        if "--port" in args and args[args.index("--port") + 1:args.index("--port") + 2] == ["8003"]:
            result.append(args)
    require(len(result) == 1, f"expected exactly one port-8003 vllm process, got {len(result)}")
    return result[0]


def verify_live_config(cond: str) -> dict[str, Any]:
    argv = live_vllm_command()
    expected_cls = "src.server.adaptive_scheduler_all_output_trace_diag.TraceAllOutputScheduler"
    require("--scheduler-cls" in argv and argv[argv.index("--scheduler-cls") + 1] == expected_cls,
            "wrong live scheduler class")
    require("--per-request-spec-decode-metrics" in argv and
            argv[argv.index("--per-request-spec-decode-metrics") + 1] == "detailed",
            "live metrics are not detailed")
    require(("--no-enable-prefix-caching" in argv) == (cond == "off"),
            "live cache flag disagrees with condition")

    # Positive engine-core evidence, not merely a 0% cache-hit rate.
    expected = "False" if cond == "off" else "True"
    end = time.monotonic() + 15
    log = ""
    while time.monotonic() < end:
        log = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        engine_lines = [x for x in log.splitlines()
                        if "Initializing a V1 LLM engine" in x]
        if engine_lines:
            break
        time.sleep(0.5)
    require(engine_lines, "effective EngineCore initialization log not available")
    last = engine_lines[-1]
    require("(v0.30.0)" in last, "pinned vLLM version v0.30.0 not confirmed")
    require(f"enable_prefix_caching={expected}" in last,
            "effective EngineCore cache config does not match condition")
    require("Qwen/Qwen2.5-7B-Instruct-AWQ" in last and
            "Qwen/Qwen2.5-0.5B-Instruct" in last,
            "target/draft models changed")
    print("LIVE_CONFIG: PASS", cond, f"enable_prefix_caching={expected}")
    return {"argv": argv, "effective_prefix_cache_enabled": cond == "on",
            "engine_log_line": last, "checked_utc": now()}


def snapshot_provenance(rep: str, cond: str) -> dict[str, Any]:
    launch, switch, probe = generated_names(rep, cond)
    paths = [
        WORKLOAD, VANILLA,
        ROOT / "src/server/adaptive_scheduler.py",
        ROOT / "src/server/adaptive_scheduler_trace_diag.py",
        ROOT / "src/server/adaptive_scheduler_all_output_trace_diag.py",
        launch, switch, probe,
    ]
    return {
        "task": "M2-03", "kind": "diagnostic_ablation_replication",
        "canonical_benchmark": False, "replication": rep, "cache_condition": cond,
        "cold_start": True, "request_count": 2, "concurrency": 1,
        "sample_id": SAMPLE_ID,
        "generation": {"max_tokens": 768, "seed": 42, "temperature": 0.0},
        "max_k": 4, "min_batch_k": 0,
        "expected_prefix_caching": cond == "on",
        "output_directory": str(diag_dir(rep, cond).relative_to(ROOT)),
        "git_sha": cmd_output(["git", "rev-parse", "HEAD"]),
        "git_status_short_before": cmd_output(["git", "status", "--short"]),
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "nvidia_smi": cmd_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"], timeout=15)
                       if shutil.which("nvidia-smi") else "UNAVAILABLE",
        "started_utc": now(),
    }


def trace_validation(diag: Path) -> dict[str, Any]:
    """Raw -> reconstructed IDs, join + step metrics; never require MATCH."""
    probes = json_lines(diag / "probes.jsonl")
    all_events = json_lines(diag / "all_output_trace.jsonl")
    step_events = json_lines(diag / "step_trace.jsonl")
    telemetry = json_lines(diag / "telemetry.jsonl")
    require(len(probes) == 2, f"expected exactly two requests, got {len(probes)}")
    expected_ref = [r for r in json_lines(VANILLA)
                    if r.get("workload") == "W3" and r.get("sample_id") == SAMPLE_ID]
    require(len(expected_ref) == 1 and expected_ref[0].get("ok"),
            "frozen Vanilla reference not uniquely available")
    ref_ids = expected_ref[0]["token_ids"]
    outcomes = []
    all_engine_ids = set()
    for attempt, p in enumerate(probes, start=1):
        require(p.get("attempt") == attempt, "probe attempts missing/out of order")
        require(p.get("http_status") == 200, f"probe {attempt} HTTP failed")
        ids = p.get("token_ids")
        require(isinstance(ids, list), f"probe {attempt} missing token IDs")
        rid = p.get("response_id")
        require(isinstance(rid, str) and rid, f"probe {attempt} missing response ID")
        rows = [e for e in all_events if e.get("request_id") == rid
                or str(e.get("request_id", "")).startswith(rid + "-")]
        engine_ids = {x.get("request_id") for x in rows}
        require(len(engine_ids) == 1, f"probe {attempt} ambiguous EngineCore trace join")
        engine_id = next(iter(engine_ids))
        require(engine_id not in all_engine_ids, "two probes joined same engine request")
        all_engine_ids.add(engine_id)
        reconstructed: list[int] = []
        for e in rows:
            start, end = e["output_start_index"], e["output_end_index_exclusive"]
            emitted = e["emitted_token_ids"]
            require(start == len(reconstructed), f"probe {attempt} trace gap/overlap at {start}")
            require(end - start == len(emitted), f"probe {attempt} invalid output range")
            reconstructed.extend(emitted)
        require(reconstructed == ids, f"probe {attempt} trace reconstruction mismatch")
        require(p.get("match_vanilla") == (ids == ref_ids),
                f"probe {attempt} stored MATCH flag inconsistent with reference")
        calculated_mismatch = next((i for i, (a, b) in enumerate(zip(ref_ids, ids))
                                   if a != b), None)
        if calculated_mismatch is None and len(ref_ids) != len(ids):
            calculated_mismatch = min(len(ref_ids), len(ids))
        require(p.get("first_mismatch_vanilla") == calculated_mismatch,
                f"probe {attempt} stored mismatch index inconsistent with reference")

        steps = sorted((x for x in step_events if x.get("request_id") == engine_id),
                       key=lambda x: x["spec_step_index"])
        spec = p.get("speculative_metrics") or {}
        accepted = spec.get("per_step_accepted")
        require(isinstance(accepted, list), f"probe {attempt} missing detailed acceptance")
        require(len(steps) == len(accepted), f"probe {attempt} step trace count mismatch")
        require([x["spec_step_index"] for x in steps] == list(range(1, len(steps) + 1)),
                f"probe {attempt} duplicate/missing speculative steps")
        require([x["num_accepted"] for x in steps] == accepted,
                f"probe {attempt} acceptance not reflected in trace")
        marked = [e.get("spec_step_index") for e in rows
                  if e.get("spec_step_index") is not None]
        require(marked == list(range(1, len(steps) + 1)),
                f"probe {attempt} all-output speculative markers mismatch step trace")
        require(len([t for t in telemetry if t.get("request_id") == engine_id]) == 1,
                f"probe {attempt} telemetry cannot be uniquely joined")
        outcomes.append({
            "attempt": attempt, "engine_id": engine_id,
            "events": len(rows), "spec_steps": len(steps), "tokens": len(ids),
            "match_vanilla": ids == ref_ids,
            "first_mismatch": p.get("first_mismatch_vanilla"),
            "token_200": ids[200] if len(ids) > 200 else None,
            "trace_reconstruction": True,
        })
    require({e.get("request_id") for e in all_events} == all_engine_ids,
            "orphan/unexpected all-output trace request IDs")
    require({e.get("request_id") for e in step_events} == all_engine_ids,
            "orphan/unexpected speculative trace request IDs")
    require({e.get("request_id") for e in telemetry} == all_engine_ids,
            "orphan/unexpected telemetry request IDs")
    return {"validated_utc": now(), "trace_integrity": "PASS", "outcomes": outcomes,
            "two_runs_equal": probes[0]["token_ids"] == probes[1]["token_ids"],
            "all_output_records": len(all_events), "spec_trace_records": len(step_events),
            "telemetry_records": len(telemetry)}


def run_cell(rep: str, cond: str, consent: bool) -> None:
    require(consent, "run-cell restarts/kills all vLLM lanes; pass --allow-stop-vllm explicitly")
    ensure_frozen_inputs()
    validate_prepared(rep, cond)
    diag = diag_dir(rep, cond)
    require(not diag.exists(), f"diagnostic evidence already exists, refusing overwrite: {diag}")
    require(LOG_FILE.is_file(), f"expected existing server log missing: {LOG_FILE}")

    provenance = snapshot_provenance(rep, cond)
    diag.mkdir(parents=True, exist_ok=False)
    write_json(diag / "manifest.json", provenance, exclusive=True)
    shutil.copyfile(LOG_FILE, diag / "pre_restart_lane3.log")
    state_path = diag / "run_state.json"
    state = {"status": "STARTING", "started_utc": now(), "replication": rep,
             "cache_condition": cond}
    write_json(state_path, state, exclusive=True)
    print(f"EVIDENCE_DIR: {diag}")
    print("WARNING: switch_lane stops ALL currently running vLLM lanes")

    _, switch, probe = generated_names(rep, cond)
    try:
        env = dict(os.environ)
        env.update({
            "ADAPTIVE_MAX_K": "4", "ADAPTIVE_MIN_BATCH_K": "0",
            "ADAPTIVE_ALL_OUTPUT_TRACE_PATH": str(diag / "all_output_trace.jsonl"),
            "ADAPTIVE_TRACE_PATH": str(diag / "step_trace.jsonl"),
            "ADAPTIVE_TELEMETRY_PATH": str(diag / "telemetry.jsonl"),
        })
        # Pass environment directly, never through shell interpolation.
        with (diag / "startup_stdout.log").open("x", encoding="utf-8") as output:
            with subprocess.Popen(["bash", str(switch)], cwd=ROOT, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as proc:
                assert proc.stdout is not None
                try:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        output.write(line)
                        output.flush()
                except BaseException:
                    proc.terminate()
                    proc.wait()
                    raise
                status = proc.wait()
        require(status == 0, f"server start failed: exit={status}")
        state["status"] = "SERVER_HEALTHY"
        live = verify_live_config(cond)
        write_json(diag / "effective_server_config.json", live, exclusive=True)
        state["status"] = "PROBING"
        write_json(state_path, state)
        shell_capture([sys.executable, str(probe)], diag / "probe_stdout.log")
        validation = trace_validation(diag)
        write_json(diag / "validation.json", validation, exclusive=True)
        print("FOUR_FIELDS: attempt, tokens, match_vanilla, first_mismatch")
        for row in validation["outcomes"]:
            print({k: row[k] for k in ("attempt", "tokens", "match_vanilla", "first_mismatch")})
        print("TRACE_JOIN_AND_RECONSTRUCTION: PASS")
        state["status"] = "VALIDATED"
    except BaseException as exc:
        state["status"] = "FAILED_OR_INTERRUPTED"
        state["reason"] = f"{type(exc).__name__}: {exc}"
        print("DIAGNOSTIC_FAILED: raw files preserved; do not overwrite", file=sys.stderr)
        raise
    finally:
        state["finished_utc"] = now()
        # A copy, not a move; the live log remains available for debugging.
        if LOG_FILE.is_file():
            shutil.copyfile(LOG_FILE, diag / "server_lane3.log")
        write_json(state_path, state)
        print(f"EVIDENCE_PRESERVED: {diag}")


def summarize(save: bool) -> None:
    entries = []
    for rep in ("R1", "R2", "R3"):
        for cond in ("on", "off"):
            diag = R1_DIRS[cond] if rep == "R1" else diag_dir(rep, cond)
            entry: dict[str, Any] = {
                "replication": rep, "cache": cond,
                "directory": str(diag.relative_to(ROOT)),
                "state": "NOT_RUN", "outcomes": None,
            }
            if diag.exists():
                try:
                    check = trace_validation(diag)
                    entry.update(state="VALIDATED", outcomes=check["outcomes"],
                                 two_runs_equal=check["two_runs_equal"])
                    if rep != "R1":
                        state_path = diag / "run_state.json"
                        require(state_path.is_file() and
                                json.loads(state_path.read_text(encoding="utf-8")).get("status") == "VALIDATED",
                                f"cell not fully validated: {diag}")
                        config = json.loads((diag / "effective_server_config.json").read_text(encoding="utf-8"))
                        require(config["effective_prefix_cache_enabled"] == (cond == "on"),
                                f"cache config provenance does not match cell {diag}")
                except Exception as exc:
                    entry["state"] = "INCOMPLETE_OR_INVALID"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
    print("\n=== M2-03 DIAGNOSTIC REPLICATIONS (not canonical) ===")
    for x in entries:
        match_pattern = (" -> ".join("MATCH" if p["match_vanilla"] else "FAIL"
                                      for p in x["outcomes"]) if x["outcomes"] else "-")
        print(f"{x['replication']:2s} CACHE_{x['cache'].upper():3s} {x['state']:22s} {match_pattern}")
    complete = all(e["state"] == "VALIDATED" for e in entries)
    report = {
        "created_utc": now(), "task": "M2-03", "kind": "diagnostic_only",
        "canonical_result_not_overridden": "r03: 44/54 exact token matches (FAIL)",
        "all_cells_validated": complete,
        "note": "Repeated ON/OFF patterns support a cache-associated effect, not causal proof of a KV-cache correctness bug.",
        "cells": entries,
    }
    if save:
        target = DIAGNOSTICS / "cache_ablation_r1_r3_summary.json"
        require(not target.exists(), f"refusing to overwrite summary: {target}")
        write_json(target, report, exclusive=True)
        print("SUMMARY_SAVED:", target)
    if complete:
        print("ALL_SIX_DIAGNOSTIC_CELLS: VALIDATED")
    else:
        print("INCOMPLETE: do not infer R2/R3 results from missing cells")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create & diff 12 new diagnostic-only scripts, without starting GPU")
    run = sub.add_parser("run-cell", help="Fresh cold-start + 2 probes for one cell")
    run.add_argument("replication", choices=REPLICATIONS)
    run.add_argument("condition", choices=CONDITIONS)
    run.add_argument("--allow-stop-vllm", action="store_true",
                     help="Explicit permission to stop all currently running vLLM servers")
    summ = sub.add_parser("summarize", help="Validate and compare R1/R2/R3 without GPU")
    summ.add_argument("--save", action="store_true", help="Write a new, non-overwriting JSON summary")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "run-cell":
            run_cell(args.replication, args.condition, args.allow_stop_vllm)
        else:
            summarize(args.save)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
