"""Historical Colab-only scalar observer smoke; NOT async-lineage validation.

Run this module from the verified source snapshot with --output on mounted
Google Drive. Each cell uses a fresh local subprocess; all logs/failures survive.
The plain UniProc configuration below reproduces the pre-Item-8 observer profile.
Use its frozen historical checkout, not the current adaptive-lineage branch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback

from experiments import modal_lanes as canonical
from experiments.scalar_cost_evidence import BOUNDARY, PROTOCOL
from experiments.scalar_context import VERSION
from experiments.scalar_smoke_evidence import sha256, validate, PINNED_SOURCES
from experiments.scalar_pinned_sources import manifest as pinned_manifest, read_source
from experiments.scalar_colab_git import source_gate as git_source_gate, checked_sha

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CODE_SHA = "16528241468e2624ccfe07a69a67fcc123bfe4a3"
TESTS = ["tests/test_scalar_context.py", "tests/test_scalar_cost_capture.py",
         "tests/test_scalar_cost_evidence.py", "tests/test_m4_cost_profiler.py",
         "tests/test_m4_cost_runner.py", "tests/test_modal_profiler.py"]
PROMPT = "Count from one to fifty, writing each number in words and separating them with commas. Begin: one, two, three,"
CONFIG = dict(model=canonical.TARGET_MODEL, revision=canonical.TARGET_REVISION,
    dtype="bfloat16", quantization="awq_marlin", gpu_memory_utilization=0.85,
    max_model_len=4096, max_num_seqs=32, tensor_parallel_size=1,
    pipeline_parallel_size=1, distributed_executor_backend="uni", async_scheduling=True,
    scheduler_cls="src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
    speculative_config=dict(canonical.LANES["adaptive"].speculative_config,
                            revision=canonical.DRAFT_REVISION),
    seed=42, per_request_spec_decode_metrics="summary")
RUNTIME_PINS = {"vllm": "0.30.0", "torch": "2.13.0+cu130",
                "transformers": "5.22.2"}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def gate_require(value, reason):
    if not value:
        raise RuntimeError("CANONICAL ENVIRONMENT MISMATCH: " + reason)


def source_gate():
    return git_source_gate(ROOT, checked_sha(os.environ.get("PTIT_SCALAR_CODE_SHA")),
                           os.environ.get("PTIT_SCALAR_RUNTIME_REL"))


def require_historical_smoke_source():
    """Keep this observer-only profile from masquerading as current lineage QA.

    Historical source/evidence bytes and CONFIG remain unchanged. Offline
    validation of old artifacts remains available from the current checkout.
    """
    require(source_gate()["git_sha"] == HISTORICAL_CODE_SHA,
            "Historical scalar observer smoke requires frozen checkout " + HISTORICAL_CODE_SHA
            + "; its plain UniProcExecutor is not current async-lineage validation. "
            "Current adaptive serving requires LineageUniProcExecutor; no diagnostic bypass is provided.")


def hardware_gate():
    command = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    rows = [row.strip() for row in command.stdout.splitlines() if row.strip()]
    gate_require(command.returncode == 0 and len(rows) == 1, "nvidia-smi / single visible GPU")
    name, driver = [part.strip() for part in rows[0].split(",", 1)]
    gate_require(name == "NVIDIA L4", "GPU must be exactly NVIDIA L4; observed " + name)
    return dict(nvidia_smi_stdout=command.stdout, nvidia_smi_stderr=command.stderr,
                gpu_name=name, driver=driver, os=platform.platform(), python=sys.version)


def environment_gate():
    env = hardware_gate()
    gate_require(platform.system() == "Linux" and sys.version_info[:2] == (3, 11), "Linux / Python 3.11 required")
    for package, version in RUNTIME_PINS.items():
        observed = importlib.metadata.version(package)
        gate_require(observed == version, package + " must equal " + version + "; observed " + observed)
    env["installed_dependency_pins"] = {}
    for line in (ROOT / "experiments/scalar_smoke_constraints.txt").read_text().splitlines():
        package, version = line.split("==", 1)
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue  # optional packages are constrained, never required
        gate_require(observed == version, "dependency pin differs: " + package)
        env["installed_dependency_pins"][package] = observed
    import torch
    import vllm
    gate_require(torch.version.cuda == "13.0" and torch.cuda.is_available(), "PyTorch CUDA 13.0 unavailable")
    gate_require(torch.cuda.get_device_name(0) == "NVIDIA L4"
            and torch.cuda.get_device_capability(0) == (8, 9), "PyTorch L4/sm89 identity")
    gate_require(os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0"
            and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "V1 in-process execution required")
    captured = pinned_manifest()
    expected = {row["path"]: row["sha256"] for row in captured["files"]}
    env["installed_source_hashes"] = {}
    for relative in PINNED_SOURCES:
        name = "vllm/" + relative
        actual = hashlib.sha256(read_source(name, installed_only=True)).hexdigest()
        gate_require(actual == expected[name], "installed pinned source differs: " + name)
        env["installed_source_hashes"][name] = actual
    env.update(torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__, architecture="sm_89")
    return env


def fixed_treatment(core, k):
    """Local fixture control only; leaves production methods/thresholds intact.

    Retain the canonical controller's memory-guard decision. The fixture's
    StepPlan replacement is explicit and recorded, never a serving policy.
    """
    from src.server.budget_planner import StepPlan
    planner = core.scheduler._planner
    original = planner.plan_step
    core._scalar_control_failures = []
    def plan(decode_req_ids, *args, **kwargs):
        ids = tuple(decode_req_ids)
        natural = original(ids, *args, **kwargs)
        if any(getattr(planner.adapter.get_controller(rid), "_memory_pressure_active", True)
               for rid in ids):
            core._scalar_control_failures.append("Memory safety guard: fixed treatment refused")
            return natural
        # Prefill has no decode rows; keep its canonical K=0 plan.
        return StepPlan(batch_k=k if ids else 0, per_request_k={rid: k for rid in ids},
                        batch_node_count=k if ids else 0,
                        per_request_node_count={rid: k for rid in ids},
                        plan_sequence=natural.plan_sequence)
    planner.plan_step = plan
    return original


def runtime_object_gate(core, env):
    from src.server.adaptive_scheduler import AdaptiveAsyncScheduler
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.spec_decode.draft_model import DraftModelProposer
    from vllm.v1.outputs import ModelRunnerOutput
    cfg = core.vllm_config
    executor = core.model_executor
    runner = executor.driver_worker.worker.model_runner
    gate_require(type(core.scheduler) is AdaptiveAsyncScheduler, "scheduler class")
    gate_require(type(executor) is UniProcExecutor and type(runner) is GPUModelRunner, "executor/runner")
    gate_require(type(runner.drafter) is DraftModelProposer, "draft_model proposer")
    gate_require(cfg.scheduler_config.async_scheduling is True and runner.use_async_spec_decode
            and runner.use_async_scheduling, "async scheduling / correction branch configuration")
    gate_require(cfg.parallel_config.tensor_parallel_size == cfg.parallel_config.pipeline_parallel_size == 1,
            "TP/PP")
    gate_require(not runner.speculative_config.disable_padded_drafter_batch
            and not runner.uses_mrope, "padded absolute-position path")
    gate_require(cfg.model_config.model == canonical.TARGET_MODEL
            and cfg.model_config.revision == canonical.TARGET_REVISION, "target config revision")
    draft = cfg.speculative_config.draft_model_config
    gate_require(draft.model == canonical.DRAFT_MODEL and draft.revision == canonical.DRAFT_REVISION,
            "draft config revision")
    gate_require(str(cfg.model_config.dtype) == str(draft.dtype) == "torch.bfloat16"
            and cfg.model_config.quantization == "awq_marlin", "dtype/quantization")
    gate_require(cfg.speculative_config.num_speculative_tokens == 4, "speculative ceiling")
    fields = ("req_ids", "req_id_to_index", "sampled_token_ids", "kv_connector_output")
    gate_require(all(name in ModelRunnerOutput.__annotations__ for name in fields), "ModelRunnerOutput fields")
    for name in ("input_batch", "prev_positions", "prev_num_draft_tokens", "query_start_loc",
                 "valid_sampled_token_count_gpu", "use_async_spec_decode", "execute_model_state"):
        gate_require(hasattr(runner, name), "worker metadata: " + name)
    effective = runner.get_model().quant_config.get_name()
    gate_require(effective == "awq_marlin", "effective quantization")
    identity = dict(gpu_name=env["gpu_name"], sm_architecture="sm_89", driver=env["driver"],
        cuda=env["cuda"], torch=env["torch"], vllm=env["vllm"],
        target_model=canonical.TARGET_MODEL, target_revision=canonical.TARGET_REVISION,
        draft_model=canonical.DRAFT_MODEL, draft_revision=canonical.DRAFT_REVISION,
        dtype="bfloat16", quantization="awq_marlin", effective_quantization=effective,
        draft_method="draft_model", scheduler_cls=CONFIG["scheduler_cls"],
        scheduler_base_cls=CONFIG["scheduler_cls"],
        runtime_config_sha256=hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
        protocol=PROTOCOL)
    env.update(identity=identity, scheduler_class=type(core.scheduler).__module__ + "." + type(core.scheduler).__name__,
        executor_mode="UniProcExecutor", enginecore_path=__import__(type(core).__module__, fromlist=["x"]).__file__,
        async_scheduling=cfg.scheduler_config.async_scheduling, model_output_fields=list(ModelRunnerOutput.__annotations__),
        effective_quantization=effective, batch_queue_size=core.batch_queue_size,
        target_model_revision=cfg.model_config.revision, draft_model_revision=draft.revision)
    return runner


def tiny_window(data, k, queue_size):
    """Serialize one primary window after capture; never runs on the measured path."""
    eligible = [event for event in data["events"] if event["status"] == "eligible"
                and event["scheduled_k"] == k]
    # Four excluded eligible decode retirements settle the async queue (2x
    # the canonical two-entry queue); then one anchor plus three intervals.
    require(queue_size <= 2, "queue capacity exceeds declared settling plan")
    selected = eligible[4:8]
    require(len(selected) == 4, "insufficient nonterminal decode retirements")
    all_events = data["events"]
    positions = [all_events.index(event) for event in selected]
    require(positions == list(range(positions[0], positions[0] + 4)), "window spans ineligible lifecycle event")
    facts = {fact["verifier_id"]: fact for fact in data["execution_facts"]}
    retirements = []
    for event in selected[1:]:
        fact = facts[event["verifier_id"]]
        require(fact["proposer_k"] == k and fact["assigned_proposer_k"] == k,
                "K treatment did not reach proposer")
        require(fact["draft_forward_calls"] >= 1, "actual draft model dispatch unobserved")
        rows = []
        for row in event["context_reconstruction"]["rows"]:
            rid = row["request_id"]
            valid = dict(event["raw_sampled_counts_before_update"])[rid]
            drafts = event["verification_slots"].get(rid, 0)
            rows.append(dict(request_id=rid, effective_k=event["planned_per_request"][rid],
                context_tokens=row["pre_execution_context"],
                query_tokens=row["verification_query_length"], drafted=drafts,
                accepted=valid - 1, emitted=event["emitted_by_request"][rid],
                pre_execution_context=row["pre_execution_context"],
                verification_query_length=row["verification_query_length"],
                raw_valid_sampled_count=valid,
                corrected_post_execution_context=row["corrected_post_execution_context"],
                prefill=False, terminal=False, stale=False))
        retirements.append(dict(verifier_id=event["verifier_id"], end_ns=event["end_ns"],
            shared_k=k, producer_rows=fact["proposer_rows"], proposer_calls=fact["proposer_calls"],
            draft_forward_calls=fact["draft_forward_calls"], requests=rows))
    return dict(cell_id=f"k{k}", status="ok", timing_boundary=BOUNDARY, clock="perf_counter_ns",
        clock_domain=f"{platform.node()}:{os.getpid()}", start_ns=selected[0]["end_ns"],
        end_ns=selected[-1]["end_ns"], retirements=retirements,
        anchor_verifier_id=selected[0]["verifier_id"],
        tree_execution_verified=False, context_source="reconstructed",
        context_reconstruction_version=VERSION)


def observer_dump(observer):
    data = observer.observer.export()
    data.update(evidence_layer="LIVE_L4_SMOKE", consumer_profile=False,
        failures=observer.failures, observer_cpu_overhead_ns=observer.overhead,
        overhead_scope="partial CPU metadata and post-retirement processing; not GPU perturbation",
        observation_id_scope="process-local only", process_id=os.getpid(),
        worker_identity=dict(process_id=os.getpid(), executor="UniProcExecutor", device="cuda:0"),
        capture_complete=not observer.collecting,
        excluded_drain_pending_ids=[entry["verifier_id"] for entry in observer.observer.pending.values()],
        execution_facts=[dict(fact, **({"preparation": asdict(fact["preparation"])}
                          if "preparation" in fact else {})) for fact in observer.facts.values()])
    return data


def cell(root, k):
    """Child process. Export only after the canonical generate call finishes."""
    from experiments.scalar_live_observer import LiveObserver
    root.mkdir(exist_ok=False)
    observer, llm = None, None
    result = dict(k=k, status="FAIL", timings="INSTRUMENTATION DIAGNOSTIC ONLY")
    try:
        source_gate()
        env = environment_gate()
        save(root / "environment.json", env)
        from huggingface_hub import snapshot_download
        env["model_snapshots"] = {}
        for name, revision in ((canonical.TARGET_MODEL, canonical.TARGET_REVISION),
                               (canonical.DRAFT_MODEL, canonical.DRAFT_REVISION)):
            path = Path(snapshot_download(name, revision=revision))
            gate_require(path.name == revision, "resolved HF snapshot revision")
            env["model_snapshots"][name] = dict(revision=path.name,
                                               config_sha256=sha256(path / "config.json"))
        save(root / "environment.json", env)
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        os.environ.update(ADAPTIVE_ENABLE_VRAM_GUARD="1", ADAPTIVE_MAX_K="4",
                          ADAPTIVE_MIN_BATCH_K="0", ADAPTIVE_PER_REQUEST="1",
                          ADAPTIVE_CONTROLLER_KWARGS="", ADAPTIVE_TELEMETRY_PATH="",
                          ADAPTIVE_STEP_TELEMETRY_PATH="")
        source_gate()  # after downloads/setup, immediately before model execution
        llm = LLM(**CONFIG)
        core = llm.llm_engine.engine_core.engine_core
        runner = runtime_object_gate(core, env)
        save(root / "environment.json", env)
        tokenizer = llm.get_tokenizer()
        token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
        fixture = dict(source="in-repository deterministic counting fixture", prompt=PROMPT,
            prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), prompt_token_ids=token_ids,
            tokenized_prompt_sha256=hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
            max_tokens=48, sampling=dict(temperature=0.0, seed=42, ignore_eos=True), fixture_slot="fixture-slot-0")
        save(root / "fixture.json", fixture)
        fixed_treatment(core, k)
        def generate():
            return llm.generate([TokensPrompt(prompt_token_ids=token_ids)],
                SamplingParams(max_tokens=48, temperature=0.0, seed=42, ignore_eos=True), use_tqdm=False)
        baseline = generate()
        require(not core._scalar_control_failures, "memory guard during disabled run")
        result["disabled_output_token_ids"] = list(baseline[0].outputs[0].token_ids)
        # A fresh request incarnation; canonical state is drained by generate.
        observer = LiveObserver(core, runner.model_config.get_vocab_size())
        observer.install()
        enabled = generate()
        require(not core._scalar_control_failures, "memory guard during enabled run")
        result["enabled_output_token_ids"] = list(enabled[0].outputs[0].token_ids)
        data = observer_dump(observer)
        save(root / "observer.json", data)
        require(not data["failures"] and not data["overflow"], "live observer transport failure")
        require(result["disabled_output_token_ids"] == result["enabled_output_token_ids"],
                "enabled/disabled output mismatch")
        raw = tiny_window(data, k, core.batch_queue_size)
        result.update(status="PASS", raw=raw, settle_retirements=4, intervals=3,
            N_reason="Three intervals plus anchor validate ordering, repeated reconstruction/yield, and stable K/shape; no variance estimate.",
            k0_semantics="draft_model remains loaded; observed dispatch counts and zero-width drafts retained; not target-only vanilla")
    except Exception:
        result["error"] = traceback.format_exc()
        save(root / "failure.json", result)
    finally:
        if observer is not None:
            if not (root / "observer.json").exists():
                save(root / "observer.json", observer_dump(observer))
            try:
                observer.close()
            except Exception:
                result.update(status="FAIL", cleanup_error=traceback.format_exc())
        save(root / "cell.json", result)
    return result["status"] == "PASS"


def pre_gpu_gate(root):
    argv = [sys.executable, "-m", "pytest", "-q", *TESTS, "-rA"]
    completed = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=300)
    (root / "pre_gpu_tests.log").write_text(completed.stdout, encoding="utf-8")
    count = re.search(r"(\d+) passed", completed.stdout)
    skipped = re.search(r"(\d+) skipped", completed.stdout)
    report = dict(argv=argv, exit_code=completed.returncode,
                  git_sha=source_gate()["git_sha"], log_sha256=sha256(root / "pre_gpu_tests.log"),
                  passed=int(count[1]) if count else 0, skipped=int(skipped[1]) if skipped else 0)
    save(root / "pre_gpu_tests.json", report)
    require(report["exit_code"] == 0 and report["passed"] >= 230 and report["skipped"] == 0,
            "230-test pre-GPU gate failed or skipped source captures")
    argv = [sys.executable, "-m", "pytest", "-q", "tests/test_scalar_colab_smoke.py", "tests/test_scalar_colab_git.py"]
    completed = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=300)
    (root / "colab_preflight_tests.log").write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0 and "skipped" not in completed.stdout,
            "Colab-specific preflight tests failed/skipped")
    save(root / "colab_preflight_tests.json", dict(exit_code=completed.returncode,
        git_sha=report["git_sha"], log_sha256=sha256(root / "colab_preflight_tests.log")))


def reuse_preflight(root, previous):
    previous = Path(previous)
    required = ["pre_gpu_tests.json", "pre_gpu_tests.log", "colab_preflight_tests.json", "colab_preflight_tests.log"]
    report = json.loads((previous / required[0]).read_text())
    small = json.loads((previous / required[2]).read_text())
    code_sha = source_gate()["git_sha"]
    require(report["git_sha"] == small["git_sha"] == code_sha and report["exit_code"] == small["exit_code"] == 0
            and report["passed"] >= 230 and report["skipped"] == 0, "Preflight provenance/result mismatch")
    require(report["argv"] == [sys.executable, "-m", "pytest", "-q", *TESTS, "-rA"], "Preflight interpreter/command mismatch")
    require(sha256(previous / required[1]) == report["log_sha256"]
            and sha256(previous / required[3]) == small["log_sha256"], "Preflight log hash mismatch")
    for name in required:
        shutil.copyfile(previous / name, root / name)


def seal(root):
    save(root / "SHA256SUMS.json", {str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
        and path.name not in ("SHA256SUMS.json", "validation.json")})


def run(root, preflight_root=None):
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(artifact_kind="scalar_observer_tiny_smoke",
        git_sha=checked_sha(os.environ.get("PTIT_SCALAR_CODE_SHA")),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), cells=[0, 2], worker_batch=1,
        intervals=3, gpu_execution_requested_from_codex=False, production_source="UNCHANGED",
        status="STARTED", consumer_profile=False)
    save(root / "manifest.json", manifest)
    save(root / "config.json", CONFIG)
    (root / "raw.jsonl").write_text("", encoding="utf-8")
    verdict = {"offline_validation": "FAIL", "verdict": "LIVE OBSERVER NOT VERIFIED"}
    try:
        save(root / "source_manifest.json", source_gate())
        save(root / "environment.json", environment_gate())
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (root / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        if preflight_root is None:
            pre_gpu_gate(root)
        else:
            reuse_preflight(root, preflight_root)
        for k in (0, 2):
            log_path = root / f"k{k}-launch.log"
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run([sys.executable, "-m", "experiments.scalar_colab_smoke", "cell", "--k", str(k),
                    "--output", str(root / f"k{k}")], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, timeout=1200, env=os.environ.copy())
            folder = root / f"k{k}"
            if folder.exists():
                (folder / "stdout-stderr.log").write_bytes(log_path.read_bytes())
            require((folder / "cell.json").exists(), f"K={k} subprocess failed before saving cell status")
            result = json.loads((folder / "cell.json").read_text())
            with (root / "raw.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result.get("raw", dict(cell_id=f"k{k}", status="error", error=result.get("error")))) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            require(completed.returncode == 0 and result["status"] == "PASS", result.get("error", f"K={k} failed; no retry or next cell"))
        manifest["status"] = "COLLECTED"
        save(root / "manifest.json", manifest)
        seal(root)
        verdict = validate(root)
    except Exception:
        error = traceback.format_exc()
        save(root / "failure.json", {"error": error, "status": "FAIL"})
        manifest["status"] = "FAIL"
        save(root / "manifest.json", manifest)
        seal(root)
        verdict["error"] = error
    save(root / "validation.json", verdict)
    print(json.dumps(verdict, indent=2))
    if "CANONICAL ENVIRONMENT MISMATCH" in verdict.get("error", ""):
        print("CANONICAL ENVIRONMENT MISMATCH")
    print(verdict["verdict"])
    return verdict["offline_validation"] == "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gate", "run", "cell", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=(0, 2))
    parser.add_argument("--preflight-dir", type=Path)
    args = parser.parse_args()
    if args.stage != "validate":
        require_historical_smoke_source()  # before hardware checks or any GPU work
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    if args.stage == "cell":
        require(args.k in (0, 2), "only K=0 or K=2")
        return 0 if cell(args.output, args.k) else 1
    if args.stage == "run":
        return 0 if run(args.output, args.preflight_dir) else 1
    if args.stage == "gate":
        save(args.output, hardware_gate())
        return 0
    try:
        report = validate(args.output)
        save(args.output / "validation.json", report)
        print(report["verdict"])
        return 0
    except Exception:
        save(args.output / "validation.json", dict(offline_validation="FAIL",
            verdict="LIVE OBSERVER NOT VERIFIED", error=traceback.format_exc()))
        print("LIVE OBSERVER NOT VERIFIED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
