"""Portable Linux/Colab component profiling CLI. No Modal dependency.

Model engines run in fresh child processes per K. Failed/OOM cells are retained,
never retried automatically. Hardware mismatches stop before model loading.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments import m4_cost_profiler as prof
from experiments import modal_lanes as lanes  # Pure configuration; imports no Modal SDK.

ROOT = Path(__file__).resolve().parents[1]
# Generated evidence only; keep every other tracked/untracked source path checked.
GENERATED_EVIDENCE_PATHS = (
    "experiments/runs/m4_cost_profiling/",
    "experiments/runs/m4_cost_profiling_setup/",
)


def integers(value):
    try:
        result = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Empty/duplicate grid")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-gpu", choices=("L4","T4"), required=True)
    result.add_argument("--mode", choices=("smoke","components"), default="smoke")
    result.add_argument("--contexts", type=integers, default=(prof.CONTEXT,))
    result.add_argument("--batches", type=integers, default=(1,))
    result.add_argument("--k-values", type=integers, default=(1,))
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--worker-k", type=int, help=argparse.SUPPRESS)
    return result


def git_identity(root):
    """Return commit and source cleanliness, excluding only M4 generated evidence."""
    git = ["git","-c",f"safe.directory={root.as_posix()}"]
    sha = subprocess.check_output([*git,"rev-parse","HEAD"],cwd=root,text=True).strip()
    dirty = subprocess.check_output(
        [*git,"status","--porcelain","--untracked-files=all","--",".",
         *(f":(top,exclude){path}" for path in GENERATED_EVIDENCE_PATHS)],
        cwd=root,text=True)
    return sha, not bool(dirty.strip())


def guard_clean_source(mode, clean):
    if mode == "components" and not clean:
        raise ValueError("Component benchmark refused: commit changes before running on dirty source")


def observe_hardware():
    """Inspection only: no model download, vLLM import or CUDA inference."""
    result = {key: None for key in ("gpu","gpu_compute_capability","gpu_memory_total_mb",
                                   "driver_version","cuda_version","torch_version","vllm_version")}
    try:
        smi = subprocess.run(["nvidia-smi"],capture_output=True,text=True,timeout=20)
        result.update(nvidia_smi_output=smi.stdout, nvidia_smi_stderr=smi.stderr,
                      nvidia_smi_returncode=smi.returncode)
        query = subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                               capture_output=True,text=True,timeout=20)
        if query.returncode == 0 and query.stdout.strip():
            result["driver_version"] = next(csv.reader(query.stdout.splitlines()))[0].strip()
    except (OSError,subprocess.TimeoutExpired) as exc:
        result["nvidia_smi_error"] = str(exc)
    for package, key in (("torch","torch_version"),("vllm","vllm_version")):
        try:
            result[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if result["cuda_available"]:
            result.update(gpu=torch.cuda.get_device_name(0),
                          gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
                          gpu_memory_total_mb=torch.cuda.get_device_properties(0).total_memory/2**20)
    except Exception as exc:
        result["hardware_error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_json(path, value):
    # Small evolving metadata only; raw samples and run directories are exclusive.
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def diagnostic_report(env, records, error):
    return dict(schema_version="m4-component-calibration-v1", artifact_kind="component_report",
                not_consumer_profile=True, canonical_profile_generated=False,
                runtime_tree_available=False, gamma_attn_status="not_calibrated_pending_formula_alignment",
                technical_alignment_status=prof.ALIGNMENT_REQUIRED, environment=env,
                status="incomplete", error=error, measurement_summary=prof.summarize(records),
                full_repeat_requirement_met=False)


def run_worker(args, out):
    env = json.loads((out/"env.json").read_text(encoding="utf-8"))
    if not env["canonical_hardware"]:
        raise ValueError("Hardware mismatch cannot launch a model worker")
    sha, _ = git_identity(ROOT)
    if sha != env["source_commit"]:
        raise ValueError("Source commit changed since run reservation")
    k = args.worker_k
    if k not in env["k_values"]:
        raise ValueError("Worker K outside declared grid")
    part = out/f"k{k}"
    part.mkdir(exist_ok=False)
    cfg = prof.engine_kwargs(env["expected_gpu"],k)
    # Admit all prompts together for the fixed-context first decode boundary.
    # KV/activation exhaustion is recorded, never replaced by a smaller batch.
    cfg["max_num_batched_tokens"] = max(4096,max(env["contexts"])*max(env["batches"]))
    local_env = dict(env,K=k,engine_kwargs=cfg)
    write_json(part/"env.json",local_env)
    with (part/"raw.jsonl").open("x",encoding="utf-8") as raw:
        try:
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            llm = LLM(**cfg)
            local_env["worker_source_hashes"] = llm.collective_rpc("m4_install")
            write_json(part/"env.json",local_env)
            def operation(prompts, max_tokens):
                return llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                    SamplingParams(max_tokens=max_tokens,temperature=0.,ignore_eos=True),use_tqdm=False)
            def emit(record):
                raw.write(json.dumps(record,allow_nan=False)+"\n")
                raw.flush()
                os.fsync(raw.fileno())
            prof.measure_repeats(local_env,k,env["batches"],env["contexts"],operation,llm.collective_rpc,emit)
            local_env["status"] = "complete"
        except Exception as exc:
            local_env.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        finally:
            write_json(part/"env.json",local_env)
    return 0 if local_env["status"] == "complete" else 1


def run(args, *, root=ROOT, inspect_hardware=observe_hardware, launch=subprocess.run):
    if args.worker_k is not None:
        return run_worker(args,prof.evidence_dir(root,args.run_id))
    prof.validate_portable_config(args.mode,args.batches,args.k_values,args.contexts,args.warmup,args.repeats)
    sha, clean = git_identity(root)
    out = prof.evidence_dir(root,args.run_id)
    out.mkdir(parents=True,exist_ok=False)
    hardware = inspect_hardware()
    identity = prof.classify_hardware(hardware.get("gpu") or "",hardware.get("gpu_compute_capability"),args.expected_gpu)
    ks = list(args.k_values)
    if args.mode == "components" and 0 not in ks:
        ks.insert(0,0)  # Baseline is mandatory; --k-values names EAGLE calibration values.
    env = dict(hardware,**identity,run_id=args.run_id, timestamp_utc=prof.utc_now(),
               source_commit=sha,working_tree_clean=clean,backend="local_cuda",
               run_type="component_benchmark" if args.mode == "components" else "smoke",
               requested_run_type="component_benchmark" if args.mode == "components" else "smoke",
               target_model=lanes.TARGET_MODEL,target_revision=lanes.TARGET_REVISION,
               draft_model=lanes.EAGLE_DRAFT_MODEL,draft_revision=lanes.EAGLE_REVISION,draft_method="eagle3",
               **lanes.DTYPE_PROFILES[prof.GPU_CONFIG[args.expected_gpu][0]],
               contexts=list(args.contexts),batches=list(args.batches),k_values=ks,
               batch_grid=list(args.batches),k_grid=ks,warmup=args.warmup,repeats=args.repeats,
               timing_method=prof.TIMING,runner_selection="VLLM_USE_V2_MODEL_RUNNER=0",
               target_boundary="model forward only; excludes logits and sampling",
               proposer_boundary="EAGLE propose call; includes preparation and sampling",
               canonical_component_run=False,
               status="inspecting")
    records, error = [], None
    manifest = dict(run_id=args.run_id,source_commit=sha,status="inspecting",started_at_utc=prof.utc_now(),
                    artifact_kind="component_measurements",canonical_profile_generated=False,workers=[])
    (out/"raw.jsonl").touch(exist_ok=False)
    write_json(out/"env.json",env)
    write_json(out/"run_manifest.json",manifest)
    try:
        if not identity["canonical_hardware"]:
            env.update(run_type="smoke",canonical_component_run=False,status="hardware_mismatch")
            error = (f"Requested canonical {args.expected_gpu} profiling but observed {hardware.get('gpu') or 'no CUDA GPU'}; "
                     "run retained as non-canonical smoke. No model loaded.")
            print(error)
            return 0
        guard_clean_source(args.mode,clean)
        if sys.platform != "linux":
            raise ValueError("vLLM execution requires Linux CUDA; hardware evidence retained")
        if hardware.get("vllm_version") != lanes.VLLM_VERSION:
            raise ValueError("Install experiments/requirements-m4-cuda.txt: vLLM must be 0.30.0")
        if hardware.get("torch_version") != "2.13.0+cu130" or hardware.get("cuda_version") != "13.0":
            raise ValueError("Pinned portable stack requires torch 2.13.0+cu130 / CUDA 13.0")
        # Preserve a resolved installation inventory alongside direct dependency pins.
        frozen = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,timeout=60)
        (out/"pip-freeze.txt").write_text(frozen.stdout+frozen.stderr,encoding="utf-8")
        for k in ks:
            command = [sys.executable,"-m","experiments.m4_cost_runner","--run-id",args.run_id,
                       "--expected-gpu",args.expected_gpu,"--worker-k",str(k)]
            child_env = dict(os.environ,VLLM_USE_V2_MODEL_RUNNER="0",PYTHONUNBUFFERED="1")
            with (out/f"k{k}.log").open("x",encoding="utf-8") as log:
                child = launch(command,cwd=root,env=child_env,stdout=log,stderr=subprocess.STDOUT)
            manifest["workers"].append(dict(K=k,returncode=child.returncode))
            part = out/f"k{k}"
            if (part/"raw.jsonl").exists():
                with (out/"raw.jsonl").open("a",encoding="utf-8") as aggregate:
                    for line in (part/"raw.jsonl").read_text(encoding="utf-8").splitlines():
                        record = json.loads(line)
                        records.append(record)
                        aggregate.write(json.dumps(record)+"\n")
            if child.returncode:
                detail = json.loads((part/"env.json").read_text(encoding="utf-8")) if (part/"env.json").exists() else {}
                raise RuntimeError(f"K={k} failed (exit {child.returncode}): {detail.get('error','see worker log')}; no automatic retry")
        prof.validate_portable_evidence(env,records)
        env.update(status="complete",canonical_component_run=args.mode == "components" and clean)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        env["status"] = "failed"
        print(error,file=sys.stderr)
        return 1
    finally:
        env["error"] = error
        manifest.update(status=env["status"],error=error,finished_at_utc=prof.utc_now(),
                        run_type=env["run_type"],canonical_hardware=env["canonical_hardware"],n_raw_records=len(records))
        write_json(out/"env.json",env)
        write_json(out/"run_manifest.json",manifest)
        report = diagnostic_report(env,records,error) if error else prof.build_component_report(env,records)
        write_json(out/"component_report.json",report)
        print(f"Artifacts retained: {out}")


def main():
    args = parser().parse_args()
    try:
        return run(args)
    except (ValueError,FileExistsError,subprocess.CalledProcessError) as exc:
        print(f"Refused: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
