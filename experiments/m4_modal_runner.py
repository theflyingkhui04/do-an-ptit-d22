"""M4 entry point reusing the project's committed image, app and Modal volumes.

Example (PowerShell):
  $env:PTIT_GPU='L4'; $env:PTIT_DTYPE_PROFILE='l4-bf16'
  modal run -m experiments.m4_modal_runner::main --run-id UNIQUE --stage smoke --ks 1
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments import modal_lanes as lanes
from experiments.modal_runner import (app, image, GPU, PROFILE, SOURCE_ROOT, SOURCE_COMMIT,
                                      SERVER_VOLUMES, artifacts_volume, REPO_ROOT)
from experiments import m4_cost_profiler as prof

if PROFILE != prof.GPU_CONFIG[GPU][0]:
    raise ValueError("GPU and dtype profile must match the frozen M4 configuration")

m4_image = image
for filename in ("modal_profiler.py", "m4_cost_profiler.py", "m4_worker_observer.py"):
    m4_image = m4_image.add_local_file(SOURCE_ROOT / "experiments" / filename,
                                      f"/root/experiments/{filename}")


@app.function(image=m4_image, gpu=GPU, timeout=3*60*60, retries=0, single_use_containers=True,
              volumes=SERVER_VOLUMES,
              env={"HF_HOME": lanes.HF_HOME, "HF_HUB_OFFLINE": "1", "PTIT_GPU": GPU,
                   "VLLM_USE_V2_MODEL_RUNNER": "0",
                   "PYTHONPATH": f"{lanes.REMOTE_PROJECT_ROOT}:/root",
                   "VLLM_CACHE_ROOT": lanes.VLLM_CACHE_ROOT})
def measure(run_id: str, k: int, batches: list[int], context: int,
            warmup: int, repeats: int, stage: str):
    prof.validate_submission(GPU,batches,[k],context,warmup,repeats,stage)
    import torch
    import pynvml
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    root = Path(lanes.ARTIFACTS_ROOT) / "m4_cost_profiling" / run_id / f"k{k}"
    root.mkdir(parents=True, exist_ok=False)
    records = []
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def decoded(value):
        return value.decode() if isinstance(value, bytes) else str(value)
    capability = torch.cuda.get_device_capability(0)
    env = dict(run_id=run_id, timestamp_utc=prof.utc_now(), source_commit=SOURCE_COMMIT,
               requested_gpu=GPU, gpu=decoded(pynvml.nvmlDeviceGetName(handle)),
               sm_architecture=f"sm_{capability[0]}{capability[1]}",
               gpu_memory_total_mb=pynvml.nvmlDeviceGetMemoryInfo(handle).total/2**20,
               driver_version=decoded(pynvml.nvmlSystemGetDriverVersion()),
               cuda_version=torch.version.cuda, torch_version=torch.__version__, vllm_version=vllm.__version__,
               dtype=lanes.DTYPE_PROFILES[PROFILE]["dtype"], quantization=lanes.DTYPE_PROFILES[PROFILE]["quantization"],
               target_model=lanes.TARGET_MODEL, target_revision=lanes.TARGET_REVISION,
               draft_model=lanes.EAGLE_DRAFT_MODEL, draft_revision=lanes.EAGLE_REVISION,
               draft_method="eagle3" if k else None, K=k, engine_kwargs=prof.engine_kwargs(GPU,k),
               runner_selection="VLLM_USE_V2_MODEL_RUNNER=0",
               context_len=context, warmup=warmup, repeats=repeats, stage=stage,
               batch_grid=batches, k_grid=[k], timing_method=prof.TIMING,
               target_boundary="model forward only; excludes logits and sampling",
               proposer_boundary="EAGLE propose call; includes preparation and sampling")
    def persist_env():
        (root / "env.json").write_text(json.dumps(env, indent=2)+"\n", encoding="utf-8")
        artifacts_volume.commit()
    persist_env()
    raw = (root / "raw.jsonl").open("x", encoding="utf-8")
    try:
        prof.validate_hardware(GPU, env["gpu"], env["sm_architecture"])
        if env["vllm_version"] != lanes.VLLM_VERSION:
            raise RuntimeError("Unpinned runtime")
        llm = LLM(**env["engine_kwargs"])
        env["worker_source_hashes"] = llm.collective_rpc("m4_install")
        # Exact batch admission is verified in worker shapes, even if capacity
        # cannot be discovered through this vLLM engine frontend.
        env["batch_admission"] = "worker-observed exact B, uniform K+1 queries, context L; reject otherwise"
        persist_env()
        def operation(prompts, max_tokens):
            return llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(max_tokens=max_tokens, temperature=0., ignore_eos=True), use_tqdm=False)
        def emit(record):
            raw.write(json.dumps(record)+"\n")
            raw.flush()
            records.append(record)
            artifacts_volume.commit()
        prof.measure_repeats(env,k,batches,[context],operation,llm.collective_rpc,emit)
        env["status"] = "complete"
    except Exception as exc:
        env.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        raw.close()
        persist_env()
    return dict(env=env, records=records)


def committed_source():
    git = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]
    branch = subprocess.check_output([*git, "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    if branch != "feat/m4-02-02-gpu-cost-profiler":
        raise ValueError("M4 requires its task branch")
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=normal"], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise ValueError("Commit source/config and evidence before submitting; dirty checkout rejected")
    sha = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if sha != SOURCE_COMMIT:
        raise ValueError("Source snapshot differs from current HEAD")
    return sha


@app.local_entrypoint()
def main(run_id: str, stage: str = "smoke", batches: str = "1", ks: str = "1",
         context: int = prof.CONTEXT, warmup: int = 5, repeats: int = 2):
    bs = tuple(int(x) for x in batches.split(","))
    kvals = tuple(int(x) for x in ks.split(","))
    prof.validate_submission(GPU, bs, kvals, context, warmup, repeats, stage)
    sha = committed_source()
    out = prof.evidence_dir(REPO_ROOT, run_id)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=run_id, source_commit=sha, requested_gpu=GPU,
                    stage=stage, batch_grid=list(bs), k_grid=list(kvals), context_len=context,
                    warmup=warmup, repeats=repeats, timestamp_utc=prof.utc_now(), status="submitted")
    (out / "submission.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    results = []
    # One fresh engine/container per K; sequential to bound credit and VRAM.
    for k in kvals:
        call = measure.spawn(run_id,k,list(bs),context,warmup,repeats,stage)
        (out / f"k{k}.submission.json").write_text(json.dumps(
            dict(manifest, K=k, call_id=call.object_id), indent=2)+"\n", encoding="utf-8")
        result = call.get()
        part = out / f"k{k}"
        part.mkdir()
        (part / "env.json").write_text(json.dumps(result["env"], indent=2)+"\n", encoding="utf-8")
        (part / "raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in result["records"]), encoding="utf-8")
        results.append(result)
        if result["env"]["status"] != "complete":
            raise RuntimeError(f"K={k} failed; retained {part}. No automatic retry.")
    records = [r for result in results for r in result["records"]]
    env = dict(results[0]["env"], **{key: manifest[key] for key in
               ("k_grid", "batch_grid", "stage", "warmup", "repeats")},
               environments={str(r["env"]["K"]): r["env"] for r in results})
    (out / "env.json").write_text(json.dumps(env, indent=2)+"\n", encoding="utf-8")
    (out / "raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    summary = prof.summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(r["n_failed"] for r in summary):
        raise RuntimeError("Failed/invalid-shape cells retained; campaign is incomplete")
