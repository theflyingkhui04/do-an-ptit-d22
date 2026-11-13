# M4-02-06 user-run Colab handoff

LOCAL-ONLY preparation. Codex has not executed these cells. Review and publish
the local task branch before starting; the source commit below must be reachable
from origin. Keep the branch name and frozen source unchanged during a campaign.
Khôi retains final freeze/tag responsibility. No Modal SDK/account is required.

`SOURCE_COMMIT = 187cfff5f8c2a72b9bfb227d97e6221458ddc283`

Create the notebook yourself and select an A100 runtime. Add a minimally scoped
`GITHUB_TOKEN` Colab Secret with read/write Contents access to this repository
and task branch. Add `HF_TOKEN` only if model access requires it. Grant notebook
access to those secrets. No Drive mount is needed. Never paste credentials into
code, output, Git configuration or remote URLs.

## Cell 1 — constants, secrets, two clones and base environment

```python
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys, tempfile
from google.colab import userdata

REPO_URL = "https://github.com/theflyingkhui04/do-an-ptit-d22.git"
BRANCH = "feat/m4-02-06-lossless-sweep-freeze"
SOURCE_COMMIT = "187cfff5f8c2a72b9bfb227d97e6221458ddc283"
RUN_REPO = Path("/content/do-an-ptit-d22-run")
EVIDENCE_REPO = Path("/content/do-an-ptit-d22-evidence")
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
WORK = Path("/content/m4-evidence-work")
SETUP = WORK / f"m4-02-06-a100-setup-{STAMP}-{SOURCE_COMMIT[:8]}"
SETUP.mkdir(parents=True, exist_ok=False)
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
# Optional: omit this try block for public model access.
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except userdata.SecretNotFoundError:
    pass

# Bootstrap only: subsequent pushes use experiments.m4_git_evidence.git_auth.
with tempfile.TemporaryDirectory(prefix="m4-askpass-") as temporary:
    helper = Path(temporary) / "askpass.py"
    helper.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "Username" in sys.argv[1] else os.environ["GITHUB_TOKEN"])\n')
    helper.chmod(0o700)
    auth_env = dict(os.environ, GIT_ASKPASS=str(helper), GIT_TERMINAL_PROMPT="0")
    for checkout in (RUN_REPO, EVIDENCE_REPO):
        assert not checkout.exists(), "Use a fresh clone path; do not delete old evidence"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--branch", BRANCH,
                        REPO_URL, str(checkout)], env=auth_env, check=True)
subprocess.run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=RUN_REPO, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RUN_REPO, text=True).strip() == SOURCE_COMMIT
os.chdir(RUN_REPO)
os.environ["PYTHONPATH"] = str(RUN_REPO)
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
probe = subprocess.check_output([sys.executable, "-c", '''
import json, torch
result = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
              cuda_available=torch.cuda.is_available())
if result['cuda_available']:
    result.update(gpu=torch.cuda.get_device_name(0),
                  capability=list(torch.cuda.get_device_capability(0)),
                  vram_bytes=torch.cuda.get_device_properties(0).total_memory)
print(json.dumps(result))
'''], text=True)
base = dict(json.loads(probe), python=sys.version, nvidia_smi=smi.stdout, source_commit=SOURCE_COMMIT)
(SETUP / "before-install.json").write_text(json.dumps(base, indent=2))
print(smi.stdout)
print({k:v for k,v in base.items() if k != "nvidia_smi"})
assert base["cuda_available"] and "NVIDIA" in base["gpu"] and "A100" in base["gpu"] and base["capability"] == [8,0], "A100_GATE FAIL — STOP"
```

If cloning/authentication fails, fix the user-owned secret/access without
printing the token. The temporary askpass file is removed even on exceptions.
No source revision may be substituted merely because a clone cannot find it.

## Cell 2 — isolated pinned runtime and installation inventory

```python
VENV = Path("/content/m4-02-06-venv")
assert not VENV.exists(), "Preserve an existing environment; use a fresh path"
subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
PY = str(VENV / "bin/python")
os.environ["PATH"] = str(VENV / "bin") + os.pathsep + os.environ["PATH"]
def run_module(module, *arguments, check=True, log=None):
    command = [PY, "-B", "-m", module, *map(str, arguments)]
    if log:
        with Path(log).open("x") as stream:
            return subprocess.run(command, cwd=RUN_REPO, stdout=stream,
                                  stderr=subprocess.STDOUT, check=check)
    return subprocess.run(command, cwd=RUN_REPO, check=check)
run_module("pip", "install", "--index-url", "https://download.pytorch.org/whl/cu130", "torch==2.13.0+cu130")
run_module("pip", "install", "--index-url", "https://pypi.org/simple", "-r", "experiments/requirements-m4-cuda.txt")
run_module("pip", "install", "aiohttp", "pytest", "psutil")
run_module("pip", "check")
with (SETUP / "pip-freeze.txt").open("x") as stream:
    subprocess.run([PY,"-m","pip","freeze"], stdout=stream, check=True)
run_module("experiments.run_m4_colab_campaign", "source-gate", "--source-commit", SOURCE_COMMIT)
run_module("experiments.m4_a100_profile", "environment", "--source-commit", SOURCE_COMMIT,
           "--out", SETUP / "after-install")
```

The environment action persists Python, driver, actual VRAM, GPU/capability,
PyTorch/CUDA and vLLM identity. `pip-freeze.txt` records transformers,
huggingface_hub and all transitive versions. A Python, wheel, driver or CUDA
incompatibility is a technical FAIL: preserve the setup log and stop. Do not
upgrade vLLM or silently substitute another PyTorch/CUDA stack.
Hardware inspection runs in short-lived subprocesses so the notebook and
orchestrator do not retain CUDA contexts alongside the serving process.

## Cell 3 — pinned source, CPU tests, manifest and model revisions

```python
source_root = subprocess.check_output([PY,"-c",
    "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent.parent)"], text=True).strip()
os.environ["M4_VLLM_SOURCE_ROOT"] = source_root
tests = ["test_budget_planner", "test_adaptive_scheduler_glue", "test_modal_runner_config",
         "test_modal_bench_jobs", "test_lineage_startup", "test_vram_memory_lifecycle",
         "test_budget_controller", "test_proposer_adapter", "test_lossless_verifier",
         "test_aggregate_results", "test_benchmark_profiles", "test_m4_cost_profiler",
         "test_m4_cost_runner", "test_cost_model_l4", "test_cost_model_solver",
         "test_m4_treatments", "test_m4_pinned_contract", "test_m4_lossless",
         "test_m4_lifecycle", "test_m4_a100_profile", "test_m4_campaign", "test_m4_readiness"]
run_module("pytest", "-p", "no:cacheprovider", "-q",
           *[f"tests/{t}.py" for t in tests], log=SETUP / "cpu-tests.log")
run_module("experiments.run_m4_colab_campaign", "freeze", "--source-commit", SOURCE_COMMIT,
           "--out", SETUP / "frozen_54_cells.json")
run_module("experiments.run_m4_colab_campaign", "models", "--source-commit", SOURCE_COMMIT,
           log=SETUP / "model-revisions.log")
```

The pinned-source test must run, not skip. Workload files must match Git blob
hashes. Do not repair mismatches by normalizing frozen files. Keep the setup
directory for failure publication even if a later stage fails.

## Cell 4 — measure and validate the actual sm80 profile

```python
PROFILE_RUN = WORK / f"m4-02-06-a100-profile-{STAMP}-{SOURCE_COMMIT[:8]}"
run_module("experiments.m4_a100_profile", "measure", "--source-commit", SOURCE_COMMIT,
           "--out", PROFILE_RUN)
PROFILE = PROFILE_RUN / "profile_sm80.json"
run_module("experiments.m4_a100_profile", "validate", "--source-commit", SOURCE_COMMIT,
           "--out", PROFILE)
```

This is measurement, not a specification-table profile. It starts fresh worker
processes sequentially for 30 B/K calibration cells, capturing exact shapes,
5 warmups and 50 repeats each. It may be expensive: do not run it before all
preceding gates pass. A failed worker stops profiling and retains failure logs;
assess it before allocating another attempt. No canonical profile is written on
incomplete evidence. Cost boundaries exclude target logits/sampling and do not
prove hardware-cost-aware scheduling; see the methodology document.

## Cell 5 — diagnostic, publish even failure, then readiness

Configure the Git author identity in EVIDENCE_REPO yourself if not already set
(`git config user.name`, `git config user.email`). These identify the evidence
author; do not use a token as either value.

```python
DIAGNOSTIC = WORK / f"m4-02-06-a100-diagnostic-{STAMP}-{SOURCE_COMMIT[:8]}"
diagnostic_result = run_module("experiments.run_m4_colab_campaign", "diagnostic",
    "--source-commit", SOURCE_COMMIT, "--profile", PROFILE, "--out", DIAGNOSTIC,
    "--full-lossless", check=False)
# Preserve setup and profile inputs together with all diagnostic successes/failures.
sys.path.insert(0, str(RUN_REPO))
from experiments.m4_git_evidence import copy_immutable
if not DIAGNOSTIC.exists():
    # A preflight failure may have happened before directory reservation.
    run_module("experiments.m4_git_evidence", SETUP, "--evidence-repo", EVIDENCE_REPO,
               "--source-commit", SOURCE_COMMIT)
    raise RuntimeError("Diagnostic preflight failed; setup persisted. STOP")
copy_immutable(SETUP, DIAGNOSTIC / "setup")
run_module("experiments.m4_git_evidence", DIAGNOSTIC, "--evidence-repo", EVIDENCE_REPO,
           "--source-commit", SOURCE_COMMIT)
EVIDENCE_COMMIT = subprocess.check_output(["git","rev-parse","HEAD"], cwd=EVIDENCE_REPO, text=True).strip()
print("SOURCE_COMMIT", SOURCE_COMMIT, "EVIDENCE_COMMIT", EVIDENCE_COMMIT)
run_module("experiments.m4_readiness", DIAGNOSTIC, "--source-commit", SOURCE_COMMIT)
assert diagnostic_result.returncode == 0, "Diagnostic failed — STOP"
```

The diagnostic includes two Vanilla captures, fixed and Adaptive captures,
native drafted/accepted counters, real proposal-buffer observations, prior-plan
feedback, controller changes, cleanup, process logs and token-ID comparison.
`--full-lossless` compares all 54 corpus prompts; omit it only for the default
6-prompt minimal diagnostic and record that scope. Constant K is insufficient:
if natural execution does not evidence two meaningful K states, readiness FAILS.
Preserve that negative result and review a separate diagnostic scenario before
rerunning. Unit-test K transitions cannot substitute for these observations.

Failed diagnostics must also be published. A failure is not permission to run
the campaign. `SOURCE_COMMIT` never becomes the newer `EVIDENCE_COMMIT`.

## Cell 6 — full sweep, only after PASS

```python
CAMPAIGN = WORK / f"m4-02-06-a100-campaign-{STAMP}-{SOURCE_COMMIT[:8]}"
run_module("experiments.run_m4_colab_campaign", "campaign", "--source-commit", SOURCE_COMMIT,
    "--profile", PROFILE, "--diagnostic", DIAGNOSTIC, "--out", CAMPAIGN,
    "--evidence-repo", EVIDENCE_REPO)
```

The driver independently re-evaluates raw readiness and verifies that the exact
diagnostic files are on the remote task branch. It publishes the manifest and
each cell's start and terminal snapshots, runs one server at a time, and stops
at the first invalid/failed/OOM/timeout cell. No values or failed attempts are
deleted. A residual process or unexplained VRAM increase blocks the next cell.

## Cell 7 — resume after interruption

Recreate the isolated runtime and clean detached RUN_REPO at the SAME
SOURCE_COMMIT. Fetch the task branch into EVIDENCE_REPO. Restore the exact
diagnostic and campaign directories from
`EVIDENCE_REPO/experiments/runs/m4_02_06/` to fresh external WORK paths using
`copy_immutable`; restore PROFILE_RUN from diagnostic `profile_inputs`.
Set DIAGNOSTIC, PROFILE, CAMPAIGN to these restored paths. Then:

```python
run_module("experiments.run_m4_colab_campaign", "resume", "--source-commit", SOURCE_COMMIT,
    "--profile", PROFILE, "--diagnostic", DIAGNOSTIC, "--out", CAMPAIGN,
    "--evidence-repo", EVIDENCE_REPO)
```

Only revalidated COMPLETE_VALID cells are skipped. Abandoned RUNNING or failed
attempts stop this command. An intentional rerun requires `--rerun-cell CELL_ID
--rerun-assessment /content/assessment.json`. The assessment must include all
`prior_attempts` IDs and nonempty `previous_failure`, `root_cause_hypothesis`,
`offline_fix_evidence`, `expected_pass_fail_signal`. Preserve original attempts.
Do not mix source commits after an implementation change: start a new campaign.

## Cell 8 — complete validation, aggregation and publication

```python
run_module("experiments.run_m4_colab_campaign", "aggregate", "--source-commit", SOURCE_COMMIT,
           "--out", CAMPAIGN)
run_module("experiments.m4_git_evidence", CAMPAIGN, "--evidence-repo", EVIDENCE_REPO,
           "--source-commit", SOURCE_COMMIT)
```

Aggregation revalidates every one of the 54 exact cells and their raw hashes;
an incomplete campaign cannot produce a canonical summary. The new output is
`analysis-<unique-id>/summary_eagle_adaptive_m4.csv`, with validation/raw hashes,
repeat variance flags and paired-Vanilla TPOT speedup denominator.

After schema/raw validation, an optional reviewed repository profile copy goes
to `EVIDENCE_REPO/src/flexsds/profiles/profile_sm80.json` (exact-path staging),
never into the detached RUN_REPO. This is a separate evidence/review addition,
not a change to the running SOURCE_COMMIT.

Hand Khôi the source/evidence SHAs, profile path/hash, diagnostic/campaign IDs,
manifest, per-attempt states, complete/failed/OOM/timeout lists, telemetry/logs,
validated summary and rerun assessments. No freeze tag is created by this flow.
