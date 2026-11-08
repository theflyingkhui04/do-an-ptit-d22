# M4 CUDA component profiling on Colab or Linux

The CLI is independent of Modal. Measurement logic lives in
`m4_cost_profiler.py` and the worker-local `m4_worker_observer.py`; the notebook
only inspects hardware, checks out source, installs dependencies, invokes the CLI,
and exports artifacts. Use `colab_m4_cost_profile.ipynb` in this directory.

Inspect GPU **before installation or model downloads**:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))'
```

T4 requires observed name Tesla T4 and capability (7,5); L4 requires NVIDIA L4
and (8,9). Colab allocation is not guaranteed. The CLI always observes hardware
again. Mismatch retains all four required artifacts, marks `canonical_hardware=false`,
changes `run_type` to `smoke`, and returns successfully **without loading models**.
It does not attempt a mismatched GPU using the wrong dtype/quantization profile.

Use a fresh Linux Python virtual environment. Do not install the repository's
broad, unpinned requirements files for this experiment:

```bash
git clone --branch feat/m4-02-02-gpu-cost-profiler https://github.com/theflyingkhui04/do-an-ptit-d22.git
cd do-an-ptit-d22
git checkout <reviewed-full-commit-sha>
python -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements-m4-cuda.txt
.venv/bin/python -m pip check
```

If the reviewed commit is still local, make it available on the task branch
before the Colab clone: `git push origin feat/m4-02-02-gpu-cost-profiler`.
The notebook requires REVISION explicitly and does not assume a remote branch
already includes local changes. No automatic push or destructive reset is performed.

The direct stack is vLLM **0.30.0**, torch **2.13.0+cu130**, nvidia-ml-py
**13.580.82**. The existing successful Modal smoke used CUDA 13.0. This is a CUDA
13 stack and needs a compatible host driver; the notebook conservatively requires
driver branch R580 or newer before installing. Do not silently select another
CUDA/vLLM version on older Colab hosts. Stop and retain the hardware inspection.
An actual installation/kernel smoke in Colab is still required; Windows tests
cannot establish that compatibility. The CLI captures resolved `pip-freeze.txt`.

Sources: [vLLM 0.30.0 package metadata](https://pypi.org/pypi/vllm/0.30.0/json),
[vLLM GPU installation and driver guidance](https://docs.vllm.ai/en/v0.30.0/getting_started/installation/gpu/).
Transitive versions come from the pinned vLLM distribution; the resulting freeze
is evidence, not a claim that arbitrary future dependency resolution is identical.

First run: T4 smoke, B=1, K=1, L=256, warmup=2, repeats=2:

```bash
.venv/bin/python -m experiments.m4_cost_runner --run-id UNIQUE-t4-smoke --expected-gpu T4 --mode smoke --contexts 256 --batches 1 --k-values 1 --warmup 2 --repeats 2
```

Only after smoke/installation validation, a **component** campaign is supported:

```bash
.venv/bin/python -m experiments.m4_cost_runner --run-id UNIQUE-t4-components --expected-gpu T4 --mode components --contexts 1024 --batches 1,2,4,8,16,32 --k-values 1,2,3,4,5 --warmup 5 --repeats 50
```

The components command automatically includes the non-speculative K0 target
baseline. K1..5 remain `linear_eagle_k1`..`linear_eagle_k5`, never topology labels.
Use `--expected-gpu L4` on an observed L4; dtype/quantization follow pinned rules
(T4 half/AWQ, L4 bfloat16/AWQ-Marlin). No frontend/HTTP concurrency is used as B.
Component benchmarks require a clean working tree before reserving the run.
Smoke may use dirty source, recorded explicitly; it is never benchmark evidence.

Artifacts remain under `experiments/runs/m4_cost_profiling/<run-id>/`:

- `raw.jsonl`: all warmup and measured records, including failed observations.
- `env.json`: observed hardware/software, source SHA/clean state, fixed grids and boundaries.
- `component_report.json`: component medians/p90/min/max/variance, no topology matrix or gamma.
- `run_manifest.json`: requested execution, worker outcomes and terminal status.
- Per-K `env.json`, `raw.jsonl`, and parent `kK.log`: retain initialization failures/OOMs.

Each K gets a fresh subprocess/engine. There are no automatic retries. Partial
results survive failure. Failed shapes/cells prevent a successful complete report.
Context-specific costs stay separate, including optional linear proposer fits.
The larger token budget in multi-batch runs admits full prompts at the controlled
first-decode boundary; activation/KV OOM is reported rather than reducing B.

Export using ordinary Linux tools, or the notebook's download cell:

```bash
tar -czf m4-smoke-artifacts.tar.gz experiments/runs/m4_cost_profiling/UNIQUE-t4-smoke
```

The prior disabled-chunked-prefill warning remains a limitation of this controlled
protocol. These observations validate component timing, not output correctness,
production graph-enabled throughput, total speculative-cycle latency, or speedup.

**TEAM TECHNICAL ALIGNMENT REQUIRED:** TreeAdaptive `CT*(1+gamma*N/64)` versus
FlexCostModel `CT*(1+factor*max(0,N-1))` remains unresolved. Arbitrary-tree EAGLE
execution is still absent. No canonical gamma or T8/T16/T32 cost is generated;
canonical profile files and the approved acceptance/solver remain unchanged.
