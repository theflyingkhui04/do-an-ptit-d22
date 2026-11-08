# M4 component cell diagnostic

## Evidence and diagnosis

Reviewed frozen source `29a10c6e8aa51983f31e2befebc105afd658dfdb` and
evidence commit `9798f071aebfb1b7d70f816dac281b921ffef731`, directory
`experiments/runs/m4_cost_profiling/l4-components-29a10c6-043738/`.
Read the environment, report, manifest, all six logs and per-K environments/raw
files. Aggregate raw is identical to concatenated per-K raw: 3,630 records,
605 successful and 3,025 failed. Hardware and dependency pins match the protocol.

For **each K=0..5**, the first failure is B=2, warmup repeat 0. Original error:
`No cycle with exact requested batch/context/query shape`.
K0 has 55 failed rows per B>=2; each speculative K has 110 (two components).
There are no thrown generation exceptions recorded. Every worker says complete
and exits zero, while the parent fails validation.

The first B=2 trace starts with only one request's prefill, followed by one
decode plus the second request's prefill. The first all-decode step has contexts
`[1025,1024]`, not `[1024,1024]`. K0 has queries `[1,1]`; K1 has `[2,2]`, etc.
No preemption is reported in these first failures. Inference proceeded but the
observer correctly rejected the unequal-context boundary. B=1 avoided this
staggered admission problem. Do not weaken the observer to accept it.

K1's allocator warning is real negative memory-pressure evidence:
`1203765248` bytes requested, `1144127488` free, at log time 05:07:49 UTC.
This lies between the final B16 result at 05:07:43.887909 UTC and the first B32
result at 05:07:54.688305 UTC; it is consistent with the first B32 generation.
It is **not** the first B2 failure (04:58:11.236765 UTC). The log has no
allocation stack, so target/proposer/KV/activation attribution is unavailable.
All later B32 attempts emit shapes, and K1 exits zero. This proves continued
execution, not a clean CUDA health assessment or a permanently poisoned context.
Do not erase this warning or interpret it as latency variance.

The old no-retry comment applies only to exceptions reaching `measure_repeats`.
Missing samples emit errors without raising, so all 55 attempts and later B
values continue. Allocator-internal recovery is not a benchmark retry, and the
old data do not establish that a thrown CUDA OOM was swallowed by this loop.

## Pinned vLLM source audit

Sources are the official `v0.30.0` tag, not latest docs:
<https://github.com/vllm-project/vllm/tree/v0.30.0>.
Downloaded audit copies are under generated evidence
`experiments/runs/m4_cost_profiling_setup/vllm-0.30.0-source-audit/`.
The following SHA256 values exactly match every per-K environment's installed
source hashes:

| File | SHA256 |
| --- | --- |
| `vllm/v1/worker/gpu_model_runner.py` | `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51` |
| `vllm/v1/spec_decode/eagle.py` | `b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378` |
| `vllm/v1/spec_decode/llm_base_proposer.py` | `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab` |

- `vllm/entrypoints/offline_utils.py`,
  `OfflineInferenceMixin._render_and_add_requests`, `_run_completion`: submits
  requests individually before draining outputs. With a background EngineCore,
  a Python list is not an admission barrier.
- `vllm/entrypoints/llm.py`, `LLM.enqueue`, `wait_for_completion`, `sleep`,
  `wake_up`: public APIs supporting a scheduling barrier.
  `enqueue_chat` explicitly documents pause level 0, enqueue, resume scheduling.
- `vllm/v1/engine/core.py`, `EngineCore.sleep`, `pause_scheduler`, `wake_up`:
  level 0 pauses scheduling only, leaves GPU memory unchanged, queues new
  requests; scheduling-only wake resumes without weight/KV offload.
- `vllm/config/scheduler.py`, `SchedulerConfig.verify_max_model_len`: with
  chunked prefill disabled, token budget must be at least max_model_len (4096).
  Thus 4096 is a pinned validation floor, not arbitrary memory tuning.
- `vllm/v1/core/sched/scheduler.py`, `Scheduler.schedule`: token budget limits
  admission; unchunked prompts that exceed the remaining budget wait. Sequence
  limits and KV allocation can also restrict admission or cause preemption.
  Atomic enqueue does not guarantee admission under insufficient KV capacity;
  the unchanged observer must still enforce actual B, query and context shape.
- `vllm/v1/worker/gpu_model_runner.py`, `GPUModelRunner.__init__`:
  max_num_batched_tokens becomes max_num_tokens, max_num_seqs becomes
  max_num_reqs, feeding persistent device buffers and InputBatch capacity.
  `profile_run` invokes `_dummy_run(self.max_num_tokens,is_profile=True)` and
  profiles sampler work; `_dummy_run` also invokes the EAGLE drafter dummy run.
- `vllm/v1/worker/gpu_input_batch.py`, `InputBatch.__init__`: request-sized
  CPU/device tensors, token storage, sampling metadata and block tables depend
  on max_num_reqs/max_num_batched_tokens. These limits are not scheduler-only.
- `vllm/v1/spec_decode/llm_base_proposer.py`,
  `SpecDecodeBaseProposer.__init__`: max_batch_size comes from max_num_seqs;
  max_num_tokens comes from max_num_batched_tokens. Persistent input IDs,
  positions, hidden states, input embeddings and request buffers use these
  capacities. Ordinary EAGLE3 remains sequential, not arbitrary-tree execution.
- `vllm/v1/worker/gpu_worker.py`, `Worker.determine_available_memory`:
  profiles runner memory, subtracts non-KV memory and applicable graph memory
  from the requested utilization budget to determine KV capacity.
  Explicit kv_cache_memory_bytes bypasses this automatic budget and still
  invokes profile_run; it is not used in this fix.
  `compile_or_warm_up_model` also warms sampler buffers with sequence capacity.
- `vllm/v1/core/kv_cache_utils.py`, `get_kv_cache_config_from_groups` and
  `get_kv_cache_configs`: convert available bytes into group/block allocations,
  checking model-length capacity. KV cache is not simply B*context allocation.

K1's original log reports 6.96 GiB model load, 4.98 GiB peak activation,
5.77 GiB KV capacity and 104,384 cache tokens. Reducing token/sequence capacity
reduces unrelated persistent/profile allocations but can increase automatically
allocated KV space within the same 0.85 budget. It does not necessarily create
extra runtime headroom. Actual forward latency may change with backend buffers
or dispatch; the sizing policy is recorded as a protocol change and old/new
measurements must not be pooled as identical engine configurations.

## Minimal implementation and evidence policy

Portable runner only: one fresh process/engine per `(K,B,context)`. Keep
`max_model_len=4096`, utilization 0.85, eager execution and all frozen model,
dtype, quantization and version pins. Set max_num_seqs=B and token capacity to
`max(4096,B*context)`. Queue all B requests while scheduling is paused, then
resume scheduling and drain outputs. The barrier is outside measured forwards.

Per-cell directories/logs are `k{K}-b{B}-c{context}`. Cell environment and every
raw row persist exact engine kwargs, source/model/GPU provenance and boundaries.
Stop on the first failed attempt; do not invent measurements for unattempted
repeat slots. All emitted warmup/measured/error rows are retained. Manifest and
report list planned and attempted counts plus each cell's status, original
error, normalized category and return code. Init failures have cell metadata
even when no timing row exists. Later cells still launch in fresh processes.

Categories include oom, shape_mismatch, scheduler/preemption,
engine_init_failure, runtime_exception and worker_exit. Original tracebacks are
saved in the cell environment/log. Worker execute exceptions log allocator
statistics and traceback before propagation, with no subsequent forward retry.
Worker-local driver free/total and torch allocated/reserved/peak/stats are
captured before measurement and after completed generations, outside timing.
Allocator warnings are also listed separately in cell results; a warning alone
does not prove that a measurement threw OOM. No observer invariant is relaxed.

Canonical topology profiles and gamma calibration remain disabled. The formula
alignment remains TEAM TECHNICAL ALIGNMENT REQUIRED and outside this change.
The source guard still excludes only the two original generated evidence paths.

## Minimal L4 diagnostic (not canonical evidence)

On the pinned Linux L4 environment, from a checkout of the reviewed fix:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 python -m experiments.m4_cost_runner --run-id l4-cell-diagnostic-UNIQUE --expected-gpu L4 --mode smoke --contexts 1024 --k-values 1 --batches 1,2 --warmup 2 --repeats 2
```

Inspect both cell directories, logs, allocator evidence and exact observed
shapes. A B2 OOM or shape failure stops progression to the full grid; retain
that diagnostic and investigate. No full GPU campaign was executed locally.

## Full L4 command (do not run before successful minimal diagnostic)

Commit the reviewed source first: components mode still refuses dirty source.
Use a new unique run ID; evidence directories cannot be overwritten.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 python -m experiments.m4_cost_runner --run-id l4-cell-components-UNIQUE --expected-gpu L4 --mode components --contexts 1024 --k-values 1,2,3,4,5 --batches 1,2,4,8,16,32 --warmup 5 --repeats 50
```

K0 is automatically added as the mandatory target baseline. True infeasible
cells remain negative boundary results; the report fails closed.

## Single-T4 impact

Do not launch a T4 campaign before the L4 diagnostic succeeds. Then use the
same sizing/admission policy, pinned half+awq profile and revisions, and
`CUDA_VISIBLE_DEVICES=0`. The portable CLI is already shared by the thin Colab
launcher. No Kaggle notebook exists in this checkout; no notebook port or T4 run
was performed here. Do not use two GPUs, microbatching, smaller effective B,
different context or memory tuning to manufacture success.

## Local verification

- `python -m pytest tests/test_m4_cost_runner.py tests/test_m4_cost_profiler.py tests/test_m4_colab_preflight.py -q`:
  final focused run **91 passed**.
- `python -m pytest -q`: **1,289 passed, 7 failed, 204 subtests passed**.
  Six failures in test_modal_smoke are existing Windows checkout byte issues:
  the W1 file has CRLF bytes (SHA256 `19808e79c743443bf9c8b989b797becc23c9f0483282b93e2c967f416b493386`),
  whereas the committed LF blob matches the manifest
  (`0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360`).
  Removing CRLF differences produces exactly the HEAD bytes. The seventh,
  test_node1_cli_exposes_ready_timeout, needs missing Modal proxy credentials.
  Relevant source/test dependencies are unchanged from HEAD. Frozen workload
  bytes and credentials were not modified in this M4 fix. The full run began
  before the final five category parametrizations; those pass in the final
  focused run above.
- `python -m pytest tests/test_modal_runner_config.py -q --tb=short`:
  **50 passed**.
- `git diff --check`: passed. All 23 recovered failed-run files were compared
  byte-for-byte with evidence commit 9798f071 and are unchanged.

Source is uncommitted. No new SHA or GPU measurement is claimed.
Readiness: **READY FOR MINIMAL L4 DIAGNOSTIC**.
