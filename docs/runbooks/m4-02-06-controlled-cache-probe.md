# M4-02-06 controlled Vanilla cache probe

Implementation readiness is separate from scientific readiness. This tooling
does not repair lossless generation or alter `ADAPTIVE_READY`. No GPU run has
been performed during implementation. Run only after technical review and
publication of an exact diagnostic commit.

## Revision baseline

Base: `8bcd86cd8965e55ef87f312bf42f72efa5da2006` (`develop`, PR #44).
Historical executable source: `187cfff5f8c2a72b9bfb227d97e6221458ddc283`.
Historical evidence: `bb16ed4feb657f6c9bfb8e92fb9a89c96fa2c43f`, under
`experiments/runs/m4_02_06/m4-02-06-a100-diagnostic-20261113T105105Z-187cfff5/`.

Between frozen source and the base, `m4_treatments`, `m4_runtime`,
`workload_adapter`, `m4_protocol`, `m4_readiness`, `src/server` and
`src/proposer` have no changes. The campaign/profile CLI changes are security
validation only. Later FlexSDS CLI, compatibility and hardware-profile loader
changes are runtime-semantic changes in that separate path; this target-only
probe neither invokes that CLI nor consumes its profile/compatibility policy.
Diagnostic/freezing evidence and tests also changed. No relevant target-only
scientific behavior changed. The existing dedicated CUDA requirement pins are
reused unchanged.

## Historical cache evidence

The preserved `vanilla/server.log` engine initialization explicitly reports
`enable_prefix_caching=True`, `enable_chunked_prefill=True`, V1 async scheduling,
BF16, AutoAWQ/Marlin, max model length 4096 and max sequences 32. Frozen
`run_m4_colab_campaign.diagnostic` keeps one Vanilla server alive across all
54 prompts of repeat 0, then all 54 prompts of repeat 1. `m0_04_001` is the
first prompt in each repeat. Thus its original prompt could be reused on
repeat 1. Historical per-request cache-hit counts are UNKNOWN.

Cache keys use chained block hashes of token IDs plus optional LoRA,
multimodal, cache-salt and prompt-embedding keys (`kv_cache_utils`, lines
600–664). This text-only, single-model, single-engine probe adds no cache salt,
LoRA, embeddings or unrelated traffic. Sampling seed/request ID are not token
block hash components. Only complete blocks are reusable. The final input
token must be recomputed for logits (`KVCacheManager.get_computed_blocks`,
lines 265–321).

## Measurement and conditioning

`load_input` reads the evidence commit with `git show`, validates recorded raw
hashes and the frozen workload request hash, selects Vanilla repeat 0, and
concatenates original prompt token IDs with generated IDs `[:549]`. It never
encodes decoded text. The two Vanilla repeats must share that exact prefix.

For W1 / `m0_04_001`, input length is **2071 = 1522 + 549**. SHA256 is over the
compact ASCII JSON integer array (no whitespace or terminal newline):
`3f52b8b8ee17eca82386c6e5ab6fa0510ee089e71dbffe60cf0efcabdd5fc75a`.
Historical next tokens are 4164 and 3752.

Both conditions use enabled prefix caching and identical engine flags, input
tokens and sampling parameters. The historical A100 server's 2048-token batch
budget is made explicit:
pinned `EngineArgs` defaults differ for `LLM_CLASS` (8192) and
`OPENAI_API_SERVER` (2048). Max sequences remains 32. Resolved compilation,
CUDA graph sizes, async/chunked-prefill flags and precision must match the
preserved Vanilla configuration or the diagnostic stops. COLD requires a
successful logical reset and request-local lookup returning zero cached tokens.
CACHE_HIT resets again,
conditions **only the original 1522-token prompt**, then submits the same
2071-token forced input. The conditioning request is aborted after its first
observed output; that output is recorded but is not used to construct input.
The measured request must reuse >0 and <=1522 tokens. Hits extending into the
generated prefix are rejected. With 16-token blocks, 1520 reused tokens would
leave 551 query tokens; this is an illustration, not an assumed runtime value.
Actual block size, block IDs, hit extent, query length and computed-token count
are recorded. COLD and CACHE_HIT are each repeated once in the same process,
with an explicit reset between conditions.

The original temperature=0, seed=42, n=1, max_tokens=1024, default EOS and
JSON-object settings remain. The generated prefix is supplied as input rather
than sampled again. The diagnostic scheduler advances each compiled JSON
grammar through those exact 549 emitted IDs once, restoring the grammar state
at the boundary. Failure to accept the prefix stops the probe. It does not
change cache lookup results, scheduler budgets, sampling tensors or token
selection. Requests abort after the first observed output, preserving sampling
parameters while limiting work. Async work already in flight can finish;
subsequent cache-hit extent is verified, never inferred from abort intent.

The generated continuation is **prefilled**, not recreated by the historical
sequence of decode forwards. This experiment tests cache-dependent target
numerics under a controlled identical prefix. It cannot alone establish why
the original decode trajectory diverged. A successful logical cache reset is
also not a fresh process, allocator or compiler state. The second pair checks
within-condition repeatability to expose first-call/order effects.

The raw measurement wraps `model.compute_logits` at its return from
`GPUModelRunner.execute_model` (lines 4497–4498). Qwen2's head projection uses
`LogitsProcessor` with scale=1 and no soft cap; the worker verifies these
properties. This is before grammar, penalties, temperature and sampling.
An additional snapshot at `_sample` entry records the grammar-masked argmax.
The original tensors and method return values are preserved. No cache/model
implementation or installed source file is patched. Default serving never
selects these classes or installs these wrappers.

An engine lifetime guard restores all four worker instance methods and the
owning scheduler's cache observer on success, capture/collection failure, and
partial hook installation. Restoration preserves preexisting instance
overrides and removes overrides of inherited methods. Pending GPU snapshots
and CPU comparison vectors are released. A weak registry ties the observer
to the exact shared UniProc engine configuration; unrelated engines are never
restored by this worker. Scheduler shutdown also restores its observer.
The guard calls the pinned CoreClient shutdown API even if restoration fails,
records `hook-cleanup.json`, and fails the run if restoration is unconfirmed.
The parent still enforces process-group termination and GPU release. These
CPU-tested cleanup contracts do not establish GPU execution correctness.

The hook clones two single-boundary vocabulary rows and the query token-ID
suffix on GPU. The actual query IDs must equal the preserved suffix implied by
the captured computed-token count; its hash is recorded after abort. This adds a small
amount of work/storage and may affect timing; this is not a performance test.
There is no `.cpu()`, `.item()` or explicit synchronization in the live capture
hook. CPU transfer/reductions occur through a worker RPC after abort. Full
vectors exist only temporarily in worker memory, for max/mean absolute
differences across conditions/repeats; they are not serialized. Artifacts store
top 20, both tracked logits, margins, dtype and finite-value counts.

## Pinned API/source contract

`experiments/m4_cache_probe_sources.json` records exact v0.30.0 source hashes,
URLs, hook locations, semantics and side effects. Unknown source bytes fail
before engine instrumentation. Relevant pinned APIs:

- `Qwen2ForCausalLM.compute_logits` and model `LogitsProcessor`: head logits.
- `GPUModelRunner.execute_model` / `sample_tokens`: raw then grammar-masked stage.
- `Sampler.greedy_sample`: argmax, not API logprobs.
- `AsyncScheduler._update_after_schedule`: grammar-state restoration boundary.
- `KVCacheManager.get_computed_blocks`: actual request-local hit observation.
- `LLM.reset_prefix_cache`: explicit reset; False is fatal.
- `LLMEngine.add_request`, `step`, `abort_request`: integer-token input/lifecycle.
- `WorkerWrapperBase.init_worker` / `LLM.collective_rpc`: explicit extension.

Reviewed upstream tests include
[test_reset_prefix_cache](https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/core/test_prefix_caching.py#L3128)
and
[test_reset_prefix_cache_e2e](https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/core/test_reset_prefix_cache_e2e.py#L15).
These were inspected, not run locally against a GPU.

## Colab execution after publication

Select A100. Replace only `DIAGNOSTIC_COMMIT` with the reviewed/published full
SHA. Use fresh paths. Public Git/model access requires no embedded credentials;
use Colab Secrets/environment if access policy requires credentials.

```python
from pathlib import Path
from datetime import datetime, timezone
import os, re, subprocess, sys

DIAGNOSTIC_COMMIT = "REPLACE_WITH_REVIEWED_PUBLISHED_40_HEX_SHA"
assert re.fullmatch(r"[0-9a-f]{40}", DIAGNOSTIC_COMMIT)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
repo = Path(f"/content/m4-cache-probe-source-{stamp}")
venv = Path(f"/content/m4-cache-probe-venv-{stamp}")
work = Path("/content/m4-evidence-work")
out = work / f"m4-02-06-cache-probe-{stamp}-{DIAGNOSTIC_COMMIT[:8]}"
assert not repo.exists() and not venv.exists() and not out.exists()
subprocess.run(["git", "clone", "--no-checkout",
    "https://github.com/theflyingkhui04/do-an-ptit-d22.git", str(repo)], check=True)
subprocess.run(["git", "fetch", "origin", DIAGNOSTIC_COMMIT], cwd=repo, check=True)
subprocess.run(["git", "checkout", "--detach", DIAGNOSTIC_COMMIT], cwd=repo, check=True)
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()==DIAGNOSTIC_COMMIT
assert not subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()
# Inspection only; do not install an incompatible runtime on a non-A100 VM.
subprocess.run([sys.executable,"-c", "import torch; assert torch.cuda.is_available(); "
    "assert 'A100' in torch.cuda.get_device_name(0); "
    "assert torch.cuda.get_device_capability(0)==(8,0)"], check=True)
subprocess.run([sys.executable,"-m","venv",str(venv)],check=True)
py = str(venv / "bin/python")
def run(*args):
    subprocess.run([py,"-B",*map(str,args)],cwd=repo,env=env,check=True)
env = dict(os.environ, PYTHONPATH=str(repo)+os.pathsep+str(repo/"src"),
           PYTHONDONTWRITEBYTECODE="1", M4_ARTIFACT_ROOT=str(work))
run("-m","pip","install","--index-url","https://download.pytorch.org/whl/cu130","torch==2.13.0+cu130")
run("-m","pip","install","--index-url","https://pypi.org/simple","-r","experiments/requirements-m4-cuda.txt")
run("-m","pip","install","pytest")
run("-m","pip","check")
run("-m","pytest","-q","tests/test_m4_cache_probe.py","tests/test_m4_readiness.py",
    "tests/test_m4_lossless.py","tests/test_m4_treatments.py")
work.mkdir(exist_ok=True)
run("-m","experiments.run_m4_cache_probe","prepare","--source-commit",DIAGNOSTIC_COMMIT,"--out",out)
run("-m","experiments.run_m4_cache_probe","run","--source-commit",DIAGNOSTIC_COMMIT,"--out",out)
```

The worker validates exact vLLM/PyTorch/CUDA dependencies, A100/sm80, installed
source hashes, clean Git identity, model revision and prepared evidence before
model execution. It records actual driver/GPU/Python drift against historical
Colab instead of claiming equality. Managed-environment drift limits historical
attribution. Nothing measures a profile or launches EAGLE/Adaptive/a sweep.

## Evidence schema and decisions

The executable schema is `m4_cache_probe.validate_result`, version
`m4.cache_probe.v1`. Every measured condition includes executed source and
instrumentation SHA, historical source SHA, run/timestamp, actual environment,
model/revisions, prompt hash/index/input hash, verified cache evidence, raw
logits/top-N/margins and command/engine/sampling provenance. Missing required
values, nonfinite logits, mismatched source/input, failed reset, unverified hit,
grammar-state mismatch, worker/lifecycle error or unresolved analyzer result
exits nonzero.

New directories/files use existing `artifact_path`/`write_new` conventions.
`input.json` contains exact token inputs, settings and raw-evidence hashes;
`cache.jsonl` records request-local lookup and grammar restoration;
four condition JSONs contain observations; vector-difference JSONs contain only
statistics; `environment.json`, `runtime-sources.json`, installation/launch,
worker log, lifecycle and final SHA256 manifest preserve provenance. A run
cannot overwrite a previous run. Preserve failures as well as successes.

- Repeated cold 4164 / hit 3752, same input/runtime, verified original-prompt
  reuse and stable within-condition vectors: `H_CACHE_SUPPORTED`.
- Same argmax but changed logits: `INCONCLUSIVE`; report actual margins and
  differences for review. No universal numerical tolerance is invented.
- Identical raw vectors/argmax across both pairs: `H_CACHE_NOT_REPRODUCED`.
- Unverified hit, mismatched environment/input/stage or unstable repeats:
  `INCONCLUSIVE`. Never label any single diagnostic `ROOT_CAUSE_PROVEN`.

Offline analysis (same reviewed source checkout, no GPU):

```text
python -m experiments.run_m4_cache_probe analyze --source-commit <diagnostic-sha> --out <run-directory>
```

Full-sweep readiness remains blocked pending a proven repair and a fresh passing
canonical readiness diagnostic. Review this diff and CPU evidence before any
publication or Colab execution.

## Audited A100 cache probe and remaining evidence limit

Reviewed source: `873cf2c373411e6f0ca6c61dbe9798e5dc5e2f30`.
Run: `m4-02-06-cache-probe-20261113T234943682317Z-873cf2c3`.
The supplied tar.gz SHA256 is
`4d6d56c0fc7f64e60c293e5fb32b5b9bdf9e07408873a649b38cc90a9fe3b209`.
All 24 manifest-listed files match their SHA256 and inventory. The prepared
input matches `load_input()` from frozen historical evidence byte-for-byte at
the integer-array level. No original artifact was edited.

| Condition (both repetitions) | Raw argmax recorded / winner in saved top 20 | 4164 | 3752 | Top-1 minus top-2 | Actual reused tokens | Computed / query tokens |
| --- | --- | --- | --- | --- | --- | --- |
| COLD | 3752 | 26.75 | 26.75 | 0 | 0 | 2048 / 23 |
| CACHE_HIT | 3752 | 26.625 | 26.75 | 0.125 | 1520 | 1520 / 551 |

Both raw and grammar-masked recorded argmaxes are 3752 in all four conditions.
The cold tie is exact at captured BF16 precision. The 1520-token hit is verified
from request-local `cache.jsonl` as 95 blocks of 16, not from request intent.
The different query shapes are observed, not inferred kernel causality.

Both vector-difference JSONs record max=0.171875 and
mean=0.028460560366511345 over 152064 entries. Both within-condition repeat
records report max=mean=0, and the complete saved logit-summary objects are
identical within each condition. **The four original full vectors are absent.**
The original collector reduces them in memory without serializing them.
Independent full-vector max/mean/repeat/global-argmax recomputation is therefore
unavailable: the historical full-vector audit remains incomplete and blocked.
Top-20 observations and saved reductions must not be presented as reconstructed
full tensors. The operator confirmed no separately preserved vectors are known.
This limitation does not prevent CPU/static design of the next diagnostic.

The unchanged analyzer at that SHA follows `analyze()` lines 228–235:
`same=True`, `changed=True`, hence the explicit same-argmax/changed-logits
`INCONCLUSIVE` branch. Validation has no reasons; both pairs agree and
`analyze_replays()` reports exact repeatability. Re-executing it on the preserved
JSON reproduces `analysis.json` exactly. The worker deliberately raises at
`run_m4_cache_probe.py:255`. `hook-cleanup.json` confirms restoration and engine
shutdown; lifecycle memory returns to 6 MiB with no GPU processes. This was a
completed measurement followed by an intentional failure status, not an
infrastructure crash. Recorded environment drift is empty, and the preserved
runtime-source manifest equals the source manifest at the executed SHA.

Supported observation: cache reuse changes the saved numerical logit evidence
under an identical forced prefix. Cache causation of the historical token flip
is unproven because even cold forced prefill selects 3752.
`ROOT_CAUSE_NOT_PROVEN`; `FULL_SWEEP_BLOCKED`.

## Minimal execution-history follow-up (disabled by default)

Opt in only through `prepare --diagnostic-mode decode_trajectory`. Existing
cache-probe mode and canonical serving retain their defaults. The trajectory
implementation is separated into `experiments/m4_decode_trajectory.py` because
sequential first-mismatch capture has different lifecycle/state contracts from
the original single forced-boundary collector. It reuses frozen input loading,
engine/environment/source gates, cache observation, forced grammar seeding,
instance-hook restoration, append-only writes and parent process/GPU cleanup.

Only W1 / m0_04_001 / target index 549 is supported. Two conditions are necessary:

1. `SEQUENTIAL_DECODE_COLD`: reset the logical prefix cache; submit only the
   original 1522 prompt IDs with the historical greedy seed/max_tokens/EOS and
   JSON-object parameters. Sample naturally. Never prefill the historical
   generated prefix, seed sequential grammar, overwrite sampled IDs or alter KV.
2. `FORCED_PREFILL_COLD`: reset again; submit original prompt plus the reference
   generated prefix at the selected boundary. Restore grammar using the reviewed
   forced-prefix mechanism. Require actual request-local lookup count zero.

No optional sequential cache-hit condition is included: it would not resolve
the primary difference between sequential cold and forced-prefill cold.

The raw measurement remains `model.compute_logits` return. `_sample` entry
captures grammar-masked logits, and `_sample` return supplies the actual token.
A constant-size device latch compares that token with reference[i] and retains
the FIRST mismatch, or boundary 549 if none occurred earlier. Raw and masked
vocabulary rows stay on GPU; there is no per-token full-vector host transfer or
device-to-host decision synchronization. Transient GPU clones/device selection
and CPU shape/mask bookkeeping add diagnostic overhead. This is not timing data.

The host compares cumulative delivered token IDs and aborts immediately at the
first delivered mismatch/target. Async work can already be in flight. The device
latch cannot be overwritten by that later work; any delivered suffix is recorded
separately and excluded. No historical tokens are forced after a mismatch.
Boundary indices must be contiguous. Recomputations/preemptions, missing hooks,
unexpected batching, premature EOS, unknown grammar-state counters and input
identity inconsistencies fail closed. There is no automatic retry.

If the first mismatch is index j<549, freeze A at j and evaluate B using
`original + reference[:j]`, not index 549. The result is
`EARLIER_DIVERGENCE_OBSERVED`, with trajectory status
`HISTORICAL_TRAJECTORY_NOT_REPRODUCED`. It cannot support an index-549 claim.

Collection occurs by worker RPC after abort, using the existing UniProc
ownership/barrier contract. Actual GPU query-ID snapshots reconstruct the full
logical input; they must equal original+reference[:j]. Store prompt, generated
prefix and full logical token arrays, their compact-JSON hashes and index.
Prefix hashes at index 549 are:

- Prompt: `ac2940b3044c3e8bd563ebbffe5d88b207c7d144375027d97fb378fa2dc55a81`.
- Generated: `ba579161a70b4aa19b19b4262ceb73c1c540dbf51ce1bce4c0dfdf75e752622b`.
- Full logical: `3f52b8b8ee17eca82386c6e5ab6fa0510ee089e71dbffe60cf0efcabdd5fc75a`.

The new `m4.decode_trajectory.v1` condition JSONs serialize **all 152064 raw
logits**, as float32 JSON values that losslessly represent captured BF16.
Masked vectors are serialized too; JSON null explicitly means negative infinity.
They also include raw/masked argmax, sampled/historical token, tracked competing
logits (including the actual first-divergence candidates), top 20, margins,
dtype, query/computed counts, cache block IDs, actual graph mode/padding and
request-owned grammar bitmask bytes/shape/hash. Unvalidated `worker-*.json`
snapshots preserve returned vectors before subsequent provenance/schema guards.
Missing evidence/guard failure produces `INCONCLUSIVE` and a nonzero worker exit.

The historical log records grammar backend configuration `auto`; it does not
prove the resolved backend or historical internal matcher state. Do not relabel
that as observed XGrammar. The follow-up records the resolved backend and grammar
classes, processed-token count, reasoning gate and actual effective bitmask at
mask preparation. Natural A and seeded B must have equal grammar class/backend,
prefix count and mask. With no reasoning parser, the initial reasoning-ended
flag may differ at index zero without changing the effective mask; later
boundaries require the flag to be true. An unsupported/missing processed-token
counter fails closed instead of changing the backend configuration.

### Pinned runtime trace for trajectory instrumentation

Every source below is v0.30.0; added installed-source hashes are in the existing
source manifest. Current/latest docs are not used as implementation proof.

| File | Function/location | Relevant invariant |
| --- | --- | --- |
| `vllm/v1/core/sched/scheduler.py` | `schedule` 561; `_update_after_schedule` 1520 | Token accounting unifies prefill/decode; advance computed counts after scheduling; chunk state includes async placeholders. |
| `vllm/v1/core/sched/async_scheduler.py` | `_update_after_schedule` 19; `_update_request_with_output` 51 | Placeholder counts are not emitted IDs; cache only computed non-placeholder tokens. |
| `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` 265; `allocate_slots` 370 | Only complete prefix blocks hit; recompute final token for logits; slot allocation tracks computed/new/lookahead regions. |
| `vllm/v1/worker/gpu_model_runner.py` | `_prepare_inputs` 1971; `_build_attention_metadata` 2297; `_get_slot_mappings` 4103 | Actual query IDs/positions depend on computed offsets; slot tables map into KV; CPU token tables can contain async placeholders. Capture actual GPU inputs. |
| same runner | `_determine_batch_execution_and_padding` 3953; `execute_model` 4282–4469 | Uniform-decode/shape selects graph mode and padding before forward; observe its original return without changing it. |
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward` 1166; `do_kv_cache_update` 1438 | Attention reads cached K/V; slot mapping drives `reshape_and_cache_flash` writes. No kernel defect is inferred. |
| `vllm/model_executor/models/qwen2.py` | `Qwen2ForCausalLM.compute_logits` 511 | Head projection calls model LogitsProcessor; reviewed return remains the raw measurement point. |
| same runner | `sample_tokens` 4566–4605; `_sample` 3658 | Grammar mutates logits before `_sample`; its return contains actual GPU sampled IDs; retain original tensors/returns. |
| `vllm/v1/sample/sampler.py` | `forward` 73; `sample` 244 | Sampler converts to float32/processes logits; greedy branch returns argmax. Raw head logits and masked decisions are distinct. |
| `vllm/v1/structured_output/__init__.py` | `grammar_init` 115; `grammar_bitmask`; `accept_tokens` 438 | Resolved backend creates grammar; mask derives from current grammar; committed generated tokens advance grammar. |
| `vllm/v1/structured_output/backend_xgrammar.py` | `accept_tokens` 157; `fill_bitmask` 208 | Processed-token count advances per accepted token; next mask derives from matcher state. Unsupported state is not guessed. |
| `vllm/model_executor/layers/quantization/auto_awq.py` | `AutoAWQMarlinLinearMethod.create_weights` 432–447; `apply` 512 | Quantized kernel selection is an alternative numerical contributor; this diagnostic changes none of it. |
| `vllm/v1/engine/input_processor.py` / `output_processor.py` | `assign_request_id` 266–279 / request construction 384 | Internal ID returned by add_request differs from the external label returned in RequestOutput. Abort/capture/cache use internal ownership. |

Pinned upstream reset-cache tests listed above were inspected, not executed.
They establish reset contracts, not GPU numerical equivalence.

### Decision precedence

First reject instrumentation/provenance/prefix/grammar mismatches as
`INCONCLUSIVE`. Then classify an earlier mismatch as
`EARLIER_DIVERGENCE_OBSERVED`. At 549, identical vectors and argmax select
`H_TRAJECTORY_NOT_SUPPORTED` (case D takes precedence over overlapping case B).
Otherwise sequential raw/sample=4164 versus forced raw/sample=3752 selects
`H_TRAJECTORY_SUPPORTED`; sequential raw/sample=3752 selects
`HISTORICAL_TRAJECTORY_NOT_REPRODUCED`. Other combinations remain
`INCONCLUSIVE`. No tolerance is invented. Every outcome retains
`ROOT_CAUSE_NOT_PROVEN` and `FULL_SWEEP_BLOCKED`.

H_TRAJECTORY is execution-history dependence at identical logical input, not an
assertion that a particular KV/kernel component is defective. Alternatives remain
quantized GEMM/attention shape, chunked prefill, CUDA graphs, allocator/block
history, first-request/compiler state, grammar state and scheduler/batching.
Equal engine flags alone do not make those histories identical.

### Review gate and future invocation

The initial implementation phase stopped at **LOCAL FOLLOW-UP DIAGNOSTIC READY
FOR REVIEW** after CPU verification. The subsequent explicitly authorized final
review permits a new commit and normal push after all publication gates pass.
No GPU, Colab, Modal, sweep, PR or Trello operation is authorized in that review.
Only after publication and another run review, with the same pinned
environment and fresh external artifact directory, prepare with:

```text
python -B -m experiments.run_m4_cache_probe prepare --diagnostic-mode decode_trajectory --source-commit <reviewed-published-sha> --out <fresh-run-directory>
```

The existing `run` command reads the prepared mode from input.json. The existing
`analyze` command verifies the final manifest and independently reduces the new
saved full vectors; it does not mutate artifacts. Source cleanliness prevents
running the current uncommitted patch as immutable GPU evidence.

CPU verification (no claim of GPU numerical correctness):

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -B -m pytest -q -rs -p no:cacheprovider tests/test_m4_decode_trajectory.py tests/test_m4_cache_probe.py tests/test_m4_readiness.py tests/test_m4_lossless.py tests/test_m4_treatments.py tests/test_m4_lifecycle.py
git diff --check
```

Initial implementation baseline: **143 passed, 4 skipped, 0 failed**. The skipped tests are
`test_m4_treatments.py:49,70,77` and `test_m4_lifecycle.py:9`; their optional
preserved installed-source cache is unavailable. New trajectory hook, teardown,
first-mismatch and serialization tests all execute on CPU. No GPU numerical
claim follows from these tests. Tracked and new-file whitespace checks are clean.

## Final pre-publication review

Two MAJOR findings were reproduced and remediated before publication:

- Offline analysis accepted matching but incorrect effective flags and did not
  bind condition source, run identity and delivered output to the prepared case.
  The analyzer now rechecks the exact frozen historical input, validates the
  reviewed effective configuration, binds executed/manifest/source/run IDs, and
  verifies actual delivered tokens, the selected decision and masked stage.
  The expected flags are shared with the live gate without changing their values.
- Collection errors could discard retained tensors before the normal RPC result
  was serialized. A failed collection now preserves transferable raw and masked
  vectors plus actual query chunks in `failed-<condition>-vectors.json`, before
  re-raising. Base64 encodes native-byte-order IEEE754 float32 bytes, with original
  captured dtype, shape and byte order recorded. This preserves NaN/Inf values
  too, without treating an invalid capture as valid scientific evidence.

The hook tests exercise indices 0, 1, 548 and 549, actual sampled-token feedback,
reused input/logit buffers, later in-flight work, exceptions and cleanup. New
regression tests cover all provenance corrections and failed-vector preservation.
No teacher forcing or production semantic change was introduced. No unresolved
BLOCKER or MAJOR remained after this review. MINOR/NIT: none requiring remediation.

Final required CPU suite: **160 passed, 4 skipped, 0 failed**. The same four
optional pinned installed-source-cache tests remain SKIPPED for the reasons
listed above. All 26 pinned reference source hashes and tracked/new-file
whitespace checks passed. GPU numerical correctness remains untested.

The publication gate authorizes only a new descendant of `873cf2c...` on
`origin/fix/m4-02-06-greedy-repro`, after confirming that remote target. It does
not authorize rewriting the original commit or starting the controlled GPU run.
