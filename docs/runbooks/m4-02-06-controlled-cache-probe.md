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
