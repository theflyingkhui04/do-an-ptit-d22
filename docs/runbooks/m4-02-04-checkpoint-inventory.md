# M4-02-04 checkpoint inventory

Entry HEAD: `b4c13637149b14b3ebab20412b044639b7e4c40f`; branch `feat/m4-02-04-scheduler-telemetry-v02`.

Classification completed before any staging. All 31 dirty/untracked entry files are listed below. No category I remains. Production source status is UNCHANGED.

| Path | Category | Purpose | Source/runtime dependency | Commit? | Reason |
|---|---|---|---|---|---|
| `logs/daijestic/M4-02/M4-02-04_worklogs.md` | F. WORKLOG | Integration review and corrected-context proof history | Recorded L1/L2 results and prior L3 artifacts | A/B (split staged hunk) | Keep integration and measurement narratives with their commits |
| `tests/test_adaptive_scheduler_glue.py` | A. OPTION-B INTEGRATION TEST | Heterogeneous lifecycle, natural K transition and memory-boundary integration | Pinned scheduler AST fixture; real project scalar classes | A | L2 coverage only; no serving edits |
| `docs/runbooks/m4-02-04-colab-observer-smoke.md` | E. RUNBOOK / TECHNICAL DESIGN | Measurement contract, source proof, observer audit or Colab handoff | Measurement helpers; canonical pinned runtime | C | Scope/limitations documented; Colab doc will replace ZIP primary workflow |
| `docs/runbooks/m4-02-04-corrected-context.md` | E. RUNBOOK / TECHNICAL DESIGN | Measurement contract, source proof, observer audit or Colab handoff | Measurement helpers; canonical pinned runtime | B | Scope/limitations documented; Colab doc will replace ZIP primary workflow |
| `docs/runbooks/m4-02-04-scalar-cost-contract.md` | E. RUNBOOK / TECHNICAL DESIGN | Measurement contract, source proof, observer audit or Colab handoff | Measurement helpers; canonical pinned runtime | B | Scope/limitations documented; Colab doc will replace ZIP primary workflow |
| `docs/runbooks/m4-02-04-scalar-observer-audit.md` | E. RUNBOOK / TECHNICAL DESIGN | Measurement contract, source proof, observer audit or Colab handoff | Measurement helpers; canonical pinned runtime | B | Scope/limitations documented; Colab doc will replace ZIP primary workflow |
| `experiments/runs/m4_02_04/20261111-scalar-observer-l2/cpu_overhead.json` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical synthetic CPU observer overhead, including earlier attempts | Self-recorded source hashes; predates corrected-context extension | B | Small historical evidence retained; explicitly not GPU/performance proof |
| `experiments/runs/m4_02_04/20261111-scalar-observer-l2/cpu_overhead_v2.json` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical synthetic CPU observer overhead, including earlier attempts | Self-recorded source hashes; predates corrected-context extension | B | Small historical evidence retained; explicitly not GPU/performance proof |
| `experiments/runs/m4_02_04/20261111-scalar-observer-l2/cpu_overhead_v3.json` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical synthetic CPU observer overhead, including earlier attempts | Self-recorded source hashes; predates corrected-context extension | B | Small historical evidence retained; explicitly not GPU/performance proof |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/SHA256SUMS.json` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/delivery-hashes.json` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/extracted-pre-gpu-tests.log` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical ZIP preparation test/operational record | Superseded local ZIP | NO | Keep local; not needed for reproducible Git handoff |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/local-preparation.json` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical ZIP preparation test/operational record | Superseded local ZIP | NO | Keep local; not needed for reproducible Git handoff |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/scalar-observer-colab-source.zip` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T115100Z-colab-observer-package/scalar_observer_l4_smoke.ipynb` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T120000Z-colab-observer-package/SHA256SUMS.json` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T120000Z-colab-observer-package/extracted-pre-gpu-tests.log` | D. RAW/DERIVED MEASUREMENT EVIDENCE | Historical ZIP preparation test/operational record | Superseded local ZIP | NO | Keep local; not needed for reproducible Git handoff |
| `experiments/runs/m4_02_04/20261111T120000Z-colab-observer-package/scalar-observer-colab-source.zip` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/runs/m4_02_04/20261111T120000Z-colab-observer-package/scalar_observer_l4_smoke.ipynb` | G. GENERATED / CACHE | Previous ZIP handoff output | Superseded local package | NO | Contains/references local captures or obsolete upload workflow |
| `experiments/scalar_colab_package.py` | B. MEASUREMENT IMPLEMENTATION | Superseded local ZIP builder | Local captures and immutable Git blobs | NO | Retain locally; Git template replaces primary ZIP transport |
| `experiments/scalar_colab_smoke.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | C | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_context.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | B | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_cost_capture.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | B | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_cost_evidence.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | B | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_cost_observer_overhead.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | B | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_live_observer.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | C | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `experiments/scalar_smoke_evidence.py` | B. MEASUREMENT IMPLEMENTATION | Pure recurrence, retirement capture, validator, CPU overhead or isolated live smoke | CPU metadata / exact vLLM 0.30.0 V1 UniProc API | C | Measurement-only modules outside production; Git gates replace ZIP assumptions |
| `tests/test_scalar_colab_smoke.py` | C. MEASUREMENT TEST | CPU recurrence/observer/evidence/transport regression | Helpers and hash-verified pinned vLLM source | C | Reproducibility guards; ZIP packaging test will become Git workflow preflight |
| `tests/test_scalar_context.py` | C. MEASUREMENT TEST | CPU recurrence/observer/evidence/transport regression | Helpers and hash-verified pinned vLLM source | B | Reproducibility guards; ZIP packaging test will become Git workflow preflight |
| `tests/test_scalar_cost_capture.py` | C. MEASUREMENT TEST | CPU recurrence/observer/evidence/transport regression | Helpers and hash-verified pinned vLLM source | B | Reproducibility guards; ZIP packaging test will become Git workflow preflight |
| `tests/test_scalar_cost_evidence.py` | C. MEASUREMENT TEST | CPU recurrence/observer/evidence/transport regression | Helpers and hash-verified pinned vLLM source | B | Reproducibility guards; ZIP packaging test will become Git workflow preflight |

## Explicit local exclusions

- H. TEMPORARY RESEARCH CAPTURE: existing ignored `experiments/runs/m4_02_04/20261110-async-lineage-design/installed/**` and its original manifest; no full upstream source will be committed. A curated hash-only manifest will bind installed vLLM bytes on Colab.
- G. GENERATED / CACHE: `__pycache__/**`, `.pyc`, `.pytest_cache/**`, previous ZIP bundles/notebooks and removed temporary extraction directories.
- Local Codex visualizations, credentials and Colab runtime output are excluded.
- No `git add .` or `git add -A`; every staged path is explicit.

## New checkpoint files

Hash-only pinned-source loader/manifest belong to B. Git workflow helper, dependency pins, notebook template and preflight tests belong to C. This inventory belongs to C. Generated SHA-filled handoff notebook will remain local after the push, per the user-approved template/handoff split.

## Reference

Reviewed `D:/Document/Four-year/do_an/colab/m4_02_02_l4_v4_git.ipynb` as a structural reference: separate checkouts, detached run SHA, Colab Secrets/GIT_ASKPASS, source gate and guarded explicit evidence staging. Its full-grid cells are not copied. The user confirmed it is not a runtime dependency.

## Explicit new-file classification

| Path | Category | Purpose | Dependency | Commit? | Reason |
|---|---|---|---|---|---|
| `experiments/scalar_pinned_sources.py` | B. MEASUREMENT IMPLEMENTATION | Verified source lookup | Committed hashes / installed vLLM metadata | B | Avoid vendoring captures |
| `experiments/scalar_vllm_source_manifest.json` | B. MEASUREMENT IMPLEMENTATION | 15-file hash authority | Preserved installed manifest | B | Metadata only |
| `experiments/scalar_colab_git.py` | B. MEASUREMENT IMPLEMENTATION | Detached source and evidence gates | Git / stdlib | C | Two-checkout workflow |
| `experiments/requirements-scalar-smoke.txt` | B. MEASUREMENT IMPLEMENTATION | Exact smoke requirements | Canonical dependency freeze | C | Reproducible isolated install |
| `experiments/scalar_smoke_constraints.txt` | B. MEASUREMENT IMPLEMENTATION | Transitive dependency constraints | Prior L4 freeze, editable checkout excluded | C | No silent version replacement |
| `experiments/colab_m4_02_04_scalar_observer_git.ipynb` | B. MEASUREMENT IMPLEMENTATION | Reviewed Git notebook template | Colab Secrets / L4 / pinned stack | C | No outputs or credentials |
| `tests/test_scalar_colab_git.py` | C. MEASUREMENT TEST | Local Git divergence/auth/source guards | Temporary local bare repositories | C | No external execution |
| `docs/runbooks/m4-02-04-checkpoint-inventory.md` | E. RUNBOOK / TECHNICAL DESIGN | Complete classification and staging rationale | Entry status and review | C | Auditable scope |
| `experiments/runs/m4_02_04/20261111-checkpoint-preparation/integration-worklog.patch` | G. GENERATED / CACHE | Temporary split-staging patch | Local worklog/index | NO | Keep local, not runtime evidence |
| SHA-filled handoff notebook and push receipt under the checkpoint-preparation directory | G. GENERATED / CACHE | Post-push delivery copy | Exact pushed CODE_SHA | NO | User-approved avoidance of self-referential SHA |

The user explicitly approved committing the template first, then rendering a local handoff/runtime copy after push, without another commit to embed its own SHA.
