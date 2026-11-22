# Dual-product PULSE campaign

The project-owned driver is `experiments.eagle_pulse_campaign`. It wraps the
active two-retirement collector and `publish_pulse_cost`; production planning,
lineage, identity, safety and admission modules are unchanged.

The frozen scope is NVIDIA A10 and NVIDIA A10G separately, B1/base3/old0,
context [0,256), q0/q2/q3, 50 predetermined slots per cell, one attempt per
slot. A failed slot immediately blocks publication and stops the campaign.
Statistics use whole cycles, 5000 bootstrap resamples, 95%, seed 731 and
`ms/completed_two_retirement_cycle`.

The explicit user resolution of 2026-11-22 governs unsupported products:
record full available observed identity, count one allocation, execute zero
scientific slots, release that worker, and continue within the frozen bound.
Repeated unsupported allocations stop at MAX_ALLOCATIONS_REACHED. Neither
product's scientific state changes on an unsupported allocation.

`eagle_pulse_campaign_freeze` runs locally after the implementation checkpoint.
It creates a self-contained Git bundle, verifies it, reconstructs an empty
checkout, checks HEAD/cleanliness/fsck/no alternates and every execution-source
manifest entry, then freezes the plan and initial state. It never imports Modal.
Existing campaign directories cannot be overwritten by preparation.

The live `run` verb requires `--resume`, a non-synthetic plan, the exact branch
and source SHA, clean executable source, verified source package, all existing
raw/profile hashes and an exclusive owner lock. No verb defaults to live.
`dry-run`, `--dry-run`, and `validate` perform local work only. `validate`
requires both products complete and re-admits their typed profiles.

The live adapter is imported only by explicit live execution. One single-use
container has max_containers=1, retries=0 and a 3600-second timeout. It observes
hardware before scientific execution and uses the source bundle's reviewed
worker. Between cohorts it waits for a driver command. The driver sends another
slot only after the previous raw bytes, manifest and state are durable.

Each cohort resets prefix cache, creates a new request/incarnation, uses the
pinned tokenizer/revision, greedy/no-stop sampling and neutral preparation,
and captures the exact production DecisionState and controller base. The
production lease applies q once, forces K0 continuation, and finalizes the two
exact native retirements. Native queue filling remains enabled. Cleanup and
later neutral work receive no third-retirement credit. The next cohort starts
only after the request and native queue drain. Worker identity stays exact.
Fresh request cohorts do not establish cross-VM nuisance independence; raw
allocation provenance preserves their actual worker grouping and order.

Every attempted slot has immutable raw files, a hashed manifest and an attempt
ID. Complete temporary JSON is flushed/closed before same-directory replace.
Resume checks the entire raw inventory and typed cycle receipts; `.done`
markers have no authority. A crash leaving RUNNING scientific work or a stale
lock fails closed for technical review. It never retries that slot or steals
the lock. A lost worker result is retained as a failed attempt and blocks
publication. No replacement slots are generated.

Live profile paths relative to the campaign root are:

- `publication/A10/eagle-pulse-profile.json`
- `publication/A10G/eagle-pulse-profile.json`

Each has a publication manifest binding the source, plan, raw manifests,
bootstrap constants and timestamp. Synthetic runs retain synthetic raw tags
and use `synthetic-publication/`; they cannot write live profile paths. Their
isolated bootstrap contract exercise never promotes synthetic cycles to live
profile evidence.

Campaign terminal state strings are COMPLETE, BLOCKED and
MAX_ALLOCATIONS_REACHED. Hard validation errors print
`SOURCE/STATE/IDENTITY INVARIANT FAILURE` and return nonzero. A BLOCKED state
contains PUBLICATION_BLOCKED product/cell states. No benchmark or online
POLICY_VALIDATION is part of this driver.

CPU evidence and source handoff metadata belong under
`audit/r6_secp_v11_gpu_readiness_20261122/dual_product_campaign_harness_v1/`.
The final report supplies the exact frozen directory, SHA and user commands
after local gates and source review. R6-GPU and R6-PERFORMANCE remain
UNVERIFIED by implementation and synthetic testing.

Adapter API cross-checks: [Modal App](https://modal.com/docs/sdk/py/latest/App),
[Modal Queue](https://modal.com/docs/sdk/py/latest/Queue), and
[Modal Function](https://modal.com/docs/sdk/py/latest/Function). The pinned
vLLM 0.30.0 source and the applied project P2R contract govern runtime semantics.
