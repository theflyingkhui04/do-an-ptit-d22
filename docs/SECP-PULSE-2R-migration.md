# SECP-PULSE-2R-v1 active migration

The frozen two-retirement objective is now applied to the active production planner. This change is scoped to B1/base3/old0/context [0,256), actions 0/2/3, same-engine K0 continuation, and explicit zero terminal value. No objective comparison or redesign was performed.

The active loader accepts only EaglePulseProfile with the pulse objective, exact initial DecisionState keys, product/runtime identity, G, cycle units, horizon 2, continuation 0 and terminal value 0. Legacy EAGLE-RW types/governor remain explicit original-contract regression utilities; they are not selected by AdaptiveBudgetPlanner or its environment loader. Historical fixtures explicitly inject that original model. New pulse tests exercise the default production governor and planner.

The numerator is `2 + sum(S_j)`, K0 baseline is 2, positive candidates use C_high and K0 uses C_low. Strict support and deterministic smaller-q ties are preserved. A pulse applies once; the next action and later native queue-fill work remain K0 until both exact outputs finalize. The lease then releases ordinary planning. Safety dominates. Recovery-clipped pulses clear cost claims and cannot arm a canonical cycle.

Pulse history is separate from legacy history and accepts only source-bound completed normal full-base3 pulse/K0 cycles in the same domain/incarnation. Calibration, probe, terminal, stale and invalid observations cannot seed it. Effective EOS must actually be disabled (the native stop function checks eos_token_id), and both tail budgets must retain at least six tokens after pending output reservation. The domain binds the resolved native queue capacity/driver and preparation/configuration contracts. Missing guards fail closed. Serial and queued domains do not alias.

The project executor observes actual proposal shapes, including width zero, without tensor readback or synchronization. Exact post-stop finalization retains pulse/continuation receipts, actual token counts, origins, physical widths, clock span and failure/censoring evidence. The single-cohort collector forbids incarnation reuse and preserves failed pulse attempts. It has no campaign loop. The pure cycle publication validator/bootstrap uses whole cycle/cohort slots, 50 fixed slots, 5000 resamples, 95%, seed 731; no live collection or profile publication was performed.

The predeclared causal fixture is `tests/fixtures/secp_pulse_causal.json`. All non-cost inputs are fixed: B1/base3/old0, survival [.75,.5,.25], candidate set [0,2,3], identity/runtime/controller/recovery state. Y=[2,3.25,3.5]. Surface A costs [1,1.1,2] yield lower scores [2,2.954545454545454,1.75] and final K2. Surface B costs [1,2,1.1] yield [2,1.625,3.1818181818181817] and final K3. Both K0 upper scores are 2. These are synthetic CPU costs, not measured GPU costs.

Validation: 746 distinct PASS = exact preceding 704 cases + 42 new pulse cases (T01 through T32, with parameterized controls). Failed/errors/skipped/deselected: 0. Ordered focused gates covered new pulse tests, cost/profile/planner, collector/bootstrap, DecisionState/pending, and safety/lineage/controller before the complete regression gate. All 19 pinned vLLM 0.30.0 files verify; git diff --check passes. Earlier reruns are not added to the distinct count.

The three active design documents record the scoped amendment application; historical audit evidence is untouched. Old/new hashes, exact CPU commands, JUnit evidence and causal inputs/scores are retained outside the source commit under `audit/pulse_migration_cpu_20261122/`. The commit contains only exact production, tests, active design and this worklog. Source checkpoint metadata is recorded after commit in that audit directory; no self-referential commit SHA is embedded here.

R6-SOURCE PASS; R6-CPU PASS; R6-GPU UNVERIFIED; R6-PERFORMANCE UNVERIFIED. CPU proofs establish source behavior, not GPU numerical prefix stability, hardware performance or survival confidence. Cost-bound conservatism does not cover survival-model uncertainty. No Modal/GPU, upload, campaign, benchmark, branch change, push, PR or Trello action was performed. No A10/A10G alias or legacy-profile promotion was introduced. No bundle is needed for this CPU-only checkpoint; a later live handoff must package the new executable source.

## Traceability

20 requirements, 20 mapped, 0 unmapped, 0 duplicate IDs. Clause IDs refer to the applied normative text in the active specification. Paths below use module/function names; unqualified module basenames are in src/proposer, src/server or src/flexsds/core as appropriate.

| Clause | Implementation | Tests |
|---|---|---|
| P2R-01 | src/proposer/pulse_policy.py:PulseCostGovernor | T01,T02,T03,T28 |
| P2R-02 | src/server/decision_state.py:PulseContextV1; eagle_consumer_profile.py:pulse_state_payload | T15,T27,T31 |
| P2R-03 | src/server/budget_planner.py:plan_step; pulse_policy.py:PulseCycleLease | T16,T17,T18,T19,T32 |
| P2R-04 | pulse_policy.py:PulseCostGovernor; PulseCycleLease | T01,T05,T16,T19 |
| P2R-05 | pulse_policy.py:PulseCycleLease.__init__/retire | T20,T21,T28,T32 |
| P2R-06 | pulse_policy.py:PulseCycleLease.confirm/verifier/retire | T20,T21,T22,T23,T24 |
| P2R-07 | pulse_policy.py:PulseCycleLease.retire; adaptive_scheduler.py:update_from_output | T20,T21,T28,T32 |
| P2R-08 | eagle_consumer_profile.py:EaglePulseProfile/PulseCostCell/publish_pulse_cost | T12,T13,T14,T28 |
| P2R-09 | pulse_policy.py:PulseCycleLease.retire | T23,T28 |
| P2R-10 | src/proposer/cost_policy.py:SurvivalEstimate.predicted_cycle_reward | T01,T02,T03 |
| P2R-11 | pulse_policy.py:PulseSurvivalHistory; budget_planner.py:_emit_finalized_cost_feedback | T26,T32 |
| P2R-12 | decision_state.py:capture_pulse_context/PulseContextV1.validate | T24,T31 |
| P2R-13 | pulse_policy.py:PulseCycleLease.invalidate/retire | T24,T25 |
| P2R-14 | budget_planner.py:plan_step; lineage.py:clear; lineage_executor.py:execute_model/sample_tokens | T25,T29,T30 |
| P2R-15 | experiments/eagle_rw_collector.py:EaglePulseCycleCollector; budget_planner.py:finish_request | T26,T28 |
| P2R-16 | eagle_consumer_profile.py:publish_pulse_cost/PulseCostCell | T25,T28 |
| P2R-17 | eagle_consumer_profile.py:EaglePulseProfile/pulse_state_payload | T12,T13,T14,T15,T27,T31 |
| P2R-18 | pulse_policy.py:PulseCostGovernor.decide | T04,T05,T06,T07,T08,T09,T10,T11 |
| P2R-19 | pulse_policy.py:PulseCycleLease; adaptive_scheduler.py:update_from_output | T16,T17,T18,T19,T20,T21,T32 |
| P2R-20 | budget_planner.py:__init__/from_env; tests/scalar_eagle_fixtures.py | T09,T10,T11,T12,T13,T14,T29,T30,T31,T32 |

## Next action

Implement and CPU-test the resumable dual-product SECP-PULSE-2R full profile campaign harness. Do not run GPU.
