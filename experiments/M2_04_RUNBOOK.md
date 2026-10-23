# M2-04: Dai's canonical T4 execution preparation

This runbook covers execution and noise readiness. Duong owns acceptance-metric
aggregation, `m2_summary.csv`, F1/F2 and report analysis. Readiness PASS here means
54 cells with at least five repeats and no cell above the frozen noise threshold;
it does **not** mean Gate M2 or the empirical W3 speedup gate passed.

## Preconditions after code review

- Use an isolated Linux Tesla T4 runtime with the existing
  `m1-t4-cu130-v1` environment contract and lane launchers. Do not bypass the
  harness environment checks. This switcher stops existing vLLM services.
- Freeze the full **reviewed execution SHA** in a clean checkout. The base
  `82aa55569e381e938a25475c279f1acd13408433` is a reference, not the SHA of the
  reviewed tooling changes. Do not edit code, manifests or workloads during a run.
- Canonical artifacts must be **byte-identical LF files matching their committed
  manifest hashes**. Git-for-Windows CRLF conversion changes the byte hashes.
  Use a fresh Linux checkout configured with `core.autocrlf=false` before checkout.
  Do not change hashes, weaken validation or normalize data at runtime.
- Artifacts: W1 `configs/w1/node1_chat_requests.jsonl`, W2
  `configs/w2_v2/chat_requests.jsonl`, W3 `configs/w3_v2/chat_requests.jsonl`, all
  under `experiments/`. The runner verifies all three manifests before switching.
- Keep launcher/controller/cache settings at the reviewed contract. Record any
  previously approved environment overrides in the execution record; do not
  introduce new tuning. Reserve ports 8001/8002/8003 for this campaign.
- Pick a unique, nonexistent evidence directory, preferably outside the checkout.
  Its basename becomes the shared experiment ID. Preserve/download the complete
  directory if running on an ephemeral host.

## Review the CPU-only plan

From the repository root, substitute the reviewed full SHA and a fresh directory:

```bash
python -m experiments.run_m2_campaign \
  --execution-sha REVIEWED_FULL_SHA \
  --output-dir /path/to/evidence/m2-04-UNIQUE_ATTEMPT
```

Without `--execute` this only prints commands: it does not start servers, inspect
the GPU or create files. It produces 54 unique cells in this order:

1. W1, then W2, then W3.
2. Within each workload: concurrency 1, 2, 4, 8, 16, 32.
3. Within each workload/concurrency: Vanilla, Static, Adaptive.

Every cell uses the same artifact, `cycle-rotate` mapping, 32 requests per repeat,
and five measured repeats (270 repeats, 8640 measured requests total). Seed 42,
temperature 0 and all other generation settings come from the frozen packages.
The existing harness warms each cell once with `max(3, concurrency)` requests
before its measured repeats. Each lane is switched and health-checked first.
The harness's single-cell invocation prints its existing “non-canonical
concurrency set” notice; the wrapper's complete matrix is the 54-cell campaign.

## Manual T4 execution after review

Run the same command with `--execute` on the reserved T4. Do not run it locally
as part of CPU verification. The wrapper refuses an existing evidence directory,
requires the reviewed clean SHA, and never passes `--overwrite` or
`--skip-env-check`. It stops on the first failed switch or measured cell.

Evidence includes `plan.json` (execution SHA and exact commands), per-cell
`raw.jsonl`, `switch.log`, `harness.log`, isolated `server/` logs, Adaptive
telemetry and a final `readiness.json`. `harness_header.csv` is only the existing
harness's empty header file; it is not Duong's analysis deliverable.
Failed/partial attempts remain intact. No checkpoint/resume machinery is added.
Stop the final server manually with `bash scripts/switch_lane.sh stop` when done
or after failure; archive the evidence directory before releasing the runtime.

## Noise check and Gate M2 handoff

The wrapper runs the read-only checker after the full campaign. It can also be
rerun against the explicitly selected 54 raw files:

```bash
python -m experiments.aggregate_results --check-m2 \
  --raw /path/to/evidence/m2-04-UNIQUE_ATTEMPT/*/raw.jsonl
```

It reports incomplete/duplicate cells, fewer than five measured repeats, invalid
measured requests and noisy cell IDs. Exit 0 means readiness PASS; exit 1 means
FAIL. Noise remains exactly:

`(max repeat TPOT p50 - min repeat TPOT p50) / median repeat TPOT p50 > 0.15`.

Exactly 15% is not noisy. No measured rows are dropped or rewritten. A failed
measured request invalidates the check; inspect its retained raw/log evidence.
For a noisy cell, rerun its workload/concurrency triplet in the same lane order,
using the commands in `plan.json` with **new** output/log/telemetry paths and a
new experiment ID. Recheck the resulting explicitly chosen 54-cell evidence set;
do not glob together old and rerun versions of a cell. Record which attempt was
selected and why; preserve every original and rerun file.

Hand the full raw evidence, execution SHA/environment records, plan and readiness
report to Duong for matched provenance/mapping checks, final summary and plots.
The six Adaptive W3 speedups must be measured against matched Vanilla cells;
values below 1.00x remain empirical failures. No W3 success is assumed here.
