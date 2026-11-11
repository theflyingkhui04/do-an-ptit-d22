"""Reproduce CPU-only observer overhead; never imports or launches vLLM/CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from types import SimpleNamespace as NS

from experiments.scalar_cost_capture import RetirementObserver


def measure(iterations=1000, repeats=7):
    if iterations < 1 or repeats < 1:
        raise ValueError("Positive iteration/repeat counts required")
    runtime = dict(vllm="0.30.0", executor="uni", runner="V1", tp=1, pp=1,
                   scheduler_base="AdaptiveAsyncScheduler", async_scheduling=True)
    measurements = []
    for batch in (1, 2):
        ids = [f"r{i}" for i in range(batch)]
        schedule = NS(num_spec_tokens_to_schedule=2,
                      num_scheduled_tokens={rid: 3 for rid in ids},
                      scheduled_spec_decode_tokens={rid: [-1, -1] for rid in ids})
        output = NS(req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)})
        result = {0: NS(outputs=[NS(request_id=rid, new_token_ids=[7], finish_reason=None)
                                 for rid in ids])}
        snapshot = dict(request_ids=ids, contexts=[40] * batch, queries=[3] * batch,
                        valid_drafts=[2] * batch, context_kind="MEASURED", prefill=False,
                        proposer_rows=batch, proposer_k=2)
        baseline, enabled = [], []
        def delegate(s, o): return result
        for repeat in range(repeats):
            observer = RetirementObserver(runtime, capacity=iterations + 1)
            def disabled():
                for _ in range(iterations): delegate(schedule, output)
            def observed():
                for _ in range(iterations):
                    observer.submit(schedule, placeholder_counts={rid: 2 for rid in ids},
                                    incarnations={rid: rid + ":1" for rid in ids})
                    observer.worker_snapshot(schedule, snapshot)
                    assert observer.update(delegate, schedule, output, unsafe_requests=()) is result
            order = [(baseline, disabled), (enabled, observed)]
            if repeat % 2:
                order.reverse()
            for samples, operation in order:
                start = time.perf_counter_ns()
                operation()
                samples.append(time.perf_counter_ns() - start)
            assert len(observer.events) == iterations and not observer.overflow
        measurements.append(dict(batch=batch, iterations=iterations, disabled_ns=baseline,
                                 enabled_ns=enabled, median_added_us_per_call=statistics.median(
                                     (e - b) / iterations / 1000 for b, e in zip(baseline, enabled))))
    root = Path(__file__).resolve().parents[1]
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]
    return dict(evidence_layer="CPU_SYNTHETIC_OVERHEAD_ONLY", not_gpu_evidence=True,
                not_profitability_evidence=True,
                git_sha=subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                source_dirty=bool(subprocess.check_output([*git, "status", "--porcelain"], cwd=root, text=True)),
                observer_sha256=hashlib.sha256((root / "experiments/scalar_cost_capture.py").read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=platform.python_version(), platform=platform.platform(), clock="perf_counter_ns",
                repeats=repeats, measurements=measurements,
                limitations="Synthetic CPU snapshots only; no installed vLLM or GPU execution. "
                "No GPU perturbation bound. Export/serialization/storage excluded from timed regions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    # Reserve before any work; preserve prior runs and partial failures.
    with args.output.open("x", encoding="utf-8") as stream:
        report = measure(args.iterations, args.repeats)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("batch", "median_added_us_per_call")}
                      for row in report["measurements"]]))


if __name__ == "__main__":
    main()
