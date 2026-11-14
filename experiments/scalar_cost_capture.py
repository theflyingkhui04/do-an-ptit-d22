"""Measurement-only, CPU in-memory retirement observer; no GPU installer.

The caller supplies execution-bound CPU snapshots. Optimistic runner contexts
stay forecasts; validated pure reconstruction is labelled separately. There is
no live GPU installer, runtime patch, future.result(), tensor readback, file I/O,
CUDA call or controller change here.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from experiments.scalar_context import ExecutionContextSnapshot, ReconstructedExecution, VERSION


class UnsupportedMeasurementMode(ValueError):
    pass


def check_capabilities(runtime):
    required = {"vllm": "0.30.0", "executor": "uni", "runner": "V1",
                "tp": 1, "pp": 1, "scheduler_base": "AdaptiveAsyncScheduler"}
    if any(runtime.get(key) != value for key, value in required.items()):
        raise UnsupportedMeasurementMode("UNSUPPORTED_MEASUREMENT_MODE")
    if type(runtime.get("async_scheduling")) is not bool:
        raise UnsupportedMeasurementMode("UNSUPPORTED_MEASUREMENT_MODE: async mode unknown")


class RetirementObserver:
    def __init__(self, runtime, *, clock=time.perf_counter_ns, intervals=8, capacity=4096):
        check_capabilities(runtime)
        if intervals < 1 or capacity < 2:
            raise ValueError("Invalid observer bounds")
        self.clock, self.intervals, self.capacity = clock, intervals, capacity
        self.pending = {}
        self.events, self.windows = [], []
        self.anchor, self.segment = None, []
        self.window_signature = None
        self.sequence = 0
        self.overflow = False

    def _append(self, record):
        if len(self.events) >= self.capacity:
            self.overflow = True
            self.anchor, self.segment = None, []
            return
        self.events.append(record)
        if record["status"] == "eligible":
            signature = (record["scheduled_k"], tuple(sorted(record["incarnations"].items())),
                         tuple(sorted(record["placeholder_counts"].items())))
            if self.anchor is not None and signature != self.window_signature:
                record.update(status="unsupported", reason="WINDOW_STATE_CHANGE")
            self.window_signature = signature
        if record["status"] != "eligible":
            self.anchor, self.segment = None, []
            return
        if self.anchor is None:
            self.anchor = record["end_ns"]
            return  # anchor output is excluded
        self.segment.append(record)
        if len(self.segment) == self.intervals:
            elapsed = record["end_ns"] - self.anchor
            if elapsed <= 0:
                self.events.append({"status": "invalid", "reason": "CLOCK_ORDER"})
            else:
                self.windows.append({"start_ns": self.anchor, "end_ns": record["end_ns"],
                                     "retirement_ids": [r["verifier_id"] for r in self.segment],
                                     "ms_per_batch": elapsed / 1e6 / self.intervals,
                                     "emitted": sum(r["emitted"] for r in self.segment)})
            self.anchor, self.segment = record["end_ns"], []

    def submit(self, schedule, *, planned_k=None, planned_per_request=None,
               placeholder_counts=None, incarnations=None):
        """Call at actual execute_model submission, not when a plan is produced."""
        if self.overflow:
            return
        if id(schedule) in self.pending:
            raise ValueError("SchedulerOutput submitted twice")
        self.sequence += 1
        self.pending[id(schedule)] = {
            "schedule": schedule,  # strong reference prevents object-id reuse
            "verifier_id": f"v{self.sequence}", "planned_k": planned_k,
            "planned_per_request": dict(planned_per_request or {}),
            "placeholder_counts": dict(placeholder_counts or {}),
            "incarnations": dict(incarnations or {}),
            "scheduled_k": schedule.num_spec_tokens_to_schedule,
            "scheduled": dict(schedule.num_scheduled_tokens),
            "verification_slots": {rid: len(tokens) for rid, tokens in
                                   schedule.scheduled_spec_decode_tokens.items()},
            "worker": None,
        }

    def worker_snapshot(self, schedule, snapshot, *, context_snapshot=None):
        """Only detached CPU containers are accepted, not tensors or live buffers."""
        if self.overflow:
            return
        def cpu_copy(value):
            if value is None or type(value) in (str, int, float, bool):
                return value
            if type(value) in (list, tuple):
                return tuple(cpu_copy(v) for v in value)
            if type(value) is dict and all(type(k) is str for k in value):
                return {k: cpu_copy(v) for k, v in value.items()}
            raise UnsupportedMeasurementMode("UNSUPPORTED_MEASUREMENT_MODE: non-CPU snapshot")
        entry = self.pending.get(id(schedule))
        if entry is None or entry["schedule"] is not schedule:
            raise ValueError("Worker snapshot without matching submission")
        if entry["worker"] is not None:
            raise ValueError("Duplicate worker snapshot")
        entry["worker"] = cpu_copy(snapshot)
        if context_snapshot is not None:
            if (not isinstance(context_snapshot, ExecutionContextSnapshot)
                    or context_snapshot.execution_id != entry["verifier_id"]
                    or context_snapshot.scheduler_output_id != str(id(schedule))):
                raise UnsupportedMeasurementMode("UNSUPPORTED_MEASUREMENT_MODE: context snapshot binding")
            entry["context_preparation_snapshot"] = context_snapshot

    def update(self, delegate, schedule, output, *, unsafe_requests=None, reconstructed_context=None):
        """Delegate exactly once; retirement timestamp precedes record processing.

        unsafe_requests comes from scheduler state before it consumes output:
        finished/aborted, stale/drop, preempted or failed-KV rows. Unknown state
        must be reported unsafe by the future installer, never assumed safe.
        """
        # Canonical future has already materialized these CPU lists. Scheduler
        # stop handling can mutate their lengths; retain counts BEFORE delegation.
        try:
            sampled = output.sampled_token_ids
            raw_counts = (tuple((rid, len(sampled[output.req_id_to_index[rid]])) for rid in output.req_ids)
                          if type(sampled) is list and all(type(row) is list for row in sampled) else None)
        except (AttributeError, KeyError, IndexError, TypeError):
            raw_counts = None
        try:
            result = delegate(schedule, output)
        except Exception as exc:
            self.failure(type(exc).__name__, schedule)
            raise
        end = self.clock()
        try:
            self._retired(schedule, output, result, end, unsafe_requests, raw_counts, reconstructed_context)
        except Exception as exc:
            self.pending.pop(id(schedule), None)
            self._append({"status": "unsupported", "reason": f"OBSERVATION_ERROR:{type(exc).__name__}",
                          "end_ns": end})
        return result

    def _retired(self, schedule, output, result, end, unsafe_requests, raw_counts, reconstructed_context):
        entry = self.pending.pop(id(schedule), None)
        record = {"status": "unsupported", "reason": None, "end_ns": end,
                  "emitted": 0, "output_rows": tuple(output.req_ids),
                  "schedule_object_id": id(schedule), "model_output_object_id": id(output),
                  "raw_sampled_counts_before_update": raw_counts,
                  "tree_execution_verified": False}
        if entry is None or entry["schedule"] is not schedule:
            record["reason"] = "MISSING_SUBMISSION_PAIR"
        else:
            record.update({k: v for k, v in entry.items() if k not in ("schedule", "context_preparation_snapshot")})
            rows = output.req_ids
            mapping = output.req_id_to_index
            valid_mapping = (len(set(rows)) == len(rows)
                             and mapping == {rid: i for i, rid in enumerate(rows)})
            emitted = {}
            terminal = False
            terminal_ids = set()
            for bundle in result.values():
                for item in bundle.outputs:
                    emitted[item.request_id] = len(item.new_token_ids)
                    terminal |= item.finish_reason is not None
                    if item.finish_reason is not None:
                        terminal_ids.add(item.request_id)
            record["emitted_by_request"] = emitted
            record["emitted"] = sum(emitted.values())
            worker = entry["worker"]
            if reconstructed_context is not None:
                ctx = reconstructed_context
                if (not isinstance(ctx, ReconstructedExecution)
                        or ctx.snapshot != entry.get("context_preparation_snapshot")
                        or ctx.snapshot.execution_id != entry["verifier_id"]
                        or ctx.snapshot.scheduler_output_id != str(id(schedule))
                        or dict(zip((r.request_id for r in ctx.snapshot.rows), ctx.valid_counts)) != dict(raw_counts or ())
                        or {r.request_id: r.incarnation for r in ctx.snapshot.rows} != entry["incarnations"]
                        or ctx.snapshot.shared_k != entry["scheduled_k"]
                        or worker is None):
                    record["reason"] = "RECONSTRUCTION_IDENTITY_MISMATCH"
                    self._append(record)
                    return
                contexts = {r.request_id: c for r, c in zip(ctx.snapshot.rows, ctx.pre_contexts)}
                queries = {r.request_id: r.query_length for r in ctx.snapshot.rows}
                host_contexts = {r.request_id: r.cpu_context for r in ctx.snapshot.rows}
                if (set(contexts) != set(worker["request_ids"])
                        or any(queries[rid] != q for rid, q in zip(worker["request_ids"], worker["queries"]))
                        or any(host_contexts[rid] != h for rid, h in zip(worker["request_ids"], worker["contexts"]))):
                    record["reason"] = "RECONSTRUCTION_SHAPE_MISMATCH"
                    self._append(record)
                    return
                worker = dict(worker, cpu_contexts_forecast=worker["contexts"],
                              contexts=tuple(contexts[rid] for rid in worker["request_ids"]),
                              context_kind="RECONSTRUCTED", context_reconstruction_version=VERSION)
                record["worker"] = worker
                record["context_reconstruction"] = ctx.record_fields()
                record["preparation_context_snapshot"] = asdict(ctx.snapshot)
                committed_before = {r.request_id: r.committed_output_before for r in ctx.snapshot.rows}
                for row in record["context_reconstruction"]["rows"]:
                    row["useful_emitted_count"] = emitted.get(row["request_id"], 0)
                    before = committed_before[row["request_id"]]
                    row["committed_output_after"] = (None if before is None else before + row["useful_emitted_count"])
                    row["next_execution_eligible"] &= row["request_id"] not in terminal_ids
            expected = set(entry["scheduled"])
            if not expected:
                record["reason"] = "EMPTY_SCHEDULE"
            elif not valid_mapping or not expected.issubset(set(rows)):
                record["reason"] = "OUTPUT_IDENTITY_MISMATCH"
            elif unsafe_requests is None:
                record["reason"] = "LIFECYCLE_STATE_UNAVAILABLE"
            elif unsafe_requests or terminal:
                record["reason"] = "LIFECYCLE_UNSAFE"
            elif set(emitted) != expected or not all(emitted.values()):
                record["reason"] = "NO_USEFUL_OUTPUT"
            elif worker is None:
                record["reason"] = "MISSING_WORKER_SHAPE"
            elif worker.get("context_kind") != "MEASURED" and not (
                    reconstructed_context is not None and worker.get("context_kind") == "RECONSTRUCTED"):
                record["reason"] = "ACTUAL_CONTEXT_UNAVAILABLE"
            elif worker.get("prefill") is not False:
                record["reason"] = "PREFILL_OR_UNKNOWN"
            elif set(worker.get("request_ids", ())) != expected:
                record["reason"] = "WORKER_IDENTITY_MISMATCH"
            elif not self._valid_shape(entry, worker):
                record["reason"] = "INCOMPLETE_OR_INCONSISTENT_WORKER_SHAPE"
            elif set(entry["incarnations"]) != expected:
                record["reason"] = "MISSING_REQUEST_INCARNATIONS"
            elif set(entry["placeholder_counts"]) != expected:
                record["reason"] = "MISSING_EFFECTIVE_BUDGETS"
            else:
                record["status"] = "eligible"
        self._append(record)

    @staticmethod
    def _valid_shape(entry, worker):
        ids = worker.get("request_ids", ())
        contexts, queries = worker.get("contexts", ()), worker.get("queries", ())
        drafts = worker.get("valid_drafts", ())
        return (len(ids) == len(set(ids)) == len(contexts) == len(queries) == len(drafts)
                and all(type(v) is int and v >= 0 for v in (*contexts, *drafts))
                and all(type(q) is int and q == d + 1 for q, d in zip(queries, drafts))
                and all(entry["scheduled"][rid] == q for rid, q in zip(ids, queries))
                and all(entry["verification_slots"].get(rid, 0) == d for rid, d in zip(ids, drafts))
                and worker.get("proposer_rows") == len(ids)
                and type(worker.get("proposer_k")) is int
                and worker["proposer_k"] == entry["scheduled_k"])

    def failure(self, reason, schedule=None):
        entry = self.pending.pop(id(schedule), None) if schedule is not None else None
        self._append({"status": "failed", "reason": reason,
                      "verifier_id": entry["verifier_id"] if entry else None,
                      "affected_inflight": tuple(e["verifier_id"] for e in self.pending.values())})
        if schedule is None:
            self.pending.clear()  # failed outer execution invalidates this observation segment

    def step(self, delegate):
        """Outer failure guard: a failed future never reaches update()."""
        try:
            return delegate()
        except Exception as exc:
            self.failure(type(exc).__name__)
            raise

    def export(self):
        """Call after the window. Serialization/storage belongs outside capture."""
        return {"evidence_layer": "L2_ADAPTER_ONLY", "consumer_profile": False,
                "overflow": self.overflow, "events": list(self.events),
                "windows": list(self.windows)}


def trajectory_signature(events):
    """Exact provenance signature, NOT an across-cohort matching predicate.

    Reference comparison also needs an explicit fixture-slot mapping between
    incarnations. Emitted counts remain outcomes, not an equal-yield condition.
    """
    return tuple(tuple(sorted((rid, event["incarnations"][rid], context,
                               event["emitted_by_request"].get(rid, 0))
                              for rid, context in zip(event["worker"]["request_ids"],
                                                      event["worker"]["contexts"])))
                 for event in events)
