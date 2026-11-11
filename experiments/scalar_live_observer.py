"""Read-only instance hooks for the pinned V1/UniProc scalar smoke.

No device reads, output materialization, I/O, or runtime source patching. Every
hook delegates exactly once. Observation errors are retained, not substituted
for engine results. Install only after the runner's source/environment gate.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
import time

from experiments.scalar_context import (
    ContextRow, ExecutionContextSnapshot, freeze_materialized_counts, reconstruct,
)
from experiments.scalar_cost_capture import RetirementObserver


class LiveObserver:
    def __init__(self, core, vocab_size, *, capacity=128):
        self.core, self.scheduler = core, core.scheduler
        self.executor = core.model_executor
        self.runner = self.executor.driver_worker.worker.model_runner
        self.vocab_size = vocab_size
        self.observer = RetirementObserver(
            dict(vllm="0.30.0", executor="uni", runner="V1", tp=1, pp=1,
                 scheduler_base="AdaptiveAsyncScheduler", async_scheduling=True),
            intervals=3, capacity=capacity)
        self.capacity = capacity
        self.facts, self.failures, self.overhead = {}, [], []
        self.undo, self.future_refs = [], []
        self.active, self.previous_preparation, self.previous = None, None, None
        self.proposing = None
        self.collecting = True
        self.eligible_retirements = 0

    def fail(self, stage, error):
        if len(self.failures) < self.capacity:
            self.failures.append({"stage": stage, "error": str(error),
                                  "type": type(error).__name__})

    def observe(self, stage, callback):
        start = time.perf_counter_ns()
        try:
            return callback()
        except Exception as error:
            self.fail(stage, error)
        finally:
            if len(self.overhead) < self.capacity * 12:
                self.overhead.append((stage, time.perf_counter_ns() - start))

    def patch(self, obj, name, wrapper):
        # Restore instance state exactly, including absence of an override.
        existed, prior = name in vars(obj), vars(obj).get(name)
        original = getattr(obj, name)
        setattr(obj, name, wrapper(original))
        self.undo.append((obj, name, existed, prior))

    def install(self):
        self.patch(self.executor, "execute_model", self.execution_hook)
        self.patch(self.executor, "sample_tokens", self.sampling_hook)
        self.patch(self.runner, "_prepare_inputs", self.preparation_hook)
        self.patch(self.runner, "_determine_batch_execution_and_padding", self.padding_hook)
        self.patch(self.runner.drafter, "propose", self.proposer_hook)
        self.patch(type(self.runner.drafter.model), "__call__", self.draft_dispatch_hook)
        self.patch(type(self.runner.model), "__call__", self.target_dispatch_hook)
        self.patch(self.scheduler, "update_from_output", self.retirement_hook)
        self.patch(self.observer, "_retired", self.record_processing_hook)
        return self

    def close(self):
        for obj, name, existed, prior in reversed(self.undo):
            if existed:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)
        self.undo.clear()
        self.future_refs.clear()

    def submission(self, schedule):
        ids = tuple(schedule.num_scheduled_tokens)
        if len(ids) > 1:
            raise ValueError("actual scheduled composition is not B=1")
        if len(self.facts) >= self.capacity:
            raise ValueError("bounded collector capacity exceeded")
        scheduler = self.scheduler
        requests = {rid: scheduler.requests[rid] for rid in ids}
        lifecycle = []
        for rid, request in requests.items():
            if (request.num_preemptions or request.num_stale_output_tokens
                    or request.is_finished()):
                lifecycle.append(rid)
        cached = schedule.scheduled_cached_reqs
        if cached.resumed_req_ids or schedule.preempted_req_ids:
            lifecycle.extend(ids)
        if lifecycle:
            raise ValueError("preemption/resume/stale/abort before execution")
        incarnations = {rid: rid + ":incarnation-1" for rid in ids}
        self.observer.submit(schedule, planned_k=scheduler._plan.batch_k,
            planned_per_request=scheduler._plan.per_request_k,
            placeholder_counts={rid: len(r.spec_token_ids) for rid, r in requests.items()},
            incarnations=incarnations)
        self.facts[id(schedule)] = {
            "verifier_id": self.observer.pending[id(schedule)]["verifier_id"],
            "schedule_object_id": id(schedule), "incarnations": incarnations,
            "prompt_lengths": {rid: r.num_prompt_tokens for rid, r in requests.items()},
            "committed_output_before": {rid: r.num_output_tokens for rid, r in requests.items()},
            "proposer_calls": 0, "draft_forward_calls": 0, "target_forward_calls": 0,
            "execution_future_id": None, "output_future_id": None,
            "materialized_output_id": None, "lifecycle_flags": [],
            "label": "EMPTY_SCHEDULE" if not ids else "PREPARATION_PENDING",
        }

    def bind_future(self, schedule, future, field):
        fact = self.facts[id(schedule)]
        fact[field] = id(future)
        self.future_refs.append(future)
        # Canonical EngineCore decides WHEN to call result. This wrapper only
        # observes its return; it never calls result on its own.
        def wrapper(delegate):
            def materialize(*args, **kwargs):
                result = delegate(*args, **kwargs)
                if result is not None:
                    self.observe("canonical_materialization", lambda: fact.update(
                        materialized_output_id=id(result)))
                return result
            return materialize
        self.patch(future, "result", wrapper)

    def execution_hook(self, delegate):
        def execute(schedule, *args, **kwargs):
            if not self.collecting:
                return delegate(schedule, *args, **kwargs)
            self.active = schedule
            self.observe("submission", lambda: self.submission(schedule))
            try:
                result = delegate(schedule, *args, **kwargs)
                self.observe("execution_future", lambda: self.bind_future(
                    schedule, result, "execution_future_id"))
                return result
            except Exception as error:
                self.fail("execute_model", error)
                raise
            finally:
                self.active = None
        return execute

    def sampling_hook(self, delegate):
        def sample(*args, **kwargs):
            if not self.collecting:
                return delegate(*args, **kwargs)
            state = self.runner.execute_model_state
            schedule = None if state is None else state.scheduler_output
            self.active = schedule
            try:
                result = delegate(*args, **kwargs)
                if schedule is not None:
                    self.observe("output_future", lambda: self.bind_future(
                        schedule, result, "output_future_id"))
                return result
            except Exception as error:
                self.fail("sample_tokens", error)
                raise
            finally:
                self.active = None
        return sample

    def preparation_hook(self, delegate):
        def prepare(schedule, scheduled_queries):
            if not self.collecting:
                return delegate(schedule, scheduled_queries)
            runner = self.runner
            # This predicate is evaluated before the canonical preparation.
            active = bool(runner.use_async_spec_decode
                          and runner.valid_sampled_token_count_gpu is not None
                          and runner.input_batch.prev_req_id_to_index)
            result = delegate(schedule, scheduled_queries)
            self.observe("preparation", lambda: self.prepared(schedule, scheduled_queries, active))
            return result
        return prepare

    def prepared(self, schedule, scheduled_queries, correction_active):
        runner = self.runner
        batch = runner.input_batch
        count = int(batch.num_reqs)
        ids = tuple(batch.req_ids[:count])
        if count != 1 or set(ids) != set(schedule.num_scheduled_tokens):
            raise ValueError("actual worker B/row mapping is not exactly one")
        fact = self.facts[id(schedule)]
        queries = tuple(int(q) for q in scheduled_queries)
        offsets = tuple(int(q) for q in runner.query_start_loc.np[:count + 1])
        if tuple(b - a for a, b in zip(offsets, offsets[1:])) != queries:
            raise ValueError("prepared query offsets disagree with scheduled queries")
        host = tuple(int(h) for h in batch.num_computed_tokens_cpu[:count])
        previous_rows = tuple(int(p) for p in runner.prev_positions.np[:count])
        previous_drafts = tuple(int(runner.prev_num_draft_tokens.np[p]) if p >= 0 else 0
                                for p in previous_rows)
        drafts = tuple(len(schedule.scheduled_spec_decode_tokens.get(rid, ())) for rid in ids)
        invalid = tuple(int((schedule.num_invalid_spec_tokens or {}).get(rid, 0)) for rid in ids)
        # Shape alone never certifies context. Prefill is excluded using the
        # actual preparation's prompt boundary; reconstructed C is checked later.
        prefill = any(h < fact["prompt_lengths"][rid] for rid, h in zip(ids, host))
        label = "PREFILL" if prefill else "DECODE_ELIGIBLE"
        if prefill and not all(h < fact["prompt_lengths"][rid] for rid, h in zip(ids, host)):
            label = "UNSUPPORTED_MIXED"
        rows = tuple(ContextRow(rid, fact["incarnations"][rid], "fixture-slot-0",
            h, q, p, d, slots, bad,
            self.observer.pending[id(schedule)]["planned_per_request"].get(rid, 0),
            fact["committed_output_before"][rid])
            for rid, h, q, p, d, slots, bad in zip(ids, host, queries, previous_rows,
                                                   previous_drafts, drafts, invalid))
        fact.update(label=label, worker_rows=count, request_ids=ids, queries=queries,
                    host_contexts=host, verification_slots=drafts, invalid_slots=invalid,
                    preparation=ExecutionContextSnapshot(
                        fact["verifier_id"], str(id(schedule)), rows, correction_active,
                        self.previous_preparation, schedule.num_spec_tokens_to_schedule))
        self.previous_preparation = fact["verifier_id"]

    def padding_hook(self, delegate):
        def padding(*args, **kwargs):
            result = delegate(*args, **kwargs)
            if not self.collecting:
                return result
            def capture():
                mode, descriptor, microbatch, _, _ = result
                if microbatch:
                    raise ValueError("unsupported microbatching")
                fact = self.facts[id(self.active)]
                fact.update(graph_mode=str(mode), padded_worker_rows=(descriptor.num_reqs
                    if descriptor.num_reqs is not None else fact["worker_rows"]),
                    padded_query_tokens=int(descriptor.num_tokens) - sum(fact["queries"]))
            self.observe("padding", capture)
            return result
        return padding

    def proposer_hook(self, delegate):
        signature = inspect.signature(delegate)
        def propose(*args, **kwargs):
            if not self.collecting:
                return delegate(*args, **kwargs)
            fact = self.facts.get(id(self.active))
            def before():
                bound = signature.bind(*args, **kwargs)
                fact["proposer_calls"] += 1
                fact["proposer_k"] = int(bound.arguments["num_speculative_tokens"])
                fact["proposer_rows"] = int(bound.arguments["common_attn_metadata"].batch_size())
            self.observe("proposer_before", before)
            drafter = self.runner.drafter
            self.proposing = fact
            try:
                result = delegate(*args, **kwargs)
            finally:
                self.proposing = None
            self.observe("proposer_after", lambda: fact.update(
                draft_output_shape=tuple(int(n) for n in result.shape),
                assigned_proposer_k=int(drafter.num_speculative_tokens)))
            return result
        return propose

    def draft_dispatch_hook(self, delegate):
        model = self.runner.drafter.model
        def dispatch(instance, *args, **kwargs):
            if instance is model and self.proposing is not None:
                self.proposing["draft_forward_calls"] += 1
            return delegate(instance, *args, **kwargs)
        return dispatch

    def record_processing_hook(self, delegate):
        def processing(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return delegate(*args, **kwargs)
            finally:
                if len(self.overhead) < self.capacity * 12:
                    self.overhead.append(("post_retirement_record", time.perf_counter_ns() - start))
        return processing

    def target_dispatch_hook(self, delegate):
        model = self.runner.model
        def dispatch(instance, *args, **kwargs):
            if instance is model and self.active is not None and self.collecting:
                fact = self.facts.get(id(self.active))
                if fact is not None:
                    fact["target_forward_calls"] += 1
            return delegate(instance, *args, **kwargs)
        return dispatch

    def retirement_hook(self, delegate):
        def update(schedule, output):
            if not self.collecting:
                return delegate(schedule, output)
            fact = self.facts.get(id(schedule))
            if fact is not None and fact["label"] == "EMPTY_SCHEDULE":
                return self.observer.update(delegate, schedule, output, unsafe_requests=())
            ctx = None
            unsafe = []
            def before():
                nonlocal ctx
                if fact["materialized_output_id"] != id(output):
                    raise ValueError("S/future/O materialization pairing mismatch")
                for rid in output.req_ids:
                    request = self.scheduler.requests.get(rid)
                    if (request is None or request.is_finished() or request.num_preemptions
                            or request.num_stale_output_tokens):
                        unsafe.append(rid)
                if output.kv_connector_output is not None:
                    raise ValueError("unsupported KV connector lifecycle")
                snapshot = replace(fact["preparation"],
                    padded_worker_rows=fact["padded_worker_rows"],
                    padded_query_tokens=fact["padded_query_tokens"])
                if fact["label"] == "DECODE_ELIGIBLE":
                    counts = freeze_materialized_counts(fact["verifier_id"], output,
                        fact["incarnations"], self.vocab_size)
                    ctx = reconstruct(snapshot, counts, self.previous)
                    if any(c < fact["prompt_lengths"][r.request_id]
                           for r, c in zip(snapshot.rows, ctx.pre_contexts)):
                        raise ValueError("corrected context contradicts decode eligibility")
                worker = dict(request_ids=fact["request_ids"], contexts=fact["host_contexts"],
                    queries=fact["queries"], valid_drafts=fact["verification_slots"],
                    context_kind="CPU_OPTIMISTIC", prefill=fact["label"] != "DECODE_ELIGIBLE",
                    proposer_rows=fact["proposer_rows"], proposer_k=fact["proposer_k"])
                self.observer.worker_snapshot(schedule, worker,
                    context_snapshot=snapshot if ctx is not None else None)
                fact["preparation"] = snapshot
            self.observe("before_retirement", before)
            result = self.observer.update(delegate, schedule, output,
                unsafe_requests=unsafe if ctx is not None else None, reconstructed_context=ctx)
            if ctx is not None:
                terminal = tuple(item.request_id for bundle in result.values()
                                 for item in bundle.outputs if item.finish_reason is not None)
                self.previous = replace(ctx, terminal_requests=terminal)
            if self.observer.events and self.observer.events[-1].get("status") == "eligible":
                self.eligible_retirements += 1
                if self.eligible_retirements == 8:
                    # Four settle + anchor + three intervals. Canonical engine
                    # drains normally, with observer capture now disabled.
                    self.collecting = False
            return result
        return update
