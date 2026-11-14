"""Pure CPU reconstruction of the pinned vLLM 0.30.0 scalar recurrence.

Consumes immutable worker-preparation snapshots and output lists ALREADY
materialized by canonical runtime, before scheduler clipping. No GPU access,
runtime installation, policy, profiling, waits or output materialization.
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = "vllm030-scalar-context-v1"


class UnsupportedShape(ValueError):
    def __init__(self, reason):
        super().__init__(f"UNSUPPORTED_SHAPE: {reason}")


def require(condition, reason):
    if not condition:
        raise UnsupportedShape(reason)


def natural(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class ContextRow:
    request_id: str
    incarnation: str
    fixture_slot: str
    cpu_context: int
    query_length: int
    previous_row: int
    previous_drafts: int
    scheduled_drafts: int
    invalid_slots: int = 0
    effective_k: int = 0
    committed_output_before: int | None = None


@dataclass(frozen=True)
class ExecutionContextSnapshot:
    execution_id: str
    scheduler_output_id: str
    rows: tuple[ContextRow, ...]
    correction_active: bool
    counts_source_execution_id: str | None
    shared_k: int
    padded_query_tokens: int = 0
    padded_worker_rows: int | None = None
    lifecycle_flags: tuple[str, ...] = ()
    mode: str = "vllm030-V1-uni-TP1-PP1-draft_model-padded-absolute"


@dataclass(frozen=True)
class MaterializedCounts:
    execution_id: str
    counts: tuple[tuple[str, str, int], ...]
    source: str = "canonical_output_before_scheduler_update"


def freeze_materialized_counts(execution_id, output, incarnations, vocab_size):
    """Read existing CPU lists; never invoke result/get_output or tensor APIs."""
    require(natural(vocab_size) and vocab_size > 0, "unknown vocabulary boundary")
    ids, mapping, tokens = output.req_ids, output.req_id_to_index, output.sampled_token_ids
    require(type(ids) is list and type(mapping) is dict and type(tokens) is list,
            "output is not canonically materialized CPU lists")
    require(len(set(ids)) == len(ids) and mapping == {rid: i for i, rid in enumerate(ids)}
            and len(tokens) == len(ids) and set(incarnations) == set(ids), "output row mapping")
    counts = []
    for rid in ids:
        row = tokens[mapping[rid]]
        require(type(row) is list and all(type(t) is int and 0 <= t < vocab_size for t in row),
                "unparsed/discarded/invalid sampled token row")
        counts.append((rid, incarnations[rid], len(row)))
    return MaterializedCounts(execution_id, tuple(counts))


@dataclass(frozen=True)
class ReconstructedExecution:
    snapshot: ExecutionContextSnapshot
    pre_contexts: tuple[int, ...]
    valid_counts: tuple[int, ...]
    terminal_requests: tuple[str, ...]
    emitted_counts: tuple[int | None, ...]

    def record_fields(self):
        """Post means prospective corrected carry, NOT a GPU tensor write now."""
        return {
            "execution_id": self.snapshot.execution_id,
            "scheduler_output_id": self.snapshot.scheduler_output_id,
            "context_source": "reconstructed", "context_reconstruction_version": VERSION,
            "shared_k": self.snapshot.shared_k,
            "padded_query_tokens": self.snapshot.padded_query_tokens,
            "padded_worker_rows": self.snapshot.padded_worker_rows,
            "rows": [dict(request_id=row.request_id, incarnation=row.incarnation,
                          fixture_slot=row.fixture_slot, pre_execution_context=context,
                          verification_query_length=row.query_length,
                          query_positions=list(range(context, context + row.query_length)),
                          verification_sequence_length=context + row.query_length,
                          gpu_context_buffer_after_execution=context,
                          corrected_post_execution_context=context + count,
                          next_execution_eligible=row.request_id not in self.terminal_requests,
                          committed_output_after=(None if row.committed_output_before is None or emitted is None
                                                  else row.committed_output_before + emitted),
                          raw_valid_sampled_count=count, useful_emitted_count=emitted,
                          valid_speculative_slots=row.scheduled_drafts - row.invalid_slots,
                          invalid_padding_slots=row.invalid_slots, effective_k=row.effective_k)
                     for row, context, count, emitted in zip(
                         self.snapshot.rows, self.pre_contexts, self.valid_counts, self.emitted_counts)],
        }


def reconstruct(snapshot, counts, previous=None, *, emitted_counts=None, terminal_requests=()):
    """Replay worker execution order, not scheduler retirement/latest-state order.

    The previous result may be supplied later after normal materialization.
    Missing dependencies raise instead of blocking, polling or guessing.
    """
    require(snapshot.mode == "vllm030-V1-uni-TP1-PP1-draft_model-padded-absolute",
            "unsupported runner/position/proposer mode")
    require(type(snapshot.rows) is tuple and snapshot.rows
            and all(isinstance(row, ContextRow) for row in snapshot.rows)
            and type(snapshot.lifecycle_flags) is tuple and not snapshot.lifecycle_flags,
            "preemption/resume/stale/abort/prefill/discard or missing rows")
    require(type(snapshot.correction_active) is bool and snapshot.execution_id
            and snapshot.scheduler_output_id, "missing execution identity or correction branch")
    require(type(snapshot.shared_k) is int and snapshot.shared_k in (0, 2, 3, 4)
            and natural(snapshot.padded_query_tokens), "invalid budget/padding")
    require(snapshot.padded_worker_rows is None or natural(snapshot.padded_worker_rows)
            and snapshot.padded_worker_rows >= len(snapshot.rows), "invalid padded row count")
    require(previous is None or previous.snapshot.execution_id != snapshot.execution_id,
            "execution ID reused")
    require(isinstance(counts, MaterializedCounts), "missing correction metadata")
    require(counts.execution_id == snapshot.execution_id
            and counts.source == "canonical_output_before_scheduler_update", "missing raw correction origin")
    require(type(counts.counts) is tuple and all(type(item) is tuple and len(item) == 3 for item in counts.counts),
            "mutable/malformed correction records")
    correction = {(rid, incarnation): value for rid, incarnation, value in counts.counts}
    keys = [(row.request_id, row.incarnation) for row in snapshot.rows]
    require(len({key[0] for key in keys}) == len(keys) and len(correction) == len(counts.counts)
            and set(correction) == set(keys), "correction/request incarnation mismatch")
    require(len({row.fixture_slot for row in snapshot.rows}) == len(keys), "fixture slot collision")
    require(set(terminal_requests).issubset({key[0] for key in keys}), "unknown terminal request")
    pre, valid, emitted = [], [], []
    for row in snapshot.rows:
        require(all(isinstance(v, str) and v for v in (row.request_id, row.incarnation, row.fixture_slot)),
                "missing request identity")
        require(natural(row.cpu_context) and natural(row.previous_drafts)
                and type(row.previous_row) is int and row.previous_row >= -1,
                "invalid pre-execution host snapshot")
        require(natural(row.scheduled_drafts) and 0 <= row.invalid_slots <= row.scheduled_drafts
                and row.query_length == row.scheduled_drafts + 1
                and row.effective_k in (0, 2, 3, 4) and row.effective_k <= snapshot.shared_k,
                "unsupported verification query/invalid slot/budget")
        require(row.committed_output_before is None or natural(row.committed_output_before),
                "invalid committed-output counter")
        context = row.cpu_context
        if previous is not None:
            require(not any((old.request_id, old.incarnation) == (row.request_id, row.incarnation)
                            and old.request_id in previous.terminal_requests for old in previous.snapshot.rows),
                    "terminal predecessor reused")
        if snapshot.correction_active and row.previous_row >= 0 and row.previous_drafts > 0:
            require(previous is not None and snapshot.counts_source_execution_id == previous.snapshot.execution_id,
                    "missing previous execution correction")
            require(previous.snapshot.execution_id != snapshot.execution_id
                    and row.previous_row < len(previous.snapshot.rows), "invalid predecessor row")
            old = previous.snapshot.rows[row.previous_row]
            require((old.request_id, old.incarnation) == (row.request_id, row.incarnation)
                    and row.request_id not in previous.terminal_requests, "reused/terminal predecessor")
            # This is the exact participating branch of the pinned tensor function.
            context = previous.pre_contexts[row.previous_row] + previous.valid_counts[row.previous_row]
        count = correction[row.request_id, row.incarnation]
        require(natural(count) and 1 <= count <= row.scheduled_drafts + 1,
                "missing/zero/inconsistent valid count; no implicit bonus")
        actual_emitted = None if emitted_counts is None else emitted_counts.get(row.request_id)
        require(actual_emitted is None or natural(actual_emitted) and actual_emitted <= count,
                "invalid scheduler emitted count")
        pre.append(context)
        valid.append(count)
        emitted.append(actual_emitted)
    return ReconstructedExecution(snapshot, tuple(pre), tuple(valid), tuple(terminal_requests), tuple(emitted))


def reference_match(reference, candidate, *, identity_equal=False):
    """Exact base-context match; different declared scalar queries are treatment.

    Requires complete mapped trajectories. No bins, tolerance, interpolation or
    equality of useful yields. This is a shape check, never profitability proof.
    """
    if not identity_equal or not reference or len(reference) != len(candidate):
        return "UNMATCHED_REFERENCE"
    for left, right in zip(reference, candidate):
        if left.snapshot.shared_k != 0 or left.terminal_requests or right.terminal_requests:
            return "UNMATCHED_REFERENCE"
        def state(value):
            return {row.fixture_slot: context for row, context in zip(value.snapshot.rows, value.pre_contexts)}
        if state(left) != state(right):
            return "UNMATCHED_REFERENCE"
        # Per-arm query validation occurred during reconstruction; no uncontrolled
        # prefill/invalid/padded queries are accepted for the initial reference.
        if (left.snapshot.mode != right.snapshot.mode
                or left.snapshot.padded_worker_rows is None or right.snapshot.padded_worker_rows is None
                or left.snapshot.padded_worker_rows != right.snapshot.padded_worker_rows
                or any(row.scheduled_drafts != row.effective_k
                       for row in (*left.snapshot.rows, *right.snapshot.rows))
                or left.snapshot.padded_query_tokens != right.snapshot.padded_query_tokens
                or any(row.invalid_slots for row in (*left.snapshot.rows, *right.snapshot.rows))):
            return "UNMATCHED_REFERENCE"
    return "MATCHED_REFERENCE"
