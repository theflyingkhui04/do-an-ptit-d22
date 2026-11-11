"""L2 CPU proof against preserved pinned tensor functions; no GPU execution."""
import __future__
import ast
import copy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace as NS

import pytest
import torch

from experiments.scalar_context import (
    ContextRow, ExecutionContextSnapshot, MaterializedCounts, UnsupportedShape,
    freeze_materialized_counts, reconstruct, reference_match,
)
from tests.test_scalar_cost_capture import source, method

UTILS = "vllm/v1/spec_decode/utils.py"
SAMPLER = "vllm/v1/sample/rejection_sampler.py"


def extracted(path, name, **scope):
    node = copy.deepcopy(next(n for n in ast.walk(source(path))
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    # Remove compile/staticmethod wrappers only. The entire function body stays
    # unchanged and executes with real torch CPU tensors, never CUDA tensors.
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


@pytest.fixture(scope="module")
def correction():
    return extracted(UTILS, "update_num_computed_tokens_for_batch_change", torch=torch)


def row(rid="a", context=40, drafts=2, previous=-1, prev_drafts=0, invalid=0):
    return ContextRow(rid, rid + ":1", "fixture-" + rid, context, drafts + 1,
                      previous, prev_drafts, drafts, invalid, drafts)


def snap(name="e1", rows=None, active=False, source_id=None, k=2, **kwargs):
    return ExecutionContextSnapshot(name, "S-" + name, tuple(rows or [row()]), active,
                                    source_id, k, padded_worker_rows=len(rows or [row()]), **kwargs)


def counts(snapshot, values):
    return MaterializedCounts(snapshot.execution_id,
                              tuple((r.request_id, r.incarnation, v) for r, v in zip(snapshot.rows, values)))


@pytest.mark.parametrize("drafts,accepted", [(2, 2), (4, 2), (4, 0), (3, 3), (0, 0)])
def test_exact_all_partial_first_rejection_bonus_and_k0(correction, drafts, accepted):
    initial = snap(rows=[row(drafts=drafts)], k=drafts)
    first = reconstruct(initial, counts(initial, [accepted + 1]))
    # For K0 there is no participating predecessor: the actual source uses H.
    host = 40 + accepted + 1 if drafts == 0 else 999
    current = snap("e2", [row(context=host, drafts=drafts, previous=0, prev_drafts=drafts)],
                   True, "e1", k=drafts)
    result = reconstruct(current, counts(current, [1]), first)
    gpu = torch.tensor(first.pre_contexts, dtype=torch.int32)
    correction(gpu, torch.ones(1, dtype=torch.int32), torch.tensor([0]),
               torch.tensor(first.valid_counts), torch.tensor([drafts]), torch.tensor([host]))
    assert result.pre_contexts == tuple(gpu.tolist()) == (40 + accepted + 1,)
    assert result.record_fields()["rows"][0]["query_positions"] == list(range(41 + accepted, 42 + accepted + drafts))


def test_heterogeneous_invalid_slots_and_reordered_rows(correction):
    initial = snap(rows=[row("a", 40, 4, invalid=1), row("b", 70, 2)], k=4)
    first = reconstruct(initial, counts(initial, [2, 3]))
    current = snap("e2", [row("b", 900, 2, 1, 2), row("a", 901, 4, 0, 4, 1)], True, "e1", k=4)
    result = reconstruct(current, counts(current, [1, 2]), first)
    gpu = torch.tensor(first.pre_contexts, dtype=torch.int32)
    correction(gpu, torch.ones(2, dtype=torch.int32), torch.tensor([1, 0]),
               torch.tensor(first.valid_counts), torch.tensor([4, 2]), torch.tensor([900, 901]))
    assert result.pre_contexts == tuple(gpu.tolist()) == (73, 42)
    fields = result.record_fields()
    assert fields["context_source"] == "reconstructed"
    assert fields["rows"][1]["valid_speculative_slots"] == 3
    assert fields["rows"][1]["verification_query_length"] == 5  # includes invalid slot


def test_bounded_cross_product_matches_unchanged_pinned_tensor_body(correction):
    combinations = 0
    for start in (0, 1, 40, 1024):
        for k in (0, 2, 3, 4):
            for invalid in range(k + 1):
                for accepted in range(k - invalid + 1):
                    for previous_row in (-1, 0):
                        for active in (False, True):
                            first_snapshot = snap(rows=[row(context=start, drafts=k, invalid=invalid)], k=k)
                            first = reconstruct(first_snapshot, counts(first_snapshot, [accepted + 1]))
                            current = snap("e2", [row(context=start + k + 1, drafts=k,
                                                       previous=previous_row, prev_drafts=k, invalid=invalid)],
                                           active, "e1", k=k)
                            result = reconstruct(current, counts(current, [1]), first)
                            gpu = torch.tensor([start], dtype=torch.int32)
                            if active:
                                correction(gpu, torch.ones(1, dtype=torch.int32), torch.tensor([previous_row]),
                                           torch.tensor([accepted + 1]), torch.tensor([k]),
                                           torch.tensor([start + k + 1]))
                            else:
                                gpu.copy_(torch.tensor([start + k + 1]))
                            assert result.pre_contexts == tuple(gpu.tolist())
                            fields = result.record_fields()["rows"][0]
                            assert fields["corrected_post_execution_context"] >= fields["pre_execution_context"]
                            combinations += 1
    assert combinations == 512


@pytest.mark.parametrize("discard", [False, True])
def test_existing_cpu_parse_preserves_padded_kernel_valid_counts(discard):
    parse = extracted(SAMPLER, "parse_output", PLACEHOLDER_TOKEN_ID=-1)
    padded = torch.tensor([[9, 10, -1, 300], [20, -1, -1, -1]], dtype=torch.int32)
    rows, _ = parse(padded, 256, [0] if discard else [])
    frozen = freeze_materialized_counts("e1", NS(req_ids=["b", "a"],
                                                req_id_to_index={"b": 0, "a": 1},
                                                sampled_token_ids=rows), {"a": "a:1", "b": "b:1"}, 256)
    expected = [0 if discard else 2, 1]
    assert [v for _, _, v in frozen.counts] == expected
    # Exact same validity predicate is present in the GPU count producer.
    kernel = next(n for n in ast.walk(source(UTILS)) if isinstance(n, ast.FunctionDef)
                  and n.name == "eagle_prepare_next_token_padded_kernel")
    mask = next(n.value for n in ast.walk(kernel) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "is_valid_mask" for t in n.targets))
    assert ast.unparse(mask) == "(token_ids != -1) & (token_ids < vocab_size) & token_mask"


@pytest.mark.parametrize("stop", ["eos", "max_tokens"])
def test_terminal_raw_count_before_clipping_differs_from_useful_output(stop):
    parse = extracted(SAMPLER, "parse_output", PLACEHOLDER_TOKEN_ID=-1)
    tokens, _ = parse(torch.tensor([[99, 8, 9]]), 256)
    output = NS(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=tokens)
    frozen = freeze_materialized_counts("e1", output, {"a": "a:1"}, 256)
    statuses = NS(FINISHED_STOPPED="stop", FINISHED_LENGTH_CAPPED="length")
    check = method("vllm/v1/core/sched/utils.py", "check_stop", RequestStatus=statuses)
    update = method("vllm/v1/core/sched/scheduler.py", "_update_request_with_output", check_stop=check)
    class Request:
        pooling_params = None
        sampling_params = NS(eos_token_id=99 if stop == "eos" else -1, stop_token_ids=[],
                             min_tokens=0, repetition_detection=None)
        max_tokens = 10 if stop == "eos" else 1
        output_token_ids = []
        @property
        def num_output_tokens(self): return len(self.output_token_ids)
        @property
        def num_tokens(self): return 40 + self.num_output_tokens
        def append_output_token_ids(self, token): self.output_token_ids.append(token)
    emitted, stopped = update(NS(max_model_len=4096), Request(), output.sampled_token_ids[0])
    assert stopped and len(emitted) == 1 and frozen.counts == (("a", "a:1", 3),)
    initial = snap(rows=[replace(row(), committed_output_before=7)])
    result = reconstruct(initial, frozen, emitted_counts={"a": 1}, terminal_requests=("a",))
    fields = result.record_fields()["rows"][0]
    assert fields["corrected_post_execution_context"] == 43
    assert fields["committed_output_after"] == 8
    assert not fields["next_execution_eligible"]
    current = snap("e2", [row(previous=0, prev_drafts=2)], True, "e1")
    with pytest.raises(UnsupportedShape, match="terminal"):
        reconstruct(current, counts(current, [1]), result)


def test_two_snapshots_inflight_resolve_without_reading_latest_host_state():
    initial = snap()
    current = snap("e2", [row(context=9000, previous=0, prev_drafts=2)], True, "e1")
    with pytest.raises(UnsupportedShape, match="previous execution"):
        reconstruct(current, counts(current, [1]))
    first = reconstruct(initial, counts(initial, [2]))  # normal E1 output arrives later
    result = reconstruct(current, counts(current, [1]), first)
    assert result.pre_contexts == (42,)
    with pytest.raises(FrozenInstanceError): current.rows[0].cpu_context = 42


@pytest.mark.parametrize("flag", ["preemption", "resume", "stale", "abort", "prefill", "discard"])
def test_lifecycle_outside_initial_protocol_is_unsupported(flag):
    initial = snap(lifecycle_flags=(flag,))
    with pytest.raises(UnsupportedShape, match="UNSUPPORTED_SHAPE"):
        reconstruct(initial, counts(initial, [1]))


@pytest.mark.parametrize("missing", ["current", "previous", "source_id", "incarnation", "bonus"])
def test_missing_correction_metadata_is_never_guessed(missing):
    initial = snap()
    first = reconstruct(initial, counts(initial, [2]))
    current = snap("e2", [row(previous=0, prev_drafts=2)], True, "e1")
    values = counts(current, [1])
    if missing == "current": values = None
    elif missing == "previous": first = None
    elif missing == "source_id": current = replace(current, counts_source_execution_id="unknown")
    elif missing == "incarnation": current = replace(current, rows=(replace(current.rows[0], incarnation="a:2"),))
    elif missing == "bonus": values = counts(current, [0])
    with pytest.raises(UnsupportedShape, match="UNSUPPORTED_SHAPE"):
        reconstruct(current, values, first)


def test_reference_match_requires_actual_mapped_trajectory_not_bins_or_equal_yield():
    ref_snapshot = snap("r", [row("ref", drafts=0)], k=0)
    cand_snapshot = snap("c", [row("candidate")])
    ref_snapshot = replace(ref_snapshot, rows=(replace(ref_snapshot.rows[0], fixture_slot="slot1"),))
    cand_snapshot = replace(cand_snapshot, rows=(replace(cand_snapshot.rows[0], fixture_slot="slot1"),))
    reference = reconstruct(ref_snapshot, counts(ref_snapshot, [1]))
    candidate = reconstruct(cand_snapshot, counts(cand_snapshot, [3]))
    assert reference_match([reference], [candidate], identity_equal=True) == "MATCHED_REFERENCE"
    assert reference_match([reference], [candidate]) == "UNMATCHED_REFERENCE"
    shifted = replace(cand_snapshot, rows=(replace(cand_snapshot.rows[0], cpu_context=41),))
    assert reference_match([reference], [reconstruct(shifted, counts(shifted, [3]))],
                           identity_equal=True) == "UNMATCHED_REFERENCE"


def test_no_new_synchronization_or_mutable_runtime_objects():
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "experiments/scalar_context.py").read_text())
    banned = {"synchronize", "item", "cpu", "numpy", "tolist", "result", "get_output", "to"}
    assert not any(isinstance(n, ast.Call) and getattr(n.func, "attr", None) in banned for n in ast.walk(tree))


def test_observer_binds_delayed_reconstruction_to_two_inflight_snapshots():
    from experiments.scalar_cost_capture import RetirementObserver
    from tests.test_scalar_cost_capture import RUNTIME, schedule, shape
    observer = RetirementObserver(RUNTIME, clock=iter([10, 20]).__next__, intervals=1)
    schedules = [schedule(str(i), ids=("a",)) for i in (1, 2)]
    snapshots = []
    for index, s in enumerate(schedules):
        observer.submit(s, placeholder_counts={"a": 2}, incarnations={"a": "a:1"})
        snapshot = snap(f"v{index + 1}", [row(context=40 if index == 0 else 999,
                                             previous=-1 if index == 0 else 0,
                                             prev_drafts=0 if index == 0 else 2)],
                        active=bool(index), source_id=None if index == 0 else "v1")
        snapshot = replace(snapshot, scheduler_output_id=str(id(s)),
                           rows=tuple(replace(r, committed_output_before=10) for r in snapshot.rows))
        snapshots.append(snapshot)
        worker = shape(s, context_kind="CPU_OPTIMISTIC")
        worker["contexts"] = [snapshot.rows[0].cpu_context]
        observer.worker_snapshot(s, worker, context_snapshot=snapshot)
    previous = None
    for s, snapshot in zip(schedules, snapshots):
        output = NS(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[1, 2]])
        result = reconstruct(snapshot, freeze_materialized_counts(snapshot.execution_id, output, {"a": "a:1"}, 256), previous)
        emitted = {0: NS(outputs=[NS(request_id="a", new_token_ids=[1, 2], finish_reason=None)])}
        assert observer.update(lambda s, o: emitted, s, output, unsafe_requests=(),
                               reconstructed_context=result) is emitted
        previous = result
    assert [r["status"] for r in observer.events] == ["eligible", "eligible"]
    assert observer.events[1]["worker"]["contexts"] == (42,)
    assert observer.events[1]["worker"]["cpu_contexts_forecast"] == (999,)
    assert observer.events[1]["worker"]["context_kind"] == "RECONSTRUCTED"
    assert observer.events[1]["context_reconstruction"]["rows"][0]["committed_output_after"] == 12
    assert observer.windows[0]["emitted"] == 2  # anchor excluded


def test_observer_keeps_raw_counts_before_delegate_mutates_output():
    from experiments.scalar_cost_capture import RetirementObserver
    from tests.test_scalar_cost_capture import RUNTIME, schedule
    observer = RetirementObserver(RUNTIME)
    s = schedule("1", ids=("a",))
    observer.submit(s)
    output = NS(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[99, 8, 9]])
    def consume(s, o):
        del o.sampled_token_ids[0][1:]
        return {0: NS(outputs=[NS(request_id="a", new_token_ids=[99], finish_reason="stop")])}
    observer.update(consume, s, output, unsafe_requests=())
    assert observer.events[0]["raw_sampled_counts_before_update"] == (("a", 3),)
    assert observer.events[0]["emitted_by_request"] == {"a": 1}


def test_shared_k_changes_do_not_replace_previous_verification_gate():
    first_snapshot = snap("e1", [replace(row(drafts=4), effective_k=0)], k=0)
    first = reconstruct(first_snapshot, counts(first_snapshot, [3]))
    second_snapshot = snap("e2", [replace(row(context=999, drafts=0, previous=0, prev_drafts=4),
                                           effective_k=2)], True, "e1", k=2)
    second = reconstruct(second_snapshot, counts(second_snapshot, [1]), first)
    assert second.pre_contexts == (43,)
    third_snapshot = snap("e3", [row(context=44, drafts=2, previous=0, prev_drafts=0)], True, "e2", k=2)
    third = reconstruct(third_snapshot, counts(third_snapshot, [2]), second)
    assert third.pre_contexts == (44,)  # nonparticipating host-copy branch
