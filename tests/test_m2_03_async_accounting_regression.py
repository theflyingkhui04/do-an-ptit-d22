"""M2-03 offline regression: pinned vLLM 0.30.0 async accounting.

Exercises the REAL method bodies (installed or hash-pinned vendored vLLM source)
inside a small, GPU-free state fixture. This is NOT an APC/KV-tensor test and
is NOT a replacement for GPU integration or the canonical 54-prompt gate.
"""

from __future__ import annotations

import ast
import importlib.metadata
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.m2_03_targeted import sources
from experiments.m2_03_targeted.common import sha

ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = ROOT / ".venv/lib/python3.13/site-packages/vllm/v1/core/sched"
BASE_SOURCE = VLLM_ROOT / "scheduler.py"
ASYNC_SOURCE = VLLM_ROOT / "async_scheduler.py"
ADAPTIVE_SOURCE = ROOT / "src/server/adaptive_scheduler.py"


def _read_method(path: Path, class_name: str, method_name: str) -> str:
    """Extract an unchanged method body from the installed pinned source."""
    if path in (BASE_SOURCE, ASYNC_SOURCE):
        name = 'v1/core/sched/' + path.name
        source = (path.read_text(encoding='utf-8') if path.is_file()
                  else sources.fixture_sources()[name])
        assert sha(source.encode()) == sources.manifest()['sha256'][name], 'pinned source drift'
    else:
        source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        pytest.fail(f"Expected exactly one {class_name} in {path}")
    methods = [node for node in classes[0].body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == method_name]
    if len(methods) != 1:
        pytest.fail(f"Expected exactly one {class_name}.{method_name} in {path}")
    method = methods[0]
    if method.decorator_list:
        pytest.fail(f"Unexpected method decorators: {class_name}.{method_name}")
    original_lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(original_lines[method.lineno - 1:method.end_lineno]))


@pytest.fixture(scope="module")
def scheduler_class():
    """Use production methods, substituting ONLY their expensive dependencies."""
    if BASE_SOURCE.is_file() or ASYNC_SOURCE.is_file():
        assert importlib.metadata.version('vllm') == '0.30.0', 'wrong installed vLLM version'
    else:
        assert sources.manifest()['tag'] == 'v0.30.0'  # No vLLM/CUDA install on Windows.

    specs = (
        ("Scheduler", None, BASE_SOURCE,
         ("_update_after_schedule", "_update_request_with_output")),
        ("AsyncScheduler", "Scheduler", ASYNC_SOURCE,
         ("_update_after_schedule", "_update_request_with_output")),
        ("AdaptiveAsyncScheduler", "AsyncScheduler", ADAPTIVE_SOURCE,
         ("_update_after_schedule",)),
    )
    chunks = ["from __future__ import annotations"]
    for name, parent, source, methods in specs:
        chunks.append(f"class {name}{f'({parent})' if parent else ''}:")
        for method in methods:
            chunks.append(textwrap.indent(_read_method(source, name, method), "    "))

    # check_stop is disabled in this fixture; all outputs are far from EOS/limit.
    # RequestStatus is used by the real AsyncScheduler cache write guard.
    namespace = {
        "__name__": "m203_async_accounting_extracted",
        "RequestStatus": SimpleNamespace(RUNNING="RUNNING", PREEMPTED="PREEMPTED"),
        "check_stop": lambda request, max_model_len: False,
    }
    script = "\n\n".join(chunks) + "\n"
    exec(compile(script, "<pinned-vllm-and-adaptive-methods>", "exec"), namespace)
    return namespace["AdaptiveAsyncScheduler"]


@dataclass(eq=False)
class FakeRequest:
    """Only state referenced by the extracted production methods."""
    request_id: str
    status: str = "RUNNING"
    num_computed_tokens: int = 13
    num_in_flight_tokens: int = 0
    num_output_placeholders: int = 3
    num_prompt_tokens: int = 10
    spec_token_ids: list[int] = field(default_factory=list)
    is_prefill_chunk: bool = False
    use_structured_output: bool = False
    _all_token_ids: list[int] = field(default_factory=lambda: list(range(12)))

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    def append_output_token_ids(self, token_id: int) -> None:
        self._all_token_ids.append(token_id)


class FakeKVCacheManager:
    def __init__(self):
        self.calls = []

    def cache_blocks(self, request: FakeRequest, boundary: int) -> None:
        self.calls.append((request.request_id, boundary))


def _make_scheduler(cls):
    scheduler = cls()  # Extracted methods have no runtime-heavy __init__.
    scheduler.requests = {
        "A": FakeRequest("A", spec_token_ids=[11, 12]),
        "B": FakeRequest("B", num_output_placeholders=0,
                         spec_token_ids=[21, 22, 23, 24]),
    }
    scheduler._inflight_prefills = set()
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.num_spec_tokens = 4
    scheduler.use_v2_model_runner = False
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler.finished_req_ids = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.max_model_len = 4096
    scheduler.kv_cache_manager = FakeKVCacheManager()
    scheduler._planner = SimpleNamespace(per_request=True)
    scheduler._plan = SimpleNamespace(per_request_k={"A": 2, "B": 4}, batch_k=4)
    return scheduler


def _schedule_two_requests(scheduler):
    """K=4 across batch, but A has only two current drafts and next k=2."""
    output = SimpleNamespace(
        num_scheduled_tokens={"A": 3, "B": 5},
        scheduled_spec_decode_tokens={"A": [11, 12], "B": [21, 22, 23, 24]},
        num_spec_tokens_to_schedule=4,
        pending_structured_output_tokens=False,
        has_structured_output_requests=False,
    )
    scheduler._update_after_schedule(output)
    return output


def test_trim_next_step_does_not_rewrite_current_step_accounting(scheduler_class):
    scheduler = _make_scheduler(scheduler_class)
    req_a = scheduler.requests["A"]
    req_b = scheduler.requests["B"]

    output = _schedule_two_requests(scheduler)

    assert output.scheduled_spec_decode_tokens["A"] == [11, 12]
    assert output.scheduled_spec_decode_tokens["B"] == [21, 22, 23, 24]
    assert req_a.spec_token_ids == [-1, -1]  # Adaptive's next-step k=2
    assert req_b.spec_token_ids == [-1] * 4  # Batch-wide next-step K=4
    assert req_a.num_computed_tokens == 16   # 13 + 3 scheduled tokens
    assert req_a.num_in_flight_tokens == 3
    assert req_a.num_output_placeholders == 6  # 3 prior + 1 bonus + 2 drafts
    assert req_b.num_output_placeholders == 5  # 0 prior + 1 bonus + 4 drafts
    assert scheduler.kv_cache_manager.calls == []  # Schedule does not cache.


def test_r14_one_token_query_preserves_real_async_accounting(scheduler_class):
    scheduler = _make_scheduler(scheduler_class)
    request = scheduler.requests['A']
    request._all_token_ids = list(range(377))  # prompt 10 + committed output 367
    request.num_computed_tokens = 377
    request.num_output_placeholders = request.num_in_flight_tokens = 1
    request.spec_token_ids = [11, 12]
    # The R14 helper is tested in scheduler glue; here consume its result using
    # the pinned schedule expression and REAL base/async/adaptive updates.
    request.spec_token_ids = []
    query = request.num_tokens + len(request.spec_token_ids) + request.num_output_placeholders - request.num_computed_tokens
    assert query == 1
    output = SimpleNamespace(num_scheduled_tokens={'A': query}, scheduled_spec_decode_tokens={},
                             num_spec_tokens_to_schedule=4, pending_structured_output_tokens=False,
                             has_structured_output_requests=False)
    scheduler._update_after_schedule(output)
    assert request.num_computed_tokens == 378
    assert request.num_in_flight_tokens == request.num_output_placeholders == 2
    assert request.spec_token_ids == [-1, -1]  # Normal next-step Adaptive plan survives.
    assert output.scheduled_spec_decode_tokens == {}
    request.num_in_flight_tokens -= 1  # update_from_output's ordinary-step share
    scheduler._update_request_with_output(request, [100], is_stale=False)
    assert request.num_output_placeholders == request.num_in_flight_tokens == 1
    assert scheduler.kv_cache_manager.calls == [('A', 377)]


def test_preempted_stale_output_does_not_decrement_or_cache(scheduler_class):
    scheduler = _make_scheduler(scheduler_class)
    request = scheduler.requests["A"]
    # Models counters already reset by upstream preemption. The stale output
    # accounting done by update_from_output is outside this focused test.
    request.status = "PREEMPTED"
    request.num_computed_tokens = 0
    request.num_output_placeholders = 0
    request.spec_token_ids = []

    returned, stopped = scheduler._update_request_with_output(
        request, [100, 101], is_stale=True
    )

    assert returned == [100, 101]
    assert stopped is False
    assert request.num_output_placeholders == 0  # no underflow/double decrement
    assert request.num_computed_tokens == 0
    assert scheduler.kv_cache_manager.calls == []  # no caching preempted state


@pytest.mark.parametrize("num_selected", [1, 2, 3])
def test_rejection_accounting_gives_safe_cache_boundary(scheduler_class, num_selected):
    scheduler = _make_scheduler(scheduler_class)
    request = scheduler.requests["A"]
    # Before step: 13 computed - 3 pending = 10 safe cached tokens.
    original_safe_boundary = request.num_computed_tokens - request.num_output_placeholders
    assert original_safe_boundary == 10

    output = _schedule_two_requests(scheduler)
    assert len(output.scheduled_spec_decode_tokens["A"]) == 2

    # The pinned Scheduler.update_from_output() rolls back rejected drafts
    # BEFORE calling AsyncScheduler._update_request_with_output().
    # num_selected includes one ordinary/bonus token; at most 2 accepted drafts.
    rejected = 2 - (num_selected - 1)
    assert 0 <= rejected <= 2
    request.num_computed_tokens -= rejected
    request.num_output_placeholders -= rejected

    generated = [100 + i for i in range(num_selected)]
    returned, stopped = scheduler._update_request_with_output(
        request, generated, is_stale=False
    )

    assert returned == generated
    assert stopped is False
    assert request.num_output_placeholders == 3  # old pending share survives
    assert request.num_computed_tokens == 13 + num_selected
    expected_boundary = original_safe_boundary + num_selected
    assert scheduler.kv_cache_manager.calls == [("A", expected_boundary)]
    assert 0 <= expected_boundary <= request.num_tokens
