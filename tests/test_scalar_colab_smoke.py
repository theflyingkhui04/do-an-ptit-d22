"""CPU transport/snapshot/packaging tests. Never live GPU evidence."""
import ast
from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.scalar_live_observer import LiveObserver
from experiments import scalar_colab_smoke as smoke
from experiments.scalar_smoke_evidence import replay
from experiments.scalar_smoke_evidence import validate, PINNED_SOURCES
from experiments.scalar_pinned_sources import manifest as pinned_manifest


class Model:
    def __call__(self):
        return None


class Harness:
    def __init__(self, k=2):
        self.k = k
        self.model = Model()
        self.materializations = 0
        self.calls = []
        self.runner = NS(input_batch=NS(num_reqs=1, req_ids=["r"],
            num_computed_tokens_cpu=np.array([40]), prev_req_id_to_index={}),
            prev_positions=NS(np=np.array([-1])), prev_num_draft_tokens=NS(np=np.array([0])),
            query_start_loc=NS(np=np.array([0, k + 1])), use_async_spec_decode=True,
            valid_sampled_token_count_gpu=None, execute_model_state=None)
        def propose(num_speculative_tokens, common_attn_metadata):
            self.runner.drafter.num_speculative_tokens = num_speculative_tokens
            for _ in range(max(1, num_speculative_tokens)):
                self.model()
            return NS(shape=(1, num_speculative_tokens))
        self.runner.model = Model()
        self.runner.drafter = NS(propose=propose, model=self.model, num_speculative_tokens=4)
        self.runner._prepare_inputs = lambda schedule, queries: (None, None, 1)
        self.runner._determine_batch_execution_and_padding = lambda: (
            "NONE", NS(num_tokens=k + 1, num_reqs=1), False, None, None)
        request = NS(num_preemptions=0, num_stale_output_tokens=0, is_finished=lambda: False,
            spec_token_ids=[-1] * k, num_prompt_tokens=10, num_output_tokens=0)
        self.scheduler = NS(requests={"r": request},
            _plan=NS(batch_k=k, per_request_k={"r": k}), update_from_output=self.retire)
        self.executor = NS(driver_worker=NS(worker=NS(model_runner=self.runner)),
                           execute_model=self.execute, sample_tokens=self.sample)
        self.core = NS(scheduler=self.scheduler, model_executor=self.executor)

    def schedule(self):
        return NS(num_scheduled_tokens={"r": self.k + 1},
            scheduled_spec_decode_tokens={"r": [-1] * self.k},
            num_spec_tokens_to_schedule=self.k, num_invalid_spec_tokens={},
            scheduled_cached_reqs=NS(resumed_req_ids=set()), preempted_req_ids=None)

    def execute(self, schedule, non_block=True):
        self.runner._prepare_inputs(schedule, np.array([self.k + 1]))
        self.runner._determine_batch_execution_and_padding()
        self.runner.model()
        self.runner.execute_model_state = NS(scheduler_output=schedule)
        future = Future()
        future.set_result(None)
        return future

    def sample(self, count=1):
        self.runner.drafter.propose(self.k, NS(batch_size=lambda: 1))
        future = Future()
        future.set_result(NS(req_ids=["r"], req_id_to_index={"r": 0},
            sampled_token_ids=[[10] * count], kv_connector_output=None))
        return future

    def retire(self, schedule, output):
        self.calls.append((schedule, output))
        # Canonical stopping may mutate raw rows. Keep this test's nonterminal
        # result deliberately clipped to prove raw V is frozen beforehand.
        output.sampled_token_ids[0][:] = output.sampled_token_ids[0][:1]
        return {0: NS(outputs=[NS(request_id="r", new_token_ids=[10], finish_reason=None)])}


def export(observer):
    result = smoke.observer_dump(observer)
    result["evidence_layer"] = "CPU_SYNTHETIC_ONLY"
    # Exercise the actual JSON roundtrip used after a cohort ends.
    return json.loads(json.dumps(result))


def test_live_hooks_two_inflight_raw_counts_pairing_and_exact_correction():
    h = Harness()
    original = Model.__call__
    observer = LiveObserver(h.core, 100).install()
    try:
        s1, s2 = h.schedule(), h.schedule()
        e1 = h.executor.execute_model(s1)
        f1 = h.executor.sample_tokens(count=1)
        h.runner.input_batch.num_computed_tokens_cpu[0] = 43
        h.runner.input_batch.prev_req_id_to_index = {"r": 0}
        h.runner.prev_positions.np[0] = 0
        h.runner.prev_num_draft_tokens.np[0] = 2
        h.runner.valid_sampled_token_count_gpu = object()  # never read
        e2 = h.executor.execute_model(s2)
        f2 = h.executor.sample_tokens(count=3)
        assert all(f["materialized_output_id"] is None for f in observer.facts.values())
        o1 = f1.result()  # canonical caller chooses when to wait
        assert h.scheduler.update_from_output(s1, o1)[0].outputs[0].new_token_ids == [10]
        o2 = f2.result()
        h.scheduler.update_from_output(s2, o2)
        assert not observer.failures
        events = observer.observer.events
        assert events[0]["worker"]["contexts"] == (40,)
        assert events[1]["worker"]["contexts"] == (41,)  # 40 + raw V1; never host 43
        assert events[1]["raw_sampled_counts_before_update"] == (("r", 3),)
        assert events[1]["emitted"] == 1
        assert len(h.calls) == 2
        assert observer.facts[id(s2)]["execution_future_id"] == id(e2)
        assert observer.facts[id(s2)]["output_future_id"] == id(f2)
        assert observer.facts[id(s2)]["draft_forward_calls"] == 2
        values = replay(export(observer)["events"])
        assert values[-1].pre_contexts == (41,)
    finally:
        observer.close()
    assert Model.__call__ is original
    assert "execute_model" in vars(h.executor)  # original instance method restored


@pytest.mark.parametrize("k", [0, 2])
def test_capture_stops_after_tiny_window_and_canonical_engine_drains(k):
    h = Harness(k)
    observer = LiveObserver(h.core, 100).install()
    try:
        for i in range(10):
            h.runner.input_batch.num_computed_tokens_cpu[0] = 40 + i
            s = h.schedule()
            h.executor.execute_model(s)
            future = h.executor.sample_tokens()
            h.scheduler.update_from_output(s, future.result())
        assert len(h.calls) == 10
        assert len(observer.observer.events) == 8
        assert not observer.collecting and not observer.failures
        fact = next(iter(observer.facts.values()))
        assert fact["draft_output_shape"] == (1, k)
        assert fact["draft_forward_calls"] >= 1  # K0 still dispatches model
    finally:
        observer.close()


def test_observation_failure_does_not_replace_canonical_output():
    h = Harness()
    observer = LiveObserver(h.core, 100).install()
    try:
        s = h.schedule()
        h.executor.execute_model(s)
        future = h.executor.sample_tokens()
        output = future.result()
        observer.facts[id(s)]["materialized_output_id"] = -1
        result = h.scheduler.update_from_output(s, output)
        assert result[0].outputs[0].new_token_ids == [10]
        assert len(h.calls) == 1 and observer.failures
        assert observer.observer.events[0]["status"] == "unsupported"
    finally:
        observer.close()


def test_empty_scheduler_output_is_retained_without_false_worker_failure():
    h = Harness()
    observer = LiveObserver(h.core, 100).install()
    try:
        schedule = h.schedule()
        schedule.num_scheduled_tokens = {}
        schedule.scheduled_spec_decode_tokens = {}
        observer.submission(schedule)
        output = NS(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        observer.retirement_hook(lambda s, o: {})(schedule, output)
        assert not observer.failures
        assert observer.observer.events[0]["reason"] == "EMPTY_SCHEDULE"
    finally:
        observer.close()


def synthetic_smoke(root):
    """Synthetic transport fixture, never written to a live evidence directory."""
    from tests.test_scalar_cost_evidence import fixture
    identity = fixture()[0]["observed_identity"]
    root.mkdir()
    smoke.save(root / "manifest.json", dict(artifact_kind="scalar_observer_tiny_smoke",
        gpu_execution_requested_from_codex=False, cells=[0, 2], worker_batch=1, intervals=3,
        git_sha="a" * 40, status="COLLECTED"))
    captured = pinned_manifest()
    installed_hashes = {row["path"]: row["sha256"] for row in captured["files"]
                        if row["path"] in {"vllm/" + path for path in PINNED_SOURCES}}
    smoke.save(root / "source_manifest.json", {"synthetic": True, "git_sha": "a" * 40,
        "pinned_vllm_sources": {name: {"sha256": digest} for name, digest in installed_hashes.items()}})
    smoke.save(root / "config.json", smoke.CONFIG)
    smoke.save(root / "environment.json", {"synthetic": True})
    smoke.save(root / "pre_gpu_tests.json", dict(exit_code=0, passed=230, skipped=0))
    (root / "pre_gpu_tests.log").write_text("SYNTHETIC TEST FIXTURE ONLY")
    records = []
    for k in (0, 2):
        folder = root / f"k{k}"
        folder.mkdir()
        h = Harness(k)
        observer = LiveObserver(h.core, 100).install()
        try:
            for i in range(8):
                h.runner.input_batch.num_computed_tokens_cpu[0] = 40 + i
                schedule = h.schedule()
                h.executor.execute_model(schedule)
                future = h.executor.sample_tokens()
                h.scheduler.update_from_output(schedule, future.result())
            data = export(observer)
        finally:
            observer.close()
        records.append(smoke.tiny_window(data, k, 2))
        smoke.save(folder / "observer.json", data)
        smoke.save(folder / "cell.json", dict(k=k, status="PASS",
            enabled_output_token_ids=[10], disabled_output_token_ids=[10]))
        smoke.save(folder / "environment.json", {"identity": identity, "installed_source_hashes": installed_hashes})
        import hashlib
        smoke.save(folder / "fixture.json", dict(prompt="synthetic", prompt_token_ids=[1],
            prompt_sha256=hashlib.sha256(b"synthetic").hexdigest(),
            tokenized_prompt_sha256=hashlib.sha256(b"[1]").hexdigest()))
        (folder / "stdout-stderr.log").write_text("SYNTHETIC TEST FIXTURE ONLY")
    (root / "raw.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    smoke.seal(root)
    return root


def test_offline_smoke_roundtrip_and_exact_reference(tmp_path):
    report = validate(synthetic_smoke(tmp_path / "synthetic"))
    assert report["offline_validation"] == "PASS"
    assert report["context_matching"] == "MATCHED_REFERENCE"
    assert report["consumer_profile"] is False


@pytest.mark.parametrize("corruption", ["pairing", "context", "hash", "lifecycle", "failure"])
def test_offline_smoke_rejects_corruption_and_retains_failures(tmp_path, corruption):
    root = synthetic_smoke(tmp_path / "synthetic")
    path = root / "k2/observer.json"
    data = json.loads(path.read_text())
    if corruption == "pairing":
        data["execution_facts"][-1]["materialized_output_id"] = -1
    elif corruption == "context":
        data["events"][-1]["worker"]["contexts"][0] += 1
    elif corruption == "lifecycle":
        data["events"][-1]["status"] = "unsupported"
    elif corruption == "failure":
        data["failures"].append({"stage": "execute_model", "error": "OOM"})
    else:
        path.write_text("corrupt")
    if corruption != "hash":
        smoke.save(path, data)
        smoke.seal(root)  # even structurally resealed corruption is rejected
    with pytest.raises((ValueError, RuntimeError)):
        validate(root)


def test_hardware_gate_rejects_substitutions_and_multiple_devices(monkeypatch):
    for text in ("NVIDIA A10, 610\n", "NVIDIA L4, 610\nNVIDIA L4, 610\n", "Tesla T4, 610\n"):
        monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **kw: NS(returncode=0, stdout=text, stderr=""))
        with pytest.raises(RuntimeError, match="CANONICAL ENVIRONMENT MISMATCH"):
            smoke.hardware_gate()


def test_hooks_have_no_device_reads_sync_io_or_early_materialization():
    path = Path(__file__).resolve().parents[1] / "experiments/scalar_live_observer.py"
    tree = ast.parse(path.read_text())
    forbidden = {"synchronize", "item", "cpu", "numpy", "tolist", "result", "get_output",
                 "open", "write", "write_text", "dumps", "dump", "sleep"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in forbidden, name


def test_git_notebook_template_compiles_and_requires_exact_sha(tmp_path):
    from experiments.scalar_colab_git import NOTEBOOK, render_handoff
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root / NOTEBOOK).read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    for i, text in enumerate(sources):
        compile(text, f"cell-{i}", "exec")
    assert "nvidia-smi" in sources[0] and "NVIDIA L4" in sources[0]
    assert "get_device_capability" in sources[0]
    assert any('"--detach", CODE_SHA' in text for text in sources)
    assert any('userdata.get("GITHUB_TOKEN")' in text for text in sources)
    assert not any("files.upload" in text or "drive.mount" in text or "import modal" in text for text in sources)
    output = tmp_path / "handoff.ipynb"
    render_handoff(root / NOTEBOOK, output, "a" * 40)
    handoff = json.loads(output.read_text())
    assert 'CODE_SHA = "' + "a" * 40 + '"' in "".join(handoff["cells"][1]["source"])
    with pytest.raises(FileExistsError):
        render_handoff(root / NOTEBOOK, output, "a" * 40)
