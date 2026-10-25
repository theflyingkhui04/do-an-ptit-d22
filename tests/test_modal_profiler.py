"""[M3-02] Forward-cost profiler: pure logic, no GPU."""

from __future__ import annotations

import pytest

from experiments import modal_lanes as lanes
from experiments import modal_profiler as prof


# ------------------------------------------------------------- grid and models
def test_default_grid_covers_workloads_and_sweep_concurrency():
    assert prof.CONTEXT_LENGTHS == (256, 512, 1024, 2048, 3072)
    assert prof.BATCH_SIZES == (1, 2, 4, 8, 16, 32)
    assert len(prof.grid()) == 30
    assert prof.grid((256,), (1, 2)) == [(256, 1), (256, 2)]


def test_grid_fits_lane_max_model_len():
    assert max(prof.CONTEXT_LENGTHS) + prof.DECODE_TOKENS + 1 <= prof.MAX_MODEL_LEN
    assert str(prof.MAX_MODEL_LEN) in lanes.COMMON_SERVING_ARGS


def test_models_use_pinned_lane_weights():
    target, draft = prof.MODELS["target"], prof.MODELS["draft"]
    assert (target.repo, target.revision) == (lanes.TARGET_MODEL, lanes.TARGET_REVISION)
    assert (draft.repo, draft.revision) == (lanes.DRAFT_MODEL, lanes.DRAFT_REVISION)
    assert target.quantization == lanes.DTYPE_PROFILES[lanes.DEFAULT_PROFILE]["quantization"]
    assert draft.quantization is None


def test_engine_kwargs_match_lane_serving_flags():
    kwargs = prof.engine_kwargs("target")

    assert kwargs["model"] == lanes.TARGET_MODEL
    assert kwargs["revision"] == lanes.TARGET_REVISION
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["quantization"] == "awq_marlin"
    assert kwargs["max_model_len"] == 4096
    assert kwargs["gpu_memory_utilization"] == 0.85
    assert kwargs["max_num_seqs"] == 32
    assert kwargs["enable_prefix_caching"] is False
    assert "quantization" not in prof.engine_kwargs("draft")


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="target"):
        prof.engine_kwargs("eagle")


# ------------------------------------------------------------- prompts
def test_prompts_have_exact_length_and_are_distinct():
    prompts = prof.prompt_token_ids(512, 4, seed=0)

    assert len(prompts) == 4
    assert all(len(p) == 512 for p in prompts)
    assert len({tuple(p) for p in prompts}) == 4


def test_prompts_are_deterministic_per_seed():
    first = prof.prompt_token_ids(256, 2, seed=7)
    again = prof.prompt_token_ids(256, 2, seed=7)
    other_seed = prof.prompt_token_ids(256, 2, seed=8)

    assert first == again
    assert first != other_seed


def test_prompt_ids_stay_in_safe_vocab_range():
    ids = [t for p in prof.prompt_token_ids(1024, 2) for t in p]
    assert min(ids) >= prof.TOKEN_ID_RANGE[0]
    assert max(ids) < prof.TOKEN_ID_RANGE[1]


# ------------------------------------------------------------- KV capacity
def test_fits_kv():
    assert prof.fits_kv(1024, 4, decode_tokens=64, capacity_tokens=4 * 1089)
    assert not prof.fits_kv(1024, 4, decode_tokens=64, capacity_tokens=4 * 1089 - 1)
    assert prof.fits_kv(3072, 32, decode_tokens=64, capacity_tokens=None)


# ------------------------------------------------------------- step time
def test_step_ms_from_two_timings():
    assert prof.step_ms(0.5, 1.78, decode_tokens=64) == pytest.approx(20.0)


@pytest.mark.parametrize("prefill_s,full_s,n", [(1.0, 1.0, 64), (1.0, 0.9, 64), (0.1, 1.0, 0)])
def test_step_ms_rejects_invalid_timings(prefill_s, full_s, n):
    with pytest.raises(ValueError):
        prof.step_ms(prefill_s, full_s, decode_tokens=n)


# ------------------------------------------------------------- measure one cell
class FakeEngine:
    """generate() advances a fake clock: 0.1 s prefill + 0.02 s per decode step."""

    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    def __call__(self, prompts, max_tokens):
        self.calls.append((len(prompts), len(prompts[0]), max_tokens))
        self.clock.now += 0.1 + 0.02 * (max_tokens - 1)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_measure_cell_runs_warmup_then_repeats():
    clock = FakeClock()
    engine = FakeEngine(clock)

    records = prof.measure_cell(engine, "target", 256, 2, warmup=3, repeats=5, decode_tokens=64, clock=clock)

    assert len(engine.calls) == (3 + 5) * 2  # each repeat = T(1) and T(1+N)
    assert {c[:2] for c in engine.calls} == {(2, 256)}
    assert {c[2] for c in engine.calls} == {1, 65}
    assert [r["repeat"] for r in records] == [1, 2, 3, 4, 5]
    assert all(r["step_ms"] == pytest.approx(20.0) for r in records)
    assert records[0]["model"] == "target"
    assert records[0]["context_len"] == 256
    assert records[0]["batch_size"] == 2


# ------------------------------------------------------------- summary
def rec(model, length, batch, step):
    return {"model": model, "context_len": length, "batch_size": batch, "step_ms": step, "status": "ok"}


def test_summarize_median_p90_and_variance():
    records = [rec("target", 256, 1, v) for v in (20.0, 21.0, 19.0, 20.5, 19.5)]

    cell = prof.summarize(records)[0]

    assert cell["step_ms_median"] == pytest.approx(20.0)
    assert cell["step_ms_min"] == pytest.approx(19.0)
    assert cell["step_ms_max"] == pytest.approx(21.0)
    assert cell["variance_ratio"] == pytest.approx(0.1)
    assert cell["variance_flag"] is False
    assert cell["n_repeats"] == 5
    assert cell["step_ms_p90"] >= cell["step_ms_median"]


def test_summarize_flags_noisy_cell_and_keeps_skipped():
    records = [rec("draft", 512, 8, v) for v in (10.0, 12.0, 10.0)]
    records.append({"model": "draft", "context_len": 3072, "batch_size": 32, "status": "skipped",
                    "reason": "kv capacity", "step_ms": None})

    cells = {(c["context_len"], c["batch_size"]): c for c in prof.summarize(records)}

    assert cells[(512, 8)]["variance_flag"] is True
    assert cells[(3072, 32)]["status"] == "skipped"
    assert cells[(3072, 32)]["step_ms_median"] is None


# ------------------------------------------------------------- evidence path
def test_evidence_dir_under_m3_02(tmp_path, monkeypatch):
    monkeypatch.setattr(prof, "REPO_ROOT", tmp_path)

    path = prof.evidence_dir("20261025-m302")

    assert path == (tmp_path / "experiments" / "runs" / "m3_02" / "20261025-m302").resolve()


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", r"x\y"])
def test_evidence_dir_rejects_path_tricks(bad):
    with pytest.raises(ValueError):
        prof.evidence_dir(bad)


# ------------------------------------------------------------- Modal wiring
def test_profiler_import_does_not_need_modal():
    import subprocess
    import sys
    from pathlib import Path

    code = "import sys, experiments.modal_profiler; assert 'modal' not in sys.modules"
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_parse_int_list():
    assert prof.parse_int_list("", (1, 2)) == (1, 2)
    assert prof.parse_int_list("256, 1024", (1,)) == (256, 1024)
    with pytest.raises(ValueError):
        prof.parse_int_list("0,5", (1,))
