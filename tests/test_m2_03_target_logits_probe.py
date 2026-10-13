"""CPU-only tests: these do not establish vLLM 0.30.0 GPU integration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import m2_03_target_logits_probe as probe

spec = importlib.util.spec_from_file_location("a3compare", ROOT / "scripts/m2_03_compare_target_logits.py")
a3compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3compare)


def fake_state(*, committed: int = 206, prefix: list[int] | None = None):
    prefix = prefix or list(range(206))
    prefix_padded = prefix + [0] * max(0, 300 - len(prefix))
    states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([6], dtype=np.int32)),
        total_len=SimpleNamespace(gpu=torch.tensor([committed])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([prefix_padded])),
    )
    runner = SimpleNamespace(req_states=states, batch_sharder=None,
                             cache_config=SimpleNamespace(enable_prefix_caching=True))
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["internal-request-1"],
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_draft_tokens=0,
        has_structured_output_reqs=False,
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32),
        positions=torch.tensor([205, 206], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1], dtype=torch.int64),
        input_ids=torch.tensor([205, 206], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    logits = torch.tensor([[0.2, 1.2, 0.7], [0.1, 0.8, 1.9]])
    return runner, batch, logits


@pytest.fixture(autouse=True)
def mock_version(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "0.30.0")
    probe._OWNED_DIRS.clear()
    probe._RUNTIME_CONFIG_CACHE = None
    monkeypatch.setattr(probe, "CONFIG_PATH", tmp_path / "m203_a3_probe_config.json")
    monkeypatch.setenv("M203_A3_TARGET", "200")
    monkeypatch.setenv("M203_A3_GIT_SHA", "testsha")


def test_target_row_correct_and_prefix_hashed(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "newvanilla"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    record = probe.capture_target_logits(runner, batch, logits)
    assert record["logits_row_index_in_forward"] == 0  # model_pos205, NOT row200
    assert record["target_generated_index_zero_based"] == 200
    assert record["target_top1_id"] == 1
    assert record["prefix_proof_status"] == "COMMITTED_TOKEN_PREFIX_HASHED"
    assert record["conditioning_token_prefix_sha256"] == probe.token_ids_sha256(list(range(206)))
    assert (tmp_path / "newvanilla" / "records.jsonl").exists()


def test_no_matching_position_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.positions = torch.tensor([203, 204])
    assert probe.capture_target_logits(runner, batch, logits) is None
    assert not (tmp_path / "missing").exists()


def test_uncommitted_draft_row_is_fail_closed(tmp_path, monkeypatch):
    runner_v, batch_v, logits_v = fake_state()
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "vanilla"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    probe.capture_target_logits(runner_v, batch_v, logits_v)

    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "draft"))
    monkeypatch.setenv("M203_A3_LANE", "adaptive")
    runner, batch, logits = fake_state(committed=198)
    batch.num_draft_tokens = 2
    rec = probe.capture_target_logits(runner, batch, logits)
    assert rec["prefix_proof_status"] == "UNVERIFIED_SPECULATIVE_CONTEXT"
    assert rec["conditioning_token_prefix_sha256"] is None
    with pytest.raises(a3compare.ComparisonError, match="conditioning prefix is not verified"):
        a3compare.compare(tmp_path / "vanilla", tmp_path / "draft")


def test_refuses_existing_output_dir(tmp_path, monkeypatch):
    p = tmp_path / "old"
    p.mkdir()
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(p))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    with pytest.raises(FileExistsError):
        probe.capture_target_logits(*fake_state())


def test_compares_same_prefix_and_different_argmax(tmp_path, monkeypatch):
    rv, bv, lv = fake_state()
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "v"))
    probe.capture_target_logits(rv, bv, lv)
    ra, ba, la = fake_state()
    la[0] = torch.tensor([2.2, 1.2, 0.7])
    monkeypatch.setenv("M203_A3_LANE", "adaptive")
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "a"))
    probe.capture_target_logits(ra, ba, la)
    result = a3compare.compare(tmp_path / "v", tmp_path / "a")
    assert result["raw_argmax_equal"] is False
    assert result["max_abs_logit_diff"] > 0
    assert result["same_kv_cache_state_proven"] is False


def test_prefix_mismatch_rejected(tmp_path, monkeypatch):
    for lane in ("vanilla", "adaptive"):
        runner, batch, logits = fake_state(prefix=list(range(206)) if lane == "vanilla" else [9] * 206)
        if lane == "adaptive":
            batch.input_ids = torch.tensor([9, 9], dtype=torch.int32)
        monkeypatch.setenv("M203_A3_LANE", lane)
        monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / lane))
        probe.capture_target_logits(runner, batch, logits)
    with pytest.raises(a3compare.ComparisonError, match="conditioning_token_prefix_sha256"):
        a3compare.compare(tmp_path / "vanilla", tmp_path / "adaptive")


def test_vector_tampering_rejected(tmp_path, monkeypatch):
    runner, batch, logits = fake_state()
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "v"))
    rec = probe.capture_target_logits(runner, batch, logits)
    path = tmp_path / "v" / rec["logits_vector_file"]
    with path.open("wb") as handle:
        np.save(handle, np.array([5, 6, 7], dtype=np.float32))
    with pytest.raises(a3compare.ComparisonError, match="SHA256"):
        a3compare._read_vector(tmp_path / "v", rec)


def test_multiple_matching_rows_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "ambiguous"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.positions = torch.tensor([205, 205])
    with pytest.raises(probe.ProbeError, match="Ambiguous"):
        probe.capture_target_logits(runner, batch, logits)
    assert not (tmp_path / "ambiguous").exists()


def test_model_position_not_tensor_row_index(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "second"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.positions = torch.tensor([204, 205])
    batch.input_ids = torch.tensor([204, 205])
    rec = probe.capture_target_logits(runner, batch, logits)
    assert rec["logits_row_index_in_forward"] == 1
    assert rec["target_top1_id"] == 2


def test_mismatched_selected_model_input_id_is_fail_closed(tmp_path, monkeypatch):
    """Matched model_position is not proof that target consumed that token."""
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "wrong-selected"))
    monkeypatch.setenv("M203_A3_LANE", "adaptive")
    runner, batch, logits = fake_state()
    assert int(runner.req_states.all_token_ids.gpu[0, 205]) == 205
    assert int(batch.positions[batch.logits_indices[0]]) == 205
    batch.input_ids[0] = 999  # model received B, committed prefix has A
    with pytest.raises(probe.ProbeError, match="input token mismatch"):
        probe.capture_target_logits(runner, batch, logits)
    assert not (tmp_path / "wrong-selected").exists()


def test_mismatched_earlier_query_input_id_is_fail_closed(tmp_path, monkeypatch):
    """Checking only the selected row misses draft-context divergence."""
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "wrong-prior"))
    monkeypatch.setenv("M203_A3_LANE", "adaptive")
    runner, batch, logits = fake_state()
    batch.positions = torch.tensor([204, 205], dtype=torch.int32)
    batch.input_ids = torch.tensor([999, 205], dtype=torch.int32)
    with pytest.raises(probe.ProbeError, match="input token mismatch"):
        probe.capture_target_logits(runner, batch, logits)
    assert not (tmp_path / "wrong-prior").exists()


def test_missing_model_input_ids_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "missing-inputs"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    del batch.input_ids
    with pytest.raises(probe.ProbeError, match="input_ids"):
        probe.capture_target_logits(runner, batch, logits)
    assert not (tmp_path / "missing-inputs").exists()


def test_comparator_rejects_missing_active_query_evidence(tmp_path, monkeypatch):
    for lane in ("vanilla", "adaptive"):
        monkeypatch.setenv("M203_A3_LANE", lane)
        monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / lane))
        probe.capture_target_logits(*fake_state())

    # Simulate a legacy record produced by the older observer, which did not
    # check the actual input_ids used by the target model.
    record_path = tmp_path / "adaptive" / "records.jsonl"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("active_query_input_ids_verified")
    record.pop("active_query_tokens_checked")
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(a3compare.ComparisonError, match="active query"):
        a3compare.compare(tmp_path / "vanilla", tmp_path / "adaptive")


@pytest.mark.parametrize("req_ids", [
    ["_warmup_0_", "_warmup_1_"],
    ["_v2_mixed_warmup_decode_", "_v2_mixed_warmup_prefill_"],
])
def test_synthetic_warmup_batch_is_skipped(tmp_path, monkeypatch, req_ids):
    out = tmp_path / "warmup"
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(out))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.req_ids = req_ids
    batch.num_reqs = len(req_ids)
    assert probe.capture_target_logits(runner, batch, logits) is None
    assert not out.exists()


def test_mixed_warmup_real_request_fails_closed(tmp_path, monkeypatch):
    out = tmp_path / "mixed"
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(out))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.req_ids = ["_warmup_0_", "real-1"]
    batch.num_reqs = 2
    with pytest.raises(probe.ProbeError, match="Mixed warmup and real"):
        probe.capture_target_logits(runner, batch, logits)
    assert not out.exists()


def test_real_multi_request_remains_fail_closed(tmp_path, monkeypatch):
    out = tmp_path / "real"
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(out))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    runner, batch, logits = fake_state()
    batch.req_ids = ["real-1", "real-2"]
    batch.num_reqs = 2
    with pytest.raises(probe.ProbeError, match="one active request"):
        probe.capture_target_logits(runner, batch, logits)
    assert not out.exists()



def test_config_file_fallback_when_env_missing(tmp_path, monkeypatch):
    for key in (
        "M203_A3_OUTPUT_DIR", "M203_A3_LANE", "M203_A3_TARGET", "M203_A3_GIT_SHA"
    ):
        monkeypatch.delenv(key, raising=False)
    probe._RUNTIME_CONFIG_CACHE = None
    cfg = tmp_path / "m203_a3_probe_config.json"
    monkeypatch.setattr(probe, "CONFIG_PATH", cfg)
    out = tmp_path / "from-config"
    cfg.write_text(json.dumps({
        "lane": "adaptive",
        "target": 200,
        "git_sha": "configsha",
        "output_dir": str(out),
    }), encoding="utf-8")

    rec = probe.capture_target_logits(*fake_state())
    assert rec["config_source"] == "config_file"
    assert rec["lane"] == "adaptive"
    assert rec["git_sha"] == "configsha"
    assert (out / "records.jsonl").exists()


def test_environment_takes_priority_over_config_file(tmp_path, monkeypatch):
    probe._RUNTIME_CONFIG_CACHE = None
    cfg = tmp_path / "m203_a3_probe_config.json"
    monkeypatch.setattr(probe, "CONFIG_PATH", cfg)
    cfg.write_text(json.dumps({
        "lane": "adaptive",
        "target": 200,
        "git_sha": "filesha",
        "output_dir": str(tmp_path / "file-out"),
    }), encoding="utf-8")
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "env-out"))
    monkeypatch.setenv("M203_A3_GIT_SHA", "envsha")
    monkeypatch.setenv("M203_A3_TARGET", "200")

    rec = probe.capture_target_logits(*fake_state())
    assert rec["config_source"] == "environment"
    assert rec["lane"] == "vanilla"
    assert rec["git_sha"] == "envsha"
    assert (tmp_path / "env-out" / "records.jsonl").exists()
    assert not (tmp_path / "file-out").exists()


def test_partial_environment_fails_closed(tmp_path, monkeypatch):
    probe._RUNTIME_CONFIG_CACHE = None
    cfg = tmp_path / "m203_a3_probe_config.json"
    monkeypatch.setattr(probe, "CONFIG_PATH", cfg)
    cfg.write_text(json.dumps({
        "lane": "adaptive",
        "target": 200,
        "git_sha": "filesha",
        "output_dir": str(tmp_path / "file-out"),
    }), encoding="utf-8")
    monkeypatch.setenv("M203_A3_LANE", "adaptive")
    monkeypatch.delenv("M203_A3_OUTPUT_DIR", raising=False)

    with pytest.raises(probe.ProbeError, match="Partial M203_A3 environment"):
        probe.capture_target_logits(*fake_state())
    assert not (tmp_path / "file-out").exists()


def test_invalid_config_file_fails_closed(tmp_path, monkeypatch):
    for key in (
        "M203_A3_OUTPUT_DIR", "M203_A3_LANE", "M203_A3_TARGET", "M203_A3_GIT_SHA"
    ):
        monkeypatch.delenv(key, raising=False)
    probe._RUNTIME_CONFIG_CACHE = None
    cfg = tmp_path / "m203_a3_probe_config.json"
    monkeypatch.setattr(probe, "CONFIG_PATH", cfg)
    cfg.write_text('{"lane":"adaptive","target":"200"}', encoding="utf-8")

    with pytest.raises(probe.ProbeError, match="output_dir"):
        probe.capture_target_logits(*fake_state())


def test_tied_logits_use_first_argmax_index(tmp_path, monkeypatch):
    monkeypatch.setenv("M203_A3_OUTPUT_DIR", str(tmp_path / "tie"))
    monkeypatch.setenv("M203_A3_LANE", "vanilla")
    probe._RUNTIME_CONFIG_CACHE = None
    runner, batch, logits = fake_state()
    logits[0] = torch.tensor([0.0, 2.0, 2.0])
    rec = probe.capture_target_logits(runner, batch, logits)
    assert rec["target_top1_id"] == 1
    assert rec["target_top2_id"] == 2
    assert rec["top2_margin"] == 0.0
