"""CPU-only v8 async position accounting and fail-closed evidence tests.

Cannot establish GPU integration; Kaggle diagnostic is still required.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from experiments import m2_03_v1_target_logits_probe_v8 as p
from scripts import m2_03_compare_a3_v8 as cmp
from scripts import m2_03_install_a3_v8 as installer


@pytest.fixture(autouse=True)
def prepare_test(monkeypatch, tmp_path):
    monkeypatch.setattr(p.importlib.metadata, "version", lambda x: "0.30.0")
    monkeypatch.setattr(p, "_load_runtime_config", lambda: {
        "lane": "adaptive", "source": "test", "target": 200,
        "output_dir": str(tmp_path / "adaptive"), "git_sha": "testsha"
    })
    created = set()
    def create(path):
        path = Path(path)
        if path not in created:
            path.mkdir(parents=True, exist_ok=False)
            created.add(path)
    monkeypatch.setattr(p, "_new_output_dir", create)


def fake(*, base=198, drafts=(298, 299), async_mode=True, optimistic=3,
         wrong_active=False, num_reqs=1, target=200):
    prompt = [10 + i for i in range(6)]
    output = [100 + i for i in range(base)]
    positions = list(range(len(prompt) + base - 1, len(prompt) + base + len(drafts)))
    ids = [output[-1]] + list(drafts)
    if wrong_active:
        ids[0] = 44444
    req = NS(prompt_token_ids=prompt,
             output_token_ids=output + ([-1] * optimistic if async_mode else []))
    runner = NS(
        input_batch=NS(num_reqs=num_reqs, req_ids=["req-001"] * num_reqs),
        requests={"req-001": req},
        uses_mrope=False, enable_prompt_embeds=False,
        use_async_scheduling=async_mode,
        vllm_config=NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                                          data_parallel_size=1, decode_context_parallel_size=1)),
        cache_config=NS(enable_prefix_caching=True),
        query_start_loc=NS(np=np.array([0, len(positions)], dtype=np.int64)),
        positions=torch.tensor(positions, dtype=torch.int32),
        input_ids=NS(gpu=torch.tensor(ids, dtype=torch.int32)),
        discard_request_mask=NS(np=np.array([False])),
    )
    scheduler = NS(total_num_scheduled_tokens=len(positions),
                   scheduled_spec_decode_tokens={"req-001": list(drafts)})
    spec = (NS(num_draft_tokens=[len(drafts)],
               logits_indices=torch.arange(len(positions)),
               cu_num_sampled_tokens=torch.tensor([len(positions)])) if drafts else None)
    logits = torch.tensor([[0., 1., 1.]] * len(positions), dtype=torch.float32)
    return runner, scheduler, spec, logits, prompt, output


def sampled(ids):
    return NS(sampled_token_ids=torch.tensor([ids], dtype=torch.int64))


def prepare_and_finalize(tmp_path, *, base=198, drafts=(298, 299), async_mode=True,
                         wrong_active=False, ids=None):
    runner, sch, spec, logits, prompt, output = fake(
        base=base, drafts=drafts, async_mode=async_mode, wrong_active=wrong_active)
    candidate = p.prepare_v1_capture(runner, sch, logits, spec, None)
    result = p.finalize_v1_capture(runner, candidate,
                                   sampled(ids if ids is not None else [298, 299, 128585]))
    return candidate, result, prompt, output


def make_vanilla(tmp_path, adaptive_record, prefix):
    va = tmp_path / "vanilla"; va.mkdir()
    logits = np.load(tmp_path / "adaptive" / adaptive_record["logits_vector_file"])
    filename = "v.npy"; np.save(va / filename, logits)
    vanilla = {
        "schema": "m2-03.a3.raw-target-logits.v1",
        "lane": "vanilla", "vllm_version": "0.30.0",
        "logits_kind": "raw_target_before_grammar_and_sampling_processors",
        "prefix_proof_status": "COMMITTED_TOKEN_PREFIX_HASHED",
        "active_query_input_ids_verified": True,
        "target_generated_index_zero_based": 200,
        "prompt_token_count": adaptive_record["prompt_token_count"],
        "model_position_zero_based": adaptive_record["model_position_zero_based"],
        "conditioning_token_prefix_sha256": cmp.token_hash(prefix),
        "prefix_cache_enabled_config": True,
        "logits_vector_file": filename,
        "logits_vector_sha256_float32": hashlib.sha256(logits.tobytes()).hexdigest(),
    }
    (va / "records.jsonl").write_text(json.dumps(vanilla) + "\n")
    return va


def save_api(tmp_path, emitted, *, req_id="req-001"):
    resp = tmp_path / "api.json"
    resp.write_text(json.dumps({"id":req_id, "choices":[{"token_ids": emitted}]}))
    return resp


def test_async_optimistic_placeholder_does_not_shift_target(tmp_path):
    c, rec, prompt, output = prepare_and_finalize(tmp_path)
    assert c.generated_base == 198
    assert c.offset == 2
    assert c.req_output_tokens_observed == 201  # WRONG index if v7 used len()
    assert c.req_has_optimistic_placeholders
    assert rec["model_position_zero_based"] == 205
    assert rec["row_role"] == "bonus"
    assert rec["prefix_proof_status"] == "PENDING_API_FULL_PREFIX_VERIFICATION"
    assert rec["conditioning_token_prefix_sha256"] is None
    assert rec["sampler_selected_step_prefix_ids"] == [298, 299]
    assert rec["sampled_token_id_at_target"] == 128585
    assert not rec["active_query_input_ids_verified"]


def test_async_candidate_selected_and_api_prefix_verified(tmp_path):
    _, rec, prompt, output = prepare_and_finalize(tmp_path)
    emitted = output + [298, 299, 128585]
    va = make_vanilla(tmp_path, rec, prompt + emitted[:200])
    report = cmp.compare(va, tmp_path / "adaptive", save_api(tmp_path, emitted))
    assert report["api_emitted_prefix_and_input_ids_verified"]
    assert report["adaptive_had_optimistic_placeholders"]
    assert report["max_abs_logit_diff"] == 0
    assert report["same_kv_cache_state_proven"] is False


def test_rejected_before_target_does_not_write(tmp_path):
    c, rec, *_ = prepare_and_finalize(tmp_path, ids=[444, -1, -1])
    assert c is not None and rec is None
    assert not (tmp_path / "adaptive").exists()


def test_wrong_forward_token_rejected_offline(tmp_path):
    _, rec, prompt, output = prepare_and_finalize(tmp_path, wrong_active=True)
    emitted = output + [298, 299, 128585]
    va = make_vanilla(tmp_path, rec, prompt + emitted[:200])
    with pytest.raises(cmp.InvalidEvidence, match="active model input differs"):
        cmp.compare(va, tmp_path / "adaptive", save_api(tmp_path, emitted))


def test_wrong_api_target_rejected(tmp_path):
    _, rec, prompt, output = prepare_and_finalize(tmp_path)
    emitted = output + [298, 299, 128371]
    va = make_vanilla(tmp_path, rec, prompt + emitted[:200])
    with pytest.raises(cmp.InvalidEvidence, match="Sampler selected token"):
        cmp.compare(va, tmp_path / "adaptive", save_api(tmp_path, emitted))


def test_wrong_api_request_id_rejected(tmp_path):
    _, rec, prompt, output = prepare_and_finalize(tmp_path)
    emitted = output + [298, 299, 128585]
    va = make_vanilla(tmp_path, rec, prompt + emitted[:200])
    with pytest.raises(cmp.InvalidEvidence, match="Expected one Adaptive record"):
        cmp.compare(va, tmp_path / "adaptive", save_api(tmp_path, emitted, req_id="other"))


def test_early_response_corruption_rejected_by_complete_prefix(tmp_path):
    _, rec, prompt, output = prepare_and_finalize(tmp_path)
    emitted = output + [298, 299, 128585]
    va = make_vanilla(tmp_path, rec, prompt + emitted[:200])
    emitted[0] += 1
    with pytest.raises(cmp.InvalidEvidence, match="complete conditioning token prefix"):
        cmp.compare(va, tmp_path / "adaptive", save_api(tmp_path, emitted))


def test_bad_input_mapping_skipped_not_fatal(tmp_path):
    runner, sch, spec, logits, *_ = fake()
    runner.positions[2] = 99
    # No 200 row, not fatal and no evidence. Could also be an engine prefill chunk.
    assert p.prepare_v1_capture(runner, sch, logits, spec, None) is None
    assert not (tmp_path / "adaptive").exists()


def test_bad_batch_and_missing_model_state_never_kill_engine(tmp_path):
    runner, sch, spec, logits, *_ = fake(num_reqs=2)
    assert p.prepare_v1_capture(runner, sch, logits, spec, None) is None
    runner, sch, spec, logits, *_ = fake()
    del runner.requests["req-001"]
    assert p.prepare_v1_capture(runner, sch, logits, spec, None) is None
    assert not (tmp_path / "adaptive").exists()


def test_non_spec_output_in_same_position(tmp_path):
    runner, sch, spec, logits, prompt, output = fake(base=200, drafts=(), optimistic=2)
    before = logits.clone()
    c = p.prepare_v1_capture(runner, sch, logits, spec, None)
    assert c.offset == 0 and c.generated_base == 200
    rec = p.finalize_v1_capture(runner, c, sampled([128371]))
    assert rec["row_role"] == "bonus"
    assert torch.equal(logits, before)


def test_async_accepted_draft_prefix_mismatch_skipped(tmp_path):
    runner, sch, spec, logits, *_ = fake()
    c = p.prepare_v1_capture(runner, sch, logits, spec, None)
    assert p.finalize_v1_capture(runner, c, sampled([111, 299, 128585])) is None
    assert not (tmp_path / "adaptive").exists()


def test_installer_refuses_drifted_source():
    with pytest.raises(RuntimeError, match="not the checked V1 v7"):
        installer.transform("fake patch source")


def test_installer_is_idempotent_on_checked_mock_source(monkeypatch, tmp_path):
    v7 = ("from experiments.m2_03_v1_target_logits_probe import prepare_v1_capture\n"
          "from experiments.m2_03_v1_target_logits_probe import finalize_v1_capture\n")
    monkeypatch.setattr(installer, "EXPECTED_V7_SHA256", hashlib.sha256(v7.encode()).hexdigest())
    patched = installer.transform(v7)
    path = tmp_path / "gpu_model_runner.py"
    path.write_text(v7)
    assert installer._status(path) == "V7_INSTALLED"
    path.write_text(patched)
    assert installer._status(path) == "V8_INSTALLED"


def test_refuses_probe_writes_in_results_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "_load_runtime_config", lambda: {
        "lane":"adaptive", "source":"test", "target":200,
        "output_dir":str(tmp_path / "results"), "git_sha":"test"
    })
    runner, sch, spec, logits, *_ = fake()
    pending = p.prepare_v1_capture(runner, sch, logits, spec, None)
    assert p.finalize_v1_capture(runner, pending, sampled([298, 299, 128585])) is None
    assert not (tmp_path / "results").exists()
