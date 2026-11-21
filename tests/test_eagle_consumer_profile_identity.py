from dataclasses import replace
import pytest
from flexsds.core.eagle_consumer_profile import ConsumerProfileError
from scalar_eagle_fixtures import planner, identity


def test_complete_synthetic_identity_admitted_without_live_claim():
    p,_=planner()
    assert p.cost_governor.profile.identity_status(p.runtime_identity) == "VALID_MATCH"
    assert "SYNTHETIC" in dict(p.runtime_identity.observed)["gpu_name"]


@pytest.mark.parametrize("field,value",[("gpu_name","NVIDIA L4"),("gpu_name","A10G"),
    ("gpu_name","NVIDIA A10"),("sm_architecture","sm_86"),("vram_total_mb",23000),
    ("driver_version","another-driver"),("effective_quantization","awq"),
    ("effective_dtype","float16"),("vllm_version","0.29.0")])
def test_name_sm_driver_vram_software_exact_identity(field,value):
    p,_=planner()
    obs=dict(p.runtime_identity.observed);obs[field]=value
    runtime=replace(p.runtime_identity,observed=tuple(obs.items()))
    assert p.cost_governor.profile.identity_status(runtime) == "IDENTITY_MISMATCH"


@pytest.mark.parametrize("field",["driver_version","effective_quantization","gpu_name"])
def test_unknown_observed_identity_never_backfilled_by_configured_or_filename(field):
    p,_=planner()
    obs=dict(p.runtime_identity.observed);obs[field]=None
    runtime=replace(p.runtime_identity,observed=tuple(obs.items()))
    assert runtime.status == "UNSUPPORTED"
    assert p.cost_governor.profile.identity_status(runtime) == "UNKNOWN_IDENTITY"
    with pytest.raises(ConsumerProfileError):
        replace(p.cost_governor.profile,identity=runtime)


@pytest.mark.parametrize("field,value",[("target_revision","main"),("draft_revision","latest"),
    ("draft_method","ngram"),("configured_quantization",None)])
def test_model_revision_method_and_configured_identity_required(field,value):
    p,_=planner()
    cfg=dict(identity().configured);cfg[field]=value
    with pytest.raises(ConsumerProfileError):
        replace(p.cost_governor.profile,identity=replace(identity(),configured=tuple(cfg.items())))
