"""CPU-only worker binding and failure witnesses; no GPU/system command runs."""
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest

from src.server import decision_state as identity_module
from src.server.decision_state import observe_runtime_identity
from flexsds.core.eagle_consumer_profile import validate_identity, ConsumerProfileError
from experiments.m4_pinned_contract import verify


@pytest.fixture(autouse=True)
def no_real_driver_query(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU tests must never invoke a real driver query")
    monkeypatch.setattr(identity_module.subprocess, "run", forbidden)


def runtime(*, loaded_name="awq_marlin", configured_name="awq_marlin", initialized=True):
    model=NS(model="synthetic-target", revision="a"*40, dtype="bfloat16",
             quantization=configured_name, enforce_eager=True, max_model_len=4096)
    config=NS(model_config=model,
        speculative_config=NS(draft_model_config=NS(model="synthetic-draft", revision="b"*40),
            method="eagle3", num_speculative_tokens=4, disable_padded_drafter_batch=False),
        scheduler_config=NS(async_scheduling=True, enable_chunked_prefill=False,
            max_num_seqs=8, max_num_batched_tokens=256),
        parallel_config=NS(tensor_parallel_size=1),
        cache_config=NS(enable_prefix_caching=False, gpu_memory_utilization=.9,
            cache_dtype="auto", block_size=16),
        compilation_config=NS(mode=0, cudagraph_mode="NONE"))
    loaded=NS(quant_config=NS(get_name=lambda:loaded_name))
    cls=type("GPUModelRunner", (), {"__module__":"vllm.v1.worker.gpu_model_runner"})
    runner=cls()
    runner.vllm_config=config
    runner.model_config=model
    runner.device=NS(type="cuda",index=0)
    runner.get_model=lambda:loaded
    def device():
        assert initialized, "Uninitialized CUDA device queried"
        return 0
    torch=NS(__version__="synthetic-torch",version=NS(cuda="synthetic-cuda"),
        cuda=NS(is_initialized=lambda:initialized,current_device=device,
            get_device_capability=lambda _:(8,6),get_device_name=lambda _:"SYNTHETIC observed GPU",
            get_device_properties=lambda _:NS(total_memory=24*2**30)))
    return config,runner,torch,loaded


def observe(config,runner,torch):
    return observe_runtime_identity(config,runner=runner,scheduler_class="synthetic.scheduler",
        executor_class="synthetic.executor",torch_module=torch,vllm_module=NS(__version__="0.30.0"))


def driver(monkeypatch, *, output="580.178.04\n", code=0, error=None):
    calls=[]
    def query(argv,**kwargs):
        calls.append((argv,kwargs))
        if error:
            raise error
        return NS(returncode=code,stdout=output)
    monkeypatch.setattr(identity_module.subprocess,"run",query)
    return calls


def test_bound_loaded_worker_can_close_existing_identity_contract(monkeypatch):
    config,runner,torch,loaded=runtime(configured_name="awq",loaded_name="awq_marlin")
    calls=driver(monkeypatch)
    result=observe(config,runner,torch)
    assert result.status=="SUPPORTED" and result.identity_key
    validate_identity(result)
    assert dict(result.observed)["driver_version"]=="580.178.04"
    assert dict(result.observed)["effective_quantization"]=="awq_marlin"
    assert dict(result.configured)["configured_quantization"]=="awq"
    assert calls==[(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
        dict(capture_output=True,text=True,timeout=2,check=False))]


@pytest.mark.parametrize("defect",["missing","proxy","other_config","other_model","cpu","other_device","uninitialized"])
def test_unbound_remote_or_uninitialized_runner_never_queries_host_driver(defect):
    config,runner,torch,_=runtime(initialized=defect!="uninitialized")
    if defect=="missing": runner=None
    elif defect=="proxy": runner=NS(**vars(runner))
    elif defect=="other_config": runner.vllm_config=NS()
    elif defect=="other_model": runner.model_config=NS()
    elif defect=="cpu": runner.device.type="cpu"
    elif defect=="other_device": runner.device.index=1
    result=observe(config,runner,torch)
    assert dict(result.observed)["driver_version"] is None
    assert dict(result.observed)["effective_quantization"] is None
    assert result.status=="UNSUPPORTED"


@pytest.mark.parametrize("output,code",[("",0),("580.1\n581.1\n",0),("CUDA 13.0\n",0),
    ("580.1\n",1),(" 580.1\n",0),("unknown\n",0)])
def test_invalid_driver_observation_remains_unknown_without_hiding_loaded_quantizer(monkeypatch,output,code):
    config,runner,torch,_=runtime()
    driver(monkeypatch,output=output,code=code)
    result=observe(config,runner,torch)
    assert dict(result.observed)["driver_version"] is None
    assert dict(result.observed)["effective_quantization"]=="awq_marlin"
    with pytest.raises(ConsumerProfileError): validate_identity(result)


@pytest.mark.parametrize("error",[FileNotFoundError(),subprocess.TimeoutExpired("nvidia-smi",2),OSError("unavailable")])
def test_driver_failure_remains_unknown(monkeypatch,error):
    config,runner,torch,_=runtime()
    driver(monkeypatch,error=error)
    result=observe(config,runner,torch)
    assert dict(result.observed)["driver_version"] is None and result.identity_key is None


def test_multiple_visible_devices_must_report_one_system_driver(monkeypatch):
    config,runner,torch,_=runtime()
    driver(monkeypatch,output="580.178.04\n580.178.04\n")
    assert dict(observe(config,runner,torch).observed)["driver_version"]=="580.178.04"


@pytest.mark.parametrize("defect",["missing_model","missing_quantizer","null_quantizer","missing_name",
    "error","empty","null_name","numeric_name","whitespace","online_composition"])
def test_unobservable_loaded_quantizer_never_uses_configured_quantization(monkeypatch,defect):
    config,runner,torch,loaded=runtime(configured_name="awq_marlin")
    driver(monkeypatch)
    if defect=="missing_model": runner.get_model=lambda:None
    elif defect=="missing_quantizer": del loaded.quant_config
    elif defect=="null_quantizer": loaded.quant_config=None
    elif defect=="missing_name": del loaded.quant_config.get_name
    elif defect=="error":
        def fail(): raise RuntimeError("loaded model unavailable")
        runner.get_model=fail
    elif defect=="online_composition": loaded.quant_config.online_quantization_config=NS()
    else:
        value={"empty":"","null_name":None,"numeric_name":42,"whitespace":" awq_marlin"}[defect]
        loaded.quant_config.get_name=lambda:value
    result=observe(config,runner,torch)
    assert dict(result.observed)["effective_quantization"] is None
    assert dict(result.configured)["configured_quantization"]=="awq_marlin"
    assert result.identity_key is None


def test_pinned_get_model_unwraps_loaded_model_before_quantizer_observation(monkeypatch):
    root=Path(os.environ["M4_VLLM_SOURCE_ROOT"])
    assert verify(root)["verified_files"]==19
    path=root/"vllm/v1/worker/gpu_model_runner.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="GPUModelRunner")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="get_model")
    class Wrapper:
        def __init__(self,model): self.loaded=model
        def unwrap(self): return self.loaded
    scope=dict(nn=NS(Module=object),CUDAGraphWrapper=Wrapper,UBatchWrapper=Wrapper,BreakableCUDAGraphWrapper=Wrapper)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),scope)
    config,runner,torch,loaded=runtime()
    runner.model=Wrapper(loaded)
    runner.get_model=lambda:scope["get_model"](runner)
    driver(monkeypatch)
    assert dict(observe(config,runner,torch).observed)["effective_quantization"]=="awq_marlin"


def test_new_metadata_observers_have_no_cuda_execution_or_synchronization():
    tree=ast.parse(Path(identity_module.__file__).read_text(encoding="utf-8"))
    banned={"synchronize","init","empty","tensor","to","cpu","item","forward","load_model"}
    assert not any(isinstance(n,ast.Call) and getattr(n.func,"attr",None) in banned for n in ast.walk(tree))
