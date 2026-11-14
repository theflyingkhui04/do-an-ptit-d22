"""Portable runner tests. Fake engines write only to pytest temporary folders."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import m4_cost_runner as runner
from experiments import m4_cost_profiler as p


def hardware(name="Tesla T4", capability=(7,5)):
    return dict(gpu=name,gpu_compute_capability=list(capability),gpu_memory_total_mb=15360,
                driver_version="610.57.04",cuda_version="13.0",torch_version="2.13.0+cu130",
                vllm_version="0.30.0",nvidia_smi_output="synthetic test only")


def arguments(*extra):
    return runner.parser().parse_args(["--run-id","synthetic-test-only","--expected-gpu","T4",*extra])


@pytest.mark.parametrize("name,cap,expected,canonical", [
    ("NVIDIA L4",(8,9),"L4",True), ("Tesla T4",(7,5),"T4",True),
    ("NVIDIA L4",(7,5),"L4",False), ("Tesla T4",(8,9),"T4",False),
    ("Tesla T4",(7,5),"L4",False), ("NVIDIA A100-SXM4-40GB",(8,0),"L4",False),
    ("",(),"T4",False)])
def test_observed_name_and_capability_required(name,cap,expected,canonical):
    result = p.classify_hardware(name,cap,expected)
    assert result["canonical_hardware"] is canonical
    assert result["observed_gpu"] == name


def test_cli_defaults_and_component_grid():
    args = arguments()
    assert (args.mode,args.warmup,args.repeats,args.batches,args.k_values) == ("smoke",2,2,(1,),(1,))
    args = arguments("--mode","components","--batches","1,2,4,8,16,32","--k-values","1,2,3,4,5",
                     "--contexts","256,1024","--warmup","5","--repeats","50")
    p.validate_portable_config(args.mode,args.batches,args.k_values,args.contexts,args.warmup,args.repeats)
    with pytest.raises(SystemExit):
        arguments("--batches","1,1")


@pytest.mark.parametrize("warmup,repeats", [(4,50),(5,49)])
def test_component_campaign_requires_full_repeats(warmup,repeats):
    with pytest.raises(ValueError):
        p.validate_portable_config("components",p.BATCHES,p.KS,[1024],warmup,repeats)


def test_clean_source_guard():
    runner.guard_clean_source("smoke",False)
    runner.guard_clean_source("components",True)
    with pytest.raises(ValueError,match="dirty source"):
        runner.guard_clean_source("components",False)


@pytest.fixture
def source_repo(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={tmp_path.as_posix()}", *args],
            cwd=tmp_path, text=True)
    git("init")
    (tmp_path / "source.py").write_text("value = 1\n")
    evidence = tmp_path / "experiments/runs/m4_cost_profiling/previous/env.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n")
    git("add", ".")
    git("-c", "user.name=M4 Test", "-c", "user.email=m4-test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic source fixture")
    return tmp_path


def test_only_m4_run_artifacts_are_source_clean(source_repo):
    # Both modified tracked evidence and new artifacts must be allowed.
    (source_repo / "experiments/runs/m4_cost_profiling/previous/env.json").write_text('{"status":"complete"}\n')
    for directory in runner.GENERATED_EVIDENCE_PATHS:
        artifact = source_repo / directory / "new-run/raw.jsonl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n")
    sha, clean = runner.git_identity(source_repo)
    assert len(sha) == 40 and clean
    runner.guard_clean_source("components", clean)


def test_modified_tracked_python_is_source_dirty(source_repo):
    (source_repo / "source.py").write_text("value = 2\n")
    assert runner.git_identity(source_repo)[1] is False


@pytest.mark.parametrize("path", ["src/new.py", "configs/new.yaml", "experiments/new.ipynb",
                                 "experiments/runs/m4_cost_profiling_other/source.py"])
def test_untracked_source_config_notebook_is_source_dirty(source_repo, path):
    source = source_repo / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("untracked source\n")
    assert runner.git_identity(source_repo)[1] is False


def fixture_setup(monkeypatch,clean=True):
    monkeypatch.setattr(runner,"git_identity",lambda root: ("a"*40,clean))
    monkeypatch.setattr(runner.sys,"platform","linux")


def forbidden_launch(*args,**kwargs):
    pytest.fail("No worker or model may launch on hardware mismatch or dirty source")


def test_mismatch_retains_all_artifacts_before_model_download(tmp_path,monkeypatch,capsys):
    fixture_setup(monkeypatch)
    args = arguments("--mode","components","--warmup","5","--repeats","50")
    result = runner.run(args,root=tmp_path,inspect_hardware=lambda:hardware("NVIDIA A100-SXM4-40GB",(8,0)),launch=forbidden_launch)
    assert result == 0
    out = p.evidence_dir(tmp_path,args.run_id)
    assert {"env.json","raw.jsonl","component_report.json","run_manifest.json"} <= {x.name for x in out.iterdir()}
    env = json.loads((out/"env.json").read_text())
    assert not env["canonical_hardware"] and env["run_type"] == "smoke"
    assert env["requested_run_type"] == "component_benchmark"
    assert "run retained as non-canonical smoke" in capsys.readouterr().out
    assert (out/"raw.jsonl").read_text() == ""
    assert json.loads((out/"run_manifest.json").read_text())["status"] == "hardware_mismatch"


def test_dirty_component_run_refuses_and_retains_env(tmp_path,monkeypatch):
    fixture_setup(monkeypatch,False)
    args = arguments("--mode","components","--warmup","5","--repeats","50")
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=forbidden_launch) == 1
    env = json.loads((p.evidence_dir(tmp_path,args.run_id)/"env.json").read_text())
    assert not env["working_tree_clean"] and env["status"] == "failed"


def fake_launch(command,**kwargs):
    args = runner.parser().parse_args(command[3:])
    out = p.evidence_dir(Path(kwargs["cwd"]),args.run_id)
    env = json.loads((out/"env.json").read_text())
    part = out/runner.cell_name(args.worker_k,args.worker_batch,args.worker_context)
    part.mkdir()
    rows=[]
    def rpc(method,args=()):
        return [{"samples":{"target":10+args_k,"propose":2+args_k},"observed_shapes":[]}]
    args_k=args.worker_k
    env.update(engine_kwargs=p.cell_engine_kwargs(env["expected_gpu"],args_k,args.worker_batch,args.worker_context))
    p.measure_repeats(env,args_k,[args.worker_batch],[args.worker_context],lambda *a:None,rpc,rows.append)
    (part/"raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (part/"env.json").write_text(json.dumps(dict(env,status="complete")))
    return SimpleNamespace(returncode=0)


def test_end_to_end_fake_multicontext_smoke_no_profiles(tmp_path,monkeypatch):
    fixture_setup(monkeypatch,False)  # Dirty smoke is explicit and never canonical.
    args=arguments("--contexts","256,1024","--k-values","1,2")
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=fake_launch) == 0
    out=p.evidence_dir(tmp_path,args.run_id)
    env=json.loads((out/"env.json").read_text())
    assert env["run_type"] == "smoke" and not env["canonical_component_run"]
    rows=[json.loads(x) for x in (out/"raw.jsonl").read_text().splitlines()]
    assert len(rows)==32 and sum(r["is_warmup"] for r in rows)==16
    report=json.loads((out/"component_report.json").read_text())
    assert set(report["linear_proposer_fits"])=={"b1_l256","b1_l1024"}
    assert "cost_matrix_ms" not in report and "gamma_attn" not in report
    assert report["not_consumer_profile"] and not report["full_repeat_requirement_met"]
    assert json.loads((out/"run_manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=fake_launch)


def test_components_add_baseline_and_never_derive_topologies(tmp_path,monkeypatch):
    fixture_setup(monkeypatch)
    args=arguments("--mode","components","--warmup","5","--repeats","50")
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=fake_launch)==0
    out=p.evidence_dir(tmp_path,args.run_id)
    env=json.loads((out/"env.json").read_text())
    assert env["k_values"]==[0,1] and env["canonical_component_run"]
    report=json.loads((out/"component_report.json").read_text())
    assert report["full_repeat_requirement_met"] and not report["canonical_profile_generated"]
    assert {r["measurement_kind"] for r in report["measurement_summary"]} == {
        "target_decode_forward","eagle_propose","linear_verify_forward"}


def test_failed_worker_keeps_logs_env_and_partial_raw(tmp_path,monkeypatch):
    fixture_setup(monkeypatch)
    def fail(command,**kwargs):
        fake_launch(command,**kwargs)
        kwargs["stdout"].write("synthetic OOM diagnostic\n")
        return SimpleNamespace(returncode=1)
    args=arguments()
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=fail)==1
    out=p.evidence_dir(tmp_path,args.run_id)
    assert (out/"raw.jsonl").stat().st_size > 0
    assert "OOM" in (out/"k1-b1-c1024.log").read_text()
    report=json.loads((out/"component_report.json").read_text())
    assert report["status"] == "incomplete" and not report["full_repeat_requirement_met"]


def test_interrupted_run_retains_terminal_manifest(tmp_path,monkeypatch):
    fixture_setup(monkeypatch)
    def interrupted(*args,**kwargs):
        raise KeyboardInterrupt()
    args=arguments()
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=interrupted)==1
    out=p.evidence_dir(tmp_path,args.run_id)
    manifest=json.loads((out/"run_manifest.json").read_text())
    assert manifest["status"] == "failed" and "KeyboardInterrupt" in manifest["error"]
    assert (out/"component_report.json").exists()


def test_shared_loop_retains_failure_and_stops_without_retry():
    env={"warmup":2,"repeats":2}
    records=[]
    calls=[]
    def rpc(method,args=()):
        calls.append(method)
    def failed_operation(*args):
        raise RuntimeError("synthetic CUDA OOM")
    with pytest.raises(RuntimeError,match="OOM"):
        p.measure_repeats(env,1,[1],[256],failed_operation,rpc,records.append)
    assert len(records)==2 and all(r["is_warmup"] and r["status"]=="error" for r in records)
    assert calls==["m4_begin"]
    assert all(r["error_category"] == "oom" for r in records)


@pytest.mark.parametrize("gpu", ["L4","T4"])
@pytest.mark.parametrize("batch", p.BATCHES)
def test_cell_capacity_uses_exact_batch_and_no_magic_memory_tuning(gpu,batch):
    cfg=p.cell_engine_kwargs(gpu,1,batch,1024)
    assert cfg["max_num_seqs"] == batch
    assert cfg["max_num_batched_tokens"] == max(4096,batch*1024)
    original=p.engine_kwargs(gpu,1)
    assert {k:v for k,v in cfg.items() if k not in ("max_num_seqs","max_num_batched_tokens")} == {
        k:v for k,v in original.items() if k not in ("max_num_seqs","max_num_batched_tokens")}


def test_all_requests_enqueued_before_scheduling_resumes():
    calls=[]
    prompts=[object(),object()]
    class Engine:
        def sleep(self,**kwargs): calls.append(("pause",kwargs))
        def enqueue(self,items,params,**kwargs):
            assert items is prompts
            calls.append(("enqueue",len(items)))
        def wake_up(self,**kwargs): calls.append(("resume",kwargs))
        def wait_for_completion(self,**kwargs):
            calls.append(("wait",kwargs))
            return "outputs"
    assert runner.generate_exact_batch(Engine(),prompts,object()) == "outputs"
    assert calls == [("pause",{"level":0}),("enqueue",2),
                     ("resume",{"tags":["scheduling"]}),("wait",{"use_tqdm":False})]


def test_oom_cell_does_not_retry_or_prevent_later_fresh_cells(tmp_path,monkeypatch):
    fixture_setup(monkeypatch)
    launched=[]
    def launch(command,**kwargs):
        args=runner.parser().parse_args(command[3:])
        identity=(args.worker_k,args.worker_batch,args.worker_context)
        launched.append(identity)
        if args.worker_batch != 2:
            return fake_launch(command,**kwargs)
        out=p.evidence_dir(tmp_path,args.run_id)
        part=out/runner.cell_name(*identity)
        part.mkdir()
        env=json.loads((out/"env.json").read_text())
        env.update(K=args.worker_k,batch_size=2,context_len=args.worker_context,
                   engine_kwargs=p.cell_engine_kwargs("T4",args.worker_k,2,args.worker_context),
                   status="failed",error="OutOfMemoryError: CUDA out of memory",error_category="oom")
        rows=[]
        def operation(*args): raise RuntimeError(env["error"])
        with pytest.raises(RuntimeError):
            p.measure_repeats(env,args.worker_k,[2],[args.worker_context],operation,lambda *a,**k:None,rows.append)
        (part/"raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (part/"env.json").write_text(json.dumps(env))
        return SimpleNamespace(returncode=1)
    args=arguments("--batches","1,2,4","--k-values","1,2")
    assert runner.run(args,root=tmp_path,inspect_hardware=hardware,launch=launch) == 1
    assert launched == [(k,b,1024) for k in (1,2) for b in (1,2,4)]
    out=p.evidence_dir(tmp_path,args.run_id)
    report=json.loads((out/"component_report.json").read_text())
    assert len(report["cell_results"]) == 6
    assert [w["error_category"] for w in report["cell_results"] if w["status"]=="failed"] == ["oom","oom"]
    rows=[json.loads(line) for line in (out/"raw.jsonl").read_text().splitlines()]
    assert all(r["engine_kwargs"]["max_num_seqs"] == r["batch_size"] for r in rows)
    assert sum(r["status"]=="error" for r in rows) == 4
    assert any(r["K"]==2 and r["batch_size"]==4 and r["status"]=="ok" for r in rows)


def test_missing_shape_sample_stops_cell_after_first_attempt():
    records=[]; calls=[]
    def operation(*args): calls.append("generate")
    def rpc(*args,**kwargs): return [dict(samples={},observed_shapes=[dict(actual_batch_size=1)])]
    with pytest.raises(RuntimeError,match="exact requested"):
        p.measure_repeats(dict(warmup=2,repeats=2),1,[2,4],[1024],operation,rpc,records.append)
    assert calls == ["generate"]
    assert len(records) == 2 and all(r["error_category"]=="shape_mismatch" for r in records)


def test_engine_init_failure_has_cell_config_and_original_error(tmp_path,monkeypatch):
    out=tmp_path
    env=dict(canonical_hardware=True,source_commit="a"*40,k_values=[1],batches=[2],contexts=[1024],
             expected_gpu="T4")
    runner.write_json(out/"env.json",env)
    monkeypatch.setattr(runner,"git_identity",lambda root:("a"*40,True))
    class LLM:
        def __init__(self,**kwargs): raise RuntimeError("synthetic init failure")
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(LLM=LLM,SamplingParams=object))
    monkeypatch.setitem(sys.modules,"vllm.inputs",SimpleNamespace(TokensPrompt=object))
    args=arguments("--worker-k","1","--worker-batch","2","--worker-context","1024")
    assert runner.run_worker(args,out) == 1
    part=out/"k1-b2-c1024"
    result=json.loads((part/"env.json").read_text())
    assert result["error_category"]=="engine_init_failure"
    assert result["error"]=="RuntimeError: synthetic init failure"
    assert result["engine_kwargs"]["max_num_seqs"]==2
    assert (part/"raw.jsonl").read_text()==""  # No fabricated measured repeats.


@pytest.mark.parametrize("error,phase,expected", [
    ("CUDA out of memory","engine_init","oom"),
    ("Unexpected proposal output shape","measurement","shape_mismatch"),
    ("Scheduler preemption","measurement","scheduler/preemption"),
    ("unrecognized failure","engine_init","engine_init_failure"),
    ("unrecognized failure","measurement","runtime_exception")])
def test_failure_categories(error,phase,expected):
    assert p.failure_category(error,phase=phase)==expected


def test_import_requires_no_modal_or_gpu_libraries():
    code="""import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('modal','torch','vllm'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.m4_cost_runner
"""
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_notebook_is_thin_pinned_and_inspects_before_setup():
    root=Path(__file__).resolve().parents[1]
    book=json.loads((root/"experiments/colab_m4_cost_profile.ipynb").read_text())
    cells=["".join(c["source"]) for c in book["cells"] if c["cell_type"]=="code"]
    assert "nvidia-smi" in cells[0] and "get_device_capability" in cells[0]
    assert "requirements-m4-cuda.txt" in cells[2]
    assert "experiments.m4_cost_runner" in cells[3]
    assert not any("perf_counter" in c or "def measure" in c for c in cells)
    for index,source in enumerate(cells):
        compile(source,f"cell-{index}","exec")
    pins=(root/"experiments/requirements-m4-cuda.txt").read_text()
    assert "vllm==0.30.0" in pins and "torch==2.13.0+cu130" in pins
