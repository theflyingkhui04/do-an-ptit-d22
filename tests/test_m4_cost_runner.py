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
    part = out/f"k{args.worker_k}"
    part.mkdir()
    rows=[]
    def rpc(method,args=()):
        return [{"samples":{"target":10+args_k,"propose":2+args_k},"observed_shapes":[]}]
    args_k=args.worker_k
    p.measure_repeats(env,args_k,env["batches"],env["contexts"],lambda *a:None,rpc,rows.append)
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
    assert "OOM" in (out/"k1.log").read_text()
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
