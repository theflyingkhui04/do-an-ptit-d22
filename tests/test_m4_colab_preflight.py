"""Execute notebook setup fragments with fake hardware/modules, never CUDA/pip."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cells():
    book = json.loads((ROOT / "experiments/colab_m4_cost_profile.ipynb").read_text())
    return ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("checked_out", [False, True])
def test_first_cell_mismatch_retains_normal_evidence(tmp_path, monkeypatch, checked_out):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "do-an-ptit-d22"
    if checked_out:
        (repo / ".git").mkdir(parents=True)
    commands = []
    def read_only(command, **kwargs):
        commands.append(command)
        assert command[0] in ("nvidia-smi", "git")
        text = ("610.57.04\n" if "--query-gpu=driver_version" in command else
                "a" * 40 + "\n" if command[0] == "git" else "synthetic nvidia-smi\n")
        return SimpleNamespace(returncode=0, stdout=text, stderr="")
    monkeypatch.setattr(subprocess, "run", read_only)
    def forbidden(*a, **kw):
        pytest.fail("No installation, checkout or model call permitted on mismatch")
    monkeypatch.setattr(subprocess, "check_call", forbidden)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.6.0+cu124", version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_name=lambda _: "NVIDIA A100-SXM4-40GB",
                             get_device_capability=lambda _: (8, 0),
                             get_device_properties=lambda _: SimpleNamespace(total_memory=40*2**30))))
    source = cells()[0].replace('RUN_ID = "CHANGE-ME-unique-t4-smoke"', 'RUN_ID = "synthetic-mismatch"')
    with pytest.raises(SystemExit, match="GPU mismatch"):
        exec(compile(source, "hardware-cell", "exec"), {})
    evidence_root = repo if checked_out else tmp_path
    out = evidence_root / "experiments/runs/m4_cost_profiling/synthetic-mismatch"
    env = json.loads((out / "env.json").read_text())
    assert env["status"] == "hardware_mismatch"
    assert env["canonical_hardware"] is False and env["benchmark_result"] is False
    assert env["model_loaded"] is False and env["run_type"] == "smoke"
    assert env["expected_gpu"] == "T4" and env["observed_gpu"] == env["gpu"] == "NVIDIA A100-SXM4-40GB"
    assert env["gpu_compute_capability"] == [8, 0] and env["sm_architecture"] == "sm_80"
    assert env["gpu_memory_total_mb"] == 40960 and env["driver_version"] == "610.57.04"
    assert env["cuda_version"] == "12.4" and env["torch_version"] == "2.6.0+cu124"
    assert env["timestamp_utc"] and env["source_commit"] == ("a"*40 if checked_out else None)
    assert json.loads((out / "hardware_inspection.json").read_text()) == env
    assert "nvidia-smi" in (out / "nvidia-smi.txt").read_text()
    if not checked_out:
        assert not repo.exists(), "Evidence must not prevent a later repository clone"
    original = (out / "env.json").read_bytes()
    with pytest.raises(FileExistsError):
        exec(compile(source, "hardware-cell", "exec"), {})
    assert (out / "env.json").read_bytes() == original


def version_script():
    setup = ast.parse(cells()[2])
    return next(ast.literal_eval(n.value) for n in setup.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VERSION_CHECK" for t in n.targets))


@pytest.mark.parametrize("torch_version,cuda_version,vllm_version,valid", [
    ("2.13.0+cu130", "13.0", "0.30.0", True),
    ("2.6.0+cu124", "12.4", "0.30.0", False),
    ("2.13.0+cu130", "12.6", "0.30.0", False),
    ("2.13.0+cu130", "13.0", "0.31.0", False)])
def test_isolated_version_proof_is_saved_before_assertion(tmp_path, monkeypatch, capsys,
                                                         torch_version, cuda_version, vllm_version, valid):
    proof = tmp_path / "software_versions.json"
    monkeypatch.setattr(sys, "argv", ["-c", str(proof)])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__=torch_version, version=SimpleNamespace(cuda=cuda_version), __file__=".venv/torch/__init__.py"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        __version__=vllm_version, __file__=".venv/vllm/__init__.py"))
    if valid:
        exec(compile(version_script(), "version-proof", "exec"), {})
    else:
        with pytest.raises(AssertionError):
            exec(compile(version_script(), "version-proof", "exec"), {})
    saved = json.loads(proof.read_text())
    assert saved["versions_match"] is valid
    assert saved["torch_version"] == torch_version and saved["cuda_version"] == cuda_version
    assert saved["vllm_version"] == vllm_version and saved["python_executable"] == sys.executable
    printed = capsys.readouterr().out
    assert torch_version in printed and cuda_version in printed and vllm_version in printed


def test_setup_uses_explicit_cuda_index_and_pinned_environment():
    setup, invoke = cells()[2:4]
    assert '"--index-url",\n                       "https://download.pytorch.org/whl/cu130", "torch==2.13.0+cu130"' in setup
    assert 'str(PYTHON), "-m", "pip", "install"' in setup
    assert "requirements-m4-cuda.txt" in setup and '"https://pypi.org/simple"' in setup
    assert 'str(PYTHON), "-c", VERSION_CHECK' in setup
    assert "checked.check_returncode()" in setup and '"versions_match"' in invoke
    assert "setup_software_versions.json" in invoke
    for index, source in enumerate(cells()):
        compile(source, f"notebook-cell-{index}", "exec")
