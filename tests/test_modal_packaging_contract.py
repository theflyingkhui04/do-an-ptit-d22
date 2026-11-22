"""Offline packaging checks: no Modal app, image build, or GPU is launched."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "scalar_vllm_source_manifest.json",
    "m4_eagle_source_manifest.json",
    "eagle_legacy_source_manifest.json",
)


@pytest.fixture(scope="module")
def installed_tree(tmp_path_factory):
    """Use setuptools' distribution mapping, not a hand-selected source copy."""
    destination = tmp_path_factory.mktemp("packaged-runtime")
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools import setup; setup()",
         "build_py", "--build-lib", str(destination)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return destination


# The real installed Modal SDK builds the declarative Image/mount objects.
# App/Volume registration is replaced so this cannot contact the service.
CONSTRUCT = r'''
import json, sys
from pathlib import Path
from unittest.mock import Mock, patch
import modal
sys.path.insert(0, sys.argv[1])
directories, files = [], []
add_dir, add_file = modal.Image.add_local_dir, modal.Image.add_local_file
def directory(image, local_path, remote_path, **kwargs):
    directories.append(dict(local=str(local_path), remote=remote_path, **kwargs))
    return add_dir(image, local_path, remote_path, **kwargs)
def file(image, local_path, remote_path, **kwargs):
    files.append(dict(local=str(local_path), remote=remote_path, **kwargs))
    return add_file(image, local_path, remote_path, **kwargs)
app = Mock()
app.function.return_value = lambda fn: fn
with patch.object(modal, 'App', return_value=app), \
     patch.object(modal, 'is_local', return_value=True), \
     patch.object(modal.Volume, 'from_name'), \
     patch.object(modal, 'web_server', side_effect=lambda *a, **kw: lambda fn: fn), \
     patch.object(modal.Image, 'add_local_dir', directory), \
     patch.object(modal.Image, 'add_local_file', file):
    import flexsds.server.modal_app as module
    # Lazy Images populate their mount layers only on hydration. Inspect the
    # declarative dependency graph instead, without loading remote objects.
    mounts, seen = [], set()
    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if type(obj).__name__ == 'Mount':
            mounts.extend(dict(local=str(entry.local_dir), remote=str(entry.remote_path))
                          for entry in obj.entries if hasattr(entry, 'local_dir'))
        dependencies = getattr(obj, '_deps_', None)
        if dependencies is not None:
            for dependency in dependencies():
                visit(dependency)
    visit(module.image)
    record = dict(module=module.__file__, directories=directories, files=files,
                  sdk_directory_mounts=mounts)
    Path(sys.argv[2]).write_text(json.dumps(record), encoding='utf-8')
'''


IMPORTS = r'''
import importlib, importlib.util, json, logging, sys, types
from pathlib import Path
project = Path(sys.argv[1]).resolve()
# Isolated interpreter: neither checkout nor staged host installation is visible.
sys.path[:0] = [str(project), str(project / 'src')]
names = ['src.server.adaptive_scheduler', 'src.server.lineage_executor',
         'src.proposer.controller', 'src.proposer.adapter']
origins = {name: importlib.util.find_spec(name).origin for name in names}
for origin in origins.values():
    assert Path(origin).is_relative_to(project), origin

# Resolve and execute all project imports. Only external vLLM classes and the
# exact-vLLM-source startup gate are stubbed; this proves packaging, not vLLM
# compatibility or EngineCore startup. No project runtime file is modified.
def module(name):
    obj = types.ModuleType(name)
    obj.__path__ = []
    sys.modules[name] = obj
    return obj
for name in ['vllm', 'vllm.v1', 'vllm.v1.core', 'vllm.v1.core.sched',
             'vllm.v1.executor']:
    module(name)
module('vllm.logger').init_logger = logging.getLogger
module('vllm.v1.core.sched.async_scheduler').AsyncScheduler = type('AsyncScheduler', (), {})
module('vllm.v1.executor.uniproc_executor').UniProcExecutor = type('UniProcExecutor', (), {})
startup = importlib.import_module('src.server.lineage_startup')
startup.install_enginecore_startup_binding = lambda: None
for name in names:
    imported = importlib.import_module(name)
    assert Path(imported.__file__).is_relative_to(project)
for name in ['experiments.protocol_b_boundary_observer', 'experiments.protocol_b',
             'experiments.lossless_verifier', 'experiments.workload_adapter']:
    assert Path(importlib.import_module(name).__file__).is_relative_to(project)
for name in ['scalar_vllm_source_manifest.json', 'm4_eagle_source_manifest.json',
             'eagle_legacy_source_manifest.json']:
    assert json.loads((project / 'experiments' / name).read_text())['files']
print(json.dumps(dict(origins=origins, imported=True,
                     evidence='CPU layout/import check; vLLM and startup gate stubbed')))
'''


@pytest.mark.parametrize("adaptive", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("layout", ["checkout", "installed"])
def test_explicit_mounts_and_isolated_container_imports(layout, adaptive, installed_tree, tmp_path):
    import os
    import shutil
    from modal import FilePatternMatcher

    source = ROOT / "src" if layout == "checkout" else installed_tree
    record_path = tmp_path / "mounts.json"
    env = dict(os.environ)
    env.pop("FLEXSDS_SERVING_CONFIG_B64", None)
    env["FLEXSDS_SERVING_CONFIG"] = json.dumps({"enable_adaptive": adaptive})
    result = subprocess.run(
        [sys.executable, "-I", "-c", CONSTRUCT, str(source), str(record_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert Path(record["module"]).is_relative_to(source)
    assert sorted(record["sdk_directory_mounts"], key=lambda x: x["remote"]) == sorted(
        [{key: item[key] for key in ("local", "remote")} for item in record["directories"]],
        key=lambda x: x["remote"],
    )
    if layout == "installed":
        assert {item["remote"] for item in record["directories"]} == {
            "/root/project/src/flexsds", "/root/project/src/server",
            "/root/project/src/proposer", "/root/project/experiments",
        }
    else:
        assert {item["remote"] for item in record["directories"]} == {
            "/root/project/src", "/root/project/experiments",
        }
    container = tmp_path / "container"
    for item in record["directories"]:
        destination = container / item["remote"].lstrip("/")
        source_dir = Path(item["local"])
        matcher = FilePatternMatcher(*item.get("ignore", []))
        for path in source_dir.rglob("*"):
            relative = path.relative_to(source_dir)
            if not path.is_file() or matcher(relative.as_posix()):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    project = container / "root/project"
    for name in MANIFESTS:
        assert (project / "experiments" / name).read_bytes() == (ROOT / "experiments" / name).read_bytes()
    result = subprocess.run(
        [sys.executable, "-I", "-c", IMPORTS, str(project)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(json.dumps(record))
    print(result.stdout)


def test_incomplete_installed_support_fails_before_cloud_registration(installed_tree, tmp_path):
    import os
    import shutil

    incomplete = tmp_path / "incomplete"
    shutil.copytree(installed_tree, incomplete)
    (incomplete / "experiments" / MANIFESTS[0]).unlink()
    env = dict(os.environ, FLEXSDS_SERVING_CONFIG='{"enable_adaptive": true}')
    env.pop("FLEXSDS_SERVING_CONFIG_B64", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", CONSTRUCT, str(incomplete), str(tmp_path / "mounts.json")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "Required adaptive runtime support file is missing: " + MANIFESTS[0] in result.stderr
