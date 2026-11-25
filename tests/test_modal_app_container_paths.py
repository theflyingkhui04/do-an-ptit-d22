"""Tests for safe container hydration and path resolution in modal_app."""
from pathlib import Path
import base64
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def test_modal_app_safe_path_resolution_container_simulation():
    """Kiểm tra logic nhận diện đường dẫn an toàn khi chạy trong container Modal (/root/modal_app.py)."""
    shallow_file = Path("/root/modal_app.py")
    parents = shallow_file.parents
    assert len(parents) == 2  # [Path('/root'), Path('/')]

    # Hàm logic phân giải đường dẫn cần test
    def resolve_paths(file_path: Path):
        resolved = file_path.resolve()
        if len(resolved.parents) > 2 and (resolved.parents[2] / "flexsds").exists():
            pkg_root = str(resolved.parents[2])
            project_root = resolved.parents[3]
        else:
            pkg_root = "/root/project/src"
            project_root = Path("/root/project")
        return pkg_root, project_root

    pkg_root, project_root = resolve_paths(shallow_file)
    assert pkg_root == "/root/project/src"
    assert project_root == Path("/root/project")


def test_modal_app_host_import():
    """Kiểm tra import trực tiếp flexsds.server.modal_app trên host."""
    import flexsds.server.modal_app as modal_app
    assert modal_app._pkg_root is not None
    assert modal_app.PROJECT_ROOT is not None
    assert str(modal_app.PROJECT_ROOT).endswith("do-an-ptit-d22") or str(modal_app.PROJECT_ROOT) == "/root/project"


@pytest.mark.parametrize("profile_name", [
    r"C:\Users\khoi.can\AppData\Local\flexsds\profiles\profile.json",
    'profile có dấu.json',
])
def test_image_config_survives_windows_paths_and_container_import(tmp_path, monkeypatch, profile_name):
    """Execute both host construction and container hydration without cloud calls."""
    from flexsds.server import modal_image
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / profile_name
    profile.write_text('{}', encoding='utf-8')
    config = dict(target_gpu='A10', profile_path=str(profile),
                  target_model='model có dấu', enable_adaptive=True)
    monkeypatch.setenv('FLEXSDS_SERVING_CONFIG', json.dumps(config))
    monkeypatch.delenv('FLEXSDS_SERVING_CONFIG_B64', raising=False)
    image = Mock()
    image.env.return_value = image
    image.add_local_dir.return_value = image
    image.add_local_file.return_value = image
    app = Mock()
    app.function.return_value = lambda fn: fn
    local = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(
        App=Mock(return_value=app), is_local=local,
        Volume=SimpleNamespace(from_name=Mock()),
        web_server=lambda *a, **kw: lambda fn: fn,
    ))
    monkeypatch.setattr(modal_image, 'serving_image', lambda: image)
    source = Path(modal_image.__file__).with_name('modal_app.py')
    host = runpy.run_path(str(source))
    assert host['SERVING_CONFIG'] == config
    image.add_local_file.assert_called_once_with(
        str(profile), remote_path='/root/project/hardware_profile.json')
    env = image.env.call_args.args[0]
    assert 'FLEXSDS_SERVING_CONFIG' not in env
    encoded = env['FLEXSDS_SERVING_CONFIG_B64']
    assert not any(char in encoded for char in ('\\', '"', "'", '\n'))
    remote = json.loads(base64.b64decode(encoded))
    assert remote == dict(config, profile_path='/root/project/hardware_profile.json')
    monkeypatch.setenv('FLEXSDS_SERVING_CONFIG_B64', encoded)
    # The encoded image value must take precedence over any legacy raw value.
    monkeypatch.setenv('FLEXSDS_SERVING_CONFIG', r'{"profile_path":"C:\Users\broken"}')
    local.return_value = False
    container = runpy.run_path(str(source))
    assert container['SERVING_CONFIG'] == remote
