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
    relative = Path(profile_name)
    if relative.is_absolute():  # Windows: keep the drive-style path inside tmp_path
        relative = Path(*relative.parts[1:])
    profile = tmp_path / relative
    profile.parent.mkdir(parents=True, exist_ok=True)
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
        concurrent=lambda *a, **kw: lambda fn: fn,
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


def _decorators_applied_to_serve(monkeypatch, serving_config):
    """Load modal_app.py with a recording fake `modal`; return decorator names, innermost first."""
    import json as _json
    import runpy
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from flexsds.server import modal_image

    order: list[tuple[str, dict]] = []

    def recorder(name):
        def factory(*args, **kwargs):
            def apply(fn):
                order.append((name, kwargs))
                return fn
            return apply
        return factory

    image = Mock()
    image.env.return_value = image
    image.add_local_dir.return_value = image
    image.add_local_file.return_value = image
    app = Mock()
    app.function.side_effect = recorder("function")
    monkeypatch.setenv("FLEXSDS_SERVING_CONFIG", _json.dumps(serving_config))
    monkeypatch.delenv("FLEXSDS_SERVING_CONFIG_B64", raising=False)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        App=Mock(return_value=app), is_local=Mock(return_value=False),
        Volume=SimpleNamespace(from_name=Mock()),
        web_server=recorder("web_server"), concurrent=recorder("concurrent")))
    monkeypatch.setattr(modal_image, "serving_image", lambda: image)
    runpy.run_path(str(modal_image.__file__).replace("modal_image.py", "modal_app.py"))
    return order


def test_modal_serve_accepts_concurrent_inputs(monkeypatch):
    """Without @modal.concurrent Modal hands a web_server container one request at a time (25 s vs 4.4 s, 26/11)."""
    order = _decorators_applied_to_serve(monkeypatch, {"target_gpu": "A10", "enable_adaptive": False})
    assert [name for name, _ in order] == ["web_server", "concurrent", "function"]
    assert dict(order)["concurrent"]["max_inputs"] == 64

    order = _decorators_applied_to_serve(
        monkeypatch, {"target_gpu": "A10", "enable_adaptive": False, "max_num_seqs": 48})
    assert dict(order)["concurrent"]["max_inputs"] == 96


def test_modal_serve_keeps_the_gpu_awake_for_the_configured_window(monkeypatch):
    """Modal's default ~60 s scale-down makes the next demo call a cold start (~4.5 min > Modal's 150 s web limit)."""
    order = _decorators_applied_to_serve(monkeypatch, {"target_gpu": "A10", "enable_adaptive": False,
                                                       "scaledown_window": 600})
    assert dict(order)["function"]["scaledown_window"] == 600
    default = _decorators_applied_to_serve(monkeypatch, {"target_gpu": "A10", "enable_adaptive": False})
    assert dict(default)["function"]["scaledown_window"] == 300
