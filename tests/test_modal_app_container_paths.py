"""Tests for safe container hydration and path resolution in modal_app."""
from pathlib import Path
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
