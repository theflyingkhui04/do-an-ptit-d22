"""CLI paths of the workload tools stay inside the repository or the system temp directory."""
import os
from pathlib import Path

import pytest

from src.pipeline.safe_paths import REPO_ROOT, contained_path

ELSEWHERE = Path(os.path.abspath(os.sep)) / "ptit-not-allowed" / "report.json"


def test_relative_paths_resolve_against_the_repository():
    assert contained_path("experiments/configs/w1") == Path(os.path.realpath(REPO_ROOT / "experiments/configs/w1"))


def test_the_temp_directory_is_allowed(tmp_path):
    assert contained_path(tmp_path / "out") == Path(os.path.realpath(tmp_path / "out"))


@pytest.mark.parametrize("value", ["../outside.json", "experiments/../../outside.json"])
def test_parent_traversal_out_of_the_repository_is_refused(value):
    with pytest.raises(ValueError, match="outside the repository"):
        contained_path(value)


def test_an_absolute_path_elsewhere_is_refused():
    with pytest.raises(ValueError, match="outside the repository"):
        contained_path(ELSEWHERE)
