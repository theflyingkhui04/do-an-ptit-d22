"""CPU tests of backend identity; no runtime qualification claims."""
import sys
from types import SimpleNamespace

import pytest

from experiments.m4_backend import execution_identity, valid_backend_evidence


@pytest.mark.parametrize("local,task", [(True, "ta-fixture"), (False, ""), (False, "bad"), (False, "ta-")])
def test_modal_requires_remote_task(monkeypatch, local, task):
    monkeypatch.setenv("M4_EXECUTION_BACKEND", "modal")
    monkeypatch.setenv("MODAL_TASK_ID", task)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(is_local=lambda: local))
    with pytest.raises(ValueError):
        execution_identity()


def test_modal_identity_does_not_change_scientific_config(monkeypatch):
    from experiments.m4_treatments import TREATMENTS, engine_kwargs
    before = {name: engine_kwargs(name) for name in TREATMENTS}
    monkeypatch.setenv("M4_EXECUTION_BACKEND", "modal")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-synthetic-fixture")
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(is_local=lambda: False))
    identity = execution_identity()
    assert valid_backend_evidence(identity)
    assert identity == dict(execution_backend="modal", modal_task_id="ta-synthetic-fixture")
    assert before == {name: engine_kwargs(name) for name in TREATMENTS}


@pytest.mark.parametrize("record", [{}, {"execution_backend": "other"},
    {"execution_backend": "modal"}, {"execution_backend": "modal", "modal_task_id": "ta-"}])
def test_missing_or_unapproved_backend_fails(record):
    assert not valid_backend_evidence(record)


def test_colab_evidence_remains_accepted():
    assert valid_backend_evidence(dict(execution_backend="google_colab"))
