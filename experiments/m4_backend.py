"""Execution identity only; scientific configuration is owned by m4_treatments."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def execution_identity() -> dict:
    backend = os.environ.get("M4_EXECUTION_BACKEND", "google_colab")
    if backend == "google_colab":
        if not Path("/content").is_dir() or (
            "google.colab" not in sys.modules and not os.environ.get("COLAB_RELEASE_TAG")
        ):
            raise ValueError("Canonical runtime must execute in Google Colab")
        return dict(execution_backend=backend)
    if backend == "modal":
        import modal
        task = os.environ.get("MODAL_TASK_ID", "")
        if modal.is_local() or not valid_backend_evidence(dict(execution_backend=backend, modal_task_id=task)):
            raise ValueError("Modal runtime requires an actual remote Modal task")
        return dict(execution_backend=backend, modal_task_id=task)
    raise ValueError("Unapproved M4 execution backend")


def valid_backend_evidence(environment: dict) -> bool:
    backend = environment.get("execution_backend")
    if backend == "google_colab":
        return True
    return (backend == "modal"
            and isinstance(environment.get("modal_task_id"), str)
            and environment["modal_task_id"].startswith("ta-")
            and len(environment["modal_task_id"]) > 3)
