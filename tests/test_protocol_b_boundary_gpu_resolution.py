"""CPU-only physical GPU selector and control preparation contracts."""
import json
import subprocess
from unittest.mock import patch

import pytest

from experiments import protocol_b_boundary as boundary
from experiments import protocol_b_capture as capture
from experiments.protocol_b import CONFIG_ENV, EvidenceError

GPU0 = "GPU-1b73acfd-9054-ffaf-70fe-79319670b86f"
GPU1 = "GPU-b306be9c-78ed-b3de-9c67-33130c066b7e"
INVENTORY = f"0, {GPU0}\n1, {GPU1}"
BOOT = "52fce488-54fb-4de1-9e13-370c5070de8d"


@pytest.mark.parametrize("selector,expected", [("0", GPU0), ("1", GPU1), (GPU0, GPU0), (GPU1, GPU1)])
def test_resolve(selector, expected):
    assert capture.resolve_gpu_selector(selector, INVENTORY) == expected


@pytest.mark.parametrize("selector", [None, "", " ", "2", "-1", "1.0", "garbage", "0,1",
    "0,", " 0", "GPU-", "GPU-1b73", "GPU-b306be9c-78ed-b3de-9c67-33130c066b7f"])
def test_invalid_selector(selector):
    with pytest.raises(EvidenceError):
        capture.resolve_gpu_selector(selector, INVENTORY)


def test_ambiguous_prefix_rejected():
    with pytest.raises(EvidenceError):
        capture.resolve_gpu_selector("GPU-1b73", INVENTORY.replace(GPU1, GPU0[:-1] + "e"))


@pytest.mark.parametrize("inventory", ["", f"0, {GPU0}\n0, {GPU1}", f"0, {GPU0}\n1, {GPU0}",
    f"1, {GPU0}", "bad"])
def test_invalid_inventory(inventory):
    with pytest.raises(EvidenceError):
        capture.resolve_gpu_selector("0", inventory)


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(1, "nvidia-smi"),
    subprocess.TimeoutExpired("nvidia-smi", 30), FileNotFoundError("nvidia-smi")])
def test_command_failure(error):
    with patch.dict(capture.os.environ, CUDA_VISIBLE_DEVICES="0"), \
            patch.object(capture.Path, "read_text", return_value=BOOT), \
            patch.object(capture, "command", side_effect=error), pytest.raises(EvidenceError):
        capture.control_gpu_selection(GPU0, BOOT)


def test_matching_session_and_provenance():
    with patch.dict(capture.os.environ, CUDA_VISIBLE_DEVICES="0"), \
            patch.object(capture.Path, "read_text", return_value=BOOT), \
            patch.object(capture, "command", return_value=INVENTORY) as command:
        selection = capture.control_gpu_selection(GPU0, BOOT)
    assert selection == dict(raw_selector="0", resolved_gpu_uuid=GPU0, expected_gpu_uuid=GPU0,
                             inventory=INVENTORY, boot_id=BOOT)
    command.assert_called_once_with(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])


@pytest.mark.parametrize("selector,boot", [("1", BOOT), ("0", "wrong-boot")])
def test_gpu_or_boot_mismatch(selector, boot):
    with patch.dict(capture.os.environ, CUDA_VISIBLE_DEVICES=selector), \
            patch.object(capture.Path, "read_text", return_value=boot), \
            patch.object(capture, "command", return_value=INVENTORY), pytest.raises(EvidenceError):
        capture.control_gpu_selection(GPU0, BOOT)


def test_prepare_records_selection(tmp_path):
    settings = dict.fromkeys(CONFIG_ENV, "fixture")
    settings.update(CUDA_VISIBLE_DEVICES=GPU0)
    records = [dict(prompt_token_ids=[i], token_ids=[7, 8]) for i in range(46)]
    plan = dict(prompts=[dict(request={}, request_sha256="a"*64)] * 46)
    baseline = dict(runs=dict(adaptive=dict(records=records, environment=dict(gpu_uuid=GPU0,
        boot_id=BOOT, config=dict(environment=settings), source_hashes={}, launcher_hashes={}))),
        manifest=dict(plan=plan), identity=dict(archive_sha256="b"*64), normal_producer_budget=3)
    with patch.dict(capture.os.environ, {**settings, "CUDA_VISIBLE_DEVICES": "0", "M203_P177_PRODUCER_K0": "1"}), \
            patch.object(capture.Path, "read_text", return_value=BOOT), \
            patch.object(capture, "command", return_value=INVENTORY), \
            patch.object(boundary, "command", return_value=""), \
            patch.object(boundary, "control_baseline", return_value=baseline), \
            patch.object(boundary, "control_scheduler_auth", return_value={}), \
            patch.object(boundary, "load_plan", return_value=plan):
        boundary.prepare_control(tmp_path / "run", "session", tmp_path / "sealed.tar.gz")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["gpu_selection"]["raw_selector"] == "0"
    assert manifest["gpu_selection"]["resolved_gpu_uuid"] == manifest["gpu_selection"]["expected_gpu_uuid"] == GPU0


def test_capture_reauthenticates_selection_before_runtime_collection():
    selection = dict(raw_selector="0", resolved_gpu_uuid=GPU0, expected_gpu_uuid=GPU0,
                     inventory=INVENTORY, boot_id=BOOT)
    settings = dict.fromkeys(CONFIG_ENV, "fixture")
    settings.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1")
    with patch.dict(capture.os.environ, settings), patch.object(capture, "command", return_value=""), \
            patch.object(capture, "control_gpu_selection", return_value={}) as resolve, \
            pytest.raises(EvidenceError, match="selection changed"):
        capture.runtime({}, "session", dict(environment=settings), gpu_selection=selection)
    resolve.assert_called_once_with(GPU0, BOOT)
