"""Calibration must preserve worker evidence, never synthesize empirical costs."""
import copy
import json
import pytest
from flexsds.core.hardware import HardwareInfo
from flexsds.core import calibration as cal
from flexsds.core import cost_model as cm


def observed_groups(config, batches):
    groups = []
    for k in cal.KS:
        records = [dict(k=k, batch=b, context=cal.CONTEXT, repeat=r, warmup=r < cal.WARMUP,
                        samples=dict(target=10.+k+b, **({'draft': 2.*k} if k else {})),
                        shapes=[dict(batch=b, query=[k+1]*b, context=[cal.CONTEXT]*b)])
                   for b in batches for r in range(cal.WARMUP+cal.REPEATS)]
        groups.append(dict(device='NVIDIA A10G', sm_architecture='sm_86', records=records,
                           vllm_version='0.30.0', target_revision='a'*40, draft_revision='b'*40))
    return groups


@pytest.fixture
def measured_profile():
    config = dict(target_model='target', draft_model='draft', draft_method='draft_model',
                  sm_architecture='sm_86', target_gpu='A10G')
    return cal.build_profile(config, observed_groups(config, cal.BATCHES), cal.BATCHES)


def test_cpu_cannot_create_empirical_profile(tmp_path, monkeypatch):
    monkeypatch.setattr('flexsds.core.compatibility.check_compatibility', lambda *a: type('R', (), {'draft_method': 'draft_model'})())
    cpu = HardwareInfo('CPU', 'cpu', 0, 0, 'N/A', False)
    with pytest.raises(RuntimeError, match='real CUDA'):
        cm.run_hardware_micro_benchmark(cpu, target_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_remote_gpu_cannot_be_calibrated_on_laptop(tmp_path):
    hw = HardwareInfo('Remote A10G', 'sm_86', 0, 0, 'N/A', True)
    with pytest.raises(ValueError, match='backend'):
        cm.run_hardware_micro_benchmark(hw, target_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_modal_dispatch_and_roundtrip(tmp_path, monkeypatch):
    from flexsds.server import modal_calibration
    monkeypatch.setattr('flexsds.server.modal_runner.ModalServingRunner.check_credentials', lambda self: True)
    monkeypatch.setattr('flexsds.core.compatibility.check_compatibility', lambda *a: type('R', (), {'draft_method': 'draft_model'})())
    calls = []
    def remote(config, batches):
        calls.append((config, batches))
        return observed_groups(config, batches)
    monkeypatch.setattr(modal_calibration, 'calibrate', remote)
    hw = HardwareInfo('Remote A10G', 'sm_86', 0, 0, 'N/A', True)
    model, path = cm.run_hardware_micro_benchmark(hw, 'target', 'draft', target_dir=tmp_path, backend='modal')
    assert calls[0][0]['target_gpu'] == 'A10G'
    assert model.matrix[(2, 1)] == 17.0
    cm.validate_measured_profile(json.loads(path.read_text()), hw, 'target', 'draft')


@pytest.mark.parametrize('damage', ['sample', 'matrix', 'missing', 'duplicate', 'shape', 'hash', 'model'])
def test_corrupt_or_incomplete_profile_rejected(measured_profile, damage):
    data = copy.deepcopy(measured_profile)
    group = data['evidence']['groups'][1]
    if damage == 'sample': group['records'][5]['samples']['target'] = float('nan')
    if damage == 'matrix': data['cost_matrix_ms']['k1_b1'] = 999
    if damage == 'missing': group['records'].pop()
    if damage == 'duplicate': group['records'].append(group['records'][0])
    if damage == 'shape': group['records'][0]['shapes'] = []
    if damage == 'hash': data['metadata']['raw_sha256'] = 'bad'
    if damage == 'model': data['models']['target'] = 'other'
    with pytest.raises(ValueError): cm.validate_measured_profile(data)


def test_failed_modal_calibration_does_not_write_profile(tmp_path, monkeypatch):
    monkeypatch.setattr('flexsds.server.modal_runner.ModalServingRunner.check_credentials', lambda self: True)
    monkeypatch.setattr('flexsds.core.compatibility.check_compatibility', lambda *a: type('R', (), {'draft_method': 'draft_model'})())
    def fail(*args): raise RuntimeError('GPU OOM')
    monkeypatch.setattr('flexsds.server.modal_calibration.calibrate', fail)
    with pytest.raises(RuntimeError, match='OOM'):
        cm.run_hardware_micro_benchmark(HardwareInfo('Remote A10G','sm_86',0,0,'',True), target_dir=tmp_path, backend='modal')
    assert not list(tmp_path.iterdir())


def test_modal_dispatch_uses_cli_when_sdk_is_in_another_pipx(monkeypatch):
    from flexsds.server.modal_calibration import calibrate
    seen = []
    def run(command, *, env, **kwargs):
        seen.append(command)
        request = json.loads(env['FLEXSDS_CALIBRATION_REQUEST'])
        assert request['config']['target_gpu'] == 'A10G'
        from pathlib import Path
        Path(env['FLEXSDS_CALIBRATION_OUTPUT']).write_text('[{"measured":true}]')
    monkeypatch.setattr('flexsds.server.modal_calibration.subprocess.run', run)
    assert calibrate({'target_gpu':'A10G'}, [1]) == [{'measured':True}]
    assert seen[0][1] == 'run'
    assert seen[0][2].endswith('modal_calibration_app.py')
