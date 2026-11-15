"""Offline fixtures exercise actual JSON, checkpoint headers and mapping bytes."""
import json
import struct
from pathlib import Path

import pytest

from flexsds.core.compatibility import check_compatibility, evaluate_serving_preflight
from flexsds.core.model_inspection import ModelFiles, InspectionError


def write_checkpoint(path, tensors):
    header, payload = {}, b''
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(raw)])
        payload += raw
    encoded = json.dumps(header).encode()
    (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def write_config(path, **updates):
    cfg = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    cfg.update(updates)
    (path / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')


@pytest.fixture
def models(tmp_path):
    target, draft = tmp_path / 'target', tmp_path / 'draft'
    target.mkdir(); draft.mkdir()
    cfg = dict(architectures=['Qwen3ForCausalLM'], hidden_size=4, vocab_size=4,
               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=2)
    (target / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    cfg.update(architectures=['LlamaForCausalLMEagle3'], num_hidden_layers=1,
               draft_vocab_size=2, target_model_name_or_path=str(target))
    (draft / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    tokenizer = dict(model=dict(type='BPE', vocab={'a': 0, 'b': 1, 'c': 2, 'd': 3}, merges=[]), added_tokens=[])
    (target / 'tokenizer.json').write_text(json.dumps(tokenizer), encoding='utf-8')
    tensors = {
        'fc.weight': ('BF16', [4, 12], b'\0' * 96),
        'd2t': ('I64', [2], struct.pack('<qq', 0, 2)),  # target IDs [0, 3]
        't2d': ('BOOL', [4], bytes([1, 0, 0, 1])),
    }
    write_checkpoint(target, {'weight': ('BF16', [4], b'\0' * 8)})
    write_checkpoint(draft, tensors)
    return target, draft, tensors


def check(models, **kwargs):
    return check_compatibility(str(models[0]), str(models[1]), **kwargs)


def test_eagle3_native_mapping_and_inherited_tokenizer(models):
    report = check(models)
    assert report.is_compatible, report.warnings
    assert report.eagle_version == 'EAGLE-3'
    assert report.draft_method == 'eagle3'
    assert 'Reduced 2/4' in report.vocabulary_status
    assert 'd2t verified' in report.vocabulary_status
    assert not report.use_heterogeneous_vocab  # native mapping is handled by EAGLE loader
    assert report.draft_vram_est_mb == (models[1] / 'model.safetensors').stat().st_size / 1024**2
    assert report.kv_cache_min_mb == 2 * 3 * 1 * 2 * 2 * 32 * 4096 / 1024**2


def test_eagle31_requires_config_and_norm_weights(models):
    target, draft, tensors = models
    write_config(draft, fc_norm=True, norm_output=True, use_post_norm_feedback=True)
    assert not check(models).is_compatible
    for i in range(3):
        tensors[f'fc_norm.{i}.weight'] = ('BF16', [4], b'\0' * 8)
    write_checkpoint(draft, tensors)
    assert check(models).eagle_version == 'EAGLE-3.1'
    assert check(models).is_compatible


def test_eagle1_and_2_share_supported_legacy_loader(models):
    target, draft, tensors = models
    tensors['fc.weight'] = ('BF16', [4, 8], b'\0' * 64)
    write_checkpoint(draft, tensors)
    write_config(draft, architectures=['LlamaForCausalLM'], draft_vocab_size=4)
    report = check(models)
    assert report.is_compatible, report.warnings
    assert report.method_label == 'EAGLE (version selection required)'
    assert report.draft_method == 'eagle'
    assert report.method_supported
    for version in ['1', '2']:
        write_config(draft, eagle_version=version)
        report = check(models)
        assert report.is_compatible, report.warnings
        assert report.method_label == f'EAGLE-{version}'
        assert report.draft_method == 'eagle'


def test_names_do_not_make_regular_models_eagle(models, monkeypatch):
    target, draft, _ = models
    write_config(draft, architectures=['Qwen3ForCausalLM'], num_hidden_layers=28)
    cfg = json.loads((draft / 'config.json').read_text())
    del cfg['draft_vocab_size']
    (draft / 'config.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(ModelFiles, 'tensor_headers', lambda self: pytest.fail('Regular draft should be rejected before checkpoint inspection'))
    report = check_compatibility(str(target), str(draft))
    assert report.eagle_version == 'Non-EAGLE'
    assert not report.is_compatible
    assert 'Not supported yet' in report.warnings[0]
    assert 'Not evaluated' in report.summary()


def test_missing_metadata_never_falls_back_to_repository_name(monkeypatch):
    monkeypatch.setattr(ModelFiles, '__init__', lambda *a, **kw: (_ for _ in ()).throw(InspectionError('offline')))
    report = check_compatibility('target', 'organization/EAGLE3-speculator')
    assert not report.is_compatible
    assert report.eagle_version == 'Unknown'
    assert 'Cannot verify' in report.warnings[0]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'out_of_range', 'mask'])
def test_bad_native_mapping_blocks_launch(models, mutation):
    target, draft, tensors = models
    if mutation == 'missing':
        del tensors['d2t']
    elif mutation == 'duplicate':
        tensors['d2t'] = ('I64', [2], struct.pack('<qq', 0, -1))
    elif mutation == 'out_of_range':
        tensors['d2t'] = ('I64', [2], struct.pack('<qq', 0, 4))
    else:
        tensors['t2d'] = ('BOOL', [4], bytes([1, 1, 0, 0]))
    write_checkpoint(draft, tensors)
    report = check(models)
    assert not report.is_compatible
    assert any('mapping' in message or 'd2t' in message for message in report.warnings)


def test_equal_vocab_size_does_not_prove_same_token_ids(models):
    target, draft, _ = models
    tokenizer = json.loads((target / 'tokenizer.json').read_text())
    tokenizer['model']['vocab'].update(a=1, b=0)
    (draft / 'tokenizer.json').write_text(json.dumps(tokenizer))
    report = check(models)
    assert not report.is_compatible
    assert 'Mismatch' in report.vocabulary_status
    assert any('Equal vocab_size' in w for w in report.warnings)


def test_different_input_vocabulary_is_not_silently_aligned(models):
    write_config(models[1], vocab_size=5)
    report = check(models)
    assert not report.is_compatible
    assert not report.use_heterogeneous_vocab
    assert 'different target/draft input vocabularies' in report.warnings[0]


def test_shared_full_vocab_and_unknown_pairing(models):
    write_config(models[1], draft_vocab_size=4)
    report = check(models)
    assert report.is_compatible
    assert report.vocabulary_status.startswith('Shared')
    write_config(models[1], target_model_name_or_path='wrong/target')
    assert not check(models).is_compatible


def test_kv_estimate_scales_with_configured_capacity(models):
    a = check(models, max_num_seqs=1, max_model_len=1024)
    b = check(models, max_num_seqs=2, max_model_len=2048)
    assert b.kv_cache_min_mb == 4 * a.kv_cache_min_mb
    assert a.target_vram_est_mb == b.target_vram_est_mb
    assert check(models, free_vram_mb=1).oom_risk


def test_one_preflight_estimate_and_specific_block_reason(models, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('flexsds.core.compatibility.pre_download_guard', lambda *a, **k: pytest.fail('Duplicate heuristic guard'))
    result = evaluate_serving_preflight(str(models[0]), str(models[1]), SimpleNamespace(), 'modal', 'A10G')
    assert result.is_safe
    assert result.download.vram_needed_mb == result.report.total_vram_needed_mb
    write_config(models[1], target_model_name_or_path='wrong/model')
    result = evaluate_serving_preflight(str(models[0]), str(models[1]), SimpleNamespace(), 'modal', 'A10G')
    assert not result.is_safe
    assert 'trained target/draft pairing' in result.reasons[0]


def test_oversized_header_rejected_without_reading_weights(models):
    path = models[1] / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', 10**9))
    report = check(models)
    assert not report.is_compatible
    assert 'header size' in report.warnings[0]


def test_http_requires_honored_range(monkeypatch):
    from unittest.mock import MagicMock
    response = MagicMock(status=200)
    response.headers = {}
    response.__enter__.return_value = response
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: response)
    with pytest.raises(InspectionError, match='bounded checkpoint range'):
        ModelFiles._http('https://huggingface.co/test', 8, (0, 7))
    response.read.assert_not_called()


def test_speculators_nested_config_is_inspected(models):
    target, draft, _ = models
    cfg = json.loads((draft / 'config.json').read_text())
    cfg.pop('target_model_name_or_path')
    nested = dict(architectures=['Eagle3Speculator'], draft_vocab_size=2, target_hidden_size=None,
                  transformer_layer_config=cfg,
                  speculators_config=dict(algorithm='eagle3', verifier=dict(name_or_path=str(target))))
    (draft / 'config.json').write_text(json.dumps(nested))
    assert check(models).is_compatible
    assert check(models).eagle_version == 'EAGLE-3'


def test_inspected_revisions_reach_static_command_and_modal_environment():
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
    cfg = dict(enable_adaptive=False, draft_method='eagle3', target_revision='a'*40, draft_revision='b'*40)
    cmd = LocalServingRunner().build_command(cfg)
    assert cmd[cmd.index('--revision') + 1] == 'a'*40
    spec = json.loads(cmd[cmd.index('--speculative-config') + 1])
    assert spec['revision'] == 'b'*40
    assert 'use_heterogeneous_vocab' not in spec
    forwarded = json.loads(ModalServingRunner().setup_environment(cfg)['FLEXSDS_SERVING_CONFIG'])
    assert forwarded['target_revision'] == 'a'*40
    assert forwarded['draft_revision'] == 'b'*40


def test_profile_cannot_silently_override_inspected_revision(tmp_path):
    from flexsds.server.local_runner import LocalServingRunner
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(dict(models=dict(target_revision='a'*40, draft_revision='b'*40))))
    cfg = dict(enable_adaptive=True, profile_path=str(path), target_revision='c'*40, draft_revision='b'*40)
    with pytest.raises(ValueError, match='Inspected target revision'):
        LocalServingRunner().build_command(cfg)
    cfg.update(target_revision='a'*40, draft_revision='c'*40)
    with pytest.raises(ValueError, match='Inspected draft revision'):
        LocalServingRunner().build_command(cfg)


def test_non_eagle_dry_run_returns_supported_error_before_runner(models, monkeypatch, capsys):
    from flexsds.cli.main import main
    from flexsds.server.modal_runner import ModalServingRunner
    target, draft, _ = models
    write_config(draft, architectures=['Qwen3ForCausalLM'], num_hidden_layers=28)
    cfg = json.loads((draft / 'config.json').read_text())
    del cfg['draft_vocab_size']
    (draft / 'config.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(ModalServingRunner, 'launch', lambda *a, **kw: pytest.fail('Unsupported draft reached runner'))
    assert main(['serve', '--backend', 'modal', '--target', str(target), '--draft', str(draft),
                 '--no-adaptive', '--dry-run']) == 2
    assert 'Not supported yet: Independent draft model' in capsys.readouterr().out


def test_wizard_displays_and_saves_exact_selected_legacy_version(models, tmp_path, monkeypatch, capsys):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    target, draft, tensors = models
    tensors['fc.weight'] = ('BF16', [4, 8], b'\0' * 64)
    write_checkpoint(draft, tensors)
    write_config(draft, architectures=['LlamaForCausalLM'], draft_vocab_size=4)
    monkeypatch.chdir(tmp_path)
    def choose(question, choices, **kwargs):
        if 'Model configuration' in question:
            return next(value for value in choices if 'Custom' in value)
        if 'Serving platform' in question:
            return next(value for value in choices if 'Local vLLM' in value)
        if 'EAGLE version' in question:
            return 'EAGLE-2'
        return choices[0]
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_string', lambda question, **kwargs: str(target if 'Target' in question else draft))
    monkeypatch.setattr(wizard, 'prompt_bool', lambda *a, **kwargs: False)
    wizard.run_interactive_wizard(HardwareInfo('GPU','sm_89',23000,23000,'',True))
    output = capsys.readouterr().out
    assert 'Speculation: EAGLE-2' in output
    assert 'version selection required' not in output
    saved = json.loads((tmp_path / 'serving_config.json').read_text())
    assert saved['eagle_version'] == '2'
    assert saved['draft_method'] == 'eagle'
