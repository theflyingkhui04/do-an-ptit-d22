"""Account clearing and post-readiness actions use temporary files and fake endpoints."""
import os
from pathlib import Path
import shutil
from unittest.mock import Mock
import pytest
from flexsds.core.credentials import CredentialsManager
from flexsds.server import modal_auth
from flexsds.cli import serving_session


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv('MODAL_CONFIG_PATH', str(tmp_path/'modal.toml'))
    monkeypatch.delenv('MODAL_PROFILE', raising=False)
    monkeypatch.delenv('MODAL_TOKEN_ID', raising=False)
    monkeypatch.delenv('MODAL_TOKEN_SECRET', raising=False)
    return CredentialsManager(str(tmp_path/'flexsds'/'credentials'))


def test_clear_blocks_inherited_tokens_but_new_login_restores_access(manager, monkeypatch):
    manager.set_credential('huggingface', 'HF_TOKEN', 'keep-me')
    manager.set_credential('modal', 'MODAL_TOKEN_ID', 'old-id')
    manager.set_credential('modal', 'MODAL_TOKEN_SECRET', 'old-secret')
    modal_auth.clear_modal(manager)
    monkeypatch.setenv('MODAL_TOKEN_ID', 'inherited-old-id')
    monkeypatch.setenv('MODAL_TOKEN_SECRET', 'inherited-old-secret')
    assert not modal_auth.prepare_modal_auth(manager)
    assert 'MODAL_TOKEN_ID' not in os.environ
    assert manager.get_stored_credential('huggingface', 'HF_TOKEN') == 'keep-me'
    manager.set_credential('modal', 'MODAL_TOKEN_ID', 'new-id')
    manager.set_credential('modal', 'MODAL_TOKEN_SECRET', 'new-secret')
    monkeypatch.setenv('MODAL_TOKEN_ID', 'inherited-old-id')
    assert modal_auth.prepare_modal_auth(manager)
    assert os.environ['MODAL_TOKEN_ID'] == 'new-id'


@pytest.mark.skipif(not shutil.which('modal'), reason='requires installed Modal CLI (no network)')
def test_native_clear_preserves_other_profiles_and_requires_pair(manager):
    path = Path(os.environ['MODAL_CONFIG_PATH'])
    path.write_text('[current]\nactive = true\ntoken_id = "id"\ntoken_secret = "secret"\nworkspace = "keep"\n'
                    '[other]\ntoken_id = "other-id"\ntoken_secret = "other-secret"\n')
    assert modal_auth.native_profile()['configured']
    modal_auth.clear_modal(manager)
    assert not modal_auth.native_profile()['configured']
    content = path.read_text()
    assert 'other-id' in content and 'other-secret' in content and 'keep' in content
    assert '"secret"' not in content
    if os.name != 'nt': assert path.stat().st_mode & 0o077 == 0


def session(output, proc=None, deploy=False, config=None):
    proc = proc or Mock()
    proc.poll.return_value = None
    serving_session.present_session(proc, output, endpoint='https://example.modal.run/v1',
                                    model='served-model', config=config or {'target_gpu': 'A10'}, deploy=deploy)
    return proc


def test_watch_prints_chat_command_and_opens_temporary_logs(monkeypatch, capsys):
    from flexsds.cli import wizard
    monkeypatch.setattr(serving_session.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(serving_session.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr(wizard, 'prompt_choice', lambda *a, **k: 'Watch log')
    output = Mock()
    session(output)
    text = capsys.readouterr().out
    assert 'Endpoint LLM created:' in text
    assert 'flexsds chat --endpoint https://example.modal.run/v1 --model served-model' in text
    output.suspend.assert_called_once()
    output.resume.assert_called_once()
    output.toggle_details.assert_called_once()


def test_chat_returns_to_actions_without_stopping_server(monkeypatch):
    from flexsds.cli import wizard
    from flexsds.client import chat
    monkeypatch.setattr(serving_session.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(serving_session.sys.stdout, 'isatty', lambda: True)
    choices = iter(['Realtime chat with deployed LLM', 'Watch log'])
    monkeypatch.setattr(wizard, 'prompt_choice', lambda *a, **k: next(choices))
    run = Mock(return_value=0)
    monkeypatch.setattr(chat, 'run_interactive_chat', run)
    proc = session(Mock(), config={'target_gpu': 'A10', 'draft_method': 'eagle3', 'eagle_version': '3.1', 'enable_adaptive': True})
    run.assert_called_once_with(
        endpoint='https://example.modal.run/v1',
        model='served-model',
        speculation_label='EAGLE-3.1 (HardwareAdaptive)',
    )
    proc.terminate.assert_not_called()


def test_clear_cli_does_not_detect_hardware_or_touch_real_account(manager, monkeypatch):
    from flexsds.cli import main
    monkeypatch.setattr(main, 'CredentialsManager', lambda: manager)
    detect = Mock(side_effect=AssertionError('must not inspect hardware'))
    monkeypatch.setattr(main, 'detect_hardware', detect)
    assert main.main(['clear', 'modal']) == 0
    assert (manager.path.parent/'modal_logged_out').exists()


def test_native_profile_relogin_ignores_stale_shell_tokens(manager, monkeypatch):
    modal_auth.clear_modal(manager)
    monkeypatch.setenv('MODAL_TOKEN_ID', 'stale-id')
    monkeypatch.setenv('MODAL_TOKEN_SECRET', 'stale-secret')
    monkeypatch.setattr(modal_auth, 'native_profile', lambda: {'configured': True})
    assert modal_auth.prepare_modal_auth(manager)
    assert 'MODAL_TOKEN_ID' not in os.environ
    assert 'MODAL_TOKEN_SECRET' not in os.environ


def test_clear_failure_does_not_claim_success(manager, monkeypatch, capsys):
    from flexsds.cli import main
    def fail(*a):
        raise RuntimeError('Cannot access the active Modal profile')
    monkeypatch.setattr(main, 'CredentialsManager', lambda: manager)
    monkeypatch.setattr(modal_auth, 'native_profile', fail)
    assert main.main(['clear', 'modal']) == 1
    assert 'disconnect failed' in capsys.readouterr().out


@pytest.mark.parametrize(
    'draft_method,eagle_version,enable_adaptive,expected_label',
    [
        ('eagle', '1', False, 'EAGLE-1 (Static)'),
        ('eagle', '1', True, 'EAGLE-1 (HardwareAdaptive)'),
        ('eagle', '2', False, 'EAGLE-2 (Static)'),
        ('eagle', '2', True, 'EAGLE-2 (HardwareAdaptive)'),
        ('eagle3', '3', False, 'EAGLE-3 (Static)'),
        ('eagle3', '3', True, 'EAGLE-3 (HardwareAdaptive)'),
        ('eagle3', '3.1', False, 'EAGLE-3.1 (Static)'),
        ('eagle3', '3.1', True, 'EAGLE-3.1 (HardwareAdaptive)'),
    ],
)
def test_build_api_config_payload_and_labels_cover_all_eagle_versions(
    draft_method, eagle_version, enable_adaptive, expected_label
):
    cfg = {
        'backend': 'modal',
        'target_model': 'Qwen/Qwen2.5-7B-Instruct-AWQ',
        'draft_model': 'org/draft-head',
        'draft_method': draft_method,
        'eagle_version': eagle_version,
        'enable_adaptive': enable_adaptive,
        'target_gpu': 'A100',
        'max_num_seqs': 16,
        'max_model_len': 8192,
        'MODAL_TOKEN_ID': 'secret-id',
        'MODAL_TOKEN_SECRET': 'secret-val',
        'HF_TOKEN': 'hf-secret',
        'api_key': 'sk-secret',
    }
    payload = serving_session.build_api_config_payload(
        endpoint='https://demo--flexsds-serving-serve.modal.run/v1',
        model='qwen2.5-7b-adaptive',
        config=cfg,
        deploy=True,
    )
    assert payload['endpoint'] == 'https://demo--flexsds-serving-serve.modal.run/v1'
    assert payload['served_model'] == 'qwen2.5-7b-adaptive'
    assert payload['target_model'] == 'Qwen/Qwen2.5-7B-Instruct-AWQ'
    assert payload['draft_model'] == 'org/draft-head'
    assert payload['draft_method'] == draft_method
    assert payload['eagle_version'] == eagle_version
    assert payload['enable_adaptive'] is enable_adaptive
    assert payload['target_gpu'] == 'A100'
    assert payload['modal_deploy'] is True
    assert payload['hardware']['device_name'] == 'NVIDIA A100'
    assert payload['hardware']['vram_gb'] == 80
    assert payload['hardware']['tier'] == 'High'
    assert payload['hardware']['sm_architecture'] == 'sm_80'
    for secret in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'HF_TOKEN', 'api_key'):
        assert secret not in payload
    assert serving_session.format_speculation_label(cfg) == expected_label


def test_build_api_config_payload_handles_vanilla_mode():
    cfg = {
        'target_model': 'Qwen/Qwen2.5-7B-Instruct-AWQ',
        'draft_model': None,
        'draft_method': 'none',
        'eagle_version': None,
        'disable_speculative': True,
        'enable_adaptive': False,
        'target_gpu': 'L40S',
    }
    payload = serving_session.build_api_config_payload(
        endpoint='https://demo.modal.run/v1',
        model='qwen2.5-7b-vanilla',
        config=cfg,
        deploy=False,
    )
    assert payload['draft_model'] is None
    assert payload['draft_method'] == 'none'
    assert payload['eagle_version'] is None
    assert payload['enable_adaptive'] is False
    assert payload['hardware']['device_name'] == 'NVIDIA L40S'
    assert payload['hardware']['vram_gb'] == 48
    assert serving_session.format_speculation_label(cfg) == 'Standard (Vanilla)'


def test_export_api_config_json_action_writes_file_and_keeps_server_alive(tmp_path, monkeypatch, capsys):
    import json
    from flexsds.cli import wizard

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serving_session.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(serving_session.sys.stdout, 'isatty', lambda: True)
    seen_choices = []

    def choose(question, choices, **kwargs):
        seen_choices.append(list(choices))
        return choices[2] if len(seen_choices) == 1 else 'Watch log'

    out_path = tmp_path / 'exported' / 'web_api_config.json'
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_string', lambda q, default_val='': str(out_path))

    proc = session(
        Mock(),
        deploy=False,
        config={
            'backend': 'modal',
            'target_model': 'Qwen/Qwen2.5-7B-Instruct-AWQ',
            'draft_model': 'duckhoi/qwen2.5-7b-eagle3.1-speculator',
            'draft_method': 'eagle3',
            'eagle_version': '3.1',
            'enable_adaptive': True,
            'target_gpu': 'A10',
            'max_num_seqs': 32,
            'max_model_len': 4096,
        },
    )
    assert 'Export API config JSON (for web demo)' in seen_choices[0]
    assert out_path.is_file()
    saved = json.loads(out_path.read_text(encoding='utf-8'))
    assert saved['endpoint'] == 'https://example.modal.run/v1'
    assert saved['served_model'] == 'served-model'
    assert saved['eagle_version'] == '3.1'
    assert saved['target_gpu'] == 'A10'
    assert saved['hardware']['device_name'] == 'NVIDIA A10'
    text = capsys.readouterr().out
    assert str(out_path.resolve()) in text
    assert 'serving_config.json' in text
    proc.terminate.assert_not_called()


def test_chat_banner_does_not_hardcode_eagle31(monkeypatch, capsys):
    from flexsds.client.chat import InteractiveChatSession

    client = Mock()
    client.base_url = 'https://example.modal.run/v1'
    client.check_health.return_value = True
    monkeypatch.setattr('builtins.input', lambda _: '/exit')

    # Unknown speculation must NOT claim EAGLE-3.1 (HardwareAdaptive).
    InteractiveChatSession(client, model='custom-model').start()
    out_unknown = capsys.readouterr().out
    assert 'custom-model' in out_unknown
    assert 'EAGLE-3.1' not in out_unknown

    # Explicit EAGLE-2 Static label must appear accurately.
    InteractiveChatSession(client, model='llama-eagle2', speculation_label='EAGLE-2 (Static)').start()
    out_eagle2 = capsys.readouterr().out
    assert 'EAGLE-2 (Static)' in out_eagle2
    assert 'EAGLE-3.1' not in out_eagle2


def test_build_api_config_payload_local_backend_preserves_local_hardware():
    cfg = {
        'backend': 'local',
        'target_model': 'Qwen/Qwen2.5-7B-Instruct-AWQ',
        'draft_model': 'duckhoi/qwen2.5-7b-eagle3.1-speculator',
        'draft_method': 'eagle3',
        'eagle_version': '3.1',
        'enable_adaptive': True,
        'target_gpu': 'A10',
        'hardware': {
            'device_name': 'NVIDIA GeForce RTX 4090',
            'sm_architecture': 'sm_89',
            'vram_gb': 24,
            'free_vram_mb': 23000.0,
        },
    }
    payload = serving_session.build_api_config_payload(
        endpoint='http://localhost:8000/v1',
        model='local-model',
        config=cfg,
        deploy=False,
    )
    assert payload['backend'] == 'local'
    assert payload['hardware']['device_name'] == 'NVIDIA GeForce RTX 4090'
    assert payload['hardware']['sm_architecture'] == 'sm_89'


@pytest.mark.parametrize(
    'eagle_version,expected_method,expected_label',
    [
        ('1', 'eagle', 'EAGLE-1 (Static)'),
        ('2', 'eagle', 'EAGLE-2 (Static)'),
        ('3', 'eagle3', 'EAGLE-3 (HardwareAdaptive)'),
        ('3.1', 'eagle3', 'EAGLE-3.1 (HardwareAdaptive)'),
    ],
)
def test_eagle_versions_e2e_compat_modal_env_and_export(
    tmp_path, eagle_version, expected_method, expected_label
):
    import json
    import struct
    from flexsds.core.compatibility import check_compatibility
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner

    def write_safetensors(directory, tensors):
        header, payload = {}, b''
        for name, (dtype, shape, raw) in tensors.items():
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(payload), len(payload) + len(raw)]}
            payload += raw
        encoded = json.dumps(header).encode()
        (directory / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)

    target, draft = tmp_path / 'target', tmp_path / 'draft'
    target.mkdir()
    draft.mkdir()
    base_cfg = {
        'architectures': ['Qwen3ForCausalLM'],
        'hidden_size': 4,
        'vocab_size': 4,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'num_key_value_heads': 1,
        'head_dim': 2,
    }
    (target / 'config.json').write_text(json.dumps(base_cfg), encoding='utf-8')
    (target / 'tokenizer.json').write_text(
        json.dumps({'model': {'type': 'BPE', 'vocab': {'a': 0, 'b': 1, 'c': 2, 'd': 3}, 'merges': []}, 'added_tokens': []}),
        encoding='utf-8',
    )
    write_safetensors(target, {'weight': ('BF16', [4], b'\0' * 8)})

    draft_cfg = dict(base_cfg, num_hidden_layers=1, target_model_name_or_path=str(target))
    tensors = {
        'd2t': ('I64', [2], struct.pack('<qq', 0, 2)),
        't2d': ('BOOL', [4], bytes([1, 0, 0, 1])),
    }
    if eagle_version in ('1', '2'):
        draft_cfg.update(architectures=['LlamaForCausalLM'], draft_vocab_size=4, eagle_version=eagle_version)
        tensors['fc.weight'] = ('BF16', [4, 8], b'\0' * 64)
    else:
        draft_cfg.update(architectures=['LlamaForCausalLMEagle3'], draft_vocab_size=2)
        tensors['fc.weight'] = ('BF16', [4, 12], b'\0' * 96)
        if eagle_version == '3.1':
            draft_cfg.update(fc_norm=True, norm_output=True, use_post_norm_feedback=True)
            for i in range(3):
                tensors[f'fc_norm.{i}.weight'] = ('BF16', [4], b'\0' * 8)

    (draft / 'config.json').write_text(json.dumps(draft_cfg), encoding='utf-8')
    write_safetensors(draft, tensors)

    report = check_compatibility(str(target), str(draft), free_vram_mb=46000.0)
    assert report.is_compatible is True, report.warnings
    assert report.draft_method == expected_method
    assert report.eagle_version == f'EAGLE-{eagle_version}'

    cfg = {
        'backend': 'modal',
        'target_model': str(target),
        'draft_model': str(draft),
        'draft_method': report.draft_method,
        'eagle_version': report.eagle_version,
        'enable_adaptive': eagle_version in ('3', '3.1'),
        'target_gpu': 'L40S',
        'port': 8000,
    }
    vllm_cmd = LocalServingRunner().build_command(cfg)
    spec_cfg = json.loads(vllm_cmd[vllm_cmd.index('--speculative-config') + 1])
    assert spec_cfg['method'] == expected_method

    env = ModalServingRunner().setup_environment(cfg)
    forwarded = json.loads(env['FLEXSDS_SERVING_CONFIG'])
    assert forwarded['draft_method'] == expected_method
    assert forwarded['eagle_version'] == f'EAGLE-{eagle_version}'

    payload = serving_session.build_api_config_payload(
        endpoint='https://demo.modal.run/v1',
        model='served-demo',
        config=forwarded,
        deploy=True,
    )
    assert payload['draft_method'] == expected_method
    assert payload['eagle_version'] == eagle_version
    assert payload['hardware']['device_name'] == 'NVIDIA L40S'
    assert serving_session.format_speculation_label(payload) == expected_label
