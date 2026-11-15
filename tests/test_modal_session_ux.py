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


def session(output, proc=None, deploy=False):
    proc = proc or Mock()
    proc.poll.return_value = None
    serving_session.present_session(proc, output, endpoint='https://example.modal.run/v1',
                                    model='served-model', config={'target_gpu': 'A10G'}, deploy=deploy)
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
    proc = session(Mock())
    run.assert_called_once_with(endpoint='https://example.modal.run/v1', model='served-model')
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
