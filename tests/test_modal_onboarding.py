"""Modal onboarding uses mocked installation and token verification, never cloud resources."""
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from flexsds.server import modal_auth as auth
from flexsds.core.credentials import CredentialsManager


def test_missing_modal_installs_with_current_python(monkeypatch):
    monkeypatch.setattr(auth.shutil, 'which', lambda _: None)
    monkeypatch.setattr('importlib.util.find_spec', lambda _: None)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(auth.subprocess, 'run', run)
    auth.ensure_modal()
    assert run.call_args.args[0] == [sys.executable, '-m', 'pip', 'install', 'modal']
    assert auth.modal_command() == [sys.executable, '-m', 'modal']


def test_installed_modal_does_not_install(monkeypatch):
    monkeypatch.setattr(auth.shutil, 'which', lambda _: '/bin/modal')
    run = Mock()
    monkeypatch.setattr(auth.subprocess, 'run', run)
    auth.ensure_modal()
    run.assert_not_called()


def test_failed_install_is_actionable(monkeypatch):
    monkeypatch.setattr(auth.shutil, 'which', lambda _: None)
    monkeypatch.setattr('importlib.util.find_spec', lambda _: None)
    monkeypatch.setattr(auth.subprocess, 'run', Mock(return_value=SimpleNamespace(returncode=1)))
    with pytest.raises(RuntimeError, match='Automatic Modal installation failed'):
        auth.ensure_modal()


@pytest.mark.parametrize('failure', [False, True, 'timeout'])
def test_verify_workspace_before_saving(tmp_path, monkeypatch, capsys, failure):
    manager = CredentialsManager(str(tmp_path/'credentials'))
    manager.set_credential('modal', 'MODAL_TOKEN_ID', 'old-id')
    manager.set_credential('modal', 'MODAL_TOKEN_SECRET', 'old-secret')
    monkeypatch.setenv('MODAL_TOKEN_ID', 'inherited-id')
    monkeypatch.setenv('MODAL_TOKEN_SECRET', 'inherited-secret')
    monkeypatch.setattr(auth, 'ensure_modal', lambda: None)
    monkeypatch.setattr(auth, 'modal_command', lambda: ['modal'])
    def run(command, **kwargs):
        assert command == ['modal', 'token', 'info']
        assert kwargs['env']['MODAL_TOKEN_ID'] == 'new-id'
        assert kwargs['env']['MODAL_TOKEN_SECRET'] == 'new-secret'
        assert manager.get_stored_credential('modal', 'MODAL_TOKEN_ID') == 'old-id'
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=int(failure), stdout='Workspace: thesis-team (ws-123)\n', stderr='new-secret')
    monkeypatch.setattr(auth.subprocess, 'run', run)
    if failure:
        with pytest.raises(RuntimeError):
            auth.save_verified_modal_token(manager, 'new-id', 'new-secret')
        assert manager.get_stored_credential('modal', 'MODAL_TOKEN_ID') == 'old-id'
        assert os.environ['MODAL_TOKEN_ID'] == 'inherited-id'
    else:
        auth.save_verified_modal_token(manager, ' new-id ', ' new-secret ')
        assert manager.get_stored_credential('modal', 'MODAL_TOKEN_ID') == 'new-id'
        assert os.environ['MODAL_TOKEN_SECRET'] == 'new-secret'
        assert 'Modal workspace: thesis-team (ws-123)' in capsys.readouterr().out
    assert 'new-secret' not in capsys.readouterr().out


def test_delete_token_from_first_menu(tmp_path, monkeypatch):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.setattr(CredentialsManager, 'DEFAULT_PATH', tmp_path/'credentials')
    monkeypatch.setenv('MODAL_CONFIG_PATH', str(tmp_path/'missing.toml'))
    manager = CredentialsManager()
    manager.set_credential('modal', 'MODAL_TOKEN_ID', 'id')
    manager.set_credential('modal', 'MODAL_TOKEN_SECRET', 'secret')
    manager.set_credential('huggingface', 'HF_TOKEN', 'keep')
    actions = iter(['Delete Modal token', 'Detect hardware', 'Measure & available profile GPU'])
    monkeypatch.setattr(wizard, 'run_profile_wizard', lambda hw: {'action': 'profile'})
    def choose(question, choices, **kwargs):
        assert question == 'Select action:'
        assert choices[3] == 'Delete Modal token'
        return next(actions)
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    result = wizard.run_interactive_wizard(HardwareInfo('CPU','cpu',0,0,'',False))
    assert result == {'action': 'profile'}
    assert manager.get_stored_credential('modal', 'MODAL_TOKEN_ID') is None
    assert manager.get_stored_credential('huggingface', 'HF_TOKEN') == 'keep'


def test_login_prints_english_steps_before_prompts(tmp_path, monkeypatch, capsys):
    from flexsds.cli.main import main
    monkeypatch.setattr(CredentialsManager, 'DEFAULT_PATH', tmp_path/'credentials')
    monkeypatch.setattr(auth, 'ensure_modal', lambda: None)
    def prompt(question):
        output = capsys.readouterr().out
        if 'ID' in question:
            assert '1. Open https://modal.com' in output
            assert '6. Paste the Token ID' in output
        return 'id' if 'ID' in question else 'secret'
    monkeypatch.setattr('getpass.getpass', prompt)
    saved = Mock()
    monkeypatch.setattr(auth, 'save_verified_modal_token', saved)
    assert main(['auth','login','--provider','modal']) == 0
    assert saved.call_args.args[1:] == ('id', 'secret')
