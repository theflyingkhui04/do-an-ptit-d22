"""Local interpreter/profile regression tests; never launch a Modal app."""
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

from flexsds.server import modal_auth as auth


def launcher(path, interpreter):
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('__main__.py', 'from modal.cli import entrypoint\n')
    # The embedded path, not an adjacent python.exe, owns the CLI environment.
    shebang = f'#!"{interpreter}"\n'.encode()
    path.write_bytes(b'MZ\0stub-with-decoy-#!ignored\n' + shebang + archive.getvalue())
    return path


@pytest.mark.parametrize('layout', ['venv', 'external'])
def test_windows_launcher_uses_recorded_interpreter(tmp_path, monkeypatch, layout):
    if layout == 'venv':
        interpreter = tmp_path / 'Modal venv' / 'Scripts' / 'python.exe'
    else:
        interpreter = tmp_path / 'Global Python' / 'python.exe'
    executable = launcher(interpreter.parent / 'Scripts' / 'modal.exe'
                          if layout == 'external' else interpreter.with_name('modal.exe'), interpreter)
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    monkeypatch.setattr(auth.sys, 'executable', str(tmp_path / 'unrelated-venv/python.exe'))
    assert auth._modal_interpreter_args(str(executable)) == [str(interpreter)]


def test_posix_shebang_retains_interpreter_flags_and_quoting(tmp_path):
    executable = tmp_path / 'modal'
    executable.write_text('#!"/opt/Modal Environment/bin/python" -s\nprint("unused")\n')
    assert auth._modal_interpreter_args(str(executable)) == ['/opt/Modal Environment/bin/python', '-s']


def test_no_cli_uses_current_interpreter():
    assert auth._modal_interpreter_args(None) == [sys.executable]


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    profile = tmp_path / 'modal.toml'
    profile.touch()  # Credentials are never needed by the mocked child.
    monkeypatch.setenv('MODAL_CONFIG_PATH', str(profile))
    monkeypatch.setenv('PYTHONPATH', 'unrelated-packages')
    executable = launcher(tmp_path / 'external/Scripts/modal.exe', tmp_path / 'external/python.exe')
    monkeypatch.setattr(auth.shutil, 'which', lambda _: str(executable))
    return executable


def test_modal_capable_child_returns_only_profile_status(profile_env, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='{"configured": true, "profile": "test"}'))
    monkeypatch.setattr(auth.subprocess, 'run', run)
    assert auth.native_profile() == {'configured': True, 'profile': 'test'}
    command = run.call_args.args[0]
    assert command == [str(profile_env.parent.parent / 'python.exe'), str(Path(auth.__file__).resolve()), 'inspect']
    assert 'PYTHONPATH' not in run.call_args.kwargs['env']
    assert run.call_args.kwargs['stdin'] == subprocess.DEVNULL
    assert run.call_args.kwargs['capture_output']


@pytest.mark.parametrize('failure', ['missing_interpreter', 'missing_modal', 'timeout'])
def test_child_failures_do_not_expose_output(profile_env, monkeypatch, capsys, failure):
    if failure == 'missing_interpreter':
        run = Mock(side_effect=FileNotFoundError('private subprocess details'))
    elif failure == 'timeout':
        run = Mock(side_effect=subprocess.TimeoutExpired(['python'], 15, output='private subprocess details'))
    else:
        run = Mock(return_value=SimpleNamespace(returncode=1, stdout='private subprocess details',
                                               stderr="ModuleNotFoundError: No module named 'modal'"))
    monkeypatch.setattr(auth.subprocess, 'run', run)
    with pytest.raises(RuntimeError) as error:
        auth.native_profile()
    assert 'private subprocess details' not in str(error.value)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


@pytest.mark.parametrize('contents', [b'MZbroken launcher', b'not a Python entrypoint', b'#!\n'])
def test_unknown_launcher_fails_without_guessing(tmp_path, contents):
    executable = tmp_path / 'modal.exe'
    executable.write_bytes(contents)
    with pytest.raises(RuntimeError, match='Cannot determine the Modal CLI interpreter'):
        auth._modal_interpreter_args(str(executable))


def test_missing_cli_file_is_actionable(tmp_path):
    with pytest.raises(RuntimeError, match='Cannot determine the Modal CLI interpreter'):
        auth._modal_interpreter_args(str(tmp_path / 'missing.exe'))
