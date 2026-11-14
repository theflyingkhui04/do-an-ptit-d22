"""Local Modal profile operations, including installations in a separate pipx venv."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def native_profile(action='inspect'):
    path = Path(os.environ.get('MODAL_CONFIG_PATH') or '~/.modal.toml').expanduser()
    if not path.exists():
        return {'configured': False}
    executable = shutil.which('modal')
    interpreter = sys.executable
    if executable:
        first = Path(executable).read_bytes().split(b'\n', 1)[0]
        if first.startswith(b'#!'):
            interpreter_args = shlex.split(first[2:].decode())
        else:
            candidate = Path(executable).parent / 'python.exe'
            interpreter_args = [str(candidate)] if candidate.exists() else [interpreter]
    else:
        interpreter_args = [interpreter]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    try:
        result = subprocess.run([*interpreter_args, str(Path(__file__).resolve()), action],
                                env=env, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Timed out reading the local Modal profile.') from exc
    if result.returncode:
        raise RuntimeError('Cannot access the active Modal profile; check the Modal CLI installation.')
    return json.loads(result.stdout)


def clear_modal(manager):
    """Remove local active-account tokens; leave other accounts and providers intact."""
    # Block inherited shell tokens on future invocations, even if native cleanup fails.
    marker = manager.path.parent / 'modal_logged_out'
    marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.touch(mode=0o600)
    manager.delete_credential('modal')
    for key in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'):
        os.environ.pop(key, None)
    return native_profile('clear')


def prepare_modal_auth(manager):
    keys = ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET')
    stored = [manager.get_stored_credential('modal', key) for key in keys]
    if all(stored):
        os.environ.update(zip(keys, stored))
        return True
    if (manager.path.parent / 'modal_logged_out').exists():
        for key in keys:
            os.environ.pop(key, None)
    elif all(os.environ.get(key) for key in keys):
        return True
    return native_profile()['configured']


if __name__ == '__main__':
    # Execute with the installed Modal CLI interpreter, without making network calls.
    import modal.config as config
    data = config._read_user_config()
    profile = config._profile
    entry = data.get(profile, {})
    if sys.argv[1] == 'clear':
        entry.pop('token_id', None)
        entry.pop('token_secret', None)
        config._write_user_config(data)
        os.chmod(config.user_config_path, 0o600)
    print(json.dumps({'configured': bool(entry.get('token_id') and entry.get('token_secret')),
                      'profile': profile}))
