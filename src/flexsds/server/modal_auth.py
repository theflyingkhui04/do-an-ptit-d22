"""Local Modal profile operations, including installations in a separate pipx venv."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def ensure_modal():
    """Install Modal into the running Python when no CLI or SDK is available."""
    if shutil.which('modal'):
        return
    import importlib.util
    if importlib.util.find_spec('modal') is not None:
        return
    from flexsds.cli.output import print_output as print
    print('Modal is not installed. Installing Modal automatically...')
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'modal'],
                            stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError('Automatic Modal installation failed. Check your connection and Python environment permissions, then retry.')


def modal_command():
    executable = shutil.which('modal')
    return [executable] if executable else [sys.executable, '-m', 'modal']


def print_token_instructions():
    from flexsds.cli.output import print_output as print
    print('\nHow to get your Modal API token:')
    print('1. Open https://modal.com and sign in to your account.')
    print('2. Select the workspace you want to use for serving.')
    print('3. Open Settings > API Tokens (https://modal.com/settings/tokens).')
    print('4. Create a new API token for that workspace.')
    print('5. Copy the Token ID (ak-...) and Token Secret (as-...). The secret is shown only once.')
    print('6. Paste the Token ID and Token Secret into the prompts below.')
    print('FlexSDS will verify the token and display its workspace before continuing.\n')


def save_verified_modal_token(manager, token_id, token_secret):
    """Verify the pasted pair, show its workspace, then persist it locally."""
    import re
    from flexsds.cli.output import print_output as print
    token_id, token_secret = token_id.strip(), token_secret.strip()
    if not token_id or not token_secret:
        raise RuntimeError('Both Modal token ID and secret are required.')
    ensure_modal()
    env = dict(os.environ, MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret,
               NO_COLOR='1', TERM='dumb', COLUMNS='240', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    env.pop('PYTHONPATH', None)
    print('Verifying Modal token and workspace...')
    try:
        result = subprocess.run([*modal_command(), 'token', 'info'], env=env,
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('Could not verify the Modal workspace. Check your connection and retry.') from exc
    # Never echo command output on failure: it may contain credential details.
    workspace = re.search(r'^\s*Workspace:\s*(.+)$', result.stdout, re.MULTILINE)
    if result.returncode or not workspace:
        raise RuntimeError('Modal token verification failed. Check the token pair, workspace access, and connection, then retry.')
    print('Modal workspace: ' + workspace.group(1).strip())
    manager.set_credential('modal', 'MODAL_TOKEN_ID', token_id)
    manager.set_credential('modal', 'MODAL_TOKEN_SECRET', token_secret)
    os.environ.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret)
    print('Modal credentials verified and saved.')


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
