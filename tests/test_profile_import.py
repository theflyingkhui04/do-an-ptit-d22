"""Profile import preserves measurements, indexes artifacts, and keeps strict validation."""
import json
from pathlib import Path
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from flexsds.core import cost_model as cm
from flexsds.core.profile_import import import_profile, read_profile_source
from flexsds.cli.profile_paste import _collect
from test_hardware_profiler_benchmark import measured_profile


def test_import_measured_profile_is_cataloged_and_discovered(measured_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    text = json.dumps(measured_profile)
    path, status = import_profile(text)
    assert status == 'measured'
    assert json.loads(path.read_text()) == measured_profile
    again, _ = import_profile(json.dumps(measured_profile, indent=2))
    assert again == path
    with sqlite3.connect(tmp_path/'profiles.sqlite3') as db:
        rows = db.execute('SELECT filename,device,status FROM profiles').fetchall()
    assert rows == [(path.name, measured_profile['metadata']['device'], 'measured')]
    model, name = cm.load_profile_for_hardware(custom_profile_path=path, strict=True)
    assert name == path.name and model is not None
    # Automatic hardware lookup must also find the imported artifact.
    from flexsds.core.hardware import HardwareInfo
    meta = measured_profile['metadata']
    hw = HardwareInfo(meta['device'],meta['sm_architecture'],0,0,'',True)
    _, name = cm.load_profile_for_hardware(hw, strict=True)
    assert name == path.name


def test_legacy_is_stored_without_bypassing_serving_validation(tmp_path):
    legacy = Path(__file__).resolve().parents[1]/'src/flexsds/profiles/profile_sm75_t4.json'
    path, status = import_profile(legacy.read_text(), tmp_path)
    assert status == 'legacy'
    assert json.loads(path.read_text()) == json.loads(legacy.read_text())
    with pytest.raises(ValueError, match='verified worker'):
        cm.load_profile_for_hardware(custom_profile_path=path, strict=True)


@pytest.mark.parametrize('text', ['not json', '[]', '{}', '{"metadata":{"sm_architecture":"../../escape"}}'])
def test_bad_input_leaves_no_artifact_or_database(tmp_path, text):
    with pytest.raises((ValueError, TypeError)):
        import_profile(text, tmp_path)
    assert not list(tmp_path.iterdir())


def test_tampered_worker_measurements_are_not_saved(measured_profile, tmp_path):
    measured_profile['cost_matrix_ms']['k0_b1'] = 9999
    with pytest.raises(ValueError):
        import_profile(json.dumps(measured_profile), tmp_path)
    assert not list(tmp_path.iterdir())


def test_local_path_and_http_url_read_utf8(tmp_path):
    content = '{"device":"GPU có dấu"}'
    file = tmp_path/'profile có dấu.json'
    file.write_text(content, encoding='utf-8-sig')
    assert read_profile_source(f'"{file}"') == content
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(content.encode())
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        assert read_profile_source(f'http://127.0.0.1:{server.server_port}/profile.json') == content
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_database_failure_rolls_back_new_artifact(tmp_path):
    (tmp_path/'profiles.sqlite3').mkdir()
    legacy = Path(__file__).resolve().parents[1]/'src/flexsds/profiles/profile_sm75_t4.json'
    with pytest.raises(sqlite3.Error):
        import_profile(legacy.read_text(), tmp_path)
    assert not list(tmp_path.glob('*.json'))


@pytest.mark.parametrize('keys,expected', [('{"x":1}\x18','{"x":1}'),
                                          ('wrong\x1b',None),
                                          ('{"x":2\x08'+'1}\x18','{"x":1}')])
def test_paste_controls_save_cancel_and_backspace(keys, expected):
    source = iter(keys)
    assert _collect(lambda:next(source),lambda *a:None) == expected


def test_import_menu_paste_and_url_share_database(tmp_path, monkeypatch):
    from flexsds.cli import wizard, profile_paste
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR',tmp_path/'profiles')
    legacy = Path(__file__).resolve().parents[1]/'src/flexsds/profiles/profile_sm75_t4.json'
    source = iter(['Import hardware profile','Paste JSON (Ctrl+X to save)',
                   'Import hardware profile','File path or URL'])
    def choose(q, choices, **kw):
        value=next(source)
        assert value in choices
        return value
    monkeypatch.setattr(wizard,'prompt_choice',choose)
    monkeypatch.setattr(profile_paste,'paste_profile',lambda:legacy.read_text())
    monkeypatch.setattr(wizard,'prompt_string',lambda *a: str(legacy))
    hw=HardwareInfo('CPU','cpu',0,0,'',False)
    first=wizard.run_profile_wizard(hw)
    second=wizard.run_profile_wizard(hw)
    assert first['profile_path'] == second['profile_path']
    with sqlite3.connect(tmp_path/'profiles/profiles.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM profiles').fetchone()[0] == 1


@pytest.mark.skipif(__import__('os').name == 'nt', reason='POSIX terminal integration')
@pytest.mark.parametrize('cancel', [False, True])
def test_paste_popup_restores_terminal_after_submit_or_cancel(tmp_path, cancel):
    import os
    import pty
    import select
    import subprocess
    import sys
    import time
    master, slave = pty.openpty()
    root = Path(__file__).resolve().parents[1]
    script = '''import json,termios
from flexsds.cli.profile_paste import paste_profile
old=termios.tcgetattr(0)
result=paste_profile()
print('RESULT',json.dumps(result),flush=True)
print('RESTORED',termios.tcgetattr(0)==old,flush=True)
'''
    proc = subprocess.Popen([sys.executable,'-c',script],stdin=slave,stdout=slave,stderr=slave,
                            env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root)))
    buffer = b''
    def until(marker):
        nonlocal buffer
        deadline = time.monotonic()+4
        while marker not in buffer and time.monotonic()<deadline:
            if select.select([master],[],[],0.1)[0]:
                try: buffer += os.read(master,65536)
                except OSError: break
        assert marker in buffer, buffer.decode(errors='replace')
    try:
        until(b'Ctrl+X: save')
        payload = '{"device":"GPU có dấu"}'
        os.write(master,b'\x1b' if cancel else b'\x1b[200~'+payload.encode()+b'\x1b[201~\x18')
        until(b'RESTORED True')
        assert proc.wait(timeout=3) == 0
        assert b'\x1b[?1049l' in buffer
        expected = json.dumps(None if cancel else payload).encode()
        assert b'RESULT '+expected in buffer
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
        os.close(master); os.close(slave)


def test_windows_paste_reader_consumes_special_keys_and_ctrl_x(monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    import sys
    from flexsds.cli import profile_paste
    keys=deque(['\xe0','H','{','}','\x18'])
    monkeypatch.setattr(profile_paste,'os',SimpleNamespace(name='nt'))
    monkeypatch.setitem(sys.modules,'msvcrt',SimpleNamespace(getwch=lambda:keys.popleft()))
    assert profile_paste._collect(profile_paste._read_key,lambda *a:None) == '{}'
    assert not keys
