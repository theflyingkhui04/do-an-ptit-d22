"""Process output is hidden by default, complete on demand, and interruption-safe."""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import pytest
from flexsds.cli import process_progress as pp


@pytest.fixture(autouse=True)
def private_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(pp,'LOG_DIR',tmp_path/'logs')


def test_subprocess_logs_are_saved_not_dumped(capsys):
    script="print('RAW_ENGINE_MARKER'); print('Calibration measured K=0, B=1')"
    pp.run_logged_process([sys.executable,'-c',script],env=os.environ.copy(),timeout=5,total_cells=1)
    output=capsys.readouterr().out
    assert 'RAW_ENGINE_MARKER' not in output
    assert '100%' in output and 'Calibration job completed' in output
    log=next(pp.LOG_DIR.glob('*.log'))
    assert 'RAW_ENGINE_MARKER' in log.read_text()
    if os.name != 'nt': assert log.stat().st_mode & 0o077 == 0


def test_details_replay_entire_history_and_stream_new_lines(capsys):
    progress=pp.ProcessProgress('calibration',2)
    try:
        for i in range(20):progress.feed(f'old-line-{i}-'+('x'*100)+'\n')
        assert 'old-line-' not in capsys.readouterr().out
        progress.toggle_details()
        progress.feed('new-live-line\n')
        output=capsys.readouterr().out
        assert 'old-line-0-' in output and 'old-line-19-' in output
        assert 'x'*100 in output and 'new-live-line' in output
        progress.toggle_details()
        progress.feed('hidden-again\n')
        assert 'hidden-again' not in capsys.readouterr().out
    finally:progress.close()


def test_progress_counts_cells_once_not_elapsed_time():
    progress=pp.ProcessProgress('calibration',2)
    try:
        progress.feed('Calibration measured K=0, B=1\n')
        first=progress.percent
        progress.feed('Calibration measured K=0, B=1\n')
        progress.feed('JIT kernel warmup: 100%\n')
        assert progress.percent==first and progress.percent<100
        progress.feed('Calibration measured K=1, B=1\n')
        assert progress.percent==95
    finally:progress.close()


def test_failure_keeps_full_log_and_reports_root_cause(capsys):
    with pytest.raises(subprocess.CalledProcessError):
        pp.run_logged_process([sys.executable,'-c',"print('RuntimeError: missing nvcc'); raise SystemExit(7)"],
                              env=os.environ.copy(),timeout=5,total_cells=1)
    output=capsys.readouterr().out
    assert 'Calibration failed: missing nvcc' in output
    assert 'Full details:' in output
    assert 'missing nvcc' in next(pp.LOG_DIR.glob('*.log')).read_text()


@pytest.mark.skipif(os.name=='nt',reason='POSIX terminal restoration')
def test_ctrl_o_and_interrupt_restore_real_terminal(tmp_path):
    import pty
    master,slave=pty.openpty()
    root=Path(__file__).resolve().parents[1]
    script='''import os,sys,termios
from pathlib import Path
from flexsds.cli import process_progress as pp
pp.LOG_DIR=Path(sys.argv[1])
old=termios.tcgetattr(0)
try:
 pp.run_logged_process([sys.executable,'-c',"import time; print('PTY_RAW_LOG',flush=True); time.sleep(10)"],env=os.environ.copy(),timeout=12,total_cells=1)
except KeyboardInterrupt:
 print('CANCELLED',flush=True)
finally:
 print('RESTORED',termios.tcgetattr(0)==old,flush=True)
'''
    env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root),PYTHONUNBUFFERED='1')
    proc=subprocess.Popen([sys.executable,'-c',script,str(tmp_path/'pty-logs')],stdin=slave,stdout=slave,stderr=slave,env=env)
    buffer=b''
    def until(marker):
        nonlocal buffer
        deadline=time.monotonic()+5
        while marker not in buffer and time.monotonic()<deadline:
            if select.select([master],[],[],0.1)[0]:
                try:buffer+=os.read(master,65536)
                except OSError:break
        assert marker in buffer,buffer.decode(errors='replace')
    try:
        until(b'Ctrl+O for details')
        assert b'PTY_RAW_LOG' not in buffer
        os.write(master,b'\x0f')
        until(b'PTY_RAW_LOG')
        os.write(master,b'\x0f')
        until(b'\x1b[?1049l')
        assert b'\x1b[?1049h' in buffer
        os.kill(proc.pid,signal.SIGINT)
        until(b'RESTORED True')
        assert proc.wait(timeout=5)==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        os.close(master);os.close(slave)


def test_modal_serving_uses_same_compact_capture(monkeypatch,capsys):
    from flexsds.server.modal_runner import ModalServingRunner
    from flexsds.server.local_runner import finish_launch
    runner=ModalServingRunner()
    monkeypatch.setattr(runner,'check_credentials',lambda:True)
    script="import time; print('RAW_RUNTIME_MARKER',flush=True); print('https://test--serve.modal.run',flush=True); time.sleep(.5)"
    monkeypatch.setattr(runner,'build_modal_command',lambda *a,**kw:[sys.executable,'-c',script])
    monkeypatch.setattr('flexsds.server.readiness.verify_endpoint',lambda *a,**kw:'served-model')
    proc=runner.launch({'enable_adaptive':False})
    assert finish_launch(proc)==0
    output=capsys.readouterr().out
    assert 'RAW_RUNTIME_MARKER' not in output
    assert 'Inference verified:' in output
    assert 'RAW_RUNTIME_MARKER' in proc._flexsds_output.log_path.read_text()
    assert proc._flexsds_output.closed


def test_profile_cancel_is_a_clean_cli_exit(monkeypatch,capsys):
    from flexsds.cli import main as cli
    def cancel(*a,**kw):raise KeyboardInterrupt
    monkeypatch.setattr(cli,'run_hardware_micro_benchmark',cancel)
    assert cli.main(['profile','--benchmark'])==130
    assert 'Calibration cancelled.' in capsys.readouterr().out


def test_close_open_popup_restores_primary_and_keeps_capture_paused(capsys):
    progress = pp.ProcessProgress('serving')
    progress.interactive = True
    progress.feed('only-in-popup\n')
    capsys.readouterr()
    progress.toggle_details()
    progress.close()
    text = capsys.readouterr().out
    assert '\x1b[?1049h' in text and '\x1b[?1049l' in text
    assert 'only-in-popup' in text.split('\x1b[?1049l')[0]
    assert 'only-in-popup' not in text.split('\x1b[?1049l')[1]
    assert progress.closed and not progress.details
