"""Production readiness waits for publication and owns its runtime log follower."""
import io
from contextlib import nullcontext
from threading import Event
from unittest.mock import Mock
import pytest
from flexsds.server import modal_runner, modal_auth, readiness


@pytest.mark.parametrize('interrupted', [False, True])
def test_runtime_logs_are_captured_and_follower_is_stopped(monkeypatch, interrupted):
    child = Mock()
    child.poll.return_value = None
    child.stdout = io.StringIO('Starting to load model\nJIT kernel warmup 50%\n')
    popen = Mock(return_value=child)
    monkeypatch.setattr(modal_runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(modal_auth, 'modal_command', lambda: ['modal'])
    from flexsds.cli.process_progress import ProcessProgress
    output = ProcessProgress('serving')
    try:
        with pytest.raises(KeyboardInterrupt) if interrupted else nullcontext():
            with modal_runner.deployment_logs(output, {'PYTHONUTF8': '1'}):
                assert output.done.wait(2)
                assert 'JIT kernel warmup 50%' in output.log_path.read_text()
                if interrupted:
                    raise KeyboardInterrupt
        assert popen.call_args.args[0] == ['modal', 'app', 'logs', 'flexsds-serving', '--follow']
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)
    finally:
        output.close()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_production_verification_starts_only_after_deploy_exit(monkeypatch, exit_code):
    monkeypatch.setattr(modal_auth, 'ensure_modal', lambda: None)
    monkeypatch.setattr(modal_runner.ModalServingRunner, 'check_credentials', lambda _: True)
    proc = Mock()
    proc.poll.return_value = exit_code
    proc.returncode = exit_code
    follower = Mock()
    follower.poll.return_value = None
    popen = Mock(side_effect=[proc, follower])
    monkeypatch.setattr(modal_runner.subprocess, 'Popen', popen)
    from flexsds.cli import process_progress
    output = Mock(done=Event())
    reader = Mock()
    def capture(child, on_line=None, **kwargs):
        if on_line:
            on_line('https://example--serve.modal.run')
        return reader
    output.capture.side_effect = capture
    monkeypatch.setattr(process_progress, 'ProcessProgress', lambda _: output)
    def verify(endpoint, **kwargs):
        reader.join.assert_called()
        assert proc.poll() == 0
        assert popen.call_count == 2
        kwargs['on_status']('Models OK; verifying actual inference')
        return 'served-model'
    verify_mock = Mock(side_effect=verify)
    monkeypatch.setattr(readiness, 'verify_endpoint', verify_mock)
    launched = modal_runner.ModalServingRunner().launch({'enable_adaptive': False}, deploy=True)
    if exit_code:
        with pytest.raises(RuntimeError, match='deployment failed'):
            launched._flexsds_readiness()
        verify_mock.assert_not_called()
        assert popen.call_count == 1
    else:
        launched._flexsds_readiness()
        assert launched._flexsds_session['model'] == 'served-model'
        follower.terminate.assert_called_once()
        output.update.assert_any_call('Deployment verified', 100)
