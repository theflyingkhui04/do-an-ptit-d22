"""Readiness requires a real completion response, and must propagate failure."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import Mock
import pytest
from flexsds.server import readiness
from flexsds.server.local_runner import finish_launch, LocalServingRunner
from flexsds.server.modal_runner import ModalServingRunner
from test_hardware_profiler_benchmark import measured_profile


def test_health_models_and_actual_completion():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'data':[{'id':'actual-model'}]} if self.path.endswith('models') else {}).encode())
        def do_POST(self):
            seen.append(self.path)
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert payload['model'] == 'actual-model'
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"choices":[{"text":"Hi"}]}')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        assert readiness.verify_endpoint(f'http://127.0.0.1:{server.server_port}', timeout=2) == 'actual-model'
        assert seen == ['/health','/v1/models','/v1/completions']
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_dead_development_process_is_not_ready():
    proc = Mock(); proc.poll.return_value = 0
    with pytest.raises(RuntimeError, match='exited'):
        readiness.verify_endpoint('http://unused', process=proc, timeout=1)


def test_timeout_without_valid_completion(monkeypatch):
    ticks = iter([0,0,0,0,10,10])
    monkeypatch.setattr(readiness.time, 'monotonic', lambda: next(ticks, 10))
    monkeypatch.setattr(readiness.time, 'sleep', lambda _: None)
    monkeypatch.setattr(readiness, 'request_json', lambda *a, **k: {})
    with pytest.raises(RuntimeError, match='timed out'):
        readiness.verify_endpoint('http://unused', timeout=1)


def test_readiness_failure_stops_process_and_returns_failure():
    proc = Mock()
    def fail(): raise RuntimeError('model load failed')
    proc._flexsds_readiness = fail
    assert finish_launch(proc) == 1
    proc.terminate.assert_called_once()


def test_profile_reaches_environment_and_controller(measured_profile, tmp_path, monkeypatch):
    from src.server.budget_planner import AdaptiveBudgetPlanner
    path = tmp_path / 'profile.json'; path.write_text(json.dumps(measured_profile))
    config = dict(target_model='target', draft_model='draft', draft_method='draft_model',
                  enable_adaptive=True, target_gpu='A10G', profile_path=str(path))
    forwarded = json.loads(ModalServingRunner().setup_environment(config)['FLEXSDS_SERVING_CONFIG'])
    assert forwarded['profile_path'] == str(path)
    env = LocalServingRunner().setup_environment(config)
    planner = AdaptiveBudgetPlanner.from_env(env)
    controller = planner.adapter.get_controller('request')
    assert controller.cost_profile.get_cost(2, 4) == measured_profile['cost_matrix_ms']['k2_b4']
    # Unmeasured intermediate batches select closest observed B, never generic constants.
    assert controller.cost_profile.get_cost(2, 3) == measured_profile['cost_matrix_ms']['k2_b2']
    cmd = LocalServingRunner().build_command(config)
    assert '--scheduler-cls' in cmd and '--enforce-eager' in cmd
    assert cmd[cmd.index('--revision')+1] == 'a'*40


def test_eagle_adaptive_cannot_silently_drop_scheduler():
    with pytest.raises(ValueError, match='EAGLE adaptive'):
        LocalServingRunner().build_command(dict(draft_method='eagle3', enable_adaptive=True))
    cmd = LocalServingRunner().build_command(dict(draft_method='eagle3', enable_adaptive=False))
    assert '--scheduler-cls' not in cmd


@pytest.mark.parametrize('decision', ['static', 'abort'])
def test_default_eagle_wizard_never_requests_useless_calibration(tmp_path, monkeypatch, decision):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.chdir(tmp_path)
    def choose(question, choices, default_idx=0):
        if 'platform' in question.lower(): return choices[0]
        if 'supported serving mode' in question.lower(): return choices[1 if decision == 'abort' else 0]
        return choices[default_idx]
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_bool', lambda question, **kw: 'HardwareAdaptive' in question)
    monkeypatch.setattr(wizard, 'load_profile_for_hardware', lambda *a, **k: pytest.fail('Static EAGLE does not need an adaptive profile'))
    result = wizard.run_interactive_wizard(HardwareInfo('CPU','cpu',0,0,'',False))
    if decision == 'abort':
        assert result == {'action':'abort','exit_code':1}
        assert not (tmp_path / 'serving_config.json').exists()
    else:
        assert result['enable_adaptive'] is False
        assert result['profile_path'] is None
