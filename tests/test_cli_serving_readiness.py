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


def test_eagle_adaptive_uses_sequential_dynamic_k_and_paired_executor():
    cmd = LocalServingRunner().build_command(dict(draft_method='eagle3', enable_adaptive=True))
    spec = json.loads(cmd[cmd.index('--speculative-config')+1])
    assert spec['method'] == 'eagle3'
    assert spec['num_speculative_tokens_per_batch_size'] == [[1,32,4]]
    assert '--async-scheduling' in cmd
    assert cmd[cmd.index('--scheduler-cls')+1] == 'src.server.adaptive_scheduler.AdaptiveAsyncScheduler'
    assert cmd[cmd.index('--distributed-executor-backend')+1] == 'src.server.lineage_executor.LineageUniProcExecutor'
    static = LocalServingRunner().build_command(dict(draft_method='eagle3', enable_adaptive=False))
    assert '--scheduler-cls' not in static
    assert 'num_speculative_tokens_per_batch_size' not in json.loads(static[static.index('--speculative-config')+1])


def test_default_eagle_wizard_reaches_missing_profile_gate(tmp_path, monkeypatch):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.chdir(tmp_path)
    def choose(question, choices, default_idx=0):
        if 'missing' in question.lower(): return 'Abort setup'
        return choices[default_idx]
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_bool', lambda question, **kw: 'HardwareAdaptive' in question)
    monkeypatch.setattr(wizard, 'load_profile_for_hardware', lambda *a, **k: (None, ''))
    assert wizard.run_interactive_wizard(HardwareInfo('CPU','cpu',0,0,'',False)) == {'action':'abort','exit_code':1}
    assert not (tmp_path/'serving_config.json').exists()


def test_cli_eagle_adaptive_with_measured_profile_reaches_modal_preview(measured_profile, tmp_path, monkeypatch):
    from flexsds.core import calibration as cal, cost_model as cm
    from flexsds.cli.main import main
    profile = measured_profile
    config = dict(profile['evidence']['config'], target_model='Qwen/Qwen2.5-7B-Instruct-AWQ',
                  draft_model='duckhoi/qwen2.5-7b-eagle3.1-speculator', draft_method='eagle3')
    data = cal.build_profile(config, profile['evidence']['groups'], cal.BATCHES)
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    path = cm.save_hardware_profile(data, 'sm_86', 'a10g')
    seen = []
    monkeypatch.setattr(ModalServingRunner, 'launch', lambda self, cfg, **kw: seen.append((cfg, kw)))
    assert main(['serve','--backend','modal','--gpu','A10G','--target',config['target_model'],
                 '--draft',config['draft_model'],'--dry-run']) == 0
    cfg = seen[0][0]['serving']
    assert cfg['enable_adaptive'] is True and cfg['draft_method'] == 'eagle3'
    assert cfg['profile_path'] == str(path)
    assert seen[0][1]['dry_run'] is True
    env = LocalServingRunner().setup_environment(cfg)
    assert env['ADAPTIVE_STRICT'] == '1' and env['VLLM_USE_V2_MODEL_RUNNER'] == '0'
    assert 'ADAPTIVE_STEP_TELEMETRY_PATH' in env and 'ADAPTIVE_STARTUP_TELEMETRY_PATH' in env


def test_partial_profile_requires_matching_serving_capacity(measured_profile, tmp_path):
    from flexsds.core import calibration as cal
    config = measured_profile['evidence']['config']
    groups = measured_profile['evidence']['groups']
    for group in groups: group['records'] = [r for r in group['records'] if r['batch'] == 1]
    data = cal.build_profile(config,groups,[1])
    path = tmp_path/'batch1.json'; path.write_text(json.dumps(data))
    serving = dict(target_model='target',draft_model='draft',draft_method='draft_model',
                   enable_adaptive=True,profile_path=str(path),max_num_seqs=1,
                   telemetry_path=str(tmp_path/'trace.jsonl'))
    assert LocalServingRunner().setup_environment(serving)['FLEXSDS_PROFILE_PATH'] == str(path)
    serving['max_num_seqs'] = 32
    with pytest.raises(ValueError,match='does not cover'):
        LocalServingRunner().setup_environment(serving)
