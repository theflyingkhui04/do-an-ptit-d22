"""A Colab notebook must reflect final wizard choices before it is uploaded."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from flexsds.cli import wizard
from flexsds.core.compatibility import CompatibilityReport
from flexsds.core.hardware import HardwareInfo
from flexsds.server.colab_runner import ColabServingRunner


@pytest.mark.parametrize('mode', ['disabled', 'enabled', 'fallback', 'abort', 'sync-failure'])
def test_colab_notebook_is_prepared_after_adaptive_and_profile_decisions(tmp_path, monkeypatch, mode):
    source = Path(__file__).resolve().parents[1] / 'notebooks/flexsds_colab_server.ipynb'
    notebook = tmp_path/'configured.ipynb'
    original = source.read_bytes()
    notebook.write_bytes(original)
    monkeypatch.chdir(tmp_path)
    events = []
    report = CompatibilityReport(
        target_model='target', draft_model='draft', draft_method='eagle3', use_heterogeneous_vocab=False,
        target_vram_est_mb=500, draft_vram_est_mb=100, kv_cache_min_mb=100,
        total_vram_needed_mb=800, free_vram_mb=15000, oom_risk=False, is_compatible=True,
        identified_method='eagle3', method_supported=True, eagle_version='EAGLE-3.1')
    monkeypatch.setattr(wizard, 'evaluate_serving_preflight', lambda *a, **kw: SimpleNamespace(
        report=report, require_safe=lambda **kw: None))
    def choose(question, choices, **kwargs):
        if question == 'Serving platform:':
            return 'Google Colab (Free GPU T4/L4)'
        if 'missing' in question:
            events.append('profile-choice')
            return 'Abort setup' if mode == 'abort' else 'Switch to static speculation (serve without HardwareAdaptive)'
        return choices[0]
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_positive_int', lambda q, v: 8 if 'batch' in q else 2048)
    def answer(question, **kwargs):
        if question == 'Enable HardwareAdaptive engine?':
            events.append('adaptive')
            return mode in ('enabled', 'fallback', 'abort')
        if 'Open containing folder' in question:
            assert 'adaptive' in events and 'notebook' in events
            events.append('open')
        return False
    monkeypatch.setattr(wizard, 'prompt_bool', answer)
    def profile(*args, **kwargs):
        assert 'adaptive' in events
        events.append('profile')
        return (object(), 'profile.json') if mode == 'enabled' else (None, None)
    monkeypatch.setattr(wizard, 'load_profile_for_hardware', profile)
    monkeypatch.setattr(wizard, 'resolve_profile_path', lambda name: tmp_path/name)
    def resolve(self, notebook_path=None):
        assert 'adaptive' in events
        if mode in ('enabled', 'fallback'):
            assert 'profile' in events
        events.append('notebook')
        return notebook
    monkeypatch.setattr(ColabServingRunner, 'resolve_notebook_path', resolve)
    if mode == 'sync-failure':
        monkeypatch.setattr(ColabServingRunner, 'sync_notebook_config', lambda *a, **kw: False)
    def tunnel(question, default_val=''):
        assert 'Paste Cloudflare' in question
        assert events.index('adaptive') < events.index('notebook') < events.index('open')
        code = '\n'.join(''.join(c.get('source', [])) for c in json.loads(notebook.read_text(encoding='utf-8'))['cells'])
        assert f'ENABLE_ADAPTIVE = {mode == "enabled"}' in code
        assert 'MAX_NUM_SEQS = 8' in code and 'MAX_MODEL_LEN = 2048' in code
        return 'https://example.trycloudflare.com'
    monkeypatch.setattr(wizard, 'prompt_string', tunnel)
    verify = Mock(return_value=True)
    monkeypatch.setattr(ColabServingRunner, 'verify_tunnel_health', verify)
    result = wizard.run_interactive_wizard(HardwareInfo('CPU', 'cpu', 0, 0, '', False))
    if mode in ('abort', 'sync-failure'):
        assert result == {'action':'abort', 'exit_code':1}
        assert notebook.read_bytes() == original
        assert not (tmp_path/'serving_config.json').exists()
        verify.assert_not_called()
    else:
        assert result['backend'] == 'colab'
        assert result['enable_adaptive'] is (mode == 'enabled')
        saved = json.loads((tmp_path/'serving_config.json').read_text())
        assert saved['enable_adaptive'] == result['enable_adaptive']
        assert saved['colab_tunnel_url'] == 'https://example.trycloudflare.com'
