"""Browse deployed web endpoints and stop only the app selected by the user."""
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from flexsds.server import modal_endpoints as backend
from flexsds.cli import endpoints as ui


def entry(app_id='ap-selected', name='serve'):
    app = dict(app_id=app_id, description='flexsds-serving', state='deployed', tasks='0')
    return dict(app=app, info=dict(description=app['description'], lifecycle=dict(
        state='deployed', deployed_at='2026-11-17 09:00:00+07:00', version=2)),
        kind='function', name=name, entry=dict(id='fu-selected', summary=dict(web_function=True, authenticated=False)))


@pytest.fixture
def service(monkeypatch):
    obj = backend.ModalEndpoints.__new__(backend.ModalEndpoints)
    obj.env = {'MODAL_TOKEN_ID': 'test-id', 'MODAL_TOKEN_SECRET': 'test-secret'}
    monkeypatch.setattr(backend, 'modal_command', lambda: ['modal'])
    return obj


def test_list_includes_idle_deployments_but_excludes_serve_stopped_and_batch_apps(service, monkeypatch):
    apps = [dict(app_id='ap-idle', state='deployed', tasks='0'),
            dict(app_id='ap-dev', state='ephemeral'), dict(app_id='ap-stopped', state='stopped'),
            dict(app_id='ap-batch', state='deployed'), dict(app_id='ap-race', state='deployed')]
    def run(*args):
        if args == ('app', 'list'):
            return apps
        if args[-1] == 'ap-idle':
            return dict(lifecycle={'state':'deployed'}, functions={
                'serve': {'id':'fu-web','summary': {'web_function':True}},
                'batch': {'id':'fu-batch','summary': {}}},
                servers={'http': {'id':'fu-server', 'summary': {}}})
        if args[-1] == 'ap-batch':
            return dict(lifecycle={'state':'deployed'}, functions={'batch':{'id':'fu-batch','summary':{}}})
        if args[-1] == 'ap-race':
            return dict(lifecycle={'state':'stopped'})
        pytest.fail(f'Unexpected call: {args}')
    monkeypatch.setattr(service, '_run', run)
    result = service.list_endpoints()
    assert [(e['kind'], e['name']) for e in result] == [('function','serve'),('server','http')]
    assert all(e['app']['tasks'] == '0' for e in result)


def test_stop_uses_selected_app_id_and_current_credentials(service, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(backend.subprocess, 'run', run)
    service.stop(entry())
    assert run.call_args.args[0] == ['modal', 'app', 'stop', 'ap-selected', '--yes']
    assert run.call_args.kwargs['env'] == service.env
    assert run.call_args.kwargs['stdin'] == subprocess.DEVNULL


@pytest.mark.parametrize('result', [SimpleNamespace(returncode=1, stdout='test-secret'),
                                    SimpleNamespace(returncode=0, stdout='not JSON'),
                                    SimpleNamespace(returncode=0, stdout='null')])
def test_invalid_responses_do_not_look_like_empty_lists(service, monkeypatch, result):
    monkeypatch.setattr(backend.subprocess, 'run', Mock(return_value=result))
    with pytest.raises(RuntimeError) as exc:
        service.list_endpoints()
    assert 'test-secret' not in str(exc.value)


@pytest.mark.parametrize('confirm', [True, False])
def test_menu_shows_details_and_only_stops_after_selection(monkeypatch, capsys, confirm):
    from flexsds.cli import wizard
    selected = entry()
    service = Mock()
    service.list_endpoints.side_effect = [[selected], []] if confirm else [[selected], [selected]]
    service.details.return_value = dict(web_info={'web_url':'https://example.modal.run'},
                                        gpus=[['A10G',1]], timeout=3600)
    monkeypatch.setattr(ui, 'ModalEndpoints', lambda: service)
    actions = iter(['pick', 'Stop deployment'] + ([] if confirm else ['Back to endpoint list']) + ['Back to main menu'])
    def choose(question, choices, **kwargs):
        action = next(actions)
        return choices[0] if action == 'pick' else action
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    def yes(question, default_val):
        assert 'ap-selected' in question and 'all its endpoints' in question
        assert default_val is False
        return confirm
    monkeypatch.setattr(wizard, 'prompt_bool', yes)
    ui.run_endpoint_menu()
    output = capsys.readouterr().out
    assert 'https://example.modal.run' in output and 'A10G x1' in output
    assert 'Deployed: 2026-11-17' in output
    if confirm:
        service.stop.assert_called_once_with(selected)
        assert 'Deployment stopped.' in output
    else:
        service.stop.assert_not_called()


def test_stop_failure_is_reported_and_list_can_be_reopened(monkeypatch, capsys):
    from flexsds.cli import wizard
    service = Mock()
    service.list_endpoints.return_value = [entry()]
    service.details.return_value = {}
    service.stop.side_effect = RuntimeError('permission denied')
    monkeypatch.setattr(ui, 'ModalEndpoints', lambda: service)
    choices = iter(['pick','Stop deployment','Back to main menu'])
    monkeypatch.setattr(wizard, 'prompt_choice', lambda q, opts, **kw: opts[0] if (c:=next(choices))=='pick' else c)
    monkeypatch.setattr(wizard, 'prompt_bool', lambda *a, **k: True)
    ui.run_endpoint_menu()
    output = capsys.readouterr().out
    assert 'permission denied' in output and 'Deployment stopped.' not in output


def test_chat_uses_selected_endpoint_then_returns_to_actions(monkeypatch):
    from flexsds.cli import wizard
    from flexsds.client import chat
    service = Mock()
    service.list_endpoints.return_value = [entry()]
    service.details.return_value = {'web_info':{'web_url':'https://example.modal.run'}}
    monkeypatch.setattr(ui, 'ModalEndpoints', lambda: service)
    choices = iter(['pick','Chat with endpoint','Back to endpoint list','Back to main menu'])
    monkeypatch.setattr(wizard, 'prompt_choice', lambda q, opts, **kw: opts[0] if (c:=next(choices))=='pick' else c)
    run = Mock()
    monkeypatch.setattr(chat, 'run_interactive_chat', run)
    ui.run_endpoint_menu()
    run.assert_called_once_with(endpoint='https://example.modal.run/v1')
    service.stop.assert_not_called()


def test_main_menu_returns_after_browsing(monkeypatch):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    actions = iter(['Running endpoint', 'Detect hardware', 'Detect hardware', 'Measure & available profile GPU'])
    monkeypatch.setattr(wizard, 'run_profile_wizard', lambda hw: {'action': 'profile'})
    def choose(q, opts, **kw):
        assert 'Running endpoint' in opts
        return next(actions)
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    run = Mock()
    monkeypatch.setattr(ui, 'run_endpoint_menu', run)
    assert wizard.run_interactive_wizard(HardwareInfo('CPU','cpu',0,0,'',False)) == {'action':'profile'}
    run.assert_called_once()
