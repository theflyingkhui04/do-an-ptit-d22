"""Inspect deployed Modal web endpoints using the same account as serving."""
import json
import subprocess
from flexsds.server.modal_auth import ensure_modal, modal_command
from flexsds.server.modal_runner import ModalServingRunner


class ModalEndpoints:
    def __init__(self):
        ensure_modal()
        runner = ModalServingRunner()
        if not runner.check_credentials():
            raise RuntimeError('Modal credentials are not configured. Run flexsds auth login --provider modal.')
        self.env = runner.setup_environment({})

    def _run(self, *args, json_output=True):
        command = [*modal_command(), *args]
        if json_output:
            command.append('--json')
        try:
            result = subprocess.run(command, env=self.env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError('Cannot reach Modal. Check your connection and retry.') from exc
        if result.returncode:
            raise RuntimeError(f'Modal {args[0]} {args[1]} failed. Check workspace access or refresh the list.')
        if not json_output:
            return
        try:
            data = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError('Modal returned invalid JSON. Check your Modal CLI installation.') from exc
        expected = list if args[:2] == ('app', 'list') else dict
        if not isinstance(data, expected):
            raise RuntimeError('Modal returned an unexpected response. Check your Modal CLI installation.')
        return data

    def list_endpoints(self):
        apps = self._run('app', 'list')
        if not isinstance(apps, list):
            raise RuntimeError('Modal returned an invalid app list.')
        endpoints = []
        for app in apps:
            if app.get('state') != 'deployed':
                continue
            info = self._run('app', 'info', app['app_id'])
            if info.get('lifecycle', {}).get('state') != 'deployed':
                continue  # A deployment may stop between list and info.
            for kind, entries in (('function', info.get('functions', {})), ('server', info.get('servers', {}))):
                for name, entry in entries.items():
                    if kind == 'function' and not entry.get('summary', {}).get('web_function'):
                        continue
                    endpoints.append(dict(app=app, info=info, kind=kind, name=name, entry=entry))
        return endpoints

    def details(self, endpoint):
        return self._run(endpoint['kind'], 'info', endpoint['entry']['id'])

    def stop(self, endpoint):
        # Use the selected immutable App ID, never the shared deployment name.
        self._run('app', 'stop', endpoint['app']['app_id'], '--yes', json_output=False)
