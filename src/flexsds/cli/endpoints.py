"""Browse, use, and stop persistent Modal endpoints from the main menu."""
from flexsds.cli.output import print_output as print
from flexsds.cli.spinner import status_spinner
from flexsds.server.modal_endpoints import ModalEndpoints


def show_endpoint(endpoint, details):
    app, info = endpoint['app'], endpoint['info']
    lifecycle = info.get('lifecycle', {})
    web = details.get('web_info') or {}
    url = web.get('web_url')
    gpus = ', '.join(f'{gpu} x{count}' for gpu, count in details.get('gpus', [])) or 'CPU'
    print('\nDeployed endpoint:')
    for label, value in (
        ('App', info.get('description') or app.get('description')),
        ('App ID', app['app_id']), ('Endpoint', endpoint['name']),
        ('Function ID', endpoint['entry']['id']),
        ('State', lifecycle.get('state')), ('URL', url or 'Not available'),
        ('GPU', gpus), ('Memory (MiB)', details.get('memory_mib')),
        ('Request timeout (s)', details.get('timeout')),
        ('Created', lifecycle.get('created_at')), ('Deployed', lifecycle.get('deployed_at')),
        ('Deployed by', lifecycle.get('deployed_by')), ('Version', lifecycle.get('version')),
        ('Active tasks (app)', app.get('tasks', '0')),
        ('Authenticated', endpoint['entry'].get('summary', {}).get('authenticated', 'Unknown')),
    ):
        if value is not None:
            print(f'- {label}: {value}')
    print('Stop deployment stops the whole app, including its other endpoints.')
    return url


def run_endpoint_menu():
    from flexsds.cli.wizard import prompt_choice, prompt_bool
    try:
        service = ModalEndpoints()
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc))
        return
    print('\nRunning endpoints in the active Modal workspace/environment.')
    while True:
        try:
            with status_spinner('Loading deployed endpoints'):
                endpoints = service.list_endpoints()
        except (OSError, RuntimeError, ValueError) as exc:
            print(str(exc))
            return
        if not endpoints:
            print('No deployed web endpoints found.')
        labels = [f"{e['app'].get('description') or 'Unnamed app'} / {e['name']} [{e['app']['app_id']}]"
                  for e in endpoints]
        choice = prompt_choice('Running endpoint:', [*labels, 'Refresh list', 'Back to main menu'])
        if choice == 'Back to main menu':
            return
        if choice == 'Refresh list':
            continue
        endpoint = endpoints[labels.index(choice)]
        try:
            with status_spinner('Loading endpoint information'):
                details = service.details(endpoint)
            url = show_endpoint(endpoint, details)
            while True:
                actions = (['Chat with endpoint'] if url else []) + ['Stop deployment', 'Back to endpoint list']
                action = prompt_choice('Endpoint action:', actions, default_idx=len(actions)-1)
                if action == 'Back to endpoint list':
                    break
                if action == 'Chat with endpoint':
                    from flexsds.client.chat import run_interactive_chat
                    base_url = url.rstrip('/')
                    run_interactive_chat(endpoint=base_url if base_url.endswith('/v1') else base_url+'/v1')
                elif prompt_bool(f"Stop deployment {endpoint['app'].get('description', '')} ({endpoint['app']['app_id']}) and all its endpoints?",
                                 default_val=False):
                    with status_spinner('Stopping deployment'):
                        service.stop(endpoint)
                    print('Deployment stopped.')
                    break
        except (OSError, RuntimeError, ValueError) as exc:
            print(f'Endpoint operation failed: {exc}')
