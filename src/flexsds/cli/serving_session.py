"""Next actions for a verified Modal endpoint, sharing the serving process lifetime."""
import shlex
import sys
from flexsds.cli.output import print_output as print


def present_session(proc, output, *, endpoint, model, config, deploy):
    output.suspend()
    command = shlex.join(['flexsds', 'chat', '--endpoint', endpoint, '--model', model])
    print('\nEndpoint LLM created:')
    for key, value in [('API', endpoint), ('Model', model), ('GPU', config.get('target_gpu', 'unknown')),
                       ('Mode', 'Persistent deployment' if deploy else 'Development (keep this terminal open)'),
                       ('App', output.app_url), ('Logs', output.log_path)]:
        if value:
            print(f'- {key}: {value}')
    print(f'\nChat in another terminal:\n  {command}\n')
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        output.resume()
        return
    from flexsds.cli.wizard import prompt_choice
    from flexsds.client.chat import run_interactive_chat
    while deploy or proc.poll() is None:
        choice = prompt_choice('What would you like to do?',
                               ['Watch log', 'Realtime chat with deployed LLM'])
        if choice == 'Watch log':
            print(f'Chat in another terminal:\n  {command}')
            if deploy:
                # The deploy command exits after publishing; follow the persistent app separately.
                import shutil
                import subprocess
                cmd = [shutil.which('modal') or 'modal', 'app', 'logs',
                       output.app_url.rstrip('/').rsplit('/', 1)[-1] if output.app_url else 'flexsds-serving']
                from flexsds.cli.process_progress import ProcessProgress
                watcher = ProcessProgress('serving')
                child = None
                try:
                    child = subprocess.Popen([*cmd, '--follow'], stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, errors='replace', bufsize=1)
                    watcher.start()
                    reader = watcher.capture(child)
                    watcher.toggle_details()
                    code = child.wait()
                    reader.join(timeout=5)
                    if code:
                        raise RuntimeError(f'Modal log watcher exited with code {code}')
                finally:
                    if child is not None and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    watcher.close()
            else:
                output.resume()
                output.toggle_details()
            return
        run_interactive_chat(endpoint=endpoint, model=model)
        print('\nChat closed. The endpoint is still available.')
