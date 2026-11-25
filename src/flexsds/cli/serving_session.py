"""Next actions for a verified Modal endpoint, sharing the serving process lifetime."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from typing import Any, Optional

from flexsds import __version__
from flexsds.cli.output import print_output as print
from flexsds.core.hardware import MODAL_GPU_NAMES, get_modal_gpu_spec


EXPORT_ACTION = 'Export API config JSON (for web demo)'


def _normalize_eagle_version(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'unknown', 'non-eagle', 'none'}:
        return None
    upper = text.upper()
    if upper.startswith('EAGLE-'):
        text = text[6:].strip()
    elif upper.startswith('EAGLE'):
        text = text[5:].strip().lstrip('_- ')
    return text or None


def _is_vanilla_config(cfg: dict[str, Any]) -> bool:
    if cfg.get('disable_speculative'):
        return True
    if cfg.get('draft_method') == 'none' and not cfg.get('draft_model'):
        return True
    if 'draft_model' in cfg and not cfg.get('draft_model'):
        return True
    return False


def format_speculation_label(config: Optional[dict[str, Any]]) -> Optional[str]:
    """Return an accurate human-readable speculation label, or None when unknown."""
    if not isinstance(config, dict):
        return None
    cfg = config.get('serving', config) if isinstance(config.get('serving'), dict) else config
    if _is_vanilla_config(cfg):
        return 'Standard (Vanilla)'

    version = _normalize_eagle_version(cfg.get('eagle_version'))
    method = str(cfg.get('draft_method') or '').strip().lower()
    if version:
        base = f'EAGLE-{version}'
    elif method == 'eagle3':
        base = 'EAGLE-3'
    elif method == 'eagle':
        base = 'EAGLE'
    elif method == 'draft_model':
        base = 'Draft model'
    else:
        return None

    mode = 'HardwareAdaptive' if cfg.get('enable_adaptive') else 'Static'
    return f'{base} ({mode})'


def _clean_hardware_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    target_gpu = str(cfg.get('target_gpu') or '').strip()
    if cfg.get('backend', 'modal') == 'modal' and target_gpu and target_gpu.upper() in MODAL_GPU_NAMES:
        spec = get_modal_gpu_spec(target_gpu)
        return {
            'device_name': f'NVIDIA {spec.name}',
            'sm_architecture': spec.sm_architecture,
            'vram_gb': spec.vram_gb,
            'tier': spec.tier,
            'free_vram_mb': spec.vram_budget_mb,
        }

    raw_hw = cfg.get('hardware') if isinstance(cfg.get('hardware'), dict) else {}
    device_name = str(raw_hw.get('device_name') or '').strip()
    if device_name.startswith('Remote '):
        bare = device_name.removeprefix('Remote ').strip()
        device_name = f'NVIDIA {bare}' if bare and not bare.upper().startswith('NVIDIA ') else bare
    elif not device_name and target_gpu:
        device_name = f'NVIDIA {target_gpu}'

    hardware: dict[str, Any] = {}
    if device_name:
        hardware['device_name'] = device_name
    for key in ('sm_architecture', 'vram_gb', 'tier', 'free_vram_mb'):
        if raw_hw.get(key) is not None:
            hardware[key] = raw_hw[key]
    return hardware


def build_api_config_payload(
    endpoint: str,
    model: str,
    config: dict[str, Any],
    deploy: bool = False,
) -> dict[str, Any]:
    """Build a secret-free JSON config compatible with the web demo's import flow."""
    cfg = config.get('serving', config) if isinstance(config.get('serving'), dict) else (config or {})
    vanilla = _is_vanilla_config(cfg)
    payload: dict[str, Any] = {
        'flexsds_version': __version__,
        'action': 'serve',
        'backend': cfg.get('backend', 'modal'),
        'endpoint': endpoint,
        'served_model': model,
        'target_model': cfg.get('target_model'),
        'draft_model': None if vanilla else cfg.get('draft_model'),
        'draft_method': 'none' if vanilla else cfg.get('draft_method'),
        'eagle_version': None if vanilla else _normalize_eagle_version(cfg.get('eagle_version')),
        'enable_adaptive': False if vanilla else bool(cfg.get('enable_adaptive', False)),
        'target_gpu': cfg.get('target_gpu'),
        'modal_deploy': bool(deploy),
        'hardware': _clean_hardware_metadata(cfg),
    }
    for key in ('max_num_seqs', 'max_model_len'):
        if cfg.get(key) is not None:
            payload[key] = cfg[key]
    return payload


def _export_api_config(endpoint: str, model: str, config: dict[str, Any], deploy: bool) -> None:
    from flexsds.cli.wizard import prompt_string

    raw_path = prompt_string('Output JSON file path:', default_val='serving_config.json').strip()
    out_path = Path(raw_path or 'serving_config.json').expanduser()
    payload = build_api_config_payload(endpoint=endpoint, model=model, config=config, deploy=deploy)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized + '\n', encoding='utf-8')
    except OSError as exc:
        print(f'\nFailed to export API config JSON to {out_path}: {exc}\n')
        return

    resolved = out_path.resolve()
    print(f'\nExported API config JSON: {resolved}')
    print(serialized)
    print(
        '\nWeb demo import steps:\n'
        f'  1. Add endpoint with URL: {endpoint}\n'
        '  2. Open Edit on that endpoint -> Import serving_config.json -> select the file above.\n'
    )


def present_session(proc, output, *, endpoint, model, config, deploy):
    output.suspend()
    cfg = config.get('serving', config) if isinstance(config, dict) and isinstance(config.get('serving'), dict) else (config or {})
    command = shlex.join(['flexsds', 'chat', '--endpoint', endpoint, '--model', model])
    print('\nEndpoint LLM created:')
    for key, value in [('API', endpoint), ('Model', model), ('GPU', cfg.get('target_gpu', 'unknown')),
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
    speculation_label = format_speculation_label(config)
    while deploy or proc.poll() is None:
        choice = prompt_choice(
            'What would you like to do?',
            ['Watch log', 'Realtime chat with deployed LLM', EXPORT_ACTION],
        )
        if choice == 'Watch log':
            print(f'Chat in another terminal:\n  {command}')
            if deploy:
                # The deploy command exits after publishing; follow the persistent app separately.
                import os
                import subprocess
                from flexsds.server.modal_auth import modal_command
                cmd = [*modal_command(), 'app', 'logs',
                       output.app_url.rstrip('/').rsplit('/', 1)[-1] if output.app_url else 'flexsds-serving']
                from flexsds.cli.process_progress import ProcessProgress
                watcher = ProcessProgress('serving')
                child = None
                try:
                    child = subprocess.Popen([*cmd, '--follow'],
                                             env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'),
                                             stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding='utf-8', errors='replace', bufsize=1)
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
        if choice == EXPORT_ACTION:
            _export_api_config(endpoint=endpoint, model=model, config=config, deploy=deploy)
            continue
        run_interactive_chat(endpoint=endpoint, model=model, speculation_label=speculation_label)
        print('\nChat closed. The endpoint is still available.')
