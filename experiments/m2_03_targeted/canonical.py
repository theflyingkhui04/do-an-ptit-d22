"""Reviewed canonical shell-launch contract; no runtime-selection overrides.

Only the two hash-pinned scripts are understood. Parsing their final command is
not a general shell interpreter. Changes to either script require review.
"""
import re
import shlex
from pathlib import PurePosixPath, PureWindowsPath

from .common import ROOT, decode, need, sha

RUNNER = 'vllm.v1.worker.gpu_model_runner.GPUModelRunner'
SCRIPTS = {
    'vanilla': ('scripts/launch_lane1_vanilla.sh',
                'c9e00bcca1a03d25a257f0e1377cbd39395c27f51ef389d96913b8ad64da3a5d'),
    'adaptive': ('scripts/launch_lane3_adaptive.sh',
                 '952d797c8be2407c2563f04817d27100e621f88a79a912764509b86acb803c0b'),
}
# These are the ONLY permitted departures from the shell launch environment.
# No runner, compilation, cache, device, model, or sampling setting is allowed.
INSTRUMENTATION_ENV = {
    'M203_TARGETED_CONFIG': 'Select the observation sink and reviewed boundary.',
    'PYTHONPATH': 'Make the committed observer module importable in workers.',
    'ADAPTIVE_TELEMETRY_PATH': 'Relocate the existing scheduler telemetry sink.',
}
EXPLICIT_ENV = {'PYTHONPATH', 'HF_HUB_ENABLE_HF_TRANSFER', 'HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE'}


def contract(lane):
    need(lane in SCRIPTS, 'unknown canonical lane')
    path, digest = SCRIPTS[lane]
    raw = (ROOT / path).read_bytes().replace(b'\r\n', b'\n')
    need(sha(raw) == digest, 'canonical launcher source drift: ' + path)
    text = raw.decode('utf-8')
    command = text[text.index('$VLLM_CMD Qwen/'):].replace('\\\n', '')
    if lane == 'adaptive':
        spec = re.search(r"SPEC_CONFIG='([^']+)'", text).group(1)
        command = command.replace('"$SPEC_CONFIG"', shlex.quote(spec))
    tokens = shlex.split(command)
    defaults = dict(re.findall(r'export (ADAPTIVE_\w+)="\$\{\1:-(.*?)\}"', text))
    defaults.pop('ADAPTIVE_TELEMETRY_PATH', None)
    return dict(source=path, source_sha256=digest, model=tokens[1],
                args=tokens[2:], adaptive_defaults=defaults)


def environment_view(env):
    result = {k: v for k, v in env.items() if k in EXPLICIT_ENV or k.startswith(
        ('VLLM_', 'CUDA_', 'TORCH_', 'PYTORCH_', 'NCCL_', 'ADAPTIVE_', 'M203_'))}
    need(not any(k.upper().endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))
                 for k in result), 'credential-bearing launch control; not recorded')
    return result


def script_environment(lane, inherited, project_root):
    """Implement only the exports in the hash-pinned canonical scripts."""
    spec = contract(lane)
    need('VLLM_USE_V2_MODEL_RUNNER' not in inherited,
         'canonical model-runner selection must remain unset')
    need(not any(k.startswith('M203_') for k in inherited),
         'inherited diagnostic environment forbidden in canonical observation')
    env = dict(inherited, HF_HUB_ENABLE_HF_TRANSFER='0')
    if lane == 'adaptive':
        env['PYTHONPATH'] = project_root + (':' + inherited['PYTHONPATH'] if inherited.get('PYTHONPATH') else '')
        for key, value in spec['adaptive_defaults'].items():
            env[key] = inherited.get(key) or value
        env['ADAPTIVE_TELEMETRY_PATH'] = inherited.get('ADAPTIVE_TELEMETRY_PATH') or (
            project_root + '/experiments/runs/m2_02/lane3_telemetry.jsonl')
    return env


def observation_environment(lane, inherited, project_root, observer_path, telemetry_path):
    env = script_environment(lane, inherited, project_root)
    if lane == 'vanilla':
        env['PYTHONPATH'] = project_root + (':' + inherited['PYTHONPATH'] if inherited.get('PYTHONPATH') else '')
    env['M203_TARGETED_CONFIG'] = observer_path
    if lane == 'adaptive':
        env['ADAPTIVE_TELEMETRY_PATH'] = telemetry_path
    return env


def launch_evidence(lane, inherited, env, project_root):
    spec = contract(lane)
    baseline = script_environment(lane, inherited, project_root)
    changed = {k: v for k, v in env.items() if baseline.get(k) != v}
    need(set(changed) <= set(INSTRUMENTATION_ENV) and set(baseline) <= set(env),
         'non-instrumentation canonical environment mutation')
    return dict(source=spec['source'], source_sha256=spec['source_sha256'],
                inherited_environment=environment_view(inherited),
                project_root=project_root, instrumentation_contract=dict(INSTRUMENTATION_ENV),
                instrumentation_environment=changed)


def verify_launch(lane, argv, env, proof, models, observer):
    """Independent source-derived validation, also usable on exported bundles."""
    spec = contract(lane)
    need(proof['source'] == spec['source'] and proof['source_sha256'] == spec['source_sha256'],
         'canonical launcher source provenance drift')
    need(models['target']['id'] == spec['model'], 'canonical target identity drift')
    expected = list(spec['args'])
    if lane == 'adaptive':
        i = expected.index('--speculative-config') + 1
        draft = decode(expected[i])
        need(models['draft']['id'] == draft['model'], 'canonical draft identity drift')
        draft['model'] = models['draft']['snapshot']
        # Compare JSON structurally while requiring the entire argument vector.
        actual = decode(argv[argv.index('--speculative-config') + 1])
        need(actual == draft, 'canonical speculative settings drift')
        expected[i] = argv[argv.index('--speculative-config') + 1]
    need(argv[1:] == ['-m', 'vllm.entrypoints.openai.api_server', '--model',
                          models['target']['snapshot'], *expected],
         'canonical launcher argv differs from source contract')
    path_type = PureWindowsPath if '\\' in observer['directory'] else PurePosixPath
    directory = path_type(observer['directory']).parent
    inherited = proof['inherited_environment']
    expected_env = observation_environment(lane, inherited, proof['project_root'],
        str(directory / 'observer.json'), str(directory / 'adaptive_telemetry.jsonl'))
    need(env == environment_view(expected_env), 'canonical launcher environment drift')
    checked = launch_evidence(lane, inherited, expected_env, proof['project_root'])
    need(proof == checked, 'canonical instrumentation environment drift')


def runner_identity(runner):
    cls = type(runner)
    identity = cls.__module__ + '.' + cls.__qualname__
    config = runner.vllm_config
    # Read the resolved setting, never the default or an environment guess.
    need(identity == RUNNER and config.use_v2_model_runner is False,
         'unsupported resolved model runner; V1 observation hooks required')
    return dict(model_runner=identity, use_v2_model_runner=False)


def verify_runner(proof, config):
    need(isinstance(proof, dict) and proof.get('use_v2_model_runner') is False
         and proof == dict(model_runner=RUNNER, use_v2_model_runner=False,
                       execution_sha=config['execution_sha'], lane=config['lane']),
         'unsupported/missing resolved model-runner identity')
