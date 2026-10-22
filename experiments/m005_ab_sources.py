"""Sealed, sample-005/index-162-only source composition; never launches a server."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from experiments.protocol_b import require, write_new, file_sha
from experiments.protocol_b_boundary_sources import HASHES as HISTORICAL_HASHES
from experiments.protocol_b_boundary_sources import CONTROL_DEPENDENCIES, _write_atomically
from src.server import t4_rowwise_attention as production

PROFILES = {c: 'm203.rowwise-regression-m005-i162.v1.' + c for c in ('A', 'B')}
SCHEDULER = 'project:src/server/adaptive_scheduler.py'
HASHES = dict(HISTORICAL_HASHES) | production.DEPENDENCY_HASHES | CONTROL_DEPENDENCIES | {
    SCHEDULER: 'ec43e9c0deb7aa58d47b78c8caed6a63f935c6beb8503a12912b8475eb60c391'}
OBSERVER = 'experiments/m005_ab_observer.py'
IMPORT = 'from experiments.m005_ab_observer import observe as _m005_observe\n'

# Each added hook supplies a required gate: dispatch, input/output equality,
# producer ownership, processed score identity, or committed output ownership.
# No model, kernel, sampler, scheduler, or controller operation is replaced.
BEFORE = {
    production.BACKEND: [
        ('        unified_attention(\n', '        _m005_observe("attention_pre", self, layer, query[:num_actual_tokens], key_cache, value_cache, attn_metadata, output[:num_actual_tokens])\n'),
        ('        return output\n\n    def _pth_key_value_caches(', '        _m005_observe("attention_post", self, layer, output)\n')],
    production.OPS: [
        ('    kernel_unified_attention[grid](\n', '    _m005_observe("dispatch", locals())\n')],
    'v1/worker/gpu_model_runner.py': [
        ('            model_output = self._model_forward(\n', '            _m005_observe("prepare", self, scheduler_output, input_ids, positions, attn_metadata, slot_mappings, cudagraph_mode)\n'),
        ('        # Apply structured output bitmasks if present.\n', '        _m005_observe("scores", self, scheduler_output, logits, spec_decode_metadata, grammar_output)\n'),
        ('        self._update_states_after_model_execute(\n', '        _m005_observe("sampled", self, sampler_output)\n'),
        ('                self._copy_draft_token_ids_to_cpu(scheduler_output)\n', '                _m005_observe("produced", self, scheduler_output)\n')],
    'v1/sample/sampler.py': [
        ('        if sampling_metadata.all_random:\n', '        _m005_observe("sample_scores", logits, sampling_metadata)\n')],
    'v1/sample/rejection_sampler.py': [
        ('        output_token_ids = rejection_sample(\n', '        _m005_observe("verify_scores", target_logits, sampling_metadata, metadata)\n')],
    'v1/core/sched/scheduler.py': [
        ('                    # Total computed tokens (local + external).\n', '                    _m005_observe("cache_lookup", request, num_new_local_computed_tokens, num_external_computed_tokens)\n'),
        ('            # Check for stop and update request status.\n', '            _m005_observe("account_before", request, new_token_ids, scheduler_output, output_is_stale, generated_token_ids)\n'),
        ('            elif request.pooling_params and pooler_output is not None:\n', '                _m005_observe("account", request, new_token_ids, scheduler_output, output_is_stale, generated_token_ids)\n')],
}
MODIFIED = tuple(BEFORE) + (production.DRAFT, SCHEDULER)
PROJECT_FILES = ('experiments/m005_ab.py', 'experiments/m005_ab_sources.py', OBSERVER,
                 'experiments/protocol_b_capture.py', 'experiments/protocol_b.py',
                 'experiments/lossless_verifier.py', 'experiments/workload_adapter.py',
                 'src/server/t4_rowwise_attention.py', 'src/server/budget_planner.py',
                 'src/proposer/controller.py', 'src/proposer/adapter.py',
                 'scripts/launch_lane1_vanilla.sh', 'scripts/launch_lane3_adaptive.sh', 'scripts/switch_lane.sh')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def add_import(text):
    nodes = ast.parse(text).body
    boundary = max([n.end_lineno for n in nodes if isinstance(n, ast.ImportFrom)
                    and n.module == '__future__'] + [0])
    if boundary == 0 and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant):
        boundary = nodes[0].end_lineno
    lines = text.splitlines(keepends=True)
    lines.insert(boundary, IMPORT)
    return ''.join(lines)


def base(name, raw, condition):
    require(condition in PROFILES and name in HASHES, 'Unknown M005 source/profile')
    require(sha(raw) == HASHES[name], 'Pristine M005 source drift: ' + name)
    if name == production.DRAFT or (condition == 'A' and name in production.HASHES):
        return production.transform(name, raw)
    return raw


def observe_transform(name, raw):
    text = raw.decode('utf-8')
    if name == SCHEDULER:
        anchor = '            _pb279_observe("early_schedule", self, scheduler_output)\n'
        require(text.count(anchor) == 1, 'Scheduler branch anchor drift')
        text = text.replace(anchor, anchor + '            _m005_observe("schedule", self, scheduler_output)\n')
        anchor = '        _pb279_observe("early_schedule", self, scheduler_output)\n'
        # The indented branch is deliberately excluded from the second match.
        lines = text.splitlines(keepends=True)
        require(lines.count(anchor) == 1, 'Scheduler native anchor drift')
        index = lines.index(anchor)
        lines.insert(index + 1, '        _m005_observe("schedule", self, scheduler_output)\n')
        text = ''.join(lines)
    elif name in BEFORE:
        for anchor, hook in BEFORE[name]:
            require(text.count(anchor) == 1, 'M005 hook anchor drift: ' + name)
            text = text.replace(anchor, hook + anchor, 1)
    else:
        return raw
    return add_import(text).encode()


def transform(name, raw, condition):
    result = observe_transform(name, base(name, raw, condition))
    compile(result, name, 'exec')
    return result


def source_path(name, upstream, project):
    return (project / name[8:]) if name.startswith('project:') else upstream / name


def install(upstream, project, directory, condition):
    upstream, project, directory = map(lambda p: Path(p).resolve(), (upstream, project, directory))
    require(condition in PROFILES, 'Unknown condition')
    originals = {n: source_path(n, upstream, project).read_bytes() for n in HASHES}
    expected = {n: transform(n, b, condition) for n, b in originals.items()}
    directory.mkdir(exist_ok=False)
    for name, raw in originals.items():
        path = directory / 'originals' / name.replace('project:', 'project/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    project_files = PROJECT_FILES
    for name in project_files:
        path = directory / 'project-sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / name).read_bytes())
    manifest = dict(profile=PROFILES[condition], condition=condition, upstream=str(upstream),
                    project=str(project), originals=HASHES,
                    composed={n: sha(b) for n, b in expected.items()},
                    project_hashes={n: file_sha(project / n) for n in project_files})
    write_new(directory / 'sources.json', manifest)
    for name in MODIFIED:
        _write_atomically(source_path(name, upstream, project), expected[name])
    return manifest


def authenticate(directory, *, restore=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'sources.json').read_text())
    condition = manifest['condition']
    require(manifest['profile'] == PROFILES.get(condition) and manifest['originals'] == HASHES,
            'M005 source manifest drift')
    require(set(manifest['project_hashes']) == set(PROJECT_FILES), 'Incomplete project source inventory')
    originals = {n: (directory / 'originals' / n.replace('project:', 'project/')).read_bytes() for n in HASHES}
    expected = {n: transform(n, b, condition) for n, b in originals.items()}
    require(manifest['composed'] == {n: sha(b) for n, b in expected.items()}, 'Composed manifest drift')
    upstream, project = Path(manifest['upstream']), Path(manifest['project'])
    for name, digest in manifest['project_hashes'].items():
        require(file_sha(project / name) == digest, 'Diagnostic project source drift: ' + name)
        require(file_sha(directory / 'project-sources' / name) == digest, 'Preserved project source drift')
    # Strictly reject partial installs/restores and unknown mutation, before writes.
    for name, raw in expected.items():
        require(source_path(name, upstream, project).read_bytes() == raw, 'Installed source drift: ' + name)
    if restore:
        require(not (directory / 'restored.json').exists(), 'Already restored')
        for name in MODIFIED:
            _write_atomically(source_path(name, upstream, project), originals[name])
        write_new(directory / 'restored.json', dict(profile=manifest['profile'], hashes=HASHES))
    return manifest


def diff(upstream, project, condition):
    return ''.join(''.join(difflib.unified_diff(
        (raw := source_path(n, Path(upstream), Path(project)).read_bytes()).decode().splitlines(True),
        transform(n, raw, condition).decode().splitlines(True), fromfile='a/' + n, tofile='b/' + n))
        for n in MODIFIED)
