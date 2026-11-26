"""Method recognition is separate from support; no GPU or network fixtures."""
import json
import pytest
from flexsds.core.speculative_method import detect_method, needs_checkpoint


@pytest.mark.parametrize('declared,label,family', [
    ('draft_model', 'Independent draft model', 'independent'),
    ('medusa', 'Medusa', 'heads'), ('mlp_speculator', 'MLP speculator', 'heads'),
    ('hydra', 'Hydra', 'heads'), ('mtp', 'MTP', 'heads'),
    ('EAGLE-1', 'EAGLE-1', 'feature_ar'), ('EAGLE-2', 'EAGLE-2', 'feature_ar'),
    ('EAGLE-3', 'EAGLE-3', 'feature_ar'), ('EAGLE-3.1', 'EAGLE-3.1', 'feature_ar'),
    ('EAGLE-4', 'EAGLE-4 (unverified variant)', 'unknown'),
    ('P-EAGLE', 'P-EAGLE', 'parallel'), ('dflash', 'DFlash', 'parallel'),
    ('dspark', 'DSpark', 'parallel'), ('PARD', 'PARD', 'parallel'),
    ('XPress', 'XPress', 'parallel'),
    ('LayerSkip', 'LayerSkip / early-exit self-speculation', 'self'),
    ('ngram', 'N-gram / prompt lookup', 'lookup'), ('suffix', 'Suffix decoding', 'lookup'),
    ('prompt_lookup', 'Prompt lookup', 'lookup'), ('retrieval', 'Retrieval drafting', 'lookup'),
    ('jacobi', 'Jacobi / parallel iterative decoding', 'iterative'),
])
def test_explicit_method_metadata(declared, label, family):
    result = detect_method({'speculators_config': {'algorithm': declared}})
    assert result.label == label
    assert result.family == family
    assert not result.verified  # A declaration alone never validates a checkpoint.


@pytest.mark.parametrize('architecture,method', [
    ('MedusaModel', 'medusa'), ('HydraModel', 'hydra'), ('MLPSpeculatorPreTrainedModel', 'mlp_speculator'),
    ('DeepSeekMTPModel', 'mtp'), ('DFlashDraftModel', 'dflash'), ('Qwen3DSparkModel', 'dspark'),
    ('PARDForCausalLM', 'pard'), ('XPressDraftModel', 'xpress'), ('PEagleModel', 'peagle'),
    ('LayerSkipModel', 'layerskip'), ('Eagle4Model', 'eagle4'),
    ('Eagle2LlamaForCausalLM', 'eagle2'), ('Eagle1LlamaForCausalLM', 'eagle1'),
])
def test_known_architecture_identification(architecture, method):
    result = detect_method({'architectures': [architecture]})
    assert result.method == method


def test_causal_lm_is_identified_even_if_not_supported_for_serving():
    result = detect_method({'architectures': ['Qwen3ForCausalLM'], 'num_hidden_layers': 28})
    assert result.method == 'draft_model'
    assert result.family == 'independent'
    assert 'inference-only' in result.evidence


def test_unknown_declaration_is_not_silently_changed_to_eagle():
    result = detect_method({'architectures': ['Eagle3Model'], 'speculators_config': {'algorithm': 'new_future_method'}})
    assert result.family == 'unknown'
    assert result.method == 'new_future_method'
    assert not needs_checkpoint({}, result)


def test_conflicting_metadata_blocks_identification():
    result = detect_method({'speculators_config': {'algorithm': 'dflash'}, 'draft_method': 'eagle3'})
    assert result.issue
    assert result.method == 'unknown'


@pytest.mark.parametrize('method,shape', [('eagle1', [4, 12]), ('eagle3', [4, 8])])
def test_wrong_declared_version_is_not_accepted(method, shape):
    result = detect_method({'hidden_size': 4, 'speculative_method': method}, {'fc.weight': {'shape': shape}})
    assert result.issue and not result.verified


def test_parallel_eagle_is_not_misclassified_as_sequential_eagle():
    config = dict(architectures=['Eagle3Model'], hidden_size=4, parallel_drafting=True)
    assert detect_method(config).method == 'peagle'
    config.pop('parallel_drafting')
    result = detect_method(config, {'fc.weight': {'shape': [4, 12]}, 'mask_hidden': {'shape': [4]}})
    assert result.method == 'peagle'
    assert not result.verified


def test_parallel_regular_model_is_pard():
    result = detect_method(dict(architectures=['Qwen3ForCausalLM'], parallel_drafting=True))
    assert result.method == 'pard'


@pytest.mark.parametrize('field,method', [('num_nextn_predict_layers', 'mtp'), ('medusa_num_heads', 'medusa'), ('assistant_early_exit', 'layerskip')])
def test_dedicated_config_fields(field, method):
    assert detect_method({field: 2, 'architectures': ['LlamaForCausalLM']}).method == method


def test_external_rest_does_not_establish_non_neural_implementation():
    result = detect_method({'speculative_method': 'external_rest'})
    assert result.method == 'external'
    assert result.family == 'unknown'


def test_inspect_policy_cli_does_not_load_model_or_detect_gpu(monkeypatch, capsys):
    from flexsds.cli.main import main
    monkeypatch.setattr('flexsds.core.model_inspection.ModelFiles.__init__', lambda *a: pytest.fail('Policy lookup loaded a model'))
    monkeypatch.setattr('flexsds.cli.main.detect_hardware', lambda: pytest.fail('Inspection requires no GPU'))
    assert main(['inspect', '--method', 'jacobi']) == 0
    output = capsys.readouterr().out
    assert 'Iterative neural decoding' in output
    assert 'Jacobi' in output
    assert 'Not supported yet' in output


@pytest.mark.parametrize('args', [['inspect'], ['inspect', 'repo/name', '--method', 'ngram']])
def test_inspect_requires_model_or_policy(args):
    from flexsds.cli.main import main
    assert main(args) == 2


def test_inspect_regular_model_cli(tmp_path, monkeypatch, capsys):
    from flexsds.cli.main import main
    (tmp_path / 'config.json').write_text(json.dumps({'architectures': ['Qwen3ForCausalLM']}))
    monkeypatch.setattr('flexsds.cli.main.detect_hardware', lambda: pytest.fail('Inspection requires no GPU'))
    assert main(['inspect', str(tmp_path)]) == 0
    assert 'Independent draft model' in capsys.readouterr().out


@pytest.mark.parametrize('version', ['1', '2'])
@pytest.mark.parametrize('adaptive', [False, True])
def test_legacy_eagle_uses_eagle_backend_in_static_and_adaptive(version, adaptive):
    from flexsds.server.local_runner import LocalServingRunner
    detected = detect_method(dict(eagle_version=version, hidden_size=4), {'fc.weight': {'shape': [4, 8]}})
    cmd = LocalServingRunner().build_command(dict(draft_method=detected.backend_method, enable_adaptive=adaptive))
    spec = json.loads(cmd[cmd.index('--speculative-config') + 1])
    assert spec['method'] == 'eagle'
    assert ('--scheduler-cls' in cmd) == adaptive
    assert ('num_speculative_tokens_per_batch_size' in spec) == adaptive


@pytest.mark.parametrize('method', ['eagle', 'eagle3'])
def test_lineage_contract_accepts_both_eagle_backends_without_gpu(method):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/server/lineage_executor.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_speculator_contract')
    calls = []
    scope = {'verify_eagle_sources': lambda: calls.append('shared'),
             'verify_legacy_eagle_sources': lambda: calls.append('legacy')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'contract', 'exec'), scope)
    spec = SimpleNamespace(method=method, parallel_drafting=False, num_speculative_tokens_per_batch_size=[[1,32,4]])
    scope['validate_speculator_contract'](spec)
    assert calls == (['shared', 'legacy'] if method == 'eagle' else ['shared'])
    spec.parallel_drafting = True
    with pytest.raises(RuntimeError, match='sequential dynamic-K'):
        scope['validate_speculator_contract'](spec)


@pytest.mark.parametrize('tampered', [False, True])
def test_legacy_source_gate_checks_actual_manifest_bytes(tmp_path, tampered):
    import ast
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1] / 'src/server/lineage_executor.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_legacy_eagle_sources')
    project = tmp_path / 'project'
    site = tmp_path / 'site'
    checkpoint_loader = site / 'vllm/model_executor/models/llama_eagle.py'
    checkpoint_loader.parent.mkdir(parents=True)
    checkpoint_loader.write_bytes(b'pinned loader source')
    manifest = project / 'experiments/eagle_legacy_source_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(files=[dict(path='vllm/model_executor/models/llama_eagle.py',
                                                   sha256=hashlib.sha256(checkpoint_loader.read_bytes()).hexdigest())])))
    if tampered:
        checkpoint_loader.write_bytes(b'modified loader')
    scope = dict(Path=Path, json=json, hashlib=hashlib,
                 __file__=str(project / 'src/server/lineage_executor.py'),
                 vllm=SimpleNamespace(__file__=str(site / 'vllm/__init__.py')))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy-source-gate', 'exec'), scope)
    if tampered:
        with pytest.raises(RuntimeError, match='legacy EAGLE source'):
            scope['verify_legacy_eagle_sources']()
    else:
        scope['verify_legacy_eagle_sources']()


@pytest.mark.parametrize('declared,label', [('medusa','Medusa'), ('hydra','Hydra'), ('mtp','MTP'),
    ('dflash','DFlash'), ('dspark','DSpark'), ('pard','PARD'), ('xpress','XPress'),
    ('layerskip','LayerSkip'), ('ngram','N-gram'), ('jacobi','Jacobi')])
def test_unsupported_serving_report_preserves_identified_method(tmp_path, monkeypatch, declared, label):
    from flexsds.core.compatibility import check_compatibility
    (tmp_path / 'config.json').write_text(json.dumps({'speculative_method': declared}))
    monkeypatch.setattr('flexsds.core.model_inspection.ModelFiles.tensor_headers', lambda *a: pytest.fail('Unsupported method must not load checkpoint metadata'))
    report = check_compatibility('unused/target', str(tmp_path))
    assert report.identified_method == declared
    assert label in report.summary()
    assert not report.method_supported
    assert not report.is_compatible
    assert 'Not supported yet' in report.blocking_reasons[0]


def test_recognition_does_not_claim_missing_legacy_qwen_runtime_loader():
    from flexsds.core.speculative_method import method_support
    config = dict(architectures=['Qwen2ForCausalLM'], num_hidden_layers=1, hidden_size=4)
    detected = detect_method(config, {'fc.weight': {'shape': [4, 8]}})
    assert detected.verified
    assert detected.backend_method == 'eagle'
    supported, reason = method_support(detected, config)
    assert not supported
    assert 'no verified loader' in reason


def test_invalid_pairing_cannot_start_calibration(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from flexsds.core.cost_model import run_hardware_micro_benchmark
    from flexsds.core.hardware import HardwareInfo
    report = SimpleNamespace(draft_method='eagle', is_compatible=False, blocking_reasons=['Tokenizer mismatch'])
    monkeypatch.setattr('flexsds.core.compatibility.check_compatibility', lambda *a: report)
    monkeypatch.setattr('flexsds.server.modal_calibration.calibrate', lambda *a: pytest.fail('Invalid pair allocated GPU calibration'))
    with pytest.raises(ValueError, match='Tokenizer mismatch'):
        run_hardware_micro_benchmark(HardwareInfo('Remote A10','sm_86',0,0,'',True), target_dir=tmp_path, backend='modal')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('version', ['1', '2'])
def test_explicit_shared_checkpoint_selection_has_exact_label(version):
    from flexsds.core.speculative_method import select_eagle_version
    result = detect_method(dict(architectures=['LlamaForCausalLM'], num_hidden_layers=1, hidden_size=4),
                           {'fc.weight': {'shape': [4, 8]}})
    selected = select_eagle_version(result, version)
    assert selected.label == f'EAGLE-{version}'
    assert selected.backend_method == 'eagle'
    assert 'user selected' in selected.evidence


def test_selection_cannot_relabel_eagle3_as_legacy():
    from flexsds.core.speculative_method import select_eagle_version
    result = detect_method(dict(architectures=['Eagle3Model'], hidden_size=4), {'fc.weight': {'shape': [4,12]}})
    with pytest.raises(ValueError, match='conflicts'):
        select_eagle_version(result, '2')


def test_report_version_selection_preserves_failed_pairing():
    from flexsds.core.compatibility import CompatibilityReport, select_report_eagle_version
    report = CompatibilityReport('target', 'draft', 'eagle', False, 1, 1, 1, 3, 23000, False, False,
                                 identified_method='eagle', checkpoint_verified=True,
                                 blocking_reasons=['unverified pairing'])
    selected = select_report_eagle_version(report, '2')
    assert selected.method_label == 'EAGLE-2'
    assert selected.eagle_version == 'EAGLE-2'
    assert not selected.is_compatible
    assert selected.blocking_reasons == ['unverified pairing']
