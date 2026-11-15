"""Evidence-based method identification, independent of FlexSDS runtime support.

Inference-only algorithms (lookup, early exit, Jacobi, EAGLE-2 tree policy) do
not necessarily leave a unique checkpoint signature. Unknown stays unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


FAMILIES = {
    'independent': 'Independent draft model',
    'heads': 'Target-attached prediction heads',
    'feature_ar': 'Autoregressive feature drafting',
    'parallel': 'Parallel / block drafting',
    'self': 'Self-speculative decoding',
    'lookup': 'Non-neural lookup / retrieval',
    'iterative': 'Iterative neural decoding',
    'unknown': 'Unknown',
}

# Exact normalized declarations, not substring matches against repository IDs.
METHODS = {
    'draft_model': ('independent', 'Independent draft model'),
    'medusa': ('heads', 'Medusa'),
    'mlp_speculator': ('heads', 'MLP speculator'),
    'hydra': ('heads', 'Hydra'),
    'mtp': ('heads', 'MTP'),
    'eagle': ('feature_ar', 'EAGLE (version selection required)'),
    'eagle1': ('feature_ar', 'EAGLE-1'),
    'eagle2': ('feature_ar', 'EAGLE-2'),
    'eagle3': ('feature_ar', 'EAGLE-3'),
    'eagle3_1': ('feature_ar', 'EAGLE-3.1'),
    'eagle4': ('unknown', 'EAGLE-4 (unverified variant)'),
    'peagle': ('parallel', 'P-EAGLE'),
    'dflash': ('parallel', 'DFlash'),
    'dspark': ('parallel', 'DSpark'),
    'pard': ('parallel', 'PARD'),
    'xpress': ('parallel', 'XPress'),
    'layerskip': ('self', 'LayerSkip / early-exit self-speculation'),
    'self_speculative': ('self', 'Self-speculative decoding'),
    'ngram': ('lookup', 'N-gram / prompt lookup'),
    'prompt_lookup': ('lookup', 'Prompt lookup'),
    'suffix': ('lookup', 'Suffix decoding'),
    'retrieval': ('lookup', 'Retrieval drafting'),
    # A REST endpoint does not establish whether its drafter uses a neural net.
    'external': ('unknown', 'External proposer (neural implementation unknown)'),
    'jacobi': ('iterative', 'Jacobi / parallel iterative decoding'),
}
ALIASES = {
    'eagle_1': 'eagle1', 'eagle_2': 'eagle2', 'eagle_3': 'eagle3',
    'eagle_3_1': 'eagle3_1', 'eagle31': 'eagle3_1', 'eagle_4': 'eagle4',
    'p_eagle': 'peagle', 'parallel_eagle': 'peagle',
    'mlpspeculator': 'mlp_speculator', 'multi_token_prediction': 'mtp',
    'ngram_gpu': 'ngram', 'n_gram': 'ngram', 'n_grams': 'ngram', 'prompt_lookup_decoding': 'prompt_lookup',
    'layer_skip': 'layerskip', 'early_exit': 'layerskip',
    'self_speculative_decoding': 'self_speculative', 'jacobi_decoding': 'jacobi',
    'rest': 'external', 'external_rest': 'external',
}


def normalize_method(value):
    value = re.sub(r'[-.\s]+', '_', str(value).strip().lower())
    return ALIASES.get(value, value)


@dataclass(frozen=True)
class MethodDetection:
    method: str
    label: str
    family: str
    evidence: str
    verified: bool = False
    issue: str = ''

    @property
    def family_label(self):
        return FAMILIES[self.family]

    @property
    def backend_method(self):
        if self.method in {'eagle', 'eagle1', 'eagle2'}:
            return 'eagle'
        if self.method in {'eagle3', 'eagle3_1'}:
            return 'eagle3'
        return self.method


def _result(method, evidence, **kwargs):
    family, label = METHODS.get(method, ('unknown', f'Unknown method ({method})'))
    return MethodDetection(method, label, family, evidence, **kwargs)


def _declarations(config):
    declarations = []
    for key in ('speculators_config', 'speculative_config'):
        nested = config.get(key) or {}
        if isinstance(nested, dict):
            value = nested.get('algorithm') or nested.get('method')
            if value:
                declarations.append((normalize_method(value), key))
    for key in ('speculative_method', 'speculators_model_type', 'draft_method'):
        if config.get(key):
            declarations.append((normalize_method(config[key]), key))
    if config.get('eagle_version'):
        declarations.append((normalize_method('eagle' + str(config['eagle_version'])), 'eagle_version'))
    return declarations


def detect_method(config, tensors=None):
    """Identify declared algorithms or recognized architecture/checkpoint layouts.

    A label can be recognized without being executable. `verified` is reserved
    for a supported EAGLE tensor layout, and is not an inference-success claim.
    """
    declared = _declarations(config)
    methods = {method for method, _ in declared}
    # Generic family declarations may coexist with a more specific version.
    if len(methods) > 1:
        if methods <= {'eagle', 'eagle1', 'eagle2'}:
            methods.discard('eagle')
        elif methods <= {'eagle3', 'eagle3_1'}:
            methods.discard('eagle3')
        else:
            return _result('unknown', 'Conflicting method declarations', issue='Conflicting speculative method metadata')
    if len(methods) > 1:
        return _result('unknown', 'Conflicting EAGLE versions', issue='Conflicting speculative method metadata')
    method = next(iter(methods), None)
    evidence = ', '.join(key for _, key in declared)
    arch = ' '.join(config.get('architectures') or []).lower()
    model_type = str(config.get('model_type', '')).lower()
    identity = arch + ' ' + model_type
    # Explicit parallel/head types take precedence over a shared EAGLE backbone.
    architecture_method = None
    for marker, candidate in (
        ('xpress', 'xpress'), ('dspark', 'dspark'), ('dflash', 'dflash'),
        ('peagle', 'peagle'), ('p_eagle', 'peagle'), ('pard', 'pard'),
        ('medusa', 'medusa'), ('mlpspeculator', 'mlp_speculator'),
        ('mlp_speculator', 'mlp_speculator'), ('hydra', 'hydra'),
        ('mtp', 'mtp'), ('layerskip', 'layerskip'), ('eagle4', 'eagle4'),
        ('eagle31', 'eagle3_1'), ('eagle3_1', 'eagle3_1'), ('eagle3', 'eagle3'),
        ('eagle2', 'eagle2'), ('eagle_2', 'eagle2'),
        ('eagle1', 'eagle1'), ('eagle_1', 'eagle1'), ('eagle', 'eagle'),
    ):
        if marker in identity:
            architecture_method = candidate
            break
    if method is None:
        method = architecture_method
        evidence = 'architectures / model_type' if method else ''
    if method is None:
        for field, candidate in (('dspark_config', 'dspark'), ('dflash_config', 'dflash'),
                                 ('medusa_config', 'medusa'), ('hydra_config', 'hydra')):
            if isinstance(config.get(field), dict):
                method, evidence = candidate, field
                break
    if method is None:
        for field, candidate in (('medusa_num_heads', 'medusa'), ('num_medusa_heads', 'medusa'),
                                 ('hydra_num_heads', 'hydra'), ('num_nextn_predict_layers', 'mtp'),
                                 ('num_mtp_layers', 'mtp'), ('assistant_early_exit', 'layerskip')):
            if isinstance(config.get(field), int) and not isinstance(config[field], bool) and config[field] > 0:
                method, evidence = candidate, field
                break
    parallel = config.get('parallel_drafting') is True or (config.get('speculative_config') or {}).get('parallel_drafting') is True
    if parallel:
        if method in {'eagle', 'eagle1', 'eagle2', 'eagle3', 'eagle3_1'}:
            method = 'peagle'
        elif method == 'draft_model' or method is None and any(a.lower().endswith('forcausallm') for a in config.get('architectures') or []):
            method = 'pard'
        evidence += ', parallel_drafting'
    if method and method not in {'eagle', 'eagle1', 'eagle2', 'eagle3', 'eagle3_1'}:
        return _result(method, evidence or 'method metadata')

    # Inspect ambiguous one-layer causal heads further, including legacy Qwen.
    possible_head = (config.get('num_hidden_layers') == 1 and any(a.lower().endswith('forcausallm') for a in config.get('architectures') or []))
    if method is None and not possible_head and 'draft_vocab_size' not in config:
        if any(a.lower().endswith('forcausallm') for a in config.get('architectures') or []):
            return _result('draft_model', 'Causal LM architecture; inference-only policies cannot be inferred')
        return _result('unknown', 'No recognized method metadata or checkpoint signature')
    if tensors is None:
        return _result(method or 'unknown', evidence or 'Potential feature head; checkpoint inspection required')

    if any(k.endswith('mask_hidden') for k in tensors):
        return _result('peagle', 'Parallel EAGLE mask_hidden checkpoint tensor')
    hidden = config.get('hidden_size', 0)
    fc = next((v for k, v in tensors.items() if k in {'fc.weight', 'model.fc.weight'}), {})
    shape = fc.get('shape', [])
    if hidden and shape == [hidden, hidden * 2]:
        if method in {'eagle3', 'eagle3_1'}:
            return _result(method, 'fc.weight shape contradicts metadata', issue='EAGLE version conflicts with checkpoint shape')
        return _result(method or 'eagle', '2H feature projection; EAGLE-1/2 share weights', verified=True)
    if hidden and shape == [hidden, hidden * 3]:
        if method in {'eagle1', 'eagle2'}:
            return _result(method, 'fc.weight shape contradicts metadata', issue='EAGLE version conflicts with checkpoint shape')
        variant31 = all(config.get(k) is True for k in ('fc_norm', 'norm_output', 'use_post_norm_feedback'))
        if method == 'eagle3_1' and not variant31:
            return _result(method, 'Incomplete EAGLE-3.1 configuration', issue='EAGLE-3.1 configuration is incomplete')
        if variant31:
            for i in range(3):
                if not any(k.endswith(f'fc_norm.{i}.weight') and v.get('shape') == [hidden]
                           for k, v in tensors.items()):
                    return _result('eagle3_1', 'Missing normalization weights', issue='EAGLE-3.1 checkpoint lacks normalization weights')
        return _result('eagle3_1' if variant31 else 'eagle3', '3H feature projection and configuration verified', verified=True)
    if method:
        return _result(method, evidence, issue='Cannot verify EAGLE checkpoint format')
    if any(a.lower().endswith('forcausallm') for a in config.get('architectures') or []):
        return _result('draft_model', 'Causal LM checkpoint without a feature-drafting projection')
    return _result('unknown', 'Checkpoint has no supported feature projection')


def needs_checkpoint(config, detection):
    identity = ' '.join(config.get('architectures') or []).lower()
    return (detection.family == 'feature_ar' and detection.method != 'eagle4'
            or detection.method == 'unknown' and not detection.issue and
            ('draft_vocab_size' in config or config.get('num_hidden_layers') == 1 and any(a.lower().endswith('forcausallm') for a in config.get('architectures') or [])))


def inspect_method(model, *, with_config=False, eagle_version=None):
    """Inspect one model without requiring a target, a GPU or a serving backend."""
    from flexsds.core.model_inspection import ModelFiles
    files = ModelFiles(model)
    config = files.config()
    detection = detect_method(config)
    if needs_checkpoint(config, detection):
        detection = detect_method(config, files.tensor_headers())
    if eagle_version:
        detection = select_eagle_version(detection, eagle_version)
    return (detection, config) if with_config else detection


def method_support(detection, config):
    """Pinned vLLM loader support, distinct from identifying the algorithm."""
    if detection.backend_method not in {'eagle', 'eagle3'}:
        return False, f'Not supported yet: {detection.label} ({detection.family_label})'
    if detection.issue or not detection.verified:
        return False, detection.issue or f'Cannot verify {detection.label} checkpoint format'
    architectures = set(config.get('architectures') or [])
    legacy = {'LlamaForCausalLM', 'EagleLlamaForCausalLM', 'Llama4ForCausalLM', 'EagleLlama4ForCausalLM'}
    modern = {'LlamaForCausalLM', 'Eagle3LlamaForCausalLM', 'LlamaForCausalLMEagle3',
              'Eagle3Speculator', 'Eagle3MiniMaxM2ForCausalLM',
              'Eagle3Qwen2_5vlForCausalLM', 'Eagle3Qwen3vlForCausalLM'}
    supported_architectures = legacy if detection.backend_method == 'eagle' else modern
    if not architectures or not architectures <= supported_architectures:
        return False, (f'Not supported yet: {detection.label} checkpoint architecture '
                       f'{sorted(architectures)} has no verified loader in the pinned FlexSDS runtime')
    return True, ''


def select_eagle_version(detection, version):
    """Resolve shared legacy weights by explicit user selection, never by guessing."""
    from dataclasses import replace
    requested = normalize_method('eagle' + version)
    if requested not in {'eagle1', 'eagle2', 'eagle3', 'eagle3_1'}:
        raise ValueError('Expected EAGLE version 1, 2, 3 or 3.1')
    if detection.method == 'eagle' and detection.verified and requested in {'eagle1', 'eagle2'}:
        return replace(detection, method=requested, label=METHODS[requested][1],
                       evidence=detection.evidence + f'; user selected EAGLE-{version}')
    if detection.method != requested:
        raise ValueError(f'Selected EAGLE-{version} conflicts with detected {detection.label}')
    return detection
