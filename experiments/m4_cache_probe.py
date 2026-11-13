"""CPU contracts for the isolated Vanilla cold/cache-hit logit diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from experiments.m4_protocol import ROOT, WORKLOADS
from experiments.m4_treatments import TARGET, TARGET_REVISION

FROZEN_SOURCE = '187cfff5f8c2a72b9bfb227d97e6221458ddc283'
EVIDENCE_COMMIT = 'bb16ed4feb657f6c9bfb8e92fb9a89c96fa2c43f'
EVIDENCE_ROOT = ('experiments/runs/m4_02_06/'
                 'm4-02-06-a100-diagnostic-20261113T105105Z-187cfff5')
SCHEMA = 'm4.cache_probe.v1'
MEASUREMENT = 'model.compute_logits return; before grammar, penalties, temperature, sampling'


def token_hash(tokens: list[int]) -> str:
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Expected nonempty nonnegative emitted token IDs')
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode('ascii')).hexdigest()


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                    'show', f'{commit}:{path}'], cwd=ROOT)


def construct_input(captures: list[dict], raw_responses: list[dict], workload: str,
                    prompt_id: str, generation_index: int) -> dict:
    """Join emitted IDs directly. Never tokenize or decode response text."""
    if type(generation_index) is not int or generation_index < 0:
        raise ValueError('Invalid generation index')
    selected = [c for c in captures if (c['workload'], c['prompt_id'], c['treatment'], c['repeat'])
                == (workload, prompt_id, 'vanilla', 0)]
    if len(selected) != 1:
        raise ValueError('Missing/ambiguous reference capture')
    capture = selected[0]
    if (capture['source_commit'] != FROZEN_SOURCE or capture['target_model'] != TARGET
            or capture['target_revision'] != TARGET_REVISION
            or capture['tokenizer_revision'] != TARGET_REVISION
            or capture['temperature'] != 0 or capture['seed'] != 42
            or capture['complete'] is not True):
        raise ValueError('Reference identity/settings drift')
    raw = [r for r in raw_responses if r['id'] == capture['request_id']]
    if len(raw) != 1 or raw[0]['choices'][0]['token_ids'] != capture['token_ids']:
        raise ValueError('Raw emitted token capture differs')
    original = raw[0]['prompt_token_ids']
    generated = capture['token_ids']
    token_hash(original); token_hash(generated)
    if generation_index >= len(generated):
        raise ValueError('Boundary outside emitted reference')
    other = [c for c in captures if (c['workload'], c['prompt_id'], c['treatment'], c['repeat'])
             == (workload, prompt_id, 'vanilla', 1)]
    if len(other) != 1 or other[0]['token_ids'][:generation_index] != generated[:generation_index]:
        raise ValueError('Vanilla repeats do not share requested prefix')
    if generation_index >= len(other[0]['token_ids']):
        raise ValueError('Candidate lacks boundary token')
    prefix = generated[:generation_index]
    forced = original + prefix
    return dict(workload=workload, prompt_id=prompt_id, prompt_hash=capture['prompt_hash'],
        generation_index=generation_index, original_prompt_token_ids=original,
        generated_prefix_token_ids=prefix, forced_input_token_ids=forced,
        input_token_count=len(forced), input_token_ids_sha256=token_hash(forced),
        original_prompt_token_ids_sha256=token_hash(original),
        reference_next_token_id=generated[generation_index],
        candidate_next_token_id=other[0]['token_ids'][generation_index],
        historical_max_tokens=capture['max_tokens'], historical_stopping=capture['stopping'],
        tokenizer_revision=capture['tokenizer_revision'], reference_request_id=capture['request_id'])


def load_input(workload='W1', prompt_id='m0_04_001', generation_index=549) -> dict:
    material = {}
    for name in ('captures.jsonl', 'vanilla/responses.jsonl', 'readiness.json'):
        material[name] = git_blob(EVIDENCE_COMMIT, f'{EVIDENCE_ROOT}/{name}')
    recorded = json.loads(material['readiness.json'])['evidence_sha256']
    for name in ('captures.jsonl', 'vanilla/responses.jsonl'):
        if hashlib.sha256(material[name]).hexdigest() != recorded[name]:
            raise ValueError('Preserved raw evidence hash mismatch')
    captures = [json.loads(s) for s in material['captures.jsonl'].splitlines()]
    raw = [json.loads(s) for s in material['vanilla/responses.jsonl'].splitlines()]
    result = construct_input(captures, raw, workload, prompt_id, generation_index)
    workload_bytes = git_blob(FROZEN_SOURCE, WORKLOADS[workload])
    corpus = [json.loads(s) for s in workload_bytes.splitlines()]
    selected = [r for r in corpus if r['sample_id'] == prompt_id]
    if len(selected) != 1:
        raise ValueError('Missing/ambiguous frozen workload')
    # Use the original request settings, including JSON-object grammar.
    result['historical_request'] = selected[0]['request']
    expected_hash = hashlib.sha256(json.dumps(selected[0]['request'],sort_keys=True,
                                              ensure_ascii=False).encode()).hexdigest()
    if (expected_hash != result['prompt_hash']
            or selected[0]['request']['max_tokens'] != result['historical_max_tokens']):
        raise ValueError('Frozen workload/request hash differs from capture')
    result['evidence_sha256'] = {k: hashlib.sha256(v).hexdigest() for k,v in material.items()}
    return result


def cache_evidence(events: list[dict], request_id: str, condition: str,
                   original_count: int, input_count: int, input_hash: str) -> dict:
    selected = [r for r in events if r['event']=='cache_lookup' and r['request_id']==request_id]
    if not selected:
        raise ValueError('Missing actual request-local cache lookup')
    if any(r['input_token_count'] != input_count or not r['prefix_cache_enabled']
           or r['input_token_ids_sha256'] != input_hash or r['skip_reading_prefix_cache'] for r in selected):
        raise ValueError('Cache lookup input/configuration mismatch')
    counts = [r['cached_tokens'] for r in selected]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Invalid cached token counter')
    if condition == 'COLD' and any(counts):
        raise ValueError('Cold request unexpectedly reused cache')
    if condition == 'CACHE_HIT' and not 0 < max(counts) <= original_count:
        raise ValueError('Hit missing or reuses beyond conditioning original prompt')
    if condition not in ('COLD','CACHE_HIT'):
        raise ValueError('Unknown condition')
    return dict(cache_hit_verified=condition=='CACHE_HIT',cached_tokens=max(counts),
                lookup_records=selected,cache_state='empty reset' if condition=='COLD' else 'original-prompt conditioned')


def validate_result(r: dict) -> None:
    if r.get('schema_version') != SCHEMA or r.get('diagnostic_name') != 'vanilla_cold_cache_hit':
        raise ValueError('Unknown result schema')
    for key in ('source_commit','instrumentation_commit','run_id','timestamp_utc','environment',
                'model','request','condition','logits','provenance'):
        if not r.get(key):
            raise ValueError(f'Missing {key}')
    for key in ('source_commit','instrumentation_commit'):
        if not re.fullmatch(r'[0-9a-f]{40}',r[key]):
            raise ValueError('Invalid executed source SHA')
    env = r['environment']
    if (any(env.get(k) is None for k in ('gpu_name','capability','driver','cuda','python','pytorch','vllm'))
            or 'A100' not in env['gpu_name'] or env['capability'] != [8,0]
            or env['pytorch'] != '2.13.0+cu130' or env['cuda'] != '13.0'):
        raise ValueError('Incomplete/pinned environment mismatch')
    request = r['request']
    if (request.get('workload') not in WORKLOADS or not request.get('prompt_id')
            or request['input_token_count'] != request['original_prompt_token_count']+request['generation_index']
            or request['generation_index'] < 0
            or not re.fullmatch(r'[0-9a-f]{64}',request['prompt_hash'])
            or not re.fullmatch(r'[0-9a-f]{64}',r['request']['input_token_ids_sha256'])
            or not r['logits'].get('dtype')):
        raise ValueError('Invalid input/logit identity')
    if (r['model']['target'] != TARGET or r['model']['revision'] != TARGET_REVISION
            or r['model']['tokenizer_revision'] != TARGET_REVISION
            or r['model'].get('dtype') != 'bfloat16' or r['model'].get('quantization') != 'awq_marlin'
            or r['environment']['vllm'] != '0.30.0'):
        raise ValueError('Pinned identity drift')
    if (r['logits']['measurement_point'] != MEASUREMENT or r['logits']['nan_count']
            or r['logits']['inf_count']):
        raise ValueError('Invalid logit measurement')
    top = r['logits']['top_tokens']
    if len(top) < 10 or len({x['token_id'] for x in top}) != len(top):
        raise ValueError('Missing unique top-N logits')
    values = [x['logit'] for x in top]
    if not all(type(v) in (int,float) and math.isfinite(v) for v in values):
        raise ValueError('Nonfinite logit')
    if values != sorted(values, reverse=True) or top[0]['token_id'] != r['logits']['argmax_token_id']:
        raise ValueError('Argmax/top-N inconsistency')
    for field in ('logit_token_4164','logit_token_3752','difference_4164_minus_3752','top1_minus_top2'):
        if not math.isfinite(r['logits'][field]):
            raise ValueError('Nonfinite tracked logit/margin')
    if not math.isclose(r['logits']['top1_minus_top2'],values[0]-values[1],abs_tol=1e-8):
        raise ValueError('Incorrect top margin')
    if not math.isclose(r['logits']['difference_4164_minus_3752'],
                       r['logits']['logit_token_4164']-r['logits']['logit_token_3752'],abs_tol=1e-8):
        raise ValueError('Incorrect tracked margin')
    c = r['condition']
    if (not c['prefix_cache_enabled'] or c.get('reset_verified') is not True or c['name'] not in ('COLD','CACHE_HIT')
            or (c['name']=='COLD' and c['cache_evidence']['cached_tokens'] != 0)
            or (c['name']=='CACHE_HIT' and (c['cache_hit_verified'] is not True
                or not 0 < c['cache_evidence']['cached_tokens'] <= r['request']['original_prompt_token_count']))):
        raise ValueError('Unverified condition')
    records = c['cache_evidence'].get('lookup_records',[])
    if not records:
        raise ValueError('Request-local cache evidence missing')
    sampling = r['provenance'].get('sampling_config',{})
    if (sampling.get('temperature') != 0 or sampling.get('seed') != 42 or sampling.get('n') != 1
            or sampling.get('max_tokens') != {'W1':1024,'W2':96,'W3':768}[request['workload']]
            or (request['workload']=='W1' and sampling.get('structured_outputs') != {'json_object':True})):
        raise ValueError('Historical sampling contract differs')
    if records[0]['request_id'] != r['provenance'].get('observed',{}).get('request_id'):
        raise ValueError('Cache lookup and observed request ownership differs')
    provenance = r['provenance']
    if (provenance.get('query_matches_forced_suffix') is not True
            or not re.fullmatch(r'[0-9a-f]{64}',provenance.get('query_token_ids_sha256',''))
            or provenance['computed_tokens_before']+provenance['scheduled_query_tokens'] != request['input_token_count']
            or provenance['scheduled_query_tokens'] <= 0
            or provenance['computed_tokens_before'] < c['cache_evidence']['cached_tokens']):
        raise ValueError('Actual query/forced-prefix proof missing or inconsistent')
    checked = cache_evidence(records,records[0]['request_id'],c['name'],
        request['original_prompt_token_count'],request['input_token_count'],request['input_token_ids_sha256'])
    if (checked['cached_tokens'] != c['cache_evidence']['cached_tokens']
            or checked['cache_hit_verified'] != c['cache_hit_verified']):
        raise ValueError('Cache summary disagrees with raw lookup records')


def analyze(cold: dict, hit: dict, differences: dict | None) -> dict:
    """Do not invent a tolerance or equate identical argmax with equal logits."""
    reasons = []
    try:
        validate_result(cold); validate_result(hit)
    except (ValueError,KeyError,TypeError) as exc:
        reasons.append(str(exc))
    for key in ('environment','model','source_commit','instrumentation_commit'):
        if cold.get(key) != hit.get(key): reasons.append(f'{key} differs')
    for key in ('input_token_ids_sha256','input_token_count','generation_index','prompt_hash'):
        if cold.get('request',{}).get(key) != hit.get('request',{}).get(key):
            reasons.append(f'request.{key} differs')
    if cold.get('condition',{}).get('name') != 'COLD' or hit.get('condition',{}).get('name') != 'CACHE_HIT':
        reasons.append('Wrong condition order')
    if cold.get('provenance',{}).get('engine_config') != hit.get('provenance',{}).get('engine_config'):
        reasons.append('Engine flags differ')
    if cold.get('provenance',{}).get('sampling_config') != hit.get('provenance',{}).get('sampling_config'):
        reasons.append('Sampling parameters differ')
    if cold.get('provenance',{}).get('effective_config') != hit.get('provenance',{}).get('effective_config'):
        reasons.append('Resolved runtime flags differ')
    if not differences or differences.get('vocabulary_equal') is not True:
        reasons.append('Full-vector comparison unavailable')
    elif any(type(differences.get(k)) not in (int,float) or not math.isfinite(differences[k])
             or differences[k] < 0 for k in ('max_absolute_difference','mean_absolute_difference')):
        reasons.append('Invalid vector difference statistics')
    if reasons:
        return dict(classification='INCONCLUSIVE',reasons=reasons)
    same = cold['logits']['argmax_token_id'] == hit['logits']['argmax_token_id']
    changed = differences['max_absolute_difference'] != 0
    if (cold['logits']['dtype'] != hit['logits']['dtype']
            or (not changed and (not same or cold['logits']['top_tokens']!=hit['logits']['top_tokens']))):
        return dict(classification='INCONCLUSIVE',reasons=['Inconsistent vector/dtype evidence'])
    # Same top-1 with changed logits is ambiguous without a reviewed tolerance.
    classification = ('H_CACHE_SUPPORTED' if not same else
                      'INCONCLUSIVE' if changed else 'H_CACHE_NOT_REPRODUCED')
    return dict(classification=classification,argmax_equal=same,
        input_hash_equal=True,environment_equal=True,vector_difference=differences,
        top_n_overlap=len({t['token_id'] for t in cold['logits']['top_tokens']} &
                          {t['token_id'] for t in hit['logits']['top_tokens']}),
        cold_logits=cold['logits'],cache_hit_logits=hit['logits'],
        interpretation='Single controlled replay; never ROOT_CAUSE_PROVEN. Managed drift limits historical attribution.')


def analyze_replays(results: list[dict],differences: list[dict]) -> dict:
    if len(results)!=4 or len(differences)!=2:
        return dict(classification='INCONCLUSIVE',reasons=['Two complete cold/hit pairs required'])
    pairs=[analyze(results[i],results[i+1],differences[i//2]) for i in (0,2)]
    repeat_stats=[results[i].get('provenance',{}).get('repeat_vector_difference') for i in (2,3)]
    stable = all(s and s.get('vocabulary_equal') is True and s.get('max_absolute_difference')==0
                 and s.get('mean_absolute_difference')==0
                 for s in repeat_stats)
    for previous,current in ((results[0],results[2]),(results[1],results[3])):
        stable &= all(previous.get(k)==current.get(k) for k in
                      ('environment','model','source_commit','instrumentation_commit','request'))
        stable &= previous.get('logits')==current.get('logits')
        stable &= previous.get('provenance',{}).get('engine_config')==current.get('provenance',{}).get('engine_config')
        stable &= all(previous['provenance'].get(k)==current['provenance'].get(k) for k in
                      ('sampling_config','effective_config','computed_tokens_before',
                       'scheduled_query_tokens','query_token_ids_sha256'))
    if not stable or pairs[0]['classification'] != pairs[1]['classification']:
        classification='INCONCLUSIVE'
    else:
        classification=pairs[0]['classification']
    return dict(classification=classification,pairs=pairs,within_condition_exact_repeatability=stable,
                repeat_vector_differences=repeat_stats,
                interpretation='Read-only observations under forced prefill; technical review still required for historical causality.')


def verify_run_manifest(directory: Path,source_commit: str) -> None:
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    if manifest['source_commit'] != source_commit:
        raise ValueError('Run manifest executed SHA differs')
    expected = manifest['files']
    actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
              if p.is_file() and p.name!='manifest.json'}
    if expected != actual:
        raise ValueError('Run artifact hash/inventory mismatch')
