"""Empirical scalar component calibration. Each K uses a fresh engine process.

Profile semantics: target forward + complete draft proposal, excluding target
sampling/logits and scheduler overhead. These are eager component costs, not
an end-to-end speedup guarantee. Raw warmup and measured repeats are preserved.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

SCHEMA = 'flexsds-component-v2'
BATCHES = [1, 2, 4, 8, 16, 32]
KS = [0, 1, 2, 3, 4]
CONTEXT = 1024
WARMUP = 5
REPEATS = 50


def profile_batch_coverage(batches, max_num_seqs):
    """Top measured batch when the grid stops below the serving batch, else None.

    The grid must be a prefix of BATCHES (1, 2, 4, ...). A shorter grid is valid: on T4, B=16/32
    do not fit the KV cache, so calibration measures 1..8 and larger batches use the nearest
    measured batch (EmpiricalCostProfile.get_cost).
    """
    measured = sorted(set(int(b) for b in batches))
    if not measured or measured != BATCHES[:len(measured)]:
        raise ValueError(f'Profile batch grid {measured} must be a prefix of {BATCHES}; recalibrate')
    if not 1 <= int(max_num_seqs) <= BATCHES[-1]:
        raise ValueError(f'Adaptive profile supports at most {BATCHES[-1]} concurrent sequences')
    return measured[-1] if measured[-1] < int(max_num_seqs) else None


def measurement_settings(config):
    mode = config.get('measurement_mode', 'full')
    if mode not in ('fast', 'full'):
        raise ValueError('Measurement mode must be fast or full')
    return (2, 5) if mode == 'fast' else (WARMUP, REPEATS)


def resolve_calibration_dtype_quant(target_model: str, sm_architecture: str):
    """Select vLLM dtype/quantization compatible with the target GPU architecture."""
    target_lower = target_model.lower()
    is_turing = str(sm_architecture or '').strip().lower() == 'sm_75'
    if 'awq' in target_lower:
        return ('float16', 'awq') if is_turing else ('bfloat16', 'awq_marlin')
    if 'gptq' in target_lower:
        return ('float16', 'gptq')
    return ('float16', None) if is_turing else ('bfloat16', None)


def check_calibration_device(config, device):
    """Modal GPUs must be exactly the requested card (A10 never A10G); other targets keep the lenient label match."""
    from flexsds.core.hardware import (MODAL_GPU_NAMES, canonical_modal_gpu_name, device_matches_exactly,
                                       matches_gpu_label)
    expected = config.get('target_gpu')
    if not expected:
        return
    token = canonical_modal_gpu_name(expected)
    if token in MODAL_GPU_NAMES:
        if not device_matches_exactly(token, device):
            wanted = {'T4': 'Tesla T4', 'A100': 'NVIDIA A100 80GB'}.get(token, f'NVIDIA {token}')
            raise RuntimeError(f'GPU mismatch: expected {wanted}, got {device}')
    elif not matches_gpu_label(expected, device):
        raise RuntimeError(f'GPU mismatch: expected {expected}, got {device}')


DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


def calibration_engine_kwargs(config, batches, *, revision, dtype, quantization):
    """vLLM engine settings for one calibration group.

    Every sequence must be prefilled in one step (no chunked prefill), so the next step has the exact
    decode shape that is timed. At B=32 x 1024 that prefill's activations leave a 16 GB T4 too little
    KV cache at the default memory share; `gpu_memory_utilization` in the config (recorded in the
    profile evidence) raises it. On T4, 0.95 gave 34.8k KV tokens but still ran out of memory at B=16
    (2.3 GiB request, 3.3 GiB reserved-but-unallocated) on 27/11, so the T4 profile stays at B<=8.
    """
    share = float(config.get('gpu_memory_utilization', DEFAULT_GPU_MEMORY_UTILIZATION))
    if not 0.5 <= share <= 0.98:
        raise ValueError(f'gpu_memory_utilization must be within 0.50..0.98, got {share}')
    cfg = dict(model=config['target_model'], revision=revision, dtype=dtype, max_model_len=4096,
               max_num_seqs=32, max_num_batched_tokens=max(4096, max(batches)*CONTEXT),
               gpu_memory_utilization=share, enforce_eager=True, generation_config='vllm',
               compilation_config={'mode': 0},
               enable_prefix_caching=False, enable_chunked_prefill=False,
               async_scheduling=False,
               worker_extension_cls='flexsds.core.calibration_worker.CalibrationWorker')
    if quantization:
        cfg['quantization'] = quantization
    return cfg


def measure_cell_group(config, k, batches):
    """Runs only inside an isolated CUDA process (local or Modal)."""
    warmup, repeats = measurement_settings(config)
    import torch
    import vllm
    if not torch.cuda.is_available():
        raise RuntimeError('Calibration requires a real CUDA GPU')
    sm = 'sm_%d%d' % torch.cuda.get_device_capability(0)
    if sm != config['sm_architecture']:
        raise RuntimeError(f"GPU mismatch: expected {config['sm_architecture']}, got {sm}")
    device = torch.cuda.get_device_name(0)
    check_calibration_device(config, device)
    if vllm.__version__ != '0.30.0':
        raise RuntimeError('Calibration requires vLLM 0.30.0')
    from huggingface_hub import HfApi
    target_revision = config.get('target_revision') or HfApi().model_info(config['target_model']).sha
    draft_revision = config.get('draft_revision') or HfApi().model_info(config['draft_model']).sha
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    dtype, quantization = resolve_calibration_dtype_quant(config['target_model'], sm)
    cfg = calibration_engine_kwargs(config, batches, revision=target_revision, dtype=dtype,
                                    quantization=quantization)
    # Keep the drafter loaded at K=0: EAGLE still synchronizes its KV cache.
    cfg['speculative_config'] = dict(method=config['draft_method'], model=config['draft_model'],
                                     num_speculative_tokens=4, draft_sample_method='greedy',
                                     revision=draft_revision,
                                     num_speculative_tokens_per_batch_size=[[1,32,k]])
    llm = LLM(**cfg)
    llm.collective_rpc('calibration_install')
    records = []
    for batch in batches:
        prompts = [TokensPrompt(prompt_token_ids=[1000 + (i+j) % 28000 for j in range(CONTEXT)])
                   for i in range(batch)]
        for repeat in range(warmup + repeats):
            llm.collective_rpc('calibration_begin', args=(batch, k, CONTEXT))
            llm.sleep(level=0)
            llm.enqueue(prompts, SamplingParams(max_tokens=2*k+4, temperature=0., ignore_eos=True), use_tqdm=False)
            llm.wake_up(tags=['scheduling'])
            llm.wait_for_completion(use_tqdm=False)
            [result] = llm.collective_rpc('calibration_finish')
            samples = result['samples']
            for key in ['target', 'draft']:
                if not positive(samples.get(key)):
                    raise RuntimeError(f'No exact measured {key} sample: K={k}, B={batch}')
            records.append(dict(k=k, batch=batch, context=CONTEXT, repeat=repeat,
                                warmup=repeat < warmup, **result))
        print(f'Calibration measured K={k}, B={batch}', flush=True)
    return dict(device=device, sm_architecture=sm, vram_total_mb=torch.cuda.get_device_properties(0).total_memory / 2**20,
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                vllm_version=vllm.__version__, target_revision=target_revision, draft_revision=draft_revision,
                engine_kwargs=cfg, records=records)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def build_profile(config, groups, batches):
    """Reject incomplete, duplicate, mismatched or non-finite observations."""
    warmup, repeats = measurement_settings(config)
    matrix, raw = {}, []
    devices = set()
    revisions = set()
    if batches != sorted(set(batches)) or not batches or not set(batches) <= set(BATCHES):
        raise ValueError('Invalid batch grid')
    for k, group in zip(KS, groups):
        if group['sm_architecture'] != config['sm_architecture']:
            raise ValueError('Calibration hardware mismatch')
        try:
            check_calibration_device(config, group['device'])
        except RuntimeError as exc:
            raise ValueError('Calibration GPU model mismatch') from exc
        devices.add(group['device'])
        revisions.add((group.get('target_revision'), group.get('draft_revision')))
        if group.get('vllm_version') != '0.30.0':
            raise ValueError('Unpinned calibration runtime')
        if not all(isinstance(v, str) and len(v) == 40 for v in (group.get('target_revision'), group.get('draft_revision'))):
            raise ValueError('Missing resolved model revisions')
        records = group['records']
        expected = {(k, b, r) for b in batches for r in range(warmup + repeats)}
        actual = {(r['k'], r['batch'], r['repeat']) for r in records}
        if actual != expected or len(records) != len(expected):
            raise ValueError('Incomplete/duplicate calibration samples')
        for r in records:
            if r['context'] != CONTEXT or r['warmup'] != (r['repeat'] < warmup):
                raise ValueError('Calibration context/warmup mismatch')
            if not any(s == dict(batch=r['batch'], query=[k+1]*r['batch'],
                                 context=[CONTEXT]*r['batch']) for s in r['shapes']):
                raise ValueError('Missing exact observed worker shape')
            if not all(positive(r['samples'].get(key)) for key in ['target','draft']):
                raise ValueError('Invalid measured latency')
        for b in batches:
            costs = [r['samples']['target'] + r['samples']['draft']
                     for r in records if r['batch'] == b and not r['warmup']]
            value = statistics.median(costs)
            if not positive(value):
                raise ValueError('Invalid aggregate component latency')
            matrix[f'k{k}_b{b}'] = value
        raw.extend(records)
    if len(groups) != len(KS) or len(devices) != 1 or len(revisions) != 1:
        raise ValueError('Missing K group or mixed devices')
    evidence = dict(config=config, groups=groups)
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(metadata=dict(schema=SCHEMA, measurement_status='measured', scalar_only=True,
                              device=groups[0]['device'], sm_architecture=config['sm_architecture'],
                              provenance='worker_cuda_synchronize_perf_counter', raw_sha256=digest,
                              timestamp_utc=datetime.now(timezone.utc).isoformat(),
                              cost_semantics='eager_target_forward_plus_draft_proposal_including_k0_cache_sync_excludes_target_logits_sampling',
                              vllm_version='0.30.0', measurement_mode=config.get('measurement_mode', 'full'),
                              warmup=warmup, repeats=repeats),
                models=dict(target=config['target_model'], draft=config['draft_model'],
                            target_revision=groups[0]['target_revision'], draft_revision=groups[0]['draft_revision']),
                draft_method=config['draft_method'], grid=dict(batches=batches, candidate_budgets=KS, contexts=[CONTEXT]),
                cost_matrix_ms=matrix, evidence=evidence)


def run_local_group(config, k, batches):
    with tempfile.TemporaryDirectory(prefix='flexsds-calibration-') as directory:
        request = Path(directory) / 'request.json'
        output = Path(directory) / 'result.json'
        request.write_text(json.dumps(dict(config=config, k=k, batches=batches)))
        env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER='0')
        package_root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
        try:
            subprocess.run([sys.executable, '-m', 'flexsds.core.calibration', str(request), str(output)],
                           env=env, check=True, timeout=3600)
        except subprocess.SubprocessError as exc:
            raise RuntimeError(f'GPU calibration process failed: {exc}') from exc
        return json.loads(output.read_text())


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    result = measure_cell_group(**request)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
