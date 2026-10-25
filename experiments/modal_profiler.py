"""[M3-02] Forward cost of the target and draft models on GPU L4 (Dương's part).

Measures the per-step decode time C_T(L, B) of Qwen2.5-7B-Instruct-AWQ and
C_D(L, B) of Qwen2.5-0.5B-Instruct with vLLM offline ``LLM`` on the same
configuration as the lane servers (experiments/modal_lanes.py). Khôi builds
C(N, B) and alpha* from these numbers; this module does not.

    step_ms = (T(1 + N) - T(1)) / N

T(m) is the wall time of one ``generate`` call over B distinct prompts of exactly
L tokens with ``max_tokens=m`` and ``ignore_eos``; prefill cancels out. Context
grows from L to L + N while measuring, so a cell is the mean over [L, L + N].

    modal run -m experiments.modal_runner::profile_costs --run-id <id>

Evidence: experiments/runs/m3_02/<run-id>/{forward_costs.jsonl,env.json};
summary: experiments/results/m3_02_forward_costs_l4.json.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from experiments import modal_lanes as lanes


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTEXT_LENGTHS = (256, 512, 1024, 2048, 3072)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DECODE_TOKENS = 64
WARMUP = 3
REPEATS = 5
VARIANCE_THRESHOLD = 0.15
MAX_MODEL_LEN = 4096
# Plain-text token ids of the Qwen2.5 vocabulary, far from special tokens.
TOKEN_ID_RANGE = (1000, 30000)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Generate = Callable[[list[list[int]], int], None]


@dataclass(frozen=True)
class ProfileModel:
    repo: str
    revision: str | None
    quantization: str | None


_DTYPE = lanes.DTYPE_PROFILES[lanes.DEFAULT_PROFILE]
MODELS = {
    "target": ProfileModel(lanes.TARGET_MODEL, lanes.TARGET_REVISION, _DTYPE["quantization"]),
    "draft": ProfileModel(lanes.DRAFT_MODEL, lanes.DRAFT_REVISION, None),
}


def grid(contexts: Sequence[int] = CONTEXT_LENGTHS, batches: Sequence[int] = BATCH_SIZES) -> list[tuple[int, int]]:
    return [(length, batch) for length in contexts for batch in batches]


def parse_int_list(text: str, default: Sequence[int]) -> tuple[int, ...]:
    """CLI helper: "256,1024" -> (256, 1024); empty -> default; values must be >= 1."""
    if not text.strip():
        return tuple(default)
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or min(values) < 1:
        raise ValueError(f"Expected positive integers, got {text!r}")
    return values


def engine_kwargs(model: str) -> dict[str, Any]:
    """``vllm.LLM`` arguments mirroring COMMON_SERVING_ARGS of the lane servers."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {sorted(MODELS)}")
    cfg = MODELS[model]
    kwargs: dict[str, Any] = {
        "model": cfg.repo,
        "revision": cfg.revision,
        "dtype": _DTYPE["dtype"],
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "generation_config": "vllm",
        # Every measurement must prefill for real.
        "enable_prefix_caching": False,
    }
    if cfg.quantization:
        kwargs["quantization"] = cfg.quantization
    return kwargs


def prompt_token_ids(length: int, batch: int, seed: int = 0) -> list[list[int]]:
    """B distinct, deterministic prompts of exactly ``length`` token ids."""
    low, high = TOKEN_ID_RANGE
    prompts = []
    for index in range(batch):
        # Benchmark input only, not security: a seeded PRNG keeps prompts reproducible.
        rng = random.Random(f"{seed}-{length}-{batch}-{index}")  # NOSONAR
        prompts.append([rng.randrange(low, high) for _ in range(length)])
    return prompts


def fits_kv(length: int, batch: int, decode_tokens: int, capacity_tokens: int | None) -> bool:
    """True when all B sequences fit the KV cache at once (else vLLM would queue them)."""
    if capacity_tokens is None:
        return True
    return batch * (length + decode_tokens + 1) <= capacity_tokens


def step_ms(prefill_s: float, full_s: float, decode_tokens: int) -> float:
    if decode_tokens < 1:
        raise ValueError("decode_tokens must be >= 1")
    if full_s <= prefill_s:
        raise ValueError(f"T(1+N)={full_s:.4f}s must exceed T(1)={prefill_s:.4f}s")
    return (full_s - prefill_s) / decode_tokens * 1000.0


def measure_cell(
    generate: Generate,
    model: str,
    length: int,
    batch: int,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    decode_tokens: int = DECODE_TOKENS,
    seed: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict]:
    """Warm-up then measured repeats of one (model, L, B) cell."""
    prompts = prompt_token_ids(length, batch, seed=seed)

    def timed(max_tokens: int) -> float:
        start = clock()
        generate(prompts, max_tokens)
        return clock() - start

    for _ in range(warmup):
        timed(1)
        timed(1 + decode_tokens)

    records = []
    for repeat in range(1, repeats + 1):
        prefill_s = timed(1)
        full_s = timed(1 + decode_tokens)
        records.append({
            "model": model, "context_len": length, "batch_size": batch, "repeat": repeat,
            "decode_tokens": decode_tokens, "t_prefill_s": round(prefill_s, 6), "t_full_s": round(full_s, 6),
            "step_ms": round(step_ms(prefill_s, full_s, decode_tokens), 4), "status": "ok",
        })
    return records


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]


def summarize(records: Iterable[dict], threshold: float = VARIANCE_THRESHOLD) -> list[dict]:
    """One row per (model, L, B): median, p90, min, max, variance_ratio, variance_flag."""
    cells: dict[tuple[str, int, int], list[dict]] = {}
    for record in records:
        cells.setdefault((record["model"], record["context_len"], record["batch_size"]), []).append(record)

    rows = []
    for (model, length, batch), group in sorted(cells.items()):
        row: dict[str, Any] = {"model": model, "context_len": length, "batch_size": batch}
        values = [r["step_ms"] for r in group if r.get("status") == "ok"]
        if not values:
            row.update(status="skipped", reason=group[0].get("reason"), n_repeats=0, step_ms_median=None,
                       step_ms_p90=None, step_ms_min=None, step_ms_max=None, variance_ratio=None,
                       variance_flag=None)
        else:
            median = statistics.median(values)
            ratio = (max(values) - min(values)) / median
            row.update(status="ok", n_repeats=len(values), step_ms_median=round(median, 4),
                       step_ms_p90=round(_p90(values), 4), step_ms_min=min(values), step_ms_max=max(values),
                       variance_ratio=round(ratio, 4), variance_flag=ratio > threshold)
        rows.append(row)
    return rows


def evidence_dir(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id or "") or ".." in run_id:
        raise ValueError(f"Invalid run-id {run_id!r}: use letters, digits, '.', '_' or '-'")
    root = (REPO_ROOT / "experiments" / "runs" / "m3_02").resolve()
    path = (root / run_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid run-id {run_id!r}: evidence must stay under {root}")
    return path
