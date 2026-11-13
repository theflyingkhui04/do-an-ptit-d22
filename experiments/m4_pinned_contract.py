"""CPU source audit of exact vLLM 0.30.0; never imports CUDA/vLLM."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.m4_protocol import ROOT, sha256


def verify(root: Path) -> dict:
    records = []
    for manifest in ("scalar_vllm_source_manifest.json", "m4_eagle_source_manifest.json"):
        for record in json.loads((ROOT / "experiments" / manifest).read_text())["files"]:
            path = root / record["path"]
            if sha256(path.read_bytes()) != record["sha256"]:
                raise ValueError(f"Pinned source mismatch: {path}")
            records.append(record)
    def text(path):
        return (root / "vllm" / path).read_text(encoding="utf-8")
    eagle = ast.parse(text("v1/spec_decode/eagle.py"))
    cls = next(n for n in eagle.body if isinstance(n, ast.ClassDef))
    assert cls.name == "EagleProposer" and ast.unparse(cls.bases[0]) == "SpecDecodeBaseProposer"
    assert "pass_hidden_states_to_model=True" in ast.unparse(cls)
    base = text("v1/spec_decode/llm_base_proposer.py")
    assert "self.num_speculative_tokens = num_speculative_tokens" in base
    assert "if self.num_speculative_tokens == 0:" in base
    assert base.index("ret_hidden_states = self.model(**model_kwargs)") < base.index("if self.num_speculative_tokens == 0:")
    runner = text("v1/worker/gpu_model_runner.py")
    assert "num_speculative_tokens=num_spec_tokens_to_schedule" in runner
    assert "torch.cat(" in runner and "aux_hidden_states" in runner
    scheduler = text("v1/core/sched/scheduler.py")
    assert "self.dynamic_sd_lookup" in scheduler
    assert "self.make_spec_decoding_stats(" in scheduler
    async_source = text("v1/core/sched/async_scheduler.py")
    assert "scheduler_output.num_spec_tokens_to_schedule" in async_source
    assert "request.spec_token_ids = self._spec_token_placeholders" in async_source
    config = text("config/speculative.py")
    assert "num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None" in config
    protocol = text("entrypoints/openai/chat_completion/protocol.py")
    assert "return_token_ids: bool | None" in protocol
    assert "token_ids: list[int] | None" in protocol
    assert "metrics: PerRequestMetrics | None" in protocol
    return dict(version="0.30.0", verified_files=len(records), source_contract="PASS",
                gpu_runtime="NOT VERIFIED", files=records)
