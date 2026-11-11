"""CPU inspection of two unmodified installed V1 methods; no lineage implementation.

Uses CPU tensors solely to observe which previous-row values pinned input
preparation copies. This is not a real-engine test or GPU compatibility result.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "20261110-async-lineage-design/installed/vllm/v1/worker/gpu_model_runner.py"
source = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source)
runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
names = {"_compute_prev_positions", "_prepare_input_ids"}
methods = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in names]
scope = {"np": np, "torch": torch, "PIN_MEMORY": False}
module = ast.Module(body=[ast.ClassDef(name="PinnedMethods", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)


class InputBuffer:
    def __init__(self, count):
        self.gpu = torch.full((count,), 7777, dtype=torch.int32)

    def copy_to_gpu(self, count):
        self.gpu[:count].fill_(7777)


def inspect_case(name, requests, previous_map, old_drafts, prefix_lengths, next_k, expected):
    obj = scope["PinnedMethods"]()
    previous_samples = torch.tensor([[91], [92]], dtype=torch.int32)
    obj.input_batch = SimpleNamespace(req_ids=requests, prev_req_id_to_index=previous_map,
                                      prev_sampled_token_ids=previous_samples)
    obj.prev_positions = SimpleNamespace(np=np.empty(len(requests), dtype=np.int32))
    obj._pp_recv_work = None
    obj.enable_prompt_embeds = False
    obj.prev_num_spec_tokens = old_drafts.shape[1]
    obj._draft_token_ids = old_drafts
    obj.device = torch.device("cpu")
    lengths = np.asarray([1 + prefix_lengths[r] for r in requests], dtype=np.int32)
    cumulative = np.cumsum(lengths)
    total = int(cumulative[-1])
    obj.input_ids = InputBuffer(total)
    output = SimpleNamespace(scheduled_spec_decode_tokens={r: [-1] * prefix_lengths[r] for r in requests},
                             num_spec_tokens_to_schedule=next_k)
    obj._compute_prev_positions(len(requests))
    obj._prepare_input_ids(output, len(requests), total, cumulative)
    actual = obj.input_ids.gpu.tolist()
    assert actual == expected, (name, actual, expected)
    return {"case": name, "current_requests": requests, "previous_positions": obj.prev_positions.np.tolist(),
            "next_producer_k": next_k, "actual_prepared_values": actual, "expected": expected}


drafts = torch.tensor([[1001, 1002], [2001, 2002]], dtype=torch.int64)
cases = [
    inspect_case("P10-K2 drafts consumed while P11 selects K4; reordered rows", ["r2", "r1"],
                 {"r1": 0, "r2": 1}, drafts, {"r1": 2, "r2": 2}, 4, [92, 2001, 2002, 91, 1001, 1002]),
    inspect_case("compaction moves r2 from previous row 1 to current row 0", ["r2"],
                 {"r2": 1}, drafts, {"r2": 2}, 3, [92, 2001, 2002]),
    inspect_case("P11 selects K0 but still verifies P10-K2", ["r1"],
                 {"r1": 0}, drafts, {"r1": 2}, 0, [91, 1001, 1002]),
    inspect_case("request prefix trimmed to one old draft", ["r1"],
                 {"r1": 0}, drafts, {"r1": 1}, 4, [91, 1001]),
    inspect_case("zero verification slots copies no old draft", ["r1"],
                 {"r1": 0}, drafts, {"r1": 0}, 4, [91]),
    inspect_case("finished old row reused by r3; prior mapping removed", ["r3"],
                 {}, drafts, {"r3": 2}, 4, [7777, 7777, 7777]),
    inspect_case("same string ID resubmitted after old mapping removal", ["r1"],
                 {}, drafts, {"r1": 2}, 4, [7777, 7777, 7777]),
]
report = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "method_line_ranges": {n.name: [n.lineno, n.end_lineno] for n in methods},
          "scope": "CPU execution of extracted unmodified input-preparation methods; simulated surrounding state; not L3 and not a lineage implementation",
          "cases": cases}
(ROOT / "row_mapping_inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
