"""Reviewed native worker; imports GPU runtime only inside LiveWorker startup."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import traceback

from experiments.eagle_pulse_campaign import CELLS, PRODUCTS, SLOTS, canonical, digest, require


class LiveWorker:
    def __init__(self, plan, project):
        self.plan, self.project, self.core = plan, Path(project), None
        self.attempted_slots, self.attempted_ids = set(), set()
        probe = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                         "--format=csv,noheader,nounits"], text=True).strip()
        lines = probe.splitlines()
        require(len(lines) == 1, "worker_gpu_count")
        product = lines[0].split(",")[0].strip()
        self.identity = dict(observed=[["gpu_name", product]], configured=[],
                             schema_version="runtime_identity_snapshot_v1", hardware_probe=probe)
        if product not in PRODUCTS:
            return
        os.environ.update(VLLM_USE_V2_MODEL_RUNNER="0", ADAPTIVE_REPLACEMENT_POLICY="1",
            ADAPTIVE_COST_POLICY_MODE="strict", ADAPTIVE_STRICT="1", ADAPTIVE_MAX_K="3",
            ADAPTIVE_ENABLE_VRAM_GUARD="1", ADAPTIVE_PULSE_DOMAIN_JSON=canonical(plan["domain"]),
            ADAPTIVE_CONTROLLER_KWARGS=canonical(dict(window_size=4, initial_alpha=.85, candidate_budgets=[0, 2, 3])))
        for key in ("FLEXSDS_PROFILE_PATH", "ADAPTIVE_CONSUMER_PROFILE_PATH", "M203_P177_PRODUCER_K0",
                    "M203_P177_ATTENTION_SUBSTITUTION", "ADAPTIVE_STEP_TELEMETRY_PATH"):
            os.environ.pop(key, None)
        import vllm
        from experiments.m4_pinned_contract import verify
        from experiments.eagle_rw_collector import EagleRWSourceProvenance
        from src.server.lineage_executor import LineageUniProcExecutor
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.core import EngineCore
        verify(Path(vllm.__file__).resolve().parent.parent)
        self.source = EagleRWSourceProvenance.from_checkout(project)
        require(self.source.source_git_sha == plan["source_git_sha"], "worker_source_sha")
        config = EngineArgs(**plan["engine_args"]).create_engine_config()
        self.core = EngineCore(config, LineageUniProcExecutor, log_stats=False)
        self.identity = asdict(self.core.scheduler._planner.runtime_identity)
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.v1.request import Request
        self.sampling_factory, self.request_factory = SamplingParams, Request
        self.tokenizer = AutoTokenizer.from_pretrained(plan["engine_args"]["model"], revision=plan["engine_args"]["revision"])
        self.prompt_ids = self.tokenizer.encode(plan["domain"]["preparation"]["prompt"], add_special_tokens=False)
        require(1 <= len(self.prompt_ids) < 128, "prompt_domain")

    def step(self):
        batches, executed = self.core.step_fn()
        self.core.post_step(executed)
        return batches

    def execute(self, product, cell, slot, attempt_id):
        from experiments.eagle_rw_collector import EaglePulseCycleCollector, EagleRWCohortDescriptor
        from src.server.decision_state import FixedCalibrationActionV1
        plan, core = self.plan, self.core
        require(dict(self.identity["observed"])["gpu_name"] == product, "worker_product")
        require(cell in {c["id"] for c in CELLS} and slot in SLOTS, "worker_slot_domain")
        key = (product, cell, slot)
        require(key not in self.attempted_slots and attempt_id not in self.attempted_ids, "worker_slot_already_attempted")
        self.attempted_slots.add(key)
        self.attempted_ids.add(attempt_id)
        cid = f"{product.removeprefix('NVIDIA ')}:{cell}:{slot}"
        rid = f"campaign:{plan['source_git_sha']}:{cid}:{attempt_id}"
        q = next(c["future_k"] for c in CELLS if c["id"] == cell)
        p = core.scheduler._planner
        original = p.plan_step
        actual_bases = {}
        def tracked_plan(*args, **kwargs):
            result = original(*args, **kwargs)
            if result.decision_state is not None:
                actual_bases[result.decision_state.scheduler_sequence] = dict(result.per_request_base_k)
            return result
        collector = None
        row = dict(cohort_id=cid, request_id=rid, incarnation=None, synthetic=False,
                   objective="SECP-PULSE-2R-v1", source_git_sha=plan["source_git_sha"],
                   campaign_plan_sha256=digest(plan), failures=[])
        log, failure = io.StringIO(), None
        try:
            with redirect_stdout(log), redirect_stderr(log):
                require(core.batch_queue_size == plan["domain"]["queue"]["capacity"], "resolved_queue_capacity")
                require(not core.scheduler.has_requests() and not core.batch_queue, "previous_cohort_not_drained")
                require(core.reset_prefix_cache(), "prefix_cache_reset_failed")
                params = self.sampling_factory(temperature=0., max_tokens=64, seed=731, ignore_eos=True,
                                        stop=[], stop_token_ids=[])
                params.eos_token_id = None
                request = self.request_factory(rid, self.prompt_ids, params, None, block_hasher=core.request_block_hasher)
                core.add_request(request)
                epoch = core.scheduler._lineage.request_key(request)
                row["incarnation"] = epoch
                p.bind_request(rid, epoch)
                # No survival history survives the new request incarnation.
                require(p.pulse_history.estimate(rid, epoch, 3).count == 0, "history_leak")
                def prepare(*args, **kwargs):
                    nonlocal collector
                    state = kwargs.get("decision_state")
                    if state is not None and state.eligible_decode_request_ids:
                        p.plan_step = tracked_plan
                        collector = EaglePulseCycleCollector(core, self.source,
                            EagleRWCohortDescriptor(cid, ((rid, epoch),)), q, canonical(plan["domain"])).install()
                        return p.plan_step(*args, **kwargs)
                    kwargs["calibration_action"] = FixedCalibrationActionV1(0, digest(plan))
                    return tracked_plan(*args, **kwargs)
                p.plan_step = prepare
                for _ in range(128):
                    self.step()
                    if collector is not None and any(
                            record.get("cohort_id") == cid for record in p.pulse_cycles.records):
                        # Source/export validation runs only after the native
                        # terminal record has sampled t1, never between queue steps.
                        exported = collector.export()
                        row = exported["records"][0]
                        break
                else:
                    raise RuntimeError("bounded_slot_engine_steps_exhausted")
                require(collector is not None, "missing_decode_pulse")
                require(row["status"] == "COMPLETE", "native_cycle_not_complete")
                require(exported["canonical_profile_input"] and exported["source_valid"], "collector_export_not_canonical")
                # The lease admits base3 on its actual StepPlan; preserve that
                # existing admission predicate rather than infer from product.
                for receipt in (row["pulse_receipt"], row["continuation_receipt"]):
                    receipt["base_k"] = actual_bases[receipt["schedule_sequence"]][rid]
                row["campaign_plan_sha256"] = digest(plan)
                row["preparation_evidence"] = dict(prefix_cache_reset=True, prompt_token_ids=self.prompt_ids,
                    tokenizer_revision=plan["engine_args"]["revision"], fresh_request=True,
                    native_queue_capacity=core.batch_queue_size, history_count=0)
        except Exception as error:
            failure = f"{type(error).__name__}:{error}"
            if collector is not None:
                collector.close()
                records = collector.export()["records"]
                if records:
                    row = records[0]
            row.update(status="INVALID", failures=[failure], campaign_plan_sha256=digest(plan))
            log.write(traceback.format_exc())
        finally:
            if collector is not None:
                collector.close()
            p.plan_step = original
            # Untimed post-cycle cleanup; drain native pending K0 work, never
            # include it as a third target retirement.
            core.abort_requests([rid])
            for _ in range(128):
                if not core.scheduler.has_requests() and not core.batch_queue:
                    break
                self.step()
            else:
                failure = failure or "cohort_cleanup_failed"
        return dict(record=row, identity=self.identity, failure=failure, log=log.getvalue())

    def close(self):
        if self.core is not None:
            self.core.shutdown()
