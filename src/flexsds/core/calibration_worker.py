"""Worker-local, eager vLLM 0.30 component timings; never times frontend RPCs."""
from __future__ import annotations
import time


class CalibrationWorker:
    def calibration_install(self):
        import torch
        import vllm
        if vllm.__version__ != '0.30.0':
            raise RuntimeError('Calibration requires vLLM 0.30.0')
        runner = self.model_runner
        if getattr(self, 'use_v2_model_runner', False):
            raise RuntimeError('Calibration requires the V1 model runner')
        state = dict(active=False, samples={}, shapes=[])
        self._flex_calibration = state
        execute = runner.execute_model

        def execute_observed(output, *args, **kwargs):
            state['scheduled'] = dict(output.num_scheduled_tokens)
            state['preempted'] = bool(output.preempted_req_ids)
            return execute(output, *args, **kwargs)
        runner.execute_model = execute_observed

        def before(module, args):
            state['eligible'] = False
            if not state['active']:
                return
            batch = runner.input_batch
            counts = [state['scheduled'].get(r, 0) for r in batch.req_ids]
            contexts = [int(v) for v in batch.num_computed_tokens_cpu[:batch.num_reqs]]
            state['eligible'] = (batch.num_reqs == state['batch'] and
                                 counts == [state['k'] + 1] * state['batch'] and
                                 contexts == [state['context']] * state['batch'] and
                                 not state['preempted'])
            if state['eligible'] and 'target' not in state['samples']:
                state['shapes'].append(dict(batch=batch.num_reqs, query=counts, context=contexts))
                torch.cuda.synchronize()
                state['start'] = time.perf_counter()

        def after(module, args, output):
            if state['active'] and state['eligible'] and 'target' not in state['samples']:
                torch.cuda.synchronize()
                state['samples']['target'] = (time.perf_counter() - state['start']) * 1000
        runner.model.register_forward_pre_hook(before)
        runner.model.register_forward_hook(after)
        if runner.speculative_config is not None:
            propose = runner.drafter.propose

            def propose_observed(*args, **kwargs):
                if not (state['active'] and state.get('eligible') and 'draft' not in state['samples']):
                    return propose(*args, **kwargs)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = propose(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                # Do not silently accept a different speculative budget.
                if tuple(result.shape) != (state['batch'], state['k']):
                    raise RuntimeError('Calibration proposal shape mismatch')
                state['samples']['draft'] = elapsed
                return result
            runner.drafter.propose = propose_observed

    def calibration_begin(self, batch, k, context):
        self._flex_calibration.update(active=True, batch=batch, k=k, context=context,
                                      samples={}, shapes=[], eligible=False)

    def calibration_finish(self):
        state = self._flex_calibration
        state['active'] = False
        return dict(samples=state['samples'], shapes=state['shapes'])
