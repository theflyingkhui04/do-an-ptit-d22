"""Pre-capture graph adapter, loaded only by the explicit diagnostic worker.

No torch/vLLM imports on import. See the source proof in the layer runbook.
The normal memory profile and KV allocation finish BEFORE this adapter exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.m4_layer_probe import LAYER_STAGE, stage_plan
from experiments.m4_cache_probe_hooks import configuration, verify_source_tree, _InstanceHooks


def verify_layer_sources():
    import vllm
    import torch
    if vllm.__version__ != '0.30.0' or torch.__version__ != '2.13.0+cu130':
        raise RuntimeError('Pinned layer diagnostic runtime required')
    manifest = json.loads(Path(__file__).with_name('m4_layer_probe_sources.json').read_text(encoding='utf8'))
    return verify_source_tree(Path(vllm.__file__).resolve().parent.parent,manifest)


def kv_identity(runner):
    """Metadata only: no read or transfer of KV contents."""
    if not runner.kv_caches:
        raise RuntimeError('Layer observer must be installed after KV allocation')
    return [(id(t),t.data_ptr(),tuple(t.shape),str(t.dtype),t.numel()*t.element_size())
            for t in runner.kv_caches]


class GraphLayerRuntime:
    def __init__(self, worker, model, reset, graph_wrappers, original_forward):
        import torch
        self.worker, self.runner, self.model = worker, worker.model_runner, model
        self.reset, self.graph_wrappers = reset, graph_wrappers
        self.original_class = type(model)
        self.original_aux_flag = self.runner.use_aux_hidden_state_outputs
        self.hooks = _InstanceHooks()
        self.current = None
        self.ready = False
        self.closed = False
        self.cleanup_complete = False
        self.owns_row_buffer = False
        self.before_kv = kv_identity(self.runner)
        self.original_cache_dirs = (model.vllm_config.compilation_config.cache_dir,
                                    model.vllm_config.compilation_config.local_cache_dir)
        self.original_traced_files = set(model.vllm_config.compilation_config.traced_files)
        width, layers = model.config.hidden_size, model.config.num_hidden_layers
        stage_plan(layers)
        if (type(width) is not int or width <= 0 or len(model.layers) != layers
                or model.start_layer != 0 or model.end_layer != layers or self.original_aux_flag):
            raise RuntimeError('Unsupported Qwen2 layer/auxiliary configuration')
        self.contract = dict(hidden_size=width,num_hidden_layers=layers,selected_layers=list(range(layers)),
            tensor_parallel_size=1,pipeline_parallel_size=1,
            installation='after_kv_allocation_before_runtime_capture',graph_modes=['FULL','PIECEWISE'])
        # These are preflight failures, not installations to tear down. Leave
        # someone else's graphs/buffer intact when rejecting an unsupported path.
        self.require_empty_graphs()
        if hasattr(model,'_m4_layer_row'):
            raise RuntimeError('Layer row buffer already exists')
        self.row = torch.zeros(1,dtype=torch.int64,device=self.runner.device)
        runtime = self

        # New code object: the unguarded baseline Dynamo callable MUST NOT be
        # reused. This wrapper delegates all original tensor operations unchanged.
        def observed_forward(instance, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            if input_ids is None or inputs_embeds is not None or intermediate_tensors is not None:
                raise RuntimeError('Layer diagnostic requires single-rank token-ID inputs')
            output = original_forward(instance,input_ids,positions,intermediate_tensors,inputs_embeds)
            hidden, auxiliary = output
            row = instance._m4_layer_row
            stamp = torch.cat((row,positions.index_select(0,row).to(torch.int64),
                input_ids.index_select(0,row).to(torch.int64),torch.full_like(row,positions.shape[0])))
            return hidden, auxiliary + [stamp]

        observed_forward.__code__ = observed_forward.__code__.replace(co_name=f'm4_layer_forward_{id(self)}')
        observed_class = type('M4LayerObservedQwen2Model',(self.original_class,),
                              {'forward':observed_forward})

        def add_row(auxiliary, index, hidden, residual):
            if (hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] != width
                    or hidden.stride() != (width,1)):
                raise RuntimeError('Unsupported layer source shape/dtype/layout')
            auxiliary.append(hidden.index_select(0,model._m4_layer_row).clone())
            if index != 0:
                if residual is None or residual.dtype != hidden.dtype or residual.shape != hidden.shape:
                    raise RuntimeError('Missing layer residual component')
                auxiliary.append(residual.index_select(0,model._m4_layer_row).clone())
            elif residual is not None:
                raise RuntimeError('Embedding boundary unexpectedly has a residual')
            return auxiliary

        original_model_forward = self.runner._model_forward

        def forward(*args, **kwargs):
            if runtime.closed or not runtime.ready:
                raise RuntimeError('Layer capture used outside its installed lifecycle')
            runtime.current = None
            probe = getattr(worker,'_m4_probe',None)
            row = 0  # Dummy/capture warmups have no request; no evidence is taken.
            if probe is not None and probe['armed'] is not None:
                output = probe['output']
                ids = runtime.runner.input_batch.req_ids
                if output is None or len(ids) != 1:
                    raise RuntimeError('Layer capture request isolation mismatch')
                rid = ids[0]
                if runtime.runner.requests[rid].prompt_token_ids != probe['armed']['prompt']:
                    raise RuntimeError('Layer capture wrong request prompt')
                query = output.num_scheduled_tokens[rid]
                if type(query) is not int or query <= 0:
                    raise RuntimeError('Invalid layer query row')
                row = query-1
            runtime.row.fill_(row)  # GPU write on the same stream, not a D2H sync.
            result = original_model_forward(*args,**kwargs)
            expected = len(stage_plan(layers))-1
            if (not isinstance(result,tuple) or len(result) != 2 or not isinstance(result[1],list)
                    or len(result[1]) != expected+1):
                raise RuntimeError('Unsupported/missing graph layer return surface')
            auxiliary = result[1]
            if (any(not isinstance(t,torch.Tensor) or t.dtype != torch.bfloat16
                    or list(t.shape) != [1,width] for t in auxiliary[:-1])
                    or auxiliary[-1].dtype != torch.int64 or list(auxiliary[-1].shape) != [4]):
                raise RuntimeError('Graph layer return shape/dtype mismatch')
            runtime.current = auxiliary
            return result

        try:
            model.__class__ = observed_class
            # A registered nonpersistent buffer is rebound as model state by
            # AOT export/load, rather than serialized as a closure constant.
            model.register_buffer('_m4_layer_row',self.row,persistent=False)
            self.owns_row_buffer = True
            self.hooks.replace(model,'_maybe_add_hidden_state',add_row)
            self.runner.use_aux_hidden_state_outputs = True
            self.reset(model)
            if model.compiled or getattr(model,'aot_compiled_fn',None) is not None:
                raise RuntimeError('Compiled baseline was not reset')
            self.hooks.replace(self.runner,'_model_forward',forward)
            self.ready = True
        except BaseException:
            self.close()
            raise

    def require_empty_graphs(self):
        for wrapper in self.graph_wrappers():
            if wrapper.vllm_config is not self.model.vllm_config:
                raise RuntimeError('Multiple graph-owning engines are unsupported')
            if wrapper.concrete_cudagraph_entries:
                raise RuntimeError('Layer instrumentation attempted after graph capture')

    def check_kv_unchanged(self):
        if kv_identity(self.runner) != self.before_kv:
            raise RuntimeError('Layer observer changed KV allocation')

    def snapshot(self, final_row):
        import torch
        if self.current is None:
            raise RuntimeError('Missing current graph layer output')
        width = self.contract['hidden_size']
        if (final_row.dtype != torch.bfloat16 or list(final_row.shape) != [width]):
            raise RuntimeError('Final norm row shape/dtype mismatch')
        # Independent storage BEFORE another graph replay can reuse its outputs.
        table = torch.cat(self.current[:-1]+[final_row.reshape(1,width)],dim=0).detach().clone()
        stamp = self.current[-1].detach().clone()
        self.current = None
        return table,stamp

    def close(self):
        if self.closed and self.cleanup_complete:
            return
        self.closed = True
        self.ready = False
        self.current = None
        errors = []
        # The trajectory has already aborted/barriered; shutdown follows even on
        # failure. Graph references must release copies before hooks are removed.
        for wrapper in self.graph_wrappers():
            try:
                if wrapper.vllm_config is self.model.vllm_config:
                    wrapper.clear_graphs()
            except BaseException as exc:
                errors.append(exc)
        try:
            self.hooks.restore()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self.owns_row_buffer and hasattr(self.model,'_m4_layer_row'):
                delattr(self.model,'_m4_layer_row')
            self.owns_row_buffer = False
        except BaseException as exc:
            errors.append(exc)
        self.runner.use_aux_hidden_state_outputs = self.original_aux_flag
        self.model.__class__ = self.original_class
        try:
            self.reset(self.model)
        except BaseException as exc:
            errors.append(exc)
        compilation = self.model.vllm_config.compilation_config
        compilation.cache_dir, compilation.local_cache_dir = self.original_cache_dirs
        compilation.traced_files.clear()
        compilation.traced_files.update(self.original_traced_files)
        self.row = None
        try:
            self.check_kv_unchanged()
        except BaseException as exc:
            errors.append(exc)
        self.cleanup_complete = not errors
        if errors:
            raise RuntimeError('Layer instrumentation cleanup failed') from errors[0]


def install_before_warmup(worker):
    from vllm.config import set_current_vllm_config
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.wrapper import reset_compile_wrapper
    from vllm.model_executor.models.qwen2 import Qwen2Model,Qwen2ForCausalLM
    from vllm import envs
    verify_layer_sources()
    config = configuration()
    vconfig = worker.vllm_config
    runner = worker.model_runner
    if (config.get('diagnostic_mode') != 'decode_trajectory' or config.get('localization_stage') != LAYER_STAGE
            or vconfig.speculative_config is not None or runner.use_aux_hidden_state_outputs
            or not runner.use_async_scheduling
            or vconfig.parallel_config.tensor_parallel_size != 1
            or vconfig.parallel_config.pipeline_parallel_size != 1
            or vconfig.parallel_config.use_ubatching or runner.lora_config is not None
            or vconfig.compilation_config.mode.name != 'VLLM_COMPILE'
            or vconfig.compilation_config.cudagraph_mode.name != 'FULL_AND_PIECEWISE'
            or vconfig.model_config.enforce_eager
            or envs.VLLM_USE_V2_MODEL_RUNNER):
        raise RuntimeError('Unsupported layer instrumentation engine/path')
    # Reject the alternate wrapper; the reviewed path is ordinary FULL +
    # compiled PIECEWISE graphs, not breakable/ubatched graphs.
    if (type(runner.model) is not CUDAGraphWrapper or runner.model.runtime_mode.name != 'FULL'
            or type(runner.model.runnable) is not Qwen2ForCausalLM):
        raise RuntimeError('Unsupported graph wrapper')
    causal = runner.model.runnable
    model = causal.model
    if type(model) is not Qwen2Model:
        raise RuntimeError('Unsupported target model')
    def reset(target):
        with set_current_vllm_config(vconfig):
            target.cleanup()  # Remove the previous wrapper's bytecode hook.
            reset_compile_wrapper(target)
    with set_current_vllm_config(vconfig):
        return GraphLayerRuntime(worker,model,reset,
            lambda:list(CUDAGraphWrapper._all_instances),Qwen2Model.forward)


def __getattr__(name):
    if name != 'LayerBoundaryWorker':
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.config import set_current_vllm_config
    class LayerBoundaryWorker(Worker):
        def compile_or_warm_up_model(self):
            if hasattr(self,'_m4_layer_runtime'):
                raise RuntimeError('Layer runtime already installed')
            with set_current_vllm_config(self.vllm_config):
                self._m4_layer_runtime = install_before_warmup(self)
                try:
                    result = super().compile_or_warm_up_model()
                    self._m4_layer_runtime.check_kv_unchanged()
                    return result
                except BaseException:
                    self._m4_layer_runtime.close()
                    raise
    return LayerBoundaryWorker
