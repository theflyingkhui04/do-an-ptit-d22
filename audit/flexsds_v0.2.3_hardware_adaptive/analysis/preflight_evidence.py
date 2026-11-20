import sys,os,json,hashlib,runpy,importlib.metadata as md
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.stdout.reconfigure(encoding='utf-8')
a=Path('audit/flexsds_v0.2.3_hardware_adaptive').resolve(); site=Path(md.distribution('flexsds').locate_file(''))
sys.path.insert(0,str(Path.cwd()))
from flexsds.core.cost_model import validate_measured_profile
from flexsds.core.hardware import HardwareInfo
from flexsds.server.local_runner import LocalServingRunner
from flexsds.server import modal_image
p=Path(r'C:\Users\daibu\.flexsds\profiles\profile_sm_89_remote_l4_fcda5486b12d_full.json'); data=json.loads(p.read_bytes())
hw=HardwareInfo(device_name='Remote L4',sm_architecture='sm_89',total_vram_mb=23000,free_vram_mb=23000,cuda_version='unknown',is_cuda_available=True)
validate_measured_profile(data,hw,data['models']['target'],data['models']['draft'])
print('PROFILE VALIDATION: PASS (expected identity; actual serving GPU unavailable)')
(a/'profiles/evidence-config.json').write_text(json.dumps(data['evidence']['config'],indent=2),encoding='utf-8')
base=dict(target_model=data['models']['target'],draft_model=data['models']['draft'],target_revision=data['models']['target_revision'],draft_revision=data['models']['draft_revision'],draft_method='eagle3',target_gpu='L4',port=8000,max_num_seqs=8,max_model_len=2048)
for enabled,name in [(True,'adaptive_on'),(False,'adaptive_off')]:
 cfg=dict(base,enable_adaptive=enabled,profile_path=str(p) if enabled else None,telemetry_path=str(a/name/'telemetry'/'serving_trace.jsonl'))
 runner=LocalServingRunner(); cmd=runner.build_command(cfg); env=runner.setup_environment(cfg)
 safe={k:v for k,v in env.items() if k.startswith('ADAPTIVE_') or k in ['FLEXSDS_PROFILE_PATH','VLLM_USE_V2_MODEL_RUNNER']}
 out=dict(provenance='OFFLINE DERIVATION from installed builder; NOT CLI-resolved, NOT launched',config=cfg,command=cmd,environment=safe)
 (a/name/'startup/derived-runtime.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(name,json.dumps(out,indent=2))
image=Mock(); image.env.return_value=image; image.add_local_dir.return_value=image; image.add_local_file.return_value=image
app=Mock(); app.function.return_value=lambda fn:fn
fake=SimpleNamespace(App=Mock(return_value=app),is_local=lambda:True,Volume=SimpleNamespace(from_name=Mock()),web_server=lambda *args,**kw:lambda fn:fn)
config=dict(base,enable_adaptive=True,profile_path=str(p))
with patch.dict(sys.modules,{'modal':fake}),patch.object(modal_image,'serving_image',lambda:image),patch.dict(os.environ,{'FLEXSDS_SERVING_CONFIG':json.dumps(config)}):
 result=runpy.run_path(str(site/'flexsds/server/modal_app.py'))
 print('INSTALLED APP PROJECT_ROOT',result['PROJECT_ROOT'])
 mounts=[dict(local=c.args[0],remote=c.kwargs['remote_path']) for c in image.add_local_dir.call_args_list]
 print('ACTUAL IMAGE DIRECTORY MOUNTS',json.dumps(mounts))
 proof=dict(project_root=str(result['PROJECT_ROOT']),directory_mounts=mounts,file_mounts=[dict(local=c.args[0],remote=c.kwargs['remote_path']) for c in image.add_local_file.call_args_list],required_but_unmounted=['/root/project/src/server','/root/project/src/proposer','/root/project/experiments'],evidence_level='LOCAL MOCKED IMAGE CONSTRUCTION; no cloud calls')
 (a/'analysis/installed-image-construction.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
for rel in ['flexsds/cli/main.py','flexsds/server/modal_app.py','flexsds/server/modal_runner.py','flexsds/server/local_runner.py','flexsds/server/modal_image.py','flexsds/core/cost_model.py','server/adaptive_scheduler.py','server/budget_planner.py','server/lineage_executor.py','server/lineage.py','server/lineage_diagnostics.py','server/lineage_startup.py','proposer/controller.py','proposer/adapter.py']:
 f=site/rel
 if f.exists(): (a/'source'/('installed__'+rel.replace('/','__')+'.txt')).write_text('\n'.join(f'{i}: {s}' for i,s in enumerate(f.read_text(encoding='utf-8').splitlines(),1)),encoding='utf-8')
from src.server.budget_planner import AdaptiveBudgetPlanner
planner=AdaptiveBudgetPlanner.from_env({'FLEXSDS_PROFILE_PATH':str(p)})
c=planner.adapter.get_controller('audit-cpu-only')
class ForbiddenCosts:
 def get_cost(self,*args,**kw): raise AssertionError('MEASURED COST QUERY')
c.cost_profile=ForbiddenCosts()
transitions=[]
for accepted,drafted in [(0,3),(2,2),(2,2)]:
 before=c.alpha_ema; c.update(accepted,drafted); dec=c.select_budget(batch_size=8)
 transitions.append(dict(evidence='CPU SIMULATION ONLY',accepted=accepted,drafted=drafted,before=before,after=c.alpha_ema,effective=c.get_effective_alpha(),selected_k=dec.k))
print('COST-QUERY PROBE: select_budget succeeded with get_cost forbidden; costs NOT queried')
print('CPU TRANSITIONS',json.dumps(transitions))
(a/'analysis/cpu-observations.json').write_text(json.dumps(dict(evidence_level='CPU SIMULATION ONLY; not GPU counters',cost_query_probe='get_cost replaced on this audit instance only with raising sentinel; no call occurred',transitions=transitions),indent=2),encoding='utf-8')
print('vllm installed locally:', next((d.version for d in md.distributions() if d.metadata['Name'].lower()=='vllm'),'NOT INSTALLED'))
