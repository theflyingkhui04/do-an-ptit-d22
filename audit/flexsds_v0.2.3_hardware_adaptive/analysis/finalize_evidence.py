import json,hashlib,subprocess
from pathlib import Path
root=Path.cwd(); a=root/'audit/flexsds_v0.2.3_hardware_adaptive'
commands=[dict(command='python -m pytest tests/test_budget_controller.py tests/test_proposer_adapter.py tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py tests/test_lineage_startup.py tests/test_local_runner.py tests/test_modal_runner_config.py tests/test_modal_app_container_paths.py tests/test_serving_profile_guard.py tests/test_modal_image.py -q --tb=short',exit=1,passed=140,failed=5,skipped=89,raw='tests/focused-head.txt',imports='flexsds from installed site-packages; src.* and experiments.* from repository; filename focused-head.txt does NOT imply HEAD-only test coverage'),dict(command='python -m pytest tests/test_memory_pressure_rule.py tests/test_hardware_profiler_benchmark.py tests/test_vram_memory_lifecycle.py -q -ra --tb=short',exit=0,passed=39,failed=0,skipped=0,raw='tests/memory-profile.txt')]
(a/'tests/commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
(a/'analysis/gpu-gate.json').write_text(json.dumps(dict(decision='STOP BEFORE GPU',reason=['CLI metadata preflight blocked by sandbox socket denial','Installed Modal image construction omits required src.server/src.proposer/experiments','ON eager/compilation settings differ from OFF defaults'],paid_serving_launches=0,calibrations=0,deployments=0,app_ids=[],chosen_gpu='L4 intended; not allocated or frozen',gpu_fallback=False,stop_actions='None required: this audit launched no Modal resource',account_wide_container_state='UNKNOWN; not queried; no unrelated resource stopped'),indent=2),encoding='utf-8')
for rel in ['src/server/lineage.py','src/server/lineage_startup.py','src/server/lineage_diagnostics.py','src/flexsds/core/calibration.py','src/flexsds/core/calibration_worker.py']:
 p=root/rel
 (a/'source'/rel.replace('/','__')).with_suffix('.txt').write_text('\n'.join(f'{i}: {v}' for i,v in enumerate(p.read_text(encoding='utf-8').splitlines(),1)),encoding='utf-8')
cmd=['git','-c',f'safe.directory={root.as_posix()}']
result=subprocess.run(cmd+['status','--short'],capture_output=True,text=True)
(a/'environment/final-git-status.txt').write_text(result.stdout,encoding='utf-8')
print(result.stdout)
print('tracked diff bytes',len(subprocess.run(cmd+['diff','HEAD','--'],capture_output=True).stdout))
