import sys,json,hashlib,subprocess,importlib.metadata as m
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
a=Path('audit/flexsds_v0.2.3_hardware_adaptive'); root=Path.cwd(); d=m.distribution('flexsds'); site=Path(d.locate_file(''))
(a/'environment/package-metadata.txt').write_text(d.read_text('METADATA') or '',encoding='utf-8')
print('direct_url:',d.read_text('direct_url.json'))
def git(*args):
 return subprocess.run(['git','-c',f'safe.directory={root.as_posix()}',*args],capture_output=True,text=True).stdout.strip()
tag=git('rev-parse','v0.2.3'); print('tag v0.2.3:',tag)
rows=[]
for pkg in ['flexsds','server','proposer']:
 for f in sorted((root/'src'/pkg).rglob('*.py')):
  rel=f.relative_to(root/'src'); installed=site/rel
  b=f.read_bytes(); ib=installed.read_bytes() if installed.exists() else None
  tb=subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','show',f'v0.2.3:src/{rel.as_posix()}'],capture_output=True).stdout if tag else None
  rows.append(dict(path=str(rel),installed_exists=ib is not None,head_sha256=hashlib.sha256(b).hexdigest(),installed_sha256=hashlib.sha256(ib).hexdigest() if ib else None,installed_equals_head=b==ib,installed_equals_tag=ib==tb if tag else None))
(a/'environment/source-comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('comparison files',len(rows),'different/missing',sum(not r['installed_equals_head'] for r in rows),'installed vs tag different',sum(not r['installed_equals_tag'] for r in rows))
for r in rows:
 if not r['installed_equals_head']: print(r['path'], 'installed_exists=',r['installed_exists'])
p=Path(r'C:\Users\daibu\.flexsds\profiles\profile_sm_89_remote_l4_fcda5486b12d_full.json'); raw=p.read_bytes(); data=json.loads(raw)
(a/'profiles'/p.name).write_bytes(raw)
summary={k:v for k,v in data.items() if k not in ['evidence','raw_measurements','measurements','samples']}
(a/'profiles/profile-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('profile keys',list(data)); print('profile sha256',hashlib.sha256(raw).hexdigest())
for k in ['schema','schema_version','measurement_status','metadata','models','draft_method','runtime','grid','measurement','measurement_config','cost_semantics','evidence_sha256']:
 print(k,json.dumps(data.get(k),ensure_ascii=False))
for f in ['src/flexsds/cli/main.py','src/flexsds/server/local_runner.py','src/flexsds/server/modal_runner.py','src/flexsds/server/modal_app.py','src/flexsds/server/modal_image.py','src/flexsds/core/cost_model.py','src/server/adaptive_scheduler.py','src/server/budget_planner.py','src/server/lineage_executor.py','src/server/execution_lineage.py','src/proposer/adapter.py','src/proposer/controller.py']:
 p=Path(f)
 if p.exists(): (a/'source'/f.replace('/','__')).with_suffix('.txt').write_text('\n'.join(f'{i}: {s}' for i,s in enumerate(p.read_text(encoding='utf-8').splitlines(),1)),encoding='utf-8')
