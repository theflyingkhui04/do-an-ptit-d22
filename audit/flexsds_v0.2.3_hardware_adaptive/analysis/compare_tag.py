import json,subprocess,importlib.metadata as m,hashlib
from pathlib import Path
root=Path.cwd(); a=root/'audit/flexsds_v0.2.3_hardware_adaptive'; site=Path(m.distribution('flexsds').locate_file('')); cmd=['git','-c',f'safe.directory={root.as_posix()}']
paths=subprocess.run(cmd+['ls-tree','-r','--name-only','v0.2.3','src/flexsds','src/server','src/proposer'],capture_output=True,text=True,check=True).stdout.splitlines()
rows=[]
for s in paths:
 if not s.endswith('.py'):continue
 f=site/Path(s).relative_to('src'); raw=subprocess.run(cmd+['show','v0.2.3:'+s],capture_output=True,check=True).stdout
 rows.append(dict(path=s,installed_exists=f.exists(),installed_equals_tag=f.exists() and f.read_bytes()==raw))
extra=[str(f.relative_to(site)) for pkg in ['flexsds','server','proposer'] for f in (site/pkg).rglob('*.py') if 'src/'+f.relative_to(site).as_posix() not in paths]
result=dict(tag_sha=subprocess.run(cmd+['rev-parse','v0.2.3'],capture_output=True,text=True).stdout.strip(),python_files=len(rows),mismatches=[r for r in rows if not r['installed_equals_tag']],extra_installed_python_files=extra)
(a/'environment/tag-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
print('PROFILE GROUP IDENTITIES')
d=json.loads((a/'profiles/profile_sm_89_remote_l4_fcda5486b12d_full.json').read_bytes())
for i,g in enumerate(d['evidence']['groups']):print(i,{k:v for k,v in g.items() if k not in ['engine_kwargs','records']})
