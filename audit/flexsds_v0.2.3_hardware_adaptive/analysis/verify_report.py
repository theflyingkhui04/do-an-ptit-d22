from pathlib import Path
import json,hashlib,subprocess
root=Path.cwd(); a=root/'audit/flexsds_v0.2.3_hardware_adaptive'; report=a/'report/audit-report.md'
s=report.read_text(encoding='utf-8').replace('@ROOT@',root.as_posix()).replace('@AUDIT@',a.as_posix()).replace('@SITE@','D:/ProgramFiles/Python3_11/Lib/site-packages')
report.write_text(s,encoding='utf-8')
p=root/'experiments/runs/m2_03/boundary279_20261018T132244Z_15f4dae9/vllm-originals/v1/core/sched/scheduler.py'
(a/'source/captured-upstream-scheduler.py.txt').write_bytes(p.read_bytes())
request=Path(r'C:\Users\daibu\.codex\attachments\03fb77c5-b04f-441d-8da8-ad098982f182\pasted-text-1.txt')
(a/'environment/audit-instructions.txt').write_bytes(request.read_bytes())
print('primary table data rows:',sum(line.startswith('| ') for line in s.splitlines())-1)
print('sections:',[line for line in s.splitlines() if line.startswith('## ')])
import re
missing=[]
for target in re.findall(r'\]\(([^)]+)\)',s):
 path=re.sub(r':\d+$','',target)
 if not Path(path).exists():missing.append(target)
print('missing linked files:',missing)
manifest={str(f.relative_to(a)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
(a/'analysis/artifact-sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('hashed artifacts',len(manifest))
cmd=['git','-c',f'safe.directory={root.as_posix()}']; print(subprocess.run(cmd+['status','--short'],capture_output=True,text=True).stdout)
print('tracked diff bytes',len(subprocess.run(cmd+['diff','HEAD','--'],capture_output=True).stdout))
