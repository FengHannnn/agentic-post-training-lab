from pathlib import Path
import zipfile,shutil,json
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='recovery-context-diagnostics-v1-20261102'
source=Path(r'C:/Users/app/Downloads/recovery-context-diagnostics-v1-20261102-review.zip')
dest=ROOT/'artifacts/reviews'/run;dest.mkdir(parents=True,exist_ok=True)
shutil.copy2(source,dest/source.name)
with zipfile.ZipFile(source) as z:
 assert z.testzip() is None
 target=(dest/'extracted').resolve()
 for n in z.namelist():
  assert (target/n).resolve().is_relative_to(target),n
 z.extractall(target)
 print('files',len(z.namelist()),'zip_bytes',source.stat().st_size)
out=dest/'extracted'/run
print('files',[p.name for p in out.iterdir()])
for n in ('run_status.json','prepare_summary.json','audit.json'):
 p=out/n
 print(n,json.loads(p.read_text()) if p.exists() else 'MISSING')
p=out/'diagnostic_summary.json'
if p.exists():
 d=json.loads(p.read_text())
 for branch,splits in d.items():
  for split,modes in splits.items():
   for mode,groups in modes.items():print(branch,split,mode,json.dumps(groups['overall'],ensure_ascii=False))
