from pathlib import Path
import json,zipfile,shutil
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='hard-recovery-sft-weighted-v1-20261103';src=Path(r'C:/Users/app/Downloads/hard-recovery-sft-weighted-v1-20261103-review.zip')
dest=ROOT/'artifacts/reviews'/run;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest/src.name)
with zipfile.ZipFile(src) as z:
 assert z.testzip() is None
 target=(dest/'extracted').resolve()
 for n in z.namelist():assert (target/n).resolve().is_relative_to(target),n
 z.extractall(target);print('zip_members',len(z.namelist()),'bytes',src.stat().st_size)
out=dest/'extracted'/run
print('files',[p.name for p in out.iterdir()])
for name in ('run_status.json','training_summary.json','audit.json','evaluation_summary.json','comparison_summary.json','stochastic_comparison.json'):
 p=out/name
 if not p.exists():print(name,'MISSING');continue
 d=json.loads(p.read_text(encoding='utf8'))
 if name=='training_summary.json':d={k:v for k,v in d.items() if k!='history'}
 print(name,json.dumps(d,ensure_ascii=False))
p=out/'checkpoint_progress.json'
if p.exists():
 d=json.loads(p.read_text(encoding='utf8'))
 for c in d['checkpoints']:
  s=c['train_greedy']['overall'];print('CHECKPOINT',c['new_sft_updates'],json.dumps(s,ensure_ascii=False))
p=out/'diagnostic_summary.json'
if p.exists():
 d=json.loads(p.read_text(encoding='utf8'))
 for b,splits in d.items():
  for split,modes in splits.items():
   for mode,g in modes.items():print('RECOVERY',b,split,mode,json.dumps(g['overall'],ensure_ascii=False))
