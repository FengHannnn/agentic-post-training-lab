import zipfile,json,pathlib,shutil
root=pathlib.Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='chosen-only-sft-v1-20261102'
dest=root/'artifacts/reviews'/run
dest.mkdir(parents=True,exist_ok=True)
zsrc=pathlib.Path(r'C:/Users/app/Downloads/chosen-only-sft-v1-20261102-review.zip')
shutil.copy2(zsrc,dest/zsrc.name)
with zipfile.ZipFile(zsrc) as z:
 assert z.testzip() is None
 out=(dest/'extracted').resolve()
 for n in z.namelist():
  p=(out/n).resolve();assert p.is_relative_to(out)
 z.extractall(out)
 print('members',len(z.namelist()))
 print('top',z.namelist()[:15])
for p in (dest/'extracted'/run).glob('*.json'):
 d=json.loads(p.read_text())
 if p.name=='training_summary.json':d={k:v for k,v in d.items() if k!='history'}
 print(p.name,json.dumps(d,ensure_ascii=False)[:22000])
