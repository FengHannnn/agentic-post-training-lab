from pathlib import Path
import json,hashlib,zipfile,shutil
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs')
paths=sorted(set([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.json'),*root.glob('tests/test*.py'),
    root/'diagnostics/diagnose_sft_v1.py',root/'diagnostics/review_sft_diagnostics_v1.py',
    *[root/n for n in ('README.md','PROJECT_REQUIREMENTS.md','V1_ARCHIVE_RECEIPT.json','bundle_manifest.json','sft_release_manifest.json','sft_diagnostics_release_manifest.json')]]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in paths}
manifest=root/'sql_sft_release_manifest.json'
manifest.write_text(json.dumps({'release':'v2-sql-sft-weighted-v1-20261101','files':files},indent=2)+'\n',encoding='utf8')
release=root/'artifacts/releases/sql-sft-weighted-v1-20261101';release.mkdir(parents=True,exist_ok=True)
archive=release/'v2-sql-sft-weighted-v1-20261101.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths+[manifest]:z.write(p,'v2/'+p.relative_to(root).as_posix())
    p=root/'reports/sql-sft-weighted-v1-20261101.md';z.write(p,'v2/reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,value in files.items():assert hashlib.sha256(z.read('v2/'+name)).hexdigest()==value
receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),'tests_passed':66,
    'branch':'SQL-weighted mixed SFT','parent_steps':63,'planned_extra_steps':63,'sql_action_weight':4}
receipt_path=release/'v2-sql-sft-weighted-v1-20261101-receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
for p in (archive,receipt_path):shutil.copyfile(p,out/p.name)
print(json.dumps(receipt))
