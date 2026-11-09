from pathlib import Path
import json,hashlib,zipfile,shutil
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs')
paths=sorted(set([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.json'),*root.glob('tests/test*.py'),
    *[root/n for n in ('README.md','PROJECT_REQUIREMENTS.md','V1_ARCHIVE_RECEIPT.json','bundle_manifest.json')]]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in paths}
manifest=root/'recovery_context_diagnostic_release_manifest.json'
manifest.write_text(json.dumps({'release':'v2-recovery-context-diagnostics-v1-20261102','files':files},indent=2)+'\n',encoding='utf8')
release=root/'artifacts/releases/recovery-context-diagnostics-v1-20261102';release.mkdir(parents=True,exist_ok=True)
archive=release/'v2-recovery-context-diagnostics-v1-20261102.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths+[manifest]:z.write(p,'v2/'+p.relative_to(root).as_posix())
    p=root/'reports/recovery-context-diagnostics-v1-20261102.md';z.write(p,'v2/reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,h in files.items():assert hashlib.sha256(z.read('v2/'+n)).hexdigest()==h
receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),
    'tests_passed':113,'training_updates':0,'train_contexts':84,'development_contexts':28,'planned_native_actions':1344,'max_wall_seconds':14400}
p=release/'v2-recovery-context-diagnostics-v1-20261102-receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
for item in (archive,p):shutil.copyfile(item,out/item.name)
print(json.dumps(receipt))
