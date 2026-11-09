from pathlib import Path
import json,hashlib,zipfile,shutil,py_compile
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs')
for n in ('src/apt_v2/dpo_recovery_training.py','scripts/train_v2_dpo_recovery_long.py','scripts/evaluate_v2_dpo_recovery.py','scripts/audit_v2_dpo_recovery.py','scripts/run_v2_dpo_recovery_train_long.py','scripts/streamed_process.py'):py_compile.compile(str(root/n),doraise=True)
paths=sorted(set([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.json'),*root.glob('tests/test*.py'),
    *[root/n for n in ('README.md','PROJECT_REQUIREMENTS.md','V1_ARCHIVE_RECEIPT.json','bundle_manifest.json')]]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in paths}
manifest=root/'dpo_recovery_train_v2_release_manifest.json'
manifest.write_text(json.dumps({'release':'v2-dpo-recovery-train-v2-20261102','files':files},indent=2)+'\n',encoding='utf8')
release=root/'artifacts/releases/dpo-recovery-train-v2-20261102';release.mkdir(parents=True,exist_ok=True)
archive=release/'v2-dpo-recovery-train-v2-20261102.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths+[manifest]:z.write(p,'v2/'+p.relative_to(root).as_posix())
    p=root/'reports/dpo-recovery-train-v2-20261102.md';z.write(p,'v2/reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,h in files.items():assert hashlib.sha256(z.read('v2/'+n)).hexdigest()==h
receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),
    'tests_passed':102,'planned_dpo_updates':129,'training_pairs':169,'pair_exposures':507,'max_wall_seconds':10800,'planned_evaluation_episodes':64,'planned_recovery_actions':224}
p=release/'v2-dpo-recovery-train-v2-20261102-receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
for item in (archive,p):shutil.copyfile(item,out/item.name)
print(json.dumps(receipt))
