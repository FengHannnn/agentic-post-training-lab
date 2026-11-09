from pathlib import Path
import json,hashlib,zipfile,shutil,py_compile
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs')
for n in ('src/apt_v2/independent_recovery_test.py','src/apt_v2/dpo_recovery.py','src/apt_v2/recovery_diagnostics.py','scripts/test_v2_hard_recovery.py','scripts/audit_v2_hard_recovery_test.py','scripts/run_v2_hard_recovery_test.py'):py_compile.compile(str(root/n),doraise=True)
paths=sorted(set([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.json'),*root.glob('tests/test*.py'),
    *[root/n for n in ('README.md','PROJECT_REQUIREMENTS.md','V1_ARCHIVE_RECEIPT.json','bundle_manifest.json')]]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in paths};manifest=root/'hard_recovery_test_release_manifest.json'
manifest.write_text(json.dumps({'release':'v2-hard-recovery-independent-test-v1-20261103','files':files},indent=2)+'\n',encoding='utf8')
release=root/'artifacts/releases/hard-recovery-independent-test-v1-20261103';release.mkdir(parents=True,exist_ok=True)
archive=release/'v2-hard-recovery-independent-test-v1-20261103.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths+[manifest]:z.write(p,'v2/'+p.relative_to(root).as_posix())
    p=root/'reports/hard-recovery-independent-test-v1-20261103.md';z.write(p,'v2/reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,h in files.items():assert hashlib.sha256(z.read('v2/'+n)).hexdigest()==h
receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),
    'verification':{'new_tests_passed':9,'full_regression_initial_passed':128,'full_regression_initial_failed':1,
        'compatibility_message_fixed':True,'targeted_followup_passed':2,'unique_tests_covered':129,'compileall_passed':True},
    'training_enabled':False,'optimizer_updates':0,'test_tasks':4,'planned_complete_episodes':72,'first_recovery_actions':288,
    'novel_continuations':64,'calendar_sql_challenges':2,'max_wall_seconds':14400}
p=release/'v2-hard-recovery-independent-test-v1-20261103-receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
for item in (archive,p):shutil.copyfile(item,out/item.name)
print(json.dumps(receipt))
