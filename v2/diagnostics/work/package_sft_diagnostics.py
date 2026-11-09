from pathlib import Path
import hashlib,json,zipfile,shutil,py_compile
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs')
for p in [root/'diagnostics/diagnose_sft_v1.py',root/'scripts/run_v2_sft_diagnostics.py']:
    py_compile.compile(str(p),doraise=True)
# Original pilot source identities stay unchanged.
old=Path(r'C:/Users/app/Documents/Codex/2026-10-30/new-chat/work/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030/run_manifest.json')
for n,h in json.loads(old.read_text(encoding='utf8'))['source_sha256'].items():
    assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
paths=sorted(set([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.json'),
    *root.glob('tests/test*.py'),root/'diagnostics/diagnose_sft_v1.py',
    *[root/n for n in ['README.md','PROJECT_REQUIREMENTS.md','V1_ARCHIVE_RECEIPT.json','bundle_manifest.json','sft_release_manifest.json']]]))
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
manifest=root/'sft_diagnostics_release_manifest.json'
manifest.write_text(json.dumps({'release':'v2-sft-diagnostics-v1-20261101','files':files},indent=2)+'\n',encoding='utf8')
release=root/'artifacts/releases/sft-diagnostics-v1-20261101';release.mkdir(parents=True,exist_ok=True)
archive=release/'v2-sft-diagnostics-v1-20261101.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths+[manifest]:z.write(p,'v2/'+p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,h in files.items():assert hashlib.sha256(z.read('v2/'+n)).hexdigest()==h
receipt={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'bytes':archive.stat().st_size,'files':len(files),'tests_passed':58,'original_pilot_sources_unchanged':True}
receipt_path=release/'v2-sft-diagnostics-v1-20261101-receipt.json'
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
for p in (archive,receipt_path):shutil.copyfile(p,out/p.name)
print(json.dumps(receipt))
