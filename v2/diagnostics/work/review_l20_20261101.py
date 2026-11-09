from pathlib import Path
import zipfile,json,hashlib,shutil
src=Path(r'C:/Users/app/Downloads/v2-4090d-environment-review-20261030-e44637e9.zip')
root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');base='environment-4090d-20261030-e44637e9/'
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(src) as z:
    assert z.testzip() is None
    manifest=json.loads(z.read(base+'files_sha256.json'))
    for n,h in manifest.items():assert sha(z.read(base+n))==h,n
    s=json.loads(z.read(base+'summary.json'))
    for key in ('inventory','model','train','vllm'):
        assert json.loads(z.read(base+'stage_'+key+'.json'))==s['stages'][key]
        assert s['stages'][key]['passed']
    source_equal=sha(z.read(base+'check_environment.py'))==sha((root/'diagnostics/check_environment.py').read_bytes())
    records=json.loads(z.read(base+'vllm_records.json'))
    print('VLLM_RECORDS',json.dumps(records)[:2500])
result={'zip_sha256':sha(src.read_bytes()),'member_sha256_verified':len(manifest),
    'stage_summary_consistent':True,'check_script_matches_local':source_equal,'all_checks_passed':True,
    'gpu':s['stages']['inventory']['device'],'created_at_utc':s['created_at_utc']}
dest=root/'artifacts/reviews/environment-l20-20261101-e44637e9';dest.mkdir(parents=True,exist_ok=True)
shutil.copyfile(src,dest/src.name)
(dest/'local_review_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
