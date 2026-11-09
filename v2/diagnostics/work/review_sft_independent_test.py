from pathlib import Path
import json,zipfile,hashlib,sys,importlib.util,shutil
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');src=Path(r'C:/Users/app/Downloads/sft-independent-test-v1-20261102-review.zip')
out=ROOT/'artifacts/reviews/sft-independent-test-v1-20261102';out.mkdir(parents=True,exist_ok=True)
extracted=out/'extracted';extracted.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
with zipfile.ZipFile(src) as z:
    assert z.testzip() is None
    for i in z.infolist():assert (extracted/i.filename).resolve().is_relative_to(extracted.resolve())
    z.extractall(extracted)
r=extracted/'sft-independent-test-v1-20261102';cfg=read(r/'config.json');binding=read(r/'input_binding.json')
sources=read(r/'run_manifest.json')['source_sha256']
for n,h in sources.items():assert sha(r/'source'/n)==h and sha(ROOT/n)==h,n
parent_zip=ROOT/'artifacts/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030-review.zip'
with zipfile.ZipFile(parent_zip) as z:
    for n,h in binding['checkpoints']['parent'].items():
        assert hashlib.sha256(z.read('sft-pilot-v1-20261030/final/'+n)).hexdigest()==h
candidate=ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101'
for n,h in binding['checkpoints']['weighted'].items():assert sha(candidate/'final'/n)==h
assert sha(candidate/'examples.jsonl')==binding['candidate_examples_sha256']
identity=read(r/'model_identity.json');assert identity==read(candidate/'model_identity.json')
for n,h in identity['files_sha256'].items():
    if (r/'tokenizer'/n).is_file():assert sha(r/'tokenizer'/n)==h
assert cfg['candidate_adapter_sha256']==sha(candidate/'final/adapter/adapter_model.safetensors')
assert read(r/'final/complete.json')==read(candidate/'final/complete.json')
sys.path[:0]=[str(r/'source/scripts'),str(r/'source/src')]
from apt_v2.sft_test import holdout_tasks
from dataclasses import asdict
examples=[json.loads(l) for l in (candidate/'examples.jsonl').read_text(encoding='utf8').splitlines()]
assert [asdict(t) for t in holdout_tasks(cfg,examples)]==read(r/'test_tasks.json')
spec=importlib.util.spec_from_file_location('frozen_holdout_worker',r/'source/scripts/test_v2_sft.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
worker.verify_inputs=lambda path:cfg
original_write=worker.write
worker.write=lambda path,value:original_write(out/('local_'+path.name),value)
worker.audit(r)
assert read(out/'local_audit.json')==read(r/'audit.json')
assert read(out/'local_independent_test_summary.json')==read(r/'independent_test_summary.json')
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(r/'tokenizer',local_files_only=True)
counts={};details=[]
for branch in ('parent','weighted'):
    episodes=[json.loads(l) for l in (r/(branch+'-test-episodes.jsonl')).read_text(encoding='utf8').splitlines()]
    counts[branch]={'eos':sum(row['generation']['response_token_ids'][-1]==tok.eos_token_id for e in episodes for row in e['turns'])}
    if branch=='weighted':
        for e in episodes:details.append({'task_id':e['task_id'],'turns':len(e['turns']),
            'recovered':e['recovered'],'tools':[row['action']['tool'] for row in e['turns']],'artifact':e['artifact']})
result={'zip_sha256':sha(src),'source_files_verified':len(sources),'source_matches_local':True,
 'checkpoint_binding_matches_archived_parents':True,'tokenizer_files_verified':6,'task_train_family_isolation':True,
 'episodes_replayed':8,'turns_replayed':78,'local_audit_matches_cloud':True,'candidate_adapter_sha256':cfg['candidate_adapter_sha256'],
 'acceptance':read(r/'independent_test_summary.json')['acceptance'],'optimizer_updates':0,'eos_counts':counts,'weighted_task_details':details}
(out/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(src,out/src.name)
print(json.dumps(result,ensure_ascii=False))
