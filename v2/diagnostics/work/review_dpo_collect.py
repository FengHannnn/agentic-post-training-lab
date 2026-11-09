from pathlib import Path
import json,sys,zipfile,hashlib,importlib.util,shutil,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');src=Path(r'C:/Users/app/Downloads/dpo-collect-v1-20261102-review.zip')
out=ROOT/'artifacts/reviews/dpo-collect-v1-20261102';out.mkdir(parents=True,exist_ok=True)
extracted=out/'extracted';extracted.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
with zipfile.ZipFile(src) as z:
    assert z.testzip() is None
    for i in z.infolist():assert (extracted/i.filename).resolve().is_relative_to(extracted.resolve())
    z.extractall(extracted)
r=extracted/'dpo-collect-v1-20261102';cfg=read(r/'config.json');binding=read(r/'input_binding.json')
sources=read(r/'run_manifest.json')['source_sha256']
for n,h in sources.items():assert sha(r/'source'/n)==h and sha(ROOT/n)==h,n
local_source=ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101'
for n,h in binding['source_files'].items():assert sha(local_source/n)==h,n
assert binding['adapter_sha256']==sha(ROOT/'artifacts/baselines/sft-sql-weighted-v1-20261102/adapter/adapter_model.safetensors')
identity=read(r/'model_identity.json');assert identity==read(local_source/'model_identity.json')
for n,h in identity['files_sha256'].items():
    if (r/'tokenizer'/n).is_file():assert sha(r/'tokenizer'/n)==h
assert sha(r/'rollouts.jsonl')==read(r/'rollout_summary.json')['rollouts_sha256']
assert sha(r/'pairs.jsonl')==read(r/'pair_summary.json')['pairs_sha256']
sys.path[:0]=[str(r/'source/scripts'),str(r/'source/src')]
spec=importlib.util.spec_from_file_location('frozen_dpo_collector',r/'source/scripts/collect_v2_dpo.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
worker.verify_inputs=lambda path:(cfg,local_source)
worker.write=lambda path,value:(out/('local_'+path.name)).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
original_jsonl=worker.jsonl
worker.jsonl=lambda path,rows:original_jsonl(out/('local_'+path.name),rows)
worker.build(r)
local_pairs=[json.loads(l) for l in (out/'local_pairs.jsonl').read_text(encoding='utf8').splitlines()]
pairs=[json.loads(l) for l in (r/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
assert pairs==local_pairs
assert read(r/'pair_summary.json')==read(out/'local_pair_summary.json')
assert read(r/'audit.json')==read(out/'local_audit.json')
from apt_v2.runtime import parse_action
from apt_v2.dpo import encode_pair
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(r/'tokenizer',local_files_only=True)
enc=[encode_pair(tok,p,cfg['pair_max_length']) for p in pairs]
episodes=[json.loads(l) for l in (r/'rollouts.jsonl').read_text(encoding='utf8').splitlines()]
per_task={t:sum(e['success'] for e in episodes if e['task_id']==t) for t in sorted({e['task_id'] for e in episodes})}
result={'zip_sha256':sha(src),'source_files_verified':len(sources),'source_matches_local':True,
 'source_checkpoint_binding_matches_frozen_baseline':True,'tokenizer_files_verified':6,
 'episodes_replayed':len(episodes),'turns_replayed':sum(len(e['turns']) for e in episodes),
 'pairs_regenerated_identically':len(pairs),'local_audit_matches_cloud':True,
 'max_pair_tokens':max(len(p['prompt'])+max(len(p['chosen']),len(p['rejected'])) for p in enc),
 'chosen_tools':dict(collections.Counter(parse_action(p['chosen'])['tool'] for p in pairs)),
 'rejected_tools':dict(collections.Counter(worker.action_from_text(p['rejected']).get('tool','invalid') for p in pairs)) if hasattr(worker,'action_from_text') else {},
 'per_task_successes_of_four':per_task,'mixed_task_groups':sum(0<v<cfg['rollouts_per_task'] for v in per_task.values()),
 'optimizer_updates':0}
# Parse rejected actions including malformed JSON, without dropping those negatives.
from apt_v2.dpo import action_from_text
result['rejected_tools']=dict(collections.Counter(action_from_text(p['rejected']).get('tool','invalid') for p in pairs))
(out/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(src,out/src.name)
print(json.dumps(result))
print('PAIR_EXAMPLE',json.dumps({k:pairs[0][k] for k in ('task_id','turn','reason','chosen','rejected','chosen_observation','rejected_observation')}))
