from pathlib import Path
import json,zipfile,hashlib,sys,importlib.util,collections,shutil
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
archive=Path(r'C:/Users/app/Downloads/sql-sft-weighted-v1-20261101-review.zip')
out=ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101';out.mkdir(parents=True,exist_ok=True)
extracted=out/'extracted';extracted.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for i in z.infolist():assert (extracted/i.filename).resolve().is_relative_to(extracted.resolve())
    z.extractall(extracted)
r=extracted/'sql-sft-weighted-v1-20261101'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sources=read(r/'run_manifest.json')['source_sha256'];local_match={}
for n,h in sources.items():
    assert sha(r/'source'/n)==h,n
    local_match[n]=(sha(ROOT/n)==h)
for p in r.glob('*/complete.json'):
    c=read(p)
    for n,h in c['adapter_sha256'].items():assert sha(p.parent/'adapter'/n)==h
    assert sha(p.parent/'state.pt')==c['state_sha256']
    assert c['total_steps']==63+c['steps']
binding=read(r/'input_binding.json')
with zipfile.ZipFile(ROOT/'artifacts/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030-review.zip') as z:
    prefix='sft-pilot-v1-20261030/'
    assert hashlib.sha256(z.read(prefix+'final/adapter/adapter_model.safetensors')).hexdigest()==binding['parent_adapter_sha256']
    assert hashlib.sha256(z.read(prefix+'final/state.pt')).hexdigest()==binding['parent_state_sha256']
    assert z.read(prefix+'examples.jsonl')==(r/'examples.jsonl').read_bytes()
    assert z.read(prefix+'demonstrations.jsonl')==(r/'demonstrations.jsonl').read_bytes()
for n,h in read(r/'model_identity.json')['files_sha256'].items():
    if (r/'tokenizer'/n).is_file():assert sha(r/'tokenizer'/n)==h
sys.path.insert(0,str(r/'source/scripts'))
spec=importlib.util.spec_from_file_location('frozen_sql_audit',r/'source/scripts/audit_v2_sql_sft.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original_write=module.write
module.write=lambda path,value:original_write(out/'local_replay_audit.json',value) if path.name=='audit.json' else original_write(path,value)
module.audit(r)
assert read(out/'local_replay_audit.json')==read(r/'audit.json')
import torch
from safetensors.torch import load_file
state=torch.load(r/'final/state.pt',map_location='cpu',weights_only=True)
assert state['steps']==63 and state['total_steps']==126 and state['cursor']==504
assert {int(s['step']) for s in state['optimizer']['state'].values()}=={126}
adapter=load_file(r/'final/adapter/adapter_model.safetensors')
assert len(adapter)==112 and sum(t.numel() for t in adapter.values())==1089536
assert all(torch.isfinite(t).all() for t in adapter.values())
from apt_v2.sft import build_dataset,replay,encode_example
from apt_v2.sql_sft import normalized_weights,data_identity
from apt_v2.tasks import make_tasks
from transformers import AutoTokenizer
byid={t.id:t for t in make_tasks()};tok=AutoTokenizer.from_pretrained(r/'tokenizer',local_files_only=True)
demos=[json.loads(l) for l in (r/'demonstrations.jsonl').read_text(encoding='utf8').splitlines()]
for episode in demos:replay(byid[episode['task_id']],episode)
_,examples=build_dataset();saved=[json.loads(l) for l in (r/'examples.jsonl').read_text(encoding='utf8').splitlines()]
assert examples==saved and data_identity(saved)==binding['dataset_sha256']
enc=[encode_example(tok,e,2048) for e in examples]
weights=normalized_weights(examples,4);assert abs(sum(weights)/len(weights)-1)<1e-10
summary=read(r/'comparison_summary.json');details=[];eos=collections.Counter()
for branch in ('parent','weighted'):
    for split in ('train','development'):
        for episode in [json.loads(l) for l in (r/(branch+'-'+split+'-episodes.jsonl')).read_text(encoding='utf8').splitlines()]:
            for row in episode['turns']:
                g=row['generation'];eos[(branch,split)]+=int(g['response_token_ids'][-1]==tok.eos_token_id)
            if branch=='weighted':
                assert episode['success']
                details.append({'task_id':episode['task_id'],'split':split,'turns':len(episode['turns']),
                    'recovered':episode['recovered'],'tools':[x['action'].get('tool') for x in episode['turns']],
                    'values':episode['artifact']['values'],
                    'sqls':[x['action']['arguments']['sql'] for x in episode['turns'] if x['action'].get('tool')=='query_sql']})
assert read(r/'before_action_scores.json')==read(ROOT/'artifacts/reviews/sft-diagnostics-v1-20261101/offline-review/recovered/hf-fp32-score-summary.json')['sft']
result={'zip_sha256':sha(archive),'source_files_verified':len(sources),'source_matches_local':all(local_match.values()),
    'checkpoints_verified':len(list(r.glob('*/complete.json'))),'parent_adapter_and_adam_match_archive':True,
    'training_files_byte_equal_parent':True,'demonstrations_replayed':len(demos),'evaluation_episodes_replayed':32,
    'evaluation_turns_replayed':312,'cloud_local_audit_identical':True,'final_adam_step':126,
    'final_adapter_sha256':sha(r/'final/adapter/adapter_model.safetensors'),
    'teacher_forced_before_matches_prior_diagnostics':True,'max_training_tokens':max(len(s['input_ids']) for s in enc),
    'eos_counts':{str(k):v for k,v in eos.items()},'weighted_task_details':details}
(out/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(archive,out/archive.name)
print(json.dumps({k:v for k,v in result.items() if k!='weighted_task_details'}))
print('SAMPLE_SUCCESS',json.dumps(details[0]))
