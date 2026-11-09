import ctypes
_sqlite_runtime=ctypes.WinDLL(r'C:/Users/app/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/DLLs/sqlite3.dll')
import sys,json,hashlib,sqlite3,math,csv,io
from pathlib import Path
original=Path(sys.argv[1]);dest=Path(sys.argv[2]);run='hard-recovery-dpo-matched-v1-20261103';out=dest/'extracted'/run
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');pf=read(out/'preflight.json');sources=read(out/'run_manifest.json')['source_sha256'];release=read(original/'matched_dpo_release_manifest.json')['files']
assert cfg==read(out/'source/configs/hard-recovery-dpo-matched-v1.json')==read(original/'configs/hard-recovery-dpo-matched-v1.json')
for n,h in sources.items():assert sha(out/'source'/n)==h==sha(original/n)==release[n],n
assert sources==binding['source_sha256']
def prior(rel):
    parts=Path(rel).parts;return original/'artifacts/reviews'/parts[1]/'extracted'/parts[1]/Path(*parts[2:])
for rel,h in pf['provenance'].items():assert sha(out/'provenance'/rel)==h==sha(prior(rel)),rel
for rel,h in binding['provenance'].items():assert sha(out/'provenance'/rel)==h==sha(prior(rel)),rel
for b,key in [('parent','initial_run'),('weighted_sft','comparison_run')]:
    p=original/'artifacts/reviews'/cfg[key]/'extracted'/cfg[key]/'final'
    for n,h in pf['checkpoints'][b].items():assert sha(p/n)==h,(b,n)
for n,key in [('config.json','config_sha256'),('weights.json','weights_sha256')]:assert sha(out/n)==binding[key]
assert sha(out/'config.json')==pf['config_sha256'] and sha(out/'recovery_contexts.jsonl')==pf['contexts_sha256']
assert sha(out/'pairs.jsonl')==cfg['pairs_sha256']==binding['pairs_sha256']
identity=read(out/'model_identity.json');assert identity['files_sha256']==pf['model_files']
assert identity==read(prior('artifacts/'+cfg['initial_run']+'/model_identity.json'))
for n,h in identity['files_sha256'].items():
    if n.endswith(('.json','.txt','.jinja')):assert sha(out/'tokenizer'/n)==h,n
for n in ['final']+[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']]:
    c=read(out/n/'complete.json');assert read(out/n/'binding.json')==binding
    if n=='final':
        for f,h in c['adapter_sha256'].items():assert sha(out/n/'adapter'/f)==h,f
    else:
        assert c['steps']==int(n.split('-')[1]) and c['cursor']==c['steps']//43*169 and c['total_sft_updates']==255 and c['total_dpo_updates']==11+c['steps']
status=read(out/'run_status.json');assert status['error'] is None and all(s['exit_code']==0 for s in status['stages']) and status['elapsed_seconds']<cfg['max_wall_seconds']
print('Frozen source, provenance, adapters, tokenizer, input binding verified; SQLite '+sqlite3.sqlite_version,flush=True)
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
from transformers import AutoTokenizer
from dataclasses import asdict
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_recovery_training import verify_reference,validate_training_pairs
from apt_v2.hard_recovery_sft import recovery_weights
from apt_v2.matched_dpo import verify_matched_updates
from audit_v2_matched_dpo import audit_results
from matched_dpo_common import load,tasks,contexts
assert read(out/'development_tasks.json')==[asdict(t) for t in tasks()] and load(out/'recovery_contexts.jsonl')==contexts()
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
pairs=load(out/'pairs.jsonl');validate_training_pairs(pairs);weights,ws=recovery_weights(pairs,cfg['hard_weight']);assert read(out/'weights.json')=={'weights':weights,'summary':ws}
encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ref=read(out/'reference_scores.json');verify_reference(pairs,encoded,ref,cfg['adapter_sha256'])
for name in ('before_pair_scores.json','after_pair_scores.json'):
    scores=read(out/name);ms=scores['metrics'];assert scores['pairs']==len(ms)==169
    for k,v in scores['means'].items():assert math.isfinite(v) and abs(sum(m[k] for m in ms)/169-v)<1e-8
    assert abs(sum(w*m['loss'] for w,m in zip(weights,ms))/169-scores['weighted_mean_loss'])<1e-8
    for m,e,r in zip(ms,encoded,ref['scores']):
        margin=m['chosen_mean_logp']*len(e['chosen'])-m['rejected_mean_logp']*len(e['rejected'])-sum(r['chosen'])+sum(r['rejected'])
        z=-cfg['beta']*m['relative_margin'];softplus=max(0,z)+math.log1p(math.exp(-abs(z)))
        assert all(math.isfinite(v) for v in m.values()) and abs(softplus-m['loss'])<1e-4 and abs(margin-m['relative_margin'])<.002
    if name.startswith('before'):assert abs(scores['means']['loss']-math.log(2))<1e-5 and max(abs(m['relative_margin']) for m in ms)<1e-4
train=read(out/'training_summary.json');final=read(out/'final/complete.json');verify_matched_updates(cfg,final,train)
assert load(out/'training_steps.jsonl')==train['history'] and train['reference_scores_sha256']==sha(out/'reference_scores.json') and train['adapter_tensors_changed']==112 and train['weight_summary']==ws
assert train['chosen_response_tokens_per_epoch']==sum(len(e['chosen']) for e in encoded) and train['rejected_response_tokens_per_epoch']==sum(len(e['rejected']) for e in encoded)
print('169 token-bound reference scores, weighted objective, 258 updates and checkpoint ancestry verified',flush=True)
summary,calendar,evidence=audit_results(out,cfg,tok)
assert summary==read(out/'comparison_summary.json')
assert dict(summary,status='awaiting_local_review',**evidence)==read(out/'evaluation_summary.json')
assert calendar==read(out/'calendar_sql_validation.json')
cloud=read(out/'audit.json');assert cloud==dict(passed=True,optimizer_updates=258,source_sft_updates=255,total_dpo_updates=269,reference_inputs_verified=169,**evidence)
stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(['branch','mode','kind','episodes','successes'])
for branch,modes in summary['branches'].items():
    for mode,s in modes.items():
        for kind in ('complete','recovery'):writer.writerow([branch,mode,kind,s[kind]['episodes'],s[kind]['successes']])
assert stream.getvalue()==(out/'results.csv').read_bytes().decode('utf-8-sig')
result={'status':'passed_local_review','archive_sha256':sha(dest/(run+'-review.zip')),'source_files_verified':len(sources),'source_matches_release_and_local':True,'provenance_files_verified':len(pf['provenance']),'tokenizer_verified':True,'reference_pairs_verified':169,'final_adapter_sha256':sha(out/'final/adapter/adapter_model.safetensors'),'actual_optimizer_state':'retained on cloud; excluded from review archive; cloud audit passed state checks','sqlite_replay_version':sqlite3.sqlite_version,'summary_recomputed_matches':True,'calendar_recomputed_matches':True,'csv_recomputed_matches':True,'training_update_history_verified':258,**evidence,'elapsed_seconds':status['elapsed_seconds']}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
