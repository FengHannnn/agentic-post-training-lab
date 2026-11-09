import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.tasks import make_tasks
from apt_v2.dpo import pair_id
from apt_v2.dpo_recovery import SCENARIOS,build_context,make_pair
from apt_v2.dpo_recovery_training import validate_training_pairs,training_batches,verify_reference,recovery_result,summarize_recovery,optimize_epoch
from apt_v2.dpo_training import reference_scores
from test_dpo_v2 import CharTokenizer

def pair_fixtures():
    t=make_tasks()[0];pairs=[]
    for i,source in enumerate(['frozen_sft_sample']*44+['scripted_failure']*84+['frozen_dpo_sample']*41):
        p={'task_id':t.id,'family_id':t.family_id,'split':'train','messages':[{'role':'user','content':'fixture '+str(i)}],
            'chosen':'{"tool":"read_policy","arguments":{}}','rejected':'bad '+str(i),'rejected_source':source}
        if source!='frozen_sft_sample':p.update(scenario='format_repair',context_id=t.id+':format_repair')
        p['pair_id']=pair_id(p);pairs.append(p)
    return pairs

def test_reviewed_mix_validation_and_tail_batch_schedule():
    pairs=pair_fixtures();assert validate_training_pairs(pairs)['scripted_failure']==84
    bad=copy.deepcopy(pairs);bad[0]['family_id']='test-family'
    with pytest.raises(ValueError,match='provenance'):validate_training_pairs(bad)
    batches=training_batches(169,4,51)
    assert len(batches)==43 and len(batches[-1])==1
    assert sorted(i for batch in batches for i in batch)==list(range(169))
    assert batches==training_batches(169,4,51)

def test_real_lora_one_epoch_43_updates_improves_margin_with_frozen_reference():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(137)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    enc={'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,5,2]};encoded=[copy.deepcopy(enc) for _ in range(169)]
    ref=reference_scores(model,enc);refs=[copy.deepcopy(ref) for _ in encoded];snapshot=copy.deepcopy(refs)
    pairs=pair_fixtures();cache={'baseline_adapter_sha256':'initial','pair_ids':[p['pair_id'] for p in pairs],'scores':refs}
    verify_reference(pairs,encoded,cache,'initial')
    bad=copy.deepcopy(cache);bad['scores'][0]['input_id']='wrong'
    with pytest.raises(ValueError,match='binding'):verify_reference(pairs,encoded,bad,'initial')
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    recorded=[];history=optimize_epoch(model,encoded,refs,opt,4,51,.1,1,recorded.append)
    assert len(history)==43 and recorded[-1]['pairs_seen']==169 and all(int(v['step'])==43 for v in opt.state.values())
    assert refs==snapshot and all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    from apt_v2.dpo_training import pair_objective
    assert pair_objective(model,enc,ref,.1)[1]['relative_margin']>0

def test_development_recovery_is_separate_from_train_pair_builder():
    task=[t for t in make_tasks() if t.split=='development'][0];tok=CharTokenizer()
    context=build_context(task,'repeat_evidence_error',allowed_split='development')
    text=context['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
    sample={'mode':'greedy','prompt_token_ids':tok.apply_chat_template(context['messages'],add_generation_prompt=True),'response_token_ids':ids,
        'raw_token_log_probs':[-.1]*len(ids),'text':text,'finish_reason':'stop'}
    result=recovery_result(task,context,sample,tok,'greedy');assert result['productive']
    with pytest.raises(ValueError,match='split'):make_pair(task,context,tok,100000)
    bad=copy.deepcopy(sample);bad['prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='prompt'):recovery_result(task,context,bad,tok,'greedy')

def test_complete_task_and_recovery_action_audit(tmp_path):
    import audit_v2_dpo_recovery as worker
    from common import file_hash
    from test_dpo_independent_v2 import fixtures
    from apt_v2.sft_diagnostics import summarize_episodes
    cfg={'eval':{'splits':{'train':12,'development':4}},'sample_eval':{'seed':52,'task_count':4,'rollouts_per_task':4,'temperature':.8},
        'recovery_eval':{'seed':53,'split':'development','task_count':4,'samples_per_context':3,'temperature':.8}}
    tok=CharTokenizer();tasks=make_tasks();dev=[t for t in tasks if t.split=='development'];tm={t.id:t for t in tasks}
    def jsonl(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    comparison={};sample_comparison={}
    for branch in ('before','after'):
        comparison[branch]={}
        for split in ('train','development'):
            episodes=fixtures([t for t in tasks if t.split==split],branch);path=tmp_path/(branch+'-'+split+'-episodes.jsonl');jsonl(path,episodes)
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)};comparison[branch][split]=summary
            (tmp_path/(branch+'-'+split+'-summary.json')).write_text(json.dumps(summary))
        episodes=fixtures(dev,branch,cfg['sample_eval']);path=tmp_path/(branch+'-sample-episodes.jsonl');jsonl(path,episodes)
        summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),'successes_by_task':{t.id:4 for t in dev}}
        sample_comparison[branch]=summary;(tmp_path/(branch+'-sample-summary.json')).write_text(json.dumps(summary))
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparison));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(sample_comparison))
    contexts=[build_context(t,s,allowed_split='development') for t in dev for s in SCENARIOS];jsonl(tmp_path/'recovery_contexts.jsonl',contexts)
    recovery={}
    for branch in ('before','after'):
        rows=[]
        for c in contexts:
            for mode in ('greedy','sample'):
                for ri in range(1 if mode=='greedy' else 3):
                    text=c['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
                    generation={'mode':'greedy' if mode=='greedy' else 'dpo_sample','prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),
                        'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),'text':text,'finish_reason':'stop'}
                    seed=53+[t.id for t in dev].index(c['task_id'])*10000+list(SCENARIOS).index(c['scenario'])*100+ri
                    rows.append({'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':'development','context_id':c['context_id'],
                        'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,'temperature':0 if mode=='greedy' else .8,
                        'generation':generation,'result':recovery_result(tm[c['task_id']],c,generation,tok,generation['mode'])})
        path=tmp_path/(branch+'-recovery-actions.jsonl');jsonl(path,rows)
        recovery[branch]={mode:{'overall':summarize_recovery([r for r in rows if r['mode']==mode]),'by_scenario':{s:summarize_recovery([r for r in rows if r['mode']==mode and r['scenario']==s]) for s in SCENARIOS}} for mode in ('greedy','sample')}
        recovery[branch]['actions_sha256']=file_hash(path)
    (tmp_path/'recovery_comparison.json').write_text(json.dumps(recovery))
    evidence=worker.audit_evaluations(tmp_path,cfg,tok)
    assert evidence['episodes_replayed']==64 and evidence['recovery_actions_replayed']==224 and evidence['development_recovery_contexts']==28
    rows[0]['sample_seed']+=1;jsonl(tmp_path/'after-recovery-actions.jsonl',rows)
    with pytest.raises(ValueError,match='identity'):worker.audit_evaluations(tmp_path,cfg,tok)

def test_runner_failure_keeps_final_adapter_and_excludes_optimizer(tmp_path,monkeypatch):
    import run_v2_dpo_recovery_train as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    model=tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json';model.parent.mkdir(parents=True);model.write_text('{}')
    for name in ('model_identity.json','final/adapter/adapter_model.safetensors'):
        p=tmp_path/'artifacts/initial'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    data=tmp_path/'artifacts/data/pairs.jsonl';data.parent.mkdir(parents=True);data.write_text('fixture')
    (tmp_path/'configs/dpo-recovery-train-v1.json').write_text(json.dumps({'run_id':'recovery-train','initial_run':'initial','data_run':'data','max_wall_seconds':60}))
    (tmp_path/'dpo_recovery_train_release_manifest.json').write_text('{"files":{}}')
    script='from pathlib import Path\np=Path("artifacts/recovery-train/final/adapter");p.mkdir(parents=True)\n(p/"adapter_model.safetensors").write_text("final")\n(p.parent/"state.pt").write_text("optimizer")\nraise RuntimeError("intentional")'
    (tmp_path/'scripts/train_v2_dpo_recovery.py').write_text(script)
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    import zipfile
    with zipfile.ZipFile(tmp_path/'recovery-train-review.zip') as z:
        assert 'recovery-train/final/adapter/adapter_model.safetensors' in z.namelist()
        assert all(not n.endswith('.pt') for n in z.namelist())
