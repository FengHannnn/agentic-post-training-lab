import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.matched_dpo import optimize_weighted_dpo,verify_matched_updates,context_environment,run_recovery,replay_recovery
from apt_v2.tasks import make_tasks,metric_sql
from apt_v2.dpo_recovery import SCENARIOS,build_context
from apt_v2.dpo import corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native

def tiny_model():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(27)
    return get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()

def test_weighted_real_lora_matches_manual_global_weight_batch_mean_and_keeps_reference():
    import torch
    from apt_v2.dpo_training import reference_scores,pair_objective
    from apt_v2.dpo_recovery_training import training_batches
    model=tiny_model();manual=copy.deepcopy(model)
    encoded=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8)]
    refs=[reference_scores(model,e) for e in encoded];snapshot=copy.deepcopy(refs);weights=[2.,.5,.5]
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    mop=torch.optim.AdamW([p for p in manual.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    expected=[]
    for epoch in range(2):
        manual.train()
        for batch in training_batches(3,2,51+epoch):
            mop.zero_grad(set_to_none=True)
            loss=sum(pair_objective(manual,encoded[i],refs[i],.1,True)[0]*weights[i] for i in batch)/len(batch)
            expected.append(loss.item());loss.backward();torch.nn.utils.clip_grad_norm_([p for p in manual.parameters() if p.requires_grad],1);mop.step()
    history=optimize_weighted_dpo(model,encoded,refs,weights,opt,2,51,.1,1,2)
    assert len(history)==4 and history[-1]['pairs_seen']==6 and [r['epoch_pairs_seen'] for r in history]==[2,3,2,3]
    assert [r['loss'] for r in history]==pytest.approx(expected,abs=1e-7)
    assert all(torch.allclose(p,dict(manual.named_parameters())[n],atol=1e-7,rtol=1e-6) for n,p in model.named_parameters())
    assert refs==snapshot and all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if n in frozen)
    assert all(int(s['step'])==4 for s in opt.state.values())
    cfg={'max_steps':4,'pair_count':3,'epochs':2,'gradient_accumulation':2,'seed':51,'source_sft_updates':255,'source_dpo_updates':11}
    ancestry={'algorithm':'hard_recovery_weighted_dpo','source_sft_updates':255,'source_dpo_updates':11,'total_sft_updates':255,'total_dpo_updates':15,'epochs_completed':2}
    final=dict(ancestry,steps=4,cursor=6);train=dict(ancestry,optimizer_start='fresh_adam',optimizer_updates=4,pair_exposures=6,frozen_parameter_gradients=0,history=history)
    verify_matched_updates(cfg,final,train,{'steps':4,'cursor':6,'history':history,'optimizer':opt.state_dict()})
    bad=copy.deepcopy(train);bad['source_sft_updates']=126
    with pytest.raises(ValueError,match='ancestry'):verify_matched_updates(cfg,final,bad)
    for badweights in ([1.], [1.,float('nan'),1.], [1.,0.,1.]):
        with pytest.raises(ValueError):optimize_weighted_dpo(model,encoded,refs,badweights,opt,2,51,.1,1,2)

def oracle(task,context,tok,mode):
    def policy(messages):
        env=context_environment(task,context)
        try:
            for m in messages[len(context['messages']):]:
                if m['role']=='assistant':env.step(json.loads(m['content']))
            return native(tok,messages,json.dumps(corrective_action(env)),mode)
        finally:env.close()
    return policy

def test_all_development_recoveries_complete_and_token_tamper_is_rejected():
    tok=CharTokenizer();count=0
    for task in [t for t in make_tasks() if t.split=='development']:
        for s in SCENARIOS:
            c=build_context(task,s,allowed_split='development');e=run_recovery(task,c,oracle(task,c,tok,'greedy'))
            assert e['success'] and e['turns'][-1]['action']['tool']=='submit'
            assert len(c['records'])+len(e['turns'])<=task.max_turns
            assert replay_recovery(task,c,e,tok,'greedy')==len(e['turns']);count+=1
    assert count==28
    bad=copy.deepcopy(e);bad['turns'][0]['generation']['prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='prompt'):replay_recovery(task,c,bad,tok,'greedy')
    test=next(t for t in make_tasks() if t.split=='test')
    with pytest.raises(ValueError,match='development'):context_environment(test,build_context(test,SCENARIOS[0],allowed_split='test'))

def test_no_progress_recovery_exhausts_original_budget():
    tok=CharTokenizer();task=next(t for t in make_tasks() if t.split=='development');c=build_context(task,'repeat_sql_error',allowed_split='development')
    e=run_recovery(task,c,lambda messages:native(tok,messages,c['scripted_rejected']))
    assert not e['success'] and len(e['turns'])+len(c['records'])==task.max_turns
    replay_recovery(task,c,e,tok,'greedy')

def test_calendar_development_requires_explicit_scope():
    from apt_v2.independent_recovery_test import calendar_sql_check
    task=next(t for t in make_tasks() if t.split=='development');sql=metric_sql(task,'current_net_cents')
    with pytest.raises(ValueError):calendar_sql_check(task,sql,['current_net_cents'])
    assert calendar_sql_check(task,sql,['current_net_cents'],allowed_split='development')['passed']

def test_paired_summary_rejects_duplicate_keys():
    from matched_dpo_common import result_summary
    cfg={'evaluation_branches':['parent','weighted_sft','dpo']}
    rows=[{'model_branch':b,'mode':m,'task_id':'development-000','rollout_index':0,'success':True,'turns':[],'tool_errors':0,'fault_mode':'none','recovered':False,'process_reward_sum':0,'total_reward':0} for b in cfg['evaluation_branches'] for m in ('greedy','sample')]
    # Empty recovery sets remain legitimate for summary-only fixture.
    result=result_summary(rows,[],[],cfg)
    assert result['paired']['weighted_sft_to_dpo']['sample']['complete']['both_success']==1
    with pytest.raises(ValueError,match='distinct'):result_summary(rows+[rows[0]],[],[],cfg)

def test_runner_prepare_failure_packages_logs_and_protects_run_id(tmp_path,monkeypatch,capsys):
    import run_v2_matched_dpo as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/hard-recovery-dpo-matched-v1.json').write_text(json.dumps({'run_id':'matched','max_wall_seconds':14400}))
    (tmp_path/'matched_dpo_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/matched_dpo_common.py').write_text("print('preflight live',flush=True)\nraise RuntimeError('missing parent')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'preflight live' in capsys.readouterr().out
    import zipfile
    with zipfile.ZipFile(tmp_path/'matched-review.zip') as z:assert 'matched/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()


def test_three_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from matched_dpo_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_matched_dpo import audit_results
    cfg={'evaluation_branches':['parent','weighted_sft','dpo'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
        'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33,'samples_per_context':1,'temperature':.8}}
    tok=CharTokenizer();ts=tasks();tm={t.id:t for t in ts};cs=contexts();allfull=[];allrs=[];allaa=[]
    for b in cfg['evaluation_branches']:
        full=[];rs=[];aa=[]
        for task in ts:
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0)
                def policy(messages):
                    env=Environment(task)
                    try:
                        for m in messages[2:]:
                            if m['role']=='assistant':env.step(json.loads(m['content']))
                        return native(tok,messages,json.dumps(corrective_action(env)),nm)
                    finally:env.close()
                e=run_episode(task,policy);e.update(model_branch=b,mode=mode,rollout_index=0,sample_seed=seed,temperature=0 if mode=='greedy' else .8)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                full.append(e)
        for c in cs:
            task=tm[c['task_id']]
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0,c)
                e=run_recovery(task,c,oracle(task,c,tok,nm));temperature=0 if mode=='greedy' else .8
                e.update(model_branch=b,mode=mode,sample_index=0,sample_seed=seed,temperature=temperature)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                rs.append(e);g=e['turns'][0]['generation'];actual,d=diagnose_action(task,c,g['text'])
                aa.append({'model_branch':b,'mode':mode,'task_id':task.id,'family_id':task.family_id,'split':'development','context_id':c['context_id'],
                    'scenario':c['scenario'],'sample_index':0,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d})
        jsonl(tmp_path/(b+'-complete-episodes.jsonl'),full);jsonl(tmp_path/(b+'-recovery-episodes.jsonl'),rs);jsonl(tmp_path/(b+'-first-actions.jsonl'),aa)
        allfull.extend(full);allrs.extend(rs);allaa.extend(aa)
    summary=result_summary(allfull,allrs,allaa,cfg);(tmp_path/'comparison_summary.json').write_text(json.dumps(summary))
    result,calendar,evidence=audit_results(tmp_path,cfg,tok)
    assert evidence['complete_episodes_replayed']==24 and evidence['recovery_episodes_replayed']==168
    assert result['paired']['weighted_sft_to_dpo']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'dpo-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)
