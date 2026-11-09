import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.dpo_continuation_ce import build_continuations,ce_schedule,coverage,token_budget,optimize_continuation_ce,verify_continuation_updates

def test_expert_ce_pool_is_complete_train_only_and_fully_exposed(tmp_path):
    from dpo_continuation_ce_common import jsonl,validate_ce_inputs,write,file_hash
    from apt_v2.dpo_investigation import build_probes
    examples,episodes=build_continuations();schedule=ce_schedule(len(examples),1014,2026110504);summary=coverage(examples,schedule)
    assert len(examples)==288 and len(episodes)==84 and all(e['success'] and e['turns'][-1]['row']['action']['tool']=='submit' for e in episodes)
    assert {e['split'] for e in examples}=={'train'} and summary['pool_tasks']==12 and summary['pool_contexts']==84
    assert summary['pool_tool_counts']=={'calculate_change':72,'check_result':84,'submit':84,'query_sql':48}
    assert summary['sampled_actions']==1014 and summary['sampled_unique_actions']==288 and summary['minimum_exposures_per_pool_action']==3 and summary['maximum_exposures_per_pool_action']==4
    assert schedule==ce_schedule(288,1014,2026110504) and schedule!=ce_schedule(288,1014,2026110505)
    assert all(len(set(schedule[k:k+288]))==288 for k in range(0,864,288))
    jsonl(tmp_path/'ce_examples.jsonl',examples);jsonl(tmp_path/'ce_episodes.jsonl',episodes);jsonl(tmp_path/'probes.jsonl',build_probes());write(tmp_path/'ce_summary.json',{'coverage':summary});write(tmp_path/'ce_schedule.json',{'indices':schedule,'seed':2026110504,'pool_sha256':file_hash(tmp_path/'ce_examples.jsonl')})
    cfg={'ce_action_exposures':1014,'ce_seed':2026110504,'ce_pool_actions':288};assert validate_ce_inputs(tmp_path,cfg)==(examples,schedule)
    examples[0]['split']='development';(tmp_path/'ce_examples.jsonl').unlink();jsonl(tmp_path/'ce_examples.jsonl',examples)
    with pytest.raises(ValueError,match='data'):validate_ce_inputs(tmp_path,cfg)

def test_continuation_ce_equals_chosen_ce_when_inputs_are_identical_real_lora():
    import torch
    from test_matched_dpo_v2 import tiny_model
    from apt_v2.dpo_training import reference_scores
    from apt_v2.dpo_chosen_ce import optimize_chosen_ce
    from apt_v2.dpo_recovery_training import training_batches
    model=tiny_model();manual=copy.deepcopy(model)
    enc=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8)]
    ce=[{'input_ids':e['prompt']+e['chosen'],'labels':[-100]*len(e['prompt'])+e['chosen'],'prompt_length':len(e['prompt']),'response_length':len(e['chosen'])} for e in enc]
    schedule=[i for epoch in range(2) for b in training_batches(3,2,51+epoch) for i in b];ref=[reference_scores(model,e) for e in enc];snapshot=copy.deepcopy(ref)
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0);mop=torch.optim.AdamW([p for p in manual.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    history=optimize_continuation_ce(model,enc,ref,[2.,.5,.5],ce,schedule,opt,2,51,.1,1,2,1);expected=optimize_chosen_ce(manual,enc,ref,[2.,.5,.5],mop,2,51,.1,1,2,1)
    assert len(history)==4 and [r['loss'] for r in history]==pytest.approx([r['loss'] for r in expected],abs=1e-6)
    assert all(torch.allclose(p,dict(manual.named_parameters())[n],atol=1e-7,rtol=1e-6) for n,p in model.named_parameters())
    assert ref==snapshot and all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if n in frozen)
    assert all(int(s['step'])==4 for s in opt.state.values()) and [i for r in history for i in r['ce_indices']]==schedule
    assert sum(r['CE_response_tokens'] for r in history)==18 and sum(r['CE_extra_forward_input_tokens'] for r in history)==30

def test_different_ce_context_updates_preserve_dpo_order_and_exact_budgets(monkeypatch):
    import torch
    import apt_v2.dpo_continuation_ce as module
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1.);calls=[];ce_calls=[]
    def dpo(m,e,r,beta,grad):calls.append(e);return (m.weight.square().mean(),m.weight.sum()*0)
    def ce(m,e):ce_calls.append(e['tag']);return m.weight.square().mean()*e['target']
    monkeypatch.setattr(module,'pair_terms',dpo);monkeypatch.setattr(module,'action_loss',ce)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0);encoded=[0,1,2];ce_data=[{'tag':k,'target':k+1,'input_ids':[1]*(k+3),'response_length':k+1} for k in range(4)];schedule=[3,2,1,0,3,2]
    history=optimize_continuation_ce(model,encoded,[{}, {}, {}],[2.,.5,.5],ce_data,schedule,opt,2,51,.1,1,2,1)
    from apt_v2.dpo_recovery_training import training_batches
    assert calls==[i for epoch in range(2) for b in training_batches(3,2,51+epoch) for i in b] and ce_calls==schedule
    assert sum(r['CE_response_tokens'] for r in history)==sum(ce_data[i]['response_length'] for i in schedule)
    assert history[-1]['ce_actions_seen']==6 and model.weight.item()!=1.

def test_continuation_counter_and_sampler_tampering_are_rejected():
    from apt_v2.dpo_recovery_training import training_batches
    cfg=json.loads((ROOT/'configs/dpo-continuation-ce-control-v1.json').read_text());schedule=ce_schedule(288,1014,cfg['ce_seed']);history=[];cursor=0
    for epoch in range(6):
        ec=0
        for step,batch in enumerate(training_batches(169,4,cfg['seed']+epoch),1):
            indices=schedule[cursor:cursor+len(batch)];cursor+=len(batch);ec+=len(batch);history.append({'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':step,'epoch_pairs_seen':ec,'loss':.9,'weighted_dpo_loss':.7,'chosen_ce':.2,'chosen_ce_coefficient':1,'ce_indices':indices,'ce_actions_seen':cursor,'grad_norm':1.})
    base={'algorithm':'weighted_dpo_continuation_ce','branch':'continuation_ce','chosen_ce_coefficient':1,'auxiliary_chosen_ce_updates':258,'source_sft_updates':255,'source_dpo_updates':11,'total_sft_updates':255,'total_dpo_updates':269,'epochs_completed':6}
    final=dict(base,steps=258,cursor=1014);train=dict(base,optimizer_start='fresh_adam',optimizer_updates=258,pair_exposures=1014,chosen_ce_pair_exposures=1014,ce_actions_seen=1014,ce_coverage={'pool_actions':288},frozen_parameter_gradients=0,history=history)
    verify_continuation_updates(cfg,final,train)
    for field,value in [('ce_indices',[0,0,0,0]),('loss',1.1),('ce_actions_seen',5)]:
        bad=copy.deepcopy(train);bad['history'][0][field]=value
        with pytest.raises(ValueError,match='schedule'):verify_continuation_updates(cfg,final,bad)
    bad=copy.deepcopy(final);bad['total_sft_updates']=513
    with pytest.raises(ValueError,match='ancestry'):verify_continuation_updates(cfg,bad,train)

from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native
def test_four_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_continuation_ce_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_continuation_ce import audit_results
    cfg={'evaluation_branches':['weighted_sft','weighted_control','dpo_ce','continuation_ce'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
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
    assert evidence['complete_episodes_replayed']==32 and evidence['recovery_episodes_replayed']==224
    assert result['paired']['dpo_ce_to_continuation_ce']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'continuation_ce-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)

from apt_v2.dpo_investigation import build_probes,digest,metrics_from_scores,summarize_probe_metrics
def test_five_model_scores_recompute_and_reject_changed_metric(tmp_path,monkeypatch):
    import audit_v2_dpo_continuation_ce as auditor
    from apt_v2.reference_numerics import compare_reference_caches
    from apt_v2.dpo_recovery import encode_recovery_pair
    from apt_v2.dpo_training import input_id
    from apt_v2.sft import encode_example
    from apt_v2.dpo_investigation import summarize_pair_metrics
    from dpo_investigation_common import jsonl
    tok=CharTokenizer();messages=[{'role':'user','content':'fixture'}];chosen='{"tool":"read_policy","arguments":{}}';rejected='bad'
    pair={'pair_id':'one','messages':messages,'chosen':chosen,'rejected':rejected,'native_prompt_token_ids':tok.apply_chat_template(messages,add_generation_prompt=True),'native_rejected_token_ids':[ord(c)+1 for c in rejected]+[0],'scenario':'stale_evidence_repair'}
    encoded=[encode_recovery_pair(tok,pair,1000000)];ref={'input_id':input_id(encoded[0]),'chosen':[-.1]*len(encoded[0]['chosen']),'rejected':[-.2]*len(encoded[0]['rejected'])}
    probe=build_probes()[0];probes=[probe];monkeypatch.setattr(auditor,'build_probes',lambda:probes)
    pe=encode_example(tok,probe,1000000);pm=[{'probe_id':probe['probe_id'],'probe_sha256':digest(probe),'token_log_probs':[-.1]*pe['response_length'],'ce':.1}]
    cfg={'max_length':1000000,'beta':.1,'checkpoints':{b:{'adapter_sha256':b} for b in ['parent','weighted_sft','weighted_control','dpo_ce','continuation_ce']}}
    cfg['checkpoints']['parent']['adapter_sha256']='p'
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    jsonl(tmp_path/'pairs.jsonl',[pair]);jsonl(tmp_path/'probes.jsonl',probes);write('weights.json',{'weights':[1.]})
    write('reference_scores.json',{'baseline_adapter_sha256':'p','pair_ids':['one'],'scores':[ref]});cache={'baseline_adapter_sha256':'p','pair_ids':['one'],'scores':[ref]};write('reference_recheck.json',{'fresh_scores':[ref],'maximum_token_logp_difference':0.,'numerics':compare_reference_caches(cache,cache)})
    summaries={}
    for branch,spec in cfg['checkpoints'].items():
        score=copy.deepcopy(ref)
        if branch!='parent':score['chosen']=[-1.]*len(ref['chosen']);score['rejected']=[-20.]*len(ref['rejected'])
        m=[metrics_from_scores(score,ref)];summary={'pairs':summarize_pair_metrics([pair],encoded,m,[ref],[1.]),'probes':summarize_probe_metrics(probes,pm)}
        saved={'branch':branch,'adapter_sha256':spec['adapter_sha256'],'pair_scores':[score],'pair_metrics':m,'probe_metrics':pm,'summary':summary}
        write(branch+'-scores.json',saved);summaries[branch]=summary
    write('scores_summary.json',{'branches':summaries,'training_enabled':False,'optimizer_updates':0,'maximum_parent_reference_difference':0.})
    actual,maximum=auditor.audit_scores(tmp_path,cfg,tok);assert actual==summaries and maximum==0.
    saved['pair_metrics'][0]['loss']+=.1;write('continuation_ce-scores.json',saved)
    with pytest.raises(ValueError,match='metrics'):auditor.audit_scores(tmp_path,cfg,tok)


def test_paired_runner_failure_archives_and_protects_both_update_counts(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_continuation_ce as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/dpo-continuation-ce-control-v1.json').write_text(json.dumps({'run_id':'ce-paired','max_wall_seconds':14400}))
    (tmp_path/'dpo_continuation_ce_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/dpo_continuation_ce_common.py').write_text("print('CE preflight',flush=True)\nraise RuntimeError('hardware differs')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'CE preflight' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/ce-paired/run_status.json').read_text());assert status['optimizer_updates']==0 and status['branch_optimizer_updates']=={'continuation_ce':0}
    import zipfile
    with zipfile.ZipFile(tmp_path/'ce-paired-review.zip') as z:assert 'ce-paired/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()


def test_e4_contract_rejects_dpo_or_ce_budget_changes_and_pins_gpu_uuid(tmp_path,monkeypatch):
    import dpo_continuation_ce_common as module
    from common import write,file_hash
    from dpo_chosen_ce_common import check_runtime
    cfg=json.loads((ROOT/'configs/dpo-continuation-ce-control-v1.json').read_text());old=json.loads((ROOT/'configs/dpo-chosen-ce-control-v1.json').read_text());folder=tmp_path/'artifacts'/cfg['comparison_source_run'];folder.mkdir(parents=True)
    write(folder/'config.json',old);write(folder/'audit.json',{'passed':True});write(folder/'run_status.json',{'error':None});write(folder/'execution_environment.json',cfg['required_execution_environment'])
    cfg.update(comparison_config_sha256=file_hash(folder/'config.json'),comparison_audit_sha256=file_hash(folder/'audit.json'),comparison_run_status_sha256=file_hash(folder/'run_status.json'));monkeypatch.setattr(module,'ROOT',tmp_path);module.check_contract(cfg)
    for key,value in [('hard_weight',2),('chosen_ce_coefficient',.5),('ce_action_exposures',1015)]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(ValueError,match='fixed'):module.check_contract(bad)
    identity=cfg['required_execution_environment'];check_runtime(cfg,identity,identity)
    for key,value in [('gpu_uuid','other-device'),('torch','other'),('allow_tf32_matmul',True)]:
        changed=copy.deepcopy(identity);changed[key]=value
        with pytest.raises(ValueError,match='identity'):check_runtime(cfg,changed,identity)
