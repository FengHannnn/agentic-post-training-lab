import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.dpo_chosen_ce import optimize_chosen_ce,combine_loss,verify_chosen_ce_updates

def test_zero_ce_real_lora_exactly_matches_original_weighted_dpo():
    import torch
    from test_matched_dpo_v2 import tiny_model
    from apt_v2.matched_dpo import optimize_weighted_dpo
    from apt_v2.dpo_training import reference_scores
    a=tiny_model();b=copy.deepcopy(a);enc=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8)]
    ref=[reference_scores(a,e) for e in enc];snapshot=copy.deepcopy(ref);weights=[2.,.5,.5]
    oa=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.0001,weight_decay=0);ob=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    before=optimize_weighted_dpo(a,enc,ref,weights,oa,2,51,.1,1,2);after=optimize_chosen_ce(b,enc,ref,weights,ob,2,51,.1,1,2,0)
    assert before==[{k:r[k] for k in before[0]} for r in after] and ref==snapshot
    assert all(torch.equal(p,dict(b.named_parameters())[n]) for n,p in a.named_parameters())
    for x,y in zip(oa.state.values(),ob.state.values()):assert all(torch.equal(x[k],y[k]) for k in x)

def test_ce_gradient_is_uniform_token_mean_and_leaves_rejected_gradient_unchanged():
    import torch
    from apt_v2.dpo import dpo_loss
    chosen=torch.tensor([-.2,-.4],requires_grad=True);rejected=torch.tensor([-.5,-.8,-.3],requires_grad=True)
    dpo=dpo_loss(chosen.sum(),rejected.sum(),torch.tensor(-.6),torch.tensor(-1.6),.1);ce=-chosen.mean()
    a=torch.autograd.grad(combine_loss(dpo,ce,3.7,0,4),(chosen,rejected),retain_graph=True)
    b=torch.autograd.grad(combine_loss(dpo,ce,3.7,1,4),(chosen,rejected))
    assert torch.allclose(b[0]-a[0],torch.full_like(chosen,-1/8)) and torch.equal(a[1],b[1])

@pytest.mark.parametrize('branch,coefficient',[('weighted_control',0),('dpo_ce',1)])
def test_joint_update_counters_and_objective_decomposition_are_enforced(branch,coefficient):
    from apt_v2.dpo_recovery_training import training_batches
    cfg=json.loads((ROOT/'configs/dpo-chosen-ce-control-v1.json').read_text());history=[];cursor=0
    for epoch in range(6):
        epoch_cursor=0
        for step,batch in enumerate(training_batches(169,4,cfg['seed']+epoch),1):
            cursor+=len(batch);epoch_cursor+=len(batch);history.append({'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':step,'epoch_pairs_seen':epoch_cursor,'loss':.7+coefficient*.2,'weighted_dpo_loss':.7,'chosen_ce':.2,'chosen_ce_coefficient':coefficient,'grad_norm':1.})
    ancestry={'branch':branch,'algorithm':'hard_recovery_weighted_dpo' if coefficient==0 else 'weighted_dpo_chosen_ce','chosen_ce_coefficient':coefficient,'source_sft_updates':255,'source_dpo_updates':11,'total_sft_updates':255,'total_dpo_updates':269,'epochs_completed':6,'auxiliary_chosen_ce_updates':258 if coefficient else 0}
    final=dict(ancestry,steps=258,cursor=1014);train=dict(ancestry,optimizer_start='fresh_adam',optimizer_updates=258,pair_exposures=1014,chosen_ce_pair_exposures=1014 if coefficient else 0,frozen_parameter_gradients=0,history=history)
    verify_chosen_ce_updates(cfg,branch,final,train)
    bad=copy.deepcopy(train);bad['history'][0]['loss']+=.01
    with pytest.raises(ValueError,match='objective'):verify_chosen_ce_updates(cfg,branch,final,bad)
    bad=copy.deepcopy(final);bad['total_sft_updates']+=258
    with pytest.raises(ValueError,match='ancestry'):verify_chosen_ce_updates(cfg,branch,bad,train)

def test_runtime_guard_rejects_gpu_and_environment_changes():
    from dpo_chosen_ce_common import check_runtime
    cfg={'required_gpu_name':'NVIDIA L40S'};expected={'gpu_name':'NVIDIA L40S','torch':'same'};check_runtime(cfg,expected,expected)
    with pytest.raises(ValueError,match='hardware'):check_runtime(cfg,dict(expected,gpu_name='NVIDIA L20'),expected)
    with pytest.raises(ValueError,match='identity'):check_runtime(cfg,dict(expected,torch='other'),expected)
    expected=dict(expected,allow_tf32_matmul=False);changed=dict(expected,allow_tf32_matmul=True)
    with pytest.raises(ValueError,match='identity'):check_runtime(cfg,changed,expected)
    check_runtime(cfg,changed,expected,precision=False)

def test_paired_runner_failure_archives_and_protects_both_update_counts(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_chosen_ce as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/dpo-chosen-ce-control-v1.json').write_text(json.dumps({'run_id':'ce-paired','max_wall_seconds':14400}))
    (tmp_path/'dpo_chosen_ce_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/dpo_chosen_ce_common.py').write_text("print('CE preflight',flush=True)\nraise RuntimeError('hardware differs')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'CE preflight' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/ce-paired/run_status.json').read_text());assert status['optimizer_updates']==0 and status['branch_optimizer_updates']=={'weighted_control':0,'dpo_ce':0}
    import zipfile
    with zipfile.ZipFile(tmp_path/'ce-paired-review.zip') as z:assert 'ce-paired/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()

from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native
def test_four_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_chosen_ce_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_chosen_ce import audit_results
    cfg={'evaluation_branches':['weighted_sft','uniform_dpo','weighted_control','dpo_ce'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
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
    assert result['paired']['weighted_control_to_dpo_ce']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'dpo_ce-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)

from apt_v2.dpo_investigation import build_probes,digest,metrics_from_scores,summarize_probe_metrics
def test_five_model_scores_recompute_and_reject_changed_metric(tmp_path,monkeypatch):
    import audit_v2_dpo_chosen_ce as auditor
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
    cfg={'max_length':1000000,'beta':.1,'checkpoints':{b:{'adapter_sha256':b} for b in ['parent','weighted_sft','uniform_dpo','weighted_control','dpo_ce']}}
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
    saved['pair_metrics'][0]['loss']+=.1;write('dpo_ce-scores.json',saved)
    with pytest.raises(ValueError,match='metrics'):auditor.audit_scores(tmp_path,cfg,tok)



def test_nonzero_ce_runs_real_lora_adam_updates_and_preserves_reference():
    import torch
    from test_matched_dpo_v2 import tiny_model
    from apt_v2.dpo_training import reference_scores
    model=tiny_model();enc=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8)]
    ref=[reference_scores(model,e) for e in enc];snapshot=copy.deepcopy(ref);frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad};initial={n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    history=optimize_chosen_ce(model,enc,ref,[2.,.5,.5],optimizer,2,51,.1,1,2,1)
    assert len(history)==4 and history[-1]['pairs_seen']==6 and ref==snapshot
    assert all(abs(row['loss']-row['weighted_dpo_loss']-row['chosen_ce'])<1e-5 for row in history)
    assert all(int(s['step'])==4 for s in optimizer.state.values())
    assert any(not torch.equal(p,initial[n]) for n,p in model.named_parameters() if n in initial)
    assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if n in frozen)
