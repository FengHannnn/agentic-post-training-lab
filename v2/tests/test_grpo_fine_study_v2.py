import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,validate_config
from apt_v2.grpo_credit import compute_credit,action_advantage,FINE,LOCAL
from apt_v2.grpo_fine_study import development_tasks,probe_plan,choose_candidate
from apt_v2.grpo_curriculum import training_tasks,frozen_inputs,assert_isolation
from grpo_fine_common import matched_configs,training_decision,cycle_seconds
from test_grpo_v2 import group,scored_group,tiny_model,episode


def test_recorded_failed_actions_lose_spurious_positive_process_credit():
    es=read(ROOT/'tests/fixtures/grpo_fine_group19.json')['episodes'];old=compute_credit(es,method=FINE);new=compute_credit(es,method=LOCAL)
    assert new['outcome_zero_variance'] and new['eligible']
    for t in (2,3,4):
        assert old['turn_advantages'][1][t]>1.7 and new['turn_advantages'][1][t]==0
    assert sum(map(sum,new['local_progress_rewards']))>0
    assert all(a==p for aa,pp in zip(new['turn_advantages'],new['local_progress_rewards']) for a,p in zip(aa,pp))


def test_local_credit_exact_formula_and_terminal_reset_not_a_penalty():
    es=group();c=compute_credit(es,method=LOCAL)
    for i,e in enumerate(es):
        seen=set()
        for t,r in enumerate(e['turns']):
            assert c['local_progress_rewards'][i][t]==.2*len(r['new_verified_metrics'])
            assert action_advantage(c,i,t)==c['advantages'][i]+c['local_progress_rewards'][i][t]
            assert not seen.intersection(r['new_verified_metrics']);seen.update(r['new_verified_metrics'])
        assert c['local_progress_rewards'][i][-1]==0
    assert c['eligibility_basis']=='nonzero_combined_action_advantage'


@pytest.mark.parametrize('bad',['duplicate','unknown','before','after','process','prefix_recount'])
def test_local_credit_rejects_inconsistent_verified_progress(bad):
    es=group();row=next(r for r in es[0]['turns'] if r['new_verified_metrics'])
    if bad=='duplicate':row['new_verified_metrics']*=2
    elif bad=='unknown':row['new_verified_metrics']=['made_up']
    elif bad=='before':row['potential_before']+=.2
    elif bad=='after':row['potential_after']+=.2
    elif bad=='process':row['process_reward']+=.2
    else:
        nxt=es[0]['turns'][es[0]['turns'].index(row)+1];nxt['new_verified_metrics']=[row['new_verified_metrics'][0]]
    with pytest.raises(ValueError):compute_credit(es,method=LOCAL)


def test_zero_progress_equal_failed_outcome_skips_and_never_trains_prefix():
    es=[episode(False,seed=100+i*1000,index=i,scenario='repeat_sql_error') for i in range(4)]
    c=compute_credit(es,method=LOCAL);assert not c['eligible']
    assert all(len(a)==len(e['turns']) for a,e in zip(c['turn_advantages'],es))


def test_local_advantage_reaches_real_lora_gradient_and_objective_audit(tmp_path):
    import torch
    from apt_v2.grpo_optimizer import response_scores,prepare_group_scores,action_objective,update_group
    from grpo_c3_common import verify_objective_metrics
    model=tiny_model();manual=copy.deepcopy(model);ref=copy.deepcopy(model).requires_grad_(False)
    es=scored_group(model);cache=prepare_group_scores(model,lambda p,r:response_scores(ref,p,r),es,'p','r');credit=compute_credit(es,method=LOCAL);total=0.
    for entry in cache['entries']:
        i=entry['episode'];new=response_scores(manual,entry['prompt_token_ids'],entry['response_token_ids'],True)
        old,reference,q=[torch.tensor(entry[k]) for k in ('old','reference','q')]
        loss,_=action_objective(new,old,reference,q,action_advantage(credit,i,entry['turn']-es[i]['prefix_turns']))
        weight=1/(4*len(es[i]['turns']));(loss*weight).backward();total+=loss.item()*weight
    torch.nn.utils.clip_grad_norm_([p for p in manual.parameters() if p.requires_grad],1.)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
    result=update_group(model,optimizer,es,cache,'p','r',credit_method=LOCAL)
    assert result['updated'] and result['loss']==pytest.approx(total,abs=1e-7)
    for a,b in zip(model.parameters(),manual.parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=1e-6)
    atomic_json(tmp_path/'config.json',{'beta':.02,'clip':.2,'tis_cap':2.,'k':4})
    atomic_json(tmp_path/'groups/group-0000.json',{'episodes':es,'credit':credit})
    atomic_json(tmp_path/'checkpoints/group-0000/scores.json',cache)
    atomic_json(tmp_path/'transitions/group-0000.json',{'updated':True,'after':{'checkpoint':'checkpoints/group-0000'},'metrics':result})
    assert verify_objective_metrics(tmp_path)['passed']


def test_frozen_configs_fresh_tasks_and_six_hour_bounds():
    cfg,branches=matched_configs(ROOT/'configs/grpo-fine-credit-v1.json');assert cfg['max_wall_seconds']==21600
    assert [c['credit_method'] for c in branches.values()]==[FINE,LOCAL]
    for c in branches.values():
        report=assert_isolation(c);assert report['train_tasks']==24 and report['development_tasks']==4
        assert len(training_tasks(c))==24 and all(t.seed>=73000 for t in development_tasks())
        inputs=frozen_inputs(c);assert len(inputs['curriculum_expert_proof.json'])==308
        assert all(r['passed'] and r['total_turns']<=12 for r in inputs['curriculum_expert_proof.json'])
        assert len(probe_plan(c))==16
        changed=copy.deepcopy(c);changed['learning_rate']=1e-5
        with pytest.raises(ValueError):validate_config(changed)


def test_budget_reserves_final_work_and_accounts_for_measured_slowdown():
    assert training_decision(0,2100,[])['continue']
    assert not training_decision(20000,2100,[])['continue']
    rows=[{'stage':s+'-legacy-0000','exit_code':0,'elapsed_seconds':v} for s,v in [('collect',300),('update',100)]]
    assert cycle_seconds(rows)>cycle_seconds([])


def test_budget_stop_audit_checks_the_reservation_arithmetic():
    from grpo_fine_common import verify_training_stop
    cfg={'budget_policy':'paired_cycle_with_final_reservation_v1'}
    record={'completed_groups':30,'maximum_groups':48,'reason':'budget_reserve','policy':cfg['budget_policy'],
            'budget_decision':training_decision(16000,2100,[])}
    assert verify_training_stop(record,cfg,2100)==30
    bad=copy.deepcopy(record);bad['budget_decision']['reserved_final_seconds']-=1
    with pytest.raises(ValueError):verify_training_stop(bad,cfg,2100)
    bad=copy.deepcopy(record);bad['budget_decision']['elapsed_seconds']=100
    with pytest.raises(ValueError):verify_training_stop(bad,cfg,2100)


def test_new_method_eligibility_does_not_imply_centered_group_variance():
    es=[episode(True,seed=100+i*1000,index=i) for i in range(4)]
    c=compute_credit(es,method=LOCAL)
    assert c['outcome_zero_variance'] and c['eligible']
    assert any(a>0 for row in c['turn_advantages'] for a in row)


@pytest.mark.parametrize('index',[23,27,39])
def test_local_credit_collection_runs_all_new_recovery_contexts(tmp_path,monkeypatch,index):
    import test_grpo_curriculum_v2 as fixture
    cfg=read(ROOT/'configs/grpo-fine-credit-local-v1.json')
    monkeypatch.setattr(fixture,'config',lambda *a:cfg)
    fixture.test_collection_worker_uses_fresh_tasks_and_native_custom_context(tmp_path,monkeypatch,index)
    saved=read(tmp_path/'groups'/f'group-{index:04}.json')
    assert saved['credit']['method']==LOCAL and saved['credit']['eligible']


def test_fine_selection_preserves_full_success_and_greedy_guards():
    from test_grpo_curriculum_v2 import counts
    result=choose_candidate({'weighted_sft':counts(full=16),'legacy':counts(full=15,recovery=70),
                             'local':counts(full=16,recovery=66)})
    assert result['primary_metric']=='recovery' and result['selected_candidate']=='local'
    assert not result['checks']['legacy']['sample_nonregression']
    result=choose_candidate({'weighted_sft':counts(),'legacy':counts(full=15),
                             'local':counts(full=16,grecovery=19)})
    assert result['selected_candidate']=='legacy' and not result['targeted_or_probe_used_for_selection']
