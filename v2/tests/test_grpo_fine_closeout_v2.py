"""Tests for the closeout advantage and the matched run's decision boundaries."""
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

from apt_v2 import grpo_fine_closeout as study
from apt_v2.grpo_credit import CLOSEOUT, action_advantage, closeout_credit
from apt_v2.grpo_curriculum import build_context, start_spec
from grpo_common import validate_config
from run_v2_grpo_fine_closeout import branch_config


def episode(actions, prefix_passed=False):
    rows = []
    for i,(tool,passed) in enumerate(actions):
        rows.append({'action':{'tool':tool},'observation':{'passed':passed,'submitted':tool=='submit' and passed},
                     'success':tool=='submit' and passed,'done':i==len(actions)-1,
                     'new_verified_metrics':[],'potential_before':0.,'potential_after':0.,'process_reward':0.})
    prefix = [{'row':rows[0] | {'action':{'tool':'check_result'},'observation':{'passed':True}}}] if prefix_passed else []
    return {'start':{'prefix':prefix},'turns':rows,'success':rows[-1]['success']}


def test_closeout_budget_and_direct_submission():
    episodes = [episode([('check_result',True),('check_result',True),('submit',True)]),
                episode([('submit',True)]),
                episode([('check_result',True),('submit',True)],prefix_passed=True),
                episode([('check_result',False),('check_result',True)])]
    credit = closeout_credit(episodes,{'advantages':[0.]*4,'zero_variance':True})
    assert credit['method']==CLOSEOUT
    assert credit['closeout_rewards']==[[.1,0.,.1],[.2],[0.,.2],[0.,0.]]
    assert action_advantage(credit,0,2)==.1


def test_only_early_valid_check_rewards_failed_rollout():
    e=episode([('check_result',True),('check_result',False),('submit',False)])
    c=closeout_credit([e],{'advantages':[-.3],'zero_variance':False})
    assert c['closeout_rewards']==[[.1,0.,0.]]
    assert action_advantage(c,0,0)==pytest.approx(-.2)


def test_frozen_schedule_and_two_panels():
    a,b=[branch_config(n,study.STATES) for n in study.BRANCHES]
    validate_config(a);validate_config(b)
    assert {k for k in a if a[k]!=b[k]}=={'run_id','credit_method'}
    plans=[[start_spec(c,i) for i in range(48)] for c in (a,b)]
    assert plans[0]==plans[1]
    assert sum(p['scenario'] is None for p in plans[0])==24
    assert Counter(p['scenario'] for p in plans[0] if p['scenario'] in study.STATES)=={s:4 for s in study.STATES}
    confirm=branch_config('confirm-closeout',[])
    validate_config(confirm)
    assert {t.family_id for t in study.panel_tasks(a['evaluation_panel'])}.isdisjoint(
        {t.family_id for t in study.panel_tasks(confirm['evaluation_panel'])})
    no_signal=branch_config('local',[])
    assert all(start_spec(no_signal,i)['scenario'] not in study.STATES for i in range(48))


def test_all_closeout_prefixes_have_an_executable_expert():
    from apt_v2.dpo import corrective_action
    from apt_v2.grpo import initial_state
    from apt_v2.grpo_repair_study import training_tasks
    for spec in study.diagnostic_plan(study.master_config()):
        task=next(t for t in training_tasks() if t.id==spec['task_id'])
        env, *_=initial_state(task,spec['scenario'],context_builder=build_context)
        try:
            while not env.done:
                env.step(corrective_action(env))
            assert env.success and len(env.records)<=12
        finally:
            env.close()


def test_preflight_retains_only_complete_mixed_states():
    groups=[]
    for spec in study.diagnostic_plan(study.master_config()):
        if spec['index']==11:
            continue
        state=spec['scenario']
        success=state==study.STATES[0]
        eps=[episode([('submit', success and i==0)]) for i in range(4)]
        groups.append({'spec':spec,'episodes':eps})
    result=study.preflight_result(groups)
    assert result['retained']==[study.STATES[0]]
    assert result['status']=='partial_timeout'
    assert result['states'][study.STATES[2]]['committed_groups']==3


def scored(counts,sha,quality=True):
    summary={m:{k:{'successes':counts[i*2+j]} for j,k in enumerate(('complete','recovery'))}
             for i,m in enumerate(('sample','greedy'))}
    return {'summary':summary,'evaluation_quality_passed':quality,'effective_bf16_change':True,
            'spec':{'policy':{'updates':0 if sha==study.W_SHA else 4,'adapter_sha256':sha}}}


def test_credit_gain_and_w_acceptance_are_independent():
    w=scored([14,60,4,22],study.W_SHA)
    local=scored([14,60,4,22],'a')
    new=scored([14,62,4,22],'b')
    d=study.selection(w,{'local':local,'closeout':new})
    assert d['credit_gain_passed'] and d['selected_candidate'] is None
    new=scored([15,62,4,22],'b')
    d=study.selection(w,{'local':local,'closeout':new})
    assert d['credit_gain_passed'] and d['selected_candidate']=='closeout'


def test_budget_stops_only_at_a_matched_block(tmp_path,monkeypatch):
    import run_v2_grpo_fine_closeout as runner
    from grpo_common import atomic_json
    job=runner.FineStudy(study.master_config(),tmp_path)
    children={name:tmp_path/name for name in study.BRANCHES}
    stopped={name:0 for name in study.BRANCHES}
    monkeypatch.setattr(runner,'states',lambda child:(stopped[child.name],{'updates':stopped[child.name]}))
    def transition(child,index,state):
        assert stopped[child.name]==index
        stopped[child.name]+=1
        return {'after':{'updates':state['updates']+1}}
    monkeypatch.setattr(runner,'load_transition',transition)
    monkeypatch.setattr(job,'stage',lambda *args:None)
    monkeypatch.setattr(job,'elapsed',lambda:22000. if sum(stopped.values())>=8 else 0.)
    result=job.train_pair(children,{'elapsed_seconds':2000.})
    assert {n:v[0] for n,v in result.items()}=={'local':4,'closeout':4}
    assert runner.read(tmp_path/'training_stop.json')=={'groups_per_arm':4,'reason':'budget_reserve'}
