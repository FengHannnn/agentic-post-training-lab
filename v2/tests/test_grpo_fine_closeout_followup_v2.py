"""Regression coverage for the 12-task frozen follow-up."""
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

from apt_v2 import grpo_fine_closeout_followup as spec
from apt_v2.grpo_evaluation import evaluation_plan
from apt_v2.grpo_repair_study import assert_isolation, start_spec
from grpo_common import read, validate_config
from run_v2_grpo_fine_closeout_followup import bind_checkpoint, branch_config, result_record, stage_pairs


def test_frozen_three_model_protocol():
    spec.validate_master(read(ROOT/spec.CONFIG))
    spec.validate_evidence(read(ROOT/spec.INPUT/'source-evidence.json'))
    plans=[]
    for name in spec.NAMES:
        cfg=branch_config(name)
        validate_config(cfg)
        assert assert_isolation(cfg)['development_tasks']==12
        plans.append(evaluation_plan(cfg))
        with pytest.raises(ValueError, match='evaluation only'):
            start_spec(cfg,0)
    assert plans[0]==plans[1]==plans[2]
    assert Counter((row['mode'],row['kind']) for row in plans[0])=={
        ('sample','complete'):48,('sample','recovery'):252,('greedy','complete'):12,('greedy','recovery'):84}
    altered={**spec.master_config(),'max_wall_seconds':21601}
    with pytest.raises(ValueError, match='contract'):
        spec.validate_master(altered)


@pytest.mark.parametrize('name',['local','closeout'])
def test_adapter_copy_binds_exact_reviewed_policy(tmp_path,name):
    assert bind_checkpoint(tmp_path,name,prepare=True)==spec.POLICIES[name]
    assert bind_checkpoint(tmp_path,name)==spec.POLICIES[name]
    p=tmp_path/spec.POLICIES[name]['checkpoint']/'adapter/adapter_model.safetensors'
    p.write_bytes(b'altered')
    with pytest.raises(ValueError):
        bind_checkpoint(tmp_path,name)


def evaluation_fixture(name, counts):
    plan=evaluation_plan(branch_config('w'))
    remaining=dict(zip(('sample_complete','sample_recovery','greedy_complete','greedy_recovery'),counts))
    summary={'sample':{},'greedy':{}}
    for mode in summary:
        for kind in ('complete','recovery'):
            summary[mode][kind]={'successes':remaining[mode+'_'+kind],
                                 'episodes':sum(p['mode']==mode and p['kind']==kind for p in plan)}
    episodes=[]
    for i,item in enumerate(plan):
        key=item['mode']+'_'+item['kind']
        success=remaining[key]>0
        remaining[key]-=success
        episodes.append({'task_id':item['task_id'],'scenario':item['scenario'],'evaluation_mode':item['mode'],
                         'rollout_index':item['rollout_index'],'seed':item['seed'],'start_id':str(i),
                         'sampling':{'temperature':item['temperature']},'start':{'prefix':[]},'success':success,
                         'turns':[{'action':{'tool':'check_result'},'observation':{'passed':success},
                                   'new_verified_metrics':['current_net_cents','prior_net_cents','change_cents']}]})
    return {'episodes':episodes,'summary':summary,'worker_proof':[{'loaded_tensor_sha256':name}],
            'evaluation_quality_passed':True,'targeted_summary':{},'spec':{'policy':spec.POLICIES[name]}}


def test_unchanged_rates_and_recovery_threshold_on_larger_panel():
    results={n:evaluation_fixture(n,c) for n,c in
             [('w',(28,171,12,61)),('local',(28,171,12,61)),('closeout',(30,177,12,61))]}
    decision=result_record(results)
    assert decision['primary_metric']=='sample_complete'
    assert decision['credit_threshold']==6 and decision['new_credit_recovery_gain']==6
    assert decision['credit_gain_passed'] and decision['selected_candidate']=='closeout'
    same=decision['paired']['local_to_closeout']['stages']['sample_recovery']
    assert same['both_certified']==252 and same['shared_certified_submit_gained']==6
    assert decision['new_training_updates']==0


def test_primary_switches_when_48_complete_episodes_are_saturated():
    results={n:evaluation_fixture(n,c) for n,c in
             [('w',(48,171,12,61)),('local',(48,171,12,61)),('closeout',(48,176,12,61))]}
    decision=result_record(results)
    assert decision['primary_metric']=='sample_recovery'
    assert decision['selected_candidate']=='closeout'
    assert not decision['credit_gain_passed']
    results['closeout']['episodes'][0]['seed']+=1
    with pytest.raises(ValueError, match='pair differs'):
        stage_pairs(results['w'],results['closeout'],branch_config('w'))
