from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.phase5_signal import (
    select_cases,sample_seed,sample_from_output,validate_records,signal_summary,run_seconds)

ROOT=Path(__file__).resolve().parents[2]


class Tokenizer:
    eos_token_id=pad_token_id=0
    def decode(self,tokens,**kw):
        text='{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}' if tokens[0]==5 else 'bad'
        return text+('<EOS>' if not kw['skip_special_tokens'] and tokens[-1]==0 else '')


def native(tokens=None,finish='stop',prompt=None):
    tokens=[5,0] if tokens is None else tokens
    return SimpleNamespace(prompt_token_ids=[1] if prompt is None else prompt,
        outputs=[SimpleNamespace(token_ids=tokens,finish_reason=finish,
            logprobs=[{t:SimpleNamespace(logprob=-.1)} for t in tokens])])


def make_group():
    case=EvalCase('case-natural-right_grouped','math','1+2','tool_call','1+2',3)
    rows=[]
    for i in range(8):
        s=sample_from_output(native(),[1],case,i,sample_seed(0,i),Tokenizer())
        rw=score_reward(s.text,case,finish_reason=s.finish_reason)
        rows.append({'sample':asdict(s),'case':asdict(case),'score':rw['score'],'semantic_pass':rw['semantic_pass']})
    return rows,[case],[[1]]


def test_all_144_inputs_are_training_arithmetic_with_balanced_wording_and_unique_seeds():
    folder=ROOT/'data/preferences/dpo_v2'
    splits=load_preference_splits(folder,json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    plan=json.loads((ROOT/'configs/phase5_rollout_signal.json').read_text(encoding='utf-8'))
    cases=select_cases(splits['train'],plan['case_ids'])
    assert len(cases)==144 and set(c.id for c in cases).isdisjoint(r.id for r in splits['development'])
    assert Counter(c.id.split('-')[-2] for c in cases)=={'natural':48,'sequence':48,'symbolic':48}
    assert set(Counter(c.category for c in cases).values())=={36}
    assert len({sample_seed(g,i) for g in range(144) for i in range(8)})==1152
    assert sample_seed(0,0)!=42
    with pytest.raises(ValueError):select_cases(splits['train'],plan['case_ids'][:-1])
    with pytest.raises(ValueError):select_cases(splits['train'],list(reversed(plan['case_ids'])))


def test_eight_samples_duplicates_and_eos_are_retained():
    rows,cases,prompts=make_group()
    groups,batch,rewards=validate_records(rows,cases,prompts,Tokenizer())
    assert len(groups[0])==8 and batch['response_mask'].sum()==16 and rewards['reward_sum']==8
    assert signal_summary(rewards)['group_outcomes']=={'all_correct':1,'all_wrong':0,'mixed':0}


def test_length_is_preserved_and_strict_reward_is_zero():
    case=make_group()[1][0]
    sample=sample_from_output(native([6]*64,'length'),[1],case,7,sample_seed(0,7),Tokenizer())
    assert sample.finish_reason=='length' and len(sample.response_token_ids)==64
    assert score_reward(sample.text,case,finish_reason=sample.finish_reason)['reward']==0
    with pytest.raises(ValueError):sample_from_output(native([6]*63,'length'),[1],case,0,42,Tokenizer())


@pytest.mark.parametrize('mutation',['missing','seed','prompt','policy','decode','target','score','index','eos'])
def test_corrupt_phase5_records_are_rejected(mutation):
    rows,cases,prompts=make_group()
    if mutation=='missing':rows.pop()
    elif mutation=='seed':rows[0]['sample']['seed']=42
    elif mutation=='prompt':rows[0]['sample']['prompt_token_ids']=[3]
    elif mutation=='policy':rows[0]['sample']['policy_token_log_probs']=[-.2,-.1]
    elif mutation=='decode':rows[0]['sample']['text']='changed'
    elif mutation=='target':rows[0]['case']['prompt']='changed'
    elif mutation=='score':rows[0]['score']={}
    elif mutation=='index':rows[0]['sample']['sample_index']=7
    else:rows[0]['sample']['response_token_ids']=[0,5]
    with pytest.raises(ValueError):validate_records(rows,cases,prompts,Tokenizer())


def test_prefix4_and_full8_comparison_uses_same_eight_samples():
    rewards={'groups':[
        {'group_id':'a-natural-x','rewards':[0,0,0,0,1,1,1,1]},
        {'group_id':'b-sequence-x','rewards':[0,1,0,1,0,0,0,0]},
        {'group_id':'c-symbolic-x','rewards':[1]*8}]}
    stats=signal_summary(rewards)
    assert stats['prefix4_mixed_groups']==1 and stats['full8_mixed_groups']==2
    assert stats['extra_mixed_groups_from_samples_5_to_8']==1
    assert stats['samples_in_nonzero_advantage_groups']==16


@pytest.mark.parametrize('price',[0,-1,float('nan'),float('inf'),1e9])
def test_invalid_prices_refused(price):
    with pytest.raises(ValueError):run_seconds(price)


def test_no_obsolete_global_spend_limit_and_finite_run_bound():
    assert run_seconds(2.18)==1790
    assert run_seconds(100)==350
