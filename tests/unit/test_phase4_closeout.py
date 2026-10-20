from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.closeout import select_cases, sample_seed, validate_records, benchmark_summary, budget_limits
from agentic_post_training.rollouts.records import RolloutSample

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_closeout_is_training_only_balanced_and_has_224_bounded_requests():
    config=json.loads((ROOT/"configs/phase4_closeout.json").read_text(encoding="utf-8"))
    folder=ROOT/"data/preferences/dpo_v2"
    splits=load_preference_splits(folder,json.loads((folder/"manifest.json").read_text(encoding="utf-8")))
    cases=select_cases(splits['train'],config['case_ids'])
    from collections import Counter
    indexed={r.id:r for r in splits['train']}
    assert len(cases)==48 and set(config['case_ids']).isdisjoint(r.id for r in splits['development'])
    assert set(Counter(indexed[c.id].family_id for c in cases).values())=={4}
    assert set(Counter(c.category for c in cases).values())=={12}
    assert set(Counter(c.id.split('-')[-2] for c in cases).values())=={16}
    assert config['generation_requests']==len(cases)*4+2*config['benchmark_samples']==224
    assert len({sample_seed(g,i) for g in range(48) for i in range(4)})==192
    with pytest.raises(ValueError): select_cases(splits['train'],config['case_ids'][:-1])
    with pytest.raises(ValueError): select_cases(splits['train'],config['case_ids'][:-1]+[splits['development'][0].id])


class Tokenizer:
    eos_token_id=0
    pad_token_id=0
    def decode(self,tokens,**kw):
        return '{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}' + ('' if kw['skip_special_tokens'] else '<EOS>')


def records():
    case=EvalCase('a','math','1+2','tool_call','1+2',3)
    rows=[]
    for i in range(4):
        sample=RolloutSample('a','a',i,sample_seed(0,i),[1],[2,0],[-.1,-.2],[-.1,-.2],0,'eos',
            Tokenizer().decode([2,0],skip_special_tokens=True),Tokenizer().decode([2,0],skip_special_tokens=False))
        reward=score_reward(sample.text,case,finish_reason='eos')
        rows.append({'sample':asdict(sample),'case':asdict(case),'score':reward['score'],'semantic_pass':reward['semantic_pass']})
    return rows,[case],[[1]]


def test_validate_retains_duplicates_eos_and_zero_variance_reward():
    rows,cases,prompts=records()
    groups,packed,reward=validate_records(rows,cases,prompts,Tokenizer(),{'max_new_tokens':64})
    assert len(groups[0])==4 and packed['response_mask'].sum()==8
    assert reward['reward_sum']==4 and reward['zero_variance_rate']==1


@pytest.mark.parametrize('mutation',['missing','seed','prompt','decode','target','score','policy','index'])
def test_corrupt_closeout_records_are_rejected(mutation):
    rows,cases,prompts=records()
    if mutation=='missing': rows.pop()
    elif mutation=='seed': rows[0]['sample']['seed']=999
    elif mutation=='prompt': rows[0]['sample']['prompt_token_ids']=[3]
    elif mutation=='decode': rows[0]['sample']['text']='wrong'
    elif mutation=='target': rows[0]['case']['prompt']='changed'
    elif mutation=='score': rows[0]['score']={}
    elif mutation=='policy': rows[0]['sample']['policy_token_log_probs']=[-.3,-.2]
    else: rows[0]['sample']['sample_index']=2
    with pytest.raises(ValueError): validate_records(rows,cases,prompts,Tokenizer(),{'max_new_tokens':64})


def test_benchmark_reports_token_changes_and_logp_changes_separately():
    rows,_,_=records(); first=RolloutSample(**rows[0]['sample'])
    result=benchmark_summary([first],[replace(first,policy_token_log_probs=[-.11,-.2])],2,4)
    assert result['batch_over_serial_speedup']==2 and result['token_reproduction_rate']==1
    assert result['logp_reproduction_rate']==0
    other=replace(first,response_token_ids=[3,0])
    result=benchmark_summary([first],[other],2,4)
    assert result['comparisons'][0]['max_logp_error'] is None
    with pytest.raises(ValueError): benchmark_summary([first],[replace(first,seed=999)],2,4)


@pytest.mark.parametrize('price,spent',[(0,0),(-1,0),(float('nan'),0),(1,200),(1,-1),(1,float('inf'))])
def test_invalid_or_exhausted_budget_rejected(price,spent):
    with pytest.raises(ValueError): budget_limits(price,spent)


def test_budget_caps_runtime_and_reserves_cleanup():
    budget,seconds=budget_limits(2,10)
    assert budget==10 and seconds==1790
    budget,seconds=budget_limits(2,199)
    assert budget==1 and seconds==1790
    budget,seconds=budget_limits(4,199)
    assert budget==1 and seconds==890
    with pytest.raises(ValueError): budget_limits(2,199.999)
