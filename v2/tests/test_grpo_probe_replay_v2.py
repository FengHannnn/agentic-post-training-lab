"""Recorded native q regression at the actual probability-probe call seam."""
import copy
import inspect
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import grpo_vllm
from apt_v2.runtime import Generation
from grpo_common import read
FIXTURE=read(ROOT/'tests/fixtures/grpo_probe_prefill_0018.json')


def probe_case(monkeypatch,mutation=None):
    fixture=copy.deepcopy(FIXTURE);g=Generation(**fixture['generation'])
    cfg=read(ROOT/'configs/grpo-outcome-curriculum-control-v1.json')
    params=[];requests=[]
    class SamplingParams:
        def __init__(self,**kwargs):self.values=kwargs;params.append(kwargs)
    module=types.ModuleType('vllm');module.SamplingParams=SamplingParams
    monkeypatch.setitem(sys.modules,'vllm',module)
    class Tokenizer:
        eos_token_id=151645
        def decode(self,ids,**kwargs):return fixture['generation']['text']
    class Engine:
        def generate(self,prompts,sampling,lora_request=None,use_tqdm=None):
            requests.append((prompts,sampling.values,lora_request))
            if len(requests)==1:
                prompt=g.prompt_token_ids+g.response_token_ids[:16]
                scores=[None]*len(g.prompt_token_ids)+[{token:SimpleNamespace(logprob=p)} for token,p in zip(
                    fixture['failed_probe']['tokens'],fixture['failed_probe']['teacher_forced_vllm_logp'])]
                return [SimpleNamespace(prompt_token_ids=prompt,prompt_logprobs=scores,
                    outputs=[SimpleNamespace(token_ids=[99])])]
            native=copy.deepcopy(fixture['generation'])
            if mutation:mutation(native)
            return [SimpleNamespace(prompt_token_ids=native['prompt_token_ids'],outputs=[SimpleNamespace(
                token_ids=native['response_token_ids'],logprobs=[{token:SimpleNamespace(logprob=p)} for token,p in zip(
                    native['response_token_ids'],native['behavior_token_log_probs'])],finish_reason=native['finish_reason'])])]
    kwargs={'seed':fixture['seed']} if 'seed' in inspect.signature(grpo_vllm.probability_probe).parameters else {}
    proof=grpo_vllm.probability_probe(Engine(),Tokenizer(),'one-fixed-adapter',g,cfg,**kwargs)
    return proof,g,cfg,requests,params


def test_captured_prefill_false_positive_requires_identical_seeded_decode(monkeypatch):
    proof,g,cfg,requests,params=probe_case(monkeypatch)
    assert proof['passed'], 'captured native q rejected only because prefill error 0.328790 exceeds 0.25'
    assert proof['max_absolute_error']==FIXTURE['failed_probe']['max_absolute_error']
    assert proof['gate']==.25 and proof['max_absolute_error']>proof['gate']
    assert len(requests)==2 and requests[1][0]==[{'prompt_token_ids':g.prompt_token_ids}]
    assert requests[1][1]['seed']==FIXTURE['seed'] and requests[1][1]['max_tokens']==256
    assert requests[1][1]['temperature']==1 and requests[1][1]['top_p']==1 and requests[1][1]['top_k']==-1
    assert requests[1][2]=='one-fixed-adapter'
    assert proof['decode_replay']['max_absolute_error']==0 and proof['decode_replay']['gate']==1e-5
    assert proof['extra_generated_tokens']==1+len(g.response_token_ids)
    assert grpo_vllm.verify_probability_probe(proof,asdict(g),cfg,FIXTURE['seed'])


@pytest.mark.parametrize('kind',['q','tokens','finish'])
def test_decode_replay_mismatch_still_fails_probability_gate(monkeypatch,kind):
    def mutate(g):
        if kind=='q':g['behavior_token_log_probs'][-1]-=.01
        elif kind=='tokens':g['response_token_ids'][-1]=151643
        else:g['finish_reason']='length'
    proof,g,cfg,_,_=probe_case(monkeypatch,mutate)
    assert not proof['passed']
    with pytest.raises(ValueError):grpo_vllm.verify_probability_probe(proof,asdict(g),cfg,FIXTURE['seed'])


@pytest.mark.parametrize('kind',['seed','prompt','generation','sampling','gate','metric','missing'])
def test_independent_probe_verifier_rejects_forged_replay(monkeypatch,kind):
    proof,g,cfg,_,_=probe_case(monkeypatch);assert proof['passed']
    p=copy.deepcopy(proof)
    if kind=='seed':p['decode_replay']['seed']+=1
    elif kind=='prompt':p['decode_replay']['prompt_token_ids'][0]+=1
    elif kind=='generation':p['decode_replay']['original_generation_sha256']='wrong'
    elif kind=='sampling':p['decode_replay']['sampling']['temperature']=.8
    elif kind=='gate':p['decode_replay']['gate']=1.
    elif kind=='metric':p['decode_replay']['max_absolute_error']=.1
    else:p.pop('decode_replay')
    with pytest.raises(ValueError):grpo_vllm.verify_probability_probe(p,asdict(g),cfg,FIXTURE['seed'])


def test_all_legacy_committed_probe_fields_remain_valid():
    cfg=read(ROOT/'configs/grpo-outcome-curriculum-control-v1.json')
    g=FIXTURE['generation'];q=g['behavior_token_log_probs'][:16]
    proof={**FIXTURE['failed_probe'],'teacher_forced_vllm_logp':q,'max_absolute_error':0.,'passed':True}
    assert grpo_vllm.verify_probability_probe(proof,g,cfg,FIXTURE['seed'])

