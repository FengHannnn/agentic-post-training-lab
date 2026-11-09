import copy
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read
from apt_v2.runtime import Generation
from grpo_sampler_probe import install_capture,capture_snapshot,verify_capture,METHOD

CFG=read(ROOT/'configs/grpo-outcome-curriculum-control-v1.json')

class TopK(torch.nn.Module):
    def forward(self,logits,generators,k,p):
        ids=torch.multinomial(logits.softmax(-1),1,generator=generators[0]).squeeze(-1)
        return ids,logits.log_softmax(-1,dtype=torch.float32)

class Sampler(torch.nn.Module):
    def __init__(self,bias=0.):
        super().__init__();self.topk_topp_sampler=TopK();self.logprobs_mode='processed_logprobs';self.bias=bias
    def forward(self,logits,sampling_metadata):
        logits=logits.float()
        if self.bias:logits=logits.clone();logits[:,0]+=self.bias
        ids,lp=self.topk_topp_sampler(logits,sampling_metadata.generators,sampling_metadata.top_k,sampling_metadata.top_p)
        greedy=lp.argmax(-1);indices=torch.stack([ids,greedy],dim=1)
        values=lp.gather(-1,indices)
        return NS(sampled_token_ids=ids[:,None].int(),logprobs_tensors=NS(logprob_token_ids=indices.int(),logprobs=values))


def worker_case(bias=0.):
    sampler=Sampler(bias);params=NS(seed=123)
    request=NS(sampling_params=params,lora_request=NS(lora_int_id=1),prompt_token_ids=[11,12])
    worker=NS(model_runner=NS(sampler=sampler,input_batch=NS(req_ids=['first']),requests={'first':request}))
    setup={'seed':123,'policy_sha256':'fixed-policy','token_limit':16,'sampling':copy.deepcopy(CFG['sampling'])}
    md=NS(temperature=torch.tensor([1.]),all_greedy=False,all_random=True,top_k=None,top_p=None,
          no_penalties=True,allowed_token_ids_mask=None,bad_words_token_ids={},max_num_logprobs=1,
          generators={0:torch.Generator().manual_seed(921)})
    return worker,setup,md


def gen_from(output,prompt=[11,12]):
    return Generation('native','sample',prompt,output.sampled_token_ids[0].tolist(),None,
                      [float(output.logprobs_tensors.logprobs[0,0])],'stop')


def test_same_forward_observer_keeps_native_output_logits_and_rng():
    worker,setup,md=worker_case();plain=Sampler();plain_md=copy.deepcopy(md)
    logits=torch.tensor([[.1,.2,1.1,-.4]],dtype=torch.bfloat16);unchanged=logits.clone()
    expected=plain(logits.clone(),plain_md)
    install_capture(worker,setup);actual=worker.model_runner.sampler(logits=logits,sampling_metadata=md)
    torch.testing.assert_close(actual.logprobs_tensors.logprobs,expected.logprobs_tensors.logprobs,rtol=0,atol=0)
    assert torch.equal(actual.sampled_token_ids,expected.sampled_token_ids)
    assert torch.equal(logits,unchanged) and torch.equal(md.generators[0].get_state(),plain_md.generators[0].get_state())
    capture=capture_snapshot(worker);check=verify_capture(capture,gen_from(actual),setup['sampling'],123,'fixed-policy')
    assert check['passed'] and check['maximum_error']<1e-5 and capture['rows'][0]['full_logits_unchanged']


def test_capture_stops_at_first_request_even_when_response_is_short():
    worker,setup,md=worker_case();install_capture(worker,setup)
    first=worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
    worker.model_runner.input_batch.req_ids=['next'];worker.model_runner.requests['next']=copy.deepcopy(worker.model_runner.requests['first'])
    worker.model_runner.requests['next'].sampling_params.seed=999
    worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
    snapshot=capture_snapshot(worker)
    assert len(snapshot['rows'])==1 and snapshot['closed']
    assert verify_capture(snapshot,gen_from(first),setup['sampling'],123,'fixed-policy')['passed']


def test_capture_stops_after_sixteen_tokens():
    worker,setup,md=worker_case();install_capture(worker,setup)
    for i in range(20):worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
    assert len(capture_snapshot(worker)['rows'])==16


def test_changed_sampler_processing_is_rejected_even_with_finite_q():
    worker,setup,md=worker_case(bias=.5);install_capture(worker,setup)
    output=worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
    with pytest.raises(ValueError,match='token/metadata'):verify_capture(capture_snapshot(worker),gen_from(output),setup['sampling'],123,'fixed-policy')


@pytest.mark.parametrize('field,value',[('temperature',torch.tensor([.8])),('top_k',torch.tensor([2])),
    ('top_p',torch.tensor([.9])),('no_penalties',False),('all_random',False),
    ('allowed_token_ids_mask',torch.tensor([[False,True,False,False]])),('bad_words_token_ids',{0:[[1]]}),('max_num_logprobs',None)])
def test_non_neutral_actual_metadata_fails_before_sampling(field,value):
    worker,setup,md=worker_case();install_capture(worker,setup);setattr(md,field,value)
    rng=md.generators[0].get_state().clone()
    with pytest.raises(ValueError,match='metadata'):worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
    assert torch.equal(rng,md.generators[0].get_state())


@pytest.mark.parametrize('field',['seed','policy','prompt','q','normalizer','token','summary'])
def test_capture_verifier_rejects_tampered_identity_or_probabilities(field):
    worker,setup,md=worker_case();install_capture(worker,setup)
    output=worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md);cap=capture_snapshot(worker);g=asdict(gen_from(output))
    if field=='seed':cap['request']['seed']+=1
    elif field=='policy':cap['setup']['policy_sha256']='wrong'
    elif field=='prompt':cap['request']['prompt_token_ids'][0]+=1
    elif field=='q':g['behavior_token_log_probs'][0]-=.01
    elif field=='normalizer':cap['rows'][0]['raw_logsumexp_fp64']+=.1
    elif field=='token':cap['rows'][0]['token_id']=(cap['rows'][0]['token_id']+1)%4
    else:cap['rows'][0]['normalization_error']=1.
    with pytest.raises(ValueError):verify_capture(cap,g,setup['sampling'],123,'fixed-policy')


def test_real_second_failure_remains_rejected_by_legacy_cross_request_gate():
    import grpo_vllm
    f=read(ROOT/'tests/fixtures/grpo_probe_repeat_attempt002.json')
    assert f['failed_probe']['decode_replay']['response_token_ids']==f['generation']['response_token_ids']
    assert f['failed_probe']['decode_replay']['max_absolute_error']>.17
    with pytest.raises(ValueError,match='probability gate'):grpo_vllm.verify_probability_probe(f['failed_probe'],f['generation'],CFG,f['seed'])


def test_native_frontend_checks_first_action_immediately(monkeypatch):
    import grpo_vllm
    worker,setup,md=worker_case();install_capture(worker,setup)
    class Params:
        def __init__(self,**kw):self.kw=kw
    module=types.ModuleType('vllm');module.SamplingParams=Params;monkeypatch.setitem(sys.modules,'vllm',module)
    class Engine:
        _grpo_sampler_capture_setup=setup;_grpo_first_probability_verified=False
        def generate(self,prompts,params,**kw):
            out=worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md)
            token=int(out.sampled_token_ids[0,0]);q=float(out.logprobs_tensors.logprobs[0,0])
            return [NS(prompt_token_ids=prompts[0]['prompt_token_ids'],outputs=[NS(token_ids=[token],logprobs=[{token:NS(logprob=q)}],finish_reason='stop')])]
        def collective_rpc(self,name,**kw):
            assert name=='grpo_probability_capture_snapshot';return [capture_snapshot(worker)]
    class Tok:
        eos_token_id=1
        def decode(self,*a,**kw):return 'native'
    engine=Engine();policy=grpo_vllm.BehaviorPolicy(engine,Tok(),1,setup['sampling'],123)
    g=policy.generate_prompt([11,12]);assert engine._grpo_first_probability_verified and policy.calls==1
    assert engine._grpo_first_probability_evidence['generation']==asdict(g)


def test_actual_same_forward_capture_overrides_cross_request_diagnostic_only(monkeypatch):
    import grpo_vllm
    worker,setup,md=worker_case();install_capture(worker,setup)
    output=worker.model_runner.sampler(torch.tensor([[1.,2.,3.,4.]]),md);g=gen_from(output);q=g.behavior_token_log_probs[0]
    class Params:
        def __init__(self,**kw):pass
    module=types.ModuleType('vllm');module.SamplingParams=Params;monkeypatch.setitem(sys.modules,'vllm',module)
    class Engine:
        def generate(self,prompts,*a,**kw):
            return [NS(prompt_token_ids=prompts[0]['prompt_token_ids'],prompt_logprobs=[None,None,{g.response_token_ids[0]:NS(logprob=q-.33)}],outputs=[NS(token_ids=[1])])]
    proof=[{'adapter_sha256':'fixed-policy','same_forward_capture':capture_snapshot(worker)}]
    p=grpo_vllm.probability_probe(Engine(),None,1,g,CFG,seed=123,worker_proof=proof)
    assert p['passed'] and not p['prefill_passed'] and p['max_absolute_error']>.25
    assert p['verification_method']==METHOD and p['same_forward_verification']['maximum_error']<1e-5
    assert grpo_vllm.verify_probability_probe(p,asdict(g),CFG,123,proof)
    bad=copy.deepcopy(p);bad['same_forward']['rows'][0]['raw_logsumexp_fp64']+=1
    with pytest.raises(ValueError):grpo_vllm.verify_probability_probe(bad,asdict(g),CFG,123,proof)


def test_full_vocabulary_fp64_rounding_is_diagnostic_not_a_sampling_change():
    worker,setup,md=worker_case();install_capture(worker,setup)
    logits=torch.randn((1,152064),generator=torch.Generator().manual_seed(919))*4+20
    output=worker.model_runner.sampler(logits,md);capture=capture_snapshot(worker)
    check=verify_capture(capture,gen_from(output),setup['sampling'],123,'fixed-policy')
    assert check['passed'] and check['maximum_error']==0.
    assert capture['rows'][0]['fp64_precision_difference']>1e-5
