import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.grpo_confirmation import (LABELS,ADAPTERS,confirmation_tasks,confirmation_plan,independence_report,
    key,paired,acceptance,summarize,expert_confirmation)
from apt_v2.grpo import run_continuation,initial_state,parse_or_invalid,digest
from apt_v2.runtime import Generation
from apt_v2.dpo import corrective_action
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.environment import Environment
from grpo_common import atomic_json,file_hash
from grpo_c4_common import verify_episode,verify_session,validate_config
from run_v2_grpo_c4 import resume_spent
CFG=json.loads((ROOT/'configs/grpo-c4-confirmation-v1.json').read_text())

class NativeTokenizer:
    eos_token_id=1;name_or_path=''
    def __init__(self):self.tokens={};self.text={}
    def token(self,s):
        if s not in self.tokens:
            n=len(self.tokens)+2;self.tokens[s]=n;self.text[n]=s
        return self.tokens[s]
    def apply_chat_template(self,messages,**kwargs):return [0,self.token(json.dumps(messages,sort_keys=True))]
    def decode(self,ids,**kwargs):return ''.join(self.text[i] for i in ids if i not in (0,1))
    def generation(self,messages,text,sampling):
        ids=[self.token(text),1];sample=sampling['temperature']>0
        return Generation(text,'sample' if sample else 'greedy',self.apply_chat_template(messages),ids,None,[-1.,-1.] if sample else None,'stop')


def real_episode(task,item,tok):
    env,_,_,_=initial_state(task,item['scenario'])
    sampling={**CFG['sampling'],'temperature':item['temperature']}
    def policy(messages):
        action=corrective_action(env);env.step(action);return tok.generation(messages,json.dumps(action),sampling)
    try:e=run_continuation(task,item['scenario'],policy,sampling,0,item['seed'],item['rollout_index'])
    finally:env.close()
    e.update(model_branch='weighted_sft',evaluation_mode=item['mode'],session_id='123',plan_index=0,worker_proof_sha256='a'*64);return e


def test_new_holdout_identity_database_and_year_boundary():
    r=independence_report();assert r['passed'] and len(set(r['database_sha256']))==8
    ts=confirmation_tasks();assert ts[0].prior_month=='2026-12' and ts[0].current_month=='2027-01'
    assert sum(t.fault_mode=='transient_query' for t in ts)==4 and all(t.split=='test' for t in ts)
    assert r['new_business_template'] is False


def test_exact_792_paired_budget_and_no_generation_seed_overlap():
    p=confirmation_plan(CFG);assert len(p)==264 and len({key(i) for i in p})==264
    assert len({i['seed'] for i in p})==264
    assert sum(i['mode']=='sample' and i['kind']=='recovery' for i in p)==168
    assert sum(i['mode']=='sample' and i['kind']=='complete' for i in p)==32
    assert 3*len(p)==792


def test_all_64_holdout_expert_paths_fit_original_turn_budget():
    proof=[expert_confirmation(t,s) for t in confirmation_tasks() for s in (None,*SCENARIOS)]
    assert len(proof)==64 and all(p['passed'] and p['total_turns']<=12 for p in proof)
    assert max(p['prefix_turns'] for p in proof)>0


def counts(full=32,recovery=120,gfull=8,grecovery=40):
    return {'sample':{'complete':{'successes':full,'episodes':32},'recovery':{'successes':recovery,'episodes':168}},
            'greedy':{'complete':{'successes':gfull,'episodes':8},'recovery':{'successes':grecovery,'episodes':56}}}


def test_practical_acceptance_exact_threshold_and_regression_guards():
    w=counts();assert not acceptance(w,counts(recovery=123))['passed']
    assert acceptance(w,counts(recovery=124))['passed']
    assert not acceptance(w,counts(full=31,recovery=124))['passed']
    assert not acceptance(w,counts(recovery=124,grecovery=39))['passed']
    assert not acceptance(w,counts(recovery=124,gfull=7))['passed']


def test_unsaturated_complete_metric_keeps_prior_priority():
    w=counts(full=30);assert not acceptance(w,counts(full=30,recovery=124))['passed']
    result=acceptance(w,counts(full=31,recovery=124));assert result['primary_metric']=='complete' and result['passed']


def test_paired_counts_and_task_cluster_interval_not_independent_rollouts():
    before=[];after=[]
    for item in confirmation_plan(CFG):
        e={**item,'evaluation_mode':item['mode'],'success':False};before.append(e);after.append({**e,'success':item['task_id']==confirmation_tasks()[0].id})
    p=paired(before,after,repetitions=2000)['sample']['recovery']
    assert p['improved']==21 and p['regressed']==0 and p['delta_rate']==pytest.approx(1/8)
    assert p['task_cluster_bootstrap_95pct'][0]==0
    with pytest.raises(ValueError,match='unique'):paired(before,after+after[:1],repetitions=2)


@pytest.mark.parametrize('scenario',[None,*SCENARIOS])
def test_native_holdout_replay_and_calendar_evidence(scenario):
    task=confirmation_tasks()[0];item=next(p for p in confirmation_plan(CFG) if p['task_id']==task.id and p['scenario']==scenario and p['mode']=='sample');tok=NativeTokenizer();e=real_episode(task,item,tok)
    result=verify_episode(task,item,e,'weighted_sft',tok,CFG,'123','a'*64)
    assert e['success'] and result['calendar']['calendar_consistent_success']
    bad=copy.deepcopy(e);bad['seed']+=1
    with pytest.raises(ValueError,match='identity'):verify_episode(task,item,bad,'weighted_sft',tok,CFG,'123','a'*64)
    bad=copy.deepcopy(e);bad['turns'][0]['generation']['response_token_ids'][-1]=99999
    with pytest.raises((ValueError,KeyError)):verify_episode(task,item,bad,'weighted_sft',tok,CFG,'123','a'*64)


def candidate_adapter():
    prepared=ROOT/CFG['models']['weighted_sft']['input_dir']/'adapter'
    return prepared if prepared.exists() else ROOT/'artifacts/inputs/grpo-w-20261105/adapter'


@pytest.fixture
def prepared_output(tmp_path):
    import shutil
    p=tmp_path/'output';p.mkdir();atomic_json(p/'config.json',CFG)
    shutil.copytree(candidate_adapter(),p/'inputs/weighted_sft/adapter')
    atomic_json(p/'execution_environment.json',{'gpu_name':'NVIDIA L40S','gpu_uuid':'test-device'})
    atomic_json(p/'driver_identity.json',{'stdout':'test-driver','stderr':'','exit_code':0})
    return p


def test_complete_session_checks_actual_bf16_projection_and_repeated_quality(prepared_output):
    import grpo_c4_common as c
    from grpo_vllm import projected_checkpoint_tensors
    p=prepared_output;tok=NativeTokenizer();quality=[];tasks={t.id:t for t in confirmation_tasks()}
    for spec in c.quality_specs(CFG):
        from apt_v2.dpo_recovery import build_context
        messages=build_context(tasks[spec['task_id']],spec['scenario'],allowed_split='test')['messages']
        g=asdict(tok.generation(messages,json.dumps({'tool':'read_policy','arguments':{}}),{'temperature':0}))
        quality.append({**spec,'generations':[g,copy.deepcopy(g)]})
    tree=projected_checkpoint_tensors(p/'inputs/weighted_sft/adapter')
    proof={'passed':True,'adapter_sha256':ADAPTERS['weighted_sft'],'registered_ids':[1],'active_slots':[1],
           'tensor_count':112,'tensors':tree,'loaded_tensor_sha256':digest(tree)}
    s={'session_id':'123','label':'weighted_sft','config_sha256':file_hash(p/'config.json'),'adapter_sha256':ADAPTERS['weighted_sft'],
       'hardware':json.loads((p/'execution_environment.json').read_text()),'driver':json.loads((p/'driver_identity.json').read_text()),
       'quality':quality,'worker_proof':[proof],'one_adapter_one_process':True,'max_loras':1,'lora_id':1}
    assert verify_session(p,'weighted_sft',s,tok,CFG)
    bad=copy.deepcopy(s);bad['worker_proof'][0]['loaded_tensor_sha256']='changed'
    with pytest.raises(ValueError,match='tensor proof'):verify_session(p,'weighted_sft',bad,tok,CFG)
    bad=copy.deepcopy(s);bad['hardware']['gpu_uuid']='other'
    with pytest.raises(ValueError,match='session binding'):verify_session(p,'weighted_sft',bad,tok,CFG)


def test_cumulative_resume_counts_unrecorded_active_stage_time():
    assert resume_spent({'elapsed_seconds':100,'active_started_unix':1000},{'elapsed_seconds':90},now=1040)==140
    assert resume_spent({'elapsed_seconds':100,'active_started_unix':None},{'elapsed_seconds':120},now=1040)==120


def test_config_cannot_change_preselected_candidate_or_acceptance():
    assert validate_config(CFG)==CFG
    bad=copy.deepcopy(CFG);bad['models']['fine_rl17']['adapter_sha256']='other'
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(CFG);bad['minimum_recovery_gain']=1
    with pytest.raises(ValueError):validate_config(bad)


def test_interrupted_inference_resumes_verified_episode_transactions(prepared_output,monkeypatch):
    import grpo_c4_common as c
    import grpo_c4_worker as worker
    import grpo_vllm as v
    from grpo_vllm import projected_checkpoint_tensors
    p=prepared_output;tok=NativeTokenizer();tree=projected_checkpoint_tensors(p/'inputs/weighted_sft/adapter')
    proof={'passed':True,'adapter_sha256':ADAPTERS['weighted_sft'],'registered_ids':[1],'active_slots':[1],
           'tensor_count':112,'tensors':tree,'loaded_tensor_sha256':digest(tree)}
    class Engine:
        def collective_rpc(self,*args,**kwargs):return [proof]
    class Behavior:
        def __init__(self,engine,tok,lora,sampling,seed,on_action=None):self.tok=tok;self.sampling=sampling;self.calls=0;self.callback=on_action
        def __call__(self,messages):
            task=next(t for t in confirmation_tasks() if t.request()==messages[1]['content']);env=Environment(task)
            try:
                for m in messages[2::2]:env.step(parse_or_invalid(m['content']))
                action=corrective_action(env)
            finally:env.close()
            g=self.tok.generation(messages,json.dumps(action),self.sampling);self.calls+=1
            if self.callback:self.callback(len(g.response_token_ids))
            return g
    monkeypatch.setattr(c,'verify_inputs',lambda *a,**k:CFG);monkeypatch.setattr(worker,'verify_inputs',lambda *a,**k:CFG)
    monkeypatch.setattr(worker,'runtime_identity',lambda:json.loads((p/'execution_environment.json').read_text()))
    monkeypatch.setattr(worker,'driver_identity',lambda:json.loads((p/'driver_identity.json').read_text()))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(v,'engine_and_tokenizer',lambda *a:(Engine(),tok,None));monkeypatch.setattr(v,'BehaviorPolicy',Behavior);monkeypatch.setattr(v,'shutdown',lambda e:None)
    real=worker.run_continuation;n=[0]
    def interrupted(*args,**kwargs):
        n[0]+=1
        if n[0]>5:raise RuntimeError('injected inference interruption')
        return real(*args,**kwargs)
    monkeypatch.setattr(worker,'run_continuation',interrupted)
    with pytest.raises(RuntimeError,match='injected'):worker.evaluate(p,'weighted_sft')
    first={q.name:q.read_bytes() for q in (p/'episodes/weighted_sft').glob('*.json')};assert len(first)==5
    monkeypatch.setattr(worker,'run_continuation',real);worker.evaluate(p,'weighted_sft')
    data=json.loads((p/'evaluations/weighted_sft.json').read_text());assert len(data['episodes'])==264 and len(data['sessions'])==2
    assert all((p/'episodes/weighted_sft'/name).read_bytes()==content for name,content in first.items())
    assert all(e['success'] for e in data['episodes']) and c.verify_model(p,'weighted_sft',tok)['passed']


def test_episode_requires_its_post_generation_loaded_weight_proof():
    task=confirmation_tasks()[0];item=confirmation_plan(CFG)[0];tok=NativeTokenizer();e=real_episode(task,item,tok)
    e['worker_proof_sha256']='b'*64
    with pytest.raises(ValueError,match='identity'):verify_episode(task,item,e,'weighted_sft',tok,CFG,'123','a'*64)
