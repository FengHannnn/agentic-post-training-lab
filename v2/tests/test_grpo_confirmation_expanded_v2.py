import copy
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.grpo_confirmation import confirmation_tasks as tasks_for,confirmation_plan,independence_report,expert_confirmation,paired,acceptance,ADAPTERS
from apt_v2.grpo import parse_or_invalid,digest
from apt_v2.dpo import corrective_action
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.environment import Environment
from grpo_common import atomic_json
from grpo_c4_common import validate_config
from test_grpo_confirmation_v2 import NativeTokenizer,candidate_adapter
CFG=json.loads((ROOT/'configs/grpo-c4-confirmation-v2.json').read_text())

def confirmation_tasks():return tasks_for(CFG)


def test_expanded_12_tasks_extend_original_8_and_cover_all_months():
    ts=confirmation_tasks();assert ts[:8]==tasks_for() and len(ts)==12
    assert {t.current_month for t in ts}=={f'2027-{i:02}' for i in range(1,13)}
    r=independence_report(CFG);assert r['passed'] and len(set(r['database_sha256']))==12
    assert sum(t.fault_mode=='transient_query' for t in ts)==6
    p=confirmation_plan(CFG);assert len(p)==396 and 3*len(p)==1188
    assert sum(x['kind']=='recovery' and x['mode']=='sample' for x in p)==252
    assert sum(x['kind']=='complete' and x['mode']=='sample' for x in p)==48
    assert len({x['seed'] for x in p})==396


def test_expanded_96_expert_contexts_fit_original_budget():
    proof=[expert_confirmation(t,s) for t in confirmation_tasks() for s in (None,*SCENARIOS)]
    assert len(proof)==96 and all(p['passed'] and p['total_turns']<=12 for p in proof)


def counts(full=48,recovery=180,gfull=12,grecovery=60):
    return {'sample':{'complete':{'successes':full,'episodes':48},'recovery':{'successes':recovery,'episodes':252}},
        'greedy':{'complete':{'successes':gfull,'episodes':12},'recovery':{'successes':grecovery,'episodes':84}}}


def test_expanded_practical_margin_preserves_238_percentage_points():
    assert 6/252==4/168
    w=counts();assert not acceptance(w,counts(recovery=185),CFG)['passed']
    assert acceptance(w,counts(recovery=186),CFG)['passed']
    assert not acceptance(w,counts(full=47,recovery=186),CFG)['passed']
    assert not acceptance(w,counts(recovery=186,grecovery=59),CFG)['passed']
    assert not acceptance(w,counts(recovery=186,gfull=11),CFG)['passed']
    assert not acceptance(counts(full=46),counts(full=46,recovery=186),CFG)['passed']
    assert acceptance(counts(full=46),counts(full=47,recovery=186),CFG)['passed']


def test_expanded_bootstrap_and_by_task_include_new_four_families():
    a=[];b=[]
    for item in confirmation_plan(CFG):
        e={**item,'evaluation_mode':item['mode'],'success':False};a.append(e)
        b.append({**e,'success':item['task_id']==confirmation_tasks()[-1].id})
    result=paired(a,b,cfg=CFG,repetitions=2000)['sample']['recovery']
    assert len(result['by_task'])==12 and result['improved']==21 and result['delta_rate']==pytest.approx(1/12)
    assert result['task_cluster_bootstrap_95pct'][0]==0


def test_expanded_config_freezes_seven_hour_bound_and_six_success_margin():
    assert validate_config(CFG)==CFG and CFG['max_wall_seconds']==25200
    for field,value in [('minimum_recovery_gain',4),('task_count',8),('max_wall_seconds',25201)]:
        bad=copy.deepcopy(CFG);bad[field]=value
        with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(CFG);bad['release_manifest']='grpo_c4_release_manifest.json'
    with pytest.raises(ValueError):validate_config(bad)


@pytest.fixture
def prepared_output(tmp_path):
    import shutil
    p=tmp_path/'output';p.mkdir();atomic_json(p/'config.json',CFG)
    shutil.copytree(candidate_adapter(),p/'inputs/weighted_sft/adapter')
    atomic_json(p/'execution_environment.json',{'gpu_name':'NVIDIA L40S','gpu_uuid':'test-device'})
    atomic_json(p/'driver_identity.json',{'stdout':'test-driver','stderr':'','exit_code':0})
    return p


def test_expanded_native_inference_resume_preserves_all_396_episodes(prepared_output,monkeypatch):
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
    data=json.loads((p/'evaluations/weighted_sft.json').read_text());assert len(data['episodes'])==396 and len(data['sessions'])==2
    assert all((p/'episodes/weighted_sft'/name).read_bytes()==content for name,content in first.items())
    assert all(e['success'] for e in data['episodes']) and c.verify_model(p,'weighted_sft',tok)['passed']
