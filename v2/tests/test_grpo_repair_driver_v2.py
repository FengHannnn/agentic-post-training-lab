import copy,json,sys,zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from grpo_common import read,atomic_json
from test_grpo_repair_study_v2 import config,result


@pytest.mark.parametrize('scenario',['repair_compound','repair_reminded','repair_simplified'])
def test_repair_prompts_replay_with_native_tokens_and_reject_tampering(scenario):
    from apt_v2.grpo_repair_study import training_tasks,context
    from apt_v2.grpo import replay
    from test_grpo_curriculum_v2 import native_episode,NativeTokenizer
    t=training_tasks()[3];tok=NativeTokenizer();e=native_episode(t,scenario,tok,config())
    assert replay(t,e,tok,context)==len(e['turns']) and e['success']
    bad=copy.deepcopy(e);bad['turns'][0]['generation']['prompt_token_ids'][0]+=1
    with pytest.raises(ValueError):replay(t,bad,tok,context)


def test_real_sft_batch_updates_only_lora_and_exact_adam_resume(tmp_path):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig,PeftModel
    from grpo_repair_worker import sft_batch
    from grpo_worker import save_policy_adapter
    cfg=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                   num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
    torch.manual_seed(42)
    model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),adapter_name='policy').eval()
    sample={'input_ids':[1,4,5,6,2],'labels':[-100,-100,-100,6,2],'prompt_length':3,'response_length':2}
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0)
    metrics=sft_batch(model,optimizer,[sample]*4)
    assert metrics['optimizer_steps']==[1] and metrics['frozen_parameter_gradients']==0
    assert all(p.equal(frozen[n]) for n,p in model.named_parameters() if n in frozen)
    save_policy_adapter(model,tmp_path/'adapter');state=copy.deepcopy(optimizer.state_dict())
    torch.manual_seed(42);loaded=PeftModel.from_pretrained(Qwen2ForCausalLM(cfg),tmp_path/'adapter',adapter_name='policy',is_trainable=True).eval()
    other=torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad],lr=1e-5,weight_decay=0);other.load_state_dict(state)
    a=sft_batch(model,optimizer,[sample]*4);b=sft_batch(loaded,other,[sample]*4)
    assert a==b and all(x.equal(y) for x,y in zip(model.parameters(),loaded.parameters()))
    # C starts from B weights with a new optimizer, without importing B's Adam state.
    copt=torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad],lr=1e-6,weight_decay=0)
    assert not copt.state


@pytest.mark.parametrize('route',['online_plain','online_reminder','warm_start'])
def test_real_driver_routes_and_b_gate_with_simulated_stages(tmp_path,monkeypatch,route):
    import run_v2_grpo_repair as runner
    cfg=read(ROOT/'configs/grpo-sql-repair-study-v1.json');study=runner.Study(cfg,tmp_path)
    w=result();w['elapsed_seconds']=2300
    b=result(full=15,target=3,first=4);b['elapsed_seconds']=2300
    calls=[]
    def child(name):
        p=tmp_path/'branches'/name;atomic_json(p/'config.json',config());return p
    def stage(name,*args,**kwargs):
        calls.append(name)
        if name=='diagnostic':atomic_json(tmp_path/'diagnostic/route.json',{'route':route})
    monkeypatch.setattr(study,'child',child);monkeypatch.setattr(study,'stage',stage)
    monkeypatch.setattr(study,'evaluate',lambda c,n,*a:b if n=='b' else w)
    def train(names,baseline):
        assert names==(['c'] if route=='warm_start' else ['plain'] if route=='online_plain' else ['control','curriculum'])
        return {n:result(full=16,target=4,first=5) for n in names},True
    monkeypatch.setattr(study,'train',train)
    outcome=study.execute();assert outcome['route']==route and outcome['selected_candidate']
    assert ('warm-start' in calls)==(route=='warm_start')


def test_b_quality_rejection_prevents_grpo_and_keeps_result(tmp_path,monkeypatch):
    import run_v2_grpo_repair as runner
    cfg=read(ROOT/'configs/grpo-sql-repair-study-v1.json');study=runner.Study(cfg,tmp_path)
    def child(name):
        p=tmp_path/'branches'/name;atomic_json(p/'config.json',config());return p
    monkeypatch.setattr(study,'child',child)
    monkeypatch.setattr(study,'stage',lambda name,*a,**k:atomic_json(tmp_path/'diagnostic/route.json',{'route':'warm_start'}))
    monkeypatch.setattr(study,'evaluate',lambda c,n,*a:dict(result(first=4,quality=n!='b'),elapsed_seconds=2300))
    monkeypatch.setattr(study,'train',lambda *a:pytest.fail('rejected B must not enter GRPO'))
    out=study.execute();assert out['stop_reason']=='b_gate_rejected' and out['selected_candidate'] is None


def test_failed_run_packages_partial_evidence_and_resume_counts_packaging(tmp_path,monkeypatch):
    import run_v2_grpo_repair as runner
    from grpo_common import file_hash
    cfg=read(ROOT/'configs/grpo-sql-repair-study-v1.json');config_path=tmp_path/'configs/master.json';atomic_json(config_path,cfg)
    atomic_json(tmp_path/runner.MANIFEST,{'files':{'configs/master.json':file_hash(config_path)}})
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'sqlite':'3.54.1','gpu_uuid':'same'})
    monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(runner.Study,'execute',lambda self:(_ for _ in ()).throw(RuntimeError('injected interruption')))
    with pytest.raises(RuntimeError,match='injected interruption'):runner.run(config_path)
    output=tmp_path/'artifacts'/cfg['run_id'];first=read(output/'run_status.json')
    assert first['elapsed_seconds']>=first['packaging_seconds'] and read(output/'selection.json')['selected_candidate'] is None
    archive=tmp_path/(cfg['run_id']+'-review-attempt-001.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        assert json.loads(z.read(cfg['run_id']+'/final_status.json'))['error']
    monkeypatch.setattr(runner.Study,'execute',lambda self:{'selected_candidate':None,'stop_reason':'completed'})
    monkeypatch.setattr(runner.Study,'stage',lambda *a,**k:None)
    runner.run(config_path,True);second=read(output/'run_status.json')
    first_receipt=read(archive.with_suffix('.receipt.json'))
    assert second['attempt']==2 and second['elapsed_seconds']>=first_receipt['elapsed_seconds_including_packaging'] and second['error'] is None
    with pytest.raises(ValueError,match='completed'):runner.run(config_path,True)
    with zipfile.ZipFile(tmp_path/(cfg['run_id']+'-review-attempt-002.zip')) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))


def test_incomplete_recipe_never_promotes_and_tie_prefers_control():
    from run_v2_grpo_repair import select
    w=result();candidate=result(full=15,target=3)
    assert select(w,{'control':candidate,'curriculum':candidate})['selected_candidate']=='control'
    assert select(w,{'control':candidate},complete=False)['selected_candidate'] is None
    assert select(w,{'control':dict(candidate,effective_bf16_change=False)})['selected_candidate'] is None


def test_new_schema_uses_same_forward_probability_capture(monkeypatch):
    import test_grpo_sampler_probe_v2 as fixture
    monkeypatch.setattr(fixture,'CFG',config())
    fixture.test_actual_same_forward_capture_overrides_cross_request_diagnostic_only(monkeypatch)


def test_prepare_c_accepts_only_audited_b_ancestry_and_records_sft537(tmp_path,monkeypatch):
    import grpo_common as common
    import apt_v2.grpo_curriculum as curriculum
    from grpo_common import atomic_json,file_hash
    c=config();initial=tmp_path/'artifacts/B';base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{}');(base/'model.safetensors').write_bytes(b'base fixture')
    (initial/'adapter').mkdir(parents=True);(initial/'adapter/adapter_model.safetensors').write_bytes(b'changed B fixture')
    (initial/'adapter/adapter_config.json').write_text('{}')
    c.update(repair_arm='warm',source_sft_updates=537,initial_dir='artifacts/B',base_dir='base',
             adapter_sha256=file_hash(initial/'adapter/adapter_model.safetensors'),profile_input_dir='profile',profile_binding={})
    complete={'total_sft_updates':537,'total_dpo_updates':11,'parent_W_sha256':c['baseline_W_sha256'],
              'adapter_sha256':{p.name:file_hash(p) for p in (initial/'adapter').iterdir()}}
    atomic_json(initial/'complete.json',complete)
    atomic_json(initial/'source_audit.json',{'passed':True,'actual_updates':24,'action_exposures':96})
    atomic_json(initial/'model_identity.json',{'model_id':c['base_model'],'resolved_revision':c['revision'],
                'files_sha256':{p.name:file_hash(p) for p in base.iterdir()}})
    atomic_json(tmp_path/'profile/audit.json',{'passed':True});atomic_json(tmp_path/'profile/run_status.json',{'error':None})
    output=tmp_path/'prepared';atomic_json(output/'config.json',c);atomic_json(output/'run_manifest.json',{'source_sha256':{}})
    actual={'gpu_name':c.get('required_gpu_name') or 'NVIDIA L40S','gpu_total_memory':c.get('required_min_gpu_bytes',0)+1,'packages':c['required_packages']}
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'runtime_identity',lambda:actual)
    monkeypatch.setattr(curriculum,'frozen_inputs',lambda cfg:{'targeted_evaluation_plan.json':[]})
    common.prepare(output)
    assert read(output/'input_binding.json')['source_sft_updates']==537
    assert read(output/'initial/complete.json')['parent_W_sha256']==c['baseline_W_sha256']
    complete['parent_W_sha256']='wrong';atomic_json(initial/'complete.json',complete)
    other=tmp_path/'bad';atomic_json(other/'config.json',c)
    with pytest.raises(ValueError,match='ancestry'):common.prepare(other)
