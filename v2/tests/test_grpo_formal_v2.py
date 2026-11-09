import copy
import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from grpo_common import read, validate_config, atomic_json, file_hash
from apt_v2.grpo_evaluation import (development_tasks, evaluation_plan, paired_evaluation,
                                    summarize_evaluation)
CFG = read(ROOT/'configs/grpo-outcome-formal-v1.json')


def test_formal_plan_matches_132_episode_legacy_seed_protocol():
    from dpo_uniform_common import seed_for
    plan = evaluation_plan(CFG)
    assert len(plan) == 132
    tasks = {t.id: t for t in development_tasks()}
    counts = {}
    for p in plan:
        context = {'scenario': p['scenario']} if p['scenario'] else None
        assert p['seed'] == seed_for(CFG, tasks[p['task_id']], p['mode'], p['rollout_index'], context)
        key = p['mode'], p['kind']
        counts[key] = counts.get(key, 0)+1
        assert p['task_id'].startswith('development-')
    assert counts == {('greedy','complete'):4, ('sample','complete'):16,
                      ('greedy','recovery'):28, ('sample','recovery'):84}
    assert CFG['group_evaluation_cursors'] == [8,24]
    assert CFG['max_groups'] == CFG['max_updates'] == 24
    validate_config(CFG)


@pytest.mark.parametrize('field,value', [('max_groups', 23), ('group_evaluation_cursors', [24,8]),
                                        ('max_updates', 12), ('formal_job_index', 4), ('max_wall_seconds',14401)])
def test_formal_config_rejects_budget_or_schedule_drift(field,value):
    bad = copy.deepcopy(CFG)
    bad[field] = value
    with pytest.raises(ValueError): validate_config(bad)


def test_paired_statistics_have_explicit_task_denominators():
    before = [{'task_id':t.id, 'scenario':None, 'evaluation_mode':'sample','rollout_index':0,'success':s}
              for t,s in zip(development_tasks(),[True,False,True,False])]
    after = copy.deepcopy(before)
    after[0]['success'] = False
    after[1]['success'] = True
    result = paired_evaluation(before, after)['sample']['complete']
    assert result['episodes'] == 4 and result['improved'] == result['regressed'] == 1
    assert result['delta_successes'] == 0
    assert result['by_task']['development-001']['improved'] == 1
    with pytest.raises(ValueError): paired_evaluation(before, after[:-1])
    with pytest.raises(ValueError): paired_evaluation(before, after+[after[0]])


def adapter_fixture(tmp_path):
    import torch
    from safetensors.torch import save_file
    from grpo_vllm import projected_checkpoint_tensors
    from apt_v2.grpo import digest
    path = tmp_path/'initial/adapter'
    path.mkdir(parents=True)
    tensors = {}
    for i in range(28):
        for p in ['q_proj','v_proj']:
            tensors[f'base_model.model.model.layers.{i}.self_attn.{p}.lora_A.weight'] = torch.full((8,4),.125)
            tensors[f'base_model.model.model.layers.{i}.self_attn.{p}.lora_B.weight'] = torch.full((4,8),.25)
    save_file(tensors, path/'adapter_model.safetensors')
    atomic_json(path/'adapter_config.json', {'r':8,'lora_alpha':16,'target_modules':['q_proj','v_proj']})
    projected = projected_checkpoint_tensors(path)
    proof = [{'passed':True, 'adapter_sha256':file_hash(path/'adapter_model.safetensors'),
              'registered_ids':[1], 'active_slots':[1], 'tensor_count':112,
              'tensors':projected, 'loaded_tensor_sha256':digest(projected)}]
    return path, proof


def test_independent_worker_projection_has_scaled_bfloat16_B(tmp_path):
    import torch,hashlib
    path,proof = adapter_fixture(tmp_path)
    data = proof[0]['tensors']
    expected = torch.full((4,8),.5,dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    assert data['model.layers.0.self_attn.qkv_proj/lora_b/0']['sha256'] == hashlib.sha256(expected).hexdigest()
    assert len(data) == 112


def evaluation_fixture(tmp_path, monkeypatch):
    from dataclasses import asdict
    import evaluate_v2_grpo as ev
    from apt_v2.grpo import initial_state,run_continuation
    from apt_v2.runtime import Generation
    from apt_v2.dpo import corrective_action
    from apt_v2.dpo_recovery import build_context
    from test_dpo_v2 import CharTokenizer
    cfg = copy.deepcopy(CFG)
    cfg['sampling'].update(max_new_tokens=10000,max_model_len=100000)
    path,proof = adapter_fixture(tmp_path)
    cfg['adapter_sha256'] = proof[0]['adapter_sha256']
    atomic_json(tmp_path/'config.json',cfg)
    full = evaluation_plan(cfg)
    task = development_tasks()[0]
    plan = [next(p for p in full if p['task_id']==task.id and p['scenario']==s and p['mode']==m)
            for s in [None,'format_repair'] for m in ['greedy','sample']]
    atomic_json(tmp_path/'evaluation_plan.json',plan)
    monkeypatch.setattr(ev,'verify_inputs',lambda *a,**k:cfg)
    monkeypatch.setattr(ev,'evaluation_plan',lambda cfg:plan)
    monkeypatch.setattr(ev,'quality_specs',lambda cfg:[{'task_id':task.id,'scenario':'format_repair'}])
    tok = CharTokenizer()
    def native(text,messages,mode):
        ids = [ord(c)+1 for c in text]+[0]
        return Generation(text,mode,tok.apply_chat_template(messages,add_generation_prompt=True),ids,None,
                          [-.1]*len(ids) if mode=='sample' else None,'stop')
    episodes = []
    for item in plan:
        oracle,_,_,_ = initial_state(task,item['scenario'])
        def policy(messages):
            action=corrective_action(oracle)
            oracle.step(action)
            return native(json.dumps(action),messages,item['mode'])
        try:
            sampling={**cfg['sampling'],'temperature':item['temperature']}
            e=run_continuation(task,item['scenario'],policy,sampling,0,item['seed'],item['rollout_index'])
        finally: oracle.close()
        e.update(model_branch='weighted_sft',evaluation_mode=item['mode'])
        episodes.append(e)
    context=build_context(task,'format_repair',allowed_split='development')
    generation=asdict(native(context['chosen'],context['messages'],'greedy'))
    seed=next(p['seed'] for p in plan if p['scenario']=='format_repair' and p['mode']=='greedy')
    quality=[{'task_id':task.id,'scenario':'format_repair','seed':seed,
              'generations':[generation,copy.deepcopy(generation)]}]
    spec={'label':'weighted_sft','group_cursor':0,'policy':{'checkpoint':'initial','version':0,'updates':0,
          'adapter_sha256':cfg['adapter_sha256']}}
    data={'spec':spec,'config_sha256':file_hash(tmp_path/'config.json'),'episodes':episodes,
          'quality':quality,'worker_proof':proof,'summary':summarize_evaluation(episodes)}
    return ev,tok,spec,data


def test_formal_evaluation_replays_native_complete_and_recovery_with_calendar(tmp_path,monkeypatch):
    ev,tok,spec,data=evaluation_fixture(tmp_path,monkeypatch)
    result=ev.verify_evaluation(tmp_path,'weighted_sft',spec,tok,data=data)
    assert result['passed'] and result['episodes_replayed']==4
    assert all(x['result']['calendar_consistent_success'] for x in result['calendar'])
    assert result['summary']['sample']['recovery']['successes']==1


@pytest.mark.parametrize('tamper',['seed','duplicate','quality','worker','summary'])
def test_formal_evaluation_detects_identity_and_quality_tampering(tmp_path,monkeypatch,tamper):
    ev,tok,spec,data=evaluation_fixture(tmp_path,monkeypatch)
    if tamper=='seed': data['episodes'][0]['seed']+=1
    if tamper=='duplicate': data['episodes'].append(copy.deepcopy(data['episodes'][0]))
    if tamper=='quality': data['quality'][0]['generations'][0]['response_token_ids'][0]+=1
    if tamper=='worker': data['worker_proof'][0]['tensors'].pop(next(iter(data['worker_proof'][0]['tensors'])))
    if tamper=='summary': data['summary']['sample']['recovery']['successes']+=1
    with pytest.raises(ValueError): ev.verify_evaluation(tmp_path,'weighted_sft',spec,tok,data=data)


def test_formal_driver_evaluates_W_and_each_group_cursor_once(tmp_path,monkeypatch):
    import run_v2_grpo as driver
    cfg=copy.deepcopy(CFG)
    cfg.update(run_id='test-formal-driver',max_groups=4,max_updates=4,group_evaluation_cursors=[2,4])
    path=tmp_path/'configs/formal.json'
    atomic_json(path,cfg)
    atomic_json(tmp_path/'grpo_release_manifest.json',{'files':{'configs/formal.json':file_hash(path)}})
    output=tmp_path/'artifacts'/cfg['run_id']
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/formal.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    calls=[]
    def stage(command,**kwargs):
        name=kwargs['log_path'].name.split('-attempt')[0]
        calls.append(name)
        kwargs['log_path'].write_text('100%\r',encoding='utf8')
        if name=='prepare': atomic_json(output/'input_binding.json',{})
        elif name.startswith('collect-'):
            i=int(name[-4:]);atomic_json(output/'groups'/f'group-{i:04}.json',{'index':i})
        elif name.startswith('update-'):
            i=int(name[-4:]);before=read(output/'policies'/f'before-group-{i:04}.json')
            atomic_json(output/'transitions'/f'group-{i:04}.json',{'index':i,'before':before,'after':before,
                'updated':False,'group_sha256':file_hash(output/'groups'/f'group-{i:04}.json'),
                'config_sha256':file_hash(output/'config.json'),'metrics':{'skip_reason':'zero_variance'}})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    driver.run(path)
    assert calls.index('evaluate-weighted_sft')<calls.index('collect-0000')
    assert calls.index('update-0001')<calls.index('evaluate-grpo_g0002')<calls.index('collect-0002')
    assert calls.count('evaluate-grpo_g0002')==calls.count('evaluate-grpo_g0004')==1
    assert calls[-1]=='audit'
    assert read(output/'runner_state.json')['next_group']==4


def test_eval_training_really_uses_activation_checkpointing(monkeypatch):
    import torch
    import torch.utils.checkpoint as checkpoint_module
    from test_grpo_v2 import tiny_model
    from grpo_worker import configure_training_memory
    from apt_v2.grpo_optimizer import response_scores
    calls=[]
    original=checkpoint_module.checkpoint
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(checkpoint_module,'checkpoint',counted)
    model=tiny_model().eval()
    configure_training_memory(model,{'explicit_eval_checkpointing':True})
    response_scores(model,[1,2,3],[4,2],True).sum().backward()
    assert not model.training
    assert calls, 'eval-mode gradient updates must actually checkpoint decoder layers'


def test_eval_checkpointing_preserves_scores_and_all_lora_gradients():
    import torch
    from test_grpo_v2 import tiny_model
    from grpo_worker import configure_training_memory
    from apt_v2.grpo_optimizer import response_scores
    plain=tiny_model().eval()
    checkpointed=copy.deepcopy(plain)
    contract=configure_training_memory(checkpointed,{'explicit_eval_checkpointing':True})
    assert contract['model_eval'] and contract['explicit_eval_checkpoint_layers']==1
    for model in [plain,checkpointed]:
        value=response_scores(model,[1,2,3,4,5],[6,7,2],True)
        value.sum().backward()
    torch.testing.assert_close(response_scores(plain,[1,2,3,4,5],[6,7,2]),
                               response_scores(checkpointed,[1,2,3,4,5],[6,7,2]),rtol=0,atol=0)
    for (name,a),(other,b) in zip(plain.named_parameters(),checkpointed.named_parameters()):
        assert name==other
        if a.requires_grad:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        else:
            assert a.grad is None and b.grad is None


def test_qwen_sdpa_eval_checkpointing_preserves_nonzero_A_B_gradients():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    from grpo_worker import configure_training_memory
    from apt_v2.grpo_optimizer import response_scores
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.,_attn_implementation='sdpa')
    plain=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
                         target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for n,p in plain.named_parameters():
            if 'lora_B' in n: p.add_(.02)
    checked=copy.deepcopy(plain)
    contract=configure_training_memory(checked,{'explicit_eval_checkpointing':True})
    assert contract['explicit_eval_checkpoint_layers']==2
    values=[]
    for model in [plain,checked]:
        value=response_scores(model,[1,2,3,4,5],[6,7,2],True)
        values.append(value.detach())
        value.sum().backward()
    torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    assert all(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in plain.named_parameters() if p.requires_grad)
    for (n,a),(other,b) in zip(plain.named_parameters(),checked.named_parameters()):
        assert n==other
        if a.requires_grad: torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        else: assert a.grad is None and b.grad is None


def test_formal_driver_resumes_due_evaluation_before_more_training(tmp_path,monkeypatch):
    import run_v2_grpo as driver
    cfg=copy.deepcopy(CFG)
    cfg.update(run_id='test-formal-resume',max_groups=4,max_updates=4,group_evaluation_cursors=[2,4])
    path=tmp_path/'configs/formal.json'
    atomic_json(path,cfg)
    atomic_json(tmp_path/'grpo_release_manifest.json',{'files':{'configs/formal.json':file_hash(path)}})
    output=tmp_path/'artifacts'/cfg['run_id']
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/formal.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    calls=[]; fail=[True]
    def stage(command,**kwargs):
        name=kwargs['log_path'].name.split('-attempt')[0]
        calls.append(name)
        kwargs['log_path'].write_text('100%\r',encoding='utf8')
        if name=='evaluate-grpo_g0002' and fail[0]:
            fail[0]=False;return 1
        if name=='prepare': atomic_json(output/'input_binding.json',{})
        elif name.startswith('collect-'):
            i=int(name[-4:]);atomic_json(output/'groups'/f'group-{i:04}.json',{'index':i})
        elif name.startswith('update-'):
            i=int(name[-4:]);before=read(output/'policies'/f'before-group-{i:04}.json')
            atomic_json(output/'transitions'/f'group-{i:04}.json',{'index':i,'before':before,'after':before,
                'updated':False,'group_sha256':file_hash(output/'groups'/f'group-{i:04}.json'),
                'config_sha256':file_hash(output/'config.json'),'metrics':{'skip_reason':'zero_variance'}})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    with pytest.raises(RuntimeError): driver.run(path)
    assert read(output/'runner_state.json')['next_group']==2
    first_count=len(calls)
    driver.run(path,resume=True)
    retry=calls[first_count:]
    assert retry.index('evaluate-grpo_g0002')<retry.index('collect-0002')
    assert calls.count('collect-0000')==calls.count('collect-0001')==1


def test_formal_audit_recomputes_paired_tables_and_ranking(tmp_path,monkeypatch):
    import audit_v2_grpo_formal as formal
    import audit_v2_grpo as train
    import transformers
    from apt_v2.grpo_evaluation import summarize_evaluation
    ev,tok,base_spec,data=evaluation_fixture(tmp_path,monkeypatch)
    cfg=copy.deepcopy(CFG)
    cfg.update(max_groups=4,max_updates=4,group_evaluation_cursors=[2,4])
    cfg['sampling'].update(max_new_tokens=10000,max_model_len=100000)
    cfg['adapter_sha256']=base_spec['policy']['adapter_sha256']
    # Keep the already frozen native evaluation fixture's config identity.
    cfg=read(tmp_path/'config.json')|{'max_groups':4,'max_updates':4,'group_evaluation_cursors':[2,4]}
    atomic_json(tmp_path/'config.json',cfg)
    monkeypatch.setattr(formal,'verify_inputs',lambda *a,**k:cfg)
    monkeypatch.setattr(ev,'verify_inputs',lambda *a,**k:cfg)
    monkeypatch.setattr(train,'audit',lambda *a,**k:{'passed':True,'actual_updates':0})
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    before=base_spec['policy']
    for i in range(4):
        group={'spec':{'kind':'complete' if i%2==0 else 'recovery','task_id':'train-000','scenario':None},
               'episodes':[data['episodes'][0]],'credit':{'mean':.93,'population_std':0,'eligible':False},
               'actions':7,'response_tokens':10}
        atomic_json(tmp_path/'groups'/f'group-{i:04}.json',group)
        atomic_json(tmp_path/'transitions'/f'group-{i:04}.json',{'index':i,'before':before,'after':before,
            'updated':False,'group_sha256':file_hash(tmp_path/'groups'/f'group-{i:04}.json'),
            'config_sha256':file_hash(tmp_path/'config.json'),'metrics':{'skip_reason':'zero_variance'}})
    for label,cursor in [('weighted_sft',0),('grpo_g0002',2),('grpo_g0004',4)]:
        spec={'label':label,'group_cursor':cursor,'policy':before}
        branch=copy.deepcopy(data)
        branch['spec']=spec;branch['config_sha256']=file_hash(tmp_path/'config.json')
        for e in branch['episodes']: e['model_branch']=label
        branch['summary']=summarize_evaluation(branch['episodes'])
        atomic_json(tmp_path/'evaluations'/(label+'-policy.json'),spec)
        atomic_json(tmp_path/'evaluations'/(label+'.json'),branch)
    result=formal.audit_formal(tmp_path,local=True)
    assert result['passed']
    assert result['candidate_ranking']==['grpo_g0002','grpo_g0004']
    assert (tmp_path/'local_training_curve.csv').exists() and (tmp_path/'local_evaluation_curve.csv').exists()
    compared=read(tmp_path/'local_comparison_summary.json')
    assert compared['paired_to_weighted_sft']['grpo_g0002']['sample']['complete']['delta_successes']==0
