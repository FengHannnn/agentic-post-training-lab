import sys,json,copy,random,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
from apt_v2.dpo_sft_start import optimize_segment,verify_history,candidate_decision,baseline_ready,select_candidate
from common import read,write

def test_real_lora_resume_preserves_adam_rng_history_and_reference(tmp_path):
    import torch
    from test_matched_dpo_v2 import tiny_model
    from apt_v2.dpo_training import reference_scores
    from apt_v2.dpo_recovery_training import training_batches
    a=tiny_model()
    for m in a.modules():
        if isinstance(m,torch.nn.Dropout):m.p=.2
    b=copy.deepcopy(a);encoded=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8,9,10,11,12,13)]
    ref=[reference_scores(a,e) for e in encoded];frozen=copy.deepcopy(ref);base={n:p.clone() for n,p in a.named_parameters() if not p.requires_grad}
    oa=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=5e-7,weight_decay=0);ob=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=5e-7,weight_decay=0)
    torch.manual_seed(73);full=optimize_segment(a,encoded,ref,oa,2,51,.1,1,4)
    torch.manual_seed(73);part=optimize_segment(b,encoded,ref,ob,2,51,.1,1,1)
    b.save_pretrained(tmp_path/'adapter',safe_serialization=True)
    p=tmp_path/'resume.pt';torch.save({'optimizer':ob.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'history':part},p)
    saved=torch.load(p,weights_only=False)
    from transformers import Qwen2ForCausalLM
    from peft import PeftModel
    torch.manual_seed(27);c=PeftModel.from_pretrained(Qwen2ForCausalLM(copy.deepcopy(b.config)),tmp_path/'adapter',is_trainable=True).eval()
    for m in c.modules():
        if isinstance(m,torch.nn.Dropout):m.p=.2
    oc=torch.optim.AdamW([q for q in c.parameters() if q.requires_grad],lr=5e-7,weight_decay=0);oc.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
    resumed=optimize_segment(c,encoded,ref,oc,2,51,.1,1,4,saved['history'])
    assert resumed==full and ref==frozen
    assert all(torch.equal(q,dict(c.named_parameters())[n]) for n,q in a.named_parameters())
    assert all(torch.equal(q,base[n]) and q.grad is None for n,q in a.named_parameters() if n in base)
    assert all(int(v['step'])==4 for v in oc.state.values())
    assert full[-1]['pairs_seen']==8 and [r['batch_indices'] for r in full]==training_batches(8,2,51)
    assert any(torch.count_nonzero(q.grad).item()>0 for q in c.parameters() if q.requires_grad)
    with pytest.raises(ValueError,match='increasing'):optimize_segment(c,encoded,ref,oc,2,51,.1,1,4,resumed)

def _summary(full=16,recovery=60,greedy=4,gr=21):
    return {'sample':{'complete':{'successes':full,'episodes':16},'recovery':{'successes':recovery,'episodes':84}},'greedy':{'complete':{'successes':greedy,'episodes':4},'recovery':{'successes':gr,'episodes':28}}}

def test_gate_practical_tolerances_early_stop_and_ranking():
    cfg=read(ROOT/'configs/dpo-sft-start-short-v1.json');a=_summary()
    assert baseline_ready(a,cfg) and not baseline_ready(_summary(15),cfg)
    assert candidate_decision(a,_summary(15,56),cfg)['passed']
    for c in (_summary(14,60),_summary(15,55),_summary(16,60,3),_summary(16,60,4,20)):
        assert not candidate_decision(a,c,cfg)['passed']
    assert candidate_decision(a,_summary(12),cfg)['severe_regression_stop']
    values={'weighted_sft':a,'dpo_0004':_summary(15,56),'dpo_0012':_summary(16,60)};decisions={b:candidate_decision(a,c,cfg) for b,c in values.items() if b!='weighted_sft'}
    assert select_candidate(decisions,values,cfg)=='dpo_0004'
    values={'weighted_sft':a,'dpo_0004':_summary(14,60),'dpo_0012':_summary(14,59)};decisions={b:candidate_decision(a,c,cfg) for b,c in values.items() if b!='weighted_sft'}
    assert select_candidate(decisions,values,cfg)=='dpo_0004'

def test_partial_pass_ancestry_and_adam_audit_rejects_tampering():
    from apt_v2.dpo_recovery_training import training_batches
    cfg=read(ROOT/'configs/dpo-sft-start-short-v1.json');batches=training_batches(169,4,cfg['seed']);history=[{'step':i+1,'pairs_seen':4*(i+1),'batch_indices':batches[i],'loss':.69,'grad_norm':1.} for i in range(4)]
    ancestry={'algorithm':cfg['algorithm'],'source_sft_updates':513,'source_dpo_updates':11,'total_sft_updates':513,'total_dpo_updates':15,'epochs_completed':0}
    complete=dict(ancestry,steps=4,cursor=16);train=dict(ancestry,optimizer_updates=4,pair_exposures=16,history=history,optimizer_start='fresh_adam_at_step_zero_then_exact_resume',frozen_parameter_gradients=0,adapter_tensors_changed=1);state={'steps':4,'cursor':16,'history':history,'optimizer':{'state':{0:{'step':4}}}}
    verify_history(cfg,complete,train,state)
    bad=copy.deepcopy(complete);bad['source_sft_updates']=255
    with pytest.raises(ValueError,match='ancestry'):verify_history(cfg,bad,train,state)
    bad=copy.deepcopy(train);bad['history'][0]['batch_indices']=batches[1]
    with pytest.raises(ValueError,match='schedule'):verify_history(cfg,complete,bad,state)
    bad=copy.deepcopy(state);bad['optimizer']['state'][0]['step']=3
    with pytest.raises(ValueError,match='Adam'):verify_history(cfg,complete,train,bad)

@pytest.mark.parametrize('passing,severe,baseline,expected',[(4,None,True,[4]),(12,None,True,[4,12]),(None,None,True,[4,12,24]),(None,4,True,[4]),(None,None,False,[])])
def test_runner_only_trains_until_predeclared_gate(tmp_path,monkeypatch,passing,severe,baseline,expected):
    import run_v2_dpo_sft_start as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();write(tmp_path/'configs/dpo-sft-start-short-v1.json',{'run_id':'short','max_wall_seconds':14400,'max_steps':24,'checkpoint_steps':[4,12,24]});write(tmp_path/'dpo_sft_start_release_manifest.json',{'files':{}});seen=[]
    def fake(command,**kwargs):
        out=Path(command[command.index('--output')+1]);script=Path(command[2]).name
        if script=='train_v2_dpo_sft_start.py':
            s=int(command[-1]);seen.append(s);(out/'training').mkdir(exist_ok=True);(out/'training'/f'segment-{s:04}-steps.jsonl').write_text(json.dumps({'step':s})+'\n')
        if script=='audit_v2_dpo_sft_start.py':
            if '--branch' in command:
                b=command[-1];write(out/(b+'-decision.json'),{'training_allowed':baseline} if b=='weighted_sft' else {'passed':int(b[-4:])==passing,'severe_regression_stop':int(b[-4:])==severe})
            else:write(out/'selection.json',{'stop_reason':'baseline_changed' if not baseline else 'first_passing_checkpoint' if passing else 'severe_regression' if severe else 'step_cap_reached'})
        return 0
    monkeypatch.setattr(runner,'run_logged',fake);runner.run();assert seen==expected
    status=read(tmp_path/'artifacts/short/run_status.json');assert status['optimizer_updates']==(expected[-1] if expected else 0) and status['error'] is None
    with pytest.raises(FileExistsError):runner.run()

def test_runner_failure_keeps_partial_absolute_update_count_and_log(tmp_path,monkeypatch):
    import run_v2_dpo_sft_start as runner
    import zipfile
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir();write(tmp_path/'configs/dpo-sft-start-short-v1.json',{'run_id':'failure','max_wall_seconds':14400,'max_steps':24,'checkpoint_steps':[4,12,24]});write(tmp_path/'dpo_sft_start_release_manifest.json',{'files':{}})
    (tmp_path/'scripts/dpo_sft_start_common.py').write_text("from pathlib import Path\nimport sys\np=Path(sys.argv[-1])/'training';p.mkdir();(p/'segment-0004-steps.jsonl').write_text('{\"step\": 2}\\n')\nprint('partial evidence',flush=True)\nraise RuntimeError('stopped')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert read(tmp_path/'artifacts/failure/run_status.json')['optimizer_updates']==2
    with zipfile.ZipFile(tmp_path/'failure-review.zip') as z:assert 'failure/prepare.log' in z.namelist()

from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native

def test_four_branch_complete_recovery_audit_and_tamper(tmp_path,monkeypatch):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_sft_start_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_sft_start import audit_results
    cfg={'evaluation_branches':['weighted_sft','dpo_0004'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
        'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33,'samples_per_context':1,'temperature':.8}}
    tok=CharTokenizer();ts=tasks();tm={t.id:t for t in ts};cs=contexts();allfull=[];allrs=[];allaa=[]
    for b in cfg['evaluation_branches']:
        full=[];rs=[];aa=[]
        for task in ts:
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0)
                def policy(messages):
                    env=Environment(task)
                    try:
                        for m in messages[2:]:
                            if m['role']=='assistant':env.step(json.loads(m['content']))
                        return native(tok,messages,json.dumps(corrective_action(env)),nm)
                    finally:env.close()
                e=run_episode(task,policy);e.update(model_branch=b,mode=mode,rollout_index=0,sample_seed=seed,temperature=0 if mode=='greedy' else .8)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                full.append(e)
        for c in cs:
            task=tm[c['task_id']]
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0,c)
                e=run_recovery(task,c,oracle(task,c,tok,nm));temperature=0 if mode=='greedy' else .8
                e.update(model_branch=b,mode=mode,sample_index=0,sample_seed=seed,temperature=temperature)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                rs.append(e);g=e['turns'][0]['generation'];actual,d=diagnose_action(task,c,g['text'])
                aa.append({'model_branch':b,'mode':mode,'task_id':task.id,'family_id':task.family_id,'split':'development','context_id':c['context_id'],
                    'scenario':c['scenario'],'sample_index':0,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d})
        jsonl(tmp_path/(b+'-complete-episodes.jsonl'),full);jsonl(tmp_path/(b+'-recovery-episodes.jsonl'),rs);jsonl(tmp_path/(b+'-first-actions.jsonl'),aa)
        allfull.extend(full);allrs.extend(rs);allaa.extend(aa)
    summary=result_summary(allfull,allrs,allaa,cfg);(tmp_path/'comparison_summary.json').write_text(json.dumps(summary))
    result,calendar,evidence=audit_results(tmp_path,cfg,tok)
    assert evidence['complete_episodes_replayed']==16 and evidence['recovery_episodes_replayed']==112
    assert result['paired']['weighted_sft_to_dpo_0004']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    # Exercise the actual single-branch audits and final earliest-pass selection.
    import audit_v2_dpo_sft_start as auditor
    import transformers
    cfg['gradient_accumulation']=4;cfg['checkpoint_steps']=[4];cfg['protocol']={'quality_contexts_per_model':2,'quality_repeats':2};cfg['selection_rule']=copy.deepcopy(read(ROOT/'configs/dpo-sft-start-short-v1.json')['selection_rule']);cfg['selection_rule'].update(minimum_sample_complete_successes=4,baseline_required_sample_complete_successes=4)
    monkeypatch.setattr(auditor,'verify_inputs',lambda out:cfg);monkeypatch.setattr(auditor,'verify_candidate_files',lambda *a,**k:{'step':4});monkeypatch.setattr(auditor,'verify_reference_cache',lambda *a:None);monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    (tmp_path/'training/checkpoint-0004').mkdir(parents=True)
    for j,b in enumerate(cfg['evaluation_branches']):
        write(tmp_path/(b+'-execution.json'),{'pid':100+j,'branch':b,'one_adapter_in_this_process':True,'max_loras':1,'lora_name':'fixed_adapter','lora_int_id':1,'training_enabled':False,'optimizer_updates':0})
        actions=[a for a in allaa if a['model_branch']==b and a['mode']=='greedy'];quality=[]
        for c in cs[:2]:
            a=next(a for a in actions if a['context_id']==c['context_id']);quality.append({'model_branch':b,'task_id':a['task_id'],'context_id':c['context_id'],'scenario':c['scenario'],'split':'development','seed':a['sample_seed'],'generations':[a['generation'],copy.deepcopy(a['generation'])]})
        jsonl(tmp_path/(b+'-quality.jsonl'),quality);decision=auditor.audit_branch(tmp_path,b)
        assert decision['training_allowed'] if b=='weighted_sft' else decision['passed']
    (tmp_path/'comparison_summary.json').unlink();auditor.audit(tmp_path)
    selected=read(tmp_path/'selection.json');assert selected['selected_candidate']=='dpo_0004' and selected['performance_target_met'] and selected['stop_reason']=='first_passing_checkpoint' and selected['optimizer_updates']==4
    assert read(tmp_path/'audit.json')['passed']
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'dpo_0004-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)



def test_evaluator_uses_one_adapter_per_process_and_constant_lora_position(tmp_path,monkeypatch):
    import evaluate_v2_dpo_sft_start as evaluator
    import transformers
    import audit_v2_dpo_sft_start as candidate_auditor
    monkeypatch.setattr(candidate_auditor,'verify_candidate_files',lambda *a,**k:None)
    from apt_v2.environment import Environment
    from test_dpo_v2 import CharTokenizer
    from test_hard_recovery_independent_v2 import native
    from apt_v2.dpo import corrective_action,action_from_text
    from apt_v2.tasks import make_tasks
    from apt_v2.dpo_recovery import build_context
    tok=CharTokenizer();task=next(t for t in make_tasks() if t.split=='development');context=build_context(task,'format_repair',allowed_split='development');captured={}
    class Engine:
        def __init__(self,**kwargs):captured['engine']=kwargs
    def Request(name,index,path):captured['request']=(name,index,path);return (name,index,path)
    fake=types.ModuleType('vllm');fake.LLM=Engine;request=types.ModuleType('vllm.lora.request');request.LoRARequest=Request
    monkeypatch.setitem(sys.modules,'vllm',fake);monkeypatch.setitem(sys.modules,'vllm.lora.request',request)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(evaluator,'shutdown',lambda e:None);monkeypatch.setattr(evaluator,'tasks',lambda:[task]);monkeypatch.setattr(evaluator,'contexts',lambda:[context])
    cfg={'protocol':{'quality_contexts_per_model':1,'quality_repeats':2},'evaluation_branches':['dpo_0004'],'eval':{'seed':31,'max_model_len':1000000,'max_new_tokens':100000,'gpu_memory_utilization':.6},'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33,'samples_per_context':1,'temperature':.8}}
    monkeypatch.setattr(evaluator,'verify_inputs',lambda out:cfg)
    def Policy(engine,tokenizer,lora,config,mode,seed,temperature):
        assert lora[0]=='fixed_adapter' and lora[1]==1
        def call(messages):
            env=Environment(task)
            try:
                for m in messages[2:]:
                    if m['role']=='assistant':env.step(action_from_text(m['content']))
                return native(tok,messages,json.dumps(corrective_action(env)),'greedy' if mode=='greedy' else 'dpo_sample')
            finally:env.close()
        return call
    monkeypatch.setattr(evaluator,'NativePolicy',Policy);evaluator.evaluate(tmp_path,'dpo_0004')
    assert captured['engine']['max_loras']==1 and captured['engine']['max_num_seqs']==1 and captured['engine']['enable_prefix_caching'] is False
    assert captured['request'][1]==1 and len((tmp_path/'dpo_0004-complete-episodes.jsonl').read_text().splitlines())==2
    assert json.loads((tmp_path/'dpo_0004-execution.json').read_text())['optimizer_updates']==0

    import audit_v2_dpo_sft_start as auditor
    monkeypatch.setattr(auditor,'contexts',lambda:[context]);monkeypatch.setattr(auditor,'tasks',lambda:[task]);result=auditor.audit_quality(tmp_path,cfg,tok)
    assert result['passed'] and result['branches']['dpo_0004']['generated_actions']==2
    p=tmp_path/'dpo_0004-quality.jsonl';data=json.loads(p.read_text());data['generations'][0]['prompt_token_ids'][0]+=1;p.write_text(json.dumps(data)+'\n')
    with pytest.raises(ValueError,match='prompt'):auditor.audit_quality(tmp_path,cfg,tok)

