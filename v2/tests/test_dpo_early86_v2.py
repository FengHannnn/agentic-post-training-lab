import sys,json,copy,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native

def test_four_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_early86_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_early86 import audit_results
    cfg={'evaluation_branches':['weighted_sft','dpo_0086'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
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
    assert result['paired']['weighted_sft_to_dpo_0086']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'dpo_0086-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)


def test_confirmation_runner_failure_has_zero_updates_and_protects_old_output(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_early86 as runner
    from common import write,read
    import zipfile
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir();write(tmp_path/'configs/dpo-early86-vs-weighted-sft-v1.json',{'run_id':'inference-only','max_wall_seconds':14400,'evaluation_branches':['chosen_control']});write(tmp_path/'dpo_early86_release_manifest.json',{'files':{}})
    (tmp_path/'scripts/dpo_early86_common.py').write_text("print('inference prepare');raise RuntimeError('changed runtime')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    status=read(tmp_path/'artifacts/inference-only/run_status.json');assert status['optimizer_updates']==0 and status['training_enabled'] is False
    with zipfile.ZipFile(tmp_path/'inference-only-review.zip') as z:assert 'inference-only/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()

def test_evaluator_uses_one_adapter_per_process_and_constant_lora_position(tmp_path,monkeypatch):
    import evaluate_v2_dpo_early86 as evaluator
    import transformers
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
    cfg={'protocol':{'quality_contexts_per_model':1,'quality_repeats':2},'evaluation_branches':['dpo_0086'],'eval':{'seed':31,'max_model_len':1000000,'max_new_tokens':100000,'gpu_memory_utilization':.6},'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33,'samples_per_context':1,'temperature':.8}}
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
    monkeypatch.setattr(evaluator,'NativePolicy',Policy);evaluator.evaluate(tmp_path,'dpo_0086')
    assert captured['engine']['max_loras']==1 and captured['engine']['max_num_seqs']==1 and captured['engine']['enable_prefix_caching'] is False
    assert captured['request'][1]==1 and len((tmp_path/'dpo_0086-complete-episodes.jsonl').read_text().splitlines())==2
    assert json.loads((tmp_path/'dpo_0086-execution.json').read_text())['optimizer_updates']==0

    import audit_v2_dpo_early86 as auditor
    monkeypatch.setattr(auditor,'contexts',lambda:[context]);monkeypatch.setattr(auditor,'tasks',lambda:[task]);result=auditor.audit_quality(tmp_path,cfg,tok)
    assert result['passed'] and result['branches']['dpo_0086']['generated_actions']==2
    p=tmp_path/'dpo_0086-quality.jsonl';data=json.loads(p.read_text());data['generations'][0]['prompt_token_ids'][0]+=1;p.write_text(json.dumps(data)+'\n')
    with pytest.raises(ValueError,match='prompt'):auditor.audit_quality(tmp_path,cfg,tok)

def test_performance_report_uses_complete_task_as_primary_and_reports_real_gap():
    from audit_v2_dpo_early86 import performance_comparison
    def results(full,recovery,n,m):return {'complete':{'episodes':n,'successes':full},'recovery':{'episodes':m,'successes':recovery}}
    pairs={mode:{kind:{'improved':0,'regressed':3} for kind in ('complete','recovery')} for mode in ('greedy','sample')}
    summary={'branches':{'weighted_sft':{'greedy':results(4,21,4,28),'sample':results(16,60,16,84)},'dpo_0086':{'greedy':results(4,21,4,28),'sample':results(13,57,16,84)}},'paired':{'weighted_sft_to_dpo_0086':pairs}}
    report=performance_comparison(summary)
    assert report['primary']=='sample.complete' and report['results']['sample']['complete']['DPO_minus_SFT_percentage_points']==-18.75 and report['performance_acceptance']=='awaiting_user_review_of_similarity' and report['optimizer_updates']==0

def test_source_contract_rejects_wrong_candidate_and_wrong_86_step_metadata(tmp_path,monkeypatch):
    import dpo_early86_common as common
    from common import write,file_hash
    cfg=json.loads((ROOT/'configs/dpo-early86-vs-weighted-sft-v1.json').read_text());old=json.loads((ROOT/'configs/dpo-regression-checkpoints-v1.json').read_text());source=tmp_path/'artifacts'/cfg['source_run'];source.mkdir(parents=True)
    origin=tmp_path/'artifacts'/cfg['origin_run'];origin.mkdir(parents=True);write(origin/'config.json',{});write(origin/'input_binding.json',{});cfg.update(origin_config_sha256=file_hash(origin/'config.json'),origin_binding_sha256=file_hash(origin/'input_binding.json'))
    for b,spec in cfg['frozen_models'].items():
        p=tmp_path/'artifacts'/spec['run']/spec['checkpoint'];p.mkdir(parents=True,exist_ok=True)
        meta={'algorithm':'hard_recovery_weighted_dpo','steps':86,'epochs_completed':2,'cursor':338,'total_sft_updates':255,'total_dpo_updates':97} if b=='dpo_0086' else {'steps':258,'total_sft_updates':513,'total_dpo_updates':11}
        write(p/'complete.json',meta);spec['complete_sha256']=file_hash(p/'complete.json');old['checkpoints'][b]['complete_sha256']=spec['complete_sha256']
    write(source/'config.json',old);write(source/'audit.json',{'passed':True});write(source/'run_status.json',{'error':None});cfg.update(source_config_sha256=file_hash(source/'config.json'),source_audit_sha256=file_hash(source/'audit.json'),source_run_status_sha256=file_hash(source/'run_status.json'));monkeypatch.setattr(common,'ROOT',tmp_path);common.verify_source(cfg)
    changed=copy.deepcopy(cfg);changed['evaluation_branches'].append('dpo_0043')
    with pytest.raises(ValueError,match='fixed SFT'):common.verify_source(changed)
    p=common.checkpoint_path(cfg['frozen_models']['dpo_0086']);meta=json.loads((p/'complete.json').read_text());meta['steps']=129;(p/'complete.json').unlink();write(p/'complete.json',meta);cfg['frozen_models']['dpo_0086']['complete_sha256']=file_hash(p/'complete.json');old['checkpoints']['dpo_0086']['complete_sha256']=file_hash(p/'complete.json');(source/'config.json').unlink();write(source/'config.json',old);cfg['source_config_sha256']=file_hash(source/'config.json')
    with pytest.raises(ValueError,match='exact early 86-step'):common.verify_source(cfg)
