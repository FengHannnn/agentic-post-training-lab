import copy,json,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,file_hash
from apt_v2.grpo_hard_preflight import tasks,context,plan,expert,frozen_inputs,summarize,LEVELS
from apt_v2.grpo_curriculum import TARGET_SCENARIOS,build_context as original_context
from apt_v2.grpo import initial_state,run_continuation,digest
from apt_v2.dpo import corrective_action
from grpo_hard_common import validate_config,W_SHA
from test_grpo_confirmation_v2 import NativeTokenizer
CFG=read(ROOT/'configs/grpo-hard-recovery-preflight-v1.json')


def test_all_24_contexts_preserve_budget_and_original_compound_bytes():
    proofs=[]
    for t in tasks():
        for s in TARGET_SCENARIOS:
            old=original_context(t,s);compound=context(t,s+'__compound');simple=context(t,s+'__simplified')
            assert compound==old and len(simple['records'])==len(old['records'])
            pair=[expert(t,s+'__'+level) for level in LEVELS];proofs+=pair
            assert pair[0]['remaining_turns']==pair[1]['remaining_turns']
            if s=='missing_alias_pair':assert pair[0]['initial_verified_metrics']==['prior_net_cents'] and pair[1]['initial_verified_metrics']==[]
            elif s=='schema_query_repeat':assert simple['records'][-2]['row']['action']['tool']=='inspect_schema'
            else:
                assert simple['records'][-1]['row']['observation']==compound['records'][-1]['row']['observation']
                assert pair[0]['initial_verified_metrics']==pair[1]['initial_verified_metrics']==['current_net_cents','prior_net_cents']
    assert len(proofs)==24 and all(p['passed'] for p in proofs) and max(p['total_turns'] for p in proofs)==11
    with pytest.raises(ValueError):context(tasks()[0],'missing_alias_pair__simplified','development')


def test_frozen_96_plan_pairs_seeds_and_balances_old_new_fault():
    p=plan(CFG);assert len(p)==96 and len({(x['task_id'],x['scenario'],x['rollout_index']) for x in p})==96
    assert [t.fault_mode for t in tasks()]==['none','none','transient_query','transient_query']
    assert len({x['seed'] for x in p})==48 and Counter(x['level'] for x in p)=={'simplified':48,'compound':48}
    for start in range(0,96,8):
        a,b=p[start:start+4],p[start+4:start+8]
        assert [x['seed'] for x in a]==[x['seed'] for x in b]
        assert a[0]['level']!=b[0]['level']
    assert len(frozen_inputs(CFG)['expert_budget_proof.json'])==24


@pytest.mark.parametrize('key,value',[('max_wall_seconds',7201),('adapter_sha256','wrong'),('k',8),('new_training_updates',1)])
def test_config_rejects_budget_model_or_training_changes(key,value):
    cfg=copy.deepcopy(CFG);cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def capture(g,seed):
    rows=[]
    for i,(token,q) in enumerate(zip(g['response_token_ids'],g['behavior_token_log_probs'])):
        rows.append({'step':i,'token_id':token,'raw_selected_logit':0.,'raw_logsumexp_fp64':-q,'raw_logprob_fp64':q,
            'raw_logprob':q,'native_processed_logprob':q,'normalization_error':0.,'fp64_precision_difference':0.,
            'full_logits_unchanged':True,'metadata_neutral':True,'raw_logits_sha256':'0'*64,'vocab_size':151936})
    return {'method':'same_forward_sampler_v2','setup':{'seed':seed,'policy_sha256':W_SHA,'token_limit':16,'sampling':CFG['sampling']},
        'installation':{'installed':True,'method':'same_forward_sampler_v2'},'request':{'seed':seed,'lora_id':1,'prompt_token_ids':g['prompt_token_ids']},'rows':rows}


def collection_fixture(tmp_path,monkeypatch,fail_once=False):
    import grpo_hard_worker as worker
    import grpo_hard_common as common
    import grpo_vllm
    import transformers
    cfg=copy.deepcopy(CFG);tok=NativeTokenizer();task_map={t.id:t for t in tasks()};calls=Counter();failed=False;environments=[]
    atomic_json(tmp_path/'config.json',cfg);atomic_json(tmp_path/'execution_environment.json',{'gpu':'L40S'});atomic_json(tmp_path/'driver_identity.json',{'driver':'fixed'})
    atomic_json(tmp_path/'initial/adapter/fixture.json',{})
    monkeypatch.setattr(worker,'verify_inputs',lambda *a,**k:cfg);monkeypatch.setattr(common,'verify_inputs',lambda *a,**k:cfg)
    monkeypatch.setattr(worker,'runtime_identity',lambda:{'gpu':'L40S'});monkeypatch.setattr(worker,'driver_identity',lambda:{'driver':'fixed'})
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    tensors={str(i):{'sha256':str(i)} for i in range(112)};monkeypatch.setattr(grpo_vllm,'projected_checkpoint_tensors',lambda p:tensors)
    class Engine:
        def __init__(self):self.queue=[p for p in plan(cfg) if not (tmp_path/'episodes'/f"{p['index']:04}.json").exists()];self.proof=None
        def collective_rpc(self,*a,**k):return self.proof
    class Policy:
        def __init__(self,engine,tok,lora,sampling,seed,on_action=None):
            self.engine=engine;self.item=engine.queue.pop(0);assert seed==self.item['seed'];self.seed=seed;self.sampling=sampling;self.on_action=on_action;self.calls=0;self.first=None
            calls[self.item['index']]+=1;self.env,_,_,_=initial_state(task_map[self.item['task_id']],self.item['scenario'],context_builder=context);environments.append(self.env)
        def __call__(self,messages):
            nonlocal failed
            if fail_once and self.item['index']==5 and not failed:failed=True;raise RuntimeError('simulated native interruption')
            action=corrective_action(self.env);self.env.step(action);self.calls+=1;generation=tok.generation(messages,json.dumps(action),self.sampling)
            if self.first is None:self.first=generation
            if self.engine.proof is None:
                self.engine.proof=[{'passed':True,'adapter_sha256':W_SHA,'registered_ids':[1],'active_slots':[1],'tensor_count':112,
                    'tensors':tensors,'loaded_tensor_sha256':digest(tensors),'same_forward_capture':capture(asdict(generation),self.seed)}]
            if self.on_action:self.on_action(len(generation.response_token_ids))
            return generation
    monkeypatch.setattr(grpo_vllm,'engine_and_tokenizer',lambda *a:(Engine(),tok,None));monkeypatch.setattr(grpo_vllm,'BehaviorPolicy',Policy)
    monkeypatch.setattr(grpo_vllm,'shutdown',lambda e:[env.close() for env in environments])
    return worker,common,tok,calls


def test_collector_resumes_native_episodes_and_recomputes_all_24_local_groups(tmp_path,monkeypatch):
    worker,common,tok,calls=collection_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError,match='simulated'):worker.collect(tmp_path)
    assert len(list((tmp_path/'episodes').glob('*.json')))==5
    hashes={p.name:file_hash(p) for p in (tmp_path/'episodes').glob('*.json')}
    worker.collect(tmp_path);result=common.verify_result(tmp_path,tok)
    assert all(r['remaining_metric_completion']==1 for r in result['summary']['episode_rows'])
    assert result['passed'] and result['episodes_replayed']==96 and result['sessions']==2 and result['new_training_updates']==0
    assert all(calls[i]==1 for i in range(5)) and calls[5]==2
    assert all(file_hash(tmp_path/'episodes'/n)==sha for n,sha in hashes.items())
    assert all(v['eligible_groups']==4 and v['successes']==16 for levels in result['summary']['by_scenario'].values() for v in levels.values())
    before=dict(calls);worker.collect(tmp_path);assert dict(calls)==before
    data=read(tmp_path/'results.json');bad=copy.deepcopy(data);bad['summary']['episodes']=95
    with pytest.raises(ValueError,match='summary'):common.verify_result(tmp_path,tok,data=bad)
    s=copy.deepcopy(data['sessions'][0]);s['worker_proof'][0]['same_forward_capture']['rows'][0]['native_processed_logprob']-=.1
    with pytest.raises(ValueError):common.verify_session(tmp_path,s,tok,CFG)
    s=copy.deepcopy(data['sessions'][0]);s['worker_proof'][0]['tensors']['0']['sha256']='changed'
    with pytest.raises(ValueError,match='tensor'):common.verify_session(tmp_path,s,tok,CFG)
    e=copy.deepcopy(data['episodes'][0]);e['seed']+=1
    with pytest.raises(ValueError,match='identity'):common.verify_episode(tmp_path,plan(CFG)[0],e,tok,CFG,data['sessions'][0])
    e=copy.deepcopy(data['episodes'][0]);e['turns'][0]['generation']['prompt_token_ids']=[999]
    with pytest.raises(ValueError):common.verify_episode(tmp_path,plan(CFG)[0],e,tok,CFG,data['sessions'][0])


def driver_fixture(tmp_path,monkeypatch,fail=False):
    import run_v2_grpo_hard_preflight as driver
    root=tmp_path/'v2';path=root/'configs/preflight.json';atomic_json(path,CFG)
    atomic_json(root/CFG['release_manifest'],{'files':{'configs/preflight.json':file_hash(path)}})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'source_files',lambda:{'configs/preflight.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None);calls=[];failed=False
    def stage(command,**kwargs):
        nonlocal failed
        name=command[3];target=Path(command[-1]);calls.append(name);kwargs['log_path'].write_text('Progress 100%\r')
        assert kwargs['timeout']<=7020
        if name=='prepare':atomic_json(target/'input_binding.json',{})
        elif name=='collect':
            if fail and not failed:
                failed=True;atomic_json(target/'episodes/0000.json',{});return 1
            for i in range(96):atomic_json(target/'episodes'/f'{i:04}.json',{})
        else:atomic_json(target/'preflight_audit.json',{'passed':True,'summary':{'episodes':96}})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    return driver,path,calls,root/'artifacts'/CFG['run_id']


def test_driver_resume_preserves_budget_and_always_packages(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError):driver.run(path)
    prior=read(output/'run_status.json')['elapsed_seconds'];archive=driver.run(path,True)
    assert archive.exists() and archive.name.endswith('attempt-002.zip') and calls.count('prepare')==1
    status=read(output/'run_status.json');assert status['error'] is None and status['committed_episodes']==96 and status['new_training_updates']==0 and status['elapsed_seconds']>=prior


def test_driver_refuses_source_change_before_new_sampling(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError):driver.run(path)
    n=len(calls);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):driver.run(path,True)
    assert len(calls)==n


def test_resume_budget_accounts_for_unrecorded_active_stage():
    from run_v2_grpo_hard_preflight import resume_spent
    assert resume_spent({'elapsed_seconds':100,'active_started_unix':900},{'elapsed_seconds':90},1000)==200


def test_real_W_and_frozen_context_bindings_are_rechecked(tmp_path):
    import shutil
    from grpo_hard_common import verify_inputs
    cfg=copy.deepcopy(CFG);atomic_json(tmp_path/'config.json',cfg)
    atomic_json(tmp_path/'run_manifest.json',{'source_sha256':{'configs/grpo-hard-recovery-preflight-v1.json':file_hash(ROOT/'configs/grpo-hard-recovery-preflight-v1.json')}})
    identity=read(ROOT/cfg['initial_dir']/'model_identity.json');atomic_json(tmp_path/'model_identity.json',identity)
    shutil.copytree(ROOT/cfg['initial_dir']/'adapter',tmp_path/'initial/adapter')
    frozen=frozen_inputs(cfg)
    for n,v in frozen.items():atomic_json(tmp_path/n,v)
    names=['config.json','model_identity.json',*frozen]
    names += [p.relative_to(tmp_path).as_posix() for p in (tmp_path/'initial').rglob('*') if p.is_file()]
    binding={'config_sha256':file_hash(tmp_path/'config.json'),'files':{n:file_hash(tmp_path/n) for n in names}}
    atomic_json(tmp_path/'input_binding.json',binding)
    assert verify_inputs(tmp_path,local=True)==cfg
    path=tmp_path/'initial/adapter/adapter_model.safetensors';original=path.read_bytes();path.write_bytes(original+b'changed')
    with pytest.raises(ValueError):verify_inputs(tmp_path,local=True)
    path.write_bytes(original)
    bad=read(tmp_path/'contexts.json');bad[0]['context']['messages'][0]['content']='tampered';atomic_json(tmp_path/'contexts.json',bad)
    binding['files']['contexts.json']=file_hash(tmp_path/'contexts.json');atomic_json(tmp_path/'input_binding.json',binding)
    with pytest.raises(ValueError,match='paired preflight input'):verify_inputs(tmp_path,local=True)


def test_driver_timeout_returns_review_with_cumulative_cap(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch)
    original=driver.run_logged
    def timeout(command,**kwargs):
        if command[3]=='collect':raise TimeoutError('simulated two-hour cap')
        return original(command,**kwargs)
    monkeypatch.setattr(driver,'run_logged',timeout)
    with pytest.raises(RuntimeError,match='two-hour cap'):driver.run(path)
    status=read(output/'run_status.json')
    assert status['stop_reason']=='timeout' and status['max_wall_seconds']==7200
    assert (output.parents[1]/status['archive']).exists()


def test_equal_failed_outcomes_without_new_progress_produce_no_local_signal():
    tok=NativeTokenizer();task_map={t.id:t for t in tasks()};rows=[]
    for item in plan(CFG):
        def bad_policy(messages):
            action={'tool':'calculate_change','arguments':{'current_evidence':'invalid','prior_evidence':'invalid'}}
            return tok.generation(messages,json.dumps(action),CFG['sampling'])
        e=run_continuation(task_map[item['task_id']],item['scenario'],bad_policy,CFG['sampling'],0,item['seed'],item['rollout_index'],context_builder=context)
        e.update(base_scenario=item['base_scenario'],level=item['level']);rows.append(e)
    result=summarize(rows)
    assert all(v['successes']==0 and v['progress_episodes']==0 and v['eligible_groups']==0 for levels in result['by_scenario'].values() for v in levels.values())
    assert all(r['remaining_metric_completion']==0 and r['repeated_failed_actions']>0 for r in result['episode_rows'])
