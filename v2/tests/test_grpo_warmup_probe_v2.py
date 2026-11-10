import copy,json,sys,types,shutil
from pathlib import Path
from collections import Counter
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,file_hash
from apt_v2.grpo_warmup_probe import *
from apt_v2.sft import encode_example


def config():return read(ROOT/'configs/grpo-sql-warmup-probe-v1.json')
@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/config()['input_dir']/'tokenizer',local_files_only=True,trust_remote_code=False)
@pytest.fixture(scope='module')
def plan():return read(ROOT/config()['input_dir']/'probe-plan.json')


def generation(tok,item):
    sample=encode_example(tok,item,8192);ids=sample['input_ids'][sample['prompt_length']:]
    return {'text':item['response'],'mode':'greedy','prompt_token_ids':sample['input_ids'][:sample['prompt_length']],
            'response_token_ids':ids,'raw_token_log_probs':None,'behavior_token_log_probs':None,'finish_reason':'stop'}


def score(prompt,response,seed,value):
    return {'source':'vllm_prompt_logprobs_T1','not_used_for_policy_updates':True,'prompt_sha256':digest(prompt+response),
            'response_token_ids':response,'token_log_probs':[value]*len(response),'auxiliary_token_ids':[151645],'seed':seed}


def row(cfg,item,tok,name='w'):
    g=generation(tok,item);seed=cfg['seed']+item['index']*100
    r={'index':item['index'],'task_id':item['task_id'],'state':item['state'],'model':name,'seed':seed,
       'plan_sha256':digest(item),'session_id':'session','generations':[g]* (2 if item['kind']=='target' else 1),
       'executions':[execute(item,g['text'])]* (2 if item['kind']=='target' else 1),'scores':None,'score_summary':None}
    if item['kind']=='target':
        v=variants(tok,item);r['scores']={'correct':score(v['prompt_ids'],v['correct_ids'],seed+17,-.1),
                                        'omitted':score(v['prompt_ids'],v['omitted_ids'],seed+18,-.2)}
        r['score_summary']=score_summary(v,r['scores'])
    return r


def test_fixed_plan_covers_all_tasks_and_both_pools_for_each_maintenance_state(plan):
    data=read(ROOT/config()['input_dir']/'sft-examples.json');assert plan==make_plan(data)
    assert len(plan)==24 and {p['task_id'] for p in plan[:12]}=={t.id for t in training_tasks()}
    assert Counter(p['state'] for p in plan[12:])==Counter({s:3 for s in STATES})
    for s in STATES:assert {p['task_id'].startswith('train-') for p in plan[12:] if p['state']==s}=={True,False}
    assert all(productive(p,execute(p,p['response'])) for p in plan)


def test_actual_native_alias_branches_share_prefix_and_counterfactual_is_not_certified(plan,tok):
    for p in plan[:12]:
        v=variants(tok,p);i=v['branch_index']
        assert v['correct_ids'][:i]==v['omitted_ids'][:i] and v['correct_ids'][i]!=v['omitted_ids'][i]
        assert len(v['alias_indices'])==5 and productive(p,execute(p,p['response']))
        assert not productive(p,execute(p,v['omitted_text']))
        assert v['correct_ids'][-1]==tok.eos_token_id and v['omitted_ids'][-1]==tok.eos_token_id


@pytest.mark.parametrize('change',[{'max_wall_seconds':1801},{'prior_elapsed_seconds':0},{'target_repeats':1},
                                   {'sampling':dict(config()['sampling'],max_new_tokens=512)},
                                   {'models':dict(config()['models'],w={'adapter_sha256':'x','source_sft_updates':513})}])
def test_frozen_probe_rejects_budget_sampling_or_model_drift(change):
    c=config();validate_config(c);c.update(change)
    with pytest.raises(ValueError):validate_config(c)


def test_prior_receipts_and_real_export_formats_are_bound():
    from grpo_warmup_probe_worker import model_identity
    c=config();prior=read(ROOT/c['input_dir']/'prior-evidence.json');assert validate_prior(prior,c)
    for name in c['models']:assert model_identity(c,ROOT/c['input_dir']/'models'/name,name)['model_id']=='Qwen/Qwen2.5-1.5B-Instruct'
    assert 'README.md' in read(ROOT/c['input_dir']/'models/w/complete.json')['adapter_sha256']
    prior['receipts'][1]['combined_elapsed_seconds']=0
    with pytest.raises(ValueError):validate_prior(prior,c)


@pytest.mark.parametrize('kind',['target','maintenance'])
def test_replay_validates_native_tokens_scores_and_public_prefix(plan,tok,kind):
    p=plan[0 if kind=='target' else 12];r=row(config(),p,tok)
    assert validate_row(config(),p,r,tok)
    bad=copy.deepcopy(r);bad['generations'][0]['prompt_token_ids'][0]+=1
    with pytest.raises(ValueError):validate_row(config(),p,bad,tok)
    bad_plan=copy.deepcopy(p);bad_plan['messages'][-1]['content']='{"observation":{"fake":1}}'
    with pytest.raises(ValueError):execute(bad_plan,p['response'])
    if kind=='target':
        for change in ('seed','nonfinite','behavior_q','summary'):
            bad=copy.deepcopy(r)
            if change=='seed':bad['scores']['correct']['seed']+=1
            elif change=='nonfinite':bad['scores']['correct']['token_log_probs'][0]=float('nan')
            elif change=='behavior_q':bad['scores']['correct']['source']='behavior_q'
            else:bad['score_summary']['branch_logp_margin']=0
            with pytest.raises(ValueError):validate_row(config(),p,bad,tok)


def test_summary_distinguishes_productive_actions_and_repeat_consistency(plan,tok):
    rows=[row(config(),p,tok) for p in plan];summary=summarize(plan,rows)
    assert summary['target_first_certifications']==12 and summary['maintenance_productive_actions']==12
    assert summary['target_repeat_identical']==12 and summary['correct_branch_preferred_contexts']==12
    assert summary['alias_token_nll_mean']==pytest.approx(.1)
    rows[0]['generations'][1]=dict(rows[0]['generations'][1],response_token_ids=[7])
    assert summarize(plan,rows)['target_repeat_identical']==11
    with pytest.raises(ValueError):summarize(plan,rows[:-1])


def test_native_forced_score_uses_exact_prompt_scores_and_counts_extra_token(monkeypatch):
    from grpo_warmup_probe_worker import forced_score
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(SamplingParams=lambda **kw:types.SimpleNamespace(**kw)))
    class Engine:
        def generate(self,inputs,params,**kw):
            ids=inputs[0]['prompt_token_ids'];assert params.temperature==1 and params.prompt_logprobs==1 and params.max_tokens==1
            return [types.SimpleNamespace(prompt_token_ids=ids,prompt_logprobs=[{i:types.SimpleNamespace(logprob=-.3)} for i in ids],
                                          outputs=[types.SimpleNamespace(token_ids=[2])])]
    result=forced_score(Engine(),object(),[11,12],[13,14],5)
    validate_score(result,[11,12],[13,14],5);assert result['token_log_probs']==[-.3,-.3]


def test_probe_worker_checks_real_bf16_projection_and_does_not_trust_boolean_pass():
    from grpo_warmup_probe_worker import worker_proof
    from grpo_vllm import projected_checkpoint_tensors
    folder=ROOT/config()['input_dir']/'models/w';tensors=projected_checkpoint_tensors(folder/'adapter')
    proof=[{'passed':True,'adapter_sha256':MODEL_IDENTITIES['w'][0],'tensor_count':112,'registered_ids':[1],
            'active_slots':[1],'tensors':tensors,'loaded_tensor_sha256':digest(tensors)}]
    assert worker_proof(proof,folder,MODEL_IDENTITIES['w'][0])
    bad=copy.deepcopy(proof);bad[0]['tensors'][next(iter(tensors))]['sha256']='wrong'
    with pytest.raises(ValueError):worker_proof(bad,folder,MODEL_IDENTITIES['w'][0])


def test_complete_model_replays_rows_and_rejects_wrong_worker_model(tmp_path,monkeypatch,plan,tok):
    import grpo_warmup_probe_worker as worker
    c=config();prepared=tmp_path/'prepared';shutil.copytree(ROOT/c['input_dir']/'models/w',prepared/'inputs/models/w')
    atomic_json(tmp_path/'config.json',c);atomic_json(prepared/'manifest.json',{'files':{}})
    hardware={'gpu_uuid':'fixed'};atomic_json(tmp_path/'execution_environment.json',hardware)
    monkeypatch.setattr(worker,'verify_inputs',lambda *a,**k:(c,plan))
    from grpo_vllm import projected_checkpoint_tensors
    tensors=projected_checkpoint_tensors(prepared/'inputs/models/w/adapter')
    proof=[{'passed':True,'adapter_sha256':MODEL_IDENTITIES['w'][0],'tensor_count':112,'registered_ids':[1],
            'active_slots':[1],'tensors':tensors,'loaded_tensor_sha256':digest(tensors)}]
    root=tmp_path/'probes/w';atomic_json(root/'sessions/session.json',{'session_id':'session','model':'w',
        'config_sha256':file_hash(tmp_path/'config.json'),'prepared_sha256':file_hash(prepared/'manifest.json'),
        'hardware':hardware,'adapter_sha256':MODEL_IDENTITIES['w'][0],'worker_proof':proof})
    rows=[row(c,p,tok) for p in plan]
    for r in rows:atomic_json(root/'rows'/f"{r['index']:04}.json",r)
    binding={p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*.json')}
    atomic_json(root/'complete.json',{'config_sha256':file_hash(tmp_path/'config.json'),'files':binding,'summary':summarize(plan,rows)})
    assert worker.verify_model(tmp_path,'w',tok)['target_first_certifications']==12
    bad=read(root/'rows/0000.json');bad['model']='b15';atomic_json(root/'rows/0000.json',bad)
    with pytest.raises(ValueError):worker.verify_model(tmp_path,'w',tok)


def test_request_accounting_includes_retries_and_marks_truncated_logs(tmp_path):
    from grpo_common import jsonl
    from grpo_warmup_probe_worker import request_accounting
    p=tmp_path/'probes/w/attempts/session.jsonl';p.parent.mkdir(parents=True)
    for _ in range(2):jsonl(p,{'phase':'greedy'})
    jsonl(p,{'phase':'correct','score':{'auxiliary_token_ids':[5]}})
    with p.open('a') as f:f.write('{truncated')
    counted=request_accounting(tmp_path)
    assert counted['logged_greedy_requests']==2 and counted['logged_auxiliary_tokens']==1 and len(counted['truncated_log_lines'])==1


def test_runner_failure_resume_and_combined_packaging_budget(tmp_path,monkeypatch):
    import run_v2_grpo_warmup_probe as runner
    import run_v2_grpo_repair as base
    c=config();p=tmp_path/'configs/master.json';atomic_json(p,c)
    prior=ROOT/c['input_dir']/'prior-evidence.json';atomic_json(tmp_path/c['input_dir']/'prior-evidence.json',read(prior))
    atomic_json(tmp_path/runner.MANIFEST,{'files':{'configs/master.json':file_hash(p)}})
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'sqlite':'3.54.1','gpu_uuid':'fixed'})
    monkeypatch.setattr(base,'resource_snapshot',lambda *a:None)
    def fail(self,name,*args,**kwargs):raise RuntimeError('injected probe interruption')
    monkeypatch.setattr(runner.Study,'stage',fail)
    with pytest.raises(RuntimeError,match='probe interruption'):runner.run(p)
    receipt=read((tmp_path/(c['run_id']+'-review-attempt-001.zip')).with_suffix('.receipt.json'))
    assert receipt['budget_seconds']==1800 and receipt['combined_elapsed_seconds']==c['prior_elapsed_seconds']+receipt['elapsed_seconds_including_packaging']
    calls=[];monkeypatch.setattr(runner.Study,'stage',lambda self,name,*a,**k:calls.append(name))
    runner.run(p,True);assert calls==['prepare','probe-w','probe-b24','probe-b15','audit']
    second=read((tmp_path/(c['run_id']+'-review-attempt-002.zip')).with_suffix('.receipt.json'))
    assert second['elapsed_seconds_including_packaging']>=receipt['elapsed_seconds_including_packaging']
    with pytest.raises(ValueError,match='completed'):runner.run(p,True)


@pytest.mark.parametrize('interrupted',[False,True])
def test_actual_probe_orchestration_commits_and_reuses_native_rows(tmp_path,monkeypatch,plan,tok,interrupted):
    import grpo_warmup_probe_worker as worker
    import grpo_vllm as native
    from transformers import AutoTokenizer
    cfg=config();prepared=tmp_path/'prepared';shutil.copytree(ROOT/cfg['input_dir']/'models/w',prepared/'inputs/models/w')
    atomic_json(tmp_path/'config.json',cfg);atomic_json(prepared/'manifest.json',{'files':{}})
    atomic_json(prepared/'branch_config.json',read(ROOT/cfg['branch_template']))
    atomic_json(tmp_path/'execution_environment.json',{'gpu_uuid':'fixed'})
    monkeypatch.setattr(worker,'verify_inputs',lambda *a,**k:(cfg,plan))
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(SamplingParams=lambda **kw:types.SimpleNamespace(**kw)))
    projected=native.projected_checkpoint_tensors(prepared/'inputs/models/w/adapter')
    proof=[{'passed':True,'adapter_sha256':MODEL_IDENTITIES['w'][0],'tensor_count':112,'registered_ids':[1],
            'active_slots':[1],'tensors':projected,'loaded_tensor_sha256':digest(projected)}]
    prompts={tuple(tok.apply_chat_template(p['messages'],tokenize=True,add_generation_prompt=True)):generation(tok,p) for p in plan}
    forcings={}
    for p in plan[:12]:
        v=variants(tok,p)
        for k,value in [('correct_ids',-.1),('omitted_ids',-.2)]:forcings[tuple(v['prompt_ids']+v[k])]=value
    class Engine:
        def __init__(self,fail=False):self.calls=0;self.fail=fail
        def collective_rpc(self,*args,**kwargs):return proof
        def generate(self,inputs,params,**kwargs):
            self.calls+=1
            if self.fail and self.calls==5:raise RuntimeError('injected native interruption')
            ids=inputs[0]['prompt_token_ids'];key=tuple(ids)
            if params.max_tokens==1:
                value=forcings[key]
                return [types.SimpleNamespace(prompt_token_ids=ids,prompt_logprobs=[{i:types.SimpleNamespace(logprob=value)} for i in ids],
                                               outputs=[types.SimpleNamespace(token_ids=[151645])])]
            g=prompts[key];response=g['response_token_ids']
            return [types.SimpleNamespace(prompt_token_ids=ids,outputs=[types.SimpleNamespace(token_ids=response,
                        logprobs=[{i:types.SimpleNamespace(logprob=-.1)} for i in response],finish_reason='stop')])]
    engines=[]
    def create(*a,**k):
        e=Engine(fail=interrupted and not engines);engines.append(e);return e,tok,object()
    monkeypatch.setattr(native,'engine_and_tokenizer',create);monkeypatch.setattr(native,'shutdown',lambda e:None)
    if interrupted:
        with pytest.raises(RuntimeError,match='native interruption'):worker.probe_model(tmp_path,'w')
        assert len(list((tmp_path/'probes/w/rows').glob('*.json')))==1
        original=file_hash(tmp_path/'probes/w/rows/0000.json')
    summary=worker.probe_model(tmp_path,'w');assert summary['target_first_certifications']==12
    if interrupted:assert file_hash(tmp_path/'probes/w/rows/0000.json')==original and engines[-1].calls==56
    else:assert engines[0].calls==60
    count=len(engines);assert worker.probe_model(tmp_path,'w')==summary and len(engines)==count
    assert worker.request_accounting(tmp_path)['logged_forced_requests']==24
