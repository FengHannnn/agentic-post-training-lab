import sys,json,copy,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import apt_v2.independent_recovery_test as it
from apt_v2.tasks import make_tasks,metric_sql
from apt_v2.dpo_recovery import build_context,make_pair
from apt_v2.dpo import corrective_action,TEMPORARY_ERROR
from apt_v2.environment import Environment
from apt_v2.runtime import Generation
from apt_v2.chosen_sft import chosen_examples
from apt_v2.recovery_diagnostics import diagnose_action
from test_dpo_recovery_training_v2 import pair_fixtures
from test_dpo_v2 import CharTokenizer

def native(tok,messages,text,mode='greedy'):
    ids=[ord(c)+1 for c in text]+[0]
    return Generation(text,mode,tok.apply_chat_template(messages,add_generation_prompt=True),ids,[-.1]*len(ids),None,'stop')

def oracle(task,context,tok,mode):
    def policy(messages):
        env=it.context_environment(task,context)
        try:
            for m in messages[len(context['messages']):]:
                if m['role']=='assistant':env.step(json.loads(m['content']))
            return native(tok,messages,json.dumps(corrective_action(env)),mode)
        finally:env.close()
    return policy

def accepted_bad_sql():
    for task in [t for t in make_tasks() if t.split=='test']:
        for month in (task.current_month,task.prior_month,'2025-12'):
            sql=metric_sql(task,'current_net_cents').replace("WHERE o.status='paid' AND r.refund_month=",
                "WHERE o.status='paid' AND o.order_month='"+month+"' AND r.refund_month=")
            env=Environment(task)
            try:
                row=env.step({'tool':'query_sql','arguments':{'sql':sql}})
                if row['tool_error']==TEMPORARY_ERROR:row=env.step({'tool':'query_sql','arguments':{'sql':sql}})
                if 'current_net_cents' in row['observation'].get('verified_metrics',[]):return task,sql
            finally:env.close()
    raise AssertionError('known calendar verifier blind spot fixture expected')

def test_explicit_test_contexts_keep_training_admission_closed():
    pairs=pair_fixtures();examples=chosen_examples(pairs);tasks=it.independent_tasks(pairs,examples)
    assert len(tasks)==4 and len(it.test_contexts(tasks))==36
    task=tasks[0];c=it.build_test_context(task,'format_repair')
    with pytest.raises(ValueError,match='split'):build_context(task,'format_repair')
    with pytest.raises(ValueError,match='split'):diagnose_action(task,c,c['chosen'])
    with pytest.raises(ValueError,match='split'):make_pair(task,c,CharTokenizer(),100000)
    assert it.diagnose_test_action(task,c,c['chosen'])[1]['productive']
    bad=copy.deepcopy(pairs);bad[0]['split']='test'
    with pytest.raises(ValueError,match='provenance'):it.independent_tasks(bad,examples)

def test_both_unverified_context_and_full_continuation_replay():
    task=next(t for t in make_tasks() if t.id=='test-001');tok=CharTokenizer()
    for scenario in it.NOVEL_SCENARIOS:
        c=it.build_test_context(task,scenario);env=it.context_environment(task,c)
        try:assert not env.verified and sum(not values for values in env.evidence.values())==2
        finally:env.close()
        _,bad=it.diagnose_test_action(task,c,c['scripted_rejected'])
        assert [s['status'] for s in bad['evidence']['slots']]==['id_exists_unverified','id_exists_unverified']
        episode=it.run_test_recovery(task,c,oracle(task,c,tok,'greedy'))
        assert episode['success'] and len(episode['turns'])==5 and episode['turns'][-1]['action']['tool']=='submit'
        assert len(c['records'])+len(episode['turns'])<=task.max_turns
        assert it.replay_test_recovery(task,c,episode,tok,'greedy')==5
        bad=copy.deepcopy(episode);bad['turns'][0]['generation']['prompt_token_ids']=[1]
        with pytest.raises(ValueError,match='prompt'):it.replay_test_recovery(task,c,bad,tok,'greedy')

def test_calendar_worlds_accept_oracle_sql_and_detect_accepted_wrong_month_filter():
    for task in [t for t in make_tasks() if t.split=='test']:
        for metric in ('current_net_cents','prior_net_cents'):
            assert it.calendar_sql_check(task,metric_sql(task,metric),[metric])['passed']
    task,sql=accepted_bad_sql()
    result=it.calendar_sql_check(task,sql,['current_net_cents'])
    assert not result['passed'] and result['metric_passed']['current_net_cents'] is False

def test_calendar_artifact_walk_checks_final_sources_and_allows_unused_bad_query():
    task,bad_sql=accepted_bad_sql()
    def episode(replace_source):
        env=Environment(task)
        try:
            env.step({'tool':'read_policy','arguments':{}});env.step({'tool':'inspect_schema','arguments':{}})
            row=env.step({'tool':'query_sql','arguments':{'sql':bad_sql}})
            if row['tool_error']==TEMPORARY_ERROR:row=env.step({'tool':'query_sql','arguments':{'sql':bad_sql}})
            if replace_source:env.step({'tool':'query_sql','arguments':{'sql':metric_sql(task,'current_net_cents')}})
            env.step({'tool':'query_sql','arguments':{'sql':metric_sql(task,'prior_net_cents')}})
            while not env.done:env.step(corrective_action(env))
            assert env.success
            return it.calendar_episode_audit(task,env.records,True,env.records[-1]['observation']['artifact'])
        finally:env.close()
    fragile=episode(False);repaired=episode(True)
    assert not fragile['calendar_consistent_success']
    assert repaired['calendar_consistent_success'] and repaired['calendar_queries_passed']<repaired['verified_query_count']

def test_native_policy_seed_alignment_and_probability_guards(monkeypatch):
    import test_v2_hard_recovery as worker
    tok=CharTokenizer();calls=[];text='{"tool":"read_policy","arguments":{}}';ids=[ord(c)+1 for c in text]+[0]
    class Params:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(SamplingParams=Params))
    class Engine:
        def generate(self,prompts,params,**kwargs):
            calls.append(params.seed)
            completion=types.SimpleNamespace(token_ids=ids,logprobs=[{t:types.SimpleNamespace(logprob=-.1)} for t in ids],finish_reason='stop')
            return [types.SimpleNamespace(prompt_token_ids=prompts[0]['prompt_token_ids'],outputs=[completion])]
    policy=worker.NativePolicy(Engine(),tok,None,{'eval':{'max_model_len':8192,'max_new_tokens':256}},'sample',31,.8)
    messages=[{'role':'user','content':'task'}]
    assert policy(messages).mode=='dpo_sample' and policy(messages).text==text and calls==[31,32]

def test_independent_task_and_recovery_audit_end_to_end_and_tamper(tmp_path):
    import test_v2_hard_recovery as generator
    import audit_v2_hard_recovery_test as auditor
    from test_dpo_independent_v2 import fixtures
    from apt_v2.sft_diagnostics import summarize_episodes
    from common import file_hash
    tok=CharTokenizer();pairs=pair_fixtures();examples=chosen_examples(pairs);tasks=it.independent_tasks(pairs,examples);tm={t.id:t for t in tasks};contexts=it.test_contexts(tasks)
    cfg={'evaluation_branches':['before','after'],'sample_eval':{'task_count':4,'rollouts_per_task':1,'seed':31,'temperature':.8},
        'recovery_eval':{'task_count':4,'samples_per_context':1,'seed':33,'temperature':.8},'eval':{'max_new_tokens':100000}}
    generator.jsonl(tmp_path/'pairs.jsonl',pairs);generator.jsonl(tmp_path/'examples.jsonl',examples);generator.jsonl(tmp_path/'contexts.jsonl',contexts)
    from dataclasses import asdict
    (tmp_path/'test_tasks.json').write_text(json.dumps([asdict(t) for t in tasks]))
    comparisons={};stochastic={};allactions=[];allepisodes=[]
    for branch in cfg['evaluation_branches']:
        for mode in ('greedy','sample'):
            name=branch+('-test' if mode=='greedy' else '-sample');episodes=fixtures(tasks,branch,cfg['sample_eval'] if mode=='sample' else None)
            p=tmp_path/(name+'-episodes.jsonl');generator.jsonl(p,episodes)
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(p)}
            if mode=='sample':summary['successes_by_task']={t.id:1 for t in tasks};stochastic[branch]=summary
            else:comparisons[branch]={'test':summary}
            (tmp_path/(name+'-summary.json')).write_text(json.dumps(summary))
        actions=[];continuations=[]
        for c in contexts:
            for mode in ('greedy','sample'):
                seed=generator.recovery_seed(cfg,tasks,c,0);temperature=0 if mode=='greedy' else .8;native_mode='greedy' if mode=='greedy' else 'dpo_sample'
                if c['scenario'] in it.NOVEL_SCENARIOS:
                    e=it.run_test_recovery(tm[c['task_id']],c,oracle(tm[c['task_id']],c,tok,native_mode))
                    e.update(model_branch=branch,mode=mode,sample_index=0,sample_seed=seed,temperature=temperature)
                    for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                    continuations.append(e);g=e['turns'][0]['generation']
                else:g=asdict(native(tok,c['messages'],c['chosen'],native_mode))
                actual,d=it.diagnose_test_action(tm[c['task_id']],c,g['text'])
                actions.append({'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':'test','context_id':c['context_id'],
                    'scenario':c['scenario'],'mode':mode,'sample_index':0,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d})
        generator.jsonl(tmp_path/(branch+'-recovery-actions.jsonl'),actions);generator.jsonl(tmp_path/(branch+'-recovery-episodes.jsonl'),continuations)
        allactions.extend(actions);allepisodes.extend(continuations)
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparisons));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(stochastic))
    (tmp_path/'recovery_comparison.json').write_text(json.dumps(generator.recovery_summary(allactions,allepisodes,cfg['evaluation_branches'])))
    evidence,summary,calendar=auditor.audit_results(tmp_path,cfg,tok)
    assert evidence['episodes_replayed']==16 and evidence['first_recovery_actions_replayed']==144 and evidence['recovery_continuations_replayed']==32
    assert summary['paired_continuation']['sample']['both_success']==8 and calendar['summary']['after']['complete']['sample']['calendar_consistent_successes']==4
    auditor.export_counts(tmp_path,summary)
    assert len((tmp_path/'recovery_counts.csv').read_text(encoding='utf-8-sig').splitlines())==41
    actions[0]['diagnostic']['productive']=False;p=tmp_path/'after-recovery-actions.jsonl';p.unlink();generator.jsonl(p,actions)
    with pytest.raises(ValueError,match='replay'):auditor.audit_results(tmp_path,cfg,tok)

def test_prepare_pins_checkpoints_and_train_test_separation(tmp_path,monkeypatch):
    import test_v2_hard_recovery as worker
    from common import file_hash
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    pairs=pair_fixtures();examples=chosen_examples(pairs)
    def put(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    for branch,total in [('parent',255),('candidate',513)]:
        folder=tmp_path/'artifacts'/branch;adapter=folder/'final/adapter/adapter_model.safetensors';adapter.parent.mkdir(parents=True);adapter.write_bytes(branch.encode())
        put(folder/'final/complete.json',{'adapter_sha256':{'adapter_model.safetensors':file_hash(adapter)},'total_sft_updates':total,'total_dpo_updates':11,'steps':129 if branch=='parent' else 258})
        put(folder/'model_identity.json',{'resolved_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','files_sha256':{}})
        for n,value in [('audit.json',{'passed':True}),('training_summary.json',{'optimizer_updates':258}),('input_binding.json',{})]:put(folder/n,value)
        worker.jsonl(folder/'pairs.jsonl',pairs);worker.jsonl(folder/'examples.jsonl',examples)
    data=tmp_path/'artifacts/data';data.mkdir();worker.jsonl(data/'pairs.jsonl',pairs);put(data/'audit.json',{'passed':True})
    cfg={'training_enabled':False,'parent_run':'parent','candidate_run':'candidate','data_run':'data',
        'adapter_sha256':file_hash(tmp_path/'artifacts/parent/final/adapter/adapter_model.safetensors'),
        'candidate_adapter_sha256':file_hash(tmp_path/'artifacts/candidate/final/adapter/adapter_model.safetensors'),
        'pairs_sha256':file_hash(data/'pairs.jsonl'),'sample_eval':{'task_count':4},'recovery_eval':{'task_count':4}}
    put(tmp_path/'artifacts/parent/config.json',{})
    put(tmp_path/'artifacts/candidate/config.json',{'initial_run':'parent','adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256']})
    put(tmp_path/'configs/hard-recovery-independent-test-v1.json',cfg);output=tmp_path/'output';put(output/'config.json',cfg);put(output/'run_manifest.json',{'source_sha256':{}})
    worker.prepare(output)
    assert len(worker.load(output/'contexts.jsonl'))==36 and worker.read(output/'prepare_summary.json')['test_contexts_in_training']==0
    (tmp_path/'artifacts/candidate/final/adapter/adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='identity'):worker.verify_inputs(output)

def test_runner_failure_preserves_live_logs_and_inference_only_status(tmp_path,monkeypatch,capsys):
    import run_v2_hard_recovery_test as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    for source in ('parent','candidate'):
        for name in ('final/complete.json','final/adapter/adapter_model.safetensors','model_identity.json','audit.json','examples.jsonl','pairs.jsonl'):
            p=tmp_path/'artifacts'/source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    cfg={'run_id':'independent','parent_run':'parent','candidate_run':'candidate','max_wall_seconds':14400}
    (tmp_path/'configs/hard-recovery-independent-test-v1.json').write_text(json.dumps(cfg));(tmp_path/'hard_recovery_test_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/test_v2_hard_recovery.py').write_text("print('independent live progress',flush=True)\nraise RuntimeError('intentional')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'independent live progress' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/independent/run_status.json').read_text())
    assert status['training_enabled'] is False and status['optimizer_updates']==0
    import zipfile
    with zipfile.ZipFile(tmp_path/'independent-review.zip') as z:assert 'independent/prepare.log' in z.namelist()

def test_fixed_test_config_budget_and_two_pinned_models():
    cfg=json.loads((ROOT/'configs/hard-recovery-independent-test-v1.json').read_text())
    assert cfg['training_enabled'] is False and cfg['max_wall_seconds']==14400 and cfg['sample_eval']['rollouts_per_task']==8
    assert cfg['candidate_adapter_sha256']=='f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f'
    assert 2*4*(1+8)==72 and 2*4*len(it.TEST_SCENARIOS)*(1+3)==288 and 2*4*len(it.NOVEL_SCENARIOS)*(1+3)==64
