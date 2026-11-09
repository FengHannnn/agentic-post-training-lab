import sys,json,copy,math
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.tasks import make_tasks,truth,metric_sql,connect
from apt_v2.environment import Environment,execute_readonly,ToolError
from apt_v2.runtime import ScriptedPolicy,run_episode,parse_action
from apt_v2.rewards import discounted_returns,assign_turn_credit,training_samples

@pytest.mark.parametrize('task',make_tasks(),ids=lambda t:t.id)
def test_reference_completes_task_and_recovers_injected_fault(task):
    episode=run_episode(task,ScriptedPolicy(task))
    assert episode['success']
    assert episode['recovered']==(task.fault_mode=='transient_query')
    assert episode['artifact']['values']==truth(task)
    assert abs(episode['process_reward_sum'])<1e-9
    assert episode['total_reward']==pytest.approx(1-.01*len(episode['turns']))

def test_family_splits_and_source_worlds_are_independent():
    tasks=make_tasks()
    groups={s:{t.family_id for t in tasks if t.split==s} for s in ('train','development','test')}
    assert not groups['train']&groups['development'] and not groups['train']&groups['test']
    t=tasks[0]
    assert truth(t)==truth(t) and truth(t)!=truth(t,1) and truth(t,1)!=truth(t,2)

@pytest.mark.parametrize('sql',[
 'DELETE FROM orders','DROP TABLE orders',"ATTACH DATABASE '/tmp/test' AS extra",
 "SELECT load_extension('something')","PRAGMA table_info(orders)",
 'SELECT 1','SELECT * FROM sqlite_master','SELECT 1; DROP TABLE orders'])
def test_sql_tools_are_read_only_and_read_source_tables(sql):
    db=connect(make_tasks()[0])
    count=db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]
    with pytest.raises(ToolError):execute_readonly(db,sql)
    assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==count
    db.close()

def test_known_answer_literal_does_not_earn_credit():
    task=make_tasks()[0];env=Environment(task)
    value=truth(task)['current_net_cents']
    sql=f'SELECT {value} + 0*SUM(o.gross_cents)+0*SUM(r.refund_cents) AS current_net_cents FROM orders o CROSS JOIN refunds r'
    row=env.step({'tool':'query_sql','arguments':{'sql':sql}})
    assert row['observation']['rows'][0]['current_net_cents']==value
    assert row['observation']['verified_metrics']==[] and row['process_reward']==0
    env.close()

def test_repeated_checks_do_not_accumulate_progress_reward():
    task=make_tasks()[0];env=Environment(task)
    action={'tool':'query_sql','arguments':{'sql':metric_sql(task,'current_net_cents')}}
    first=env.step(action);second=env.step(action)
    assert first['process_reward']==pytest.approx(.2)
    assert second['process_reward']==0 and second['reward']==pytest.approx(-.01)
    env.close()

def test_failure_at_budget_cancels_partial_progress():
    task=replace(make_tasks()[0],max_turns=2);env=Environment(task)
    env.step({'tool':'query_sql','arguments':{'sql':metric_sql(task,'current_net_cents')}})
    env.step({'tool':'read_policy','arguments':{}})
    assert env.done and not env.success
    assert sum(r['process_reward'] for r in env.records)==pytest.approx(0.)
    assert sum(r['reward'] for r in env.records)==pytest.approx(-.02)
    env.close()

def test_numeric_answer_requires_provenance_and_business_policy():
    task=make_tasks()[0];env=Environment(task)
    result=env.step({'tool':'submit','arguments':{'values':truth(task),'evidence':{},'policy_id':'revenue-v1'}})
    assert not result['success'] and not result['done']
    assert not result['observation']['passed']
    env.close()

def test_error_feedback_is_recoverable_without_answer_leakage():
    env=Environment(make_tasks()[0])
    row=env.step({'tool':'query_sql','arguments':{'sql':'SELECT missing_column FROM orders'}})
    assert row['observation']['recoverable'] and 'error' in row['observation']
    assert 'current_net_cents' not in row['observation']
    env.close()

@pytest.mark.parametrize('text',['{}','[]','{"tool":"read_policy","tool":"submit","arguments":{}}','```json\n{}\n```'])
def test_malformed_actions_are_rejected(text):
    with pytest.raises(ValueError):parse_action(text)

def test_outcome_and_process_modes_have_same_episode_total():
    task=make_tasks()[0]
    a=run_episode(task,ScriptedPolicy(task),'outcome')
    b=run_episode(task,ScriptedPolicy(task),'potential')
    assert a['total_reward']==pytest.approx(b['total_reward'])
    assert [t['reward'] for t in a['turns']]!=[t['reward'] for t in b['turns']]
    assert discounted_returns(b['turns'])[0]==pytest.approx(b['total_reward'])

def test_partial_episode_and_greedy_tokens_are_not_admitted():
    task=make_tasks()[0];episode=run_episode(task,ScriptedPolicy(task))
    with pytest.raises(ValueError):discounted_returns(episode['turns'][:-1])
    with pytest.raises(ValueError):training_samples(episode)
    with pytest.raises(ValueError):assign_turn_credit([episode])

def test_real_lora_update_uses_only_generated_action_tokens():
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import get_peft_model,LoraConfig
    from apt_v2.trainer import prepare_scores,update,response_scores
    torch.set_num_threads(1);torch.manual_seed(2026)
    model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,
        _attn_implementation='eager')),LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
            target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    reference=copy.deepcopy(model).requires_grad_(False)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    episodes=[]
    for i,reward in enumerate((.99,-.01)):
        prompt=[1,2,3]
        with torch.no_grad():
            logps=model(input_ids=torch.tensor([prompt])).logits[0,-1].log_softmax(-1)
            token=torch.multinomial(logps.exp(),1).item()
        generation={'mode':'sample','prompt_token_ids':prompt,'response_token_ids':[token],
            'behavior_token_log_probs':[logps[token].item()]}
        episodes.append({'task_id':'same-task','turns':[{'turn':0,'done':True,'reward':reward,
            'outcome_reward':reward,'generation':generation}]})
        torch.testing.assert_close(response_scores(model,prompt,[token]),logps[token].reshape(1))
    cache=prepare_scores(model,reference,episodes)
    parameters=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=1e-3)
    metrics=update(model,optimizer,episodes,cache)
    assert metrics['updated'] and metrics['grad_norm']>0
    assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)
    changed=copy.deepcopy(episodes);changed[0]['turns'][0]['generation']['response_token_ids']=[31]
    with pytest.raises(ValueError,match='identity'):update(model,optimizer,changed,cache)

@pytest.mark.parametrize('failed_stage',[0,1,2])
def test_cloud_driver_stops_and_packages_stage_failure(tmp_path,monkeypatch,failed_stage):
    import zipfile
    import run_v2_baseline as driver
    from common import file_hash
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/baseline.json').write_text(json.dumps({'max_wall_seconds':1800,'max_job_cost_yuan':10}))
    stub=tmp_path/'stub';stub.write_text('data')
    (tmp_path/'bundle_manifest.json').write_text(json.dumps({'files':{'stub':file_hash(stub)}}))
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'OUTPUT',tmp_path/'artifacts/new')
    monkeypatch.setattr(driver,'preflight',lambda:{'passed':True})
    monkeypatch.setattr(driver,'source_files',lambda:{'stub':file_hash(stub)})
    calls=[]
    def spawn(*args,**kwargs):
        index=len(calls);calls.append(args)
        return SimpleNamespace(wait=lambda **kw:1 if index==failed_stage else 0)
    monkeypatch.setattr(driver.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError):driver.run(download_model=True)
    assert len(calls)==failed_stage+1
    with zipfile.ZipFile(tmp_path/'v2-baseline-review-20261029.zip') as z:
        result=json.loads(z.read('new/cost_and_run.json'))
        assert result['error'] and not result['automatic_retry']

def test_policy_failure_preserves_the_partial_episode():
    task=make_tasks()[0]
    def policy(messages):
        if len(messages)>2:raise RuntimeError('context budget exhausted')
        from apt_v2.runtime import Generation
        return Generation('{"tool":"read_policy","arguments":{}}')
    with pytest.raises(RuntimeError) as error:run_episode(task,policy)
    evidence=error.value.episode_evidence
    assert evidence['status']=='interrupted' and len(evidence['turns'])==1
    assert not evidence['training_admitted']

def test_saved_baseline_replay_detects_modified_rewards(tmp_path,monkeypatch):
    import audit_v2
    import transformers
    from common import file_hash
    task=next(t for t in make_tasks() if t.split=='development')
    episode=run_episode(task,ScriptedPolicy(task))
    texts={}
    for i,row in enumerate(episode['turns'],1):
        text=row['generation']['text'];texts[i]=text
        row['generation'].update(mode='greedy',prompt_token_ids=[1],response_token_ids=[i],
            raw_token_log_probs=[-.5],behavior_token_log_probs=None)
    class Tokenizer:
        def decode(self,ids,**kwargs):return texts[ids[0]]
        def apply_chat_template(self,*args,**kwargs):return [1]
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    output=tmp_path/'output';output.mkdir()
    (tmp_path/'configs').mkdir();(output/'source').mkdir();(output/'tokenizer').mkdir()
    cfg={'split':'development','task_count':1,'reward_mode':'potential','model_id':'model','revision':'abc'}
    def save(path,value):path.write_text(json.dumps(value),encoding='utf8')
    save(tmp_path/'configs/baseline.json',cfg)
    stub=tmp_path/'stub';stub.write_text('source')
    (output/'source/stub').write_bytes(stub.read_bytes())
    save(output/'run_manifest.json',{'config':cfg,'source_sha256':{'stub':file_hash(stub)}})
    tokenizer_path=output/'tokenizer/tokenizer_config.json';save(tokenizer_path,{})
    save(output/'model_identity.json',{'model_id':'model','resolved_revision':'abcdef',
        'files_sha256':{'tokenizer_config.json':file_hash(tokenizer_path)}})
    path=output/'episodes.jsonl';path.write_text(json.dumps(episode)+'\n',encoding='utf8')
    save(output/'summary.json',{'tasks':1,'successes':1,'fault_tasks':0,'fault_tasks_recovered':0,
        'total_tool_errors':0,'total_turns':len(episode['turns']),'response_tokens':len(episode['turns']),
        'episodes_sha256':file_hash(path)})
    monkeypatch.setattr(audit_v2,'ROOT',tmp_path)
    result=audit_v2.audit(output)
    assert result['passed'] and result['tokenizer_checks']
    episode['turns'][0]['reward']+=1
    path.write_text(json.dumps(episode)+'\n',encoding='utf8')
    with pytest.raises(ValueError,match='replay'):audit_v2.audit(output)

def test_cloud_driver_default_uses_cache_without_price(tmp_path,monkeypatch):
    import run_v2_baseline as driver
    from common import file_hash
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/baseline.json').write_text(json.dumps({'max_wall_seconds':1800}))
    cached=tmp_path/'.cache/qwen2.5-1.5b-instruct'
    cached.mkdir(parents=True);(cached/'model_identity.json').write_text('{}')
    stub=tmp_path/'stub';stub.write_text('data')
    (tmp_path/'bundle_manifest.json').write_text(json.dumps({'files':{'stub':file_hash(stub)}}))
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'OUTPUT',tmp_path/'artifacts/new')
    monkeypatch.setattr(driver,'preflight',lambda:{'passed':True})
    monkeypatch.setattr(driver,'source_files',lambda:{'stub':file_hash(stub)})
    calls=[]
    def spawn(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(wait=lambda **kw:0)
    monkeypatch.setattr(driver.subprocess,'Popen',spawn)
    driver.run()
    assert [Path(c[1]).name for c in calls]==['evaluate_v2.py','audit_v2.py']
    status=json.loads((driver.OUTPUT/'cost_and_run.json').read_text())
    assert status['max_job_seconds']==1790 and status['cost_estimation_enabled'] is False
    assert 'hourly_price_yuan' not in status and 'estimated_job_cost_yuan' not in status
    assert status['error'] is None
