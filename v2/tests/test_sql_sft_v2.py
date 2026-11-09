import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.sft import build_dataset,action_loss
from apt_v2.sql_sft import normalized_weights,data_identity,validate_examples,audit_episode,backward_batch
from apt_v2.runtime import ScriptedPolicy,Generation,run_episode,parse_action
from apt_v2.tasks import make_tasks

@pytest.fixture(scope='module')
def dataset():return build_dataset()

def test_weights_preserve_data_and_global_scale(dataset):
    examples=dataset[1];before=copy.deepcopy(examples);weights=normalized_weights(examples,4)
    assert len(weights)==252 and sum(weights)/len(weights)==pytest.approx(1)
    sql=[w for e,w in zip(examples,weights) if parse_action(e['response'])['tool']=='query_sql']
    other=[w for e,w in zip(examples,weights) if parse_action(e['response'])['tool']!='query_sql']
    assert len(sql)==72 and len(other)==180 and sql[0]/other[0]==4
    assert examples==before and normalized_weights(examples,1)==[1.]*252
    for invalid in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):normalized_weights(examples,invalid)

def test_dataset_identity_rejects_changes_and_leakage(dataset):
    examples=dataset[1];summary={'dataset_sha256':data_identity(examples)}
    assert validate_examples(examples,summary)['query_sql']==72
    changed=copy.deepcopy(examples);changed[0]['split']='development'
    with pytest.raises(ValueError):validate_examples(changed,summary)
    changed=copy.deepcopy(examples);changed[0]['response']='{}'
    with pytest.raises(ValueError):validate_examples(changed,summary)

class CharTokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):
        text=''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:'
        return [ord(c)+1 for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i-1) for i in ids if i)

def recorded_episode(task,scripted=True):
    tok=CharTokenizer();expert=ScriptedPolicy(task)
    def policy(messages):
        text=expert(messages).text if scripted else json.dumps({'tool':'calculate_change','arguments':{}})
        ids=[ord(c)+1 for c in text]+[0]
        return Generation(text,'greedy',tok.apply_chat_template(messages),ids,[-.2]*len(ids),None,'stop')
    return run_episode(task,policy),tok

def test_replay_accepts_success_and_budget_failure():
    for scripted in (True,False):
        task=make_tasks()[1];episode,tok=recorded_episode(task,scripted)
        assert audit_episode(task,episode,tok)==len(episode['turns'])
        assert episode['success']==scripted

def test_replay_detects_generation_and_observation_tampering():
    task=make_tasks()[0];episode,tok=recorded_episode(task)
    bad=copy.deepcopy(episode);bad['turns'][0]['generation']['prompt_token_ids']=[3]
    with pytest.raises(ValueError,match='prompt differs'):audit_episode(task,bad,tok)
    bad=copy.deepcopy(episode);bad['turns'][0]['observation']['policy_id']='changed'
    with pytest.raises(ValueError,match='observation'):audit_episode(task,bad,tok)

def test_weighted_backward_matches_manual_objective():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(71)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    samples=[{'input_ids':[1,3,4,2],'labels':[-100,-100,4,2],'prompt_length':2,'response_length':2},
        {'input_ids':[1,5,6,2],'labels':[-100,-100,6,2],'prompt_length':2,'response_length':2}]
    weights=[1.6,.4];params=[p for p in model.parameters() if p.requires_grad]
    value=backward_batch(model,samples,weights,[0,1]);actual=[p.grad.clone() for p in params]
    model.zero_grad(set_to_none=True)
    expected=(action_loss(model,samples[0])*1.6+action_loss(model,samples[1])*.4)/2
    expected.backward()
    assert value==pytest.approx(expected.item(),abs=1e-6)
    assert all(torch.allclose(a,p.grad,atol=1e-7,rtol=1e-5) for a,p in zip(actual,params))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    with pytest.raises(ValueError,match='nonempty'):backward_batch(model,samples,weights,[])


def test_sql_runner_stops_after_failed_evaluation_and_packages_evidence(tmp_path,monkeypatch):
    import run_v2_sql_sft as runner
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'.cache/qwen2.5-1.5b-instruct').mkdir(parents=True)
    (tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json').write_text('{}')
    parent=tmp_path/'artifacts/pilot';(parent/'final/adapter').mkdir(parents=True)
    for name in ('examples.jsonl','demonstrations.jsonl','final/state.pt','final/adapter/adapter_model.safetensors'):
        (parent/name).write_text('fixture')
    (tmp_path/'configs/sql-sft-weighted-v1.json').write_text(json.dumps({'run_id':'weighted','parent_run':'pilot','max_wall_seconds':60}))
    (tmp_path/'sql_sft_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/train_v2_sql_sft.py').write_text('print("train stage complete")')
    (tmp_path/'scripts/evaluate_v2_sql_sft.py').write_text('raise RuntimeError("intentional eval failure")')
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    with pytest.raises(RuntimeError,match='evaluate failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/weighted/run_status.json').read_text())
    assert status['stages']==[{'stage':'train','exit_code':0},{'stage':'evaluate','exit_code':1}]
    assert not (tmp_path/'artifacts/weighted/audit.log').exists()
    import zipfile
    with zipfile.ZipFile(tmp_path/'weighted-review.zip') as z:assert 'weighted/evaluate.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()
