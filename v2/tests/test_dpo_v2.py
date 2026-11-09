import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from apt_v2.dpo import verified_pairs,encode_pair,balanced_pairs,dpo_loss,response_logps
from apt_v2.runtime import ScriptedPolicy,Generation,parse_action,run_episode
from apt_v2.tasks import make_tasks

class CharTokenizer:
    eos_token_id=0
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        ids=[]
        for m in messages:
            ids += [ord(c)+1 for c in m['role']+':'+m['content']]+[0]
        if add_generation_prompt:ids += [ord(c)+1 for c in 'assistant:']
        return ids
    def decode(self,ids,**kwargs):return ''.join(chr(i-1) for i in ids if i)

def rollout(task,error=None):
    tok=CharTokenizer();expert=ScriptedPolicy(task);injected=False
    def policy(messages):
        nonlocal injected
        if len(messages)>2:
            obs=json.loads(messages[-1]['content'])['observation']
            if 'rows' in obs and not obs.get('verified_metrics'):expert.phase-=1
        sample=expert(messages);action=parse_action(sample.text)
        if error and not injected and action['tool']=='query_sql':
            injected=True
            if error=='table':action['arguments']['sql']='SELECT current_net_cents FROM metrics'
            elif error=='semantic':action['arguments']['sql']='SELECT SUM(gross_cents) AS current_net_cents FROM orders'
            elif error=='json':return make('invalid JSON',messages)
        return make(json.dumps(action),messages)
    def make(text,messages):
        ids=[ord(c)+1 for c in text]+[0]
        return Generation(text,'dpo_sample',tok.apply_chat_template(messages,add_generation_prompt=True),ids,[-.1]*len(ids),None,'stop')
    episode=run_episode(task,policy);episode['rollout_id']=task.id+'-sample-00'
    return episode,tok

def test_counterfactual_pairs_share_exact_context_and_verify_progress():
    task=make_tasks()[0];episode,tok=rollout(task,'table')
    pairs=verified_pairs(task,episode,tok,100000,2)
    assert episode['success'] and len(pairs)==1
    pair=pairs[0];assert pair['reason']=='tool_error'
    assert pair['chosen_observation']['verified_metrics']==['current_net_cents']
    assert 'no such table' in pair['rejected_observation']['error']
    encoded=encode_pair(tok,pair,100000)
    assert encoded['prompt']==pair['native_prompt_token_ids']
    assert encoded['rejected']==pair['native_rejected_token_ids']
    assert encoded['chosen'][-1]==encoded['rejected'][-1]==tok.eos_token_id
    assert pair['chosen_source']=='scripted_counterfactual' and pair['rejected_source']=='frozen_sft_sample'

def test_semantically_wrong_sql_is_rejected_even_when_sql_executes():
    task=make_tasks()[0];episode,tok=rollout(task,'semantic')
    pair=verified_pairs(task,episode,tok,100000)[0]
    assert pair['reason']=='sql_unverified' and pair['rejected_observation']['verified_metrics']==[]
    assert pair['chosen_observation']['verified_metrics']==['current_net_cents']

def test_transient_service_fault_is_not_preference_error():
    episode,tok=rollout(make_tasks()[1])
    assert episode['recovered']
    assert verified_pairs(make_tasks()[1],episode,tok,100000)==[]

def test_cross_context_leakage_and_duplicate_caps_rejected():
    task=make_tasks()[0];episode,tok=rollout(task,'table');pair=verified_pairs(task,episode,tok,100000)[0]
    bad=copy.deepcopy(pair);bad['native_prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='contexts differ'):encode_pair(tok,bad,100000)
    assert len(balanced_pairs([pair,pair],4))==1
    bad=copy.deepcopy(pair);bad['split']='development'
    with pytest.raises(ValueError,match='split'):balanced_pairs([bad],4)
    with pytest.raises(ValueError,match='train task'):verified_pairs(make_tasks()[12],episode,tok,100000)

def test_malformed_actions_valid_negative_and_sample_tamper_detected():
    task=make_tasks()[0];episode,tok=rollout(task,'json')
    pair=verified_pairs(task,episode,tok,100000)[0]
    assert pair['rejected']=='invalid JSON'
    bad=copy.deepcopy(episode);bad['turns'][0]['generation']['prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='prompt differs'):verified_pairs(task,bad,tok,100000)
    bad=copy.deepcopy(episode);bad['turns'][0]['reward']+=1
    with pytest.raises(ValueError,match='reward'):verified_pairs(task,bad,tok,100000)
    assert verified_pairs(task,episode,tok,20)==[]

def test_dpo_formula_gradient_directions_and_frozen_reference():
    import torch
    chosen=torch.tensor(-2.,requires_grad=True);rejected=torch.tensor(-3.,requires_grad=True)
    rc=torch.tensor(-2.);rr=torch.tensor(-3.)
    loss=dpo_loss(chosen,rejected,rc,rr,.1)
    assert loss.item()==pytest.approx(math.log(2))
    loss.backward();assert chosen.grad<0 and rejected.grad>0
    assert dpo_loss(torch.tensor(-1.),rejected.detach(),rc,rr,.1).item()<loss.item()
    with pytest.raises(ValueError,match='reference'):dpo_loss(chosen,rejected,rc.requires_grad_(True),rr)

def test_real_qwen_dpo_update_improves_pair_margin_and_preserves_reference():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(93)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    prompt=[1,7];chosen=[3,4,2];rejected=[3,5,2]
    rc=response_logps(reference,prompt,chosen).sum();rr=response_logps(reference,prompt,rejected).sum()
    full=model(input_ids=torch.tensor([prompt+chosen])).logits[0,len(prompt)-1:-1].float().log_softmax(-1)
    expected=full.gather(-1,torch.tensor(chosen)[:,None]).squeeze(-1)
    assert torch.allclose(response_logps(model,prompt,chosen),expected,atol=1e-6)
    before=response_logps(model,prompt,chosen).sum()-response_logps(model,prompt,rejected).sum()
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    loss=dpo_loss(response_logps(model,prompt,chosen,True).sum(),response_logps(model,prompt,rejected,True).sum(),rc,rr)
    loss.backward();optimizer.step()
    after=response_logps(model,prompt,chosen).sum()-response_logps(model,prompt,rejected).sum()
    assert after>before
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    assert all(p.grad is None for p in reference.parameters())
    assert torch.equal(response_logps(reference,prompt,chosen).sum(),rc)


def test_preference_build_stage_replays_rollouts_and_writes_audit(tmp_path,monkeypatch):
    import collect_v2_dpo as worker
    import transformers
    from dataclasses import asdict
    from apt_v2.sft_diagnostics import summarize_episodes
    tasks=make_tasks()[:2]
    cfg={'seed':91,'rollouts_per_task':2,'temperature':.8,'adapter_sha256':'pinned',
        'pair_max_length':100000,'max_pairs_per_episode':2,'max_pairs_per_task':4,
        'min_pairs_for_training':1,'min_task_coverage_for_training':1}
    monkeypatch.setattr(worker,'verify_inputs',lambda output:(cfg,None))
    monkeypatch.setattr(worker,'make_tasks',lambda:tasks)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:CharTokenizer())
    (tmp_path/'tasks.json').write_text(json.dumps([asdict(t) for t in tasks]))
    episodes=[]
    for ti,task in enumerate(tasks):
        for ri in range(2):
            e,_=rollout(task,'table' if ri==0 else None);seed=91+ti*10000+ri*100
            e.update(rollout_index=ri,rollout_id=f'{task.id}-sample-{ri:02}',sample_seed=seed,temperature=.8,adapter_sha256='pinned')
            for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
            episodes.append(e)
    (tmp_path/'rollouts.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    (tmp_path/'rollout_summary.json').write_text(json.dumps(summarize_episodes(episodes)))
    worker.build(tmp_path)
    assert json.loads((tmp_path/'audit.json').read_text())['episodes_replayed']==4
    summary=json.loads((tmp_path/'pair_summary.json').read_text())
    assert summary['pairs']==1 and summary['task_coverage']==1 and summary['ready_for_training_review']

def test_dpo_runner_failure_keeps_logs_and_no_training(tmp_path,monkeypatch):
    import run_v2_dpo_collect as runner
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    source=tmp_path/'artifacts/source';(source/'final/adapter').mkdir(parents=True)
    for name in ('examples.jsonl','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        (source/name).write_text('fixture')
    (tmp_path/'configs/dpo-collect-v1.json').write_text(json.dumps({'run_id':'collection','source_run':'source','max_wall_seconds':60}))
    (tmp_path/'dpo_collect_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/collect_v2_dpo.py').write_text('raise RuntimeError("intentional failure")')
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/collection/run_status.json').read_text())
    assert status['optimizer_updates']==0 and status['training_enabled'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'collection-review.zip') as z:assert 'collection/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()
