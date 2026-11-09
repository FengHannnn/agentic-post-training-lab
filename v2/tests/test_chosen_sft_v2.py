import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.chosen_sft import chosen_examples,encode_chosen,optimize_sft,verify_sft_updates
from apt_v2.dpo import pair_id
from apt_v2.tasks import make_tasks
from apt_v2.dpo_recovery import SCENARIOS,build_context
from apt_v2.dpo_recovery_training import recovery_result,summarize_recovery
from test_dpo_recovery_training_v2 import pair_fixtures
from test_dpo_v2 import CharTokenizer

def test_projection_and_encoding_ignore_rejected_and_mask_prompt():
    pairs=pair_fixtures();examples=chosen_examples(pairs);encoded=encode_chosen(CharTokenizer(),examples,100000)
    assert len(examples)==169 and [e['source_pair_id'] for e in examples]==[p['pair_id'] for p in pairs]
    assert all('rejected' not in e for e in examples)
    for sample in encoded:
        n=sample['prompt_length'];assert sample['labels'][:n]==[-100]*n and sample['labels'][n:]==sample['input_ids'][n:]
    changed=copy.deepcopy(pairs)
    for i,p in enumerate(changed):p['rejected']='different rejected '+str(i);p['pair_id']=pair_id(p)
    assert encode_chosen(CharTokenizer(),chosen_examples(changed),100000)==encoded

def test_real_chosen_only_lora_129_updates_reduce_cross_entropy():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    from apt_v2.sft import action_loss
    torch.manual_seed(331)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    sample={'input_ids':[1,7,3,4,2],'labels':[-100,-100,3,4,2],'prompt_length':2,'response_length':3}
    before=action_loss(model,sample).item()
    logits=model(input_ids=torch.tensor([sample['input_ids']])).logits[0,1:-1].float()
    expected=torch.nn.functional.cross_entropy(logits,torch.tensor([3,4,2]))
    assert action_loss(model,sample).item()==pytest.approx(expected.item(),abs=1e-6)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    history=optimize_sft(model,[copy.deepcopy(sample) for _ in range(169)],opt,4,51,1,3)
    assert len(history)==129 and history[-1]['pairs_seen']==507 and action_loss(model,sample).item()<before
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    cfg={'max_steps':129,'pair_count':169,'epochs':3,'gradient_accumulation':4,'seed':51}
    final={'algorithm':'chosen_only_sft','steps':129,'cursor':507,'source_dpo_updates':11,'sft_updates':129,'epochs_completed':3}
    train={'algorithm':'chosen_only_sft','optimizer_start':'fresh_adam','optimizer_updates':129,'pair_exposures':507,'frozen_parameter_gradients':0,'history':history}
    state={'steps':129,'cursor':507,'history':history,'optimizer':opt.state_dict()};verify_sft_updates(cfg,final,train,state)
    bad=copy.deepcopy(train);bad['history'][43]['epoch']=1
    with pytest.raises(ValueError,match='schedule'):verify_sft_updates(cfg,final,bad)

def test_same_run_three_model_task_and_recovery_audit(tmp_path):
    import audit_v2_dpo_recovery as worker
    from common import file_hash
    from test_dpo_independent_v2 import fixtures
    from apt_v2.sft_diagnostics import summarize_episodes
    cfg={'evaluation_branches':['before','dpo','after'],'eval':{'splits':{'train':12,'development':4}},'sample_eval':{'seed':52,'task_count':4,'rollouts_per_task':4,'temperature':.8},
        'recovery_eval':{'seed':53,'split':'development','task_count':4,'samples_per_context':3,'temperature':.8}}
    tok=CharTokenizer();tasks=make_tasks();dev=[t for t in tasks if t.split=='development'];tm={t.id:t for t in tasks}
    def jsonl(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    comparison={};sample_comparison={}
    for branch in ('before','dpo','after'):
        comparison[branch]={}
        for split in ('train','development'):
            episodes=fixtures([t for t in tasks if t.split==split],branch);path=tmp_path/(branch+'-'+split+'-episodes.jsonl');jsonl(path,episodes)
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)};comparison[branch][split]=summary
            (tmp_path/(branch+'-'+split+'-summary.json')).write_text(json.dumps(summary))
        episodes=fixtures(dev,branch,cfg['sample_eval']);path=tmp_path/(branch+'-sample-episodes.jsonl');jsonl(path,episodes)
        summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),'successes_by_task':{t.id:4 for t in dev}}
        sample_comparison[branch]=summary;(tmp_path/(branch+'-sample-summary.json')).write_text(json.dumps(summary))
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparison));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(sample_comparison))
    contexts=[build_context(t,s,allowed_split='development') for t in dev for s in SCENARIOS];jsonl(tmp_path/'recovery_contexts.jsonl',contexts)
    recovery={}
    for branch in ('before','dpo','after'):
        rows=[]
        for c in contexts:
            for mode in ('greedy','sample'):
                for ri in range(1 if mode=='greedy' else 3):
                    text=c['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
                    generation={'mode':'greedy' if mode=='greedy' else 'dpo_sample','prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),
                        'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),'text':text,'finish_reason':'stop'}
                    seed=53+[t.id for t in dev].index(c['task_id'])*10000+list(SCENARIOS).index(c['scenario'])*100+ri
                    rows.append({'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':'development','context_id':c['context_id'],
                        'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,'temperature':0 if mode=='greedy' else .8,
                        'generation':generation,'result':recovery_result(tm[c['task_id']],c,generation,tok,generation['mode'])})
        path=tmp_path/(branch+'-recovery-actions.jsonl');jsonl(path,rows)
        recovery[branch]={mode:{'overall':summarize_recovery([r for r in rows if r['mode']==mode]),'by_scenario':{s:summarize_recovery([r for r in rows if r['mode']==mode and r['scenario']==s]) for s in SCENARIOS}} for mode in ('greedy','sample')}
        recovery[branch]['actions_sha256']=file_hash(path)
    (tmp_path/'recovery_comparison.json').write_text(json.dumps(recovery))
    evidence=worker.audit_evaluations(tmp_path,cfg,tok)
    assert evidence['episodes_replayed']==96 and evidence['recovery_actions_replayed']==336 and evidence['development_recovery_contexts']==28
    assert evidence['paired_dpo_vs_sft_development_sample']['both_success']==16
    rows[0]['sample_seed']+=1;jsonl(tmp_path/'after-recovery-actions.jsonl',rows)
    with pytest.raises(ValueError,match='identity'):worker.audit_evaluations(tmp_path,cfg,tok)


def test_chosen_runner_failure_preserves_live_progress_and_archive(tmp_path,monkeypatch,capsys):
    import run_v2_chosen_sft as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    model=tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json';model.parent.mkdir(parents=True);model.write_text('{}')
    for name in ('model_identity.json','final/adapter/adapter_model.safetensors'):
        p=tmp_path/'artifacts/initial'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    for name in ('final/adapter/adapter_model.safetensors','final/complete.json','model_identity.json','reference_scores.json'):
        p=tmp_path/'artifacts/comparison'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    data=tmp_path/'artifacts/data/pairs.jsonl';data.parent.mkdir(parents=True);data.write_text('fixture')
    cfg={'run_id':'chosen','initial_run':'initial','comparison_run':'comparison','data_run':'data','max_wall_seconds':10800}
    (tmp_path/'configs/chosen-only-sft-v1.json').write_text(json.dumps(cfg));(tmp_path/'chosen_sft_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/train_v2_chosen_sft.py').write_text("print('SFT live progress',flush=True)\nraise RuntimeError('intentional')")
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    assert 'SFT live progress' in capsys.readouterr().out
    import zipfile
    with zipfile.ZipFile(tmp_path/'chosen-review.zip') as z:assert 'chosen/train.log' in z.namelist()

def test_control_configuration_and_three_pinned_checkpoints(tmp_path,monkeypatch):
    import evaluate_v2_dpo_recovery as evaluator
    cfg=json.loads((ROOT/'configs/chosen-only-sft-v1.json').read_text())
    dpo=json.loads((ROOT/'configs/dpo-recovery-train-v2.json').read_text())
    assert 'beta' not in cfg and cfg['algorithm']=='chosen_only_sft'
    for key in ('initial_run','adapter_sha256','pairs_sha256','learning_rate','epochs','max_steps','seed','eval','sample_eval','recovery_eval','max_wall_seconds'):assert cfg[key]==dpo[key]
    monkeypatch.setattr(evaluator,'ROOT',tmp_path)
    paths=evaluator.checkpoints(cfg,tmp_path/'output')
    assert list(paths)==['before','dpo','after'] and paths['dpo']==tmp_path/'artifacts'/cfg['comparison_run']/'final'
