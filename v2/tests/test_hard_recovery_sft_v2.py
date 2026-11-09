import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.hard_recovery_sft import recovery_weights,optimize_weighted_sft,verify_weighted_updates,HARD_SCENARIOS
from apt_v2.dpo import pair_id
from apt_v2.tasks import make_tasks
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.recovery_diagnostics import diagnostic_contexts,messages_id,diagnose_action,diagnostic_summary,summarize
from test_dpo_recovery_training_v2 import pair_fixtures
from test_dpo_v2 import CharTokenizer

def weighted_pairs():
    pairs=pair_fixtures()
    names=[s for s in SCENARIOS for _ in range(12)]+['format_repair']*4+['sql_unverified_requery']*12+['stale_evidence_repair']+['repeat_sql_error']*12+['failed_check_repair']*12
    for p,s in zip(pairs[44:],names):
        p.update(scenario=s,context_id=p['task_id']+':'+s);p['pair_id']=pair_id(p)
    return pairs

def test_scenario_weights_preserve_data_and_normalize_global_scale():
    pairs=weighted_pairs();snapshot=copy.deepcopy(pairs);weights,summary=recovery_weights(pairs,4)
    assert pairs==snapshot and summary['hard_examples']==84 and summary['other_examples']==85
    assert sum(weights)/169==pytest.approx(1) and summary['hard_weight_share']==pytest.approx(336/421)
    assert all(w==pytest.approx((4 if p.get('scenario') in HARD_SCENARIOS else 1)/(421/169)) for p,w in zip(pairs,weights))
    assert recovery_weights(pairs,1)[0]==[1.]*169
    with pytest.raises(ValueError,match='positive'):recovery_weights(pairs,float('nan'))

def test_weighted_batch_gradient_matches_hand_computed_objective(monkeypatch):
    import torch
    import apt_v2.sft as sft
    class Scalar(torch.nn.Module):
        def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.25))
    model=Scalar();monkeypatch.setattr(sft,'action_loss',lambda model,sample:(model.value-sample['target'])**2)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    history=optimize_weighted_sft(model,[{'target':0.},{'target':2.}],[.4,1.6],optimizer,2,73,100,1)
    assert history[0]['loss']==pytest.approx(2.4625) and history[0]['grad_norm']==pytest.approx(2.7)
    assert model.value.item()==pytest.approx(.52)
    with pytest.raises(ValueError,match='aligned'):optimize_weighted_sft(model,[{}],[],optimizer,1,1,1,1)

def test_real_lora_258_weighted_updates_keep_frozen_parameters_and_adam_steps():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    from apt_v2.sft import action_loss
    torch.manual_seed(409)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    sample={'input_ids':[1,7,3,4,2],'labels':[-100,-100,3,4,2],'prompt_length':2,'response_length':3}
    encoded=[copy.deepcopy(sample) for _ in range(169)];weights,_=recovery_weights(weighted_pairs(),4)
    before=action_loss(model,sample).item();frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    history=optimize_weighted_sft(model,encoded,weights,optimizer,4,301,1,6)
    assert len(history)==258 and history[-1]['pairs_seen']==1014 and action_loss(model,sample).item()<before
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    cfg={'max_steps':258,'pair_count':169,'epochs':6,'gradient_accumulation':4,'seed':301,'source_sft_updates':255,'source_dpo_updates':11}
    final={'algorithm':'hard_recovery_weighted_sft','steps':258,'cursor':1014,'epochs_completed':6,'source_sft_updates':255,'source_dpo_updates':11,'total_sft_updates':513,'total_dpo_updates':11}
    train={'algorithm':'hard_recovery_weighted_sft','optimizer_start':'fresh_adam','optimizer_updates':258,'pair_exposures':1014,'frozen_parameter_gradients':0,'history':history}
    state={'steps':258,'cursor':1014,'history':history,'optimizer':optimizer.state_dict()}
    verify_weighted_updates(cfg,final,train,state)
    bad=copy.deepcopy(train);bad['history'][43]['epoch']=1
    with pytest.raises(ValueError,match='schedule'):verify_weighted_updates(cfg,final,bad)
    bad=copy.deepcopy(final);bad['total_sft_updates']=512
    with pytest.raises(ValueError,match='ancestry'):verify_weighted_updates(cfg,bad,train)

def test_checkpoint_native_replay_progress_and_two_split_final_diagnostics(tmp_path,monkeypatch):
    import audit_v2_hard_recovery_sft as worker
    import diagnose_v2_recovery_contexts as shared
    from common import file_hash
    tasks=[next(t for t in make_tasks() if t.split==s) for s in ('train','development')]
    monkeypatch.setattr(shared,'make_tasks',lambda:tasks);monkeypatch.setattr(worker,'ROOT',tmp_path)
    cfg={'task_counts':{'train':1,'development':1},'evaluation_branches':['before','after'],'initial_run':'parent','source_sft_updates':255,
        'checkpoint_steps':[43,86,129,172,215],'max_steps':258,'recovery_eval':{'seed':53,'split_seed_offsets':{'train':1000000,'development':0},'samples_per_context':3,'temperature':.8},
        'eval':{'max_new_tokens':100000}}
    contexts=diagnostic_contexts(tasks);tok=CharTokenizer();tm={t.id:t for t in tasks};pairs=weighted_pairs()
    for i,c in enumerate(c for c in contexts if c['split']=='train'):
        pairs[i].update(messages=c['messages'],chosen=c['chosen']);pairs[i]['pair_id']=pair_id(pairs[i])
    seen={messages_id(p['messages']) for p in pairs}
    membership=[{'context_id':c['context_id'],'task_id':c['task_id'],'split':c['split'],'messages_sha256':messages_id(c['messages']),
        'seen_in_training_pairs':messages_id(c['messages']) in seen} for c in contexts]
    shared.jsonl(tmp_path/'contexts.jsonl',contexts);shared.jsonl(tmp_path/'context_membership.jsonl',membership);shared.jsonl(tmp_path/'pairs.jsonl',pairs)
    branches=['before','after',*[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']]]
    middle={};allrows=[]
    for branch in branches:
        rows=[]
        for c in contexts:
            if branch.startswith('checkpoint') and c['split']!='train':continue
            for mode in (('greedy',) if branch.startswith('checkpoint') else ('greedy','sample')):
                for ri in range(1 if mode=='greedy' else 3):
                    text=c['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
                    g={'text':text,'mode':'greedy' if mode=='greedy' else 'dpo_sample','prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),
                        'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),'finish_reason':'stop'}
                    actual,d=diagnose_action(tm[c['task_id']],c,text)
                    rows.append({'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':c['split'],'context_id':c['context_id'],
                        'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':shared.sampling_seed(cfg,tasks,c,ri),
                        'temperature':0 if mode=='greedy' else .8,'generation':g,'row':actual,'diagnostic':d})
        path=tmp_path/(branch+'-actions.jsonl');shared.jsonl(path,rows)
        p=tmp_path/'artifacts/parent/final' if branch=='before' else tmp_path/('final' if branch=='after' else branch)
        p.mkdir(parents=True);(p/'complete.json').write_text(json.dumps({'adapter_sha256':{'adapter_model.safetensors':branch}}))
        if branch.startswith('checkpoint'):
            middle[branch]={'overall':summarize(rows),'by_scenario':{s:summarize([r for r in rows if r['scenario']==s]) for s in SCENARIOS},'actions_sha256':file_hash(path)}
        else:allrows.extend(rows)
    (tmp_path/'checkpoint_diagnostics.json').write_text(json.dumps(middle))
    (tmp_path/'diagnostic_summary.json').write_text(json.dumps(diagnostic_summary(allrows,['before','after'])))
    assert worker.audit_checkpoint_actions(tmp_path,cfg,tok)==35
    evidence=shared.audit_diagnostics(tmp_path,{**cfg,'checkpoints':{'before':{},'after':{}}},tok)
    assert evidence['actions_replayed']==112 and evidence['training_contexts_exactly_in_pairs']==7
    progress=worker.checkpoint_progress(tmp_path,cfg)
    assert [r['new_sft_updates'] for r in progress['checkpoints']]==[0,43,86,129,172,215,258] and progress['selection']=='fixed_final'
    rows[0]['diagnostic']['tool_matches_expert']=False
    path=tmp_path/'checkpoint-0215-actions.jsonl';path.unlink();shared.jsonl(path,rows)
    with pytest.raises(ValueError,match='replay'):worker.audit_checkpoint_actions(tmp_path,cfg,tok)

def test_complete_task_evaluation_audit_and_seed_tamper(tmp_path,monkeypatch):
    import audit_v2_hard_recovery_sft as worker
    from test_dpo_independent_v2 import fixtures
    from apt_v2.sft_diagnostics import summarize_episodes
    from common import file_hash
    tasks=[next(t for t in make_tasks() if t.split==s) for s in ('train','development')];monkeypatch.setattr(worker,'make_tasks',lambda:tasks)
    cfg={'evaluation_branches':['before','after'],'eval':{'splits':{'train':1,'development':1}},'sample_eval':{'seed':52,'task_count':1,'rollouts_per_task':2,'temperature':.8}}
    comparison={};stochastic={}
    for branch in cfg['evaluation_branches']:
        comparison[branch]={}
        for split in ('train','development'):
            episodes=fixtures([t for t in tasks if t.split==split],branch);p=tmp_path/(branch+'-'+split+'-episodes.jsonl');p.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
            s={**summarize_episodes(episodes),'episodes_sha256':file_hash(p)};comparison[branch][split]=s;(tmp_path/(branch+'-'+split+'-summary.json')).write_text(json.dumps(s))
        dev=[t for t in tasks if t.split=='development'];episodes=fixtures(dev,branch,cfg['sample_eval']);p=tmp_path/(branch+'-sample-episodes.jsonl');p.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
        s={**summarize_episodes(episodes),'episodes_sha256':file_hash(p),'successes_by_task':{dev[0].id:2}};stochastic[branch]=s
        (tmp_path/(branch+'-sample-summary.json')).write_text(json.dumps(s))
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparison));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(stochastic))
    result=worker.audit_task_evaluations(tmp_path,cfg,CharTokenizer())
    assert result['episodes_replayed']==8 and result['paired_development_sample']['both_success']==2
    episodes[0]['sample_seed']+=1;p.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    with pytest.raises(ValueError,match='identity'):worker.audit_task_evaluations(tmp_path,cfg,CharTokenizer())

def test_runner_keeps_streamed_failure_log_and_final_adapter_only(tmp_path,monkeypatch,capsys):
    import run_v2_hard_recovery_sft as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    required={'parent':('model_identity.json','audit.json','final/complete.json','final/adapter/adapter_model.safetensors','after_chosen_scores.json'),
        'data':('audit.json','pairs.jsonl'),'diagnostic':('audit.json','config.json','input_binding.json')}
    for source,names in required.items():
        for name in names:
            p=tmp_path/'artifacts'/source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    cfg={'run_id':'weighted','initial_run':'parent','data_run':'data','diagnostic_run':'diagnostic','max_wall_seconds':14400}
    (tmp_path/'configs/hard-recovery-sft-weighted-v1.json').write_text(json.dumps(cfg));(tmp_path/'hard_recovery_sft_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/train_v2_hard_recovery_sft.py').write_text("from pathlib import Path\np=Path('artifacts/weighted/final/adapter');p.mkdir(parents=True);(p/'adapter_model.safetensors').write_bytes(b'final')\np=Path('artifacts/weighted/checkpoint-0043/adapter');p.mkdir(parents=True);(p/'adapter_model.safetensors').write_bytes(b'intermediate')\nprint('weighted live progress',flush=True)\nraise RuntimeError('intentional')")
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    assert 'weighted live progress' in capsys.readouterr().out
    import zipfile
    with zipfile.ZipFile(tmp_path/'weighted-review.zip') as z:
        assert 'weighted/train.log' in z.namelist() and 'weighted/final/adapter/adapter_model.safetensors' in z.namelist()
        assert 'weighted/checkpoint-0043/adapter/adapter_model.safetensors' not in z.namelist()

def test_fixed_config_ancestry_schedule_and_probe_budget():
    cfg=json.loads((ROOT/'configs/hard-recovery-sft-weighted-v1.json').read_text())
    assert cfg['adapter_sha256']=='078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55'
    assert cfg['source_sft_updates']==255 and cfg['source_dpo_updates']==11 and cfg['max_steps']==6*43 and cfg['hard_weight']==4
    assert cfg['learning_rate']==5e-6 and cfg['checkpoint_selection']=='fixed_final' and cfg['max_wall_seconds']==14400
    assert len(cfg['checkpoint_steps'])*84+2*112*4==1316 and 2*(12+4+16)==64
