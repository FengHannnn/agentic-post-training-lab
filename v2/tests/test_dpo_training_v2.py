import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.dpo_training import reference_scores,pair_objective,input_id

def test_cached_reference_stays_fixed_during_real_dpo_update():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(29)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    encoded={'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,5,2]}
    cache=reference_scores(model,encoded);saved=copy.deepcopy(cache)
    before,metrics=pair_objective(model,encoded,cache,.1,True)
    assert before.item()==pytest.approx(math.log(2),abs=1e-6) and metrics['relative_margin']==pytest.approx(0)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    before.backward();opt.step()
    _,after=pair_objective(model,encoded,cache,.1)
    assert after['relative_margin']>0 and cache==saved
    bad=copy.deepcopy(encoded);bad['prompt']=[1,8]
    with pytest.raises(ValueError,match='binding'):pair_objective(model,bad,cache,.1)

def test_complete_greedy_and_stochastic_audit(tmp_path,monkeypatch):
    import audit_v2_dpo as worker
    import transformers
    from test_dpo_v2 import rollout,CharTokenizer
    from apt_v2.tasks import make_tasks
    from apt_v2.dpo import verified_pairs,encode_pair
    from apt_v2.sft_diagnostics import summarize_episodes
    from common import file_hash
    tasks=[make_tasks()[0],make_tasks()[12]];train=tasks[0]
    cfg={'parent_run':'source','eval':{'splits':{'train':1,'development':1}},
        'sample_eval':{'seed':71,'temperature':.8,'rollouts_per_task':2},'max_length':100000}
    source=tmp_path/'artifacts/source/final/adapter';source.mkdir(parents=True)
    (source/'adapter_model.safetensors').write_bytes(b'frozen')
    cfg['adapter_sha256']=file_hash(source/'adapter_model.safetensors')
    (tmp_path/'config.json').write_text(json.dumps(cfg));(tmp_path/'run_manifest.json').write_text('{"source_sha256":{}}')
    (tmp_path/'model_identity.json').write_text('{"files_sha256":{}}')
    monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(worker,'make_tasks',lambda:tasks)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:CharTokenizer())
    greedy={};sampled={}
    for branch in ('sft','dpo'):
        greedy[branch]={}
        for task in tasks:
            e,_=rollout(task);e['model_branch']=branch
            for row in e['turns']:row['generation']['mode']='greedy'
            path=tmp_path/(branch+'-'+task.split+'-episodes.jsonl');path.write_text(json.dumps(e)+'\n')
            summary={**summarize_episodes([e]),'episodes_sha256':file_hash(path)}
            (tmp_path/(branch+'-'+task.split+'-summary.json')).write_text(json.dumps(summary));greedy[branch][task.split]=summary
        episodes=[]
        for ri in range(2):
            e,_=rollout(train);seed=71+ri*100;e.update(model_branch=branch,rollout_index=ri,sample_seed=seed,temperature=.8)
            for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
            episodes.append(e)
        path=tmp_path/(branch+'-sample-episodes.jsonl');path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
        sampled[branch]={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),'successes_by_task':{train.id:2}}
        (tmp_path/(branch+'-sample-summary.json')).write_text(json.dumps(sampled[branch]))
    (tmp_path/'comparison_summary.json').write_text(json.dumps(greedy));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(sampled))
    bad_episode,tok=rollout(train,'table');pair=verified_pairs(train,bad_episode,tok,100000)[0]
    path=tmp_path/'pairs.jsonl';path.write_text(json.dumps(pair)+'\n');cfg['pairs_sha256']=file_hash(path)
    (tmp_path/'config.json').write_text(json.dumps(cfg));enc=encode_pair(tok,pair,100000)
    ref={'input_id':input_id(enc),'chosen':[-.2]*len(enc['chosen']),'rejected':[-.2]*len(enc['rejected'])}
    (tmp_path/'reference_scores.json').write_text(json.dumps({'baseline_adapter_sha256':cfg['adapter_sha256'],'pair_ids':[pair['pair_id']],'scores':[ref]}))
    for name in ('before_pair_scores.json','after_pair_scores.json'):
        (tmp_path/name).write_text(json.dumps({'metrics':[{'loss':math.log(2),'relative_margin':0.}], 'means':{'loss':math.log(2),'relative_margin':0.}}))
    final=tmp_path/'final';(final/'adapter').mkdir(parents=True);(final/'adapter/adapter_model.safetensors').write_bytes(b'candidate')
    (final/'state.pt').write_bytes(b'state')
    (final/'complete.json').write_text(json.dumps({'steps':11,'adapter_sha256':{'adapter_model.safetensors':file_hash(final/'adapter/adapter_model.safetensors')},'state_sha256':file_hash(final/'state.pt')}))
    (tmp_path/'training_summary.json').write_text('{"optimizer_updates":11,"optimizer_start":"fresh_adam"}')
    worker.audit(tmp_path)
    assert json.loads((tmp_path/'audit.json').read_text())['episodes_replayed']==8

def test_dpo_runner_stops_failed_stage_and_keeps_final_adapter_only(tmp_path,monkeypatch):
    import run_v2_dpo as runner
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'.cache/qwen2.5-1.5b-instruct').mkdir(parents=True)
    (tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json').write_text('{}')
    parent=tmp_path/'artifacts/parent';(parent/'final/adapter').mkdir(parents=True)
    (parent/'model_identity.json').write_text('{}');(parent/'final/adapter/adapter_model.safetensors').write_bytes(b'base')
    data=tmp_path/'artifacts/data';data.mkdir();(data/'pairs.jsonl').write_text('fixture')
    (tmp_path/'configs/dpo-train-v1.json').write_text(json.dumps({'run_id':'dpo','parent_run':'parent','data_run':'data','max_wall_seconds':60}))
    (tmp_path/'dpo_train_release_manifest.json').write_text('{"files":{}}')
    code="import sys;from pathlib import Path;p=Path(sys.argv[sys.argv.index('--output')+1]);"
    code+="(p/'final/adapter').mkdir(parents=True);(p/'final/adapter/adapter_model.safetensors').write_bytes(b'candidate');"
    code+="(p/'state.pt').write_bytes(b'optimizer');(p/'old.safetensors').write_bytes(b'old');raise RuntimeError('intentional failure')"
    (tmp_path/'scripts/train_v2_dpo.py').write_text(code)
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    import zipfile
    with zipfile.ZipFile(tmp_path/'dpo-review.zip') as z:
        assert 'dpo/final/adapter/adapter_model.safetensors' in z.namelist()
        assert not any(n.endswith('.pt') or n.endswith('old.safetensors') for n in z.namelist())
