import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
from apt_v2.sft import build_dataset,replay,encode_example,action_loss
from apt_v2.tasks import make_tasks

@pytest.fixture(scope='module')
def dataset():return build_dataset()

def test_train_only_replay_and_repairs(dataset):
    episodes,examples=dataset
    assert len(episodes)==36 and all(e['success'] for e in episodes)
    assert len({e['task_id'] for e in episodes})==12
    assert all(e['split']=='train' and e['source']=='scripted' for e in examples)
    assert {e['variant'] for e in episodes}=={'expert','sql_repair','evidence_repair'}
    by_task={t.id:t for t in make_tasks()}
    for e in episodes:
        replay(by_task[e['task_id']],e)
        admitted={x['turn'] for x in examples if x['task_id']==e['task_id'] and x['variant']==e['variant']}
        assert all(r['turn'] not in admitted for r in e['turns'] if r['tool_error'])
        if e['variant']!='expert':assert e['recovered']
        if e['variant']=='sql_repair':assert any('no such table: metrics' in (r['tool_error'] or '') for r in e['turns'])
    assert any('missing' in json.dumps(x['messages']) for x in examples)

def test_replay_rejects_tampering(dataset):
    e=copy.deepcopy(dataset[0][0]);e['turns'][0]['observation']['policy_id']='tampered'
    with pytest.raises(ValueError,match='replay mismatch'):replay(make_tasks()[0],e)

class Tokenizer:
    eos_token_id=2
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        ids=[1]
        for m in messages:
            ids += [3 if m['role']=='assistant' else 4]+[10+ord(c)%31 for c in m['content']]+[2]
        return ids+[3] if add_generation_prompt else ids

def test_mask_and_overlength(dataset):
    sample=encode_example(Tokenizer(),dataset[1][0],100000)
    p=sample['prompt_length'];assert sample['labels'][:p]==[-100]*p
    assert sample['labels'][p:]==sample['input_ids'][p:] and sample['labels'][-1]==2
    with pytest.raises(ValueError,match='context exceeds'):encode_example(Tokenizer(),dataset[1][0],5)

def test_real_peft_loss_and_save_reload(tmp_path):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig,PeftModel
    torch.manual_seed(42)
    cfg=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
    model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.eval();s={'input_ids':[1,4,5,6,2],'labels':[-100,-100,-100,6,2],'prompt_length':3,'response_length':2}
    full=model(input_ids=torch.tensor([s['input_ids']])).logits[0,2:4].float()
    expected=torch.nn.functional.cross_entropy(full,torch.tensor([6,2]))
    assert torch.allclose(action_loss(model,s),expected,atol=1e-6)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    action_loss(model,s).backward();opt.step();opt.zero_grad(set_to_none=True)
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    model.save_pretrained(tmp_path/'adapter')
    torch.manual_seed(42);loaded=PeftModel.from_pretrained(Qwen2ForCausalLM(cfg),tmp_path/'adapter',is_trainable=True);loaded.eval()
    assert torch.equal(action_loss(model,s),action_loss(loaded,s))
    torch.save(opt.state_dict(),tmp_path/'optimizer.pt')
    restored=torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad],lr=.001)
    restored.load_state_dict(torch.load(tmp_path/'optimizer.pt',weights_only=True))
    action_loss(model,s).backward();opt.step()
    action_loss(loaded,s).backward();restored.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters()))


def test_runner_packages_failed_stage(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_v2_sft as runner
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'.cache/qwen2.5-1.5b-instruct').mkdir(parents=True)
    (tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json').write_text('{}')
    (tmp_path/'configs/sft-pilot-v1.json').write_text(json.dumps({'run_id':'test-run','max_wall_seconds':60}))
    (tmp_path/'sft_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/train_v2_sft.py').write_text('raise RuntimeError("intentional stage failure")')
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/test-run/run_status.json').read_text())
    assert status['stages']==[{'stage':'train','exit_code':1}]
    import zipfile
    with zipfile.ZipFile(tmp_path/'test-run-review.zip') as z:
        assert 'test-run/train.log' in z.namelist() and 'test-run/run_status.json' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()
