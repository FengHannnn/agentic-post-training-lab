import sys,json,copy,io,os,time,threading,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.dpo_recovery_training import optimize_epochs,verify_update_history
from apt_v2.dpo_training import reference_scores

def test_three_epoch_real_lora_keeps_reference_and_first_epoch_matches_one_pass():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    from apt_v2.dpo_recovery_training import optimize_epoch
    torch.manual_seed(231)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    first=copy.deepcopy(model);enc={'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,5,2]};encoded=[copy.deepcopy(enc) for _ in range(169)]
    refs=[copy.deepcopy(reference_scores(model,enc)) for _ in encoded];snapshot=copy.deepcopy(refs)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    first_opt=torch.optim.AdamW([p for p in first.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    optimize_epoch(first,encoded,refs,first_opt,4,51,.1,1)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    captured={}
    def record(row):
        if row['step']==43:captured.update({n:p.detach().clone() for n,p in model.named_parameters()})
    history=optimize_epochs(model,encoded,refs,opt,4,51,.1,1,3,record)
    assert len(history)==129 and history[-1]['pairs_seen']==507 and history[-1]['epoch']==3
    assert [r['pairs_seen'] for r in history if r['epoch_step']==43]==[169,338,507]
    assert all(torch.equal(captured[n],p) for n,p in first.named_parameters())
    assert refs==snapshot and all(int(s['step'])==129 for s in opt.state.values())
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad)
    cfg={'max_steps':129,'pair_count':169,'epochs':3,'gradient_accumulation':4,'seed':51}
    final={'steps':129,'cursor':507,'source_dpo_updates':11,'total_dpo_updates':140}
    train={'optimizer_start':'fresh_adam','optimizer_updates':129,'pair_exposures':507,'frozen_parameter_gradients':0,'history':history}
    state={'steps':129,'cursor':507,'history':history,'optimizer':opt.state_dict()}
    verify_update_history(cfg,final,train,state)
    bad=copy.deepcopy(train);bad['history'][43]['epoch']=1
    with pytest.raises(ValueError,match='schedule'):verify_update_history(cfg,final,bad)

def test_streamed_progress_is_visible_before_child_exits_and_preserves_carriage_returns(tmp_path,monkeypatch):
    import streamed_process
    marker=threading.Event();seen=[]
    class Console(io.StringIO):
        def write(self,text):
            value=super().write(text)
            if '1/2' in text:marker.set()
            return value
    console=Console();monkeypatch.setattr(streamed_process.sys,'stdout',console)
    log=tmp_path/'live.log'
    def observe():
        if marker.wait(5):seen.append(log.read_bytes())
    observer=threading.Thread(target=observe);observer.start()
    command=[sys.executable,'-u','-c',"import sys,time;sys.stdout.write('progress 1/2'+chr(13));sys.stdout.flush();time.sleep(.5);print('progress 2/2 测试',flush=True)"]
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    code=streamed_process.run_logged(command,cwd=tmp_path,env=env,log_path=log,timeout=10);observer.join(5)
    assert code==0 and seen and b'1/2' in seen[0] and b'2/2' not in seen[0]
    assert b'\r' in log.read_bytes() and '测试' in log.read_text(encoding='utf8') and '2/2' in console.getvalue()

def test_streamed_timeout_stops_process_and_preserves_partial_log(tmp_path):
    from streamed_process import run_logged
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    started=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable,'-u','-c',"import time;print('started',flush=True);time.sleep(20)"],cwd=tmp_path,env=env,log_path=tmp_path/'timeout.log',timeout=1)
    assert time.monotonic()-started<10 and 'started' in (tmp_path/'timeout.log').read_text()

def test_extended_runner_failure_keeps_live_output_and_review_archive(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_recovery_train_long as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    model=tmp_path/'.cache/qwen2.5-1.5b-instruct/model_identity.json';model.parent.mkdir(parents=True);model.write_text('{}')
    for name in ('model_identity.json','final/adapter/adapter_model.safetensors'):
        p=tmp_path/'artifacts/initial'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    data=tmp_path/'artifacts/data/pairs.jsonl';data.parent.mkdir(parents=True);data.write_text('fixture')
    cfg={'run_id':'long','initial_run':'initial','data_run':'data','max_wall_seconds':10800}
    (tmp_path/'configs/dpo-recovery-train-v2.json').write_text(json.dumps(cfg))
    (tmp_path/'dpo_recovery_train_v2_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/train_v2_dpo_recovery_long.py').write_text("print('live training progress',flush=True)\nraise RuntimeError('intentional')")
    with pytest.raises(RuntimeError,match='train failed'):runner.run()
    assert 'live training progress' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/long/run_status.json').read_text());assert status['max_wall_seconds']==10800
    import zipfile
    with zipfile.ZipFile(tmp_path/'long-review.zip') as z:assert 'long/train.log' in z.namelist()
