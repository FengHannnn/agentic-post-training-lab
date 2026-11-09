import argparse,importlib.util,json,sys,zipfile,subprocess
from pathlib import Path
import pytest
PATH=Path(__file__).resolve().parents[1]/'check_environment.py'
spec=importlib.util.spec_from_file_location('check_environment_4090d',PATH)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

def args(**changes):
    values=dict(max_seconds=900.,hourly_price=None,download_model=False,worker=None,output=None)
    values.update(changes);return argparse.Namespace(**values)

def test_verdict_distinguishes_inference_from_training():
    stages={k:{'passed':True} for k in ('inventory','model','vllm')}
    stages['train']={'passed':False,'error_type':'OutOfMemoryError'}
    result=check.verdict(stages)
    assert result['v2_baseline_ready'] and not result['short_context_lora_ready'] and not result['all_checks_passed']
    stages['train']={'passed':True}
    assert check.verdict(stages)['all_checks_passed']

@pytest.mark.parametrize('seconds,price',[(0,None),(float('nan'),None),(1,None),(900,0),(900,float('nan'))])
def test_invalid_time_or_cost_budget_is_rejected(seconds,price):
    with pytest.raises(ValueError):check.run(args(max_seconds=seconds,hourly_price=price))

def test_worker_preserves_exception_in_json(tmp_path,monkeypatch):
    def fail(output):raise RuntimeError('CUDA kernel failed')
    monkeypatch.setattr(check,'inventory',fail)
    code=check.worker(args(worker='inventory',output=tmp_path))
    saved=check.read(tmp_path/'stage_inventory.json')
    assert code==1 and saved['error_type']=='RuntimeError' and 'CUDA kernel failed' in saved['traceback']

def test_parent_keeps_inference_probe_after_training_oom(tmp_path,monkeypatch):
    monkeypatch.setattr(check,'ROOT',tmp_path)
    calls=[]
    def spawn(command,**kwargs):
        stage=command[command.index('--worker')+1]
        output=Path(command[command.index('--output')+1])
        calls.append(stage)
        check.save(output/('stage_'+stage+'.json'),{'passed':stage!='train'})
        return argparse.Namespace(wait=lambda **kw:0 if stage!='train' else 1)
    monkeypatch.setattr(check.subprocess,'Popen',spawn)
    assert check.run(args())==2
    assert calls==['inventory','model','train','vllm']
    archive=next(tmp_path.glob('v2-4090d-environment-review-*.zip'))
    with zipfile.ZipFile(archive) as z:
        path=next(p for p in z.namelist() if p.endswith('/summary.json'))
        result=json.loads(z.read(path))
        assert result['verdict']['v2_baseline_ready'] and not result['verdict']['short_context_lora_ready']
        assert result['persistent_training_weights_modified'] is False

def test_missing_model_skips_heavy_workers_and_still_packages(tmp_path,monkeypatch):
    monkeypatch.setattr(check,'ROOT',tmp_path)
    calls=[]
    def spawn(command,**kwargs):
        stage=command[command.index('--worker')+1]
        output=Path(command[command.index('--output')+1]);calls.append(stage)
        check.save(output/('stage_'+stage+'.json'),{'passed':stage=='inventory'})
        return argparse.Namespace(wait=lambda **kw:0)
    monkeypatch.setattr(check.subprocess,'Popen',spawn)
    assert check.run(args())==2 and calls==['inventory','model']
    assert len(list(tmp_path.glob('v2-4090d-environment-review-*.zip')))==1

def test_timeout_stops_worker_and_preserves_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(check,'ROOT',tmp_path)
    killed=[]
    class Process:
        pid=123456
        def __init__(self):self.waits=0
        def wait(self,**kw):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('worker',kw.get('timeout',0))
            return 0
        def terminate(self):killed.append('terminate')
    monkeypatch.setattr(check.subprocess,'Popen',lambda *a,**kw:Process())
    monkeypatch.setattr(check.os,'killpg',lambda pid,sig:killed.append(pid),raising=False)
    assert check.run(args())==2 and killed
    path=next((tmp_path/'artifacts').glob('*/summary.json'))
    assert 'TimeoutError' in check.read(path)['error']

def test_model_digest_changes_only_for_the_selected_parameters():
    import torch
    model=torch.nn.Linear(3,2)
    model.weight.requires_grad_(False)
    frozen=check.model_digest(model,True)
    full=check.model_digest(model)
    with torch.no_grad():model.bias.add_(1)
    assert check.model_digest(model,True)==frozen and check.model_digest(model)!=full
    with torch.no_grad():model.weight.add_(1)
    assert check.model_digest(model,True)!=frozen

def test_native_prompt_budget_is_preserved():
    class Tokenizer:
        def apply_chat_template(self,*a,**kw):return list(range(20))
        def encode(self,*a,**kw):return [5,6,7]
    tok=Tokenizer()
    assert len(check.make_prompt(tok,512))==512
    assert check.make_prompt(tok,512)[-10:]==list(range(10,20))
    with pytest.raises(ValueError):check.make_prompt(tok,1)
