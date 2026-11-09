"""One bounded DPO pass over the reviewed 44 action preferences."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.dpo import encode_pair,pair_id
from apt_v2.dpo_training import reference_scores,pair_objective

def train(output):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=read(output/'config.json')
    if cfg['epochs']!=1 or cfg['max_steps']!=11 or cfg['gradient_accumulation']!=4:raise ValueError('one-pass 11-step pilot configuration required')
    data=ROOT/'artifacts'/cfg['data_run'];parent=ROOT/'artifacts'/cfg['parent_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',complete['adapter_sha256'])
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('frozen SFT identity differs')
    identity=read(parent/'model_identity.json');verify_files(folder,identity['files_sha256'])
    if file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed preference data required')
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
    if len(pairs)!=cfg['pair_count'] or len({p['pair_id'] for p in pairs})!=len(pairs):raise ValueError('44 unique pairs required')
    if any(p['split']!='train' or p['pair_id']!=pair_id(p) for p in pairs):raise ValueError('train pair identity differs')
    from apt_v2.tasks import make_tasks
    allowed={(t.id,t.family_id) for t in make_tasks() if t.split=='train'}
    if any((p['task_id'],p['family_id']) not in allowed for p in pairs):raise ValueError('train task provenance differs')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    encoded=[encode_pair(tok,p,cfg['max_length']) for p in pairs]
    import shutil
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');write(output/'model_identity.json',identity)
    binding={'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],
        'config_sha256':file_hash(output/'config.json'),'source_sha256':read(output/'run_manifest.json')['source_sha256']}
    write(output/'input_binding.json',binding)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,parent/'final/adapter',is_trainable=True).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('only fixed-structure LoRA may train')
    initial=[p.detach().cpu().clone() for p in parameters]
    reference=[reference_scores(model,e) for e in encoded]
    write(output/'reference_scores.json',{'baseline_adapter_sha256':cfg['adapter_sha256'],
        'pair_ids':[p['pair_id'] for p in pairs],'scores':reference,'contract':'detached FP32 scores before any update'})
    def score_all(name):
        model.eval();metrics=[pair_objective(model,e,r,cfg['beta'])[1] for e,r in zip(encoded,reference)]
        write(output/name,{'pairs':len(metrics),'metrics':metrics,
            'means':{k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}})
    score_all('before_pair_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    indices=list(range(len(pairs)));random.Random(cfg['seed']).shuffle(indices)
    steps=cursor=0;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train()
    def save(name):
        p=output/name;p.mkdir();model.save_pretrained(p/'adapter',safe_serialization=True)
        write(p/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'steps':steps,'cursor':cursor,'history':history,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},p/'state.pt')
        write(p/'complete.json',{'algorithm':'dpo','steps':steps,'source_sft_updates':126,'cursor':cursor,
            'adapter_sha256':{f.name:file_hash(f) for f in (p/'adapter').iterdir() if f.is_file()},'state_sha256':file_hash(p/'state.pt')})
    while cursor<len(indices) and steps<cfg['max_steps']:
        batch=indices[cursor:cursor+cfg['gradient_accumulation']];optimizer.zero_grad(set_to_none=True);value=0.
        for i in batch:
            loss,_=pair_objective(model,encoded[i],reference[i],cfg['beta'],True)
            if not torch.isfinite(loss):raise ValueError('nonfinite DPO objective')
            (loss/len(batch)).backward();value+=loss.detach().item()/len(batch)
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'],error_if_nonfinite=True)
        if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
        optimizer.step();steps+=1;cursor+=len(batch)
        row={'step':steps,'pairs_seen':cursor,'loss':value,'grad_norm':norm.item()};history.append(row);print(json.dumps(row),flush=True)
        if steps%cfg['checkpoint_every']==0:save(f'checkpoint-{steps:04}')
    save('final');score_all('after_pair_scores.json')
    write(output/'training_summary.json',{'algorithm':'dpo','optimizer_start':'fresh_adam','optimizer_updates':steps,
        'source_sft_updates':126,'pair_exposures':cursor,'beta':cfg['beta'],'history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    verify_files(parent/'final/adapter',complete['adapter_sha256']);verify_files(ROOT,binding['source_sha256'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);train(p.parse_args().output)
