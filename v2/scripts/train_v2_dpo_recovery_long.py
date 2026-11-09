"""Three recovery DPO passes with visible progress and a fixed DPO11 reference."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_recovery_training import validate_training_pairs,optimize_epochs
from apt_v2.dpo_training import reference_scores,pair_objective

def train(output):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs/dpo-recovery-train-v2.json') or cfg['pair_count']!=169:raise ValueError('fixed recovery training config differs')
    if cfg['epochs']!=3 or cfg['max_steps']!=129 or cfg['gradient_accumulation']!=4:raise ValueError('three-pass 129-step recovery configuration required')
    data=ROOT/'artifacts'/cfg['data_run'];parent=ROOT/'artifacts'/cfg['initial_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',complete['adapter_sha256'])
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('frozen DPO11 identity differs')
    if complete['algorithm']!='dpo' or complete['steps']!=11:raise ValueError('reviewed DPO11 initialization required')
    if not read(data/'audit.json')['passed']:raise ValueError('audited recovery data required')
    identity=read(parent/'model_identity.json');verify_files(folder,identity['files_sha256'])
    if file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed preference data required')
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
    validate_training_pairs(pairs)
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
    import shutil
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');write(output/'model_identity.json',identity)
    binding={'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],
        'config_sha256':file_hash(output/'config.json'),'source_sha256':read(output/'run_manifest.json')['source_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in [parent/'model_identity.json',parent/'final/complete.json',data/'audit.json',data/'pair_summary.json',data/'pairs.jsonl']}}
    for n,h in identity['files_sha256'].items():
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,p)
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
    reference=[reference_scores(model,e) for e in tqdm(encoded,desc='Reference cache',unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
    write(output/'reference_scores.json',{'baseline_adapter_sha256':cfg['adapter_sha256'],
        'pair_ids':[p['pair_id'] for p in pairs],'scores':reference,'contract':'detached FP32 scores before any update'})
    def score_all(name):
        model.eval();metrics=[pair_objective(model,e,r,cfg['beta'])[1] for e,r in tqdm(zip(encoded,reference),total=len(encoded),desc=name.removesuffix('.json'),unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
        write(output/name,{'pairs':len(metrics),'metrics':metrics,
            'means':{k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}})
    score_all('before_pair_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    steps=cursor=0;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train()
    def save(name):
        p=output/name;p.mkdir();model.save_pretrained(p/'adapter',safe_serialization=True)
        write(p/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'steps':steps,'cursor':cursor,'history':history,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},p/'state.pt')
        write(p/'complete.json',{'algorithm':'dpo','steps':steps,'source_sft_updates':126,'source_dpo_updates':11,'total_dpo_updates':11+steps,'training_stage':'recovery-3ep','epochs_completed':steps//43,'cursor':cursor,
            'adapter_sha256':{f.name:file_hash(f) for f in (p/'adapter').iterdir() if f.is_file()},'state_sha256':file_hash(p/'state.pt')})
    with (output/'training_steps.jsonl').open('x',encoding='utf8') as stream, tqdm(total=cfg['max_steps'],desc='DPO training',unit='step',file=sys.stdout,disable=False,dynamic_ncols=True,mininterval=.2) as progress:
        def record_step(row):
            nonlocal steps,cursor
            steps=row['step'];cursor=row['pairs_seen'];history.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            progress.set_postfix(epoch=f"{row['epoch']}/{cfg['epochs']}",loss=f"{row['loss']:.4f}",pairs=f"{cursor}/{169*cfg['epochs']}",refresh=False);progress.update(1)
            if steps%cfg['checkpoint_every']==0:
                if steps<cfg['max_steps']:save(f'checkpoint-{steps:04}')
                tqdm.write(f"Epoch {row['epoch']}/{cfg['epochs']} completed; step {steps}/{cfg['max_steps']}; pairs {cursor}",file=sys.stdout)
        optimize_epochs(model,encoded,reference,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['beta'],cfg['max_grad_norm'],cfg['epochs'],record_step)
    if steps!=129 or cursor!=507:raise ValueError('three complete recovery data passes required')
    save('final');score_all('after_pair_scores.json')
    write(output/'training_summary.json',{'algorithm':'dpo','optimizer_start':'fresh_adam','optimizer_updates':steps,
        'source_sft_updates':126,'source_dpo_updates':11,'total_dpo_updates':11+steps,'pair_exposures':cursor,'epochs_completed':3,'beta':cfg['beta'],'history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    verify_files(parent/'final/adapter',complete['adapter_sha256']);verify_files(ROOT,binding['source_sha256']);verify_files(ROOT,binding['provenance'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);train(p.parse_args().output)
