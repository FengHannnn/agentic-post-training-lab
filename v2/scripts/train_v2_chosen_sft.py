"""Three-pass chosen-only SFT control initialized from the same reviewed DPO11."""
import sys,json,random,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256,optimize_sft
from apt_v2.dpo import response_logps

def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def train(output):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs/chosen-only-sft-v1.json') or cfg['algorithm']!='chosen_only_sft' or cfg['epochs']!=3 or cfg['max_steps']!=129:raise ValueError('fixed chosen-only control required')
    data=ROOT/'artifacts'/cfg['data_run'];parent=ROOT/'artifacts'/cfg['initial_run'];comparison=ROOT/'artifacts'/cfg['comparison_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    initial_complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',initial_complete['adapter_sha256'])
    dpo_complete=read(comparison/'final/complete.json');verify_files(comparison/'final/adapter',dpo_complete['adapter_sha256'])
    if initial_complete['algorithm']!='dpo' or initial_complete['steps']!=11 or file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('DPO11 initialization differs')
    if dpo_complete['algorithm']!='dpo' or dpo_complete['steps']!=129 or file_hash(comparison/'final/adapter/adapter_model.safetensors')!=cfg['comparison_adapter_sha256']:raise ValueError('DPO140 comparison differs')
    if not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed recovery pairs required')
    identity=read(parent/'model_identity.json')
    if identity!=read(comparison/'model_identity.json'):raise ValueError('comparison base model differs')
    verify_files(folder,identity['files_sha256'])
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()];examples=chosen_examples(pairs)
    if len(examples)!=169:raise ValueError('169 chosen examples required')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False);encoded=encode_chosen(tok,examples,cfg['max_length'])
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');jsonl(output/'examples.jsonl',examples);write(output/'model_identity.json',identity)
    provenance=[parent/'model_identity.json',parent/'final/complete.json',data/'audit.json',data/'pairs.jsonl',comparison/'model_identity.json',comparison/'final/complete.json',comparison/'reference_scores.json']+[comparison/'final/adapter'/n for n in dpo_complete['adapter_sha256']]
    binding={'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],'examples_sha256':data_sha256(examples),
        'encoded_sha256':data_sha256(encoded),'config_sha256':file_hash(output/'config.json'),'source_sha256':read(output/'run_manifest.json')['source_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance}}
    write(output/'input_binding.json',binding)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,p)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,parent/'final/adapter',is_trainable=True).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('fixed LoRA parameters required')
    initial=[p.detach().cpu().clone() for p in parameters]
    def score_all(name):
        model.eval();metrics=[]
        for sample in tqdm(encoded,desc=name.removesuffix('.json'),unit='example',file=sys.stdout,disable=False,dynamic_ncols=True):
            n=sample['prompt_length'];probs=response_logps(model,sample['input_ids'][:n],sample['input_ids'][n:])
            metrics.append({'input_id':data_sha256(sample),'token_log_probs':probs.cpu().tolist(),'loss':-probs.mean().item()})
        write(output/name,{'examples':len(metrics),'metrics':metrics,'mean_loss':sum(m['loss'] for m in metrics)/len(metrics)})
    score_all('before_chosen_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0);steps=cursor=0;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def save(name):
        p=output/name;p.mkdir();model.save_pretrained(p/'adapter',safe_serialization=True);write(p/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'steps':steps,'cursor':cursor,'history':history,'python_rng':random.getstate(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},p/'state.pt')
        write(p/'complete.json',{'algorithm':'chosen_only_sft','steps':steps,'source_sft_updates':126,'source_dpo_updates':11,'sft_updates':steps,
            'total_sft_updates':126+steps,'total_dpo_updates':11,'epochs_completed':steps//43,'cursor':cursor,
            'adapter_sha256':{f.name:file_hash(f) for f in (p/'adapter').iterdir() if f.is_file()},'state_sha256':file_hash(p/'state.pt')})
    with (output/'training_steps.jsonl').open('x',encoding='utf8') as stream,tqdm(total=129,desc='Chosen SFT training',unit='step',file=sys.stdout,disable=False,dynamic_ncols=True,mininterval=.2) as progress:
        def record(row):
            nonlocal steps,cursor
            steps=row['step'];cursor=row['pairs_seen'];history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            progress.set_postfix(epoch=f"{row['epoch']}/3",loss=f"{row['loss']:.4f}",examples=f"{cursor}/507",refresh=False);progress.update(1)
            if steps%43==0:
                if steps<129:save(f'checkpoint-{steps:04}')
                tqdm.write(f"Epoch {row['epoch']}/3 completed; step {steps}/129; examples {cursor}",file=sys.stdout)
        optimize_sft(model,encoded,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['max_grad_norm'],cfg['epochs'],record)
    if steps!=129 or cursor!=507:raise ValueError('three complete chosen-only passes required')
    save('final');score_all('after_chosen_scores.json')
    write(output/'training_summary.json',{'algorithm':'chosen_only_sft','optimizer_start':'fresh_adam','optimizer_updates':steps,'source_sft_updates':126,
        'source_dpo_updates':11,'sft_updates':129,'total_sft_updates':255,'total_dpo_updates':11,'pair_exposures':cursor,'epochs_completed':3,
        'loss_reduction':'mean tokens per chosen response; mean examples per batch','rejected_used_in_loss':False,'reference_used_in_loss':False,
        'supervised_tokens_per_epoch':sum(s['response_length'] for s in encoded),'history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    verify_files(ROOT,binding['source_sha256']);verify_files(ROOT,binding['provenance']);verify_files(parent/'final/adapter',initial_complete['adapter_sha256'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);train(p.parse_args().output)
