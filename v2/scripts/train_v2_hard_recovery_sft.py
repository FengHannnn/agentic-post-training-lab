"""Six-epoch scenario-weighted continuation from the reviewed chosen-only SFT."""
import sys,json,random,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.hard_recovery_sft import recovery_weights,optimize_weighted_sft
from apt_v2.dpo import response_logps
from apt_v2.recovery_diagnostics import diagnostic_contexts,messages_id
from apt_v2.tasks import make_tasks

def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def train(output):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs/hard-recovery-sft-weighted-v1.json') or cfg['algorithm']!='hard_recovery_weighted_sft':raise ValueError('fixed weighted recovery config required')
    data=ROOT/'artifacts'/cfg['data_run'];parent=ROOT/'artifacts'/cfg['initial_run'];diag=ROOT/'artifacts'/cfg['diagnostic_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',complete['adapter_sha256'])
    if complete['algorithm']!='chosen_only_sft' or complete['total_sft_updates']!=cfg['source_sft_updates'] or complete['total_dpo_updates']!=cfg['source_dpo_updates'] or file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:
        raise ValueError('reviewed SFT255/DPO11 initialization differs')
    if not read(parent/'audit.json')['passed'] or not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed parent and recovery data required')
    if not read(diag/'audit.json')['passed'] or read(diag/'config.json')['checkpoints']['sft']['adapter_sha256']!=cfg['adapter_sha256']:raise ValueError('reviewed diagnostic source differs')
    identity=read(parent/'model_identity.json');verify_files(folder,identity['files_sha256'])
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
    examples=chosen_examples(pairs);weights,weight_summary=recovery_weights(pairs,cfg['hard_weight'])
    if len(examples)!=cfg['pair_count']:raise ValueError('reviewed pair count differs')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False);encoded=encode_chosen(tok,examples,cfg['max_length'])
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');jsonl(output/'examples.jsonl',examples)
    write(output/'weights.json',{'weights':weights,'summary':weight_summary});write(output/'model_identity.json',identity)
    contexts=diagnostic_contexts([t for t in make_tasks() if t.split in cfg['task_counts']]);seen={messages_id(p['messages']) for p in pairs}
    membership=[{'context_id':c['context_id'],'task_id':c['task_id'],'split':c['split'],'messages_sha256':messages_id(c['messages']),
        'seen_in_training_pairs':messages_id(c['messages']) in seen} for c in contexts]
    if any(m['seen_in_training_pairs']!=(m['split']=='train') for m in membership):raise ValueError('exact recovery context membership differs')
    jsonl(output/'contexts.jsonl',contexts);jsonl(output/'context_membership.jsonl',membership)
    provenance=[parent/'model_identity.json',parent/'audit.json',parent/'final/complete.json',parent/'after_chosen_scores.json',data/'audit.json',data/'pairs.jsonl',
        diag/'audit.json',diag/'config.json',diag/'input_binding.json']+[parent/'final/adapter'/n for n in complete['adapter_sha256']]
    binding={'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],'examples_sha256':data_sha256(examples),
        'encoded_sha256':data_sha256(encoded),'weights_sha256':file_hash(output/'weights.json'),'contexts_sha256':file_hash(output/'contexts.jsonl'),
        'membership_sha256':file_hash(output/'context_membership.jsonl'),'config_sha256':file_hash(output/'config.json'),
        'source_sha256':read(output/'run_manifest.json')['source_sha256'],'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance}}
    write(output/'input_binding.json',binding)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            target=output/'tokenizer'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,target)
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
        write(output/name,{'examples':len(metrics),'metrics':metrics,'mean_loss':sum(m['loss'] for m in metrics)/len(metrics),
            'mean_weighted_loss':sum(m['loss']*w for m,w in zip(metrics,weights))/len(metrics)})
    score_all('before_chosen_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    steps=cursor=0;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def save(name):
        p=output/name;p.mkdir();model.save_pretrained(p/'adapter',safe_serialization=True);write(p/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'steps':steps,'cursor':cursor,'history':history,'python_rng':random.getstate(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},p/'state.pt')
        write(p/'complete.json',{'algorithm':'hard_recovery_weighted_sft','steps':steps,'source_sft_updates':cfg['source_sft_updates'],
            'source_dpo_updates':cfg['source_dpo_updates'],'sft_updates':steps,'total_sft_updates':cfg['source_sft_updates']+steps,
            'total_dpo_updates':cfg['source_dpo_updates'],'epochs_completed':steps//cfg['checkpoint_every'],'cursor':cursor,
            'adapter_sha256':{f.name:file_hash(f) for f in (p/'adapter').iterdir() if f.is_file()},'state_sha256':file_hash(p/'state.pt')})
    with (output/'training_steps.jsonl').open('x',encoding='utf8') as stream,tqdm(total=cfg['max_steps'],desc='Hard recovery SFT',unit='step',file=sys.stdout,disable=False,dynamic_ncols=True,mininterval=.2) as progress:
        def record(row):
            nonlocal steps,cursor
            steps=row['step'];cursor=row['pairs_seen'];history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            progress.set_postfix(epoch=f"{row['epoch']}/{cfg['epochs']}",weighted_loss=f"{row['loss']:.4f}",examples=cursor,refresh=False);progress.update(1)
            if steps%cfg['checkpoint_every']==0:
                if steps<cfg['max_steps']:save(f'checkpoint-{steps:04}')
                tqdm.write(f"Epoch {row['epoch']}/{cfg['epochs']} completed; step {steps}/{cfg['max_steps']}; examples {cursor}",file=sys.stdout)
        optimize_weighted_sft(model,encoded,weights,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['max_grad_norm'],cfg['epochs'],record)
    if steps!=cfg['max_steps'] or cursor!=cfg['pair_count']*cfg['epochs']:raise ValueError('complete weighted recovery epochs required')
    save('final');score_all('after_chosen_scores.json')
    write(output/'training_summary.json',{'algorithm':'hard_recovery_weighted_sft','optimizer_start':'fresh_adam','optimizer_updates':steps,
        'source_sft_updates':cfg['source_sft_updates'],'source_dpo_updates':cfg['source_dpo_updates'],'sft_updates':steps,
        'total_sft_updates':cfg['source_sft_updates']+steps,'total_dpo_updates':cfg['source_dpo_updates'],'pair_exposures':cursor,'epochs_completed':cfg['epochs'],
        'loss_reduction':'mean response tokens; normalized scenario weight; mean examples per batch',
        'rejected_used_in_loss':False,'reference_used_in_loss':False,'token_weighting':False,'supervised_tokens_per_epoch':sum(s['response_length'] for s in encoded),
        'weight_summary':weight_summary,'history':history,'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    verify_files(ROOT,binding['source_sha256']);verify_files(ROOT,binding['provenance'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);train(p.parse_args().output)
