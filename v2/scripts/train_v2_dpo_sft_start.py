"""Short DPO segments continue one Adam/RNG trajectory from best weighted SFT."""
import sys,json,random,time,shutil
from pathlib import Path
from common import read,write,file_hash,verify_files
from dpo_sft_start_common import ROOT,CONFIG,verify_inputs,load,checkpoint,branch_name,normalize_config,verify_reference_cache,runtime_identity
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_training import reference_scores
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.dpo_sft_start import optimize_segment,verify_history

def train(output,target):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from tqdm.auto import tqdm
    cfg=verify_inputs(output,precision=True)
    if target not in cfg['checkpoint_steps']:raise ValueError('fixed short DPO target required')
    position=cfg['checkpoint_steps'].index(target);previous=cfg['checkpoint_steps'][position-1] if position else 0
    if not read(output/'weighted_sft-decision.json')['training_allowed'] or not read(output/'weighted_sft-audit.json')['passed']:raise ValueError('stable audited SFT baseline required before training')
    if previous:
        prior_decision=read(output/(branch_name(previous)+'-decision.json'))
        if prior_decision['passed'] or prior_decision['severe_regression_stop'] or not read(output/(branch_name(previous)+'-audit.json'))['passed']:raise ValueError('continuation after stopping rule')
    dest=checkpoint(output,target)
    if dest.exists():raise FileExistsError('short DPO checkpoint already exists')
    parent=output/'adapters/weighted_sft' if not previous else checkpoint(output,previous)/'adapter'
    state=None
    if previous:
        meta=read(checkpoint(output,previous)/'complete.json');verify_files(checkpoint(output,previous)/'adapter',meta['adapter_sha256'])
        if file_hash(checkpoint(output,previous)/'state.pt')!=meta['state_sha256']:raise ValueError('resume state hash differs')
        state=torch.load(checkpoint(output,previous)/'state.pt',map_location='cpu',weights_only=False)
        verify_history(cfg,meta,read(checkpoint(output,previous)/'training_summary.json'),state)
        if state['reference_scores_sha256']!=file_hash(output/'reference_scores.json') or state['config_sha256']!=file_hash(output/'config.json'):raise ValueError('resume reference or config differs')
    elif (output/'reference_scores.json').exists():raise FileExistsError('step-zero reference already exists')
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False);pairs=load(output/'pairs.jsonl');encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,parent,is_trainable=True).eval();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('only fixed 1089536 LoRA parameters may train')
    initial=[p.detach().cpu().clone() for p in parameters]
    if not previous:
        # Scores come from the exact policy initialization in eval mode before step 1.
        reference=[reference_scores(model,e) for e in tqdm(encoded,desc='Fresh weighted SFT reference',unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
        cache={'baseline_adapter_sha256':cfg['adapter_sha256'],'pair_ids':[p['pair_id'] for p in pairs],'scores':reference}
        verify_reference(pairs,encoded,cache,cfg['adapter_sha256']);write(output/'reference_scores.json',cache)
        write(output/'reference_binding.json',{'reference_scores_sha256':file_hash(output/'reference_scores.json'),'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],'before_optimizer_update':0,'dtype':'float32','mode':'eval'})
    verify_reference_cache(output,cfg,tok);reference=read(output/'reference_scores.json')['scores'];reference_digest=file_hash(output/'reference_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    if state:
        optimizer.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    history=[] if state is None else state['history'];started=time.perf_counter();torch.cuda.reset_peak_memory_stats();(output/'training').mkdir(exist_ok=True)
    with (output/'training'/f'segment-{target:04}-steps.jsonl').open('x',encoding='utf8') as stream,tqdm(total=target,initial=previous,desc='Short DPO',unit='step',file=sys.stdout,disable=False,dynamic_ncols=True) as progress:
        def record(row):
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();progress.set_postfix(loss=f"{row['loss']:.4f}",pairs=row['pairs_seen'],refresh=False);progress.update(1)
        history=optimize_segment(model,encoded,reference,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['beta'],cfg['max_grad_norm'],target,history,record)
    dest.mkdir();model.save_pretrained(dest/'adapter',safe_serialization=True)
    cursor=history[-1]['pairs_seen'];state={'optimizer':optimizer.state_dict(),'steps':target,'cursor':cursor,'history':history,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'reference_scores_sha256':reference_digest,'config_sha256':file_hash(output/'config.json')};torch.save(state,dest/'state.pt')
    ancestry={'algorithm':cfg['algorithm'],'source_sft_updates':513,'source_dpo_updates':11,'total_sft_updates':513,'total_dpo_updates':11+target,'epochs_completed':0}
    complete={**ancestry,'steps':target,'cursor':cursor,'pair_fraction':cursor/cfg['pair_count'],'adapter_sha256':{p.name:file_hash(p) for p in (dest/'adapter').iterdir() if p.is_file()},'state_sha256':file_hash(dest/'state.pt'),'reference_scores_sha256':reference_digest,'config_sha256':file_hash(output/'config.json'),'initial_adapter_sha256':cfg['adapter_sha256'],'resumed_from_step':previous}
    summary={**ancestry,'optimizer_start':'fresh_adam_at_step_zero_then_exact_resume','optimizer_updates':target,'pair_exposures':cursor,'history':history,'reference_scores_sha256':reference_digest,'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),'segment_elapsed_seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'learning_rate':cfg['learning_rate'],'beta':cfg['beta'],'uniform_pair_weight':1,'loss_reduction':'mean(sequence-sum-logp DPO)','resumed_from_step':previous}
    verify_history(cfg,complete,summary,state);write(dest/'complete.json',complete);write(dest/'training_summary.json',summary)
    infer=output/'adapters'/branch_name(target);infer.mkdir();shutil.copyfile(dest/'adapter/adapter_model.safetensors',infer/'adapter_model.safetensors');write(infer/'adapter_config.json',normalize_config(read(dest/'adapter/adapter_config.json')))
    write(dest/'inference_binding.json',{'adapter_sha256':file_hash(infer/'adapter_model.safetensors'),'normalized_config_sha256':file_hash(infer/'adapter_config.json')})
    write(dest/'execution_environment.json',runtime_identity())
    if file_hash(output/'reference_scores.json')!=reference_digest:raise ValueError('reference mutated during short DPO')
    verify_inputs(output,precision=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--target-step',type=int,required=True);a=p.parse_args();train(a.output,a.target_step)
