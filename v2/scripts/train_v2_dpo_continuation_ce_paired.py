"""Fixed-budget expert continuation CE with unchanged six-epoch DPO."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_recovery_training import validate_training_pairs
from apt_v2.dpo_training import reference_scores,pair_objective
from apt_v2.dpo_continuation_ce import optimize_continuation_ce,ce_schedule,coverage,token_budget
from apt_v2.sft import encode_example
from apt_v2.dpo_chosen_ce import optimize_chosen_ce
from apt_v2.reference_numerics import compare_reference_caches
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.hard_recovery_sft import recovery_weights
from dpo_continuation_ce_paired_common import verify_preflight,training_path,runtime_identity,check_runtime

def train(root_output,branch):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=verify_preflight(root_output);coefficient=cfg['training_variants'][branch]['chosen_ce_coefficient'];algorithm='weighted_dpo_chosen_ce' if branch=='chosen_control' else 'weighted_dpo_continuation_ce'
    output=training_path(root_output,branch);output.mkdir(parents=True)
    write(output/'config.json',dict(cfg,branch=branch,chosen_ce_coefficient=coefficient,branch_algorithm=algorithm))
    write(output/'run_manifest.json',read(root_output/'run_manifest.json'))
    execution=runtime_identity();check_runtime(cfg,execution,read(root_output/'execution_environment.json'));write(output/'execution_environment.json',execution)
    if cfg!=read(ROOT/'configs/dpo-continuation-ce-paired-v1.json') or cfg['pair_count']!=169:raise ValueError('fixed recovery training config differs')
    if cfg['epochs']!=6 or cfg['max_steps']!=258 or cfg['gradient_accumulation']!=4:raise ValueError('six-pass 258-step recovery configuration required')
    data=ROOT/'artifacts'/cfg['data_run'];parent=ROOT/'artifacts'/cfg['initial_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',complete['adapter_sha256'])
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('frozen chosen-only parent identity differs')
    if complete['algorithm']!='chosen_only_sft' or complete['total_sft_updates']!=cfg['source_sft_updates'] or complete['total_dpo_updates']!=cfg['source_dpo_updates'] or not read(parent/'audit.json')['passed']:raise ValueError('reviewed SFT255/DPO11 initialization required')
    if not read(data/'audit.json')['passed']:raise ValueError('audited recovery data required')
    identity=read(parent/'model_identity.json');verify_files(folder,identity['files_sha256'])
    if file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed preference data required')
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
    validate_training_pairs(pairs)
    weights,weight_summary=recovery_weights(pairs,cfg['hard_weight']);write(output/'weights.json',{'weights':weights,'summary':weight_summary})
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
    import shutil
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');write(output/'model_identity.json',identity)
    binding={'branch':branch,'chosen_ce_coefficient':coefficient,'root_input_files_sha256':read(root_output/'preflight.json')['input_files_sha256'],'reference_scores_sha256':cfg['reference_scores_sha256'],'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],
        'weights_sha256':file_hash(output/'weights.json'),'config_sha256':file_hash(output/'config.json'),'source_sha256':read(output/'run_manifest.json')['source_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in [parent/'model_identity.json',parent/'final/complete.json',parent/'audit.json',data/'audit.json',data/'pair_summary.json',data/'pairs.jsonl']}}
    for n,h in identity['files_sha256'].items():
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,p)
    write(output/'input_binding.json',binding)
    if branch=='chosen_control':
        ce_coverage={'source':'original169_chosen','sampled_actions':1014,'pool_actions':169};budget=read(root_output/'control_ce_summary.json');write(output/'ce_summary.json',budget)
    else:
        ce_examples=[json.loads(l) for l in (root_output/'ce_examples.jsonl').read_text(encoding='utf8').splitlines()]
        ce_encoded=[encode_example(tok,e,cfg['max_length']) for e in ce_examples];schedule=read(root_output/'ce_schedule.json')['indices']
        if schedule!=ce_schedule(len(ce_examples),1014,cfg['ce_seed']):raise ValueError('frozen CE schedule differs')
        ce_coverage=coverage(ce_examples,schedule);budget=token_budget(encoded,ce_encoded,schedule,cfg['epochs'])
        for name in ('ce_examples.jsonl','ce_schedule.json','ce_summary.json'):shutil.copyfile(root_output/name,output/name)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,parent/'final/adapter',is_trainable=True).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('only fixed-structure LoRA may train')
    initial=[p.detach().cpu().clone() for p in parameters]
    fresh=[reference_scores(model,e) for e in tqdm(encoded,desc=branch+' reference precheck',unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
    canonical=read(root_output/'reference_scores.json');verify_reference(pairs,encoded,canonical,cfg['adapter_sha256'])
    numerical=compare_reference_caches(dict(canonical,scores=fresh),canonical,cfg['beta'])
    write(output/'reference_precheck.json',{'fresh_scores':fresh,'numerics':numerical,'canonical_sha256':cfg['reference_scores_sha256'],'before_any_optimizer_update':True})
    shutil.copyfile(root_output/'reference_scores.json',output/'reference_scores.json');reference=canonical['scores']
    reference_digest=file_hash(output/'reference_scores.json')
    def score_all(name):
        model.eval();metrics=[pair_objective(model,e,r,cfg['beta'])[1] for e,r in tqdm(zip(encoded,reference),total=len(encoded),desc=name.removesuffix('.json'),unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
        write(output/name,{'pairs':len(metrics),'metrics':metrics,
            'means':{k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]},'weighted_mean_loss':sum(w*m['loss'] for w,m in zip(weights,metrics))/len(metrics)})
    score_all('before_pair_scores.json')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    steps=cursor=0;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train()
    def save(name):
        p=output/name;p.mkdir();model.save_pretrained(p/'adapter',safe_serialization=True)
        write(p/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'steps':steps,'cursor':cursor,'history':history,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},p/'state.pt')
        write(p/'complete.json',{'algorithm':algorithm,'branch':branch,'chosen_ce_coefficient':coefficient,'auxiliary_chosen_ce_updates':steps if coefficient else 0,'steps':steps,'source_sft_updates':cfg['source_sft_updates'],'source_dpo_updates':cfg['source_dpo_updates'],'total_sft_updates':cfg['source_sft_updates'],'total_dpo_updates':cfg['source_dpo_updates']+steps,'training_stage':'paired-'+branch+'-6ep','epochs_completed':steps//43,'cursor':cursor,
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
        if branch=='chosen_control':optimize_chosen_ce(model,encoded,reference,weights,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['beta'],cfg['max_grad_norm'],cfg['epochs'],coefficient,record_step)
        else:optimize_continuation_ce(model,encoded,reference,weights,ce_encoded,schedule,optimizer,cfg['gradient_accumulation'],cfg['seed'],cfg['beta'],cfg['max_grad_norm'],cfg['epochs'],coefficient,record_step)
    if steps!=258 or cursor!=1014:raise ValueError('six complete recovery data passes required')
    save('final');score_all('after_pair_scores.json')
    write(output/'training_summary.json',{'algorithm':algorithm,'branch':branch,'chosen_ce_coefficient':coefficient,'auxiliary_chosen_ce_updates':steps if coefficient else 0,'optimizer_start':'fresh_adam','optimizer_updates':steps,
        'source_sft_updates':cfg['source_sft_updates'],'source_dpo_updates':cfg['source_dpo_updates'],'total_sft_updates':cfg['source_sft_updates'],'total_dpo_updates':cfg['source_dpo_updates']+steps,'pair_exposures':cursor,'chosen_ce_pair_exposures':cursor if coefficient else 0,'epochs_completed':cfg['epochs'],'reference_scores_sha256':reference_digest,'beta':cfg['beta'],'history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'ce_actions_seen':cursor,'ce_coverage':ce_coverage,'token_budget':budget,'weight_summary':weight_summary,'loss_reduction':'mean(weight*DPO + token-mean CE); source='+branch,
        'chosen_response_tokens_per_epoch':sum(len(e['chosen']) for e in encoded),'rejected_response_tokens_per_epoch':sum(len(e['rejected']) for e in encoded),
        'sequence_tokens_per_epoch':sum(2*len(e['prompt'])+len(e['chosen'])+len(e['rejected']) for e in encoded),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    if file_hash(output/'reference_scores.json')!=reference_digest:raise ValueError('reference cache changed during training')
    verify_files(parent/'final/adapter',complete['adapter_sha256']);verify_files(ROOT,binding['source_sha256']);verify_files(ROOT,binding['provenance'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--branch',choices=['chosen_control','continuation_ce'],required=True);a=p.parse_args();train(a.output,a.branch)
