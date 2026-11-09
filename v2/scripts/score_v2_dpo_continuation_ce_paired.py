"""FP32 checkpoint scores on fixed training pairs and expert workflow probes."""
import sys,random,math
from pathlib import Path
from dpo_continuation_ce_paired_common import ROOT,read,write,load,verify_inputs,checkpoint_path,model_specs
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.dpo_training import reference_scores
from apt_v2.reference_numerics import compare_reference_caches
from apt_v2.sft import encode_example
from apt_v2.dpo import response_logps
from apt_v2.dpo_investigation import metrics_from_scores,summarize_pair_metrics,summarize_probe_metrics,digest

def score(output):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from tqdm.auto import tqdm
    cfg=verify_inputs(output);cfg={**cfg,'checkpoints':model_specs(cfg,output)}
    write(output/'score_model_specs.json',cfg['checkpoints'])
    if not torch.cuda.is_available():raise ValueError('CUDA scoring required')
    torch.manual_seed(cfg['eval']['seed']);random.seed(cfg['eval']['seed'])
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    pairs=load(output/'pairs.jsonl');encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];probes=load(output/'probes.jsonl')
    pencoded=[encode_example(tok,p,cfg['max_length']) for p in probes];ref=read(output/'reference_scores.json');weights=read(output/'weights.json')['weights']
    verify_reference(pairs,encoded,ref,cfg['checkpoints']['parent']['adapter_sha256'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,checkpoint_path(cfg['checkpoints']['parent'])/'adapter',adapter_name='parent',is_trainable=False).eval()
    fresh=[reference_scores(model,e) for e in tqdm(encoded,desc='Parent reference recheck',unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True)]
    maximum=max(abs(x-y) for a,b in zip(fresh,ref['scores']) for k in ('chosen','rejected') for x,y in zip(a[k],b[k]))
    numerical=compare_reference_caches(dict(ref,scores=fresh),ref,cfg['beta'])
    write(output/'reference_recheck.json',{'maximum_token_logp_difference':maximum,'fresh_scores':fresh,'numerics':numerical})
    summaries={};torch.cuda.reset_peak_memory_stats()
    for branch,spec in cfg['checkpoints'].items():
        if branch!='parent':model.load_adapter(checkpoint_path(spec)/'adapter',adapter_name=branch,is_trainable=False)
        model.set_adapter(branch);model.eval();scores=[];metrics=[]
        for e,r in tqdm(zip(encoded,ref['scores']),total=len(encoded),desc=branch+' pair scores',unit='pair',file=sys.stdout,disable=False,dynamic_ncols=True):
            current=reference_scores(model,e);m=metrics_from_scores(current,r);scores.append(current);metrics.append(m)
        pm=[]
        for p,e in tqdm(zip(probes,pencoded),total=len(probes),desc=branch+' workflow probes',unit='action',file=sys.stdout,disable=False,dynamic_ncols=True):
            n=e['prompt_length'];values=response_logps(model,e['input_ids'][:n],e['input_ids'][n:]).cpu().tolist()
            pm.append({'probe_id':p['probe_id'],'probe_sha256':digest(p),'token_log_probs':values,'ce':-sum(values)/len(values)})
        summary={'pairs':summarize_pair_metrics(pairs,encoded,metrics,ref['scores'],weights),'probes':summarize_probe_metrics(probes,pm)}
        write(output/(branch+'-scores.json'),{'branch':branch,'adapter_sha256':spec['adapter_sha256'],'pair_scores':scores,'pair_metrics':metrics,'probe_metrics':pm,'summary':summary})
        summaries[branch]=summary
    write(output/'scores_summary.json',{'branches':summaries,'training_enabled':False,'optimizer_updates':0,'maximum_parent_reference_difference':maximum,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    verify_inputs(output)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);score(p.parse_args().output)
