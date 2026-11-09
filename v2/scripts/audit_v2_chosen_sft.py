"""Audit chosen-only supervision and same-run DPO11/DPO140/SFT evaluations."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from evaluate_v2_dpo_recovery import verify_inputs
from audit_v2_dpo_recovery import audit_evaluations,load
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256,verify_sft_updates

def audit(output):
    import torch
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);binding=read(output/'input_binding.json');pairs=load(output/'pairs.jsonl')
    if file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed source pairs differ')
    examples=chosen_examples(pairs)
    if examples!=load(output/'examples.jsonl') or data_sha256(examples)!=binding['examples_sha256']:raise ValueError('chosen-only projection differs')
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False);encoded=encode_chosen(tok,examples,cfg['max_length'])
    if data_sha256(encoded)!=binding['encoded_sha256']:raise ValueError('chosen-only encoding differs')
    for sample in encoded:
        n=sample['prompt_length']
        if sample['labels'][:n]!=[-100]*n or sample['labels'][n:]!=sample['input_ids'][n:] or sample['response_length']!=len(sample['input_ids'])-n:raise ValueError('assistant-only labels required')
    for name in ('before_chosen_scores.json','after_chosen_scores.json'):
        scores=read(output/name)
        if scores['examples']!=169 or len(scores['metrics'])!=169:raise ValueError('169 chosen scores required')
        for sample,metric in zip(encoded,scores['metrics']):
            probs=metric['token_log_probs']
            if metric['input_id']!=data_sha256(sample) or len(probs)!=sample['response_length'] or any(not math.isfinite(v) or v>1e-6 for v in probs):raise ValueError('chosen score token binding differs')
            if not math.isfinite(metric['loss']) or abs(-sum(probs)/len(probs)-metric['loss'])>1e-6:raise ValueError('chosen cross entropy differs')
        if abs(sum(m['loss'] for m in scores['metrics'])/169-scores['mean_loss'])>1e-8:raise ValueError('chosen mean loss differs')
    ref=read(ROOT/'artifacts'/cfg['comparison_run']/'reference_scores.json')
    if ref['baseline_adapter_sha256']!=cfg['adapter_sha256'] or ref['pair_ids']!=[p['pair_id'] for p in pairs] or len(ref['scores'])!=169:raise ValueError('comparison initialization binding differs')
    for metric,score in zip(read(output/'before_chosen_scores.json')['metrics'],ref['scores']):
        if len(metric['token_log_probs'])!=len(score['chosen']) or any(abs(a-b)>1e-5 for a,b in zip(metric['token_log_probs'],score['chosen'])):raise ValueError('SFT/DPO initial chosen probabilities differ')
    final=read(output/'final/complete.json');train=read(output/'training_summary.json')
    if file_hash(output/'final/state.pt')!=final['state_sha256']:raise ValueError('SFT optimizer state differs')
    state=torch.load(output/'final/state.pt',map_location='cpu',weights_only=True);verify_sft_updates(cfg,final,train,state)
    if train['rejected_used_in_loss'] is not False or train['reference_used_in_loss'] is not False or train['supervised_tokens_per_epoch']!=sum(s['response_length'] for s in encoded):raise ValueError('chosen-only training objective differs')
    if load(output/'training_steps.jsonl')!=train['history']:raise ValueError('streamed metrics differ')
    for step,epoch in ((43,1),(86,2)):
        p=output/f'checkpoint-{step:04}';complete=read(p/'complete.json')
        if complete['algorithm']!='chosen_only_sft' or complete['steps']!=step or complete['epochs_completed']!=epoch or complete['cursor']!=169*epoch:raise ValueError('SFT checkpoint differs')
        verify_files(p/'adapter',complete['adapter_sha256'])
    evidence=audit_evaluations(output,cfg,tok)
    if evidence['episodes_replayed']!=96 or evidence['recovery_actions_replayed']!=336:raise ValueError('complete three-model evaluation required')
    write(output/'evaluation_summary.json',{**evidence,'status':'awaiting_local_review','algorithm':'chosen_only_sft','optimizer_updates':129,
        'recovery_comparison':read(output/'recovery_comparison.json')})
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,**{k:v for k,v in evidence.items() if not k.startswith('paired_')},'chosen_examples':169,
        'supervised_tokens_per_epoch':sum(s['response_length'] for s in encoded),'optimizer_updates':129,'chosen_only_supervision':True,
        'initial_chosen_scores_match_dpo':True,'initial_and_comparison_checkpoints_unchanged':True,'test_split_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
