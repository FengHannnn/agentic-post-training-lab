"""Single bounded stored-rollout GRPO update, reference, resume and generation comparison."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from collect_phase5_signal import ROOT, load_run, read, write, check_gpu, file_hash
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.trainer_scores import score_response, no_update_metrics, pack_trainer_scores
from agentic_post_training.rollouts.phase5_signal import run_seconds
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (
    guarded_step, lora_parameters, save_optimizer_checkpoint, restore_optimizer_checkpoint,
    adapter_snapshot, compare_adapter_snapshot,optimizer_snapshot,compare_optimizer_snapshot)

PLAN=ROOT/'configs/grpo_first_update.json'
ARCHIVE=ROOT/'grpo-first-review-20261027.zip'


def context():
    import torch
    plan=read(PLAN)
    fixed={'contract_version':'grpo-first-update-v1','learning_rate':1e-5,'beta':.02,'clip_epsilon':.2,
        'tis_cap':2.,'max_grad_norm':1.,'stop_sampled_k3':.05,'stop_ratio_clip_fraction':.5,
        'checkpoint_step':1,'resume_probe_step':2,'max_logical_steps':16,'seed':20261027,
        'reload_logp_tolerance':1e-5,'max_wall_seconds':1800,'max_run_cost_yuan':10.,
        'auto_promote_baseline':False,'rollout_policy_iterations':1}
    if any(plan[k]!=v for k,v in fixed.items()):raise ValueError('frozen first-update plan differs')
    source=ROOT/plan['source_rollout'];summary=read(source/'summary.json')
    if file_hash(source/'summary.json')!=plan['source_summary_sha256'] or not summary['signal_collection_interface_passed']:
        raise ValueError('bound audited signal batch required')
    for rel,digest in summary['files_sha256'].items():
        if file_hash(source/rel)!=digest:raise ValueError('source signal artifact changed')
    cfg,_,baseline,base,tok,rows,groups,_,rewards=load_run(source)
    device=check_gpu(cfg)
    if device!=read(source/'run_manifest.json')['device']:raise ValueError('use same GPU model as stored old scoring')
    records=[json.loads(x) for x in (source/'trainer_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    targets={(r['group_id'],r['sample_index']):r for r in records}
    rw={(r['group_id'],r['sample_index']):r for r in rewards['records']}
    if len(records)!=1152 or len(targets)!=1152:raise ValueError('all trainer records required')
    for g in groups:
        for s in g:
            row=targets[s.group_id,s.sample_index];reward=rw[s.group_id,s.sample_index]
            if (row['sampler_behavior_token_log_probs']!=s.behavior_token_log_probs
                or any(row[k]!=reward[k] for k in ('reward','advantage','eligible'))):raise ValueError('sample/reward mapping differs')
            old=torch.tensor(row['trainer_old_token_log_probs'],dtype=torch.float32)
            new=torch.tensor(row['trainer_new_token_log_probs'],dtype=torch.float32,requires_grad=True)
            if no_update_metrics(old,new,1e-5)!=row['no_update_check']:raise ValueError('saved trainer scores differ')
    packed=pack_trainer_scores(groups,records,tok.pad_token_id)
    saved=torch.load(source/'trainer_score_batch.pt',map_location='cpu',weights_only=True)
    if set(saved)!=set(packed) or any(not torch.equal(v,saved[k]) for k,v in packed.items()):raise ValueError('stored masks/scores differ')
    if read(source/'reward_audit.json')!=rewards:raise ValueError('stored rewards differ')
    mixed={r['group_id'] for r in rewards['groups'] if r['eligible'] and not r['zero_variance']}
    selected=[key for batch in plan['group_batches'] for key in batch]
    if (len(selected)!=62 or len(set(selected))!=62 or set(selected)!=mixed or len(plan['group_batches'])!=16
        or any(not 1<=len(b)<=4 for b in plan['group_batches'])):raise ValueError('one pass over all 62 mixed groups required')
    folder=ROOT/'data/preferences/dpo_v2'
    splits=load_preference_splits(folder,read(folder/'manifest.json'))
    train={r.id:r.target() for r in splits['train']};dev={r.id:r.target() for r in splits['development']}
    if plan['evaluation_development_case_ids']!=list(dev):raise ValueError('all 56 development cases required')
    if ({(c.system,c.prompt) for c in train.values()} & {(c.system,c.prompt) for c in dev.values()}):raise ValueError('train/dev prompt overlap')
    suites={'train_diagnostic':[train[k] for k in plan['training_train_diagnostic_case_ids']],
            'development':[dev[k] for k in plan['evaluation_development_case_ids']]}
    return plan,source,cfg,baseline,base,tok,groups,targets,suites


def load_model(base,adapter,trainable):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager')
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable,autocast_adapter_dtype=False).to('cuda').eval()
    if not trainable:model.requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 parameters required')
    if any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules()):raise ValueError('dropout must be zero')
    return model


def predict(model,tok,case):
    import torch
    ids=tok.apply_chat_template([{'role':'system','content':case.system},{'role':'user','content':case.prompt}],
        tokenize=True,add_generation_prompt=True)
    inputs=torch.tensor([ids],device=next(model.parameters()).device,dtype=torch.long)
    with torch.no_grad():
        output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,num_beams=1,
            use_cache=True,max_new_tokens=64,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    tokens=output[0,len(ids):].tolist()
    finish='eos' if tokens and tokens[-1]==tok.eos_token_id else 'length'
    text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return {'case_id':case.id,'category':case.category,'response_token_ids':tokens,'finish_reason':finish,
        'text':text,**score_reward(text,case,finish_reason=finish)}


def evaluate(model,tok,suites,path):
    results={}
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for name,cases in suites.items():
            results[name]=[]
            for i,c in enumerate(cases):
                row={'suite':name,**predict(model,tok,c)};results[name].append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                if (i+1)%16==0:print(f"Evaluation {path.stem} {name} {i+1}/{len(cases)}",flush=True)
    return results


def compare_evaluations(before,after):
    result={}
    for suite,left in before.items():
        right=after[suite]
        if [r['case_id'] for r in left]!=[r['case_id'] for r in right]:raise ValueError('evaluation identity differs')
        passed=lambda r:r['strict_pass'] and r['finish_reason']=='eos'
        transitions=[]
        for a,b in zip(left,right):
            if passed(a)!=passed(b) or a['text']!=b['text']:
                transitions.append({'case_id':a['case_id'],'before_strict':passed(a),'after_strict':passed(b),
                    'new_structure_failure':any(a['score'][k] and not b['score'][k] for k in ('json_valid','schema_valid','route_correct'))})
        result[suite]={'cases':len(left),'before_strict':sum(passed(r) for r in left),'after_strict':sum(passed(r) for r in right),
            'before_semantic':sum(r['semantic_pass'] for r in left),'after_semantic':sum(r['semantic_pass'] for r in right),
            'new_structure_failure_cases':sum(any(a['score'][k] and not b['score'][k] for k in ('json_valid','schema_valid','route_correct')) for a,b in zip(left,right)),
            'before_length_stops':sum(r['finish_reason']=='length' for r in left),'after_length_stops':sum(r['finish_reason']=='length' for r in right),
            'changed_cases':transitions}
    return result


def train(output):
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261027)
    plan,source,cfg,baseline,base,tok,groups,targets,suites=context()
    checkpoint_binding=hashlib.sha256((plan['source_summary_sha256']+file_hash(PLAN)).encode('ascii')).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    sources=read(source/'run_manifest.json')['source_sha256'].copy()
    for p in (Path(__file__),PLAN,ROOT/'src/agentic_post_training/trainers/grpo.py'):
        sources[str(p.relative_to(ROOT)).replace('\\','/')]=file_hash(p)
    inv={'plan':plan,'device':check_gpu(cfg),'source_summary_sha256':file_hash(source/'summary.json'),
        'source_manifest_sha256':file_hash(source/'run_manifest.json'),'source_sha256':sources,
        'input_rollout_unchanged':True,'baseline_manifest_sha256':file_hash(baseline/'manifest.json'),
        'checkpoint_input_digest':checkpoint_binding}
    write(output/'run_manifest.json',inv)
    for rel,digest in sources.items():
        if file_hash(ROOT/rel)!=digest:raise ValueError('source changed before training')
        dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    reference=load_model(base,baseline/'checkpoint/adapter',False)
    policy=load_model(base,baseline/'checkpoint/adapter',True)
    initial=parameter_digest(policy);ref_initial=parameter_digest(reference)
    expected=read(ROOT/'configs/trainer_probability_contract.json')['frozen_fp32_parameter_sha256']
    if initial!=expected or ref_initial!=expected or sum(p.numel() for p in lora_parameters(policy))!=230400:
        raise ValueError('original SFT/reference identity differs')
    frozen_initial=parameter_digest(policy,frozen_only=True)
    indexed={g[0].group_id:g for g in groups}
    selected=[s for batch in plan['group_batches'] for key in batch for s in indexed[key]]
    reference_cache={};initial_max_error=0.
    for s in selected:
        key=s.group_id,s.sample_index
        ref=score_response(reference,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu()
        fresh=score_response(policy,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu()
        old=torch.tensor(targets[key]['trainer_old_token_log_probs'],dtype=torch.float32)
        initial_max_error=max(initial_max_error,(fresh-old).abs().max().item(),(ref-old).abs().max().item())
        reference_cache[key]=ref
    if initial_max_error>1e-5:raise ValueError('current original policy/reference scores differ from stored old')
    before=evaluate(policy,tok,suites,output/'before_predictions.jsonl')
    optimizer=torch.optim.AdamW(lora_parameters(policy),lr=plan['learning_rate'],weight_decay=0.)
    history=[];probe=None;probe_snapshot=None;optimizer_calls=0;backward_calls=0;stopped=None
    with (output/'history.jsonl').open('x',encoding='utf-8',newline='\n') as log:
        for step,batch_ids in enumerate(plan['group_batches'],1):
            samples=[s for key in batch_ids for s in indexed[key]]
            metric=guarded_step(policy,optimizer,samples,targets,reference_cache,plan);backward_calls+=len(samples)
            metric.update(step=step,group_ids=batch_ids)
            if not metric['updated']:
                stopped=metric;log.write(json.dumps(metric,allow_nan=False)+'\n');log.flush();break
            optimizer_calls+=1
            if step==2:
                # Discard this uninterrupted branch after capturing its next-step result.
                probe_snapshot=adapter_snapshot(policy);probe_optimizer=optimizer_snapshot(optimizer);probe=metric.copy()
                del optimizer,policy;gc.collect();torch.cuda.empty_cache()
                policy=load_model(base,output/'checkpoint-step1/adapter',True)
                optimizer=torch.optim.AdamW(lora_parameters(policy),lr=plan['learning_rate'],weight_decay=0.)
                restored_step=restore_optimizer_checkpoint(policy,optimizer,output/'checkpoint-step1',checkpoint_binding)
                if restored_step!=1:raise ValueError('checkpoint step differs')
                resumed=guarded_step(policy,optimizer,samples,targets,reference_cache,plan);backward_calls+=len(samples)
                optimizer_calls+=int(resumed['updated'])
                resume_check=compare_adapter_snapshot(policy,probe_snapshot,1e-6)
                resume_check.update(compare_optimizer_snapshot(optimizer,probe_optimizer,1e-7))
                resume_check.update(uninterrupted_loss=probe['loss'],restored_loss=resumed['loss'],
                    restored_step=restored_step,extra_probe_optimizer_steps=1)
                if not resumed['updated'] or not resume_check['passed'] or not resume_check['optimizer_state_passed'] or abs(probe['loss']-resumed['loss'])>1e-6:
                    raise ValueError('restored next optimizer step differs from uninterrupted branch')
                write(output/'resume_check.json',resume_check)
                metric={**resumed,'step':step,'group_ids':batch_ids}
            history.append(metric);log.write(json.dumps(metric,allow_nan=False)+'\n');log.flush()
            print(json.dumps({'step':step,'loss':metric['loss'],'sampled_k3':metric['sampled_k3'],
                'clip_fraction':metric['ratio_clip_fraction'],'grad_norm':metric['grad_norm_before_clip']}),flush=True)
            if step==1:save_optimizer_checkpoint(policy,optimizer,output/'checkpoint-step1',step,checkpoint_binding)
    if not history:raise ValueError('no optimizer update completed')
    final=parameter_digest(policy);frozen_final=parameter_digest(policy,frozen_only=True)
    reference_final=parameter_digest(reference)
    intact=(frozen_initial==frozen_final and reference_final==ref_initial
        and all(not p.requires_grad and p.grad is None for p in reference.parameters())
        and all(p.grad is None for p in policy.parameters() if not p.requires_grad))
    if not intact or final==initial:raise ValueError('frozen/reference changed or policy did not update')
    save_optimizer_checkpoint(policy,optimizer,output/'final-checkpoint',len(history),checkpoint_binding)
    tok.save_pretrained(output/'tokenizer')
    after=evaluate(policy,tok,suites,output/'after_predictions.jsonl')
    expected_scores={}
    for s in selected[:8]:expected_scores[s.group_id,s.sample_index]=score_response(policy,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu()
    del policy,optimizer;gc.collect();torch.cuda.empty_cache()
    restored=load_model(base,output/'final-checkpoint/adapter',True)
    max_error=0.
    for s in selected[:8]:
        actual=score_response(restored,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu()
        max_error=max(max_error,(actual-expected_scores[s.group_id,s.sample_index]).abs().max().item())
    reload_token_checks=[]
    for i in [0,1,2,3,52,53,54,55]:
        c=suites['development'][i];actual=predict(restored,tok,c)
        reload_token_checks.append(actual['response_token_ids']==after['development'][i]['response_token_ids'])
    reloaded_digest=parameter_digest(restored)
    reload_passed=(max_error<=1e-5 and all(reload_token_checks) and reloaded_digest==final)
    write(output/'reload_check.json',{'passed':reload_passed,'max_abs_logp_error':max_error,
        'greedy_token_checks':reload_token_checks,'reloaded_parameter_sha256':reloaded_digest})
    resume_passed=(output/'resume_check.json').is_file() and read(output/'resume_check.json')['passed']
    comparison=compare_evaluations(before,after)
    summary={'contract_version':plan['contract_version'],'logical_optimizer_steps':len(history),
        'actual_optimizer_step_calls_including_discarded_probe':optimizer_calls,'model_backward_calls':backward_calls,
        'processed_candidate_groups':sum(len(r['group_ids']) for r in history),'available_mixed_groups':62,
        'source_groups':144,'source_samples':1152,'skipped_zero_variance_groups':82,
        'selected_group_policy':'all mixed groups once in a precommitted order; stop guard may shorten it',
        'initial_parameter_sha256':initial,'final_parameter_sha256':final,
        'frozen_parameters_unchanged':frozen_initial==frozen_final,'reference_unchanged':reference_final==ref_initial,
        'reference_frozen_and_no_grad':all(not p.requires_grad and p.grad is None for p in reference.parameters()),
        'initial_reference_old_max_logp_error':initial_max_error,'policy_changed':final!=initial,
        'resume_next_step_passed':resume_passed,'final_adapter_reload_passed':reload_passed,'early_stop':stopped,
        'engineering_checks_passed':bool(intact and final!=initial and resume_passed and reload_passed),
        'evaluation':comparison,'baseline_promoted':False,
        'behavior_safety_gate_passed':all(r['new_structure_failure_cases']==0 for r in comparison.values()) and comparison['development']['after_strict']>=comparison['development']['before_strict'],
        'policy_iterations':1,'rollout_refreshed_after_update':False,
        'original_cross_engine_gate_passed':False,'source_summary_sha256':plan['source_summary_sha256'],
        'monitor_note':'k3 is a sampled penalty on stored histories, not an unbiased full-policy KL; rollout reward is static',
        'files_sha256':{str(p.relative_to(output)).replace('\\','/'):file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('files_sha256','evaluation')}),flush=True)
    if not summary['engineering_checks_passed']:raise SystemExit(2)


def run(args):
    if args.output.exists() or ARCHIVE.exists():raise FileExistsError('new output/archive required; no automatic retry')
    price=args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
    seconds=run_seconds(price);started=time.perf_counter();error=None
    try:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','train','--output',str(args.output.resolve())],
            cwd=ROOT,start_new_session=(os.name!='nt'))
        try:code=proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
            else:proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
            raise TimeoutError('bounded training job stopped; no automatic retry')
        if code:raise RuntimeError(f'train/check/evaluate exited {code}')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if args.output.exists():
            elapsed=time.perf_counter()-started
            write(args.output/'cost_and_run.json',{'elapsed_run_seconds':elapsed,'hourly_price_yuan':price,
                'estimated_run_cost_yuan':elapsed*price/3600,'error':error,'max_run_seconds':seconds,
                'billing_scope':'job estimate only; boot/idle/storage/rounding excluded','instance_automatically_shutdown':False})
            with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(args.output.rglob('*')):
                    if p.is_file():z.write(p,str(p.relative_to(args.output.parent)))
            print(f'Download from project root: {ARCHIVE}',flush=True)
    if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('run','train'),default='run')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/checkpoints/grpo-first-20261027-01')
    parser.add_argument('--hourly-price',type=float)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.stage=='run':run(args)
    else:train(args.output)


if __name__=='__main__':main()
