"""Independent GPU stages: supervised warm start, fresh vLLM rollout, bounded GRPO, audit."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import torch
from balanced_warmstart_common import (ROOT,OUTPUT,context,identity,load_rollouts,compare_metrics,demonstration_schedule)
from collect_vllm_rollout import read,write,effective_config
import explore_grpo_sampling as exploration
from reward_comparison_common import evaluation_metrics,import_comparison_adam
from train_grpo_first import load_model,evaluate,predict,compare_evaluations
from agentic_post_training.data import collate_sft
from agentic_post_training.rollouts.trainer_scores import score_response,no_update_metrics,sampler_trainer_difference
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.policy_iteration import file_hash,verify_files,digest_json
from agentic_post_training.trainers.grpo import (lora_parameters,guarded_step,save_optimizer_checkpoint,
    restore_optimizer_checkpoint,adapter_snapshot,optimizer_snapshot,compare_adapter_snapshot,compare_optimizer_snapshot)

def configure(ctx):
    exploration.check_device(ctx['cfg'])
    torch.manual_seed(ctx['plan']['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def seal(folder,values):
    write(folder/'summary.json',{**values,'files_sha256':{
        p.relative_to(folder).as_posix():file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}})

def verified_summary(folder):
    summary=read(folder/'summary.json')
    verify_files(folder,summary['files_sha256'])
    return summary

def validate_predictions(path,ctx):
    from agentic_post_training.rewards.contract import score_reward
    rows=exploration.jsonl(path)
    result={}
    offset=0
    for suite,cases in ctx['suites'].items():
        result[suite]=rows[offset:offset+len(cases)]
        offset+=len(cases)
        if len(result[suite])!=len(cases):raise ValueError('incomplete evaluation')
        for row,case in zip(result[suite],cases):
            tokens=row['response_token_ids']
            if (row['suite']!=suite or row['case_id']!=case.id or not tokens or len(tokens)>64
                or ctx['tok'].eos_token_id in tokens[:-1]
                or (tokens[-1]==ctx['tok'].eos_token_id)!=(row['finish_reason']=='eos')
                or row['finish_reason'] not in ('eos','length')
                or (row['finish_reason']=='length' and len(tokens)!=64)
                or row['text']!=ctx['tok'].decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)):
                raise ValueError('evaluation case/tokens/finish identity differs')
            if any(row[k]!=v for k,v in score_reward(row['text'],case,finish_reason=row['finish_reason']).items()):
                raise ValueError('evaluation reward differs')
    if offset!=len(rows):raise ValueError('unexpected evaluation records')
    return result

def check_reload(policy,ctx,folder,predictions):
    digest=parameter_digest(policy)
    probes=ctx['encoded'][:2]
    prompt_lengths=[row['loss_mask'].index(1) for row in probes]
    scores=[score_response(policy,row['input_ids'][:n],row['input_ids'][n:],grad_enabled=False).cpu()
            for row,n in zip(probes,prompt_lengths)]
    reloaded=load_model(ctx['base'],folder/'final-checkpoint/adapter',True)
    error=max((score_response(reloaded,row['input_ids'][:n],row['input_ids'][n:],grad_enabled=False).cpu()-old).abs().max().item()
        for row,n,old in zip(probes,prompt_lengths,scores))
    tokens_match=all(predict(reloaded,ctx['tok'],cases[0])['response_token_ids']==predictions[suite][0]['response_token_ids']
        for suite,cases in ctx['suites'].items())
    if parameter_digest(reloaded)!=digest or error>1e-5 or not tokens_match:
        raise ValueError('saved adapter reload differs')
    del reloaded
    gc.collect();torch.cuda.empty_cache()
    return {'passed':True,'max_abs_logp_error':error,'greedy_tokens_match':tokens_match}

def sft_step(policy,optimizer,examples,pad):
    policy.eval()  # All dropout is zero, same eager execution as the probability gates.
    optimizer.zero_grad(set_to_none=True)
    counts=[sum(row['loss_mask'][1:]) for row in examples]
    total=sum(counts)
    loss_value=0.
    device=next(policy.parameters()).device
    for row,count in zip(examples,counts):
        batch={k:v.to(device) for k,v in collate_sft([row],pad_token_id=pad).items()}
        loss=policy(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'],use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss')
        (loss*(count/total)).backward()
        loss_value+=loss.detach().item()*count/total
    norm=torch.nn.utils.clip_grad_norm_(lora_parameters(policy),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('nonzero finite supervised gradient required')
    optimizer.step()
    return {'loss':loss_value,'response_tokens':total,'grad_norm_before_clip':norm.item()}

def warmstart(root):
    ctx=context();configure(ctx)
    inv=identity(ctx)
    write(root/'run_manifest.json',inv)
    for relative in inv['source_sha256']:
        dest=root/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,dest)
    folder=root/'warmstart';folder.mkdir()
    policy=load_model(ctx['base'],ctx['parent']/'final-checkpoint/adapter',True)
    initial=parameter_digest(policy);frozen=parameter_digest(policy,frozen_only=True)
    if initial!=ctx['plan']['parent_parameter_sha256']:raise ValueError('parent full parameter identity differs')
    before=evaluate(policy,ctx['tok'],ctx['suites'],folder/'a_predictions.jsonl')
    # Anchor original fixed suites to historical CUDA outputs before any new training.
    saved=exploration.jsonl(ctx['parent']/'current_predictions.jsonl')
    for suite in ('train_diagnostic','development'):
        previous=[r for r in saved if r['suite']==suite]
        if [r['response_token_ids'] for r in previous]!=[r['response_token_ids'] for r in before[suite]]:
            raise ValueError('A fixed CUDA greedy predictions differ from parent')
    optimizer=torch.optim.AdamW(lora_parameters(policy),lr=ctx['plan']['sft_learning_rate'],weight_decay=0.)
    binding=digest_json({'input':inv,'stage':'supervised'})
    history=[]
    schedule=demonstration_schedule(len(ctx['selected']),ctx['plan']['sft_epochs'],ctx['plan']['seed'])
    with (folder/'history.jsonl').open('x',encoding='utf8') as stream:
        for epoch,indices in schedule:
            row={'step':len(history)+1,'epoch':epoch,'case_ids':[ctx['selected'][i].id for i in indices],
                **sft_step(policy,optimizer,[ctx['encoded'][i] for i in indices],ctx['tok'].pad_token_id)}
            history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(json.dumps({'stage':'warmstart',**row}),flush=True)
    final=parameter_digest(policy)
    if parameter_digest(policy,frozen_only=True)!=frozen or final==initial:raise ValueError('SFT parameter invariant failed')
    save_optimizer_checkpoint(policy,optimizer,folder/'final-checkpoint',18,binding)
    after=evaluate(policy,ctx['tok'],ctx['suites'],folder/'b_predictions.jsonl')
    reload_check=check_reload(policy,ctx,folder,after)
    comparison=compare_evaluations(before,after)
    behavior_safe=all(v['new_structure_failure_cases']==0 for v in comparison.values())
    behavior_safe=behavior_safe and evaluation_metrics(after,ctx['suites'])['development']['verified_correct']>=evaluation_metrics(before,ctx['suites'])['development']['verified_correct']
    behavior_safe=behavior_safe and comparison['development']['after_strict']>=comparison['development']['before_strict']
    verify_files(ROOT,inv['source_sha256'])
    seal(folder,{'engineering_checks_passed':True,'initial_parameter_sha256':initial,'final_parameter_sha256':final,
        'frozen_parameter_sha256':frozen,'frozen_parameters_unchanged':True,'logical_sft_steps':18,
        'sft_optimizer_mode':'fresh AdamW; parent GRPO optimizer preserved separately','adam_step_at_end':18,
        'reload_check':reload_check,'before':evaluation_metrics(before,ctx['suites']),'after':evaluation_metrics(after,ctx['suites']),
        'comparison_to_a':comparison,'behavior_safety_gate_passed':behavior_safe,'baseline_promoted':False})
    if not behavior_safe:raise RuntimeError('warm start behavior safety gate failed; no sampling or GRPO')

def verify_run(root,ctx):
    inv=read(root/'run_manifest.json')
    if inv!=identity(ctx):raise ValueError('run input identity differs')
    verify_files(root/'source',inv['source_sha256'])

def generate(root):
    ctx=context();configure(ctx);verify_run(root,ctx)
    warm=verified_summary(root/'warmstart')
    if not warm['engineering_checks_passed'] or not warm['behavior_safety_gate_passed']:
        raise ValueError('verified safe warm start required')
    folder=root/'rollout';folder.mkdir()
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    import torch.distributed as dist
    engine=LLM(model=str(ctx['base']),tokenizer=str(ctx['baseline']/'checkpoint/tokenizer'),dtype='half',
        trust_remote_code=False,enable_lora=True,max_lora_rank=8,max_loras=1,max_model_len=512,
        max_num_seqs=4,max_num_batched_tokens=512,gpu_memory_utilization=.25,enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=True,seed=ctx['plan']['seed'],
        generation_config='vllm',logprobs_mode='raw_logprobs')
    try:
        effective=effective_config(engine)
        if effective!=exploration.EFFECTIVE:raise ValueError('effective native engine configuration differs')
        adapter=root/'warmstart/final-checkpoint/adapter'
        request=LoRARequest('correct-demonstration-warmstart',1,str(adapter))
        shared=exploration.sampling(ctx['tok'].eos_token_id,1.)
        entries=[(g,i,ctx['plan']['seed']+1000*g+i) for g in range(24) for i in range(32)]
        with (folder/'rollouts.jsonl').open('x',encoding='utf8') as stream:
            for start in range(0,768,64):
                chunk=entries[start:start+64]
                outputs=engine.generate([{'prompt_token_ids':ctx['prompts'][g]} for g,i,seed in chunk],
                    [SamplingParams(seed=seed,**shared) for g,i,seed in chunk],lora_request=request,use_tqdm=False)
                if len(outputs)!=len(chunk):raise ValueError('incomplete native outputs')
                for output,(g,i,seed) in zip(outputs,chunk):
                    row=exploration.native_record(output,ctx['prompts'][g],ctx['cases'][g],'warmstart_t100',i,seed,1.,ctx['tok'])
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush();print(f'Fresh warm-start rollout {start+len(chunk)}/768',flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized():dist.destroy_process_group()
    seal(folder,{'policy_parameter_sha256':warm['final_parameter_sha256'],
        'adapter_sha256':file_hash(adapter/'adapter_model.safetensors'),
        'warmstart_summary_sha256':file_hash(root/'warmstart/summary.json'),
        'effective_engine':effective,'sampling_parameters':shared,'samples':768,'engine_shutdown_called':True})
    load_rollouts(root,ctx)

def grpo(root):
    ctx=context();configure(ctx);verify_run(root,ctx)
    warm=verified_summary(root/'warmstart')
    groups,targets,batches,reward_audit=load_rollouts(root,ctx)
    folder=root/'grpo';folder.mkdir()
    write(folder/'reward_audit.json',{'groups':reward_audit,'batches':batches,'contract':ctx['plan']['reward_contract']})
    policy=load_model(ctx['base'],root/'warmstart/final-checkpoint/adapter',True)
    reference=load_model(ctx['base'],ctx['baseline']/'checkpoint/adapter',False)
    initial=parameter_digest(policy);frozen=parameter_digest(policy,frozen_only=True)
    reference_digest=parameter_digest(reference)
    expected_reference=read(ROOT/'configs/trainer_probability_contract.json')['frozen_fp32_parameter_sha256']
    if initial!=warm['final_parameter_sha256'] or reference_digest!=expected_reference:
        raise ValueError('warm-start policy or original frozen reference differs')
    references={};cached=[]
    with (folder/'trainer_scores.jsonl').open('x',encoding='utf8') as stream:
        for group in groups:
            for sample in group:
                key=sample.group_id,sample.sample_index
                old=score_response(policy,sample.prompt_token_ids,sample.response_token_ids,grad_enabled=False)
                new=score_response(policy,sample.prompt_token_ids,sample.response_token_ids,grad_enabled=True)
                check=no_update_metrics(old,new,1e-5)
                if not check['passed']:raise ValueError('warm-start old/new probability gate failed')
                ref=score_response(reference,sample.prompt_token_ids,sample.response_token_ids,grad_enabled=False).cpu()
                targets[key]['trainer_old_token_log_probs']=old.cpu().tolist()
                references[key]=ref
                row={'group_id':sample.group_id,'sample_index':sample.sample_index,
                    'response_token_ids':sample.response_token_ids,'behavior_token_log_probs':sample.behavior_token_log_probs,
                    **targets[key],'reference_token_log_probs':ref.tolist(),
                    'trainer_new_token_log_probs':new.detach().cpu().tolist(),'no_update_check':check,
                    'sampler_trainer_difference':sampler_trainer_difference(sample.behavior_token_log_probs,old.cpu().tolist())}
                cached.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                del old,new,ref
            stream.flush()
            print(f'Fresh trainer score group {len(cached)//32}/24',flush=True)
    if parameter_digest(policy)!=initial or parameter_digest(reference)!=reference_digest or any(p.grad is not None for p in policy.parameters()):
        raise ValueError('probability preparation changed model or gradients')
    del reference;gc.collect();torch.cuda.empty_cache()
    optimizer=torch.optim.AdamW(lora_parameters(policy),lr=ctx['plan']['learning_rate'],weight_decay=0.)
    binding=digest_json({'input':identity(ctx),'warmstart':file_hash(root/'warmstart/summary.json'),
        'rollout':file_hash(root/'rollout/summary.json'),'reward_audit':reward_audit,'batches':batches})
    lineage=import_comparison_adam(policy,optimizer,ctx['parent']/'final-checkpoint/optimizer.pt',
        expected_hash=ctx['plan']['parent_optimizer_sha256'],
        old_binding=read(ctx['parent']/'run_manifest.json')['checkpoint_input_digest'],new_binding=binding)
    write(folder/'optimizer_lineage.json',lineage)
    save_optimizer_checkpoint(policy,optimizer,folder/'initial-checkpoint',0,binding)
    indexed={g[0].group_id:g for g in groups}
    history=[];stopped=None;resume=None;backwards=0;calls=0
    with (folder/'history.jsonl').open('x',encoding='utf8') as stream:
        for step,batch in enumerate(batches,1):
            samples=[s for key in batch for s in indexed[key]]
            metric=guarded_step(policy,optimizer,samples,targets,references,ctx['plan'])
            backwards+=len(samples);calls+=int(metric['updated'])
            if not metric['updated']:
                stopped={'step':step,'group_ids':batch,**metric};stream.write(json.dumps(stopped)+'\n');break
            if step==1:
                expected_adapter=adapter_snapshot(policy);expected_optimizer=optimizer_snapshot(optimizer)
                loss=metric['loss']
                del policy,optimizer;gc.collect();torch.cuda.empty_cache()
                policy=load_model(ctx['base'],folder/'initial-checkpoint/adapter',True)
                optimizer=torch.optim.AdamW(lora_parameters(policy),lr=ctx['plan']['learning_rate'],weight_decay=0.)
                if restore_optimizer_checkpoint(policy,optimizer,folder/'initial-checkpoint',binding)!=0:
                    raise ValueError('GRPO initial restore local step differs')
                replay=guarded_step(policy,optimizer,samples,targets,references,ctx['plan'])
                backwards+=len(samples);calls+=int(replay['updated'])
                resume={**compare_adapter_snapshot(policy,expected_adapter,1e-6),
                    **compare_optimizer_snapshot(optimizer,expected_optimizer,1e-7),'loss_abs_error':abs(loss-replay['loss'])}
                if not resume['passed'] or not resume['optimizer_state_passed'] or resume['loss_abs_error']>1e-6 or not replay['updated']:
                    raise ValueError('GRPO resume replay differs')
                metric=replay
            if any(v['step'].item()!=31+step for v in optimizer.state.values()):
                raise ValueError('GRPO cumulative Adam step differs')
            row={'step':step,'group_ids':batch,**metric};history.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();print(json.dumps({'stage':'grpo',**row}),flush=True)
    final=parameter_digest(policy)
    if parameter_digest(policy,frozen_only=True)!=frozen:raise ValueError('GRPO changed frozen base')
    save_optimizer_checkpoint(policy,optimizer,folder/'final-checkpoint',len(history),binding)
    after=evaluate(policy,ctx['tok'],ctx['suites'],folder/'c_predictions.jsonl')
    reload_check=check_reload(policy,ctx,folder,after)
    before=validate_predictions(root/'warmstart/b_predictions.jsonl',ctx)
    comparison=compare_evaluations(before,after)
    verify_files(ROOT,ctx['sources'])
    seal(folder,{'engineering_checks_passed':True,'initial_parameter_sha256':initial,'final_parameter_sha256':final,
        'reference_parameter_sha256':reference_digest,'frozen_parameters_unchanged':True,'logical_grpo_steps':len(history),
        'planned_steps':len(batches),'mixed_groups':sum(r['mixed'] for r in reward_audit),'early_stop':stopped,
        'adam_step_at_start':31,'adam_step_at_end':31+len(history),'sft_moments_imported':False,
        'actual_optimizer_step_calls_including_discarded_probe':calls,'model_backward_calls':backwards,
        'resume_check':resume,'reload_check':reload_check,'after':evaluation_metrics(after,ctx['suites']),
        'comparison_to_b':comparison,'baseline_promoted':False,'checkpoint_input_digest':binding,
        'no_signal_skip':not bool(batches)})

def audit(root):
    ctx=context()
    inv=read(root/'run_manifest.json')
    if inv!=identity(ctx):raise ValueError('run identity differs')
    verify_files(root/'source',inv['source_sha256']);verify_files(ROOT,inv['source_sha256'])
    warm=verified_summary(root/'warmstart');rl=verified_summary(root/'grpo')
    groups,targets,batches,rewards=load_rollouts(root,ctx)
    if read(root/'grpo/reward_audit.json')!={'groups':rewards,'batches':batches,'contract':ctx['plan']['reward_contract']}:
        raise ValueError('binary reward audit differs')
    if not warm['engineering_checks_passed'] or not warm['behavior_safety_gate_passed'] or not rl['engineering_checks_passed']:
        raise ValueError('engineering or warm-start safety gate failed')
    supervised=exploration.jsonl(root/'warmstart/history.jsonl')
    schedule=demonstration_schedule(36,2,ctx['plan']['seed'])
    expected_batches=[[ctx['selected'][i].id for i in indices] for epoch,indices in schedule]
    if (len(supervised)!=18 or warm['initial_parameter_sha256']!=ctx['plan']['parent_parameter_sha256']
        or warm['final_parameter_sha256']==warm['initial_parameter_sha256']
        or any(r['step']!=i or r['epoch']!=(i-1)//9+1 or r['case_ids']!=expected_batches[i-1]
               for i,r in enumerate(supervised,1))):
        raise ValueError('supervised schedule or starting policy differs')
    a=validate_predictions(root/'warmstart/a_predictions.jsonl',ctx)
    b=validate_predictions(root/'warmstart/b_predictions.jsonl',ctx)
    c=validate_predictions(root/'grpo/c_predictions.jsonl',ctx)
    metrics=[evaluation_metrics(rows,ctx['suites']) for rows in (a,b,c)]
    if warm['before']!=metrics[0] or warm['after']!=metrics[1] or rl['after']!=metrics[2]:
        raise ValueError('evaluation metrics differ from recomputation')
    if warm['comparison_to_a']!=compare_evaluations(a,b) or rl['comparison_to_b']!=compare_evaluations(b,c):
        raise ValueError('case transition comparison differs')
    binding=digest_json({'input':identity(ctx),'warmstart':file_hash(root/'warmstart/summary.json'),
        'rollout':file_hash(root/'rollout/summary.json'),'reward_audit':rewards,'batches':batches})
    history=exploration.jsonl(root/'grpo/history.jsonl')
    updated=[row for row in history if row['updated']]
    if (len(updated)!=rl['logical_grpo_steps'] or len(updated)>len(batches) or len(batches)>6
        or any(row['step']!=i or row['group_ids']!=batches[i-1] for i,row in enumerate(history,1))
        or rl['initial_parameter_sha256']!=warm['final_parameter_sha256']
        or rl['checkpoint_input_digest']!=binding or rl['planned_steps']!=len(batches)
        or rl['mixed_groups']!=sum(r['mixed'] for r in rewards)
        or bool(rl['no_signal_skip'])!=bool(not batches)
        or (updated and (not rl['resume_check'] or not rl['resume_check']['passed'] or not rl['resume_check']['optimizer_state_passed']))):
        raise ValueError('GRPO history, phase lineage or budget differs')
    for phase,count,step,digest in [('warmstart',18,18,digest_json({'input':inv,'stage':'supervised'})),
                                  ('grpo',len(updated),31+len(updated),binding)]:
        saved=torch.load(root/phase/'final-checkpoint/optimizer.pt',map_location='cpu',weights_only=True)
        if (saved['step']!=count or saved['input_digest']!=digest
            or not saved['optimizer']['state'] or any(v['step'].item()!=step for v in saved['optimizer']['state'].values())):
            raise ValueError('phase checkpoint optimizer counters/binding differ')
    records=exploration.jsonl(root/'grpo/trainer_scores.jsonl')
    if len(records)!=768:raise ValueError('complete old/new/reference cache required')
    for sample,row in zip([s for group in groups for s in group],records):
        key=sample.group_id,sample.sample_index
        old=torch.tensor(row['trainer_old_token_log_probs'],dtype=torch.float32)
        new=torch.tensor(row['trainer_new_token_log_probs'],dtype=torch.float32,requires_grad=True)
        ref=torch.tensor(row['reference_token_log_probs'],dtype=torch.float32)
        if ((row['group_id'],row['sample_index'])!=key or row['response_token_ids']!=sample.response_token_ids
            or row['behavior_token_log_probs']!=sample.behavior_token_log_probs
            or any(row[k]!=v for k,v in targets[key].items())
            or old.shape!=(len(sample.response_token_ids),) or ref.shape!=old.shape or not torch.isfinite(ref).all()
            or (ref>1e-6).any() or no_update_metrics(old,new,1e-5)!=row['no_update_check'] or not row['no_update_check']['passed']):
            raise ValueError('fresh probability cache/admission differs')
    comparison={'artifact_audit_passed':True,'gpu_reexecuted_by_audit':False,
        'a_parent':metrics[0],'b_sft_only':metrics[1],'c_sft_then_grpo':metrics[2],
        'sft_increment_b_minus_a':compare_metrics(metrics[0],metrics[1]),
        'grpo_increment_c_minus_b':compare_metrics(metrics[1],metrics[2]),
        'combined_increment_c_minus_a':compare_metrics(metrics[0],metrics[2]),
        'mixed_groups':rl['mixed_groups'],'logical_sft_steps':18,'logical_grpo_steps':len(updated),
        'grpo_early_stop':rl['early_stop'],'no_signal_skip':rl['no_signal_skip'],
        'claim_limit':'One seed, targeted training cases; no significance or GRPO improvement claim from SFT gains.',
        'baseline_promoted':False,'automatic_next_phase':False,
        'grpo_behavior_safety_gate_passed':all(v['new_structure_failure_cases']==0 for v in rl['comparison_to_b'].values())
            and metrics[2]['development']['strict']>=metrics[1]['development']['strict']
            and metrics[2]['development']['verified_correct']>=metrics[1]['development']['verified_correct']}
    write(root/'comparison.json',comparison)
    return comparison

if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_NO_USAGE_STATS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('warmstart','generate','grpo','audit'),required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    globals()[args.stage](args.output)
