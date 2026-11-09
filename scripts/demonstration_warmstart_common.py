"""Frozen inputs and strict reward admission for demonstration warm start plus fresh GRPO."""
from dataclasses import asdict
import math
import torch
import explore_grpo_sampling as exploration
from collect_vllm_rollout import ROOT, read, write
from graded_arithmetic_reward import grade_answer
from iteration_common import evaluation_suites
from agentic_post_training.algorithms.losses import group_relative_advantages
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files, digest_json

PLAN = ROOT / 'configs/experiments/grpo_demonstration_warmstart.json'
OUTPUT = ROOT / 'artifacts/checkpoints/grpo-demonstration-warmstart-20261028-01'
NEW_FILES = ('scripts/demonstration_warmstart_common.py', 'scripts/train_demonstration_warmstart.py',
             'scripts/run_demonstration_warmstart.py', 'configs/experiments/grpo_demonstration_warmstart.json')
FIXED = dict(contract_version='grpo-demonstration-warmstart-v1', sft_epochs=3,
    sft_batch_size=4, sft_learning_rate=5e-5, samples_per_group=32, groups_per_step=4,
    max_grpo_steps=6, learning_rate=1e-5, beta=.02, clip_epsilon=.2, tis_cap=2.,
    max_grad_norm=1., stop_sampled_k3=.05, stop_ratio_clip_fraction=.5,
    seed=202610300, max_wall_seconds=1800, max_run_cost_yuan=10.,
    reward_contract='verified-correct-binary-v1', auto_promote_baseline=False,
    grpo_optimizer_mode='carry_parent_31_adam', reference_mode='original_frozen_sft')

def encode_demonstration(example, tok):
    case = example.target()
    if grade_answer(example.chosen, case, finish_reason='eos')['verified_equivalent_reward'] != 1.:
        raise ValueError('a verified correct training demonstration is required')
    prompt = tok.apply_chat_template([{'role':'system','content':case.system},
        {'role':'user','content':case.prompt}], tokenize=True, add_generation_prompt=True)
    response = tok.encode(example.chosen, add_special_tokens=False)
    if not response or tok.eos_token_id in response or tok.decode(response,
        skip_special_tokens=True, clean_up_tokenization_spaces=False) != example.chosen:
        raise ValueError('demonstration tokenization is not lossless')
    response = response + [tok.eos_token_id]
    if len(response) > 64 or len(prompt) + len(response) > 512:
        raise ValueError('demonstration exceeds the native generation budget')
    ids = prompt + response
    return {'input_ids':ids, 'attention_mask':[1]*len(ids),
        'labels':[-100]*len(prompt)+response, 'loss_mask':[0]*len(prompt)+[1]*len(response)}

def context():
    ctx = exploration.setup()
    source_plan, cfg, parent, baseline, base, tok, selected, cases, prompts = ctx
    plan = read(PLAN)
    if any(plan.get(k) != v for k,v in FIXED.items()):
        raise ValueError('fixed demonstration experiment contract differs')
    if (plan['case_ids'] != [c.id for c in cases]
        or plan['parent_summary_sha256'] != file_hash(parent/'summary.json')
        or plan['parent_parameter_sha256'] != source_plan['candidate_parameter_sha256']
        or plan['parent_optimizer_sha256'] != file_hash(parent/'final-checkpoint/optimizer.pt')
        or plan['reward_scorer_sha256'] != file_hash(ROOT/'scripts/graded_arithmetic_reward.py')):
        raise ValueError('parent, training cases or reward scorer identity differs')
    encoded = [encode_demonstration(row,tok) for row in selected]
    suites = {'actual_training_cases':cases, **evaluation_suites(read(ROOT/'configs/grpo_policy_iteration.json'))}
    sources = read(parent/'run_manifest.json')['source_sha256'].copy()
    for rel in (*NEW_FILES, 'scripts/explore_grpo_sampling.py', 'configs/experiments/grpo_sampling_exploration.json',
                'scripts/graded_arithmetic_reward.py', 'scripts/reward_comparison_common.py'):
        sources[rel] = file_hash(ROOT/rel)
    verify_files(ROOT,sources)
    return dict(plan=plan,cfg=cfg,parent=parent,baseline=baseline,base=base,tok=tok,
        selected=selected,cases=cases,prompts=prompts,encoded=encoded,suites=suites,sources=sources)

def identity(ctx):
    return {'plan':ctx['plan'], 'source_sha256':ctx['sources'],
        'parent_summary_sha256':file_hash(ctx['parent']/'summary.json'),
        'baseline_manifest_sha256':file_hash(ctx['baseline']/'manifest.json'),
        'data_manifest_sha256':file_hash(ROOT/'data/preferences/dpo_v2/manifest.json'),
        'demonstrations_sha256':digest_json([{'case':asdict(r.target()),'chosen':r.chosen} for r in ctx['selected']]),
        'evaluation_case_ids':{k:[c.id for c in v] for k,v in ctx['suites'].items()}}

def choose_batches(groups, case_ids):
    if (len(groups)!=len(case_ids) or any(not g for g in groups) or [g[0].group_id for g in groups]!=case_ids
        or len(set(case_ids))!=len(case_ids)):
        raise ValueError('fresh groups must exactly match frozen case order')
    selected, records = [], []
    for group in groups:
        if len(group)!=32 or [s.sample_index for s in group]!=list(range(32)):
            raise ValueError('32 ordered native completions per group required')
        values=[s.reward for s in group] if hasattr(group[0],'reward') else None
        if values is None:
            raise ValueError('scored groups required')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v not in (0.,1.) for v in values):
            raise ValueError('only verified binary rewards are admitted')
        if len(set(values))==2: selected.append(group[0].group_id)
        records.append({'group_id':group[0].group_id,'rewards':values,'mixed':len(set(values))==2})
    return [selected[i:i+4] for i in range(0,len(selected),4)], records

def load_rollouts(root, ctx):
    folder=root/'rollout'
    summary=read(folder/'summary.json')
    verify_files(folder,summary['files_sha256'])
    warm=read(root/'warmstart/summary.json')
    if (summary['policy_parameter_sha256']!=warm['final_parameter_sha256']
        or summary['adapter_sha256']!=file_hash(root/'warmstart/final-checkpoint/adapter/adapter_model.safetensors')
        or summary['warmstart_summary_sha256']!=file_hash(root/'warmstart/summary.json')
        or summary['effective_engine']!=exploration.EFFECTIVE
        or summary['sampling_parameters']!=exploration.sampling(ctx['tok'].eos_token_id,1.)):
        raise ValueError('fresh rollout is not bound to the warm-start policy')
    rows=exploration.jsonl(folder/'rollouts.jsonl')
    if len(rows)!=768: raise ValueError('all 768 fresh native completions required')
    groups, rewards=[],{}
    for g,case in enumerate(ctx['cases']):
        group=[]
        for index in range(32):
            row=rows[g*32+index]
            if (row['case']!=asdict(case) or row['case_id']!=case.id or row['sample_index']!=index
                or row['prompt_token_ids']!=ctx['prompts'][g] or row['seed']!=ctx['plan']['seed']+1000*g+index
                or row['temperature']!=1. or row['arm']!='warmstart_t100'
                or row['behavior_token_log_probs']!=row['sampler_raw_model_token_log_probs']
                or row['training_admitted']):
                raise ValueError('fresh native record identity differs')
            sample=RolloutSample(case.id,case.id,index,row['seed'],row['prompt_token_ids'],row['response_token_ids'],
                row['sampler_raw_model_token_log_probs'],row['behavior_token_log_probs'],row['eos_token_id'],
                row['finish_reason'],row['text'],row['raw_text'])
            tok=ctx['tok']
            if (sample.eos_token_id!=tok.eos_token_id or len(sample.response_token_ids)>64
                or row['response_mask']!=[True]*len(sample.response_token_ids)
                or (sample.finish_reason=='length' and len(sample.response_token_ids)!=64)
                or tok.decode(sample.response_token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=sample.text
                or tok.decode(sample.response_token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=sample.raw_text):
                raise ValueError('native EOS, mask or decode differs')
            grade=grade_answer(sample.text,case,finish_reason=sample.finish_reason)
            rewards[case.id,index]=grade['verified_equivalent_reward']
            group.append(sample)
        groups.append(group)
    pack_rollout_groups(groups,ctx['tok'].pad_token_id)
    # Use a separate scored view; never modify native RolloutSample or forge teacher proposal probabilities.
    from types import SimpleNamespace
    views=[[SimpleNamespace(group_id=s.group_id,sample_index=s.sample_index,reward=rewards[s.group_id,s.sample_index])
            for s in group] for group in groups]
    batches,audit=choose_batches(views,ctx['plan']['case_ids'])
    targets={}
    for group in groups:
        values=[rewards[s.group_id,s.sample_index] for s in group]
        advantages=group_relative_advantages(torch.tensor([values],dtype=torch.float64))[0].tolist()
        for s,reward,adv in zip(group,values,advantages):
            targets[s.group_id,s.sample_index]={'reward':reward,'advantage':adv}
    return groups,targets,batches,audit

def compare_metrics(before,after):
    return {suite:{key:after[suite][key]-before[suite][key]
        for key in ('strict','verified_correct','completed_semantic','length_stops')} for suite in before}
