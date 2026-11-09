"""Isolated single-adapter paired development evaluation at frozen group cursors."""
import argparse
import copy
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from grpo_common import (read, atomic_json, jsonl, file_hash, verify_files, verify_inputs, under)
from apt_v2.grpo import run_continuation, replay, initial_state, validate_native
from apt_v2.grpo_evaluation import (development_tasks, evaluation_plan, evaluation_key,
                                    plan_key, summarize_evaluation)
from apt_v2.dpo_recovery import SCENARIOS, build_context
from apt_v2.independent_recovery_test import calendar_episode_audit


def study_context(cfg):
    if cfg.get('curriculum_schema')=='repair_study_v1':
        from apt_v2.grpo_repair_study import context
        return context
    return build_context


def quality_specs(cfg):
    task = development_tasks(cfg)[0]
    return [{'task_id': task.id, 'scenario': s} for s in SCENARIOS[:cfg['protocol']['quality_contexts_per_model']]]


def finish_quality_check(output,label,spec,check,targeted_check):
    """A rejected alias model remains rejected while the other arm is evaluated."""
    cfg=read(output/'config.json')
    if cfg.get('curriculum_schema') not in ('alias_curriculum_v1','repair_study_v1'):
        if not check['passed']:
            raise ValueError('evaluation quality/calendar gate failed: '+str(check['checks']))
        return
    failed=[key for key,passed in check['checks'].items() if not passed]
    if targeted_check is not None and not targeted_check['passed']:
        failed.append('targeted_calendar_final_evidence')
    status={'passed':not failed,'structural_verification_passed':True,'failed_checks':failed,
            'config_sha256':file_hash(output/'config.json'),'policy_sha256':spec['policy']['adapter_sha256'],
            'evaluation_sha256':file_hash(output/'evaluations'/(label+'.json')),
            'original_quality_thresholds_preserved':True}
    atomic_json(output/'evaluations'/(label+'-quality-status.json'),status)
    print('Evaluation '+label+': quality '+('PASSED' if status['passed'] else 'REJECTED; saved outcome retained'),flush=True)
    return status


def verify_evaluation(output, label, spec, tok, local=False, data=None):
    cfg = verify_inputs(output, runtime=False, local=local)
    if not re.fullmatch(r'weighted_sft|grpo_g[0-9]{4}', label):
        raise ValueError('safe frozen evaluation label required')
    data = data if data is not None else read(output/'evaluations'/(label+'.json'))
    policy = spec['policy']
    adapter = under(output, policy['checkpoint'])/'adapter'
    if (data['spec'] != spec or data['config_sha256'] != file_hash(output/'config.json')
            or file_hash(adapter/'adapter_model.safetensors') != policy['adapter_sha256']):
        raise ValueError('evaluation model/config binding differs')
    plan = evaluation_plan(cfg)
    if read(output/'evaluation_plan.json') != plan:
        raise ValueError('frozen development plan differs')
    episodes = data['episodes']
    expected = {plan_key(p): p for p in plan}
    actual = {evaluation_key(e): e for e in episodes}
    if len(actual) != len(episodes) or actual.keys() != expected.keys():
        raise ValueError('distinct complete development coverage required')
    tasks = {t.id: t for t in development_tasks(cfg)}
    calendar = []
    for key, e in actual.items():
        item = expected[key]
        s = {**cfg['sampling'], 'temperature': item['temperature']}
        if (e['sampling'] != s or e['seed'] != item['seed'] or e['policy_version'] != policy['version']
                or e['model_branch'] != label or e['split'] != 'development'):
            raise ValueError('paired evaluation seed/sampling/version differs')
        task = tasks[e['task_id']]
        replay(task, e, tok, study_context(cfg))
        records = [p['row'] for p in e['start']['prefix']]+e['turns']
        audit = calendar_episode_audit(task, records, e['success'], e['artifact'], allowed_split='development')
        calendar.append({'key': list(key), 'result': audit})
    quality = data['quality']
    qs = quality_specs(cfg)
    if len(quality) != len(qs):
        raise ValueError('fixed quality context count differs')
    qreport = []
    for row, item in zip(quality, qs):
        if row['task_id'] != item['task_id'] or row['scenario'] != item['scenario'] or len(row['generations']) != 2:
            raise ValueError('fixed repeated greedy quality inputs differ')
        context = study_context(cfg)(tasks[item['task_id']], item['scenario'], allowed_split='development')
        s = {**cfg['sampling'], 'temperature': 0}
        seed = next(p['seed'] for p in plan if p['task_id'] == item['task_id'] and p['scenario'] == item['scenario'] and p['mode'] == 'greedy')
        if row['seed'] != seed:
            raise ValueError('quality input seed differs')
        for generation in row['generations']:
            validate_native(tok, context['messages'], generation, s)
        left, right = row['generations']
        identical = left['response_token_ids'] == right['response_token_ids']
        qreport.append({'task_id': item['task_id'], 'scenario': item['scenario'], 'identical_tokens': identical})
    proof = data['worker_proof']
    if not proof or any(not p['passed'] or p['adapter_sha256'] != policy['adapter_sha256']
                        or p['registered_ids'] != [1] or 1 not in p['active_slots'] or p['tensor_count'] != 112 for p in proof):
        raise ValueError('evaluation worker checkpoint proof differs')
    from grpo_vllm import projected_checkpoint_tensors
    from apt_v2.grpo import digest
    projected = projected_checkpoint_tensors(adapter)
    if any(p['tensors'] != projected or p['loaded_tensor_sha256'] != digest(projected) for p in proof):
        raise ValueError('independent evaluation worker tensor projection differs')
    summary = summarize_evaluation(episodes,cfg)
    if data['summary'] != summary:
        raise ValueError('evaluation summary recomputation differs')
    checks = {'complete_episode_coverage': len(episodes) == len(plan),
              'repeated_greedy_tokens_identical': all(q['identical_tokens'] for q in qreport),
              'calendar_final_evidence': all(not x['result']['environment_success'] or x['result']['calendar_consistent_success'] for x in calendar)}
    return {'passed': all(checks.values()), 'checks': checks, 'summary': summary,
            'quality': qreport, 'calendar': calendar,
            'episodes_replayed': len(episodes), 'actions_replayed': sum(len(e['turns']) for e in episodes)}


def evaluate(output, label, spec_path):
    from tqdm.auto import tqdm
    from grpo_vllm import engine_and_tokenizer, BehaviorPolicy, shutdown
    from transformers import AutoTokenizer
    cfg = verify_inputs(output)
    if cfg['phase'] != 'formal' or not re.fullmatch(r'weighted_sft|grpo_g[0-9]{4}', label):
        raise ValueError('formal frozen evaluation required')
    spec = read(spec_path)
    if spec['label'] != label:
        raise ValueError('evaluation label differs')
    target = output/'evaluations'/(label+'.json')
    if target.exists():
        tok = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True, trust_remote_code=False)
        result = verify_evaluation(output, label, spec, tok)
        if not result['passed'] and cfg.get('curriculum_schema') not in ('alias_curriculum_v1','repair_study_v1'):
            raise ValueError('committed evaluation quality check failed')
        verify_closing_evaluation(output,label,spec,tok)
        targeted_check=verify_targeted_evaluation(output,label,spec,tok)
        finish_quality_check(output,label,spec,result,targeted_check)
        return
    adapter = under(output, spec['policy']['checkpoint'])/'adapter'
    if file_hash(adapter/'adapter_model.safetensors') != spec['policy']['adapter_sha256']:
        raise ValueError('evaluation checkpoint hash differs')
    engine_cfg = copy.deepcopy(cfg)
    engine_cfg['seed'] = cfg['eval']['seed']
    engine_cfg['inference']['gpu_memory_utilization'] = cfg['eval']['gpu_memory_utilization']
    engine = None
    started = time.perf_counter()
    episodes, quality, closing_episodes, targeted_episodes = [], [], [], []
    from apt_v2.grpo_curriculum import targeted_plan,summarize_targeted
    target_plan=targeted_plan(cfg)
    plan = evaluation_plan(cfg)
    tasks = {t.id: t for t in development_tasks(cfg)}
    partial = output/'evaluations'/f'{label}-partial-{time.time_ns()}.jsonl'
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine, tok, lora = engine_and_tokenizer(engine_cfg, adapter)
        with tqdm(total=len(plan)+len(target_plan)+2*len(quality_specs(cfg))+(4 if cfg.get('closing_evaluation',{}).get('enabled') else 0), desc='Evaluate '+label,
                  unit='episode/action', file=sys.stdout, disable=False, dynamic_ncols=True) as bar:
            for item in plan:
                sampling = {**cfg['sampling'], 'temperature': item['temperature']}
                def show_action(count):
                    bar.set_postfix(kind=item['kind'], mode=item['mode'], turn=policy.calls, response_tokens=count, refresh=True)
                policy = BehaviorPolicy(engine, tok, lora, sampling, item['seed'], show_action)
                try:
                    episode = run_continuation(tasks[item['task_id']], item['scenario'], policy, sampling,
                                               spec['policy']['version'], item['seed'], item['rollout_index'], cfg['tool_cost'], study_context(cfg))
                except Exception as exc:
                    jsonl(partial, getattr(exc, 'episode_evidence', {'error': str(exc), 'status': 'interrupted'}))
                    raise
                episode.update(model_branch=label, evaluation_mode=item['mode'])
                jsonl(partial, episode)
                replay(tasks[item['task_id']], episode, tok, study_context(cfg))
                episodes.append(episode)
                bar.set_postfix(kind=item['kind'], mode=item['mode'], success=episode['success'], refresh=False)
                bar.update(1)
            for item in quality_specs(cfg):
                task = tasks[item['task_id']]
                context = study_context(cfg)(task, item['scenario'], allowed_split='development')
                seed = next(p['seed'] for p in plan if p['task_id'] == task.id and p['scenario'] == item['scenario'] and p['mode'] == 'greedy')
                generations = []
                for _ in range(2):
                    policy = BehaviorPolicy(engine, tok, lora, {**cfg['sampling'], 'temperature': 0}, seed)
                    generations.append(asdict(policy(context['messages'])))
                    bar.set_postfix(kind='quality', mode='greedy', refresh=False)
                    bar.update(1)
                quality.append({**item, 'seed': seed, 'generations': generations})
                jsonl(partial, {'quality': quality[-1]})
            if cfg.get('curriculum_schema'):
                from apt_v2.grpo_curriculum import build_context as target_context
                for item in target_plan:
                    sampling={**cfg['sampling'],'temperature':item['temperature']}
                    policy=BehaviorPolicy(engine,tok,lora,sampling,item['seed'],
                        lambda n:bar.set_postfix(kind='targeted',mode=item['mode'],turn=policy.calls,response_tokens=n,refresh=True))
                    try:
                        episode=run_continuation(tasks[item['task_id']],item['scenario'],policy,sampling,
                            spec['policy']['version'],item['seed'],item['rollout_index'],cfg['tool_cost'],target_context)
                    except Exception as exc:
                        jsonl(partial,getattr(exc,'episode_evidence',{'status':'interrupted','error':str(exc)}));raise
                    episode.update(model_branch=label,evaluation_mode=item['mode'])
                    replay(tasks[item['task_id']],episode,tok,target_context)
                    targeted_episodes.append(episode);jsonl(partial,{'targeted':episode});bar.update(1)
            if cfg.get('closing_evaluation',{}).get('enabled'):
                from apt_v2.period_closing import closing_tasks,run_closing
                for i,task in enumerate(closing_tasks()):
                    seed=cfg['closing_evaluation']['seed']+i*1000
                    sampling={**cfg['sampling'],'temperature':0}
                    policy=BehaviorPolicy(engine,tok,lora,sampling,seed,
                        lambda n:bar.set_postfix(kind='closing',turn=policy.calls,response_tokens=n,refresh=True))
                    try:
                        episode=run_closing(task,policy,sampling,spec['policy']['version'],seed,tok)
                    except Exception as exc:
                        jsonl(partial,getattr(exc,'episode_evidence',{'status':'interrupted','error':str(exc)}));raise
                    episode['model_branch']=label;closing_episodes.append(episode);jsonl(partial,{'closing':episode});bar.update(1)
        proof = engine.collective_rpc('grpo_verify_lora_tensors',
                    args=(str(adapter.resolve()), spec['policy']['adapter_sha256']), timeout=60)
        data = {'spec': spec, 'config_sha256': file_hash(output/'config.json'), 'episodes': episodes,
                'quality': quality, 'targeted_episodes':targeted_episodes,
                'targeted_summary':summarize_targeted(targeted_episodes) if target_plan else None,
                'closing_episodes':closing_episodes, 'summary': summarize_evaluation(episodes,cfg), 'worker_proof': proof,
                'elapsed_seconds': time.perf_counter()-started,
                'quality_response_tokens': sum(len(g['response_token_ids']) for q in quality for g in q['generations']),
                'one_adapter_one_process': True,
                'max_loras': 1, 'lora_id': 1, 'seed_protocol': 'same_as_dpo_sft_start_development'}
        check = verify_evaluation(output, label, spec, tok, data=data)
        verify_closing_evaluation(output,label,spec,tok,data)
        targeted_check=verify_targeted_evaluation(output,label,spec,tok,data)
        atomic_json(output/'evaluations'/(label+'-audit.json'), check)
        atomic_json(target, data)
        finish_quality_check(output,label,spec,check,targeted_check)
    finally:
        if engine is not None:
            shutdown(engine)




def verify_closing_evaluation(output,label,spec,tok,data=None):
    from apt_v2.period_closing import closing_tasks,replay_closing
    from dataclasses import asdict
    cfg=read(output/'config.json'); data=data if data is not None else read(output/'evaluations'/(label+'.json'))
    if not cfg.get('closing_evaluation',{}).get('enabled'): return None
    tasks=closing_tasks(); rows=data.get('closing_episodes',[])
    if len(rows)!=len(tasks) or read(output/'closing_tasks.json')!=[asdict(t) for t in tasks]:
        raise ValueError('fixed closing development task coverage differs')
    for i,(task,e) in enumerate(zip(tasks,rows)):
        if (e['task']!=asdict(task) or e['model_branch']!=label or e['policy_version']!=spec['policy']['version']
                or e['seed']!=cfg['closing_evaluation']['seed']+i*1000
                or e['sampling']!={**cfg['sampling'],'temperature':0}):
            raise ValueError('closing evaluation policy/seed/sampling differs')
        replay_closing(e,tok)
    return {'passed':True,'episodes':len(rows),'successes':sum(e['success'] for e in rows),
            'valid_closings':sum(e['closing_valid'] for e in rows),'actions':sum(len(e['turns']) for e in rows)}




def verify_targeted_evaluation(output,label,spec,tok,data=None):
    from apt_v2.grpo_curriculum import targeted_plan,build_context as target_context,summarize_targeted
    cfg=read(output/'config.json')
    if not cfg.get('curriculum_schema'):return None
    data=data if data is not None else read(output/'evaluations'/(label+'.json'))
    plan=targeted_plan(cfg)
    if read(output/'targeted_evaluation_plan.json')!=plan:raise ValueError('targeted plan differs')
    rows=data.get('targeted_episodes',[])
    actual={evaluation_key(e):e for e in rows};expected={plan_key(p):p for p in plan}
    if len(actual)!=len(rows) or actual.keys()!=expected.keys():raise ValueError('targeted evaluation coverage differs')
    tasks={t.id:t for t in development_tasks(cfg)}
    calendar=[]
    for key,e in actual.items():
        p=expected[key];task=tasks[e['task_id']]
        if (e['sampling']!={**cfg['sampling'],'temperature':p['temperature']} or e['seed']!=p['seed']
                or e['policy_version']!=spec['policy']['version'] or e['model_branch']!=label
                or e['split']!='development'):
            raise ValueError('targeted evaluation sampling/model binding differs')
        replay(task,e,tok,target_context)
        cal=calendar_episode_audit(task,[x['row'] for x in e['start']['prefix']]+e['turns'],
                                   e['success'],e['artifact'],allowed_split='development')
        calendar.append({'key':list(key),'result':cal})
        if e['success'] and not cal['calendar_consistent_success'] and cfg.get('curriculum_schema') not in ('alias_curriculum_v1','repair_study_v1'):
            raise ValueError('targeted calendar evidence differs')
    summary=summarize_targeted(rows)
    if data.get('targeted_summary')!=summary:raise ValueError('targeted summary differs')
    result={'passed':all(not r['result']['environment_success'] or r['result']['calendar_consistent_success'] for r in calendar),
            'summary':summary,'episodes_replayed':len(rows),'actions_replayed':sum(len(e['turns']) for e in rows)}
    if cfg.get('curriculum_schema') in ('alias_curriculum_v1','repair_study_v1'):result['calendar']=calendar
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    evaluate(args.output, args.label, args.spec)
