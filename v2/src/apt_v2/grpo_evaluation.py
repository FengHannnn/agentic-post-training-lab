"""Fixed development evaluation schedule, task-level summaries and paired outcomes."""
from .tasks import make_tasks
from .dpo_recovery import SCENARIOS


def development_tasks(cfg=None):
    if cfg and cfg.get('curriculum_schema'):
        from .grpo_curriculum import development_tasks as curriculum_tasks
        return curriculum_tasks(cfg)
    return [t for t in make_tasks() if t.split == 'development']


def evaluation_plan(cfg):
    result = []
    for ti, task in enumerate(development_tasks(cfg)):
        for scenario in [None, *SCENARIOS]:
            for mode in ('greedy', 'sample'):
                count = 1 if mode == 'greedy' else (cfg['sample_eval']['rollouts_per_task'] if scenario is None
                                                    else cfg['recovery_eval']['samples_per_context'])
                for ri in range(count):
                    seed = (cfg['recovery_eval']['seed']+ti*100000+SCENARIOS.index(scenario)*1000+ri*100
                            if scenario else (cfg['eval']['seed']+ti*10000 if mode == 'greedy'
                                              else cfg['sample_eval']['seed']+ti*10000+ri*100))
                    result.append({'task_id': task.id, 'scenario': scenario,
                                   'kind': 'complete' if scenario is None else 'recovery', 'mode': mode,
                                   'rollout_index': ri, 'seed': seed,
                                   'temperature': 0 if mode == 'greedy' else cfg['sample_eval' if scenario is None else 'recovery_eval']['temperature']})
    return result


def evaluation_key(episode):
    return (episode['task_id'], episode['scenario'], episode['evaluation_mode'], episode['rollout_index'])


def plan_key(item):
    return (item['task_id'], item['scenario'], item['mode'], item['rollout_index'])


def summarize_evaluation(episodes, cfg=None):
    result = {}
    def stats(rows):
        return {'episodes': len(rows), 'successes': sum(e['success'] for e in rows),
                'actions': sum(len(e['turns']) for e in rows),
                'response_tokens': sum(len(r['generation']['response_token_ids']) for e in rows for r in e['turns']),
                'tool_errors': sum(e['tool_errors'] for e in rows)}
    for mode in ('greedy', 'sample'):
        result[mode] = {}
        for kind in ('complete', 'recovery'):
            rows = [e for e in episodes if e['evaluation_mode'] == mode and bool(e['scenario']) == (kind == 'recovery')]
            result[mode][kind] = {**stats(rows),
                'by_task': {t.id: stats([e for e in rows if e['task_id'] == t.id]) for t in development_tasks(cfg)},
                'by_scenario': {s: stats([e for e in rows if e['scenario'] == s]) for s in SCENARIOS} if kind == 'recovery' else {}}
    return result


def paired_evaluation(before, after, cfg=None):
    a = {evaluation_key(e): e['success'] for e in before}
    b = {evaluation_key(e): e['success'] for e in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('distinct paired task/scenario/mode/rollout keys required')
    result = {}
    def stats(keys):
        return {'episodes': len(keys), 'both_success': sum(a[k] and b[k] for k in keys),
                'both_failure': sum(not a[k] and not b[k] for k in keys),
                'improved': sum(not a[k] and b[k] for k in keys),
                'regressed': sum(a[k] and not b[k] for k in keys),
                'delta_successes': sum(int(b[k])-int(a[k]) for k in keys)}
    for mode in ('greedy', 'sample'):
        result[mode] = {}
        for kind in ('complete', 'recovery'):
            keys = [k for k in a if k[2] == mode and bool(k[1]) == (kind == 'recovery')]
            result[mode][kind] = {**stats(keys), 'by_task': {
                t.id: stats([k for k in keys if k[0] == t.id]) for t in development_tasks(cfg)}}
    return result
