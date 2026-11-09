"""Independent test contracts for a pinned SFT candidate."""
import json
from .tasks import make_tasks
from .sft_diagnostics import summarize_episodes

def holdout_tasks(cfg,training_examples):
    if cfg['training_enabled'] is not False or cfg['eval']['splits']!={'test':4}:
        raise ValueError('inference-only four-task test evaluation required')
    tasks=make_tasks();test=[t for t in tasks if t.split=='test']
    train=[t for t in tasks if t.split=='train']
    if len(test)!=4 or sum(t.fault_mode!='none' for t in test)!=2:raise ValueError('fixed test set differs')
    if {t.id for t in test}&{t.id for t in train} or {t.family_id for t in test}&{t.family_id for t in train}:
        raise ValueError('task/family leakage')
    allowed={(t.id,t.family_id) for t in train}
    if not training_examples or any(e['split']!='train' or (e['task_id'],e['family_id']) not in allowed for e in training_examples):
        raise ValueError('train-only candidate provenance required')
    return test

def holdout_summary(episodes,task_ids):
    if len(episodes)!=len(task_ids) or {e['task_id'] for e in episodes}!=set(task_ids):
        raise ValueError('complete distinct test episodes required')
    if any(e['split']!='test' for e in episodes):raise ValueError('test-only evaluation required')
    rows=[r for e in episodes for r in e['turns']]
    return {**summarize_episodes(episodes),
        'valid_checks':sum(r['action'].get('tool')=='check_result' and r['observation'].get('passed',False) for r in rows),
        'valid_submits':sum(r['action'].get('tool')=='submit' and r['observation'].get('passed',False) for r in rows)}

def acceptance(summary,criteria):
    checks={key:summary[key]>=value for key,value in criteria.items()}
    checks['strict_format']=summary['strict_format']==summary['turns']
    return {'criteria':criteria,'checks':checks,'eligible_for_local_review':all(checks.values())}
