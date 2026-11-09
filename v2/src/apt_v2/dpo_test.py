"""Native-token replay and paired outcome summaries for frozen DPO holdout evaluation."""
import json,math
from .environment import Environment
from .runtime import SYSTEM
from .dpo import action_from_text
from .sft_diagnostics import summarize_episodes

def replay_episode(task,episode,tokenizer,mode):
    if episode['task_id']!=task.id or episode['family_id']!=task.family_id or episode['split']!=task.split or episode['fault_mode']!=task.fault_mode:
        raise ValueError('test task identity differs')
    env=Environment(task,episode['reward_mode'])
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    try:
        for row in episode['turns']:
            sample=row['generation'];ids=sample['response_token_ids'];raw=sample['raw_token_log_probs']
            if sample['mode']!=mode or not ids or raw is None or len(ids)!=len(raw):raise ValueError('native token alignment differs')
            if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite native log probabilities required')
            prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            if sample['prompt_token_ids']!=prompt:raise ValueError('native prompt differs')
            if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=sample['text']:raise ValueError('native decode differs')
            actual=env.step(action_from_text(sample['text']))
            for key in ('turn','action','observation','reward','done','success','tool_error','process_reward','outcome_reward','new_verified_metrics','termination'):
                if actual[key]!=row[key]:raise ValueError('replay differs: '+key)
            messages += [{'role':'assistant','content':sample['text']},{'role':'user','content':json.dumps({'observation':row['observation']})}]
        if not env.done or env.success!=episode['success'] or env.tool_errors!=episode['tool_errors']:
            raise ValueError('episode completion differs')
        if (env.success and env.progress_after_error)!=episode['recovered']:raise ValueError('recovery differs')
        if episode['total_reward']!=sum(r['reward'] for r in env.records) or episode['process_reward_sum']!=sum(r['process_reward'] for r in env.records):raise ValueError('episode reward sum differs')
        artifact=env.records[-1]['observation'].get('artifact') if env.success else None
        if artifact!=episode['artifact']:raise ValueError('episode artifact differs')
        return len(episode['turns'])
    finally:env.close()

def validate_set(episodes,tasks,branch,sample_cfg=None):
    task_ids=[t.id for t in tasks]
    if any(e['model_branch']!=branch or e['split']!='test' for e in episodes):raise ValueError('test branch differs')
    if sample_cfg is None:
        if len(episodes)!=len(tasks) or {e['task_id'] for e in episodes}!=set(task_ids):raise ValueError('complete distinct greedy set required')
    else:
        expected={(t,i) for t in task_ids for i in range(sample_cfg['rollouts_per_task'])}
        if len(episodes)!=len(expected) or {(e['task_id'],e['rollout_index']) for e in episodes}!=expected:raise ValueError('complete distinct sample set required')
        for e in episodes:
            seed=sample_cfg['seed']+task_ids.index(e['task_id'])*10000+e['rollout_index']*100
            if e['sample_seed']!=seed or e['temperature']!=sample_cfg['temperature']:raise ValueError('sample identity differs')
            if any(row['sampling_seed']!=seed+i for i,row in enumerate(e['turns'])):raise ValueError('turn seed differs')
    return summarize_episodes(episodes)

def paired_outcomes(sft,dpo,sampled=False):
    key=lambda e:(e['task_id'],e['rollout_index']) if sampled else e['task_id']
    left={key(e):e for e in sft};right={key(e):e for e in dpo}
    if len(left)!=len(sft) or len(right)!=len(dpo) or left.keys()!=right.keys():raise ValueError('paired identities differ')
    counts={'both_success':0,'both_failure':0,'improved':0,'regressed':0};changes=[]
    for k,a in left.items():
        b=right[k]
        label=('both_success' if a['success'] else 'both_failure') if a['success']==b['success'] else ('improved' if b['success'] else 'regressed')
        counts[label]+=1
        if label in ('improved','regressed'):changes.append({'task_id':a['task_id'],'rollout_index':a.get('rollout_index'),'sft_success':a['success'],'dpo_success':b['success']})
    return {**counts,'changes':changes}
