"""SQL-weighted mixed SFT with unchanged expert data and normalized action weights."""
import math
from collections import Counter
from .runtime import parse_action

def normalized_weights(examples,sql_weight):
    if not examples or not math.isfinite(sql_weight) or sql_weight<=0:raise ValueError('positive finite SQL weight and nonempty examples required')
    raw=[float(sql_weight) if parse_action(e['response'])['tool']=='query_sql' else 1. for e in examples]
    mean=sum(raw)/len(raw)
    return [w/mean for w in raw]

def data_identity(examples):
    import json,hashlib
    return hashlib.sha256(json.dumps(examples,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def validate_examples(examples,parent_summary):
    from .sft import build_dataset
    _,rebuilt=build_dataset()
    if examples!=rebuilt:raise ValueError('unchanged replay-verified pilot examples required')
    if any(e['split']!='train' for e in examples):raise ValueError('train-only examples required')
    if data_identity(examples)!=parent_summary['dataset_sha256']:raise ValueError('parent dataset identity differs')
    return dict(Counter(parse_action(e['response'])['tool'] for e in examples))


def audit_episode(task,episode,tokenizer):
    import json,math
    from .environment import Environment
    from .runtime import SYSTEM
    if episode['task_id']!=task.id or episode['split']!=task.split:raise ValueError('task identity differs')
    env=Environment(task,episode['reward_mode'])
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    try:
        for row in episode['turns']:
            sample=row['generation'];ids=sample['response_token_ids'];probs=sample['raw_token_log_probs']
            if sample['mode']!='greedy' or not ids or probs is None or len(ids)!=len(probs):raise ValueError('greedy native token alignment required')
            if any(not math.isfinite(v) or v>1e-6 for v in probs):raise ValueError('finite native probabilities required')
            if sample['prompt_token_ids']!=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True):raise ValueError('prompt differs')
            if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=sample['text']:raise ValueError('decode differs')
            try:action=parse_action(sample['text'])
            except (TypeError,ValueError):action={'invalid_action':sample['text']}
            actual=env.step(action)
            for key in ('action','observation','done','success','reward','tool_error','process_reward','outcome_reward','new_verified_metrics','termination'):
                if actual[key]!=row[key]:raise ValueError('replay differs: '+key)
            messages += [{'role':'assistant','content':sample['text']},
                {'role':'user','content':json.dumps({'observation':row['observation']})}]
        if not env.done or env.success!=episode['success'] or env.tool_errors!=episode['tool_errors']:
            raise ValueError('episode completion differs')
        if (env.success and env.progress_after_error)!=episode['recovered']:raise ValueError('recovery differs')
        return len(episode['turns'])
    finally:env.close()


def backward_batch(model,samples,weights,indices):
    import torch
    from .sft import action_loss
    if not indices:raise ValueError('nonempty batch required')
    value=0.
    for i in indices:
        loss=action_loss(model,samples[i])*weights[i]/len(indices)
        if not torch.isfinite(loss):raise ValueError('nonfinite weighted loss')
        loss.backward();value+=loss.item()
    return value
