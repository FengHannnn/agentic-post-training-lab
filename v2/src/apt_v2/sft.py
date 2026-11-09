"""Replay-verified demonstrations and runtime-aligned assistant action supervision."""
import json
from .runtime import ScriptedPolicy,Generation,SYSTEM,run_episode,parse_action
from .environment import Environment
from .tasks import make_tasks

class RepairPolicy(ScriptedPolicy):
    def __init__(self,task,variant):
        super().__init__(task);self.variant=variant;self.injected=False
    def __call__(self,messages):
        sample=super().__call__(messages);action=parse_action(sample.text)
        target={'sql_repair':'query_sql','evidence_repair':'calculate_change'}.get(self.variant)
        eligible=action['tool']==target and (target!='query_sql' or 'AS prior_net_cents' in action['arguments']['sql'])
        if not self.injected and eligible:
            self.injected=True
            if target=='query_sql':action['arguments']['sql']='SELECT current_net_cents FROM metrics'
            else:action['arguments']={'current_evidence':'missing','prior_evidence':'missing'}
            return Generation(json.dumps(action))
        return sample

def replay(task,episode):
    env=Environment(task,episode['reward_mode'])
    try:
        for row in episode['turns']:
            actual=env.step(parse_action(row['generation']['text']))
            for key in ('action','observation','done','success','tool_error','reward'):
                if actual[key]!=row[key]:raise ValueError('replay mismatch: '+key)
        if not env.success or not env.done:raise ValueError('successful complete demonstration required')
    finally:env.close()

def build_dataset():
    episodes=[];examples=[]
    for task in make_tasks():
        if task.split!='train':continue
        for variant in ('expert','sql_repair','evidence_repair'):
            episode=run_episode(task,RepairPolicy(task,variant));replay(task,episode)
            episode['source']='scripted';episode['variant']=variant
            episodes.append(episode)
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
            for row in episode['turns']:
                text=row['generation']['text'];obs=row['observation']
                # Exclude erroneous actions and failed checks; preserve them in later contexts.
                admitted=row['tool_error'] is None and obs.get('passed',True)
                if admitted:
                    examples.append({'task_id':task.id,'family_id':task.family_id,'split':'train',
                        'source':'scripted','variant':variant,'turn':row['turn'],
                        'messages':[dict(m) for m in messages],'response':text})
                messages += [{'role':'assistant','content':text},
                    {'role':'user','content':json.dumps({'observation':obs})}]
    return episodes,examples

def encode_example(tokenizer,example,max_length):
    messages=example['messages']
    prompt=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    complete=list(tokenizer.apply_chat_template(messages+[{'role':'assistant','content':example['response']}],
        tokenize=True,add_generation_prompt=False))
    if complete[:len(prompt)]!=prompt:raise ValueError('chat template prefix mismatch')
    response=complete[len(prompt):]
    if not response or tokenizer.eos_token_id not in response:raise ValueError('assistant EOS required')
    response=response[:response.index(tokenizer.eos_token_id)+1]
    complete=prompt+response
    if len(complete)>max_length:raise ValueError(f'context exceeds max_length: {len(complete)} > {max_length}')
    return {'input_ids':complete,'labels':[-100]*len(prompt)+response,
        'prompt_length':len(prompt),'response_length':len(response)}

def action_loss(model,sample):
    import torch
    device=next(model.parameters()).device
    ids=torch.tensor([sample['input_ids']],device=device)
    n=sample['response_length']
    # Qwen supports logits_to_keep: project only the supervised suffix to vocabulary.
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
        logits_to_keep=n+1).logits[0,-n-1:-1].float()
    target=torch.tensor(sample['labels'][-n:],device=device)
    return torch.nn.functional.cross_entropy(logits,target)
