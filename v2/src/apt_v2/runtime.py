"""Agent runtime shared by scripted, model and future training policies."""
from dataclasses import dataclass,asdict
import json
from .environment import Environment,TOOLS
from .tasks import metric_sql

SYSTEM=(
 "You are a business reconciliation agent. Use tools to gather evidence, calculate and validate your answer. "
 "At every turn return exactly one JSON object with keys tool and arguments, without markdown. "
 "Tools: "+json.dumps(TOOLS)+
 ". query_sql must return exact metric aliases current_net_cents and/or prior_net_cents. "
 "Use calculate_change for their difference. Query results include evidence IDs and verified_metrics. "
 "Submit values, evidence IDs for every metric, and policy_id. If a tool fails, inspect the error and retry or repair."
)

@dataclass
class Generation:
    text:str
    mode:str='scripted'
    prompt_token_ids:list|None=None
    response_token_ids:list|None=None
    raw_token_log_probs:list|None=None
    behavior_token_log_probs:list|None=None
    finish_reason:str|None=None

def parse_action(text):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    value=json.loads(text,object_pairs_hook=unique)
    if not isinstance(value,dict) or set(value)!={'tool','arguments'}:raise ValueError('one tool action required')
    return value

def run_episode(task,policy,reward_mode='potential'):
    env=Environment(task,reward_mode)
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    try:
        while not env.done:
            sample=policy(messages)
            try:action=parse_action(sample.text)
            except (ValueError,TypeError):action={'invalid_action':sample.text}
            row=env.step(action)
            row['generation']=asdict(sample)
            messages.append({'role':'assistant','content':sample.text})
            messages.append({'role':'user','content':json.dumps({'observation':row['observation']})})
        return {'task_id':task.id,'family_id':task.family_id,'split':task.split,'fault_mode':task.fault_mode,
            'reward_mode':reward_mode,'success':env.success,'turns':env.records,
            'tool_errors':env.tool_errors,'recovered':env.success and env.progress_after_error,
            'total_reward':sum(r['reward'] for r in env.records),
            'process_reward_sum':sum(r['process_reward'] for r in env.records),
            'artifact':env.records[-1]['observation'].get('artifact') if env.success else None}
    except Exception as exc:
        exc.episode_evidence={'task_id':task.id,'status':'interrupted','turns':env.records,
            'error':str(exc),'training_admitted':False}
        raise
    finally:env.close()

class ScriptedPolicy:
    """Reference tool plan; it reads public task fields and tool observations."""
    def __init__(self,task):
        self.task=task;self.phase=0;self.evidence={};self.values={}
    def __call__(self,messages):
        if len(messages)>2:
            observation=json.loads(messages[-1]['content'])['observation']
            if 'error' in observation:self.phase-=1
            else:
                for metric in observation.get('verified_metrics',[]):
                    self.evidence[metric]=observation['evidence_id']
                    if 'rows' in observation:self.values[metric]=observation['rows'][0][metric]
                    else:self.values[metric]=observation[metric]
        phase=self.phase;self.phase+=1
        if phase==0:action={'tool':'read_policy','arguments':{}}
        elif phase==1:action={'tool':'inspect_schema','arguments':{}}
        elif phase in (2,3):
            metric='current_net_cents' if phase==2 else 'prior_net_cents'
            action={'tool':'query_sql','arguments':{'sql':metric_sql(self.task,metric)}}
        elif phase==4:
            action={'tool':'calculate_change','arguments':{
                'current_evidence':self.evidence['current_net_cents'],'prior_evidence':self.evidence['prior_net_cents']}}
        else:
            action={'tool':'check_result' if phase==5 else 'submit','arguments':{
                'values':dict(self.values),'evidence':dict(self.evidence),'policy_id':'revenue-v1'}}
        return Generation(json.dumps(action))
