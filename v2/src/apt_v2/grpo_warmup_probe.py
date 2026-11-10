"""Train-only warm-up diagnostics with execution-verified actions and alias token branches."""
import copy,json,math
from .grpo import digest,parse_or_invalid,validate_native
from .grpo_repair_study import training_tasks,W_SHA
from .grpo_retention_study import STATES,verify_dataset
from .environment import Environment
from .runtime import SYSTEM,parse_action

SCHEMA='warmup_probe_v1'
RUN_ID='grpo-sql-warmup-probe-v1-20261110'
MODEL_IDENTITIES={'w':(W_SHA,513),'b24':('d81a31b392ca5e4e043a43aa2c8fa75ee2968e97b09cbe857f732d6e22dd5226',537),
                  'b15':('38a9a6d68ed1a0ea0620aaa726ca6443fe701691a88c533634761663c6a72b31',528)}


def validate_config(cfg):
    if (cfg.get('schema')!=SCHEMA or cfg.get('run_id')!=RUN_ID or cfg.get('max_wall_seconds')!=1800
            or cfg.get('prior_elapsed_seconds')!=9500.091520188143 or cfg.get('total_authorized_seconds')!=21600
            or cfg.get('seed')!=2026111061 or cfg.get('target_repeats')!=2
            or cfg.get('branch_template')!='configs/grpo-sql-repair-branch-v1.json'
            or cfg.get('input_dir')!='artifacts/inputs/grpo-warmup-probe-20261110'
            or cfg.get('sampling')!={'temperature':1,'top_p':1,'top_k':-1,'min_p':0,'repetition_penalty':1,
                 'presence_penalty':0,'frequency_penalty':0,'min_tokens':0,'max_new_tokens':256,'max_model_len':8192,
                 'logprobs_mode':'processed_logprobs'}
            or cfg.get('models')!={k:{'adapter_sha256':s,'source_sft_updates':n} for k,(s,n) in MODEL_IDENTITIES.items()}):
        raise ValueError('frozen warm-up probe contract differs')
    if cfg['max_wall_seconds']+cfg['prior_elapsed_seconds']>cfg['total_authorized_seconds']:
        raise ValueError('combined budget exceeded')


def validate_prior(data,cfg):
    first,second=data['receipts']
    if (first['sha256']!='77d7e4cea2c9d0d481ee95a20583d00e8cc27a992c35ae46d983179bc00b9920'
            or second['sha256']!='0246e5f7b2fad61fecddc684d64e459fc71da43aa48dca1c9b1ea61d7eb98a62'
            or first['elapsed_seconds_including_packaging']!=5578.0134817191865
            or second['elapsed_seconds_including_packaging']!=3922.078038468957
            or second['combined_elapsed_seconds']!=cfg['prior_elapsed_seconds']
            or first['elapsed_seconds_including_packaging']+second['elapsed_seconds_including_packaging']!=cfg['prior_elapsed_seconds']
            or not all(x['passed'] for x in data['local_audits'])):
        raise ValueError('prior receipts/audits differ')
    return True


def make_plan(examples):
    verify_dataset(examples);by={(e['task_id'],e['state']):e for e in examples};plan=[]
    tasks=training_tasks()
    for i,t in enumerate(tasks):
        e=copy.deepcopy(by[t.id,'repair_compound']);e.update(index=i,kind='target');plan.append(e)
    for i,t in enumerate(tasks):
        # Every maintenance state has three tasks and both old/new pools.
        state=STATES[(i//2+(2 if i%2 else 0))%4]
        e=copy.deepcopy(by[t.id,state]);e.update(index=12+i,kind='maintenance');plan.append(e)
    return plan


def execute(plan,text):
    tasks={t.id:t for t in training_tasks()};task=tasks[plan['task_id']];messages=plan['messages'];env=Environment(task)
    try:
        if messages[:2]!=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]:
            raise ValueError('public train prefix differs')
        if len(messages)%2:raise ValueError('complete tool exchanges required')
        for i in range(2,len(messages),2):
            if messages[i]['role']!='assistant' or messages[i+1]['role']!='user':raise ValueError('prefix roles differ')
            row=env.step(parse_action(messages[i]['content']))
            if {'observation':row['observation']}!=json.loads(messages[i+1]['content']):raise ValueError('prefix execution differs')
        return env.step(parse_or_invalid(text))
    finally:env.close()


def productive(plan,row):
    if row['tool_error']:return False
    state=plan['state'];tool=row['action'].get('tool');obs=row['observation']
    if state in ('repair_compound','repeat_sql_error'):
        return bool(set(row['new_verified_metrics'])&{'current_net_cents','prior_net_cents'})
    if state=='stale_evidence_repair':return 'change_cents' in row['new_verified_metrics']
    if state=='failed_check_repair':return tool in ('check_result','submit') and bool(obs.get('passed'))
    if state=='checked_submit':return bool(row['success'] and obs.get('submitted'))
    raise ValueError('unknown probe state')


def variants(tok,plan):
    from .sft import encode_example
    action=parse_action(plan['response']);sql=action['arguments']['sql'];needle=' AS current_net_cents'
    if plan['kind']!='target' or sql.count(needle)!=1:raise ValueError('one explicit target alias required')
    omitted=copy.deepcopy(action);omitted['arguments']['sql']=sql.replace(needle,'')
    good=encode_example(tok,plan,8192);bad=encode_example(tok,dict(plan,response=json.dumps(omitted)),8192)
    prompt=good['input_ids'][:good['prompt_length']]
    if prompt!=bad['input_ids'][:bad['prompt_length']]:raise ValueError('matched teacher-forcing prompt required')
    a=good['input_ids'][good['prompt_length']:];b=bad['input_ids'][bad['prompt_length']:]
    branch=0
    while branch<min(len(a),len(b)) and a[branch]==b[branch]:branch+=1
    if branch==min(len(a),len(b)):raise ValueError('distinct nonempty alias branches required')
    start=plan['response'].index('AS current_net_cents');end=start+len('AS current_net_cents')
    spans=tok(plan['response'],add_special_tokens=False,return_offsets_mapping=True)
    if spans['input_ids']!=a[:-1]:raise ValueError('native response offset alignment differs')
    alias=[i for i,(x,y) in enumerate(spans['offset_mapping']) if x<end and y>start]
    if not alias:raise ValueError('alias tokens required')
    return {'prompt_ids':prompt,'correct_ids':a,'omitted_ids':b,'branch_index':branch,'alias_indices':alias,
            'omitted_text':json.dumps(omitted)}


def validate_score(score,prompt,response,seed):
    if (score['source']!='vllm_prompt_logprobs_T1' or score['not_used_for_policy_updates'] is not True
            or score['prompt_sha256']!=digest(prompt+response) or score['response_token_ids']!=response
            or score['seed']!=seed or len(score['token_log_probs'])!=len(response)
            or len(score['auxiliary_token_ids'])!=1
            or any(type(i) is not int or i<0 for i in score['auxiliary_token_ids'])
            or any(not math.isfinite(x) or x>1e-6 for x in score['token_log_probs'])):
        raise ValueError('teacher-forced diagnostic score differs')


def score_summary(v,scores):
    good=scores['correct']['token_log_probs'];bad=scores['omitted']['token_log_probs'];j=v['branch_index']
    return {'alias_token_nll':-math.fsum(good[i] for i in v['alias_indices'])/len(v['alias_indices']),
            'correct_response_token_nll':-math.fsum(good)/len(good),'correct_branch_token':v['correct_ids'][j],
            'omitted_branch_token':v['omitted_ids'][j],
            'branch_logp_margin':good[j]-bad[j],
            'branch_index':j,'alias_token_count':len(v['alias_indices'])}


def validate_row(cfg,plan,row,tok):
    seed=cfg['seed']+plan['index']*100
    if (row['index']!=plan['index'] or row['task_id']!=plan['task_id'] or row['state']!=plan['state']
            or row['seed']!=seed or row['plan_sha256']!=digest(plan) or row['model'] not in cfg['models']):
        raise ValueError('diagnostic row plan differs')
    sampling=dict(cfg['sampling'],temperature=0)
    generations=row['generations'];expected=2 if plan['kind']=='target' else 1
    if len(generations)!=expected or len(row['executions'])!=expected:raise ValueError('diagnostic repetition count differs')
    for g,execution in zip(generations,row['executions']):
        validate_native(tok,plan['messages'],g,sampling)
        if execute(plan,g['text'])!=execution:raise ValueError('diagnostic action replay differs')
    if plan['kind']=='target':
        v=variants(tok,plan)
        validate_score(row['scores']['correct'],v['prompt_ids'],v['correct_ids'],seed+17)
        validate_score(row['scores']['omitted'],v['prompt_ids'],v['omitted_ids'],seed+18)
        if row['score_summary']!=score_summary(v,row['scores']):raise ValueError('score summary differs')
    elif row['scores'] is not None or row['score_summary'] is not None:raise ValueError('maintenance has no forced alias score')
    return True


def summarize(plan,rows):
    if len(rows)!=24 or sorted(r['index'] for r in rows)!=list(range(24)):raise ValueError('24 complete probe contexts required')
    targets=[r for r in rows if r['index']<12];maintenance=[r for r in rows if r['index']>=12]
    positive=lambda rs:sum(productive(plan[r['index']],r['executions'][0]) for r in rs)
    return {'target_first_certifications':positive(targets),'target_contexts':12,
            'maintenance_productive_actions':positive(maintenance),'maintenance_contexts':12,
            'maintenance_by_state':{state:{'contexts':3,'productive':positive([r for r in maintenance if r['state']==state])} for state in STATES},
            'target_repeat_identical':sum(r['generations'][0]['response_token_ids']==r['generations'][1]['response_token_ids'] for r in targets),
            'alias_token_nll_mean':math.fsum(r['score_summary']['alias_token_nll'] for r in targets)/12,
            'branch_logp_margin_mean':math.fsum(r['score_summary']['branch_logp_margin'] for r in targets)/12,
            'correct_branch_preferred_contexts':sum(r['score_summary']['branch_logp_margin']>0 for r in targets),
            'committed_generated_actions':36,'committed_forced_score_requests':24,'committed_auxiliary_generated_tokens':24,
            'not_used_for_policy_updates':True}
