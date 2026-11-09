"""Frozen C4 inputs, native trajectory verification and independent audit."""
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,under,runtime_identity,validate_runtime
from apt_v2.grpo_confirmation import (LABELS,ADAPTERS,VERSIONS,confirmation_tasks,confirmation_plan,
    independence_report,key,summarize,paired,acceptance,expert_confirmation)
from apt_v2.grpo import replay,validate_native,digest
from apt_v2.dpo_recovery import SCENARIOS,build_context
from apt_v2.independent_recovery_test import calendar_episode_audit


def validate_config(cfg):
    version=cfg.get('protocol_version',1)
    profiles={1:(8,4,14400,'grpo_c4_release_manifest.json'),2:(12,6,25200,'grpo_c4_v2_release_manifest.json')}
    if version not in profiles:raise ValueError('known C4 protocol version required')
    tasks,gain,budget,manifest=profiles[version]
    if cfg.get('release_manifest',manifest)!=manifest:raise ValueError('frozen release manifest differs')
    if (cfg['study_phase']!='C4' or not re.fullmatch(r'grpo-c4-[a-z0-9-]+',cfg['run_id'])
            or cfg['new_training_updates']!=0 or cfg['task_count']!=tasks or tuple(cfg['models'])!=LABELS
            or cfg['complete_rollouts']!=4 or cfg['recovery_rollouts']!=3
            or cfg['minimum_recovery_gain']!=gain or cfg['max_wall_seconds']>budget
            or cfg['max_wall_seconds']<=120 or cfg['packaging_reserve_seconds']<120
            or cfg['required_gpu_name']!='NVIDIA L40S' or cfg['quality_contexts_per_model']!=7
            or cfg['sampling']['max_new_tokens']!=256 or cfg['sampling']['max_model_len']!=8192
            or cfg['confirmation_seed']!=2026110700 or cfg['bootstrap_seed']!=2026110711
            or cfg['bootstrap_repetitions']!=10000 or cfg['tool_cost']!=.01):raise ValueError('frozen C4 confirmation contract differs')
    expected_acceptance={'mode':'practical','sample_recovery_minimum_gain':gain,'sample_recovery_denominator':tasks*21,
        'sample_complete_no_regression':True,'greedy_complete_no_regression':True,'greedy_recovery_no_regression':True,
        'primary':'complete unless W sample complete is saturated; otherwise recovery','primary_must_improve':True}
    if cfg['acceptance']!=expected_acceptance:raise ValueError('frozen practical acceptance differs')
    for label in LABELS:
        m=cfg['models'][label]
        if m['adapter_sha256']!=ADAPTERS[label] or m['version']!=VERSIONS[label]:raise ValueError('preselected C4 candidate differs')
    return cfg


def driver_identity():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    if p.returncode:raise ValueError('GPU driver identity unavailable')
    return {'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode}


def verify_inputs(output,local=False):
    cfg=validate_config(read(output/'config.json'));verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    binding=read(output/'input_binding.json');verify_files(output,binding['files'])
    if binding['config_sha256']!=file_hash(output/'config.json'):raise ValueError('frozen C4 config differs')
    if read(output/'tasks.json')!=[asdict(t) for t in confirmation_tasks(cfg)] or read(output/'plan.json')!=confirmation_plan(cfg):raise ValueError('frozen confirmation tasks or plan differ')
    if read(output/'independence.json')!=independence_report(cfg):raise ValueError('confirmation identity or database isolation differs')
    for label in LABELS:
        if file_hash(output/'inputs'/label/'adapter/adapter_model.safetensors')!=ADAPTERS[label]:raise ValueError('frozen candidate weights differ')
    if not local:
        validate_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'))
        if driver_identity()!=read(output/'driver_identity.json'):raise ValueError('within-job driver identity differs')
        verify_files(under(ROOT,cfg['base_dir']),{n:h for n,h in read(output/'model_identity.json')['files_sha256'].items() if not n.endswith('.safetensors')})
    return cfg


def quality_specs(cfg):
    t=confirmation_tasks(cfg)[0];plan=confirmation_plan(cfg)
    return [{'task_id':t.id,'scenario':s,'seed':next(p['seed'] for p in plan if p['task_id']==t.id and p['scenario']==s and p['mode']=='greedy')} for s in SCENARIOS]


def verify_session(output,label,session,tok,cfg):
    from grpo_vllm import projected_checkpoint_tensors
    if (session['label']!=label or session['config_sha256']!=file_hash(output/'config.json')
            or session['adapter_sha256']!=ADAPTERS[label] or session['hardware']!=read(output/'execution_environment.json')
            or session['driver']!=read(output/'driver_identity.json') or session['one_adapter_one_process'] is not True
            or session['max_loras']!=1 or session['lora_id']!=1):raise ValueError('C4 generation session binding differs')
    if not re.fullmatch(r'[0-9]+',session['session_id']):raise ValueError('safe immutable session ID required')
    proof=session['worker_proof'];projected=projected_checkpoint_tensors(output/'inputs'/label/'adapter')
    if not proof or any(not p['passed'] or p['adapter_sha256']!=ADAPTERS[label] or p['registered_ids']!=[1]
            or 1 not in p['active_slots'] or p['tensor_count']!=112 or p['tensors']!=projected
            or p['loaded_tensor_sha256']!=digest(projected) for p in proof):raise ValueError('independent C4 worker tensor proof differs')
    quality=session['quality'];specs=quality_specs(cfg);tasks={t.id:t for t in confirmation_tasks(cfg)}
    if len(quality)!=len(specs):raise ValueError('seven repeated greedy quality contexts required')
    for row,spec in zip(quality,specs):
        if {k:row[k] for k in spec}!=spec or len(row['generations'])!=2:raise ValueError('frozen quality context or seed differs')
        messages=build_context(tasks[spec['task_id']],spec['scenario'],allowed_split='test')['messages']
        for g in row['generations']:validate_native(tok,messages,g,{**cfg['sampling'],'temperature':0})
        if row['generations'][0]['response_token_ids']!=row['generations'][1]['response_token_ids']:raise ValueError('repeated greedy response tokens differ')
    return True


def verify_episode(task,item,episode,label,tok,cfg,session_id,worker_proof_sha256):
    if (key(episode)!=key(item) or episode['seed']!=item['seed'] or episode['sampling']!={**cfg['sampling'],'temperature':item['temperature']}
            or episode['policy_version']!=VERSIONS[label] or episode['model_branch']!=label
            or episode['session_id']!=session_id or episode['split']!='test'
            or episode['start']['tool_cost']!=cfg['tool_cost']
            or episode['worker_proof_sha256']!=worker_proof_sha256):raise ValueError('C4 paired native episode identity differs')
    actions=replay(task,episode,tok)
    records=[p['row'] for p in episode['start']['prefix']]+episode['turns']
    calendar=calendar_episode_audit(task,records,episode['success'],episode['artifact'],allowed_split='test')
    if episode['success'] and not calendar['calendar_consistent_success']:raise ValueError('C4 final calendar evidence differs')
    return {'actions':actions,'calendar':calendar}


def verify_model(output,label,tok,local=False,data=None):
    cfg=verify_inputs(output,local);data=data if data is not None else read(output/'evaluations'/(label+'.json'))
    if data['label']!=label or data['config_sha256']!=file_hash(output/'config.json'):raise ValueError('model evaluation identity differs')
    if data['new_training_updates']!=0:raise ValueError('evaluation-only C4 contract differs')
    rows=data['episodes'];plan=confirmation_plan(cfg);expected={key(p):p for p in plan};actual={key(e):e for e in rows}
    if len(actual)!=len(rows) or actual.keys()!=expected.keys():raise ValueError('complete distinct C4 evaluation coverage required')
    sessions=data['sessions']
    if not sessions or len({s['session_id'] for s in sessions})!=len(sessions):raise ValueError('distinct bound generation sessions required')
    for s in sessions:
        if read(output/'sessions'/label/(s['session_id']+'.json'))!=s:raise ValueError('session transaction differs')
        verify_session(output,label,s,tok,cfg)
    ids={s['session_id'] for s in sessions};tasks={t.id:t for t in confirmation_tasks(cfg)};calendar=[]
    for k,e in actual.items():
        if type(e['plan_index']) is not int or not 0<=e['plan_index']<len(plan):raise ValueError('valid C4 plan cursor required')
        if e['session_id'] not in ids:raise ValueError('episode has no verified loaded-weight session')
        if read(output/'episodes'/label/(str(e['plan_index']).zfill(4)+'.json'))!=e or plan[e['plan_index']]!=expected[k]:raise ValueError('episode transaction or plan cursor differs')
        calendar.append(verify_episode(tasks[e['task_id']],expected[k],e,label,tok,cfg,e['session_id'],digest(next(s['worker_proof'] for s in sessions if s['session_id']==e['session_id']))))
    summary=summarize(rows,cfg)
    if data['summary']!=summary:raise ValueError('C4 summary recomputation differs')
    return {'passed':True,'episodes_replayed':len(rows),'actions_replayed':sum(x['actions'] for x in calendar),
            'sessions':len(sessions),'quality_actions':14*len(sessions),'calendar_successes':sum(e['success'] for e in rows),
            'summary':summary,'adapter_sha256':ADAPTERS[label]}


def audit(output,local=False):
    import csv
    from transformers import AutoTokenizer
    cfg=verify_inputs(output,local);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    checks={label:verify_model(output,label,tok,local) for label in LABELS}
    proof=[expert_confirmation(t,s) for t in confirmation_tasks(cfg) for s in (None,*SCENARIOS)]
    if read(output/'expert_budget_proof.json')!=proof:raise ValueError('confirmation expert/budget proof differs')
    data={label:read(output/'evaluations'/(label+'.json'))['episodes'] for label in LABELS}
    summaries={label:checks[label]['summary'] for label in LABELS}
    comparisons={label+'_vs_W':paired(data['weighted_sft'],data[label],cfg=cfg) for label in LABELS[1:]}
    comparisons['fine_vs_outcome']=paired(data['outcome_rl16'],data['fine_rl17'],cfg=cfg)
    decisions={label:acceptance(summaries['weighted_sft'],summaries[label],cfg) for label in LABELS[1:]}
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_files',
        'new_training_updates':0,'verification':checks,'summaries':summaries,'paired':comparisons,
        'acceptance':decisions,'at_least_one_grpo_beats_W':any(d['passed'] for d in decisions.values()),
        'preselected_candidates':list(LABELS),'independence':read(output/'independence.json'),
        'expert_contexts_proved':len(proof),'hardware':read(output/'execution_environment.json'),
        'thresholds_frozen_before_inference':True,'bootstrap_role':'descriptive task-family clustering; practical acceptance uses frozen counts'}
    prefix='local_' if local else ''
    atomic_json(output/(prefix+'c4_audit.json'),result)
    cases=[];base={key(e):e for e in data['weighted_sft']}
    for label in LABELS[1:]:
        for e in data[label]:
            before=base[key(e)]
            if e['success']!=before['success']:cases.append({'model':label,'task_id':e['task_id'],'scenario':e['scenario'],'mode':e['evaluation_mode'],'rollout_index':e['rollout_index'],'before_success':before['success'],'after_success':e['success'],'before_actions':len(before['turns']),'after_actions':len(e['turns'])})
    atomic_json(output/(prefix+'paired_changed_cases.json'),cases)
    rows=[{'model':label,'mode':mode,'kind':kind,**{k:v[k] for k in ('episodes','successes','actions','response_tokens','tool_errors')}} for label,modes in summaries.items() for mode,kinds in modes.items() for kind,v in kinds.items()]
    with (output/(prefix+'confirmation_summary.csv')).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result
