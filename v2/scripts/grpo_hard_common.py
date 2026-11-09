"""Hard-recovery preflight configuration, native replay and immutable-model audit."""
import csv,re,sys
from pathlib import Path
from dataclasses import asdict
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,under,runtime_identity,validate_runtime
from grpo_c4_common import driver_identity
from apt_v2.grpo import replay,validate_native,digest
from apt_v2.grpo_hard_preflight import tasks,plan,context,frozen_inputs,summarize
from apt_v2.grpo_credit import LOCAL

W_SHA='f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f'


def validate_config(cfg):
    sampling={'temperature':1,'top_p':1,'top_k':-1,'min_p':0,'repetition_penalty':1,'presence_penalty':0,
              'frequency_penalty':0,'min_tokens':0,'max_new_tokens':256,'max_model_len':8192,'logprobs_mode':'processed_logprobs'}
    if (cfg['run_id']!='grpo-hard-recovery-preflight-v1-20261109' or cfg['study_phase']!='hard_recovery_preflight'
            or cfg['max_wall_seconds']!=7200 or cfg['packaging_reserve_seconds']!=180 or cfg['seed']!=2026110917
            or cfg['new_training_updates']!=0 or cfg['expected_episodes']!=96 or cfg['k']!=4 or cfg['epsilon']!=1e-8
            or cfg['credit_method']!=LOCAL or cfg['tool_cost']!=.01 or cfg['sampling']!=sampling
            or cfg['adapter_sha256']!=W_SHA or cfg['required_gpu_name']!='NVIDIA L40S'
            or cfg['base_model']!='Qwen/Qwen2.5-1.5B-Instruct' or cfg['revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
            or cfg['release_manifest']!='grpo_hard_release_manifest.json'):
        raise ValueError('frozen train-only 96-episode two-hour preflight differs')
    for name in ('base_dir','initial_dir','contract_report','validation_report'):under(ROOT,cfg[name])
    return cfg


def verify_inputs(output,local=False):
    cfg=validate_config(read(output/'config.json'));verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    binding=read(output/'input_binding.json');verify_files(output,binding['files'])
    if binding['config_sha256']!=file_hash(output/'config.json'):raise ValueError('preflight config binding differs')
    for name,value in frozen_inputs(cfg).items():
        if read(output/name)!=value:raise ValueError('frozen paired preflight input differs: '+name)
    identity=read(output/'model_identity.json')
    if identity['model_id']!=cfg['base_model'] or identity['resolved_revision']!=cfg['revision']:raise ValueError('preflight base identity differs')
    if file_hash(output/'initial/adapter/adapter_model.safetensors')!=W_SHA:raise ValueError('immutable W changed')
    if not local:
        validate_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'))
        if driver_identity()!=read(output/'driver_identity.json'):raise ValueError('same-job GPU driver differs')
        verify_files(under(ROOT,cfg['base_dir']),{n:h for n,h in identity['files_sha256'].items() if not n.endswith('.safetensors')})
    return cfg


def verify_session(output,session,tok,cfg):
    from grpo_vllm import projected_checkpoint_tensors
    from grpo_sampler_probe import verify_capture
    if (not re.fullmatch(r'[0-9]+',session['session_id']) or session['config_sha256']!=file_hash(output/'config.json')
            or session['adapter_sha256']!=W_SHA or session['hardware']!=read(output/'execution_environment.json')
            or session['driver']!=read(output/'driver_identity.json') or session['new_training_updates']!=0
            or session['max_loras']!=1 or session['lora_id']!=1):raise ValueError('frozen preflight session differs')
    projected=projected_checkpoint_tensors(output/'initial/adapter');proof=session['worker_proof']
    if (len(proof)!=1 or not proof[0]['passed'] or proof[0]['adapter_sha256']!=W_SHA
            or proof[0]['registered_ids']!=[1] or 1 not in proof[0]['active_slots'] or proof[0]['tensor_count']!=112
            or proof[0]['tensors']!=projected or proof[0]['loaded_tensor_sha256']!=digest(projected)):
        raise ValueError('native W tensor proof differs')
    index=session['first_plan_index']
    if type(index) is not int or not 0<=index<96:raise ValueError('valid first session plan index required')
    item=plan(cfg)[index];task=next(t for t in tasks() if t.id==item['task_id']);messages=context(task,item['scenario'])['messages']
    validate_native(tok,messages,session['first_generation'],cfg['sampling'])
    verify_capture(proof[0]['same_forward_capture'],session['first_generation'],cfg['sampling'],item['seed'],W_SHA)
    return True


def verify_episode(output,item,e,tok,cfg,session):
    if (e['plan_index']!=item['index'] or e['task_id']!=item['task_id'] or e['scenario']!=item['scenario']
            or e['base_scenario']!=item['base_scenario'] or e['level']!=item['level'] or e['rollout_index']!=item['rollout_index']
            or e['seed']!=item['seed'] or e['sampling']!=cfg['sampling'] or e['policy_version']!=0 or e['split']!='train'
            or e['model_branch']!='weighted_sft' or e['session_id']!=session['session_id']
            or e['worker_proof_sha256']!=digest(session['worker_proof']) or e['start']['tool_cost']!=cfg['tool_cost']):
        raise ValueError('paired native episode identity differs')
    if item['index']==session['first_plan_index'] and e['turns'][0]['generation']!=session['first_generation']:
        raise ValueError('first generation session binding differs')
    task=next(t for t in tasks() if t.id==item['task_id'])
    return replay(task,e,tok,context)


def verify_result(output,tok,local=False,data=None):
    cfg=verify_inputs(output,local);data=read(output/'results.json') if data is None else data
    if data['config_sha256']!=file_hash(output/'config.json') or data['new_training_updates']!=0:raise ValueError('frozen result identity differs')
    rows=data['episodes'];sessions=data['sessions'];expected=plan(cfg)
    if len(rows)!=96 or not sessions or len({s['session_id'] for s in sessions})!=len(sessions):raise ValueError('complete distinct preflight coverage required')
    by={s['session_id']:s for s in sessions}
    for s in sessions:
        if read(output/'sessions'/(s['session_id']+'.json'))!=s:raise ValueError('session transaction differs')
        verify_session(output,s,tok,cfg)
    if {p.name for p in (output/'episodes').glob('*.json')}!={f'{i:04}.json' for i in range(96)}:
        raise ValueError('exact committed episode file coverage required')
    if {p.stem for p in (output/'sessions').glob('*.json')}!=set(by):raise ValueError('exact generation session coverage required')
    from tqdm.auto import tqdm
    actions=0
    for item,e in tqdm(zip(expected,rows),total=96,desc='Replay paired episodes',unit='episode',file=sys.stdout,disable=False):
        if e['session_id'] not in by or read(output/'episodes'/f"{item['index']:04}.json")!=e:raise ValueError('committed episode transaction differs')
        actions+=verify_episode(output,item,e,tok,cfg,by[e['session_id']])
    summary=summarize(rows)
    if data['summary']!=summary:raise ValueError('local credit or paired summary recomputation differs')
    return {'passed':True,'scope':'local_returned_files' if local else 'cloud_files_and_runtime','episodes_replayed':96,
            'actions_replayed':actions,'sessions':len(sessions),'new_training_updates':0,'immutable_W_sha256':W_SHA,'summary':summary}


def audit(output,local=False):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    result=verify_result(output,tok,local);prefix='local_' if local else ''
    atomic_json(output/(prefix+'preflight_audit.json'),result)
    with (output/(prefix+'episode_summary.csv')).open('w',newline='',encoding='utf8') as f:
        rows=result['summary']['episode_rows'];writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result
