import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
from paired_runtime import check_job_runtime

def identity():return json.loads((ROOT/'configs/dpo-continuation-ce-control-v1.json').read_text())['required_execution_environment']
def test_user_uuid_change_allows_new_matched_job_but_pins_within_job():
    old=identity();actual=dict(old,gpu_uuid='92e14e85-628d-e5a4-af16-0db4defb2afb');cfg={'required_gpu_name':'NVIDIA L40S','required_execution_environment':old}
    report=check_job_runtime(cfg,actual)
    assert report['passed'] and report['allowed_differences']=={'gpu_uuid':{'expected':old['gpu_uuid'],'actual':actual['gpu_uuid']}}
    check_job_runtime(cfg,actual,actual)
    with pytest.raises(ValueError,match='gpu_uuid'):check_job_runtime(cfg,dict(actual,gpu_uuid='third-card'),actual)

@pytest.mark.parametrize('key,value',[('torch','other'),('gpu_name','NVIDIA L20'),('allow_tf32_matmul',True),('gpu_uuid',None)])
def test_new_job_rejects_non_uuid_drift_with_actionable_details(key,value):
    old=identity();actual=copy.deepcopy(old);actual[key]=value;cfg={'required_gpu_name':'NVIDIA L40S','required_execution_environment':old}
    with pytest.raises(ValueError,match=key):check_job_runtime(cfg,actual)

def test_precision_relaxation_is_limited_to_post_inference_checks():
    old=identity();cfg={'required_gpu_name':'NVIDIA L40S','required_execution_environment':old};changed=dict(old,allow_tf32_matmul=True)
    with pytest.raises(ValueError,match='allow_tf32_matmul'):check_job_runtime(cfg,changed,old)
    assert check_job_runtime(cfg,changed,old,precision=False)['passed']

def test_prepare_records_actual_runtime_and_diff_before_rejection(tmp_path,monkeypatch):
    import dpo_continuation_ce_paired_common as common
    from common import write,read
    cfg=json.loads((ROOT/'configs/dpo-continuation-ce-paired-v1.json').read_text());(tmp_path/'configs').mkdir();write(tmp_path/'configs/dpo-continuation-ce-paired-v1.json',cfg)
    output=tmp_path/'output';output.mkdir();write(output/'config.json',cfg);actual=dict(cfg['required_execution_environment'],torch='changed-version')
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'runtime_identity',lambda:actual)
    with pytest.raises(ValueError,match='torch'):common.prepare(output)
    assert read(output/'execution_environment.json')==actual
    report=read(output/'runtime_precheck.json');assert not report['passed'] and report['rejected_differences']['torch']['actual']=='changed-version'
    assert not (output/'training').exists()

from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native
def test_four_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_continuation_ce_paired_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_continuation_ce_paired import audit_results
    cfg={'evaluation_branches':['weighted_sft','historical_dpo_ce','chosen_control','continuation_ce'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
        'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33,'samples_per_context':1,'temperature':.8}}
    tok=CharTokenizer();ts=tasks();tm={t.id:t for t in ts};cs=contexts();allfull=[];allrs=[];allaa=[]
    for b in cfg['evaluation_branches']:
        full=[];rs=[];aa=[]
        for task in ts:
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0)
                def policy(messages):
                    env=Environment(task)
                    try:
                        for m in messages[2:]:
                            if m['role']=='assistant':env.step(json.loads(m['content']))
                        return native(tok,messages,json.dumps(corrective_action(env)),nm)
                    finally:env.close()
                e=run_episode(task,policy);e.update(model_branch=b,mode=mode,rollout_index=0,sample_seed=seed,temperature=0 if mode=='greedy' else .8)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                full.append(e)
        for c in cs:
            task=tm[c['task_id']]
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,task,mode,0,c)
                e=run_recovery(task,c,oracle(task,c,tok,nm));temperature=0 if mode=='greedy' else .8
                e.update(model_branch=b,mode=mode,sample_index=0,sample_seed=seed,temperature=temperature)
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                rs.append(e);g=e['turns'][0]['generation'];actual,d=diagnose_action(task,c,g['text'])
                aa.append({'model_branch':b,'mode':mode,'task_id':task.id,'family_id':task.family_id,'split':'development','context_id':c['context_id'],
                    'scenario':c['scenario'],'sample_index':0,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d})
        jsonl(tmp_path/(b+'-complete-episodes.jsonl'),full);jsonl(tmp_path/(b+'-recovery-episodes.jsonl'),rs);jsonl(tmp_path/(b+'-first-actions.jsonl'),aa)
        allfull.extend(full);allrs.extend(rs);allaa.extend(aa)
    summary=result_summary(allfull,allrs,allaa,cfg);(tmp_path/'comparison_summary.json').write_text(json.dumps(summary))
    result,calendar,evidence=audit_results(tmp_path,cfg,tok)
    assert evidence['complete_episodes_replayed']==32 and evidence['recovery_episodes_replayed']==224
    assert result['paired']['chosen_control_to_continuation_ce']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'continuation_ce-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)

from apt_v2.dpo_investigation import build_probes,digest,metrics_from_scores,summarize_probe_metrics
def test_five_model_scores_recompute_and_reject_changed_metric(tmp_path,monkeypatch):
    import audit_v2_dpo_continuation_ce_paired as auditor
    from apt_v2.reference_numerics import compare_reference_caches
    from apt_v2.dpo_recovery import encode_recovery_pair
    from apt_v2.dpo_training import input_id
    from apt_v2.sft import encode_example
    from apt_v2.dpo_investigation import summarize_pair_metrics
    from dpo_investigation_common import jsonl
    tok=CharTokenizer();messages=[{'role':'user','content':'fixture'}];chosen='{"tool":"read_policy","arguments":{}}';rejected='bad'
    pair={'pair_id':'one','messages':messages,'chosen':chosen,'rejected':rejected,'native_prompt_token_ids':tok.apply_chat_template(messages,add_generation_prompt=True),'native_rejected_token_ids':[ord(c)+1 for c in rejected]+[0],'scenario':'stale_evidence_repair'}
    encoded=[encode_recovery_pair(tok,pair,1000000)];ref={'input_id':input_id(encoded[0]),'chosen':[-.1]*len(encoded[0]['chosen']),'rejected':[-.2]*len(encoded[0]['rejected'])}
    probe=build_probes()[0];probes=[probe];monkeypatch.setattr(auditor,'build_probes',lambda:probes)
    pe=encode_example(tok,probe,1000000);pm=[{'probe_id':probe['probe_id'],'probe_sha256':digest(probe),'token_log_probs':[-.1]*pe['response_length'],'ce':.1}]
    cfg={'max_length':1000000,'beta':.1,'checkpoints':{b:{'adapter_sha256':b} for b in ['parent','weighted_sft','historical_dpo_ce','chosen_control','continuation_ce']}}
    cfg['checkpoints']['parent']['adapter_sha256']='p'
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    jsonl(tmp_path/'pairs.jsonl',[pair]);jsonl(tmp_path/'probes.jsonl',probes);write('weights.json',{'weights':[1.]})
    write('reference_scores.json',{'baseline_adapter_sha256':'p','pair_ids':['one'],'scores':[ref]});cache={'baseline_adapter_sha256':'p','pair_ids':['one'],'scores':[ref]};write('reference_recheck.json',{'fresh_scores':[ref],'maximum_token_logp_difference':0.,'numerics':compare_reference_caches(cache,cache)})
    summaries={}
    for branch,spec in cfg['checkpoints'].items():
        score=copy.deepcopy(ref)
        if branch!='parent':score['chosen']=[-1.]*len(ref['chosen']);score['rejected']=[-20.]*len(ref['rejected'])
        m=[metrics_from_scores(score,ref)];summary={'pairs':summarize_pair_metrics([pair],encoded,m,[ref],[1.]),'probes':summarize_probe_metrics(probes,pm)}
        saved={'branch':branch,'adapter_sha256':spec['adapter_sha256'],'pair_scores':[score],'pair_metrics':m,'probe_metrics':pm,'summary':summary}
        write(branch+'-scores.json',saved);summaries[branch]=summary
    write('scores_summary.json',{'branches':summaries,'training_enabled':False,'optimizer_updates':0,'maximum_parent_reference_difference':0.})
    actual,maximum=auditor.audit_scores(tmp_path,cfg,tok);assert actual==summaries and maximum==0.
    saved['pair_metrics'][0]['loss']+=.1;write('continuation_ce-scores.json',saved)
    with pytest.raises(ValueError,match='metrics'):auditor.audit_scores(tmp_path,cfg,tok)


def test_paired_runner_failure_archives_and_protects_both_update_counts(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_continuation_ce_paired as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/dpo-continuation-ce-paired-v1.json').write_text(json.dumps({'run_id':'ce-paired','max_wall_seconds':14400}))
    (tmp_path/'dpo_continuation_ce_paired_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/dpo_continuation_ce_paired_common.py').write_text("print('CE preflight',flush=True)\nraise RuntimeError('hardware differs')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'CE preflight' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/ce-paired/run_status.json').read_text());assert status['optimizer_updates']==0 and status['branch_optimizer_updates']=={'chosen_control':0,'continuation_ce':0}
    import zipfile
    with zipfile.ZipFile(tmp_path/'ce-paired-review.zip') as z:assert 'ce-paired/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()



def test_paired_runner_preserves_completed_control_when_second_training_fails(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_continuation_ce_paired as runner
    from common import write,read
    import zipfile
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir();write(tmp_path/'configs/dpo-continuation-ce-paired-v1.json',{'run_id':'paired-failure','max_wall_seconds':14400});write(tmp_path/'dpo_continuation_ce_paired_release_manifest.json',{'files':{}})
    (tmp_path/'scripts/dpo_continuation_ce_paired_common.py').write_text("print('prepare ok')")
    (tmp_path/'scripts/train_v2_dpo_continuation_ce_paired.py').write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--branch');a=p.parse_args();o=Path(a.output)/'training'/a.branch;o.mkdir(parents=True)\nif a.branch=='chosen_control':\n (o/'training_summary.json').write_text(json.dumps({'optimizer_updates':258}))\nelse:\n (o/'training_steps.jsonl').write_text('{}\\n{}\\n');raise RuntimeError('second branch stopped')\n")
    with pytest.raises(RuntimeError,match='continuation-ce failed'):runner.run()
    status=read(tmp_path/'artifacts/paired-failure/run_status.json');assert status['optimizer_updates']==260 and status['branch_optimizer_updates']=={'chosen_control':258,'continuation_ce':2}
    with zipfile.ZipFile(tmp_path/'paired-failure-review.zip') as z:assert 'paired-failure/training/chosen_control/training_summary.json' in z.namelist()

def test_cross_job_uuid_flexibility_requires_both_fresh_training_variants(tmp_path,monkeypatch):
    import dpo_continuation_ce_paired_common as common
    from common import write,file_hash
    cfg=json.loads((ROOT/'configs/dpo-continuation-ce-paired-v1.json').read_text());old=json.loads((ROOT/'configs/dpo-chosen-ce-control-v1.json').read_text());origin=tmp_path/'artifacts'/cfg['comparison_source_run'];origin.mkdir(parents=True)
    for name,value in [('config.json',old),('audit.json',{'passed':True}),('run_status.json',{'error':None}),('execution_environment.json',cfg['required_execution_environment'])]:write(origin/name,value)
    cfg.update(comparison_config_sha256=file_hash(origin/'config.json'),comparison_audit_sha256=file_hash(origin/'audit.json'),comparison_run_status_sha256=file_hash(origin/'run_status.json'));monkeypatch.setattr(common,'ROOT',tmp_path);common.check_contract(cfg)
    for key,value in [('training_variants',{'continuation_ce':{'chosen_ce_coefficient':1}}),('maximum_optimizer_updates_this_job',258),('ce_action_exposures',1015),('hard_weight',2)]:
        changed=copy.deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError,match='fixed'):common.check_contract(changed)
