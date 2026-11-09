import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from dpo_uniform_common import intervention

def test_only_weight_change_is_accepted_and_learning_rate_or_protocol_change_is_rejected():
    old=json.loads((ROOT/'configs/hard-recovery-dpo-matched-v1.json').read_text());new=json.loads((ROOT/'configs/dpo-uniform-control-v1.json').read_text())
    assert intervention(old,new)['changed_training_fields']==['hard_weight']
    for key in ('learning_rate','seed','beta','epochs','adapter_sha256'):
        bad=copy.deepcopy(new);bad[key]=old[key]+1 if isinstance(old[key],(int,float)) else 'other'
        with pytest.raises(ValueError,match='only hard_weight'):intervention(old,bad)
    bad=copy.deepcopy(new);bad['sample_eval']['seed']+=1
    with pytest.raises(ValueError,match='protocol'):intervention(old,bad)

def test_all_one_weights_match_existing_unweighted_optimizer_and_adam_state():
    import torch
    from test_matched_dpo_v2 import tiny_model
    from apt_v2.matched_dpo import optimize_weighted_dpo
    from apt_v2.dpo_recovery_training import optimize_epochs
    from apt_v2.dpo_training import reference_scores
    a=tiny_model();b=copy.deepcopy(a);encoded=[{'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,i,2]} for i in (5,6,8)]
    ref=[reference_scores(a,e) for e in encoded];snapshot=copy.deepcopy(ref)
    oa=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.0001,weight_decay=0);ob=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.0001,weight_decay=0)
    ha=optimize_weighted_dpo(a,encoded,ref,[1.]*3,oa,2,51,.1,1,2);hb=optimize_epochs(b,encoded,ref,ob,2,51,.1,1,2)
    assert ha==hb and ref==snapshot
    assert all(torch.equal(p,dict(b.named_parameters())[n]) for n,p in a.named_parameters())
    for sa,sb in zip(oa.state.values(),ob.state.values()):
        assert all(torch.equal(sa[k],sb[k]) for k in sa)

def test_uniform_runner_failure_archive_and_existing_output_protection(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_uniform as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{});monkeypatch.setattr(runner,'resource_snapshot',lambda *a:None)
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/dpo-uniform-control-v1.json').write_text(json.dumps({'run_id':'uniform','max_wall_seconds':14400}))
    (tmp_path/'dpo_uniform_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/dpo_uniform_common.py').write_text("print('uniform preflight',flush=True)\nraise RuntimeError('missing control')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'uniform preflight' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/uniform/run_status.json').read_text());assert status['optimizer_updates']==0 and status['error']
    import zipfile
    with zipfile.ZipFile(tmp_path/'uniform-review.zip') as z:assert 'uniform/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()

from test_matched_dpo_v2 import oracle,run_recovery,make_tasks,corrective_action
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native
def test_four_branch_complete_recovery_audit_and_tamper(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from dpo_uniform_common import tasks,contexts,seed_for,jsonl,result_summary
    from apt_v2.recovery_diagnostics import diagnose_action
    from audit_v2_dpo_uniform import audit_results
    cfg={'evaluation_branches':['parent','weighted_sft','weighted_dpo','dpo'],'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},
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
    assert result['paired']['weighted_dpo_to_dpo']['sample']['recovery']['both_success']==28
    assert all(x['result']['calendar_consistent_success'] for x in calendar)
    aa[0]['diagnostic']['productive']=False;p=tmp_path/'dpo-first-actions.jsonl';p.unlink();jsonl(p,aa)
    with pytest.raises(ValueError,match='binding'):audit_results(tmp_path,cfg,tok)
