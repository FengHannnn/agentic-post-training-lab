import sys,json,copy,math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.dpo_investigation import build_probes,digest,metrics_from_scores,summarize_probe_metrics,evaluation_summary
from test_dpo_v2 import CharTokenizer
from test_hard_recovery_independent_v2 import native
from test_matched_dpo_v2 import oracle

def test_deterministic_expert_probes_cover_completion_and_keep_test_out():
    probes=build_probes();assert probes==build_probes() and len(probes)==238
    assert len({p['probe_id'] for p in probes})==238
    assert sum(p['kind']=='recovery_first' for p in probes)==112
    assert sum(p['kind']=='recovery_continuation' for p in probes)==96
    assert sum(p['expected_tool']=='submit' for p in probes)==32
    assert {p['split'] for p in probes}=={'train','development'}
    assert sum(p['split']=='train' for p in probes)==84

def test_relative_objective_can_improve_when_both_responses_lose_probability():
    import torch
    from apt_v2.dpo import dpo_loss
    ref={'input_id':'same','chosen':[-.1,-.1],'rejected':[-.2,-.2]}
    current={'input_id':'same','chosen':[-1.,-1.],'rejected':[-4.,-4.]}
    m=metrics_from_scores(current,ref)
    expected=dpo_loss(torch.tensor(-2.),torch.tensor(-8.),torch.tensor(-.2),torch.tensor(-.4),.1).item()
    assert m['loss']==pytest.approx(expected,abs=1e-7) and m['loss']<math.log(2)
    assert m['chosen_mean_logp']<sum(ref['chosen'])/2 and m['relative_margin']>0
    bad=copy.deepcopy(current);bad['input_id']='other'
    with pytest.raises(ValueError,match='binding'):metrics_from_scores(bad,ref)

def test_probe_summary_rejects_unbound_tokens_and_wrong_ce():
    p=build_probes()[:1];m=[{'probe_id':p[0]['probe_id'],'probe_sha256':digest(p[0]),'token_log_probs':[-.1,-.2],'ce':.15}]
    assert summarize_probe_metrics(p,m)['train']['recovery_first']['calculate_change']['examples']==1
    bad=copy.deepcopy(m);bad[0]['ce']=.2
    with pytest.raises(ValueError,match='CE'):summarize_probe_metrics(p,bad)
    bad=copy.deepcopy(m);bad[0]['probe_sha256']='wrong'
    with pytest.raises(ValueError,match='binding'):summarize_probe_metrics(p,bad)

def test_evaluation_replay_catches_seed_tampering_and_locates_real_budget_failure(tmp_path):
    from apt_v2.environment import Environment
    from apt_v2.runtime import run_episode
    from apt_v2.dpo import corrective_action
    from apt_v2.tasks import metric_sql
    from apt_v2.matched_dpo import run_recovery
    from dpo_investigation_common import jsonl,seed_for
    from apt_v2.dpo_investigation import investigation_tasks,investigation_contexts
    from audit_v2_dpo_investigation import audit_evaluations,onset_report
    branches={'parent':{'adapter_sha256':'p','new_dpo_steps':0},'weighted_sft':{'adapter_sha256':'s','new_dpo_steps':0},'dpo_0043':{'adapter_sha256':'a','new_dpo_steps':43},'dpo_0258':{'adapter_sha256':'z','new_dpo_steps':258}}
    cfg={'checkpoints':branches,'eval':{'seed':31,'max_new_tokens':100000,'max_model_len':1000000},'sample_eval':{'seed':32,'rollouts_per_task':1,'temperature':.8},'recovery_eval':{'seed':33}}
    tok=CharTokenizer();ts=investigation_tasks();cs=investigation_contexts();tm={t.id:t for t in ts};allfull=[];allrs=[];jsonl(tmp_path/'recovery_contexts.jsonl',cs)
    for b,spec in branches.items():
        full=[];rs=[]
        for t in ts:
            for mode in ('greedy','sample'):
                nm='greedy' if mode=='greedy' else 'dpo_sample';seed=seed_for(cfg,t,mode,0)
                def policy(messages):
                    env=Environment(t)
                    try:
                        for item in messages[2:]:
                            if item['role']=='assistant':env.step(json.loads(item['content']))
                        action=corrective_action(env)
                        if b=='dpo_0258' and t.id=='development-002' and action['tool']=='calculate_change':
                            action={'tool':'query_sql','arguments':{'sql':metric_sql(t,'current_net_cents').replace('AS current_net_cents','AS change_cents')}}
                        return native(tok,messages,json.dumps(action),nm)
                    finally:env.close()
                e=run_episode(t,policy);e.update(model_branch=b,mode=mode,rollout_index=0,sample_seed=seed,temperature=0 if mode=='greedy' else .8,adapter_sha256=spec['adapter_sha256'])
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                full.append(e)
        for c in cs:
            t=tm[c['task_id']];seed=seed_for(cfg,t,'greedy',0,c);e=run_recovery(t,c,oracle(t,c,tok,'greedy'))
            e.update(model_branch=b,mode='greedy',sample_index=0,sample_seed=seed,temperature=0,adapter_sha256=spec['adapter_sha256'])
            for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
            rs.append(e)
        jsonl(tmp_path/(b+'-complete-episodes.jsonl'),full);jsonl(tmp_path/(b+'-recovery-episodes.jsonl'),rs);allfull.extend(full);allrs.extend(rs)
    summary=evaluation_summary(allfull,allrs,branches);(tmp_path/'comparison_summary.json').write_text(json.dumps(summary))
    actual,calendar,first,evidence=audit_evaluations(tmp_path,cfg,tok)
    assert actual==summary and evidence['complete_episodes_replayed']==32 and evidence['recovery_episodes_replayed']==112
    assert summary['branches']['dpo_0258']['complete']['greedy']['successes']==3
    scores={b:{'pairs':{'by_scenario':{'stale_evidence_repair':{'chosen_ce':.01 if b!='dpo_0258' else 1.}}}} for b in branches}
    onset=onset_report(cfg,scores,summary)
    assert onset['first_task_greedy_regression_step']==258 and onset['first_task_sample_regression_step']==258
    assert onset['first_recovery_greedy_regression_step'] is None
    full[0]['sample_seed']+=1;p=tmp_path/'dpo_0258-complete-episodes.jsonl';p.unlink();jsonl(p,full)
    with pytest.raises(ValueError,match='seed'):audit_evaluations(tmp_path,cfg,tok)

def test_checkpoint_validator_checks_tensor_and_ancestry(tmp_path,monkeypatch):
    import dpo_investigation_common as common
    from common import file_hash
    monkeypatch.setattr(common,'ROOT',tmp_path)
    p=tmp_path/'artifacts/origin/checkpoint-0043';(p/'adapter').mkdir(parents=True);(p/'adapter/a.json').write_text('{}');(p/'adapter/adapter_model.safetensors').write_bytes(b'weights')
    c={'algorithm':'hard_recovery_weighted_dpo','steps':43,'cursor':169,'epochs_completed':1,'source_sft_updates':255,'source_dpo_updates':11,'total_sft_updates':255,'total_dpo_updates':54,'adapter_sha256':{n:file_hash(p/'adapter'/n) for n in ('a.json','adapter_model.safetensors')}}
    (p/'complete.json').write_text(json.dumps(c));spec={'run':'origin','checkpoint':'checkpoint-0043','new_dpo_steps':43,'total_sft_updates':255,'total_dpo_updates':54,'complete_sha256':file_hash(p/'complete.json'),'adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors')}
    assert common.validate_checkpoint(spec)['adapter/adapter_model.safetensors']==spec['adapter_sha256']
    bad=copy.deepcopy(spec);bad['total_dpo_updates']=55
    with pytest.raises(ValueError,match='ancestry'):common.validate_checkpoint(bad)
    (p/'adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):common.validate_checkpoint(spec)

def test_runner_failure_archives_logs_and_keeps_zero_updates(tmp_path,monkeypatch,capsys):
    import run_v2_dpo_investigation as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'configs/dpo-regression-checkpoints-v1.json').write_text(json.dumps({'run_id':'investigate','max_wall_seconds':14400}))
    (tmp_path/'dpo_investigation_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/dpo_investigation_common.py').write_text("print('diagnosis preflight',flush=True)\nraise RuntimeError('missing checkpoint')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'diagnosis preflight' in capsys.readouterr().out
    status=json.loads((tmp_path/'artifacts/investigate/run_status.json').read_text());assert status['optimizer_updates']==0 and status['training_enabled'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'investigate-review.zip') as z:assert 'investigate/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()


def test_score_audit_recomputes_raw_logps_and_rejects_changed_metric(tmp_path,monkeypatch):
    import audit_v2_dpo_investigation as auditor
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
    cfg={'max_length':1000000,'checkpoints':{'parent':{'adapter_sha256':'p'},'dpo_0043':{'adapter_sha256':'d'}}}
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    jsonl(tmp_path/'pairs.jsonl',[pair]);jsonl(tmp_path/'probes.jsonl',probes);write('weights.json',{'weights':[1.]})
    write('reference_scores.json',{'baseline_adapter_sha256':'p','pair_ids':['one'],'scores':[ref]});write('reference_recheck.json',{'fresh_scores':[ref],'maximum_token_logp_difference':0.})
    summaries={}
    for branch,spec in cfg['checkpoints'].items():
        score=copy.deepcopy(ref)
        if branch!='parent':score['chosen']=[-1.]*len(ref['chosen']);score['rejected']=[-20.]*len(ref['rejected'])
        m=[metrics_from_scores(score,ref)];summary={'pairs':summarize_pair_metrics([pair],encoded,m,[ref],[1.]),'probes':summarize_probe_metrics(probes,pm)}
        saved={'branch':branch,'adapter_sha256':spec['adapter_sha256'],'pair_scores':[score],'pair_metrics':m,'probe_metrics':pm,'summary':summary}
        write(branch+'-scores.json',saved);summaries[branch]=summary
    write('scores_summary.json',{'branches':summaries,'training_enabled':False,'optimizer_updates':0,'maximum_parent_reference_difference':0.})
    actual,maximum=auditor.audit_scores(tmp_path,cfg,tok);assert actual==summaries and maximum==0.
    saved['pair_metrics'][0]['loss']+=.1;write('dpo_0043-scores.json',saved)
    with pytest.raises(ValueError,match='metrics'):auditor.audit_scores(tmp_path,cfg,tok)


def test_real_peft_adapter_switch_preserves_parent_scores_and_disk_weights(tmp_path):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig,PeftModel
    from apt_v2.dpo_training import reference_scores
    from common import file_hash
    torch.manual_seed(73)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32))
    clean=copy.deepcopy(base);model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    pa=tmp_path/'parent';pb=tmp_path/'later';model.save_pretrained(pa)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.add_(.3)
    model.save_pretrained(pb)
    hashes={str(p):file_hash(p) for d in (pa,pb) for p in d.iterdir() if p.is_file()}
    scored=PeftModel.from_pretrained(clean,pa,adapter_name='parent',is_trainable=False).eval();scored.load_adapter(pb,adapter_name='dpo_0043',is_trainable=False)
    e={'prompt':[1,7],'chosen':[3,4,2],'rejected':[3,5,2]}
    scored.set_adapter('parent');scored.eval();before=reference_scores(scored,e)
    scored.set_adapter('dpo_0043');scored.eval();later=reference_scores(scored,e)
    scored.set_adapter('parent');scored.eval();again=reference_scores(scored,e)
    assert before==again and before['input_id']==later['input_id']
    assert max(abs(a-b) for k in ('chosen','rejected') for a,b in zip(before[k],later[k]))>1e-6
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
