"""Synthetic CPU integration check of the final cloud artifact auditor."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from agentic_post_training.rollouts.closeout import sample_seed, benchmark_summary, validate_records
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import no_update_metrics, pack_trainer_scores
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward


def test_final_auditor_rechecks_full_192_sample_pack_and_rejects_corruption(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('closeout_test_script',root/'scripts/finish_phase4.py')
    script=importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    class Tokenizer:
        eos_token_id=pad_token_id=0
        def decode(self,ids,**kw):
            return '{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}'+('' if kw['skip_special_tokens'] else '<EOS>')
    tok=Tokenizer(); cases=[EvalCase(f'case-{g}','math','1+2','tool_call','1+2',3) for g in range(48)]
    prompts=[[1] for _ in cases]; rows=[]; groups=[]; records=[]
    for g,case in enumerate(cases):
        group=[]
        for i in range(4):
            s=RolloutSample(case.id,case.id,i,sample_seed(g,i),[1],[2,0],[-.1,-.2],[-.1,-.2],0,'eos',
                tok.decode([2,0],skip_special_tokens=True),tok.decode([2,0],skip_special_tokens=False))
            reward=score_reward(s.text,case,finish_reason='eos')
            rows.append({'sample':asdict(s),'case':asdict(case),'score':reward['score'],'semantic_pass':reward['semantic_pass']})
            old=torch.tensor([-.1,-.2]); new=old.clone().requires_grad_()
            records.append({'group_id':s.group_id,'sample_index':i,'response_token_ids':[2,0],
                'sampler_behavior_token_log_probs':[-.1,-.2], 'trainer_old_token_log_probs':old.tolist(),
                'trainer_new_token_log_probs':old.tolist(),'no_update_check':no_update_metrics(old,new,1e-5),
                'eligible':True,'reward':1.,'advantage':0.})
            group.append(s)
        groups.append(group)
    config={'max_new_tokens':64}
    _,packed,rewards=validate_records(rows,cases,prompts,tok,config)
    closeout={'contract_version':'phase4-closeout-48x4-v1'}
    monkeypatch.setattr(script,'load_run',lambda out:(config,closeout,None,None,tok,rows,groups,packed,rewards))
    monkeypatch.setattr(script,'setup',lambda:(config,closeout,None,None,tok,cases,prompts))
    def write(name,data): (tmp_path/name).write_text(json.dumps(data),encoding='utf-8')
    (tmp_path/'trainer_scores.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    (tmp_path/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    write('reward_audit.json',rewards)
    flat=[s for group in groups for s in group]; bench=flat[:16]
    write('generation_checks.json',{'effective_engine':{'dtype':'torch.float16','logprobs_mode':'raw_logprobs',
        'max_model_len':512,'enforce_eager':True,'enable_prefix_caching':False,'enable_chunked_prefill':True,
        'max_num_seqs':4,'max_num_batched_tokens':512,'max_lora_rank':8},
        'sampling_parameters':dict(n=1,temperature=1,top_p=1,top_k=-1,max_tokens=64,min_tokens=0,
            repetition_penalty=1,presence_penalty=0,frequency_penalty=0,logprobs=1,stop_token_ids=[0],ignore_eos=False,detokenize=False),
        'benchmark_batch_samples':[asdict(s) for s in bench],'benchmark_serial_samples':[asdict(s) for s in bench],
        'benchmark':benchmark_summary(bench,bench,1.,2.),'generation_seconds':2.,'generation_requests':224,
        'generation_response_tokens_per_second':192.,'engine_shutdown_called':True})
    digest=json.loads((root/'configs/trainer_probability_contract.json').read_text())['frozen_fp32_parameter_sha256']
    write('trainer_checks.json',{'initial_parameter_sha256':digest,'final_parameter_sha256':digest,
        'no_grad_accumulated':True,'model_backward_calls':0,'optimizer_steps':0,'scoring_seconds':2.})
    write('run_manifest.json',{'synthetic_unit_test':True})
    batch=pack_trainer_scores(groups,records,0); torch.save(batch,tmp_path/'trainer_score_batch.pt')
    script.audit(tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['new_contract_interface_passed'] and result['artifact_audit_passed']
    assert result['zero_variance_rate']==1 and not result['training_admitted']
    assert result['samples']==192 and result['response_tokens']==384
    batch['response_mask'][0,0,0]=False; torch.save(batch,tmp_path/'trainer_score_batch.pt')
    with pytest.raises(ValueError,match='saved score tensors or masks differ'): script.audit(tmp_path)
