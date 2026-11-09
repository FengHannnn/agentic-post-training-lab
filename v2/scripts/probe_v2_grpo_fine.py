"""Sixteen paired diagnostic episodes at the same actual update count; no training."""
import argparse,sys,time
from pathlib import Path
from grpo_common import read,atomic_json,jsonl,verify_inputs,file_hash,under
from grpo_fine_common import states_at
from apt_v2.grpo_fine_study import development_tasks,probe_plan
from apt_v2.grpo import run_continuation,replay,digest
from apt_v2.grpo_evaluation import evaluation_key,plan_key
from apt_v2.grpo_curriculum import build_context


def state_for_update(output,updates=12):
    return next(((i,s) for i,s in enumerate(states_at(output)) if s['updates']==updates),None)


def verify_probe(output,spec,tok,local=False,data=None):
    from grpo_vllm import projected_checkpoint_tensors
    cfg=verify_inputs(output,runtime=False,local=local);path=output/'diagnostics/common-update-0012.json'
    data=data if data is not None else read(path);adapter=under(output,spec['policy']['checkpoint'])/'adapter'
    expected_state=state_for_update(output)
    if (expected_state is None or spec!={'group_cursor':expected_state[0],'policy':expected_state[1],'actual_updates':12}
            or data['spec']!=spec or data['config_sha256']!=file_hash(output/'config.json')
            or file_hash(adapter/'adapter_model.safetensors')!=spec['policy']['adapter_sha256']):
        raise ValueError('common-update probe checkpoint differs')
    plan=probe_plan(cfg)
    if read(output/'common_update_probe_plan.json')!=plan:raise ValueError('frozen diagnostic plan differs')
    actual={evaluation_key(e):e for e in data['episodes']};expected={plan_key(p):p for p in plan}
    if len(actual)!=16 or len(data['episodes'])!=16 or actual.keys()!=expected.keys():raise ValueError('common probe paired coverage differs')
    tasks={t.id:t for t in development_tasks()}
    for key,e in actual.items():
        p=expected[key];sampling={**cfg['sampling'],'temperature':p['temperature']}
        if (e['sampling']!=sampling or e['seed']!=p['seed'] or e['rollout_index']!=p['rollout_index']
                or e['policy_version']!=12 or e['model_branch']!='common_update_0012' or e['split']!='development'):
            raise ValueError('common probe sampling identity differs')
        replay(tasks[e['task_id']],e,tok,build_context)
    proof=data['worker_proof'];projected=projected_checkpoint_tensors(adapter)
    if (len(proof)!=1 or not proof[0]['passed'] or proof[0]['adapter_sha256']!=spec['policy']['adapter_sha256']
            or proof[0]['registered_ids']!=[1] or 1 not in proof[0]['active_slots']
            or proof[0]['tensors']!=projected or proof[0]['loaded_tensor_sha256']!=digest(projected)):
        raise ValueError('common probe native adapter tensors differ')
    return {'passed':True,'actual_updates':12,'group_cursor':spec['group_cursor'],'episodes':16,
            'successes':sum(e['success'] for e in data['episodes']),'used_for_selection':False,
            'by_kind':{kind:{'episodes':sum(bool(e['scenario'])==(kind=='targeted') for e in data['episodes']),
                'successes':sum(e['success'] for e in data['episodes'] if bool(e['scenario'])==(kind=='targeted'))} for kind in ('complete','targeted')}}


def run(output,spec):
    from tqdm.auto import tqdm
    from grpo_vllm import engine_and_tokenizer,BehaviorPolicy,shutdown
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);path=output/'diagnostics/common-update-0012.json'
    if path.exists():return verify_probe(output,spec,AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True))
    adapter=under(output,spec['policy']['checkpoint'])/'adapter';engine=None;episodes=[];start=time.perf_counter()
    partial=output/'diagnostics'/f'partial-0012-{time.time_ns()}.jsonl';partial.parent.mkdir(parents=True,exist_ok=True)
    try:
        engine,tok,lora=engine_and_tokenizer(cfg,adapter);tasks={t.id:t for t in development_tasks()}
        with tqdm(total=16,desc='Common update 12',unit='episode',file=sys.stdout,disable=False,dynamic_ncols=True) as bar:
            for item in probe_plan(cfg):
                sampling={**cfg['sampling'],'temperature':item['temperature']}
                policy=BehaviorPolicy(engine,tok,lora,sampling,item['seed'],lambda n:bar.set_postfix(turn=policy.calls,tokens=n,refresh=True))
                e=run_continuation(tasks[item['task_id']],item['scenario'],policy,sampling,12,item['seed'],item['rollout_index'],cfg['tool_cost'],build_context)
                e.update(model_branch='common_update_0012',evaluation_mode=item['mode']);replay(tasks[item['task_id']],e,tok,build_context)
                episodes.append(e);jsonl(partial,e);bar.update(1)
        proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),spec['policy']['adapter_sha256']),timeout=60)
        data={'spec':spec,'config_sha256':file_hash(output/'config.json'),'episodes':episodes,'worker_proof':proof,'elapsed_seconds':time.perf_counter()-start}
        check=verify_probe(output,spec,tok,data=data);atomic_json(path,data);atomic_json(output/'diagnostics/common-update-0012-audit.json',check);return check
    finally:
        if engine is not None:shutdown(engine)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();run(a.output,read(a.spec))
