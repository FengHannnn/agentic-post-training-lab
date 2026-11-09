"""Isolated C2 collection/update stages; GPU memory is released on process exit."""
import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from dataclasses import asdict
from grpo_common import (ROOT, read, atomic_json, jsonl, file_hash, verify_files, verify_inputs,
                         under, prepare, start_spec, checkpoint_binding)
from apt_v2.grpo import (digest, initial_state, run_continuation, replay, group_advantages,
                         validate_native)
from apt_v2.tasks import make_tasks
from apt_v2.grpo_credit import compute_credit


def task_for(task_id, cfg=None):
    if cfg and cfg.get('curriculum_schema'):
        from apt_v2.grpo_curriculum import training_tasks
        return next(t for t in training_tasks(cfg) if t.id == task_id)
    return next(t for t in make_tasks() if t.id == task_id)


def planned_spec(cfg, index):
    if cfg.get('curriculum_schema'):
        from apt_v2.grpo_curriculum import start_spec as curriculum_spec
        return curriculum_spec(cfg, index)
    return start_spec(index, cfg['seed'], cfg['recovery_scenarios'])


def context_builder(cfg):
    if cfg.get('curriculum_schema'):
        from apt_v2.grpo_curriculum import build_context
        return build_context
    return None


def verify_group(output, index, before, tokenizer=None, local=False):
    cfg = verify_inputs(output, runtime=False, local=local)
    g = read(output / 'groups' / f'group-{index:04}.json')
    spec = planned_spec(cfg, index)
    if g['spec'] != spec or g['policy'] != before or g['config_sha256'] != file_hash(output/'config.json'):
        raise ValueError('frozen group schedule or checkpoint differs')
    task = task_for(spec['task_id'], cfg)
    for i, episode in enumerate(g['episodes']):
        if (episode['seed'] != spec['seed'] + i*1000 or episode['rollout_index'] != i
                or episode['sampling'] != cfg['sampling'] or episode['policy_version'] != before['version']
                or episode['task_id'] != spec['task_id'] or episode['scenario'] != spec['scenario']):
            raise ValueError('independent rollout policy/seed/start binding differs')
        replay(task, episode, tokenizer, context_builder(cfg))
    credit = compute_credit(g['episodes'], cfg['k'], cfg['epsilon'], cfg.get('credit_method','outcome'))
    if credit != g['credit']:
        raise ValueError('group advantages differ')
    proof = g['worker_proof']
    if (not proof or any(not p['passed'] or p['adapter_sha256'] != before['adapter_sha256']
                        or p['registered_ids'] != [1] or 1 not in p['active_slots']
                        or p['tensor_count'] != 112 or len(p['tensors']) != 112
                        or p['loaded_tensor_sha256'] != digest(p['tensors']) for p in proof)):
        raise ValueError('worker tensor identity differs')
    from grpo_vllm import verify_probability_probe
    verify_probability_probe(g['probability_probe'],g['episodes'][0]['turns'][0]['generation'],cfg,spec['seed'],g['worker_proof'])
    return g


def collect(output, index, before):
    from tqdm.auto import tqdm
    from grpo_vllm import engine_and_tokenizer, BehaviorPolicy, probability_probe, shutdown
    cfg = verify_inputs(output)
    target = output / 'groups' / f'group-{index:04}.json'
    if target.exists():
        verify_group(output, index, before)
        print('Verified committed rollout group', index, flush=True)
        return
    spec = planned_spec(cfg, index)
    task = task_for(spec['task_id'], cfg)
    adapter = under(output, before['checkpoint'])/'adapter'
    if file_hash(adapter/'adapter_model.safetensors') != before['adapter_sha256']:
        raise ValueError('sampling checkpoint weight hash differs')
    engine = None
    partial = output/'groups'/f'partial-{index:04}-{time.time_ns()}.jsonl'
    partial.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    episodes = []
    tokens = 0
    try:
        if cfg.get('curriculum_schema') in ('outcome_curriculum_v1','fine_credit_ablation_v1','alias_curriculum_v1','repair_study_v1'):
            setup={'seed':spec['seed'],'policy_sha256':before['adapter_sha256'],'token_limit':16,'sampling':cfg['sampling']}
            engine, tok, lora = engine_and_tokenizer(cfg, adapter, setup)
        else:
            engine, tok, lora = engine_and_tokenizer(cfg, adapter)
        with tqdm(total=cfg['k'], desc=f'Group {index:04} {spec["kind"]}', unit='rollout',
                  file=sys.stdout, disable=False, dynamic_ncols=True) as bar:
            def action(count):
                nonlocal tokens
                tokens += count
                bar.set_postfix(tokens=tokens, turn=policy.calls, refresh=True)
            first = None
            for i in range(cfg['k']):
                seed = spec['seed'] + i*1000
                policy = BehaviorPolicy(engine, tok, lora, cfg['sampling'], seed, action)
                e = run_continuation(task, spec['scenario'], policy, cfg['sampling'], before['version'], seed, i, cfg['tool_cost'], context_builder(cfg))
                jsonl(partial, e)
                replay(task, e, tok, context_builder(cfg))
                episodes.append(e)
                if first is None:
                    first = policy.first
                bar.set_postfix(success=sum(x['success'] for x in episodes), tokens=tokens, refresh=False)
                bar.update(1)
        proof = engine.collective_rpc('grpo_verify_lora_tensors', args=(str(adapter.resolve()), before['adapter_sha256']), timeout=60)
        probe = probability_probe(engine, tok, lora, first, cfg, seed=spec['seed'],worker_proof=proof)
        atomic_json(output/'groups'/f'probability-probe-{index:04}-{time.time_ns()}.json', probe)
        if not probe['passed']:
            raise ValueError('temperature-one processed-q probe exceeded tolerance')
        if file_hash(adapter/'adapter_model.safetensors') != before['adapter_sha256']:
            raise ValueError('immutable sampling adapter changed')
        group = {'spec': spec, 'policy': before, 'config_sha256': file_hash(output/'config.json'),
                 'episodes': episodes, 'credit': compute_credit(episodes, cfg['k'], cfg['epsilon'], cfg.get('credit_method','outcome')),
                 'worker_proof': proof, 'probability_probe': probe,
                 'elapsed_seconds': time.perf_counter()-started, 'response_tokens': tokens,
                 'actions': sum(len(e['turns']) for e in episodes),
                 'max_context_tokens': max(len(r['generation']['prompt_token_ids']) for e in episodes for r in e['turns']),
                 'pid': os.getpid(), 'fresh_engine': True, 'adapter_path': str(adapter.resolve()),
                 'sampling_policy': 'one_adapter_one_fresh_process_constant_lora_id_1'}
        atomic_json(target, group)
        verify_group(output, index, before, tok)
    except Exception as exc:
        evidence=getattr(engine,'_grpo_first_probability_evidence',None) if engine is not None else None
        if evidence is not None:atomic_json(output/'groups'/f'first-probability-evidence-{index:04}-{time.time_ns()}.json',evidence)
        atomic_json(output/'groups'/f'interrupted-{index:04}-{time.time_ns()}.json',
                    getattr(exc, 'episode_evidence', {'status': 'interrupted', 'error': str(exc),
                                                     'training_admitted': False, 'partial_file': partial.name}))
        raise
    finally:
        if engine is not None:
            shutdown(engine)


def tensor_digest(tensors):
    import torch
    return digest({n: {'shape': list(p.shape), 'dtype': str(p.dtype),
                       'sha': __import__('hashlib').sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                   for n, p in sorted(tensors.items())})


def save_policy_adapter(model, adapter, adapter_name='policy'):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    adapter = Path(adapter)
    adapter.mkdir(parents=True, exist_ok=True)
    weights = {n: p.detach().cpu().contiguous() for n, p in
               get_peft_model_state_dict(model, adapter_name=adapter_name).items()}
    save_file(weights, adapter/'adapter_model.safetensors', metadata={'format': 'pt'})
    adapter_cfg = copy.deepcopy(model.peft_config[adapter_name])
    adapter_cfg.inference_mode = True
    adapter_cfg.save_pretrained(adapter)
    # Normalize the module order used by the isolated vLLM protocol.
    config = read(adapter/'adapter_config.json')
    config['target_modules'] = sorted(config['target_modules'])
    atomic_json(adapter/'adapter_config.json', config)
    return file_hash(adapter/'adapter_model.safetensors')


def rng_state(torch):
    return {'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(torch, state):
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def configure_training_memory(model, cfg):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    layers = 0
    if cfg.get('explicit_eval_checkpointing', False):
        from apt_v2.grpo_optimizer import enable_eval_checkpointing
        layers = enable_eval_checkpointing(model)
    model.eval()
    return {'model_eval': not model.training, 'explicit_eval_checkpoint_layers': layers}


def update(output, index, before):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import save_file
    from tqdm.auto import tqdm
    from apt_v2.grpo_optimizer import prepare_group_scores, update_group, response_scores
    from grpo_common import load_transition
    cfg = verify_inputs(output)
    transition_path = output/'transitions'/f'group-{index:04}.json'
    if transition_path.exists():
        load_transition(output, index, before)
        return
    g = verify_group(output, index, before)
    eligible = g['credit']['eligible']
    root = output/'checkpoints'/f'group-{index:04}'
    # A checkpoint is committed before its transition. Resume can adopt it exactly.
    if root.exists():
        complete = read(root/'complete.json')
        verify_files(root, complete['files'])
        if (complete['group_sha256'] != file_hash(output/'groups'/f'group-{index:04}.json')
                or complete['config_sha256'] != file_hash(output/'config.json') or complete['before'] != before):
            raise ValueError('orphan checkpoint binding differs')
        result = read(root/'transition.json')
        atomic_json(transition_path, result)
        return
    result = {'index': index, 'before': before, 'after': dict(before), 'updated': False,
              'group_sha256': file_hash(output/'groups'/f'group-{index:04}.json'),
              'config_sha256': file_hash(output/'config.json')}
    if not eligible:
        result['metrics'] = {'updated': False, 'skip_reason': 'zero_variance', 'credit': g['credit']}
        atomic_json(transition_path, result)
        return
    if before['updates'] >= cfg['max_updates']:
        raise ValueError('optimizer update cap reached')
    folder = under(ROOT, cfg['base_dir'])
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    for e in g['episodes']:
        replay(task_for(e['task_id'], cfg), e, tok, context_builder(cfg))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    base = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32, attn_implementation='sdpa').cuda()
    model = PeftModel.from_pretrained(base, under(output, before['checkpoint'])/'adapter',
                                     adapter_name='policy', is_trainable=True).eval()
    model.load_adapter(output/'initial/adapter', adapter_name='reference', is_trainable=False)
    def activate(name):
        model.set_adapter(name)
        for n, p in model.named_parameters():
            p.requires_grad_(name == 'policy' and 'lora_' in n and '.policy.' in n)
    activate('policy')
    memory_contract = configure_training_memory(model, cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 1089536:
        raise ValueError('1089536 trainable LoRA parameters required')
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    reference_before = tensor_digest(get_peft_model_state_dict(model, adapter_name='reference'))
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0)
    resumed = False
    if before['updates']:
        parent = under(output, before['checkpoint'])
        meta = read(parent/'complete.json')
        if file_hash(parent/'state.pt') != meta['state_sha256']:
            raise ValueError('resume Adam/RNG state hash differs')
        state = torch.load(parent/'state.pt', map_location='cpu', weights_only=False)
        if (state['config_sha256'] != file_hash(output/'config.json') or state['actual_updates'] != before['updates']
                or state['policy_version'] != before['version'] or state['group_cursor'] > index
                or state['adapter_sha256'] != before['adapter_sha256']):
            raise ValueError('resume state/version/cursor differs')
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(torch, state)
        resumed = True
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    actions = g['actions']
    with tqdm(total=2*actions, desc=f'Score/update group {index:04}', unit='action',
              file=sys.stdout, disable=False, dynamic_ncols=True) as bar:
        def progress(stage):
            bar.set_postfix(stage=stage, refresh=False)
            bar.update(1)
        def reference_score(prompt, response):
            activate('reference')
            try:
                return response_scores(model, prompt, response)
            finally:
                activate('policy')
        cache = prepare_group_scores(model, reference_score, g['episodes'], before['adapter_sha256'],
                                     cfg['adapter_sha256'], progress, release_cuda_cache=cfg.get('release_cuda_cache_between_actions', False))
        metrics = update_group(model, optimizer, g['episodes'], cache, before['adapter_sha256'],
                               cfg['adapter_sha256'], k=cfg['k'], epsilon=cfg['epsilon'], beta=cfg['beta'],
                               clip=cfg['clip'], tis_cap=cfg['tis_cap'], max_grad_norm=cfg['max_grad_norm'],
                               max_k3=cfg['max_sampled_k3'], max_clip_fraction=cfg['max_clip_fraction'], progress=progress,
                               release_cuda_cache=cfg.get('release_cuda_cache_between_actions', False),
                               credit_method=cfg.get('credit_method','outcome'))
    if reference_before != tensor_digest(get_peft_model_state_dict(model, adapter_name='reference')):
        raise ValueError('frozen reference tensors changed')
    metrics.update(training_memory_contract=memory_contract,
                   release_cuda_cache_between_actions=cfg.get('release_cuda_cache_between_actions', False),
                   elapsed_seconds=time.perf_counter()-started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   reference_tensor_sha256=reference_before, reference_unchanged=True,
                   resumed_adam_rng=resumed, same_path_max_error=cache['same_path_max_error'],
                   frozen_parameter_gradients=sum(p.grad is not None for p in model.parameters() if not p.requires_grad))
    current = {n: p.detach().cpu() for n, p in model.named_parameters() if n in initial}
    delta2 = sum((current[n].double()-p.double()).square().sum().item() for n, p in initial.items())
    metrics['adapter_delta_l2'] = delta2**.5
    metrics['changed_tensors'] = sum(not torch.equal(current[n], p) for n, p in initial.items())
    result['metrics'] = metrics
    if not metrics['updated']:
        if delta2:
            raise ValueError('skipped optimizer changed parameters')
        atomic_json(transition_path, result)
        return
    if not delta2 or not metrics['changed_tensors']:
        raise ValueError('actual LoRA tensor update required')
    # Stage a self-contained checkpoint; directory rename is the commit boundary.
    temp = output/'checkpoints'/f'.group-{index:04}-{time.time_ns()}'
    adapter = temp/'adapter'
    new_sha = save_policy_adapter(model, adapter)
    after = {'checkpoint': root.relative_to(output).as_posix(), 'version': before['version']+1,
             'updates': before['updates']+1, 'adapter_sha256': new_sha}
    result.update(after=after, updated=True)
    state = {**rng_state(torch), 'optimizer': optimizer.state_dict(), 'actual_updates': after['updates'],
             'policy_version': after['version'], 'adapter_sha256': new_sha, 'group_cursor': index+1,
             'config_sha256': file_hash(output/'config.json'), 'sampling_version': before['version'],
             'next_group_spec': planned_spec(cfg, index+1)}
    torch.save(state, temp/'state.pt')
    loaded = torch.load(temp/'state.pt', map_location='cpu', weights_only=False)
    check_optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0)
    check_optimizer.load_state_dict(loaded['optimizer'])
    steps = [int(v['step']) for v in check_optimizer.state.values()]
    if not steps or any(s != after['updates'] for s in steps):
        raise ValueError('Adam actual update counter differs')
    metrics['optimizer_state_count'] = len(steps)
    metrics['optimizer_steps'] = sorted(set(steps))
    metrics['optimizer_roundtrip_verified'] = True
    restored_rng = loaded['torch_rng']
    if not torch.equal(restored_rng, state['torch_rng']) or loaded['python_rng'] != state['python_rng'] or any(
            not torch.equal(a, b) for a, b in zip(loaded['cuda_rng'], state['cuda_rng'])):
        raise ValueError('saved RNG roundtrip differs')
    metrics['rng_roundtrip_verified'] = True
    atomic_json(temp/'scores.json', cache)
    atomic_json(temp/'transition.json', result)
    atomic_json(temp/'complete.json', {'files': checkpoint_binding(temp), 'state_sha256': file_hash(temp/'state.pt'),
                'policy_version': after['version'], 'actual_updates': after['updates'], 'before': before,
                'group_cursor': index+1, 'group_sha256': result['group_sha256'],
                'config_sha256': result['config_sha256'], 'total_sft_updates': cfg['source_sft_updates'], 'total_dpo_updates': cfg['source_dpo_updates'],
                'total_rl_updates': after['updates']})
    root.parent.mkdir(parents=True, exist_ok=True)
    temp.rename(root)
    atomic_json(transition_path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'collect', 'update'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group-index', type=int)
    p.add_argument('--policy-file', type=Path)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.output)
    else:
        if args.group_index is None or args.policy_file is None:
            p.error('group index and policy file required')
        before = read(args.policy_file)
        {'collect': collect, 'update': update}[args.stage](args.output, args.group_index, before)


if __name__ == '__main__':
    main()
