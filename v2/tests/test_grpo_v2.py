import copy
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from apt_v2.tasks import make_tasks
from apt_v2.runtime import ScriptedPolicy, Generation, run_episode
from apt_v2.grpo import (run_continuation, replay, group_advantages, start_spec,
                         validate_native, initial_state)
from grpo_common import read, validate_config, atomic_json

CFG = read(ROOT/'configs/grpo-outcome-preflight-v1.json')
SAMPLING = CFG['sampling']
TASK = make_tasks()[0]


def episode(success=True, seed=100, index=0, scenario=None, extra=False):
    # Real SQLite/environment trajectories. Token fixtures are scored independently
    # by the tiny causal model in the optimizer tests below.
    expert = ScriptedPolicy(TASK)
    if scenario is None:
        def policy(messages):
            if not success:
                return Generation(json.dumps({'tool': 'missing', 'arguments': {}}))
            if extra and len(messages) == 2:
                return Generation(json.dumps({'tool': 'read_policy', 'arguments': {}}))
            # ScriptedPolicy's state machine expects its own observation sequence.
            adjusted = messages[:2] + messages[4:] if extra else messages
            return expert(adjusted)
    else:
        from apt_v2.dpo import corrective_action
        env, _, _, _ = initial_state(TASK, scenario)
        def policy(messages):
            action = corrective_action(env) if success else {'tool': 'missing', 'arguments': {}}
            env.step(action)
            return Generation(json.dumps(action))
    try:
        e = run_continuation(TASK, scenario, policy, SAMPLING, 0, seed, index)
    finally:
        if scenario:
            env.close()
    for row in e['turns']:
        token = 4 if success else 5
        row['generation'].update(mode='sample', prompt_token_ids=[1, 2, 3], response_token_ids=[token, 2],
                                 behavior_token_log_probs=[-2., -2.], raw_token_log_probs=None, finish_reason='stop')
    return e


def group():
    return [episode(i % 2 == 0, seed=100+i*1000, index=i) for i in range(4)]


def test_group_population_std_and_trajectory_credit():
    es = group()
    c = group_advantages(es)
    assert c['returns'] == pytest.approx([.93, -.12, .93, -.12])
    assert c['mean'] == pytest.approx(.405)
    assert c['population_std'] == pytest.approx(.525)
    assert c['advantages'] == pytest.approx([1., -1., 1., -1.], abs=3e-8)
    assert sum(c['advantages']) == pytest.approx(0.)
    assert c['eligible']


@pytest.mark.parametrize('success', [True, False])
def test_exact_zero_variance_has_no_late_action_credit(success):
    es = [episode(success, seed=100+i*1000, index=i) for i in range(4)]
    c = group_advantages(es)
    assert c['zero_variance'] and not c['eligible']
    assert c['advantages'] == [0.]*4


def test_all_success_different_lengths_are_not_zero_variance():
    es = [episode(True, seed=100+i*1000, index=i, extra=bool(i%2)) for i in range(4)]
    c = group_advantages(es)
    assert all(e['success'] for e in es)
    assert not c['zero_variance']
    assert c['advantages'][0] > 0 > c['advantages'][1]


@pytest.mark.parametrize('scenario', CFG['recovery_scenarios'])
def test_train_recovery_prefix_replay_and_remaining_budget(scenario):
    e = episode(True, scenario=scenario)
    assert e['success']
    assert e['prefix_turns'] > 0
    assert e['turns'][0]['turn'] == e['prefix_turns']
    assert e['total_turns'] == e['prefix_turns'] + len(e['turns']) <= 12
    assert replay(TASK, e) == len(e['turns'])
    bad = copy.deepcopy(e)
    bad['turns'][0]['reward'] += .1
    with pytest.raises(ValueError, match='replay'):
        replay(TASK, bad)


@pytest.mark.parametrize('mutation', ['task', 'state', 'version', 'seed', 'sampling', 'split', 'incomplete', 'q', 'return'])
def test_group_rejects_mixed_or_unqualified_rollouts(mutation):
    es = group()
    e = es[-1]
    if mutation == 'task': e['task_id'] = 'train-001'
    if mutation == 'state': e['start_id'] = 'wrong'
    if mutation == 'version': e['policy_version'] += 1
    if mutation == 'seed': e['seed'] = es[0]['seed']
    if mutation == 'sampling': e['sampling'] = {**SAMPLING, 'temperature': .8}
    if mutation == 'split': e['split'] = 'development'
    if mutation == 'incomplete': e['turns'][-1]['done'] = False
    if mutation == 'q': e['turns'][0]['generation']['behavior_token_log_probs'] = None
    if mutation == 'return': e['return'] += .01
    with pytest.raises(ValueError): group_advantages(es)


def test_interruption_saves_partial_evidence_and_excludes_group():
    calls = []
    def policy(messages):
        if calls:
            raise RuntimeError('inference service interrupted')
        calls.append(1)
        return Generation(json.dumps({'tool': 'read_policy', 'arguments': {}}))
    with pytest.raises(RuntimeError) as exc:
        run_continuation(TASK, None, policy, SAMPLING, 0, 100, 0)
    evidence = exc.value.episode_evidence
    assert evidence['status'] == 'interrupted' and not evidence['training_admitted']
    assert len(evidence['turns']) == 1


class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, *args, **kwargs): return [1, 3]
    def decode(self, ids, **kwargs): return 'action'


def native(ids, finish):
    return {'mode': 'sample', 'prompt_token_ids': [1, 3], 'response_token_ids': ids,
            'text': 'action', 'raw_token_log_probs': None, 'behavior_token_log_probs': [-1.]*len(ids),
            'finish_reason': finish}


def test_native_eos_and_truncated_action_masks_are_real_tokens():
    s = {**SAMPLING, 'max_new_tokens': 3}
    validate_native(Tokenizer(), [], native([4, 2], 'stop'), s)
    validate_native(Tokenizer(), [], native([4, 5, 6], 'length'), s)
    for ids, finish in [([4], 'stop'), ([2, 4, 2], 'stop'), ([4, 5], 'length'), ([4], 'abort')]:
        with pytest.raises(ValueError): validate_native(Tokenizer(), [], native(ids, finish), s)


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import get_peft_model, LoraConfig
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.,
                         _attn_implementation='eager')
    return get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
                          lora_dropout=0, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()


def scored_group(model):
    from apt_v2.grpo_optimizer import response_scores
    es = group()
    for e in es:
        for row in e['turns']:
            g = row['generation']
            g['behavior_token_log_probs'] = response_scores(model, g['prompt_token_ids'], g['response_token_ids']).tolist()
    return es


def test_suffix_scores_match_full_logits_and_prompt_has_no_loss():
    import torch
    from apt_v2.grpo_optimizer import response_scores
    model = tiny_model()
    prompt, response = [1, 2, 3, 4, 5], [6, 7, 2]
    logits = model(input_ids=torch.tensor([prompt+response])).logits
    expected = logits[0, len(prompt)-1:-1].float().log_softmax(-1).gather(-1, torch.tensor(response).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(response_scores(model, prompt, response), expected)
    assert response_scores(model, prompt, response).numel() == len(response)


def test_actual_lora_update_frozen_reference_and_cache_binding():
    import torch
    from apt_v2.grpo_optimizer import response_scores, prepare_group_scores, update_group
    model = tiny_model()
    ref = copy.deepcopy(model).requires_grad_(False)
    es = scored_group(model)
    cache = prepare_group_scores(model, lambda p, r: response_scores(ref, p, r), es, 'policy', 'ref')
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    reference = {n: p.detach().clone() for n, p in ref.named_parameters()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0)
    result = update_group(model, optimizer, es, cache, 'policy', 'ref')
    assert result['updated'] and result['grad_norm'] > 0
    assert any(not torch.equal(a, b) for a, b in zip(initial, parameters))
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    assert all(torch.equal(p, reference[n]) and p.grad is None for n, p in ref.named_parameters())
    bad = copy.deepcopy(cache)
    bad['entries'][-1]['q'][0] -= 1
    with pytest.raises(ValueError, match='identity'):
        update_group(model, optimizer, es, bad, 'policy', 'ref')
    with pytest.raises(ValueError, match='identity'):
        update_group(model, optimizer, es, cache, 'different_checkpoint', 'ref')


def test_zero_variance_and_kl_guard_do_not_advance_adam():
    import torch
    from apt_v2.grpo_optimizer import response_scores, prepare_group_scores, update_group
    model = tiny_model()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    es = [episode(False, seed=100+i*1000, index=i) for i in range(4)]
    result = update_group(model, optimizer, es, {}, 'p', 'r')
    assert not result['updated'] and not optimizer.state
    es = scored_group(model)
    cache = prepare_group_scores(model, lambda p, r: response_scores(model, p, r)+1., es, 'p', 'r')
    result = update_group(model, optimizer, es, cache, 'p', 'r', max_k3=.01)
    assert not result['updated'] and result['skip_reason'] == 'kl_or_clip_guard' and not optimizer.state
    assert all(p.grad is None for p in model.parameters())


def test_episode_normalization_matches_manual_gradient():
    import torch
    from apt_v2.grpo_optimizer import action_objective
    x = torch.tensor([-.3, -.6], requires_grad=True)
    old = x.detach().clone()
    q = old.clone()
    # A 1-action trajectory and a 3-action trajectory each carry half the mass.
    a = action_objective(x, old, old, q, 1., beta=0)[0]/2
    a.backward()
    grad_one = x.grad.clone()
    x.grad = None
    for _ in range(3):
        (action_objective(x, old, old, q, 1., beta=0)[0]/6).backward()
    torch.testing.assert_close(x.grad, grad_one)


def test_adam_and_rng_resume_reproduces_next_real_update(tmp_path):
    import torch
    from apt_v2.grpo_optimizer import response_scores, prepare_group_scores, update_group
    from grpo_worker import rng_state, restore_rng
    model = tiny_model()
    ref = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    def step(m, opt):
        es = scored_group(m)
        cache = prepare_group_scores(m, lambda p, r: response_scores(ref, p, r), es, 'p', 'r')
        return update_group(m, opt, es, cache, 'p', 'r')
    assert step(model, optimizer)['updated']
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), **rng_state(torch)}, tmp_path/'state.pt')
    loaded = torch.load(tmp_path/'state.pt', weights_only=False)
    baseline_random = (random.random(), torch.rand(3))
    assert step(model, optimizer)['updated']
    resumed = tiny_model()
    resumed.load_state_dict(loaded['model'])
    opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    opt.load_state_dict(loaded['optimizer'])
    restore_rng(torch, loaded)
    assert random.random() == baseline_random[0]
    torch.testing.assert_close(torch.rand(3), baseline_random[1], rtol=0, atol=0)
    assert step(resumed, opt)['updated']
    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert {int(s['step']) for s in opt.state.values()} == {2}


def test_shared_base_reference_adapter_stays_frozen():
    import torch
    from peft import LoraConfig
    from apt_v2.grpo_optimizer import response_scores, prepare_group_scores, update_group
    model = tiny_model()
    model.add_adapter('reference', LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    def activate(name):
        model.set_adapter(name)
        for n, p in model.named_parameters(): p.requires_grad_(name == 'default' and '.default.' in n and 'lora_' in n)
    activate('default')
    ref = {n: p.detach().clone() for n, p in model.named_parameters() if '.reference.' in n}
    es = scored_group(model)
    def reference_score(p, r):
        activate('reference')
        try: return response_scores(model, p, r)
        finally: activate('default')
    cache = prepare_group_scores(model, reference_score, es, 'p', 'r')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    assert update_group(model, optimizer, es, cache, 'p', 'r')['updated']
    assert all(torch.equal(p, ref[n]) and not p.requires_grad and p.grad is None for n, p in model.named_parameters() if n in ref)


def test_schedule_is_train_only_balanced_and_seeds_disjoint():
    specs = [start_spec(i, CFG['seed'], CFG['recovery_scenarios']) for i in range(24)]
    assert sum(s['kind'] == 'complete' for s in specs) == 12
    assert sum(s['kind'] == 'recovery' for s in specs) == 12
    assert all(s['task_id'].startswith('train-') for s in specs)
    seeds = [s['seed'] + i*1000+j for s in specs for i in range(4) for j in range(12)]
    assert len(set(seeds)) == len(seeds)
    validate_config(CFG)
    bad = copy.deepcopy(CFG)
    bad['max_wall_seconds'] = 3601
    with pytest.raises(ValueError): validate_config(bad)


def test_failure_zip_keeps_partial_progress_and_excludes_state(tmp_path):
    import zipfile
    from run_v2_grpo import package_review
    output = tmp_path/'run'
    (output/'groups').mkdir(parents=True)
    atomic_json(output/'groups/partial.json', {'status': 'interrupted', 'training_admitted': False})
    (output/'state.pt').write_bytes(b'large optimizer kept on cloud')
    (output/'progress.log').write_text('Training: 50%\r', encoding='utf8')
    package_review(output, tmp_path/'review.zip')
    with zipfile.ZipFile(tmp_path/'review.zip') as z:
        assert 'run/groups/partial.json' in z.namelist()
        assert 'run/progress.log' in z.namelist()
        assert not any(n.endswith('.pt') for n in z.namelist())


def test_real_child_progress_is_streamed_to_terminal_and_log(tmp_path, capsys):
    import os
    from streamed_process import run_logged
    log = tmp_path/'stage.log'
    code = run_logged([sys.executable, '-u', '-c', "import sys; sys.stdout.write('Training: 40%\\rTraining: 100%\\n'); sys.stdout.flush()"],
                      cwd=tmp_path, env=dict(os.environ), log_path=log, timeout=20)
    assert code == 0
    assert 'Training: 40%' in log.read_text(encoding='utf8')
    assert 'Training: 100%' in capsys.readouterr().out


@pytest.mark.parametrize('fail_at', ['prepare', 'collect-0000', 'update-0000'])
def test_driver_failure_packages_status_and_progress(tmp_path, monkeypatch, fail_at):
    import zipfile
    import run_v2_grpo as driver
    from grpo_common import file_hash
    cfg = copy.deepcopy(CFG)
    cfg['run_id'] = 'test-grpo-failure'
    config = tmp_path/'configs/preflight.json'
    atomic_json(config, cfg)
    atomic_json(tmp_path/'grpo_release_manifest.json', {'files': {'configs/preflight.json': file_hash(config)}})
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'source_files', lambda: {'configs/preflight.json': file_hash(config)})
    monkeypatch.setattr(driver, 'resource_snapshot', lambda *a: None)
    called = []
    def stage(command, **kwargs):
        name = kwargs['log_path'].name.split('-attempt')[0]
        called.append(name)
        kwargs['log_path'].write_text('Progress: 50%\r', encoding='utf8')
        return 1 if name == fail_at else 0
    monkeypatch.setattr(driver, 'run_logged', stage)
    with pytest.raises(RuntimeError): driver.run(config)
    with zipfile.ZipFile(tmp_path/(cfg['run_id']+'-review.zip')) as z:
        status = json.loads(z.read(cfg['run_id']+'/run_status.json'))
        assert status['error'] and status['actual_updates'] == 0
        assert any(n.endswith('.log') for n in z.namelist())
    assert called[-1] == fail_at
    assert read(tmp_path/'artifacts'/cfg['run_id']/'runner_state.json')['next_group'] == 0


def test_driver_resume_keeps_committed_groups_and_counter(tmp_path, monkeypatch):
    import run_v2_grpo as driver
    from grpo_common import file_hash
    cfg = copy.deepcopy(CFG)
    cfg.update(run_id='test-grpo-resume', max_groups=2, max_updates=1)
    config = tmp_path/'configs/preflight.json'
    atomic_json(config, cfg)
    atomic_json(tmp_path/'grpo_release_manifest.json', {'files': {'configs/preflight.json': file_hash(config)}})
    output = tmp_path/'artifacts'/cfg['run_id']
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'source_files', lambda: {'configs/preflight.json': file_hash(config)})
    monkeypatch.setattr(driver, 'resource_snapshot', lambda *a: None)
    fail = [True]
    calls = []
    def stage(command, **kwargs):
        name = kwargs['log_path'].name.split('-attempt')[0]
        calls.append(name)
        kwargs['log_path'].write_text('Progress: 100%\r', encoding='utf8')
        if name == 'prepare':
            atomic_json(output/'input_binding.json', {})
        elif name.startswith('collect-'):
            i = int(name[-4:])
            if i == 1 and fail[0]:
                fail[0] = False
                return 1
            atomic_json(output/'groups'/f'group-{i:04}.json', {'index': i})
        elif name.startswith('update-'):
            i = int(name[-4:])
            before = read(output/'policies'/f'before-group-{i:04}.json')
            atomic_json(output/'transitions'/f'group-{i:04}.json',
                        {'index': i, 'before': before, 'after': before, 'updated': False,
                         'config_sha256': file_hash(output/'config.json'),
                         'group_sha256': file_hash(output/'groups'/f'group-{i:04}.json'),
                         'metrics': {'skip_reason': 'zero_variance'}})
        return 0
    monkeypatch.setattr(driver, 'run_logged', stage)
    with pytest.raises(RuntimeError): driver.run(config)
    assert read(output/'runner_state.json')['next_group'] == 1
    archive = driver.run(config, resume=True)
    assert archive.name.endswith('attempt-002.zip')
    assert calls.count('collect-0000') == calls.count('update-0000') == 1
    state = read(output/'runner_state.json')
    assert state['next_group'] == 2 and state['policy']['updates'] == 0


def test_production_adapter_export_reloads_exact_policy(tmp_path):
    import torch
    from peft import PeftModel, LoraConfig
    from apt_v2.grpo_optimizer import response_scores
    from grpo_worker import save_policy_adapter
    model = tiny_model()
    model.add_adapter('policy', LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                         target_modules=['v_proj', 'q_proj'], task_type='CAUSAL_LM'))
    model.set_adapter('policy')
    with torch.no_grad():
        for n, p in model.named_parameters():
            if '.policy.' in n: p.add_(.01)
    expected = response_scores(model, [1, 3, 4], [5, 2])
    assert save_policy_adapter(model, tmp_path/'adapter')
    config = read(tmp_path/'adapter/adapter_config.json')
    assert config['target_modules'] == ['q_proj', 'v_proj'] and config['inference_mode']
    # A clean base with the same initialization; load only the exported policy.
    base = tiny_model().unload()
    loaded = PeftModel.from_pretrained(base, tmp_path/'adapter', is_trainable=True).eval()
    torch.testing.assert_close(response_scores(loaded, [1, 3, 4], [5, 2]), expected, rtol=0, atol=0)


def test_worker_proof_detects_stale_registered_adapter(tmp_path, monkeypatch):
    import torch
    import types
    from grpo_vllm import worker_tensor_proof
    from grpo_common import file_hash
    (tmp_path/'adapter_model.safetensors').write_bytes(b'checkpoint')
    fake_request = types.ModuleType('vllm.lora.request')
    fake_request.LoRARequest = lambda *a: a
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', fake_request)
    expected = SimpleNamespace(loras={'layer': SimpleNamespace(lora_a=torch.ones(2, 3), lora_b=torch.ones(3, 2))})
    actual = copy.deepcopy(expected)
    registry = SimpleNamespace(list_adapters=lambda: {1: actual}, lora_index_to_id=[1],
                               _create_merged_loras_inplace=lambda adapter: None)
    manager = SimpleNamespace(_adapter_manager=registry, _load_adapter=lambda request: copy.deepcopy(expected))
    worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager))
    proof = worker_tensor_proof(worker, str(tmp_path), file_hash(tmp_path/'adapter_model.safetensors'))
    assert proof['passed'] and proof['tensor_count'] == 2
    actual.loras['layer'].lora_b.add_(.01)
    with pytest.raises(ValueError, match='tensors differ'):
        worker_tensor_proof(worker, str(tmp_path), file_hash(tmp_path/'adapter_model.safetensors'))
