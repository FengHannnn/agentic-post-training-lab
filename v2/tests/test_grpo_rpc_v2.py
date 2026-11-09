"""Regression at the real collector RPC expression, without a GPU dependency."""
import ast
import inspect
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def test_collector_rpc_request_uses_primitive_transport_payload(tmp_path):
    import grpo_worker
    import grpo_vllm
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(grpo_worker.collect)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'collective_rpc']
    assert len(calls) == 1
    class PrimitiveTransport:
        def collective_rpc(self, method, args=(), timeout=None):
            # Match the supported primitive payload boundary used with pickle fallback disabled.
            json.dumps({'method': method, 'args': args, 'timeout': timeout})
            assert isinstance(method, str)
            assert method == 'grpo_verify_lora_tensors'
            return [{'passed': True}]
    actual_call = ast.Expression(calls[0])
    ast.fix_missing_locations(actual_call)
    result = eval(compile(actual_call, '<actual collector RPC>', 'eval'),
                  {'engine': PrimitiveTransport(), 'worker_tensor_proof': grpo_vllm.worker_tensor_proof,
                   'adapter': tmp_path, 'before': {'adapter_sha256': 'bound-checkpoint'}})
    assert result == [{'passed': True}]


def test_engine_registers_importable_named_worker_extension(monkeypatch, tmp_path):
    import copy
    import types
    import transformers
    import grpo_vllm
    from grpo_common import read
    cfg = read(ROOT/'configs/grpo-outcome-preflight-v1.json')
    captured = {}
    class Engine:
        def __init__(self, **kwargs): captured.update(kwargs)
    vllm = types.ModuleType('vllm')
    vllm.LLM = Engine
    request = types.ModuleType('vllm.lora.request')
    request.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', request)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    grpo_vllm.engine_and_tokenizer(cfg, tmp_path)
    assert captured['worker_extension_cls'] == 'grpo_vllm.GRPOWorkerExtension'
    extension = getattr(grpo_vllm, 'GRPOWorkerExtension')
    assert callable(getattr(extension, 'grpo_verify_lora_tensors'))


def test_full_collector_commits_group_through_named_worker_rpc(tmp_path, monkeypatch):
    import copy
    import types
    import torch
    from types import SimpleNamespace
    import grpo_worker
    import grpo_vllm
    from grpo_common import read, atomic_json, file_hash
    from apt_v2.runtime import ScriptedPolicy, Generation
    cfg = read(ROOT/'configs/grpo-outcome-preflight-v1.json')
    adapter = tmp_path/'initial/adapter'
    adapter.mkdir(parents=True)
    (adapter/'adapter_model.safetensors').write_bytes(b'fixed-test-adapter')
    cfg['adapter_sha256'] = file_hash(adapter/'adapter_model.safetensors')
    atomic_json(tmp_path/'config.json', cfg)
    before = {'checkpoint': 'initial', 'version': 0, 'updates': 0, 'adapter_sha256': cfg['adapter_sha256']}
    monkeypatch.setattr(grpo_worker, 'verify_inputs', lambda *a, **k: cfg)
    class Tokenizer:
        eos_token_id = 2
        def __init__(self): self.keys = {}; self.texts = {}
        def apply_chat_template(self, messages, **kwargs):
            key = json.dumps(messages, sort_keys=True)
            if key not in self.keys: self.keys[key] = len(self.keys)+10
            return [1, self.keys[key]]
        def response(self, text):
            key = len(self.texts)+100
            self.texts[key] = text
            return [key, 2]
        def decode(self, ids, **kwargs): return self.texts[ids[0]]
    tok = Tokenizer()
    expected = SimpleNamespace(loras={f'layer-{i}': SimpleNamespace(lora_a=torch.ones(2, 3),
                                  lora_b=torch.ones(3, 2)) for i in range(56)})
    registry = SimpleNamespace(list_adapters=lambda: {1: expected}, lora_index_to_id=[1],
                               _create_merged_loras_inplace=lambda adapter: None)
    manager = SimpleNamespace(_adapter_manager=registry, _load_adapter=lambda request: copy.deepcopy(expected))
    worker = grpo_vllm.GRPOWorkerExtension()
    worker.model_runner = SimpleNamespace(lora_manager=manager)
    request = types.ModuleType('vllm.lora.request')
    request.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', request)
    calls = []
    class Engine:
        def collective_rpc(self, method, args=(), timeout=None):
            payload = json.loads(json.dumps({'method': method, 'args': args}))
            calls.append(payload['method'])
            return [getattr(worker, payload['method'])(*payload['args'])]
    monkeypatch.setattr(grpo_vllm, 'engine_and_tokenizer', lambda *a: (Engine(), tok, 'id1'))
    monkeypatch.setattr(grpo_vllm, 'shutdown', lambda engine: None)
    task = grpo_worker.task_for('train-000')
    class Policy:
        def __init__(self, engine, tokenizer, lora, sampling, seed, on_action):
            self.expert = ScriptedPolicy(task)
            self.calls = 0
            self.first = None
            self.on_action = on_action
        def __call__(self, messages):
            text = self.expert(messages).text
            ids = tok.response(text)
            g = Generation(text, 'sample', tok.apply_chat_template(messages), ids, None, [-.1, -.1], 'stop')
            self.calls += 1
            if self.first is None: self.first = g
            self.on_action(len(ids))
            return g
    monkeypatch.setattr(grpo_vllm, 'BehaviorPolicy', Policy)
    def probe(engine, tokenizer, lora, g, config, seed=None, worker_proof=None):
        return {'temperature': 1, 'top_p': 1, 'top_k': -1, 'tokens': g.response_token_ids,
                'behavior_q': g.behavior_token_log_probs, 'teacher_forced_vllm_logp': g.behavior_token_log_probs,
                'max_absolute_error': 0., 'gate': config['probability_probe_max_error'],
                'passed': True, 'extra_generated_tokens': 1}
    monkeypatch.setattr(grpo_vllm, 'probability_probe', probe)
    grpo_worker.collect(tmp_path, 0, before)
    group = read(tmp_path/'groups/group-0000.json')
    assert calls == ['grpo_verify_lora_tensors']
    assert len(group['episodes']) == 4 and all(e['success'] for e in group['episodes'])
    assert group['worker_proof'][0]['tensor_count'] == 112
    assert group['credit']['zero_variance'] and group['credit']['advantages'] == [0.]*4
    assert group['actions'] == 28


def test_repair_preflight_preserves_total_hour_and_original_identity():
    import copy
    import pytest
    from grpo_common import read, validate_config
    cfg = read(ROOT/'configs/grpo-outcome-preflight-v2.json')
    validate_config(cfg)
    original = read(ROOT/'configs/grpo-outcome-preflight-v1.json')
    assert cfg['adapter_sha256'] == original['adapter_sha256']
    assert cfg['sampling'] == original['sampling']
    assert cfg['max_groups'] == 8 and cfg['max_updates'] == 2
    spent = sum(r['elapsed_seconds'] for r in cfg['prior_preflight_runs'])
    assert spent + cfg['max_wall_seconds'] <= 3600
    bad = copy.deepcopy(cfg)
    bad['max_wall_seconds'] = 3600
    with pytest.raises(ValueError, match='combined preflight'):
        validate_config(bad)
