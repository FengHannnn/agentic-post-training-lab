"""Single-adapter inference process with processed q and worker-side tensor proof."""
import hashlib
import math
from pathlib import Path
from grpo_common import file_hash
from apt_v2.runtime import Generation


def tensor_tree(adapter):
    import torch
    result = {}
    def visit(name, value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for i, tensor in enumerate(value):
                visit(name+'/'+str(i), tensor)
        elif torch.is_tensor(value):
            t = value.detach().cpu().contiguous()
            result[name] = {'shape': list(t.shape), 'dtype': str(t.dtype),
                            'sha256': hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
        else:
            raise ValueError('unexpected vLLM LoRA tensor container')
    for name, layer in sorted(adapter.loras.items()):
        for field in ('lora_a', 'lora_b'):
            visit(name+'/'+field, getattr(layer, field))
    return result


def worker_tensor_proof(worker, adapter_dir, expected_sha):
    # Pinned to vLLM 0.11.0 internals; fail with evidence if this interface changes.
    from vllm.lora.request import LoRARequest
    from apt_v2.grpo import digest
    manager = worker.model_runner.lora_manager
    registry = manager._adapter_manager
    if file_hash(Path(adapter_dir)/'adapter_model.safetensors') != expected_sha:
        raise ValueError('worker adapter source hash differs')
    loaded = registry.list_adapters()
    if set(loaded) != {1} or 1 not in registry.lora_index_to_id:
        raise ValueError('one loaded active LoRA ID=1 required')
    expected = manager._load_adapter(LoRARequest('fixed_adapter', 1, adapter_dir))
    registry._create_merged_loras_inplace(expected)
    actual_tree, expected_tree = tensor_tree(loaded[1]), tensor_tree(expected)
    if not actual_tree or actual_tree != expected_tree:
        raise ValueError('worker-loaded LoRA tensors differ from immutable checkpoint')
    proof={'passed': True, 'adapter_sha256': expected_sha, 'registered_ids': [1],
            'active_slots': registry.lora_index_to_id, 'tensor_count': len(actual_tree),
            'loaded_tensor_sha256': digest(actual_tree), 'tensors': actual_tree}
    from grpo_sampler_probe import capture_snapshot
    capture=capture_snapshot(worker)
    if capture is not None:proof['same_forward_capture']=capture
    return proof


class GRPOWorkerExtension:
    """Importable worker methods; RPC transports only the method name and primitives."""
    def grpo_verify_lora_tensors(self, adapter_dir, expected_sha):
        return worker_tensor_proof(self, adapter_dir, expected_sha)

    def grpo_install_probability_capture(self, setup):
        from grpo_sampler_probe import install_capture
        return install_capture(self,setup)

    def grpo_probability_capture_snapshot(self):
        from grpo_sampler_probe import capture_snapshot
        return capture_snapshot(self)


def engine_and_tokenizer(cfg, adapter, capture_setup=None):
    from transformers import AutoTokenizer
    from vllm import LLM
    from grpo_common import ROOT, under
    s = cfg['sampling']
    folder = under(ROOT, cfg['base_dir'])
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    engine = LLM(model=str(folder), tokenizer=str(folder), dtype='bfloat16', trust_remote_code=False,
                 max_model_len=s['max_model_len'], max_num_seqs=1,
                 max_num_batched_tokens=s['max_model_len'],
                 gpu_memory_utilization=cfg['inference']['gpu_memory_utilization'], enforce_eager=True,
                 enable_prefix_caching=False, generation_config='vllm', logprobs_mode='processed_logprobs',
                 seed=cfg['seed'], enable_lora=True, max_lora_rank=8, max_loras=1,
                 lora_dtype=cfg['inference']['lora_dtype'],
                 worker_extension_cls='grpo_vllm.GRPOWorkerExtension')
    if capture_setup is not None:
        installed=engine.collective_rpc('grpo_install_probability_capture',args=(capture_setup,),timeout=60)
        if len(installed)!=1 or not installed[0]['installed']:
            raise ValueError('single worker same-forward capture installation failed')
        engine._grpo_sampler_capture_setup=capture_setup
        engine._grpo_first_probability_verified=False
    from vllm.lora.request import LoRARequest
    return engine, tok, LoRARequest('fixed_adapter', 1, str(adapter.resolve()))


class BehaviorPolicy:
    def __init__(self, engine, tok, lora, sampling, seed, on_action=None):
        self.engine, self.tok, self.lora = engine, tok, lora
        self.sampling, self.seed, self.calls = sampling, seed, 0
        self.on_action = on_action
        self.first = None

    def __call__(self, messages):
        prompt = list(self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        return self.generate_prompt(prompt)

    def generate_prompt(self, prompt):
        from vllm import SamplingParams
        s = self.sampling
        prompt = list(prompt)
        if len(prompt)+s['max_new_tokens'] > s['max_model_len']:
            raise ValueError('inference context budget exceeded; group interrupted')
        params = SamplingParams(n=1, temperature=s['temperature'], top_p=s['top_p'], top_k=s['top_k'],
                                min_p=s['min_p'], repetition_penalty=s['repetition_penalty'],
                                presence_penalty=s['presence_penalty'], frequency_penalty=s['frequency_penalty'],
                                min_tokens=s['min_tokens'], max_tokens=s['max_new_tokens'], logprobs=1,
                                stop_token_ids=[self.tok.eos_token_id], detokenize=False, seed=self.seed+self.calls)
        native = self.engine.generate([{'prompt_token_ids': prompt}], params, lora_request=self.lora, use_tqdm=False)[0]
        self.calls += 1
        if list(native.prompt_token_ids) != prompt or len(native.outputs) != 1:
            raise ValueError('native inference request binding differs')
        result = native.outputs[0]
        ids = list(result.token_ids)
        if not ids or result.logprobs is None or len(result.logprobs) != len(ids):
            raise ValueError('native processed probabilities required')
        q = [float(p[token].logprob) for token, p in zip(ids, result.logprobs)]
        if any(not math.isfinite(p) or p > 1e-6 for p in q):
            raise ValueError('finite processed probabilities required')
        stochastic = s['temperature'] > 0
        generation = Generation(self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                                'sample' if stochastic else 'greedy', prompt, ids, None,
                                q if stochastic else None, result.finish_reason)
        setup=getattr(self.engine,'_grpo_sampler_capture_setup',None)
        if setup is not None and not self.engine._grpo_first_probability_verified:
            from dataclasses import asdict
            from grpo_sampler_probe import verify_capture
            snapshots=self.engine.collective_rpc('grpo_probability_capture_snapshot',args=(),timeout=60)
            if len(snapshots)!=1:raise ValueError('one worker probability capture required')
            self.engine._grpo_first_probability_evidence={'capture':snapshots[0],'generation':asdict(generation),'seed':self.seed}
            verify_capture(snapshots[0],generation,self.sampling,self.seed,setup['policy_sha256'])
            self.engine._grpo_first_probability_verified=True
        if self.first is None:
            self.first = generation
        if self.on_action:
            self.on_action(len(ids))
        return generation


def probability_probe(engine, tok, lora, generation, cfg, seed=None, worker_proof=None):
    from vllm import SamplingParams
    # With no processors and T=1, processed decode q should match teacher-forced
    # raw prompt scores on the same native tokens, up to BF16 kernel differences.
    count = min(16, len(generation.response_token_ids))
    prefix = generation.prompt_token_ids
    tokens = generation.response_token_ids[:count]
    params = SamplingParams(temperature=1., top_p=1., top_k=-1, max_tokens=1,
                            prompt_logprobs=1, logprobs=1, detokenize=False, seed=cfg['seed']+999)
    result = engine.generate([{'prompt_token_ids': prefix+tokens}], params,
                             lora_request=lora, use_tqdm=False)[0]
    if list(result.prompt_token_ids) != prefix+tokens:
        raise ValueError('probability probe prompt binding differs')
    reference = [float(result.prompt_logprobs[len(prefix)+j][token].logprob) for j, token in enumerate(tokens)]
    q = generation.behavior_token_log_probs[:count]
    error = max(abs(a-b) for a, b in zip(q, reference))
    proof = {'temperature': 1, 'top_p': 1, 'top_k': -1, 'tokens': tokens, 'behavior_q': q,
             'teacher_forced_vllm_logp': reference, 'max_absolute_error': error,
             'gate': cfg['probability_probe_max_error'], 'passed': error <= cfg['probability_probe_max_error'],
             'extra_generated_tokens': len(result.outputs[0].token_ids)}
    if not all(math.isfinite(x) for x in reference):
        raise ValueError('nonfinite probability probe')
    if cfg.get('curriculum_schema') in ('outcome_curriculum_v1','fine_credit_ablation_v1','alias_curriculum_v1') and worker_proof is not None:
        from grpo_sampler_probe import verify_capture,METHOD
        from dataclasses import asdict
        if len(worker_proof)!=1 or 'same_forward_capture' not in worker_proof[0]:
            raise ValueError('same-forward worker capture missing')
        capture=worker_proof[0]['same_forward_capture']
        check=verify_capture(capture,asdict(generation),cfg['sampling'],seed,worker_proof[0]['adapter_sha256'])
        proof.update(verification_method=METHOD,same_forward=capture,same_forward_verification=check,
                     prefill_passed=proof['passed'],passed=check['passed'])
        return proof
    if (not proof['passed'] and cfg.get('curriculum_schema') == 'outcome_curriculum_v1'):
        if type(seed) is not int:
            raise ValueError('original request seed required for decode replay')
        # Replay the identical request through the same sampling/decode implementation.
        # Keep the prefill scores and original native behavior q unchanged.
        repeated = BehaviorPolicy(engine, tok, lora, cfg['sampling'], seed).generate_prompt(prefix)
        rq = repeated.behavior_token_log_probs
        original = generation.behavior_token_log_probs
        aligned = len(rq) == len(original)
        replay_error = max(abs(a-b) for a,b in zip(original,rq)) if aligned else None
        from dataclasses import asdict
        from apt_v2.grpo import digest
        replay = {'method':'same_request_decode_replay_v1','seed':seed,'sampling':dict(cfg['sampling']),
                  'prompt_token_ids':repeated.prompt_token_ids,'response_token_ids':repeated.response_token_ids,
                  'behavior_token_log_probs':rq,'finish_reason':repeated.finish_reason,
                  'original_generation_sha256':digest(asdict(generation)),
                  'max_absolute_error':replay_error,'gate':1e-5,
                  'passed':(repeated.response_token_ids == generation.response_token_ids
                            and repeated.finish_reason == generation.finish_reason
                            and aligned and replay_error <= 1e-5)}
        proof.update(verification_method='prefill_with_same_request_decode_replay_v1',
                     decode_replay=replay,prefill_passed=False,passed=replay['passed'],
                     extra_generated_tokens=proof['extra_generated_tokens']+len(repeated.response_token_ids))
    return proof


def verify_probability_probe(probe, generation, cfg, seed, worker_proof=None):
    from apt_v2.grpo import digest
    if not isinstance(generation,dict):
        from dataclasses import asdict
        generation=asdict(generation)
    count=min(16,len(generation['response_token_ids']))
    expected=generation['behavior_token_log_probs'][:count]
    reference=probe['teacher_forced_vllm_logp']
    if (probe['tokens']!=generation['response_token_ids'][:count] or probe['behavior_q']!=expected
            or len(reference)!=count or not all(math.isfinite(v) and v<=1e-6 for v in reference)
            or probe['gate']!=cfg['probability_probe_max_error']
            or (probe['temperature'],probe['top_p'],probe['top_k'])!=(1,1,-1)):
        raise ValueError('probability probe token/sampling identity differs')
    error=max(abs(a-b) for a,b in zip(expected,reference))
    if error!=probe['max_absolute_error']:
        raise ValueError('probability probe prefill recomputation differs')
    from grpo_sampler_probe import METHOD,verify_capture
    if probe.get('verification_method')==METHOD:
        if (worker_proof is None or len(worker_proof)!=1 or probe['same_forward']!=worker_proof[0].get('same_forward_capture')
                or probe['prefill_passed']!=(error<=probe['gate']) or not probe['passed'] or probe['extra_generated_tokens']!=1):
            raise ValueError('same-forward worker/probe binding differs')
        checked=verify_capture(probe['same_forward'],generation,cfg['sampling'],seed,worker_proof[0]['adapter_sha256'])
        if checked!=probe['same_forward_verification']:raise ValueError('same-forward summary differs')
        return True
    replay=probe.get('decode_replay')
    if replay is None:
        if not probe['passed'] or error>probe['gate']:
            raise ValueError('probability probe recomputation failed')
        return True
    sampling=cfg['sampling']
    if (cfg.get('curriculum_schema')!='outcome_curriculum_v1' or error<=probe['gate']
            or probe.get('verification_method')!='prefill_with_same_request_decode_replay_v1'
            or probe.get('prefill_passed') is not False
            or (sampling['temperature'],sampling['top_p'],sampling['top_k'],sampling['min_p'],
                sampling['min_tokens'],sampling['repetition_penalty'],sampling['presence_penalty'],sampling['frequency_penalty'])
               !=(1,1,-1,0,0,1,0,0)
            or sampling['logprobs_mode']!='processed_logprobs'
            or replay['method']!='same_request_decode_replay_v1' or replay['seed']!=seed
            or replay['sampling']!=sampling or replay['gate']!=1e-5
            or replay['original_generation_sha256']!=digest(generation)
            or replay['prompt_token_ids']!=generation['prompt_token_ids']
            or replay['response_token_ids']!=generation['response_token_ids']
            or replay['finish_reason']!=generation['finish_reason']):
        raise ValueError('same-request decode replay identity differs')
    q=generation['behavior_token_log_probs'];rq=replay['behavior_token_log_probs']
    if len(q)!=len(rq) or not all(math.isfinite(v) and v<=1e-6 for v in rq):
        raise ValueError('finite aligned decode replay probabilities required')
    actual=max(abs(a-b) for a,b in zip(q,rq))
    if (actual!=replay['max_absolute_error'] or actual>1e-5 or not replay['passed'] or not probe['passed']
            or probe['extra_generated_tokens']!=1+len(replay['response_token_ids'])):
        raise ValueError('same-request decode replay probability gate failed')
    return True


def shutdown(engine):
    engine.llm_engine.engine_core.shutdown()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def projected_checkpoint_tensors(adapter_dir):
    """Independent CPU projection for the fixed Qwen q/v LoRA BF16 worker layout."""
    import torch
    from safetensors.torch import load_file
    from grpo_common import read
    weights = load_file(str(Path(adapter_dir)/'adapter_model.safetensors'))
    cfg = read(Path(adapter_dir)/'adapter_config.json')
    if cfg['r'] != 8 or cfg['lora_alpha'] != 16 or sorted(cfg['target_modules']) != ['q_proj', 'v_proj']:
        raise ValueError('fixed Qwen q/v r8 alpha16 projection required')
    result = {}
    for name, tensor in weights.items():
        module, suffix = name.removeprefix('base_model.model.').split('.lora_')
        if not module.endswith(('.q_proj', '.v_proj')) or suffix not in ('A.weight', 'B.weight'):
            raise ValueError('fixed q/v adapter keys required')
        slot = 0 if module.endswith('.q_proj') else 2
        module = module.rsplit('.', 1)[0]+'.qkv_proj'
        value = tensor.to(torch.bfloat16)
        field = 'lora_a' if suffix == 'A.weight' else 'lora_b'
        if field == 'lora_b':
            value = value*(cfg['lora_alpha']/cfg['r'])
        result[f'{module}/{field}/{slot}'] = {
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    if len(result) != 112:
        raise ValueError('112 projected worker tensors required')
    return result
