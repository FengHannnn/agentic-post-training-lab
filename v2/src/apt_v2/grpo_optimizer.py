"""Response-only, episode-balanced clipped GRPO objective and immutable score cache."""
import math
import torch
from .grpo import digest, group_advantages
from .rewards import training_samples
from .grpo_credit import compute_credit, action_advantage


def response_scores(model, prompt, response, grad=False):
    if not prompt or not response:
        raise ValueError('nonempty prompt and response required')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
    with torch.set_grad_enabled(grad):
        # Qwen2.5 and the local tiny-Llama test support suffix-only vocabulary logits.
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(response) + 1).logits
        positions = logits[0, :-1].float().log_softmax(-1)
        if len(positions) != len(response):
            raise ValueError('suffix causal response alignment differs')
        return positions.gather(-1, torch.tensor(response, device=device).unsqueeze(-1)).squeeze(-1)


def prepare_group_scores(model, reference_score, episodes, policy_sha, reference_sha, progress=None, release_cuda_cache=False):
    model.eval()
    entries = []
    same_path_max = 0.
    for i, episode in enumerate(episodes):
        for sample in training_samples(episode):
            args = sample['prompt_token_ids'], sample['response_token_ids']
            old = response_scores(model, *args)
            fresh = response_scores(model, *args, grad=True)
            error = (old - fresh.detach()).abs().max().item()
            same_path_max = max(same_path_max, error)
            if not torch.allclose(old, fresh.detach(), atol=1e-5, rtol=0):
                raise ValueError('same-path old/new gate failed')
            del fresh
            reference = reference_score(*args).detach()
            if old.shape != reference.shape or not torch.isfinite(old).all() or not torch.isfinite(reference).all():
                raise ValueError('finite aligned old/reference scores required')
            entries.append({'episode': i, 'turn': sample['turn'], 'prompt_token_ids': list(args[0]),
                            'response_token_ids': list(args[1]), 'old': old.cpu().tolist(),
                            'reference': reference.cpu().tolist(),
                            'q': sample['behavior_token_log_probs']})
            del old, reference
            if release_cuda_cache and next(model.parameters()).device.type == 'cuda':
                torch.cuda.empty_cache()
            if progress:
                progress('score')
    return {'group_sha256': digest(episodes), 'policy_sha256': policy_sha,
            'reference_sha256': reference_sha, 'same_path_max_error': same_path_max, 'entries': entries}


def action_objective(new, old, ref, q, advantage, beta=.02, clip=.2, tis_cap=2.):
    if (new.shape != old.shape or new.shape != ref.shape or new.shape != q.shape
            or any(t.requires_grad for t in (old, ref, q))):
        raise ValueError('aligned detached old/ref/q required')
    if not math.isfinite(advantage) or not all(torch.isfinite(t).all() for t in (new, old, ref, q)):
        raise ValueError('finite probabilities and advantage required')
    ratio = (new - old).exp()
    weights = (old - q).exp().clamp(max=tis_cap).detach()
    surrogate = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    delta = ref.double() - new.double()
    k3 = torch.expm1(delta) - delta
    loss = (weights * (surrogate + beta * k3.to(new.dtype))).mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite GRPO action objective')
    return loss, {'sampled_k3': k3.detach().mean().item(),
                  'clip_fraction': ((ratio.detach()-1).abs() > clip).float().mean().item(),
                  'tis_mean': weights.mean().item(), 'tis_cap_fraction': (weights >= tis_cap).float().mean().item(),
                  'old_minus_q_abs_mean': (old-q).abs().mean().item(),
                  'old_minus_q_abs_max': (old-q).abs().max().item()}


def update_group(model, optimizer, episodes, cache, policy_sha, reference_sha, k=4, epsilon=1e-8,
                 beta=.02, clip=.2, tis_cap=2., max_grad_norm=1., max_k3=.05,
                 max_clip_fraction=.5, progress=None, release_cuda_cache=False, credit_method='outcome'):
    if (not all(math.isfinite(v) for v in (beta, clip, tis_cap, max_grad_norm, max_k3, max_clip_fraction))
            or beta < 0 or not 0 < clip < 1 or tis_cap <= 0 or max_grad_norm <= 0
            or max_k3 < 0 or not 0 <= max_clip_fraction <= 1):
        raise ValueError('invalid GRPO optimizer settings')
    credit = compute_credit(episodes, k, epsilon, credit_method)
    if not credit['eligible']:
        optimizer.zero_grad(set_to_none=True)
        return {'updated': False, 'skip_reason': 'zero_variance', 'credit': credit, 'actions': 0}
    if (cache['group_sha256'] != digest(episodes) or cache['policy_sha256'] != policy_sha
            or cache['reference_sha256'] != reference_sha):
        raise ValueError('immutable score cache identity differs')
    parameters = [p for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError('LoRA-only trainable parameters required')
    expected = [(i, s) for i, e in enumerate(episodes) for s in training_samples(e)]
    if len(expected) != len(cache['entries']):
        raise ValueError('score cache action count differs')
    # Validate every entry before any backward pass.
    for (i, s), entry in zip(expected, cache['entries']):
        if (entry['episode'] != i or entry['turn'] != s['turn']
                or any(entry[key] != s[key] for key in ('prompt_token_ids', 'response_token_ids'))
                or entry['q'] != s['behavior_token_log_probs']):
            raise ValueError('cached action tokens or behavior identity differs')
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    metrics = {'loss': 0., 'actions': 0, 'response_tokens': 0, 'sampled_k3': 0., 'clip_fraction': 0.,
               'tis_mean': 0., 'tis_cap_fraction': 0., 'old_minus_q_abs_mean': 0., 'old_minus_q_abs_max': 0.}
    device = next(model.parameters()).device
    try:
        for (i, s), entry in zip(expected, cache['entries']):
            new = response_scores(model, s['prompt_token_ids'], s['response_token_ids'], True)
            old, ref, q = [torch.tensor(entry[key], device=device).detach() for key in ('old', 'reference', 'q')]
            loss, stats = action_objective(new, old, ref, q, action_advantage(credit, i, s['turn']-episodes[i]['prefix_turns']), beta, clip, tis_cap)
            weight = 1. / (k * len(episodes[i]['turns']))
            (loss * weight).backward()
            metrics['loss'] += loss.detach().item() * weight
            for key, value in stats.items():
                if key.endswith('_max'):
                    metrics[key] = max(metrics[key], value)
                else:
                    metrics[key] += value * weight
            metrics['actions'] += 1
            metrics['response_tokens'] += len(s['response_token_ids'])
            del new, old, ref, q, loss
            if release_cuda_cache and device.type == 'cuda':
                torch.cuda.empty_cache()
            if progress:
                progress('backward')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('frozen parameter received a gradient')
        if metrics['sampled_k3'] > max_k3 or metrics['clip_fraction'] > max_clip_fraction:
            optimizer.zero_grad(set_to_none=True)
            return {**metrics, 'credit': credit, 'updated': False, 'skip_reason': 'kl_or_clip_guard'}
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True).item()
        if norm <= 0:
            raise ValueError('nonzero actual LoRA gradient required')
        optimizer.step()
        return {**metrics, 'credit': credit, 'updated': True, 'grad_norm': norm, 'skip_reason': None}
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise


def enable_eval_checkpointing(model):
    """Checkpoint decoder forwards explicitly while retaining deterministic eval mode."""
    import functools
    import torch.utils.checkpoint as checkpoint_module
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    layers = base.model.layers
    count = 0
    for layer in layers:
        if not getattr(layer, '_grpo_eval_checkpoint_wrapped', False):
            original = layer.forward
            @functools.wraps(original)
            def checkpointed(*args, _forward=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint_module.checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)
            layer.forward = checkpointed
            layer._grpo_eval_checkpoint_wrapped = True
        count += 1
    return count
