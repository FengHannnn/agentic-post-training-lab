"""Scenario-weighted DPO with the reviewed weighted SFT exposure schedule."""
import math
from .dpo_recovery_training import training_batches, verify_update_history
from .dpo_training import pair_objective


def optimize_weighted_dpo(model, encoded, reference, weights, optimizer, batch_size,
                          seed, beta, max_grad_norm, epochs, on_step=None):
    import torch
    if not encoded or len(encoded) != len(reference) or len(encoded) != len(weights):
        raise ValueError('aligned nonempty data, reference and weights required')
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('positive finite weights required')
    if type(epochs) is not int or epochs <= 0 or not math.isfinite(beta) or beta <= 0:
        raise ValueError('positive epochs and beta required')
    parameters = [p for p in model.parameters() if p.requires_grad]
    history = []; cursor = 0; model.train()
    for epoch in range(epochs):
        epoch_cursor = 0
        for epoch_step, batch in enumerate(training_batches(len(encoded), batch_size, seed + epoch), 1):
            optimizer.zero_grad(set_to_none=True); value = 0.
            for i in batch:
                loss, _ = pair_objective(model, encoded[i], reference[i], beta, True)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite weighted DPO loss')
                weighted = loss * weights[i] / len(batch)
                weighted.backward(); value += weighted.detach().item()
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            if norm.item() <= 0:
                raise ValueError('nonzero LoRA gradient required')
            optimizer.step(); cursor += len(batch); epoch_cursor += len(batch)
            row = {'step': len(history) + 1, 'pairs_seen': cursor, 'epoch': epoch + 1,
                   'epoch_step': epoch_step, 'epoch_pairs_seen': epoch_cursor,
                   'loss': value, 'grad_norm': norm.item()}
            history.append(row)
            if on_step is not None:
                on_step(row)
    return history


def verify_matched_updates(cfg, final, train, state=None):
    verify_update_history(cfg, final, train, state)
    for obj in (final, train):
        if (obj['algorithm'] != 'hard_recovery_weighted_dpo'
                or obj['source_sft_updates'] != cfg['source_sft_updates']
                or obj['source_dpo_updates'] != cfg['source_dpo_updates']
                or obj['total_sft_updates'] != cfg['source_sft_updates']
                or obj['total_dpo_updates'] != cfg['source_dpo_updates'] + cfg['max_steps']
                or obj['epochs_completed'] != cfg['epochs']):
            raise ValueError('matched DPO ancestry differs')
    for row in train['history']:
        if row['epoch_pairs_seen'] != row['pairs_seen'] - (row['epoch'] - 1) * cfg['pair_count']:
            raise ValueError('matched exposure schedule differs')


import json
from dataclasses import asdict
from .dpo_recovery_training import validate_generation
from .dpo_recovery import build_context
from .dpo import action_from_text
from .environment import Environment


def context_environment(task,context):
    if task.split!='development' or context!=build_context(task,context['scenario'],allowed_split='development'):raise ValueError('fixed development recovery context required')
    env=Environment(task)
    try:
        for item in context['records']:
            if env.step(action_from_text(item['text']))!=item['row']:raise ValueError('recovery prefix replay differs')
        if env.done:raise ValueError('unfinished recovery prefix required')
        return env
    except Exception:env.close();raise


def run_recovery(task,context,policy):
    env=context_environment(task,context);messages=[dict(m) for m in context['messages']];rows=[];prefix_errors=env.tool_errors
    try:
        while not env.done:
            g=policy(messages);row=env.step(action_from_text(g.text));row['generation']=asdict(g);rows.append(row)
            messages.extend([{'role':'assistant','content':g.text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        return {'task_id':task.id,'family_id':task.family_id,'split':task.split,'context_id':context['context_id'],'scenario':context['scenario'],
            'success':env.success,'turns':rows,'tool_errors':env.tool_errors-prefix_errors,'total_tool_errors':env.tool_errors,
            'termination':rows[-1]['termination'],'artifact':rows[-1]['observation'].get('artifact') if env.success else None}
    except Exception as exc:
        exc.episode_evidence={'task_id':task.id,'context_id':context['context_id'],'status':'interrupted','turns':rows,'error':str(exc)};raise
    finally:env.close()


def replay_recovery(task,context,episode,tokenizer,mode):
    for k,v in {'task_id':task.id,'family_id':task.family_id,'split':'development','context_id':context['context_id'],'scenario':context['scenario']}.items():
        if episode[k]!=v:raise ValueError('recovery identity differs')
    env=context_environment(task,context);messages=[dict(m) for m in context['messages']];prefix_errors=env.tool_errors
    try:
        if not episode['turns']:raise ValueError('nonempty complete recovery required')
        for row in episode['turns']:
            if env.done:raise ValueError('extra recovery turn after termination')
            g=row['generation'];validate_generation(tokenizer,messages,g,mode);actual=env.step(action_from_text(g['text']))
            if actual!={k:v for k,v in row.items() if k not in ('generation','sampling_seed')}:raise ValueError('recovery continuation replay differs')
            messages.extend([{'role':'assistant','content':g['text']},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if not env.done or env.success!=episode['success'] or env.tool_errors-prefix_errors!=episode['tool_errors'] or env.tool_errors!=episode['total_tool_errors'] or env.records[-1]['termination']!=episode['termination']:raise ValueError('complete recovery outcome differs')
        artifact=env.records[-1]['observation'].get('artifact') if env.success else None
        if artifact!=episode['artifact']:raise ValueError('recovery artifact differs')
        return len(episode['turns'])
    finally:env.close()
