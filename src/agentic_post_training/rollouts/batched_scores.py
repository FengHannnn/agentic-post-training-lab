"""Right-padded full-forward scoring of several original responses at once."""

import torch

from agentic_post_training.algorithms.losses import causal_token_log_probs


def score_responses(model,prompts,responses,pad_token_id,*,grad_enabled):
    if model.training or not prompts or len(prompts)!=len(responses):
        raise ValueError('eval model and aligned nonempty prompt/response batches required')
    if type(pad_token_id) is not int or pad_token_id<0:
        raise ValueError('valid padding token ID required')
    for tokens in [*prompts,*responses]:
        if not tokens or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('nonempty valid token sequences required')
    sequences=[p+r for p,r in zip(prompts,responses)]
    device=next(model.parameters()).device
    width=max(map(len,sequences))
    ids=torch.full((len(sequences),width),pad_token_id,dtype=torch.long,device=device)
    attention=torch.zeros_like(ids)
    for i,tokens in enumerate(sequences):
        ids[i,:len(tokens)]=torch.tensor(tokens,device=device)
        attention[i,:len(tokens)]=1  # EOS may equal PAD: use lengths, not token identity.
    with torch.set_grad_enabled(grad_enabled):
        logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits
        logps=causal_token_log_probs(logits.float(),ids)
        return [logps[i,len(p)-1:len(p)-1+len(r)] for i,(p,r) in enumerate(zip(prompts,responses))]
